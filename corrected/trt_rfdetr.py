#!/usr/bin/env python3
"""
trt_rfdetr.py — TensorRT inference wrapper for RF-DETR (corrected).

RF-DETR ONNX outputs:
  dets   [1, Q, 4]  (cx, cy, w, h) normalised to [0, 1]
  labels [1, Q, C]  class logits (sigmoid, NOT softmax)

Decoding follows the reference RF-DETR post-processor: sigmoid over the logits, then the
top-Q of all (query, class) pairs above the threshold.  No NMS (set prediction).
Returned class ids are RAW output indices; map them to names with a names file, because
C is usually (#classes + 1 dummy slot) and the ONNX carries no names.
"""

import numpy as np

import cv2

from trt_common import TRTEngineBase

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def rfdetr_preprocess(img_bgr, input_h, input_w):
    """BGR uint8 -> (float32 NCHW, ImageNet-normalised, meta=(orig_w, orig_h)). Plain resize."""
    oh, ow = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    img = resized[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    blob = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...])
    return blob, (ow, oh)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class TRTRFDETREngine(TRTEngineBase):
    def __init__(self, engine_path, conf_thresh=0.5, num_classes=None):
        super().__init__(engine_path, tag="RF-DETR TRT")
        self.conf = conf_thresh
        self.input_h, self.input_w = self.input_shape[2], self.input_shape[3]

        # identify the box tensor and the logits tensor
        if "dets" in self.output_names and "labels" in self.output_names:
            self.dets_name, self.labels_name = "dets", "labels"
        else:
            box = [n for n in self.output_names if self.bindings[n]["shape"][-1] == 4]
            lab = [n for n in self.output_names if self.bindings[n]["shape"][-1] != 4]
            if len(box) != 1 or len(lab) != 1:
                raise RuntimeError(f"Cannot tell RF-DETR outputs apart: "
                                   f"{[(n, self.bindings[n]['shape']) for n in self.output_names]}")
            self.dets_name, self.labels_name = box[0], lab[0]

        _, self.num_queries, self.num_classes = self.bindings[self.labels_name]["shape"]
        if num_classes is not None and num_classes != self.num_classes:
            print(f"  [WARN] --rfdetr-num-classes={num_classes} but the engine has "
                  f"{self.num_classes} logits per query; using {self.num_classes}.")
        print(f"[RF-DETR TRT] {engine_path}\n  input={self.input_shape}  queries={self.num_queries}"
              f"  logits/query={self.num_classes}  load={self.load_time * 1000:.0f} ms")

    def preprocess(self, image_bgr):
        return rfdetr_preprocess(image_bgr, self.input_h, self.input_w)

    def postprocess(self, outputs, meta, conf=None, iou=None):          # iou unused (no NMS)
        thr = self.conf if conf is None else conf
        ow, oh = meta
        boxes = outputs[self.dets_name][0].astype(np.float32)           # (Q, 4) cxcywh norm
        prob = _sigmoid(outputs[self.labels_name][0].astype(np.float32))  # (Q, C)
        flat = prob.reshape(-1)

        keep = np.flatnonzero(flat >= thr)
        if keep.size == 0:
            return (np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int32))
        keep = keep[np.argsort(-flat[keep], kind="stable")[: self.num_queries]]

        q = keep // self.num_classes
        cls = (keep % self.num_classes).astype(np.int32)
        b = boxes[q]
        out = np.empty((len(keep), 4), np.float32)
        out[:, 0] = np.clip((b[:, 0] - b[:, 2] / 2) * ow, 0, ow)
        out[:, 1] = np.clip((b[:, 1] - b[:, 3] / 2) * oh, 0, oh)
        out[:, 2] = np.clip((b[:, 0] + b[:, 2] / 2) * ow, 0, ow)
        out[:, 3] = np.clip((b[:, 1] + b[:, 3] / 2) * oh, 0, oh)
        return out, flat[keep].astype(np.float32), cls


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from names_util import resolve_names

    ap = argparse.ArgumentParser(description="Run one image through an RF-DETR engine")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--classes", type=int, default=None, help="expected logits per query (sanity check)")
    ap.add_argument("--names", default=None, help="names file (.json/.yaml/.txt) or comma list, "
                                                  "indexed by RF-DETR output index")
    args = ap.parse_args()

    names, src = resolve_names(args.names, engine_path=args.engine)
    print(f"class names: {src}")
    eng = TRTRFDETREngine(args.engine, conf_thresh=args.conf, num_classes=args.classes)
    eng.warmup(5)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    blob, meta = eng.preprocess(img)
    outputs, t = eng.infer(blob)
    boxes, scores, cls = eng.postprocess(outputs, meta)

    print(f"\nTimings: H2D={t['h2d_ms']:.2f}ms  Infer={t['infer_ms']:.2f}ms  D2H={t['d2h_ms']:.2f}ms")
    print(f"Detections: {len(boxes)}")
    for i in range(len(boxes)):
        label = (names or {}).get(int(cls[i]), str(int(cls[i])))
        x1, y1, x2, y2 = boxes[i]
        print(f"  {label:<18s} conf={scores[i]:.3f}  box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {scores[i]:.2f}", (int(x1), max(int(y1) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite("rfdetr_test_output.jpg", img)
    print("Saved rfdetr_test_output.jpg")
    eng.close()
