#!/usr/bin/env python3
"""
trt_common.py — shared pieces for the corrected Jetson benchmark.

  * TRTEngineBase  : engine load / buffer binding / timed inference (TensorRT >= 8.5, incl. 10.x)
  * letterbox, yolo_preprocess
  * nms, decode_yolo : decoding for BOTH classic YOLO heads (1, 4+nc, N) and
                       end-to-end heads such as YOLO26 (1, 300, 6)

tensorrt / pycuda are imported lazily so the decode helpers can be unit-tested on a PC.
"""

import time

import cv2
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Pre-processing
# ═════════════════════════════════════════════════════════════════════════════
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize with aspect-preserving padding. Returns (padded, ratio, (left, top))."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def yolo_preprocess(img_bgr, input_h, input_w):
    """BGR uint8 HWC -> (float32 NCHW RGB in [0,1], meta for decode_yolo)."""
    padded, ratio, (left, top) = letterbox(img_bgr, (input_h, input_w))
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob[np.newaxis, ...])
    return blob, (ratio, left, top, img_bgr.shape[:2])


# ═════════════════════════════════════════════════════════════════════════════
# NMS + YOLO decode
# ═════════════════════════════════════════════════════════════════════════════
def nms(boxes, scores, iou_thresh):
    """Greedy NMS on xyxy boxes. Returns kept indices (score-descending)."""
    if len(boxes) == 0:
        return np.empty(0, np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes, scores, class_ids, iou_thresh):
    """Per-class NMS (same behaviour as Ultralytics' default, agnostic=False)."""
    if len(boxes) == 0:
        return np.empty(0, np.int64)
    offset = class_ids.astype(np.float32)[:, None] * 7680.0      # > any box coordinate (Ultralytics' value)
    return nms(boxes + offset, scores, iou_thresh)


def is_end2end_shape(shape):
    """
    shape: output shape WITHOUT the batch dim.
      classic head   -> (4+nc, N)   e.g. (12, 8400)
      end-to-end head-> (K, 6)      e.g. (300, 6)   rows = x1,y1,x2,y2,score,cls
    """
    return len(shape) == 2 and shape[1] == 6 and shape[0] > 6


_EMPTY = (np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int32))


def decode_yolo(out, meta, conf, iou, end2end=None, max_det=300, max_nms=3000):
    """
    Decode a raw YOLO output tensor into (boxes_xyxy, scores, class_ids) in ORIGINAL
    image coordinates.  `meta` comes from yolo_preprocess().
    """
    ratio, left, top, (oh, ow) = meta
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    if end2end is None:
        end2end = is_end2end_shape(out.shape)

    if end2end:
        # rows: x1, y1, x2, y2, score, class  (already NMS-free, letterboxed pixels)
        mask = out[:, 4] >= conf
        if not mask.any():
            return _EMPTY
        sel = out[mask]
        order = np.argsort(-sel[:, 4], kind="stable")[:max_det]
        sel = sel[order]
        boxes = sel[:, :4].copy()
        scores = sel[:, 4]
        class_ids = np.rint(sel[:, 5]).astype(np.int32)
    else:
        if out.shape[0] < out.shape[1]:
            out = out.T                                     # -> (N, 4+nc)
        cls_scores = out[:, 4:]
        max_scores = cls_scores.max(axis=1)
        mask = max_scores >= conf
        if not mask.any():
            return _EMPTY
        xywh = out[mask, :4]
        scores = max_scores[mask]
        class_ids = cls_scores[mask].argmax(axis=1).astype(np.int32)
        if len(scores) > max_nms:                           # cap candidates before NMS
            top_k = np.argsort(-scores)[:max_nms]
            xywh, scores, class_ids = xywh[top_k], scores[top_k], class_ids[top_k]
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        keep = batched_nms(boxes, scores, class_ids, iou)[:max_det]
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    # undo letterbox, clip to the original frame
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / ratio, 0, ow)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / ratio, 0, oh)
    return boxes.astype(np.float32), scores.astype(np.float32), class_ids.astype(np.int32)


# ═════════════════════════════════════════════════════════════════════════════
# TensorRT engine base
# ═════════════════════════════════════════════════════════════════════════════
class TRTEngineBase:
    """Loads a serialized engine, allocates pinned/device buffers, runs timed inference."""

    def __init__(self, engine_path, tag="TRT"):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)

        self.trt, self.cuda, self.tag = trt, cuda, tag
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)

        t0 = time.perf_counter()
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)         # must outlive the engine
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.load_time = time.perf_counter() - t0
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize {engine_path}. Engines are tied to the exact "
                f"TensorRT version + GPU: rebuild it on this Jetson.")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        self.stream = cuda.Stream()
        self._ev_start, self._ev_end = cuda.Event(), cuda.Event()

        self.bindings = {}
        self.input_name = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Tensor '{name}' has dynamic shape {shape}; "
                                   f"rebuild the engine from a static-shape ONNX.")
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            n = int(np.prod(shape))
            host = cuda.pagelocked_empty(n, dtype)
            device = cuda.mem_alloc(n * dtype.itemsize)
            self.bindings[name] = {"shape": shape, "dtype": dtype, "host": host, "device": device}
            self.context.set_tensor_address(name, int(device))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        if self.input_name is None or not self.output_names:
            raise RuntimeError("Engine has no input or no output tensors")
        self.input_shape = self.bindings[self.input_name]["shape"]   # (1, 3, H, W)

    # ── timed inference ──────────────────────────────────────────────────────
    def infer(self, blob):
        cuda = self.cuda
        timings = {}
        inp = self.bindings[self.input_name]
        np.copyto(inp["host"], blob.ravel(), casting="same_kind")

        t0 = time.perf_counter()
        cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        self.stream.synchronize()
        timings["h2d_ms"] = (time.perf_counter() - t0) * 1000

        self._ev_start.record(self.stream)
        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        self._ev_end.record(self.stream)
        self._ev_end.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        timings["infer_ms"] = self._ev_start.time_till(self._ev_end)

        t0 = time.perf_counter()
        for name in self.output_names:
            o = self.bindings[name]
            cuda.memcpy_dtoh_async(o["host"], o["device"], self.stream)
        self.stream.synchronize()
        timings["d2h_ms"] = (time.perf_counter() - t0) * 1000

        outputs = {n: self.bindings[n]["host"].reshape(self.bindings[n]["shape"]).copy()
                   for n in self.output_names}
        return outputs, timings

    def warmup(self, n=10):
        dummy = np.random.rand(*self.input_shape).astype(np.float32)
        for _ in range(n):
            self.infer(dummy)

    def close(self):
        """Release GPU buffers (call between models to keep Orin Nano memory low)."""
        for b in getattr(self, "bindings", {}).values():
            try:
                b["device"].free()
            except Exception:
                pass
        self.bindings = {}
        self.context = None
        self.engine = None
        self.runtime = None
