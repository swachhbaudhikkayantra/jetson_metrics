#!/usr/bin/env python3
"""
benchmark.py (corrected) — YOLOv8 / YOLO26 / RF-DETR TensorRT benchmark for Jetson Orin Nano.

  Live webcam overlay:
      python3 benchmark.py --mode live --model yolov8 --engine engines/best_fp16.engine
  Headless benchmark:
      python3 benchmark.py --mode headless \
          --yolov8-engine engines/best_fp16.engine --yolov8-pt Models/Yolov8/best.pt \
          --yolo26-engine engines/yolo26n_fp16.engine --yolo26-pt Models/Yolov11/yolo26n.pt \
          --rfdetr-engine engines/rfdetr_fp16.engine --rfdetr-names rfdetr_names.json \
          --source images/ --val-dir val/images --val-ann val/annotations.json
"""

import argparse
import csv
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict

import cv2
import numpy as np

from names_util import build_class_map, parse_names, resolve_names
from trt_common import TRTEngineBase, decode_yolo, is_end2end_shape, yolo_preprocess

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0: Utilities
# ═════════════════════════════════════════════════════════════════════════════
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def file_size_mb(p):
    return os.path.getsize(p) / (1024 ** 2) if p and os.path.exists(p) else 0.0


def compute_stats(arr):
    keys = ["mean", "median", "std", "min", "max", "p90", "p95", "p99"]
    if len(arr) == 0:
        return {k: 0.0 for k in keys}
    a = np.asarray(arr, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def collect_images(path):
    """All images below `path` (case-insensitive extensions), sorted."""
    out = []
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _color_for_class(cls_id):
    rng = np.random.RandomState(int(cls_id) + 42)           # no global seeding side-effects
    return tuple(int(c) for c in rng.randint(50, 255, 3))


def draw_boxes(frame, boxes, scores, class_ids, names=None):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cid = int(class_ids[i])
        color = _color_for_class(cid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{(names or {}).get(cid, cid)}:{scores[i]:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def collect_system_info():
    from importlib import metadata
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "platform": platform.platform(), "cv2": cv2.__version__}
    for pkg in ("tensorrt", "torch", "ultralytics", "pycuda", "numpy"):
        try:
            info[pkg] = metadata.version(pkg)
        except Exception:                                   # noqa: BLE001
            info[pkg] = None
    if info.get("tensorrt") is None:                        # apt python3-libnvinfer has no dist-info
        try:
            import tensorrt
            info["tensorrt"] = tensorrt.__version__
        except Exception:                                   # noqa: BLE001
            pass
    try:
        with open("/etc/nv_tegra_release") as f:
            info["l4t"] = f.readline().strip()
    except OSError:
        info["l4t"] = None
    try:
        with open("/proc/device-tree/model", "rb") as f:
            info["board"] = f.read().decode(errors="ignore").strip("\x00 \n")
    except OSError:
        info["board"] = None
    try:
        info["jetpack"] = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],
            stderr=subprocess.DEVNULL, universal_newlines=True).strip() or None
    except Exception:                                       # noqa: BLE001
        info["jetpack"] = None
    return info


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: TensorRT YOLO wrapper (classic AND end-to-end heads)
# ═════════════════════════════════════════════════════════════════════════════
class TRTYOLOEngine(TRTEngineBase):
    """YOLOv8 / YOLO11 (classic head) and YOLO26 (end-to-end head) detection engines."""

    def __init__(self, engine_path, conf=0.25, iou=0.45):
        super().__init__(engine_path, tag="YOLO TRT")
        if len(self.output_names) != 1:
            raise RuntimeError(f"{engine_path}: expected 1 output tensor (detection model), got "
                               f"{len(self.output_names)} — segmentation/pose engines are not supported.")
        self.conf, self.iou = conf, iou
        self.input_h, self.input_w = self.input_shape[2], self.input_shape[3]
        self.out_name = self.output_names[0]
        out_shape = self.bindings[self.out_name]["shape"]
        self.end2end = is_end2end_shape(out_shape[1:])
        print(f"[YOLO TRT] {engine_path}\n  input={self.input_shape} output={out_shape} "
              f"head={'end-to-end (NMS-free)' if self.end2end else 'classic (+NMS)'}  "
              f"load={self.load_time * 1000:.0f} ms")

    def preprocess(self, img_bgr):
        return yolo_preprocess(img_bgr, self.input_h, self.input_w)

    def postprocess(self, outputs, meta, conf=None, iou=None):
        return decode_yolo(outputs[self.out_name], meta,
                           self.conf if conf is None else conf,
                           self.iou if iou is None else iou, end2end=self.end2end)


def load_engine(model_type, path, args):
    if model_type == "rfdetr":
        from trt_rfdetr import TRTRFDETREngine
        return TRTRFDETREngine(path, conf_thresh=args.conf, num_classes=args.rfdetr_num_classes)
    return TRTYOLOEngine(path, conf=args.conf, iou=args.iou)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Frame sources
# ═════════════════════════════════════════════════════════════════════════════
class ImageFolderSource:
    """Iterates over images in a folder (skips unreadable files)."""

    def __init__(self, path, loop=False):
        self.files = collect_images(path)
        if not self.files:
            raise FileNotFoundError(f"No images in {path}")
        self.idx, self.loop = 0, loop
        self.frames_read = self.frames_dropped = 0
        self.actual_fps = 0
        print(f"[ImageFolder] {len(self.files)} images from {path}")

    def read(self):
        for _ in range(len(self.files)):
            if self.idx >= len(self.files):
                if not self.loop:
                    return None
                self.idx = 0
            img = cv2.imread(self.files[self.idx])
            self.idx += 1
            if img is not None:
                self.frames_read += 1
                return img
        return None

    def release(self):
        pass


def gst_csi_pipeline(width=1920, height=1080, fps=30):
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1" % (width, height, fps))


class CameraCapture:
    """
    Threaded reader.  Live sources (USB index, CSI, rtsp/http) drop old frames to stay current;
    files are read losslessly (blocking) and optionally looped.
    """

    def __init__(self, source=0, width=1920, height=1080, csi=False, loop=False):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.live = csi or isinstance(source, int) or \
            str(source).lower().startswith(("rtsp://", "http://", "https://"))
        self.loop = loop and not self.live
        if csi:
            self.cap = cv2.VideoCapture(gst_csi_pipeline(width, height), cv2.CAP_GSTREAMER)
        elif isinstance(source, int) and sys.platform.startswith("linux"):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {'CSI camera' if csi else source}")
        if isinstance(source, int):
            # MJPG is required for 1080p30 on most USB cameras (YUYV is limited to ~5 fps)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        print(f"[Camera] {'live' if self.live else 'file'} source, {self.width}x{self.height} "
              f"@ {self.actual_fps:.1f} fps")
        self.q = queue.Queue(maxsize=2)
        self.frames_read = self.frames_dropped = 0
        self._stop, self._eof = threading.Event(), threading.Event()
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.loop and self.frames_read > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                if not ok:
                    break
            self.frames_read += 1
            if self.live:
                if self.q.full():
                    try:
                        self.q.get_nowait()
                        self.frames_dropped += 1
                    except queue.Empty:
                        pass
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    self.frames_dropped += 1
            else:
                while not self._stop.is_set():
                    try:
                        self.q.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        self._eof.set()

    def read(self, timeout=5.0):
        deadline = time.perf_counter() + timeout
        while True:
            try:
                return self.q.get(timeout=0.1)
            except queue.Empty:
                if self._eof.is_set() and self.q.empty():
                    return None
                if time.perf_counter() > deadline:
                    return None

    def release(self):
        self._stop.set()
        self.t.join(timeout=2)
        self.cap.release()


def open_source(args):
    p = args.source
    if os.path.isdir(p):
        return ImageFolderSource(p, loop=True)
    if os.path.isfile(p):
        return CameraCapture(p, loop=True)
    if args.csi:
        return CameraCapture(0, csi=True)
    return CameraCapture(int(p) if p.isdigit() else p)


def grab_sample_frames(args, n=20):
    src = open_source(args)
    frames = []
    try:
        for _ in range(n):
            f = src.read()
            if f is None:
                break
            frames.append(f.copy())
    finally:
        src.release()
    return frames


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Resource monitor (tegrastats)
# ═════════════════════════════════════════════════════════════════════════════
_RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
_RE_CPU = re.compile(r"CPU \[([^\]]*)\]")
_RE_TEMP = re.compile(r"\b([A-Za-z][\w]*)@(-?\d+(?:\.\d+)?)C")
_RE_RAIL = re.compile(r"\b([A-Z][A-Z0-9_]*)\s+(\d+)mW(?:/\d+mW)?")


def parse_tegrastats(line):
    """One tegrastats line -> sample dict (None if it is not a stats line)."""
    m = _RE_RAM.search(line)
    if not m:
        return None
    s = {"ts": time.time(), "ram_used_mb": float(m.group(1)), "ram_total_mb": float(m.group(2))}
    g = _RE_GR3D.search(line)
    s["gpu_util"] = float(g.group(1)) if g else None
    c = _RE_CPU.search(line)
    cores = []
    if c:
        for tok in c.group(1).split(","):
            mm = re.match(r"\s*(\d+)%", tok)
            if mm:                                          # "off" cores are skipped
                cores.append(float(mm.group(1)))
    s["cpu_util"] = cores
    # disabled sensors print e.g. 'GPU@-256C' on Orin/JetPack 5: not a temperature
    temps = {k.lower(): float(v) for k, v in _RE_TEMP.findall(line) if float(v) > -40}
    s["temp_cpu"] = temps.get("cpu", temps.get("tj"))
    s["temp_gpu"] = temps.get("gpu", temps.get("tj"))
    rails = {k: float(v) for k, v in _RE_RAIL.findall(line)}
    total = rails.get("VDD_IN", rails.get("POM_5V_IN"))
    s["power_total_mw"] = total
    s["power_rails_mw"] = rails
    return s


class ResourceMonitor:
    """Samples tegrastats in a background thread (psutil fallback off-Jetson)."""

    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.samples = []
        self.source = "none"
        self._stop = threading.Event()
        self._thread = None
        self._proc = None

    def start(self):
        self.samples = []
        self._stop.clear()
        exe = shutil.which("tegrastats")
        self._proc = None
        if exe:
            try:
                self._proc = subprocess.Popen([exe, "--interval", str(self.interval_ms)],
                                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                              universal_newlines=True)
            except OSError as e:
                print(f"  [WARN] cannot start tegrastats: {e}")
        self.source = "tegrastats" if self._proc is not None else "psutil"
        self._thread = threading.Thread(
            target=self._run_tegrastats if self._proc is not None else self._run_psutil, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:                               # noqa: BLE001
                pass
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self.source == "tegrastats" and not self.samples:
            print("  [WARN] tegrastats produced no data (try running with sudo) — "
                  "power/temperature/GPU columns will be n/a.")

    def latest(self):
        return self.samples[-1] if self.samples else {}

    def _run_tegrastats(self):
        proc = self._proc
        for line in proc.stdout:
            if self._stop.is_set():
                break
            s = parse_tegrastats(line)
            if s:
                self.samples.append(s)

    def _run_psutil(self):
        try:
            import psutil
        except ImportError:
            return
        while not self._stop.is_set():
            self.samples.append({"ts": time.time(), "gpu_util": None,
                                 "cpu_util": psutil.cpu_percent(percpu=True),
                                 "ram_used_mb": psutil.virtual_memory().used / 1024 ** 2,
                                 "temp_cpu": None, "temp_gpu": None, "power_total_mw": None})
            self._stop.wait(self.interval_ms / 1000)

    def summarize(self):
        if not self.samples:
            return {"available": False, "source": self.source}

        def vals(key):
            return [s[key] for s in self.samples if s.get(key) is not None]

        def agg(key, fn):
            v = vals(key)
            return float(fn(v)) if v else None

        cpu_avgs = [float(np.mean(s["cpu_util"])) for s in self.samples if s.get("cpu_util")]
        p_avg, p_max = agg("power_total_mw", np.mean), agg("power_total_mw", np.max)
        return {
            "available": True, "source": self.source, "num_samples": len(self.samples),
            "gpu_util_avg": agg("gpu_util", np.mean), "gpu_util_max": agg("gpu_util", np.max),
            "cpu_util_avg": float(np.mean(cpu_avgs)) if cpu_avgs else None,
            "cpu_util_max": float(np.max(cpu_avgs)) if cpu_avgs else None,
            "ram_peak_mb": agg("ram_used_mb", np.max),
            "power_avg_w": p_avg / 1000 if p_avg is not None else None,
            "power_peak_w": p_max / 1000 if p_max is not None else None,
            "temp_cpu_avg": agg("temp_cpu", np.mean), "temp_cpu_max": agg("temp_cpu", np.max),
            "temp_gpu_avg": agg("temp_gpu", np.mean), "temp_gpu_max": agg("temp_gpu", np.max),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Ground truth + accuracy (COCO mAP, proper P/R/F1)
# ═════════════════════════════════════════════════════════════════════════════
def find_yolo_label(img_path):
    """Label next to the image, or the Ultralytics layout images/... -> labels/..."""
    base = os.path.splitext(img_path)[0] + ".txt"
    if os.path.exists(base):
        return base
    parts = img_path.replace("\\", "/").split("/")
    for i in range(len(parts) - 2, -1, -1):                 # last 'images' directory component
        if parts[i] == "images":
            cand = "/".join(parts[:i] + ["labels"] + parts[i + 1:])
            cand = os.path.splitext(cand)[0] + ".txt"
            if os.path.exists(cand):
                return cand
            break
    return None


def _yolo_line_to_xywh(parts, w, h):
    """YOLO txt line -> COCO bbox [x, y, w, h] in pixels (supports polygon labels)."""
    if len(parts) == 5:
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        return [x1, y1, bw * w, bh * h]
    if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
        xs = [float(v) * w for v in parts[1::2]]
        ys = [float(v) * h for v in parts[2::2]]
        return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    return None


def load_ground_truth(val_dir, val_ann=None, val_names=None, max_images=0):
    from pycocotools.coco import COCO
    img_files = collect_images(val_dir)
    if max_images > 0:
        img_files = img_files[:max_images]
    if not img_files:
        raise FileNotFoundError(f"No images in --val-dir {val_dir}")

    if val_ann:
        coco = COCO(val_ann)
        by_name = {}
        for iid, info in coco.imgs.items():
            by_name.setdefault(os.path.basename(info["file_name"]), iid)
        images = [(p, by_name[os.path.basename(p)]) for p in img_files
                  if os.path.basename(p) in by_name]
        if not images:
            raise ValueError("No image in --val-dir matches a file_name in --val-ann")
        cat_names = {c["id"]: c["name"] for c in coco.cats.values()}
        return {"coco": coco, "images": images, "cat_names": cat_names,
                "source": f"COCO json {val_ann}", "images_without_labels": 0}

    names = parse_names(val_names) or {}
    cat_names = dict(names)
    imgs, anns, images, ann_id, labelled = [], [], [], 1, 0
    for p in img_files:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        iid = len(imgs) + 1
        imgs.append({"id": iid, "file_name": os.path.basename(p), "width": w, "height": h})
        images.append((p, iid))
        lbl = find_yolo_label(p)
        if lbl is None:
            continue
        labelled += 1
        with open(lbl) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                bbox = _yolo_line_to_xywh(parts, w, h)
                if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                cid = int(float(parts[0]))
                cat_names.setdefault(cid, str(cid))
                anns.append({"id": ann_id, "image_id": iid, "category_id": cid, "bbox": bbox,
                             "area": bbox[2] * bbox[3], "iscrowd": 0})
                ann_id += 1
    if labelled == 0:
        raise ValueError(f"No YOLO label (.txt) files found for the images in {val_dir}. Expected "
                         f"next to the images or in the parallel labels/ folder.")
    from pycocotools.coco import COCO as _COCO
    coco = _COCO()
    coco.dataset = {"images": imgs, "annotations": anns,
                    "categories": [{"id": i, "name": n} for i, n in sorted(cat_names.items())]}
    coco.createIndex()
    return {"coco": coco, "images": images, "cat_names": cat_names,
            "source": f"YOLO txt labels in {val_dir}", "images_without_labels": len(images) - labelled}


def _iou_xywh(box, gts):
    x1 = np.maximum(box[0], gts[:, 0])
    y1 = np.maximum(box[1], gts[:, 1])
    x2 = np.minimum(box[0] + box[2], gts[:, 0] + gts[:, 2])
    y2 = np.minimum(box[1] + box[3], gts[:, 1] + gts[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    return inter / (box[2] * box[3] + gts[:, 2] * gts[:, 3] - inter + 1e-9)


def precision_recall_at_conf(coco_gt, dets, img_ids, cat_ids, cat_names, conf, iou_thr=0.5):
    """Real TP/FP/FN: greedy score-ordered matching at IoU>=iou_thr, per image and class."""
    gt_by, dt_by = defaultdict(list), defaultdict(list)
    for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_ids, catIds=cat_ids)):
        if not a.get("iscrowd", 0):
            gt_by[(a["image_id"], a["category_id"])].append(a["bbox"])
    for d in dets:
        if d["score"] >= conf:
            dt_by[(d["image_id"], d["category_id"])].append((d["score"], d["bbox"]))
    per = defaultdict(lambda: [0, 0, 0])                    # cat -> [tp, fp, fn]
    for key in set(gt_by) | set(dt_by):
        gts = np.asarray(gt_by.get(key, []), dtype=np.float64).reshape(-1, 4)
        used = np.zeros(len(gts), bool)
        tp = 0
        for _, bb in sorted(dt_by.get(key, []), key=lambda t: -t[0]):
            if len(gts):
                ious = _iou_xywh(bb, gts)
                ious[used] = -1
                j = int(np.argmax(ious))
                if ious[j] >= iou_thr:
                    used[j] = True
                    tp += 1
        n_dt = len(dt_by.get(key, []))
        per[key[1]][0] += tp
        per[key[1]][1] += n_dt - tp
        per[key[1]][2] += len(gts) - tp

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                "tp": tp, "fp": fp, "fn": fn}

    tot = [sum(v[i] for v in per.values()) for i in range(3)]
    out = prf(*tot)
    out["per_class"] = {cat_names.get(c, str(c)): prf(*v) for c, v in sorted(per.items())}
    return out


def _clean(v):
    return None if v is None or v < 0 else float(v)


def evaluate_accuracy(detect_fn, model_names, gt, display_conf, tag=""):
    """
    detect_fn(img_bgr) -> boxes_xyxy, scores, class_ids   (run at a LOW threshold, e.g. 0.001)
    Model classes are matched to dataset categories by NAME; classes the dataset does not
    contain are ignored (and reported), so the model is only scored on what it can predict.
    """
    from pycocotools.cocoeval import COCOeval

    if not model_names:
        return {"skipped": "model class names unknown — supply --<model>-names"}
    cmap, report = build_class_map(model_names, gt["cat_names"])
    result = {"class_mapping": report}
    if not cmap:
        result["skipped"] = "no model class name matches a dataset class (different label sets)"
        return result
    print(f"  [{tag}] scoring on {len(cmap)} shared class(es): {sorted(report['matched'])}")
    if report["gt_classes_not_predictable"]:
        print(f"  [{tag}] dataset classes this model cannot predict: "
              f"{report['gt_classes_not_predictable']}")

    dets, ids, n_disp = [], [], 0
    total = len(gt["images"])
    for k, (path, img_id) in enumerate(gt["images"], 1):
        img = cv2.imread(path)
        if img is None:
            continue
        ids.append(img_id)
        boxes, scores, cls = detect_fn(img)
        n_disp += int((scores >= display_conf).sum())
        for b, s, c in zip(boxes, scores, cls):
            cid = cmap.get(int(c))
            if cid is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in b)
            dets.append({"image_id": img_id, "category_id": cid,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
        if k % 100 == 0:
            print(f"    {k}/{total} images")

    cat_ids = sorted(set(cmap.values()))
    result.update({"images_evaluated": len(ids), "eval_classes": [gt["cat_names"][c] for c in cat_ids],
                   "avg_dets_per_img_at_conf": n_disp / len(ids) if ids else 0.0,
                   "total_dets_scored": len(dets)})
    if not dets:
        result.update({"mAP50": 0.0, "mAP50_95": 0.0})
        return result

    E = COCOeval(gt["coco"], gt["coco"].loadRes(dets), "bbox")
    E.params.imgIds = ids
    E.params.catIds = cat_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    st = E.stats
    result.update({"mAP50_95": _clean(st[0]), "mAP50": _clean(st[1]), "mAP75": _clean(st[2]),
                   "mAP_small": _clean(st[3]), "mAP_medium": _clean(st[4]), "mAP_large": _clean(st[5]),
                   "AR_1": _clean(st[6]), "AR_10": _clean(st[7]), "AR_100": _clean(st[8])})

    prec = E.eval["precision"]                               # [T, R, K, A, M]
    per_class = {}
    for k, cid in enumerate(E.params.catIds):
        p_all, p50 = prec[:, :, k, 0, 2], prec[0, :, k, 0, 2]
        per_class[gt["cat_names"][cid]] = {
            "AP50_95": float(p_all[p_all > -1].mean()) if (p_all > -1).any() else None,
            "AP50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else None}
    result["per_class"] = per_class

    prf = precision_recall_at_conf(gt["coco"], dets, ids, cat_ids, gt["cat_names"], display_conf)
    result.update({"precision_at_conf": prf["precision"], "recall_at_conf": prf["recall"],
                   "f1_at_conf": prf["f1"], "tp": prf["tp"], "fp": prf["fp"], "fn": prf["fn"],
                   "conf_thresh_used": display_conf, "per_class_prf": prf["per_class"]})
    return result


def make_trt_detector(eng, conf):
    def fn(img):
        blob, meta = eng.preprocess(img)
        outputs, _ = eng.infer(blob)
        return eng.postprocess(outputs, meta, conf=conf)
    return fn


def make_pt_detector(model, conf, iou, imgsz, device):
    def fn(img):
        b = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, device=device,
                          verbose=False, max_det=300, rect=False)[0].boxes
        return (b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                b.cls.cpu().numpy().astype(np.int32))
    return fn


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: PyTorch baseline (comparable stages, real frames)
# ═════════════════════════════════════════════════════════════════════════════
def pytorch_baseline(model, device, frames, conf, iou, imgsz, num_frames=100, warmup=10):
    """
    Uses Ultralytics' own per-stage timers (r.speed) so 'inference' is comparable with the
    TensorRT infer_ms, and end-to-end is comparable with TensorRT preprocess+infer+postprocess.
    rect=False forces the same square letterboxed input TensorRT gets (Ultralytics otherwise
    shrinks a 16:9 frame to 384x640 and does ~40% less work).
    """
    for i in range(warmup):
        model.predict(frames[i % len(frames)], imgsz=imgsz, conf=conf, iou=iou,
                      device=device, verbose=False, rect=False)
    e2e, inf = [], []
    for i in range(num_frames):
        t0 = time.perf_counter()
        r = model.predict(frames[i % len(frames)], imgsz=imgsz, conf=conf, iou=iou,
                          device=device, verbose=False, rect=False)[0]
        e2e.append((time.perf_counter() - t0) * 1000)
        inf.append(r.speed["inference"])
    s_e2e, s_inf = compute_stats(e2e), compute_stats(inf)
    return {"device": str(device), "imgsz": imgsz, "pt_e2e_ms": s_e2e, "pt_infer_ms": s_inf,
            "pt_e2e_fps": 1000.0 / s_e2e["mean"] if s_e2e["mean"] > 0 else 0.0,
            "pt_infer_fps": 1000.0 / s_inf["mean"] if s_inf["mean"] > 0 else 0.0}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Latency benchmark
# ═════════════════════════════════════════════════════════════════════════════
def _one_pass(eng, frame):
    blob, meta = eng.preprocess(frame)
    outputs, _ = eng.infer(blob)
    return eng.postprocess(outputs, meta)


def run_benchmark(eng, source, num_frames, warmup, monitor=None):
    """Cold first frame -> warmup -> timed loop (resource monitor covers only the loop)."""
    frame = source.read()
    if frame is None:
        raise RuntimeError("Could not read the first frame from the source")

    t0 = time.perf_counter()
    _one_pass(eng, frame)
    first_frame_ms = (time.perf_counter() - t0) * 1000       # cold: includes lazy CUDA init
    print(f"\n  First-frame (cold) latency: {first_frame_ms:.1f} ms")
    print(f"  Warmup: {warmup} frames...")
    for _ in range(warmup):
        _one_pass(eng, frame)

    timings = defaultdict(list)
    det_counts = []
    read0 = getattr(source, "frames_read", 0)               # camera counters for the timed loop only
    drop0 = getattr(source, "frames_dropped", 0)
    if monitor:
        monitor.start()
    wall_start = time.perf_counter()
    print(f"  Running {num_frames} frames...")
    try:
        for _ in range(num_frames):
            t_all = time.perf_counter()
            t0 = time.perf_counter()
            frame = source.read()
            if frame is None:
                break
            timings["capture_ms"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            blob, meta = eng.preprocess(frame)
            timings["preprocess_ms"].append((time.perf_counter() - t0) * 1000)

            outputs, it = eng.infer(blob)
            timings["h2d_ms"].append(it["h2d_ms"])
            timings["infer_ms"].append(it["infer_ms"])
            timings["d2h_ms"].append(it["d2h_ms"])

            t0 = time.perf_counter()
            boxes, _, _ = eng.postprocess(outputs, meta)
            timings["postprocess_ms"].append((time.perf_counter() - t0) * 1000)
            det_counts.append(len(boxes))

            timings["total_ms"].append((time.perf_counter() - t_all) * 1000)
            timings["pipeline_ms"].append(timings["total_ms"][-1] - timings["capture_ms"][-1])
        wall = time.perf_counter() - wall_start
    finally:
        if monitor:
            monitor.stop()

    frames_done = len(timings["total_ms"])
    result = {"frames": frames_done, "wall_time_s": wall,
              "wall_fps": frames_done / wall if wall > 0 else 0.0,
              "first_frame_ms": first_frame_ms,
              "avg_dets_per_frame": float(np.mean(det_counts)) if det_counts else 0.0}
    for stage, vals in timings.items():
        result[stage] = compute_stats(vals)

    def fps(stage):
        m = result.get(stage, {}).get("mean", 0)
        return 1000.0 / m if m > 0 else 0.0

    result["infer_fps"] = fps("infer_ms")
    result["pipeline_fps"] = fps("pipeline_ms")             # pre + H2D + infer + D2H + post
    result["e2e_fps"] = fps("total_ms")                     # + frame capture / disk read

    totals = np.asarray(timings["total_ms"])
    if totals.size:
        med = float(np.median(totals))
        result["jitter_std_ms"] = float(totals.std())
        result["jitter_range_ms"] = float(totals.max() - totals.min())
        result["stutter_count"] = int((totals > 2 * med).sum())
    else:
        result.update({"jitter_std_ms": 0.0, "jitter_range_ms": 0.0, "stutter_count": 0})
    result["camera_frames_read"] = getattr(source, "frames_read", 0) - read0
    result["camera_frames_dropped"] = getattr(source, "frames_dropped", 0) - drop0
    result["camera_fps"] = getattr(source, "actual_fps", 0)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: Sustained-load (thermal) test
# ═════════════════════════════════════════════════════════════════════════════
def sustained_load_test(eng, source, duration_s, monitor):
    log, frames = [], 0
    monitor.start()
    t_start = win_t = time.perf_counter()
    win_n, win_lat = 0, 0.0
    print(f"\n  Sustained-load test: {duration_s}s ...")
    try:
        while time.perf_counter() - t_start < duration_s:
            frame = source.read()
            if frame is None:
                break
            t0 = time.perf_counter()
            _one_pass(eng, frame)
            lat = (time.perf_counter() - t0) * 1000
            frames += 1
            win_n += 1
            win_lat += lat
            now = time.perf_counter()
            if now - win_t >= 1.0:
                s = monitor.latest()
                pw = s.get("power_total_mw")
                log.append({"time_s": now - t_start, "fps": win_n / (now - win_t),
                            "latency_ms": win_lat / win_n, "temp_gpu": s.get("temp_gpu"),
                            "gpu_util": s.get("gpu_util"), "power_w": pw / 1000 if pw is not None else None})
                win_t, win_n, win_lat = now, 0, 0.0
    finally:
        monitor.stop()
    summary = {"frames": frames, "duration_s": time.perf_counter() - t_start}
    if len(log) >= 10:
        k = max(1, len(log) // 10)
        first = float(np.mean([e["fps"] for e in log[:k]]))
        last = float(np.mean([e["fps"] for e in log[-k:]]))
        summary.update({"fps_first_10pct": first, "fps_last_10pct": last,
                        "fps_drop_pct": (first - last) / first * 100 if first > 0 else 0.0})
    summary.update(monitor.summarize())
    print(f"  Sustained test done: {frames} frames, {len(log)} log points")
    return log, summary


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: Output (table / CSV / JSON / summary.md / plots)
# ═════════════════════════════════════════════════════════════════════════════
def _g(d, *path):
    for p in path:
        if not isinstance(d, dict) or d.get(p) is None:
            return None
        d = d[p]
    return d


def _div(a, b):
    return a / b if a is not None and b else None


def _fmt(v, fmt):
    return "n/a" if v is None else format(v, fmt)


METRICS = [
    ("Infer Latency (ms)", lambda r: _g(r, "latency", "infer_ms", "mean"), ".2f"),
    ("Pipeline Latency (ms)", lambda r: _g(r, "latency", "pipeline_ms", "mean"), ".2f"),
    ("E2E Latency incl. capture (ms)", lambda r: _g(r, "latency", "total_ms", "mean"), ".2f"),
    ("Infer FPS", lambda r: _g(r, "latency", "infer_fps"), ".1f"),
    ("Pipeline FPS", lambda r: _g(r, "latency", "pipeline_fps"), ".1f"),
    ("E2E FPS incl. capture", lambda r: _g(r, "latency", "e2e_fps"), ".1f"),
    ("P99 Total Latency (ms)", lambda r: _g(r, "latency", "total_ms", "p99"), ".2f"),
    ("First Frame (ms)", lambda r: _g(r, "latency", "first_frame_ms"), ".1f"),
    ("Jitter Std (ms)", lambda r: _g(r, "latency", "jitter_std_ms"), ".2f"),
    ("Stutter Count", lambda r: _g(r, "latency", "stutter_count"), "d"),
    ("mAP@0.5 (TRT)", lambda r: _g(r, "accuracy", "mAP50"), ".3f"),
    ("mAP@0.5:0.95 (TRT)", lambda r: _g(r, "accuracy", "mAP50_95"), ".3f"),
    ("mAP@0.5 (PyTorch)", lambda r: _g(r, "accuracy_pytorch", "mAP50"), ".3f"),
    ("Precision @conf", lambda r: _g(r, "accuracy", "precision_at_conf"), ".3f"),
    ("Recall @conf", lambda r: _g(r, "accuracy", "recall_at_conf"), ".3f"),
    ("GPU Util %", lambda r: _g(r, "resources", "gpu_util_avg"), ".1f"),
    ("Power Avg (W, board)", lambda r: _g(r, "resources", "power_avg_w"), ".2f"),
    ("FPS/W (wall-clock)", lambda r: _g(r, "resources", "fps_per_watt"), ".2f"),
    ("Energy / frame (mJ)", lambda r: _g(r, "resources", "energy_per_frame_mj"), ".1f"),
    ("Temp GPU Max (C)", lambda r: _g(r, "resources", "temp_gpu_max"), ".1f"),
    ("PT/TRT speedup (infer)", lambda r: _g(r, "pytorch_baseline", "speedup_infer"), ".2f"),
    ("PT/TRT speedup (e2e)", lambda r: _g(r, "pytorch_baseline", "speedup_e2e"), ".2f"),
    ("Engine Size (MB)", lambda r: _g(r, "model_info", "engine_mb"), ".1f"),
    ("Engine Load (ms)", lambda r: _g(r, "model_info", "load_time_ms"), ".0f"),
]


def print_comparison_table(all_results):
    from tabulate import tabulate
    models = [m for m in all_results if "error" not in all_results[m]]
    if not models:
        return
    rows = [[label] + [_fmt(fn(all_results[m]), fmt) for m in models] for label, fn, fmt in METRICS]
    print("\n" + "=" * 80 + "\n BENCHMARK COMPARISON\n" + "=" * 80)
    print(tabulate(rows, headers=["Metric"] + models, tablefmt="grid") + "\n")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "_"))
        elif not isinstance(v, (list, tuple)):
            out[key] = v
    return out


def save_results(all_results, out_dir, system_info=None, args=None):
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    payload = {"system": system_info, "args": vars(args) if args else None, "results": all_results}
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    rows = [dict(model=m, timestamp=ts, **_flatten(r)) for m, r in all_results.items()]
    csv_path = os.path.join(out_dir, "results.csv")
    if rows:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    ok = {m: r for m, r in all_results.items() if "error" not in r}
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Summary\n\nGenerated: {ts}\n\n")
        if system_info:
            f.write(f"Board: {system_info.get('board')}  |  L4T: {system_info.get('l4t')}  |  "
                    f"JetPack: {system_info.get('jetpack')}  |  TensorRT: {system_info.get('tensorrt')}\n\n")
        f.write("## Winners by Metric\n\n| Metric | Winner | Value |\n|---|---|---|\n")
        for label, fn, hi in [
            ("Lowest inference latency (ms)", METRICS[0][1], False),
            ("Highest pipeline FPS", METRICS[4][1], True),
            ("Best mAP@0.5 (TRT, only where scored)", METRICS[10][1], True),
            ("Best FPS/W", METRICS[17][1], True),
            ("Smallest engine (MB)", METRICS[22][1], False)]:
            vals = {m: fn(r) for m, r in ok.items() if fn(r) is not None}
            if vals:
                w_ = (max if hi else min)(vals, key=vals.get)
                f.write(f"| {label} | **{w_}** | {vals[w_]:.3f} |\n")
        f.write("\n## All Metrics\n\n| Metric | " + " | ".join(ok) + " |\n|---|" + "---|" * len(ok) + "\n")
        for label, fn, fmt in METRICS:
            f.write(f"| {label} | " + " | ".join(_fmt(fn(r), fmt) for r in ok.values()) + " |\n")
        for m, r in all_results.items():
            if "error" in r:
                f.write(f"\n**{m} FAILED:** {r['error']}\n")
        f.write("\nNotes: power is whole-module VDD_IN from tegrastats; mAP is only comparable between "
                "models scored on the same classes (see `accuracy.eval_classes` in results.json).\n")
    print(f"\n  Results saved: {json_path}, {csv_path}, {md_path}")


def generate_plots(all_results, sustained_logs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    models = [m for m in all_results if "error" not in all_results[m]]
    if not models:
        return
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(models))
    for stage, label in [("preprocess_ms", "Preprocess"), ("h2d_ms", "H2D"), ("infer_ms", "Inference"),
                         ("d2h_ms", "D2H"), ("postprocess_ms", "Postprocess")]:
        v = np.array([_g(all_results[m], "latency", stage, "mean") or 0 for m in models])
        ax.bar(x, v, bottom=bottoms, label=label, width=0.5)
        bottoms += v
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Stage Latency Breakdown (mean)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_stacked.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.27
    for off, (key, label) in zip((-w, 0, w), [("infer_fps", "Inference-only"),
                                              ("pipeline_fps", "Pipeline (pre+infer+post)"),
                                              ("e2e_fps", "End-to-end incl. capture")]):
        ax.bar(x + off, [_g(all_results[m], "latency", key) or 0 for m in models], w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("FPS")
    ax.set_title("Throughput Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fps_comparison.png"), dpi=150)
    plt.close()

    fpw = [_g(all_results[m], "resources", "fps_per_watt") for m in models]
    if any(v is not None for v in fpw):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(models, [v or 0 for v in fpw], color="teal")
        ax.set_ylabel("Wall-clock FPS / Watt (board power)")
        ax.set_title("Energy Efficiency")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fps_per_watt.png"), dpi=150)
        plt.close()

    pts = [(m, _g(all_results[m], "latency", "infer_ms", "mean"), _g(all_results[m], "accuracy", "mAP50"))
           for m in models]
    pts = [p for p in pts if p[1] is not None and p[2] is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(8, 5))
        for m, lat, mp in pts:
            ax.scatter(lat, mp, s=100)
            ax.annotate(m, (lat, mp), textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("Inference Latency (ms)")
        ax.set_ylabel("mAP@0.5 (shared classes only)")
        ax.set_title("Accuracy vs Speed")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "map_vs_latency.png"), dpi=150)
        plt.close()

    for name, log in sustained_logs.items():
        if not log:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        ts = [e["time_s"] for e in log]
        nan = float("nan")
        axes[0].plot(ts, [e["fps"] for e in log], color="blue")
        axes[0].set_ylabel("FPS")
        axes[0].set_title(f"Sustained Load — {name}")
        axes[1].plot(ts, [e["latency_ms"] for e in log], color="orange")
        axes[1].set_ylabel("Latency (ms)")
        axes[2].plot(ts, [e["temp_gpu"] if e["temp_gpu"] is not None else nan for e in log],
                     color="red", label="GPU temp (C)")
        axes[2].set_ylabel("Temp (C)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend(loc="upper left")
        ax2 = axes[2].twinx()
        ax2.plot(ts, [e["power_w"] if e["power_w"] is not None else nan for e in log],
                 color="green", label="Power (W)")
        ax2.set_ylabel("Power (W)")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sustained_{name}.png"), dpi=150)
        plt.close()
    print(f"  Plots saved to {out_dir}/")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: Power modes
# ═════════════════════════════════════════════════════════════════════════════
def _run_priv(cmd):
    """Run a privileged command; sudo -n never blocks on a password prompt."""
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    return subprocess.run(cmd, check=True, capture_output=True, universal_newlines=True,
                          timeout=60, stdin=subprocess.DEVNULL)   # never wait on a prompt


def get_current_power_mode():
    """-> {'name': str, 'id': int|None} or None."""
    for prefix in ([], ["sudo", "-n"]):
        try:
            out = subprocess.check_output(prefix + ["nvpmodel", "-q"], universal_newlines=True,
                                          stderr=subprocess.DEVNULL, timeout=15)
        except Exception:                                   # noqa: BLE001
            continue
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        name = next((ln.split(":", 1)[1].strip() for ln in lines if "NV Power Mode" in ln), None)
        ident = int(lines[-1]) if lines and lines[-1].isdigit() else None
        if name or ident is not None:
            return {"name": name or "unknown", "id": ident}
    return None


def set_power_mode(mode_id, jetson_clocks=False):
    try:
        _run_priv(["nvpmodel", "-m", str(mode_id)])
        if jetson_clocks:
            _run_priv(["jetson_clocks"])
        time.sleep(5)                                       # let clocks settle
        cur = get_current_power_mode()
        if cur is not None and cur.get("id") is not None and cur["id"] != mode_id:
            print(f"  [WARN] requested power mode {mode_id} but the board reports {cur}")
            return None
        return cur
    except Exception as e:                                  # noqa: BLE001
        print(f"  [WARN] could not set power mode {mode_id}: {e}\n"
              f"         Run the benchmark with sudo, or allow passwordless sudo for nvpmodel.")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: Live mode
# ═════════════════════════════════════════════════════════════════════════════
def run_live(eng, args, names):
    cap = CameraCapture(args.camera, 1920, 1080, csi=args.csi)
    monitor = ResourceMonitor()
    monitor.start()
    print("\n  Live mode — press 'q' to quit")
    fps_smooth = 0.0
    try:
        while True:
            frame = cap.read()
            if frame is None:
                break
            t0 = time.perf_counter()
            blob, meta = eng.preprocess(frame)
            outputs, timings = eng.infer(blob)
            boxes, scores, class_ids = eng.postprocess(outputs, meta)
            total_ms = (time.perf_counter() - t0) * 1000
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps

            draw_boxes(frame, boxes, scores, class_ids, names)
            s = monitor.latest()
            pw, tg, gu = s.get("power_total_mw"), s.get("temp_gpu"), s.get("gpu_util")
            for i, line in enumerate([
                    f"FPS: {fps_smooth:.1f}",
                    f"Infer: {timings['infer_ms']:.1f}ms  Total: {total_ms:.1f}ms",
                    f"GPU: {_fmt(gu, '.0f')}%  Power: {_fmt(pw / 1000 if pw is not None else None, '.1f')}W",
                    f"Temp: {_fmt(tg, '.0f')}C  Dets: {len(boxes)}"]):
                cv2.putText(frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            try:
                cv2.imshow("Benchmark Live", frame)
            except cv2.error:
                print("  [ERROR] cv2.imshow unavailable — OpenCV has no GUI support "
                      "(pip 'opencv-python-headless' installed?) or no display attached.")
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        monitor.stop()
        cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def get_model_info(pt_path, engine_path):
    info = {"engine_mb": file_size_mb(engine_path)}
    if pt_path and os.path.exists(pt_path):
        info["pt_mb"] = file_size_mb(pt_path)
        try:
            from ultralytics import YOLO
            m = YOLO(pt_path)
            info["parameters"] = int(sum(p.numel() for p in m.model.parameters()))
        except Exception:                                   # noqa: BLE001
            info["parameters"] = None
    return info


def benchmark_one(name, cfg, args, gt, mode_name, out_dir):
    """Full benchmark of one engine. Returns the result dict + its sustained log."""
    print(f"\n{'─' * 60}\n  Model: {name}   Engine: {cfg['engine']}\n{'─' * 60}")
    names, names_src = resolve_names(cfg.get("names"), cfg["engine"], cfg.get("pt"))
    print(f"  Class names: {names_src}" + (f"  ({len(names)} classes)" if names else ""))

    eng = load_engine(cfg["type"], cfg["engine"], args)
    result = {"power_mode": mode_name, "conf": args.conf, "iou": args.iou,
              "input_hw": [getattr(eng, "input_h", None), getattr(eng, "input_w", None)],
              "class_names": names, "class_names_source": names_src}
    sustained_log = []
    try:
        source = open_source(args)
        monitor = ResourceMonitor()
        try:
            lat = run_benchmark(eng, source, args.num_frames, args.warmup, monitor)
        finally:
            source.release()
        result["latency"] = lat

        res = monitor.summarize()
        pwr, wfps = res.get("power_avg_w"), lat.get("wall_fps", 0)
        if pwr and wfps > 0:                                # power is averaged over WALL time
            res["energy_per_frame_mj"] = pwr / wfps * 1000
            res["fps_per_watt"] = wfps / pwr
        result["resources"] = res
        result["model_info"] = get_model_info(cfg.get("pt"), cfg["engine"])
        result["model_info"]["load_time_ms"] = eng.load_time * 1000

        if args.sustained:
            sdir = args.sustained_source or (args.source if os.path.isdir(args.source) else args.val_dir)
            if not sdir or not os.path.isdir(sdir):
                print("  [WARN] --sustained needs an image folder (--source or --sustained-source)")
            else:
                sustained_log, sres = sustained_load_test(
                    eng, ImageFolderSource(sdir, loop=True), args.sustained_duration, ResourceMonitor())
                result["sustained"] = sres

        if gt is not None:
            print("\n  Accuracy (TensorRT engine)...")
            result["accuracy"] = evaluate_accuracy(make_trt_detector(eng, args.eval_conf), names,
                                                   gt, args.conf, tag=f"{name}/TRT")
    finally:
        eng.close()
        del eng

    # PyTorch reference: latency on real frames and accuracy through the same pipeline
    if cfg.get("pt") and not args.skip_pt_baseline:
        try:
            from ultralytics import YOLO
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
            model = YOLO(cfg["pt"])
            imgsz = result["input_hw"][0] or 640
            frames = grab_sample_frames(args)
            if frames:
                print(f"\n  PyTorch baseline ({device})...")
                pt = pytorch_baseline(model, device, frames, args.conf, args.iou, imgsz)
                trt_inf = _g(result, "latency", "infer_ms", "mean")
                trt_pipe = _g(result, "latency", "pipeline_ms", "mean")
                pt["speedup_infer"] = _div(pt["pt_infer_ms"]["mean"], trt_inf)
                pt["speedup_e2e"] = _div(pt["pt_e2e_ms"]["mean"], trt_pipe)
                result["pytorch_baseline"] = pt
            if gt is not None:
                print("\n  Accuracy (PyTorch reference)...")
                result["accuracy_pytorch"] = evaluate_accuracy(
                    make_pt_detector(model, args.eval_conf, args.iou, imgsz, device),
                    {int(k): v for k, v in model.names.items()}, gt, args.conf, tag=f"{name}/PT")
                a, b = _g(result, "accuracy", "mAP50"), _g(result, "accuracy_pytorch", "mAP50")
                if a is not None and b is not None:
                    result["map50_drop_trt_vs_pt"] = b - a
            del model
            if device != "cpu":
                torch.cuda.empty_cache()
        except Exception as e:                              # noqa: BLE001
            print(f"  [WARN] PyTorch baseline failed: {e}")
            result["pytorch_baseline_error"] = str(e)
    return result, sustained_log


def build_parser():
    p = argparse.ArgumentParser(description="Jetson Orin Nano object-detection benchmark (corrected)")
    p.add_argument("--mode", choices=["live", "headless"], default="headless")
    for m, label in (("yolov8", "YOLOv8"), ("yolo26", "YOLO26"), ("rfdetr", "RF-DETR")):
        p.add_argument(f"--{m}-engine", help=f"{label} TensorRT engine")
        p.add_argument(f"--{m}-names", help=f"{label} class names: .json/.yaml/.txt file or comma list "
                                            f"(default: sidecar / .pt / ONNX metadata)")
    p.add_argument("--yolov8-pt", help="YOLOv8 .pt (PyTorch baseline + class names)")
    p.add_argument("--yolo26-pt", help="YOLO26 .pt (PyTorch baseline + class names)")
    p.add_argument("--model", choices=["yolov8", "yolo26", "rfdetr"], help="live mode: model type")
    p.add_argument("--engine", help="live mode: engine path")
    p.add_argument("--source", default="0", help="camera index, video file, or image folder")
    p.add_argument("--camera", type=int, default=0, help="camera index for live mode")
    p.add_argument("--csi", action="store_true", help="use the CSI camera via GStreamer (nvarguscamerasrc)")
    p.add_argument("--conf", type=float, default=0.25, help="display / P-R-F1 confidence threshold")
    p.add_argument("--eval-conf", type=float, default=0.001, help="threshold used while computing mAP")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--num-frames", type=int, default=500)
    p.add_argument("--rfdetr-num-classes", type=int, default=None, help="sanity check of logits/query")
    p.add_argument("--val-dir", help="validation images")
    p.add_argument("--val-ann", help="COCO annotation json for --val-dir (else YOLO .txt labels)")
    p.add_argument("--val-names", help="dataset class names (YOLO-txt GT); default: YOLOv8 model names")
    p.add_argument("--val-max-images", type=int, default=0, help="limit images used for accuracy (0 = all)")
    p.add_argument("--skip-pt-baseline", action="store_true")
    p.add_argument("--sustained", action="store_true")
    p.add_argument("--sustained-duration", type=int, default=600)
    p.add_argument("--sustained-source", help="image folder for the sustained test")
    p.add_argument("--cooldown", type=int, default=0, help="seconds to idle between models")
    p.add_argument("--power-modes", type=int, nargs="+", help="nvpmodel modes to sweep (needs sudo)")
    p.add_argument("--jetson-clocks", action="store_true")
    p.add_argument("--out-dir", default="results")
    return p


def main():
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)

    # ── live ──
    if args.mode == "live":
        if not args.engine:
            sys.exit("--engine is required for live mode")
        mtype = args.model or "yolov8"
        pt = getattr(args, f"{mtype}_pt", None) if mtype != "rfdetr" else None
        names, src = resolve_names(getattr(args, f"{mtype}_names"), args.engine, pt)
        print(f"Class names: {src}")
        eng = load_engine(mtype, args.engine, args)
        eng.warmup(args.warmup)
        try:
            run_live(eng, args, names)
        finally:
            eng.close()
        return

    # ── headless ──
    print("=" * 70 + "\n Jetson Orin Nano — Headless Benchmark (corrected)\n" + "=" * 70)
    system_info = collect_system_info()
    orig_mode = get_current_power_mode()
    print(f"  Power mode: {orig_mode}")

    engines = {}
    for m, typ in (("yolov8", "yolo"), ("yolo26", "yolo"), ("rfdetr", "rfdetr")):
        if getattr(args, f"{m}_engine"):
            engines[m] = {"engine": getattr(args, f"{m}_engine"), "type": typ,
                          "pt": getattr(args, f"{m}_pt", None) if m != "rfdetr" else None,
                          "names": getattr(args, f"{m}_names")}
    if not engines:
        sys.exit("[ERROR] No engines given. Use --yolov8-engine / --yolo26-engine / --rfdetr-engine")
    for name, cfg in engines.items():
        if not os.path.isfile(cfg["engine"]):
            print(f"  [WARN] engine not found for {name}: {cfg['engine']} — it will be reported as failed")

    gt = None
    if args.val_dir:
        try:
            ds_names = parse_names(args.val_names)
            if ds_names is None and not args.val_ann and "yolov8" in engines:
                c = engines["yolov8"]
                ds_names, s = resolve_names(c["names"], c["engine"], c["pt"])
                print(f"  Dataset class names (YOLO-txt labels) taken from the YOLOv8 model: {s}")
            gt = load_ground_truth(args.val_dir, args.val_ann, ds_names, args.val_max_images)
            print(f"  Ground truth: {gt['source']}  images={len(gt['images'])}  "
                  f"classes={sorted(gt['cat_names'].values())}  unlabeled={gt['images_without_labels']}")
        except Exception as e:                              # noqa: BLE001
            print(f"  [WARN] accuracy evaluation disabled: {e}")

    modes = args.power_modes if args.power_modes else [None]
    if args.jetson_clocks and not args.power_modes:
        print("  [WARN] --jetson-clocks only takes effect together with --power-modes "
              "(run `sudo jetson_clocks` yourself for a single mode)")
    if args.power_modes and not (orig_mode and orig_mode.get("id") is not None):
        print("  [WARN] current power mode could not be read — it will NOT be restored afterwards")
    if args.power_modes and args.jetson_clocks:
        try:
            _run_priv(["jetson_clocks", "--store"])
        except Exception as e:                              # noqa: BLE001
            print(f"  [WARN] could not store jetson_clocks state: {e}")

    all_modes = {}
    try:
        for pm in modes:
            if pm is not None:
                cur = set_power_mode(pm, args.jetson_clocks)
                if cur is None:
                    print(f"\n  [SKIP] power mode {pm}: could not be applied — not benchmarking it "
                          f"under a wrong label")
                    continue
                mode_name = cur["name"]
                print(f"\n  ════ Power mode {pm}: {mode_name} ════")
            else:
                cur = get_current_power_mode()
                mode_name = cur["name"] if cur else "unknown"
            out_dir = os.path.join(args.out_dir, f"mode{pm}") if pm is not None else args.out_dir
            ensure_dir(out_dir)

            all_results, sustained_logs = {}, {}
            for i, (name, cfg) in enumerate(engines.items()):
                if i and args.cooldown:
                    print(f"\n  Cooling down {args.cooldown}s ...")
                    time.sleep(args.cooldown)
                try:
                    all_results[name], slog = benchmark_one(name, cfg, args, gt, mode_name, out_dir)
                    sustained_logs[name] = slog
                except Exception as e:                      # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    all_results[name] = {"error": f"{type(e).__name__}: {e}"}
                    print(f"  [ERROR] {name} failed: {e}")

            save_results(all_results, out_dir, system_info, args)
            try:
                print_comparison_table(all_results)
            except Exception as e:                          # noqa: BLE001
                print(f"  [WARN] comparison table failed: {e}")
            try:
                generate_plots(all_results, sustained_logs, out_dir)
                if sustained_logs:
                    with open(os.path.join(out_dir, "sustained_load_log.json"), "w", encoding="utf-8") as f:
                        json.dump(sustained_logs, f, indent=2, default=_json_default)
            except Exception as e:                          # noqa: BLE001
                print(f"  [WARN] plotting failed: {e}")
            all_modes["default" if pm is None else f"mode{pm}"] = all_results
    finally:
        if args.power_modes:
            if orig_mode and orig_mode.get("id") is not None:
                set_power_mode(orig_mode["id"], False)
                print(f"  Restored power mode {orig_mode}")
            if args.jetson_clocks:
                try:
                    _run_priv(["jetson_clocks", "--restore"])
                except Exception:                           # noqa: BLE001
                    pass

    if len(modes) > 1:
        path = os.path.join(args.out_dir, "results_all_modes.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"system": system_info, "modes": all_modes}, f, indent=2, default=_json_default)
        print(f"\n  All-modes results: {path}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
