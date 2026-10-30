# INTENT — what this code is supposed to do

This file is the reference both the original code (`../sby-deploy/deployment/`) and the corrected
code (this folder) are judged against. It was written from the original README, the original
docstrings, and the owner's stated goals. It does not describe any implementation.

## 1. Goal

Benchmark three object detectors on a **Jetson Orin Nano** using **TensorRT**, and produce
trustworthy numbers for latency, throughput, accuracy, resource use and thermal stability, so the
owner can choose a model for deployment.

Models under comparison:

| Key      | Model                                                            | Source file in the zip               |
|----------|------------------------------------------------------------------|--------------------------------------|
| `yolov8` | Custom YOLOv8n, 8 classes (aluminium_foil, bottle, cement_dust, floor, obstacle, paper, wooden_dust, wrapper) | `Yolov8/best.pt` (+ `last.pt`, pre-exported ONNX) |
| `yolo26` | YOLO26n, COCO 80-class, end-to-end (NMS-free) head, 640 px       | `Yolov11/yolo26n.pt` (the original code mislabels this "YOLO11") |
| `rfdetr` | RF-DETR, ONNX only, input 1x3x448x448, outputs `dets[1,300,4]` (normalised cxcywh) and `labels[1,300,10]` (logits) | `Rfdetr/Copy of rfdetr_fp32.onnx` |

## 2. Environment constraints

* The code is written and unit-tested on a Windows PC (no TensorRT, no Jetson), then copied to a
  Jetson Orin Nano (aarch64, JetPack 5.x or 6.x; exact version not yet known). It must not rely on
  x86 wheels, and must cope with TensorRT 8.5/8.6 (JetPack 5/6.0) **and** TensorRT 10.x (JetPack 6.1+).
* TensorRT engines must be built **on the Jetson**, from the **FP32** ONNX / `.pt` (the INT8 ONNX files in
  the zip are not valid TensorRT inputs: they use ONNX-Runtime-only ops; the FP16 ONNX files are parseable
  but are not the baseline to build from).
* Orin Nano has no DLA; everything runs on the GPU. Memory is shared (8 GB or 4 GB).
* Python may be as old as 3.8 (JetPack 5). OpenCV, TensorRT, torch and numpy come from JetPack /
  NVIDIA wheels and must **not** be replaced by pip installs.

## 3. Required behaviour, per component

### 3.1 Setup (`setup.sh`)
* Detect JetPack/L4T, verify CUDA-enabled torch, TensorRT bindings, system OpenCV (with GUI +
  GStreamer for live mode), `trtexec`, `tegrastats`.
* Install only what is missing, without overwriting JetPack-provided packages (torch, tensorrt,
  OpenCV, numpy<2). Report failures instead of hiding them.

### 3.2 Engine export (`export_engines.py`)
* Convert YOLO `.pt` (or a given `.onnx`) and the RF-DETR FP32 ONNX into TensorRT engines on the Jetson.
* Precisions: FP16 always; FP32 optional; INT8 optional **with a real calibrator** fed by real images
  preprocessed exactly as at inference time.
* Work on both TensorRT 8.5+ and 10.x, with static shapes, `/usr/src/tensorrt/bin/trtexec` not
  required. Refuse unusable ONNX (ONNX-Runtime quantized ops).
* Engine names must be predictable (`<model>_<precision>.engine`) and match the README commands.
* Persist class names (sidecar `<base>.names.json`) so downstream code can show and match names.
* Record file sizes and build times.

### 3.3 Inference wrappers
* Load an engine, bind static I/O, run timed H2D / GPU / D2H inference using CUDA events for GPU time.
* YOLOv8 (classic `[1,4+nc,N]`) **and** YOLO26 (end-to-end `[1,300,6]` = x1,y1,x2,y2,score,cls)
  must both decode to correct boxes in original-image pixels (letterbox undone, clipped).
  Classic YOLO uses per-class NMS (Ultralytics default).
* RF-DETR: resize (no letterbox), ImageNet mean/std, sigmoid on logits, normalised cxcywh -> pixel xyxy,
  no NMS. Class indices are raw output indices; names come from a user-supplied mapping.

### 3.4 Benchmark (`benchmark.py`)
Headless mode, for each engine, must report:
* **Latency** per stage (capture, preprocess, H2D, infer, D2H, postprocess, total): mean, median, std,
  min, max, p90, p95, p99; inference-only FPS, pipeline FPS, end-to-end FPS, wall FPS; first-frame
  (cold) latency; jitter and stutter count; camera frame drops (live sources only).
* **Resources** from `tegrastats`: GPU %, CPU %, RAM peak, whole-module power (W), CPU/GPU temperature,
  FPS/W and energy per frame. Missing data must show as n/a, never as a wrong number.
* **Accuracy** (when a validation set is given): COCO mAP@0.5, mAP@0.5:0.95 (and breakdowns), per-class
  AP with class **names**, and real precision / recall / F1 at the display threshold. mAP must use a
  low confidence threshold (~0.001). Model classes are matched to dataset classes **by name**; a model
  is only scored on classes it can predict, and the report states which. Accuracy of the TensorRT
  engine and of the original `.pt` are both measured through the same pipeline so the FP16 accuracy
  drop is visible.
* **PyTorch baseline**: latency measured on real frames with comparable stages
  (PT inference vs TRT infer; PT end-to-end vs TRT pre+infer+post), speedup factors.
* **Sustained load** (optional): FPS, latency, GPU temperature and power over time, plus the FPS drop
  between the first and last 10 % of the run.
* **Power-mode sweep** (optional): run every model under each `nvpmodel` mode, keep every mode's results
  in its own folder, restore the original mode afterwards, never hang on a sudo prompt.
* Outputs: terminal comparison table, `results.json` (with system / TensorRT / JetPack info and args),
  `results.csv`, `summary.md` (valid markdown), and plots (latency stack, FPS, FPS/W, mAP-vs-latency,
  sustained time series).
* One model failing must not abort the others.

Live mode: camera capture (USB with MJPG, CSI via GStreamer), boxes drawn with **class names**,
overlay with FPS, latency, GPU %, power, temperature.

### 3.5 Naming
* "YOLO11" is wrong for `yolo26n.pt`; the corrected code calls it `yolo26`.
* Class names must be shown wherever detections are shown or scored.

## 4. Deliverables and process
1. `../sby-deploy/` (original code) stays **byte-for-byte untouched**.
2. `corrected/` contains only code that needed correcting, plus helper modules the corrected code
   needs and a README.
3. `run_both.sh` runs the original and the corrected benchmark against the **same engines and data**
   on the Jetson, writing `results_original/` and `results_corrected/` so the two can be compared.
4. Everything that can be tested on the PC (decoding, NMS, name mapping, accuracy evaluation, metrics,
   sources, tegrastats parsing, shell syntax) is tested there. What cannot (TensorRT, pycuda, tegrastats
   on a real Orin, nvpmodel) is identified explicitly as untested.

## 5. Success criteria
* Numbers the benchmark prints are correct or marked n/a; none are silently wrong.
* No step in the documented run order crashes for a reason that can be foreseen from the code.
* The corrected code does what this document says; where the original did not, the difference is listed.
