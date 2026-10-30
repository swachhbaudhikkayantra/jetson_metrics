# Corrected Jetson benchmark

Corrected counterpart of `../sby-deploy/deployment/` (which is left untouched).
What this code must do is defined in [INTENT.md](INTENT.md).

## Files

| File | Replaces / adds |
|---|---|
| `setup.sh` | replaces `setup.sh` — no silent failures, never overwrites JetPack torch / tensorrt / OpenCV / numpy |
| `export_engines.py` | replaces `export_engines.py` — Python builder API (TensorRT 8.5+ and 10.x), real INT8 calibration, stable engine names, names sidecar |
| `trt_common.py` | new — shared engine plumbing + YOLO decode (classic **and** end-to-end YOLO26) + per-class NMS |
| `trt_rfdetr.py` | replaces `trt_rfdetr.py` — shares the base engine, reference-style decoding, clipping, names |
| `names_util.py` | new — class names (the original never used them) and name-based class matching |
| `benchmark.py` | replaces `benchmark.py` — see the fix table below |
| `run_both.sh` | new — runs original + corrected on the same engines/data |

## Run order (on the Jetson)

```bash
bash setup.sh                                  # read every [FAIL]/[WARN] line
sudo nvpmodel -m 0 && sudo jetson_clocks       # optional; check modes: sudo nvpmodel -q --verbose

# 1. engines (built on the Jetson, from FP32 sources)
python3 export_engines.py \
    --yolov8 Models/Yolov8/best.pt \
    --yolo26 Models/Yolov11/yolo26n.pt \
    --rfdetr-onnx "Models/Rfdetr/Copy of rfdetr_fp32.onnx" \
    --rfdetr-names rfdetr_names.json \
    --out-dir engines/
# -> engines/best_fp16.engine  yolo26n_fp16.engine  rfdetr_fp16.engine  + *.names.json

# 2. benchmark
python3 benchmark.py --mode headless \
    --yolov8-engine engines/best_fp16.engine  --yolov8-pt Models/Yolov8/best.pt \
    --yolo26-engine engines/yolo26n_fp16.engine --yolo26-pt Models/Yolov11/yolo26n.pt \
    --rfdetr-engine engines/rfdetr_fp16.engine --rfdetr-names rfdetr_names.json \
    --source images/ --val-dir val/images --val-ann val/annotations.json \
    --num-frames 500 --warmup 50 --out-dir results/

# 3. live
python3 benchmark.py --mode live --model yolov8 --engine engines/best_fp16.engine --camera 0
python3 benchmark.py --mode live --model yolov8 --engine engines/best_fp16.engine --csi   # CSI camera

# 4. RF-DETR on one image
python3 trt_rfdetr.py --engine engines/rfdetr_fp16.engine --image test.jpg --names rfdetr_names.json
```

### Run original and corrected side by side
```bash
bash run_both.sh all        # export (corrected exporter) then both benchmarks
# -> results_original/  results_corrected/  logs/
```
Both benchmarks use the **same engines and data**, so the difference you see is the benchmark code.
`run_both.sh` sets `PYTHONDONTWRITEBYTECODE=1` so the original folder stays clean (check with
`git -C ../sby-deploy status --porcelain`).

## RF-DETR names (you must provide these)
The ONNX has 10 logits per query and no names. RF-DETR usually has one dummy slot, so a 9-class model
has 10 outputs. Create `rfdetr_names.json` listing the name for **every output index**, e.g.
`["__dummy__", "aluminium_foil", "bottle", ...]` — confirm the order against the training dataset
(`_annotations.coco.json` categories). Without it RF-DETR shows raw indices and is not scored.

## What was fixed (numbers = rows of the earlier review table)

| # | Problem in original | Fix here |
|---|---|---|
| 1 | YOLO26 end-to-end output `(1,300,6)` decoded as classic head | `decode_yolo` auto-detects both heads (tested against Ultralytics) |
| 2, 3 | `trtexec` not on PATH / `--workspace` removed in TRT 10 | builder Python API, `set_memory_pool_limit` |
| 4, 5 | no working INT8 path; INT8 ONNX unusable | real `IInt8*Calibrator` on real images; ORT-quantized ONNX rejected up front |
| 6 | broken `.pth` route | removed (you only have ONNX) |
| 7 | mAP at conf 0.25 | `--eval-conf 0.001` for mAP, `--conf` for display/PRF |
| 8 | precision always 1.0 | real TP/FP/FN matching at IoU 0.5 |
| 9 | class ids not mapped | model classes matched to dataset classes **by name**; only shared classes are scored and reported |
| 10 | wrong `labels/` lookup | `images/` -> `labels/` substitution; polygon labels supported |
| 11 | class-agnostic NMS | per-class NMS |
| 12, 13 | `jtop` interval / voltage read as power | `tegrastats` parser (VDD_IN); disabled sensors (`-256C`) ignored; n/a when no data |
| 14 | PyTorch speedup on noise, mismatched stages | real frames, Ultralytics per-stage timers, `rect=False` so PyTorch gets the same square 640x640 input, two speedups (infer, e2e) + PyTorch mAP for the FP16 accuracy drop |
| 15 | build time always 0 | recorded in `engines/export_meta.json` (the benchmark table no longer shows a build-time row) |
| 16 | power sweep overwrote files | per-mode sub-folders, original mode restored |
| 17 | broken `summary.md` table | valid markdown, full metric table |
| 18 | no class names | names everywhere (sidecar / `.pt` / ONNX metadata / explicit file) |
| 19 | USB 1080p without MJPG, no CSI | MJPG, `--csi`, files read losslessly and looped |
| 20 | `sudo` hang | `sudo -n`, clear warning |
| 21-24 | `setup.sh` clobbered OpenCV/tensorrt/numpy, hid errors | rewritten |
| 25 | "YOLO11" label, engine-name mismatch | `yolo26`; `<model>_<precision>.engine` |
| 26 | unfair accuracy comparison | scored only on shared classes; the report says which |

## Known limits (be aware)
* TensorRT / pycuda / tegrastats / nvpmodel code paths could not be executed on the PC — first run on the
  Jetson is their first test. Decode, NMS, name matching, accuracy evaluation, tegrastats parsing,
  image-folder and video-file sources and reporting were tested on the PC; USB/CSI cameras were not.
* **RF-DETR on JetPack 5 (TensorRT 8.5):** the ONNX has 51 `LayerNormalization` nodes (native support
  arrived in TensorRT 8.6) and the YOLOv8 ONNX shipped in the zip is opset 20. The exporter warns, but the
  RF-DETR build may need a re-export at opset <= 16. JetPack 6 is the safer target.
* Energy per frame and FPS/W use wall-clock FPS (power is averaged over wall time), so they include
  frame-capture / disk-read time.
* A COCO-pretrained YOLO26 shares only `bottle` with your dataset, so its mAP is for that one class.
  For a fair three-way accuracy comparison fine-tune YOLO26 on the same 8 classes.
* Power is whole-module `VDD_IN`. INT8 and RF-DETR FP16 can lose accuracy — compare mAP vs the FP32/PyTorch row.
* The PyTorch reference is forced to the same square 640x640 letterbox (`rect=False`); tiny box differences
  from library resize paths remain possible.
