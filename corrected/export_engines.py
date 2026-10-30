#!/usr/bin/env python3
"""
export_engines.py (corrected) — build TensorRT engines ON THE JETSON.

    python3 export_engines.py \
        --yolov8 Models/Yolov8/best.pt \
        --yolo26 Models/Yolov11/yolo26n.pt \
        --rfdetr-onnx "Models/Rfdetr/Copy of rfdetr_fp32.onnx" \
        --rfdetr-names rfdetr_names.json \
        --out-dir engines/

Engines written:  <name>_<precision>.engine   (best_fp16.engine, yolo26n_fp16.engine,
                                               rfdetr_fp16.engine, ...)
Sidecars:         <name>.names.json, <name>.onnx, export_meta.json

Works with TensorRT 8.5+ and 10.x (Python builder API; trtexec is not required).
--yolov8 / --yolo26 accept a .pt (exported here) or an FP32 .onnx (used as is).
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from names_util import names_from_onnx, parse_names

UNSUPPORTED_ONNX_OPS = {"ConvInteger", "MatMulInteger", "DynamicQuantizeLinear", "QLinearConv",
                        "QLinearMatMul"}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ONNX sanity check
# ─────────────────────────────────────────────────────────────────────────────
def check_onnx(path):
    """Refuse ONNX files TensorRT cannot use (needs the `onnx` package; skipped if absent).
    Returns {'ops': set, 'opset': int} for the version warnings, or None."""
    try:
        import onnx
    except ImportError:
        print("    [WARN] `onnx` not installed — skipping ONNX sanity check")
        return None
    m = onnx.load(path, load_external_data=False)
    ops = {n.op_type for n in m.graph.node}
    bad = sorted(ops & UNSUPPORTED_ONNX_OPS)
    if bad:
        raise RuntimeError(
            f"{os.path.basename(path)} uses ONNX-Runtime-only quantized ops {bad}, which TensorRT "
            f"cannot parse. Build from the FP32 ONNX instead (INT8 is produced here via calibration).")
    inits = {i.name for i in m.graph.initializer}
    for i in m.graph.input:
        if i.name in inits:
            continue
        dims = [d.dim_value if d.dim_value else -1 for d in i.type.tensor_type.shape.dim]
        if any(d < 0 for d in dims):
            raise RuntimeError(f"ONNX input '{i.name}' has dynamic dims {dims}; a static shape is required")
        if i.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            print(f"    [WARN] ONNX input '{i.name}' is not FP32 — build from the FP32 export for a fair baseline")
    opset = max([o.version for o in m.opset_import if o.domain in ("", "ai.onnx")] or [0])
    return {"ops": ops, "opset": opset}


# ─────────────────────────────────────────────────────────────────────────────
# Calibration preprocessing (identical to inference-time preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
def list_images(d, count):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for root, _, fs in os.walk(d):
        files += [os.path.join(root, f) for f in fs if os.path.splitext(f)[1].lower() in exts]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No calibration images in {d}")
    step = max(1, len(files) // count)                       # spread across the set
    return files[::step][:count]


def make_preprocess(kind, h, w):
    import cv2
    if kind == "rfdetr":
        from trt_rfdetr import rfdetr_preprocess
        return lambda path: rfdetr_preprocess(cv2.imread(path), h, w)[0]
    from trt_common import yolo_preprocess
    return lambda path: yolo_preprocess(cv2.imread(path), h, w)[0]


def make_calibrator(trt, kind, images, input_shape, cache_file):
    """Entropy-2 for CNN (YOLO), MinMax for the transformer (RF-DETR)."""
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda

    base = trt.IInt8MinMaxCalibrator if kind == "rfdetr" else trt.IInt8EntropyCalibrator2
    prep = make_preprocess(kind, input_shape[2], input_shape[3])

    class _Calibrator(base):
        def __init__(self):
            super().__init__()
            self.files, self.i = images, 0
            self.dev = cuda.mem_alloc(int(np.prod(input_shape)) * 4)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            while self.i < len(self.files):
                path = self.files[self.i]
                self.i += 1
                try:
                    blob = np.ascontiguousarray(prep(path), dtype=np.float32)
                except Exception as e:                       # noqa: BLE001
                    print(f"    calibrator: skipping {path}: {e}")
                    continue
                cuda.memcpy_htod(self.dev, blob)
                return [int(self.dev)]
            return None

        def read_calibration_cache(self):
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(cache_file, "wb") as f:
                f.write(cache)

    return _Calibrator()


# ─────────────────────────────────────────────────────────────────────────────
# Engine builder (TensorRT Python API, 8.5+ / 10.x)
# ─────────────────────────────────────────────────────────────────────────────
def build_engine(onnx_path, engine_path, precision, kind, workspace_mb=2048,
                 calib_dir=None, calib_count=100):
    import tensorrt as trt

    info = check_onnx(onnx_path)
    if info:
        try:
            trt_ver = tuple(int(x) for x in trt.__version__.split(".")[:2])
        except ValueError:
            trt_ver = (99, 0)
        if trt_ver < (8, 6) and "LayerNormalization" in info["ops"]:
            print("    [WARN] this ONNX uses LayerNormalization (needs TensorRT >= 8.6). On TensorRT "
                  f"{trt.__version__} the parse will probably fail — re-export the model at ONNX opset <= 16.")
        if trt_ver < (8, 6) and info["opset"] > 17:
            print(f"    [WARN] ONNX opset {info['opset']} is newer than TensorRT {trt.__version__} supports "
                  f"(<= 17). Re-export at opset 17.")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    try:                                                     # TensorRT 8.x needs the flag; 10.x ignores it
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        flags = 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(os.path.abspath(onnx_path)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    in_shape = tuple(network.get_input(0).shape)

    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)              # true FP32 baseline
    elif precision == "fp16":
        if not getattr(builder, "platform_has_fast_fp16", True):
            print("    [WARN] platform reports no fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not calib_dir:
            raise RuntimeError("INT8 needs --calib-dir (real images); refusing to build with random data")
        if not getattr(builder, "platform_has_fast_int8", True):
            print("    [WARN] platform reports no fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)                # FP16 fallback for layers INT8 cannot take
        cache = engine_path.replace(".engine", "_calib.cache")
        calibrator = make_calibrator(                        # keep a reference until the build ends
            trt, kind, list_images(calib_dir, calib_count), in_shape, cache)
        config.int8_calibrator = calibrator
    else:
        raise ValueError(precision)

    print(f"    building {precision.upper()} engine (input {in_shape}) — can take several minutes...")
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed (see log above)")
    with open(engine_path, "wb") as f:
        f.write(bytes(blob))
    return time.time() - t0


def describe_engine(engine_path):
    """Deserialize the fresh engine and print its I/O — catches a broken build immediately."""
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    if eng is None:
        raise RuntimeError(f"{engine_path} does not deserialize")
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        print(f"      {'IN ' if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT else 'OUT'} {n}: "
              f"{tuple(eng.get_tensor_shape(n))} {eng.get_tensor_dtype(n)}")


def build_all(onnx_path, out_dir, base, kind, precisions, args, extra):
    res = dict(extra, onnx=onnx_path, onnx_mb=size_mb(onnx_path), kind=kind)
    for prec in precisions:
        engine_path = os.path.join(out_dir, f"{base}_{prec}.engine")
        print(f"\n  [{base}] {prec.upper()} -> {engine_path}")
        try:
            bt = build_engine(onnx_path, engine_path, prec, kind, args.workspace_mb,
                              args.calib_dir, args.calib_count)
            describe_engine(engine_path)
            res[f"engine_{prec}"], res[f"engine_{prec}_mb"] = engine_path, size_mb(engine_path)
            res[f"build_time_{prec}"] = bt
            print(f"    done in {bt:.0f}s ({size_mb(engine_path):.1f} MB)")
        except Exception as e:                               # noqa: BLE001
            print(f"    [ERROR] {prec} build failed: {e}")
            res[f"error_{prec}"] = str(e)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Per-model preparation
# ─────────────────────────────────────────────────────────────────────────────
def export_yolo(src, label, args, precisions):
    base = os.path.splitext(os.path.basename(src))[0]
    print(f"\n{'=' * 60}\n {label}: {src}\n{'=' * 60}")
    onnx_path = os.path.join(args.out_dir, base + ".onnx")
    extra = {"model_type": label, "source": src}
    if src.lower().endswith(".onnx"):
        if os.path.abspath(src) != os.path.abspath(onnx_path):
            shutil.copy(src, onnx_path)
        try:
            names = names_from_onnx(onnx_path)
        except Exception:                                    # noqa: BLE001
            names = None
    else:
        os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")   # no surprise pip installs
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO
        model = YOLO(src)
        names = {int(k): v for k, v in model.names.items()}
        extra["pt_mb"] = size_mb(src)
        exported = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=17,
                                dynamic=False, half=False)
        exported = str(exported)
        if os.path.abspath(exported) != os.path.abspath(onnx_path):
            shutil.move(exported, onnx_path)                 # keep the Models folder clean
    if names:
        with open(os.path.join(args.out_dir, base + ".names.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in sorted(names.items())}, f, indent=2)
        print(f"    {len(names)} class names saved -> {base}.names.json")
    else:
        print("    [WARN] no class names found for this model")
    extra["name"] = base
    return build_all(onnx_path, args.out_dir, base, "yolo", precisions, args, extra)


def export_rfdetr(onnx_src, args, precisions):
    print(f"\n{'=' * 60}\n RF-DETR: {onnx_src}\n{'=' * 60}")
    onnx_path = os.path.join(args.out_dir, "rfdetr.onnx")
    shutil.copy(onnx_src, onnx_path)
    if args.rfdetr_names:
        names = parse_names(args.rfdetr_names)
        with open(os.path.join(args.out_dir, "rfdetr.names.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in sorted(names.items())}, f, indent=2)
        print(f"    {len(names)} class names saved -> rfdetr.names.json")
    else:
        print("    [WARN] --rfdetr-names not given: RF-DETR detections will show raw class indices and "
              "its accuracy cannot be scored. The ONNX has 10 logits per query; supply a list of 10 names "
              "indexed by output index (including the unused/dummy slot).")
    return build_all(onnx_path, args.out_dir, "rfdetr", "rfdetr", precisions, args,
                     {"model_type": "rfdetr", "name": "rfdetr", "source": onnx_src})


def main():
    ap = argparse.ArgumentParser(description="Build TensorRT engines on the Jetson")
    ap.add_argument("--yolov8", help="YOLOv8 .pt or FP32 .onnx")
    ap.add_argument("--yolo26", help="YOLO26 .pt or FP32 .onnx")
    ap.add_argument("--rfdetr-onnx", help="RF-DETR FP32 .onnx (NOT the fp16/int8 files)")
    ap.add_argument("--rfdetr-names", help="RF-DETR class names indexed by output index "
                                           "(.json/.yaml/.txt or comma list)")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO export size (RF-DETR size is fixed in its ONNX)")
    ap.add_argument("--out-dir", default="./engines")
    ap.add_argument("--fp32", action="store_true", help="also build FP32 engines")
    ap.add_argument("--int8", action="store_true", help="also build INT8 engines (needs --calib-dir)")
    ap.add_argument("--calib-dir", help="folder of representative images for INT8 calibration")
    ap.add_argument("--calib-count", type=int, default=200)
    ap.add_argument("--workspace-mb", type=int, default=2048,
                    help="builder workspace; lower it (e.g. 1024) on 4 GB modules")
    args = ap.parse_args()

    if not (args.yolov8 or args.yolo26 or args.rfdetr_onnx):
        ap.error("give at least one of --yolov8 / --yolo26 / --rfdetr-onnx")
    if args.int8 and not args.calib_dir:
        ap.error("--int8 requires --calib-dir (a folder of real images)")
    ensure_dir(args.out_dir)

    precisions = (["fp32"] if args.fp32 else []) + ["fp16"] + (["int8"] if args.int8 else [])
    results = []
    for label, src in (("yolov8", args.yolov8), ("yolo26", args.yolo26)):
        if src:
            if not os.path.isfile(src):
                print(f"[ERROR] {label}: file not found: {src}")
                continue
            try:
                results.append(export_yolo(src, label, args, precisions))
            except Exception as e:                           # noqa: BLE001
                print(f"[ERROR] {label} export failed: {e}")
                results.append({"model_type": label, "name": label, "error": str(e)})
    if args.rfdetr_onnx:
        if not os.path.isfile(args.rfdetr_onnx):
            print(f"[ERROR] rfdetr: file not found: {args.rfdetr_onnx}")
        else:
            try:
                results.append(export_rfdetr(args.rfdetr_onnx, args, precisions))
            except Exception as e:                           # noqa: BLE001
                print(f"[ERROR] rfdetr export failed: {e}")
                results.append({"model_type": "rfdetr", "name": "rfdetr", "error": str(e)})

    print(f"\n{'=' * 60}\n Export Summary\n{'=' * 60}")
    failed = False
    for r in results:
        print(f"\n  {r.get('name')} ({r.get('model_type')})")
        if "error" in r:
            print(f"    FAILED: {r['error']}")
            failed = True
            continue
        for p in precisions:
            if f"engine_{p}" in r:
                print(f"    {p:5s}: {r[f'engine_{p}_mb']:7.1f} MB  build={r[f'build_time_{p}']:.0f}s  "
                      f"[{r[f'engine_{p}']}]")
            else:
                print(f"    {p:5s}: NOT BUILT ({r.get(f'error_{p}', 'unknown')})")
                failed = True
    with open(os.path.join(args.out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Metadata saved: {os.path.join(args.out_dir, 'export_meta.json')}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
