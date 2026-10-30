#!/usr/bin/env python3
"""
paper_metrics.py — measure, on the Jetson, every metric the paper reports for the object-detection models
(Section "Edge Deployment Optimisation" tables + per-class tables) for each TensorRT engine.

  python3 paper_metrics.py                       # all engines found in engines/ (FP16 and INT8)
  python3 paper_metrics.py --models yolov8       # one model

Protocol (same as the paper): 10 discarded warm-up passes, then 100 timed passes over the validation images
(original 1280x720 camera frames); mean and standard deviation of the per-image latency.  Each engine is measured
in its own fresh process so peak CPU / peak RAM belong to that engine only.  Images are decoded outside the timed
region, one at a time (RAM is not inflated by holding the set in memory).

Accuracy: overall table on the validation split (175 images, 8 classes), per-class table on the test split.

Outputs (project root):
  results_paper_metrics.csv     one row per model x precision (footprint, latency, FPS, CPU, RAM, accuracy, power ...)
  results_paper_per_class.csv   per-class precision / recall / F1 / AP, for validation and test splits
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENG = os.path.join(HERE, "engines")
DS = os.path.join(ROOT, "dataset_v3")
VAL_DIR, TEST_DIR = os.path.join(DS, "valid"), os.path.join(DS, "test")
VAL_ANN = os.path.join(VAL_DIR, "_annotations.coco.json")
TEST_ANN = os.path.join(TEST_DIR, "_annotations.coco.json")
PAPER_CLASSES = ["aluminium_foil", "bottle", "cement_dust", "floor", "obstacle", "paper", "wooden_dust", "wrapper"]
PRECS = ["fp16", "int8"]
# key, label in the tables, wrapper type, engine base name, ONNX used for the parameter count
MODELS = [
    ("yolov8", "YOLOv8n", "yolo", "best", "best.onnx"),
    ("yolo11seg", "YOLOv11n-seg (box head)", "yolo", "yolo11seg", "yolo11seg.onnx"),
    ("yolo26", "YOLO26n (COCO, reference)", "yolo", "yolo26n", "yolo26n.onnx"),
    ("rfdetr", "RF-DETR Medium", "rfdetr", "rfdetr", "rfdetr_slim.onnx"),
]
KNOWN_FAILURES = {("rfdetr", "int8"): "INT8 engine build failed: TensorRT internal error "
                                      "('region should have been removed from Graph::regions')",
                  ("yolo11seg", "int8"): "INT8 engine build failed: TensorRT 'Could not find any implementation' "
                                         "for the mask-head node /model.23/proto/cv3"}
NOTES = {"yolov8": "all 8 classes scored",
         "yolo11seg": "box head decoded and scored; mask head not evaluated; all 8 classes scored",
         "yolo26": "generic COCO model (not trained on the dataset): only 'bottle' can be scored",
         "rfdetr": "all 8 classes scored"}
NA = "n/a"


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "min": float(a.min()), "max": float(a.max())}


# ═════════════════════════════════════════════════════════════════════════════
# worker: measure ONE engine (runs in its own process)
# ═════════════════════════════════════════════════════════════════════════════
def worker(a):
    import resource
    import threading
    import types

    import cv2
    import psutil

    sys.path.insert(0, HERE)
    import benchmark as bm
    from names_util import resolve_names

    key, prec = a.worker
    _, label, typ, base, _onnx = next(m for m in MODELS if m[0] == key)
    path = os.path.join(ENG, f"{base}_{prec}.engine")
    names, _ = resolve_names(None, path, None)
    args = types.SimpleNamespace(conf=0.25, iou=0.45, rfdetr_num_classes=None)

    gt_val = bm.load_ground_truth(VAL_DIR, VAL_ANN)
    gt_test = bm.load_ground_truth(TEST_DIR, TEST_ANN)
    paths = [p for p, _ in gt_val["images"]]
    eng = bm.load_engine(typ, path, args)

    def load(i):                                              # decoded outside the timed region
        return cv2.imread(paths[i % len(paths)])

    # cold first frame (includes lazy CUDA init)
    first = load(0)
    t0 = time.perf_counter()
    blob, meta = eng.preprocess(first)
    out, _ = eng.infer(blob)
    eng.postprocess(out, meta)
    first_ms = (time.perf_counter() - t0) * 1000

    def one(img):
        t = time.perf_counter()
        blob, meta = eng.preprocess(img)
        t_pre = time.perf_counter()
        out, it = eng.infer(blob)
        t_post0 = time.perf_counter()
        eng.postprocess(out, meta)
        t_end = time.perf_counter()
        return ((t_end - t) * 1000, it["infer_ms"], (t_pre - t) * 1000, (t_end - t_post0) * 1000)

    for i in range(a.warmup):
        one(load(i))

    # timed passes with process CPU/RSS sampling and tegrastats (power, GPU %, temperature)
    proc = psutil.Process()
    proc.cpu_percent(None)
    cpu_samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            time.sleep(0.1)
            cpu_samples.append(proc.cpu_percent(None))       # % of ONE core, summed over threads (can exceed 100)

    th = threading.Thread(target=sample, daemon=True)
    mon = bm.ResourceMonitor(interval_ms=200)
    mon.start()
    th.start()
    wall0 = time.perf_counter()
    rows = [one(load(i)) for i in range(a.timed)]
    wall = time.perf_counter() - wall0
    stop.set()
    th.join(timeout=2)
    mon.stop()
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # KB -> MB (Linux), before accuracy stage
    rows = np.array(rows)
    e2e, gpu, pre, post = (_stats(rows[:, i]) for i in range(4))
    res = mon.summarize()
    pwr = res.get("power_avg_w")
    fps_e2e = 1000.0 / e2e["mean"]

    out = {
        "model": label, "key": key, "precision": prec.upper(), "status": "ok",
        "engine_mb": os.path.getsize(path) / 1048576, "engine_load_ms": eng.load_time * 1000,
        "input_size": f"{eng.input_w}x{eng.input_h}", "first_frame_ms": first_ms,
        "latency_e2e": e2e, "latency_gpu": gpu, "preprocess_ms": pre["mean"], "postprocess_ms": post["mean"],
        "fps_e2e": fps_e2e, "fps_gpu_only": 1000.0 / gpu["mean"],
        "peak_cpu_pct_process": float(max(cpu_samples)) if cpu_samples else None,
        "avg_cpu_pct_process": float(np.mean(cpu_samples)) if cpu_samples else None,
        "peak_rss_mb_process": peak_rss_mb,
        "gpu_util_avg_pct": res.get("gpu_util_avg"), "gpu_util_peak_pct": res.get("gpu_util_max"),
        "power_avg_w": pwr, "power_peak_w": res.get("power_peak_w"),
        "temp_gpu_max_c": res.get("temp_gpu_max"), "temp_cpu_max_c": res.get("temp_cpu_max"),
        "fps_per_watt": (fps_e2e / pwr) if pwr else None,
        "mj_per_frame": (pwr / fps_e2e * 1000) if pwr else None,
        "timed_wall_s": wall, "n_timed": a.timed, "n_warmup": a.warmup,
        "n_val_images": len(gt_val["images"]), "n_test_images": len(gt_test["images"]),
    }
    if not a.no_accuracy:
        det = bm.make_trt_detector(eng, 0.001)
        print("  accuracy on the validation split ...", flush=True)
        out["accuracy"] = bm.evaluate_accuracy(det, names, gt_val, 0.25, tag=f"{key}/{prec}/val")
        print("  accuracy on the test split ...", flush=True)
        out["accuracy_test"] = bm.evaluate_accuracy(det, names, gt_test, 0.25, tag=f"{key}/{prec}/test")
    eng.close()
    print("PAPER_JSON:" + json.dumps(out, default=str), flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# master: run every engine, build the CSVs
# ═════════════════════════════════════════════════════════════════════════════
def onnx_params_m(path):
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        return sum(int(np.prod(i.dims)) for i in m.graph.initializer) / 1e6
    except Exception:                                         # noqa: BLE001
        return None


def g(d, *path):
    for p in path:
        if not isinstance(d, dict) or d.get(p) is None:
            return None
        d = d[p]
    return d


def f(v, nd=3):
    return NA if v is None else (round(v, nd) if isinstance(v, (int, float)) else v)


def master(a):
    results = []
    for key, label, typ, base, onnx_name in MODELS:
        if a.models and key not in a.models:
            continue
        params = onnx_params_m(os.path.join(ENG, onnx_name))
        for prec in PRECS:
            path = os.path.join(ENG, f"{base}_{prec}.engine")
            print(f"\n{'=' * 70}\n {label} — {prec.upper()}\n{'=' * 70}", flush=True)
            r = {"model": label, "key": key, "precision": prec.upper(), "params_m": params,
                 "onnx_mb": os.path.getsize(os.path.join(ENG, onnx_name)) / 1048576
                 if os.path.exists(os.path.join(ENG, onnx_name)) else None}
            if not os.path.exists(path):
                r["status"] = KNOWN_FAILURES.get((key, prec), "no engine file")
                print("  skipped:", r["status"])
                results.append(r)
                continue
            p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--worker", key, prec,
                                "--warmup", str(a.warmup), "--timed", str(a.timed)],
                               capture_output=True, text=True)
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PAPER_JSON:")), None)
            if p.returncode != 0 or line is None:
                r["status"] = "measurement failed: " + (p.stderr.strip().splitlines() or ["unknown"])[-1][:200]
                print("  FAILED:", r["status"])
            else:
                r.update(json.loads(line[len("PAPER_JSON:"):]))
                print(f"  e2e {r['latency_e2e']['mean']:.2f} ms  FPS {r['fps_e2e']:.1f}  "
                      f"val mAP50 {f(g(r, 'accuracy', 'mAP50'))}  test mAP50 {f(g(r, 'accuracy_test', 'mAP50'))}  "
                      f"power {f(r.get('power_avg_w'), 2)} W", flush=True)
            results.append(r)
            time.sleep(a.cooldown)

    # paper-style "FPS pre-quantisation (FP16) / post-quantisation (INT8)" per model
    fps = {(r["key"], r["precision"]): r.get("fps_e2e") for r in results if r.get("status") == "ok"}
    cols = ["model", "precision", "status", "engine_mb", "onnx_mb", "params_m", "input_size",
            "quantisation", "weights_format", "train_epochs_batch_optimizer_time",
            "engine_load_ms", "first_frame_ms",
            "latency_mean_ms", "latency_std_ms", "latency_median_ms", "latency_p95_ms", "latency_p99_ms",
            "latency_min_ms", "latency_max_ms", "gpu_infer_mean_ms", "gpu_infer_std_ms",
            "preprocess_mean_ms", "postprocess_mean_ms",
            "fps_pre_quant_fp16", "fps_post_quant_int8", "fps_this_row", "fps_gpu_only",
            "peak_cpu_pct_process", "avg_cpu_pct_process", "peak_ram_mb_process",
            "gpu_util_avg_pct", "gpu_util_peak_pct", "power_avg_w", "power_peak_w", "fps_per_watt",
            "mj_per_frame", "temp_gpu_max_c", "temp_cpu_max_c",
            "val_precision@0.25", "val_recall@0.25", "val_f1@0.25",
            "val_mAP@0.5", "val_mAP@0.75", "val_mAP@0.5:0.95",
            "test_precision@0.25", "test_recall@0.25", "test_f1@0.25",
            "test_mAP@0.5", "test_mAP@0.75", "test_mAP@0.5:0.95",
            "classes_scored", "val_images", "test_images", "warmup_passes", "timed_passes", "notes"]
    out1 = os.path.join(ROOT, "results_paper_metrics.csv")
    with open(out1, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            ok = r.get("status") == "ok"
            va, te = r.get("accuracy") or {}, r.get("accuracy_test") or {}
            lat, gpu = r.get("latency_e2e") or {}, r.get("latency_gpu") or {}
            w.writerow([r["model"], r["precision"], r.get("status"), f(r.get("engine_mb")), f(r.get("onnx_mb")),
                        f(r.get("params_m")), r.get("input_size", NA), r["precision"] + " (TensorRT)",
                        "TensorRT engine", "n/a (training-time, not a quantised-model metric)",
                        f(r.get("engine_load_ms"), 1), f(r.get("first_frame_ms"), 1),
                        f(lat.get("mean")), f(lat.get("std")), f(lat.get("median")), f(lat.get("p95")),
                        f(lat.get("p99")), f(lat.get("min")), f(lat.get("max")),
                        f(gpu.get("mean")), f(gpu.get("std")), f(r.get("preprocess_ms")), f(r.get("postprocess_ms")),
                        f(fps.get((r["key"], "FP16")), 1), f(fps.get((r["key"], "INT8")), 1),
                        f(r.get("fps_e2e"), 1), f(r.get("fps_gpu_only"), 1),
                        f(r.get("peak_cpu_pct_process"), 1), f(r.get("avg_cpu_pct_process"), 1),
                        f(r.get("peak_rss_mb_process"), 1), f(r.get("gpu_util_avg_pct"), 1),
                        f(r.get("gpu_util_peak_pct"), 1), f(r.get("power_avg_w"), 2), f(r.get("power_peak_w"), 2),
                        f(r.get("fps_per_watt"), 2), f(r.get("mj_per_frame"), 1), f(r.get("temp_gpu_max_c"), 1),
                        f(r.get("temp_cpu_max_c"), 1),
                        f(va.get("precision_at_conf")), f(va.get("recall_at_conf")), f(va.get("f1_at_conf")),
                        f(va.get("mAP50")), f(va.get("mAP75")), f(va.get("mAP50_95")),
                        f(te.get("precision_at_conf")), f(te.get("recall_at_conf")), f(te.get("f1_at_conf")),
                        f(te.get("mAP50")), f(te.get("mAP75")), f(te.get("mAP50_95")),
                        ";".join(va.get("eval_classes", [])) or NA, r.get("n_val_images", NA),
                        r.get("n_test_images", NA), r.get("n_warmup", NA), r.get("n_timed", NA),
                        NOTES.get(r["key"], "") if ok else r.get("status", "")])

    out2 = os.path.join(ROOT, "results_paper_per_class.csv")
    with open(out2, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "precision_mode", "split", "class", "precision", "recall", "f1", "AP@0.5",
                    "AP@0.5:0.95", "tp", "fp", "fn", "note"])
        for r in results:
            if r.get("status") != "ok":
                continue
            for split, acc in (("test", r.get("accuracy_test") or {}), ("valid", r.get("accuracy") or {})):
                prf, ap = acc.get("per_class_prf") or {}, acc.get("per_class") or {}
                scored = set(acc.get("eval_classes", []))
                for c in PAPER_CLASSES:
                    if c in scored:
                        p_, a_ = prf.get(c, {}), ap.get(c, {})
                        w.writerow([r["model"], r["precision"], split, c, f(p_.get("precision")), f(p_.get("recall")),
                                    f(p_.get("f1")), f(a_.get("AP50")), f(a_.get("AP50_95")),
                                    p_.get("tp", NA), p_.get("fp", NA), p_.get("fn", NA), ""])
                    else:
                        w.writerow([r["model"], r["precision"], split, c] + [NA] * 8 +
                                   ["model cannot predict this class (COCO labels)"])
    json.dump(results, open(os.path.join(HERE, "results_paper_metrics_raw.json"), "w"), indent=2, default=str)
    print(f"\nWrote {out1}\n      {out2}")


def latency_master(a):
    """Latency / FPS / power / CPU / RAM only, repeated --repeats times per engine (run it with clocks locked);
    reports the mean of the runs and the standard deviation ACROSS runs."""
    metric_defs = [  # column name, getter on one run's JSON
        ("latency_mean_ms", lambda r: r["latency_e2e"]["mean"]),
        ("latency_std_within_run_ms", lambda r: r["latency_e2e"]["std"]),
        ("latency_p95_ms", lambda r: r["latency_e2e"]["p95"]),
        ("gpu_infer_mean_ms", lambda r: r["latency_gpu"]["mean"]),
        ("preprocess_mean_ms", lambda r: r["preprocess_ms"]),
        ("postprocess_mean_ms", lambda r: r["postprocess_ms"]),
        ("fps_e2e", lambda r: r["fps_e2e"]),
        ("fps_gpu_only", lambda r: r["fps_gpu_only"]),
        ("first_frame_ms", lambda r: r["first_frame_ms"]),
        ("peak_cpu_pct_process", lambda r: r["peak_cpu_pct_process"]),
        ("peak_ram_mb_process", lambda r: r["peak_rss_mb_process"]),
        ("gpu_util_avg_pct", lambda r: r["gpu_util_avg_pct"]),
        ("power_avg_w", lambda r: r["power_avg_w"]),
        ("fps_per_watt", lambda r: r["fps_per_watt"]),
        ("mj_per_frame", lambda r: r["mj_per_frame"]),
        ("temp_gpu_max_c", lambda r: r["temp_gpu_max_c"]),
    ]
    out_rows = []
    for key, label, typ, base, onnx_name in MODELS:
        if a.models and key not in a.models:
            continue
        for prec in PRECS:
            if not os.path.exists(os.path.join(ENG, f"{base}_{prec}.engine")):
                continue
            runs = []
            for k in range(a.repeats):
                print(f"{label} {prec.upper()}  repeat {k + 1}/{a.repeats}", flush=True)
                p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--worker", key, prec,
                                    "--warmup", str(a.warmup), "--timed", str(a.timed), "--no-accuracy"],
                                   capture_output=True, text=True)
                line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PAPER_JSON:")), None)
                if p.returncode == 0 and line:
                    r = json.loads(line[len("PAPER_JSON:"):])
                    runs.append(r)
                    print(f"   e2e {r['latency_e2e']['mean']:.2f} ms  gpu {r['latency_gpu']['mean']:.2f} ms  "
                          f"FPS {r['fps_e2e']:.1f}  {r['power_avg_w']:.2f} W", flush=True)
                else:
                    print("   FAILED:", (p.stderr.strip().splitlines() or ["unknown"])[-1][:160], flush=True)
                time.sleep(a.cooldown)
            if runs:
                row = {"model": label, "precision": prec.upper(), "repeats_ok": len(runs)}
                for name, fn in metric_defs:
                    vals = [fn(r) for r in runs if fn(r) is not None]
                    row[name + "_mean_of_runs"] = round(float(np.mean(vals)), 3) if vals else NA
                    row[name + "_std_across_runs"] = round(float(np.std(vals, ddof=1)), 3) if len(vals) > 1 else NA
                out_rows.append(row)
    path = os.path.join(ROOT, "results_paper_latency_locked_clocks.csv")
    if out_rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", nargs=2, metavar=("KEY", "PREC"), help=argparse.SUPPRESS)
    ap.add_argument("--models", nargs="*", help="subset of: " + ", ".join(m[0] for m in MODELS))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--cooldown", type=int, default=15, help="seconds between engines")
    ap.add_argument("--latency-only", action="store_true", help="skip accuracy; repeat latency/power runs")
    ap.add_argument("--repeats", type=int, default=3, help="runs per engine in --latency-only mode")
    ap.add_argument("--no-accuracy", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(a)
    elif a.latency_only:
        latency_master(a)
    else:
        master(a)
