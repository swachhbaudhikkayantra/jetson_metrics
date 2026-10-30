# Benchmark Summary

Generated: 20261030_211354

Board: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super  |  L4T: # R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic, EABI: aarch64, DATE: Mon Jun 16 16:07:13 UTC 2025  |  JetPack: None  |  TensorRT: 10.3.0

## Winners by Metric

| Metric | Winner | Value |
|---|---|---|
| Lowest inference latency (ms) | **yolov8** | 8.545 |
| Highest pipeline FPS | **yolov8** | 53.141 |
| Best mAP@0.5 (TRT, only where scored) | **yolov8** | 0.846 |
| Best FPS/W | **yolov8** | 5.098 |
| Smallest engine (MB) | **yolov8** | 4.599 |

## All Metrics

| Metric | yolov8 | yolo26 |
|---|---|---|
| Infer Latency (ms) | 8.55 | 11.74 |
| Pipeline Latency (ms) | 18.82 | 23.07 |
| E2E Latency incl. capture (ms) | 21.66 | 25.97 |
| Infer FPS | 117.0 | 85.2 |
| Pipeline FPS | 53.1 | 43.4 |
| E2E FPS incl. capture | 46.2 | 38.5 |
| P99 Total Latency (ms) | 24.26 | 39.08 |
| First Frame (ms) | 53.1 | 86.0 |
| Jitter Std (ms) | 1.12 | 4.00 |
| Stutter Count | 0 | 0 |
| mAP@0.5 (TRT) | 0.846 | 0.480 |
| mAP@0.5:0.95 (TRT) | 0.611 | 0.303 |
| mAP@0.5 (PyTorch) | n/a | n/a |
| Precision @conf | 0.794 | 1.000 |
| Recall @conf | 0.767 | 0.357 |
| GPU Util % | 40.1 | 53.5 |
| Power Avg (W, board) | 9.05 | 9.26 |
| FPS/W (wall-clock) | 5.10 | 4.16 |
| Energy / frame (mJ) | 196.2 | 240.7 |
| Temp GPU Max (C) | 62.8 | 62.8 |
| PT/TRT speedup (infer) | n/a | n/a |
| PT/TRT speedup (e2e) | n/a | n/a |
| Engine Size (MB) | 4.6 | 5.4 |
| Engine Load (ms) | 65 | 68 |

Notes: power is whole-module VDD_IN from tegrastats; mAP is only comparable between models scored on the same classes (see `accuracy.eval_classes` in results.json).
