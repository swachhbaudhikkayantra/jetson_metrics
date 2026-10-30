#!/usr/bin/env bash
# ============================================================================
# run_both.sh — run the ORIGINAL and the CORRECTED benchmark on the same engines + data
#
#   Step 1 (once):   bash run_both.sh export     # builds engines with the corrected exporter
#   Step 2:          bash run_both.sh bench      # runs both benchmarks
#   (or both:)       bash run_both.sh all
#
# Configure with environment variables (defaults shown):
#   MODELS=<auto>                          folder holding Yolov8/ Yolov11/ Rfdetr/ (auto-detects
#                                          ../Models*/Models, else ../Models)
#   ORIG_DIR=../sby-deploy/deployment      original code
#   ENGINES=./engines                      where engines are written / read
#   SOURCE=./data/images                   image folder (or video) for timing
#   VAL_DIR=                               validation images (optional; empty = no accuracy)
#   VAL_ANN=                               COCO json for VAL_DIR (optional; else YOLO .txt labels)
#   RFDETR_NAMES=./rfdetr_names.json       10 names, indexed by RF-DETR output index
#   NUM_FRAMES=500  WARMUP=50  CONF=0.25  IOU=0.45
#
# Outputs:  results_original/   results_corrected/   logs/
# ============================================================================
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1          # never write __pycache__ into the ORIGINAL folder
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${MODELS:-}" ]; then
    MODELS="$(ls -d "$HERE"/../Models*/Models 2>/dev/null | head -1)"
    MODELS="${MODELS:-$HERE/../Models}"
fi
ORIG_DIR="${ORIG_DIR:-$HERE/../sby-deploy/deployment}"
ENGINES="${ENGINES:-$HERE/engines}"
SOURCE="${SOURCE:-$HERE/data/images}"
VAL_DIR="${VAL_DIR:-}"
VAL_ANN="${VAL_ANN:-}"
RFDETR_NAMES="${RFDETR_NAMES:-$HERE/rfdetr_names.json}"
NUM_FRAMES="${NUM_FRAMES:-500}"
WARMUP="${WARMUP:-50}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

stage="${1:-all}"

do_export() {
    local names_arg=()
    [ -f "$RFDETR_NAMES" ] && names_arg=(--rfdetr-names "$RFDETR_NAMES")
    ( cd "$HERE" && python3 export_engines.py \
        --yolov8 "$MODELS/Yolov8/best.pt" \
        --yolo26 "$MODELS/Yolov11/yolo26n.pt" \
        --rfdetr-onnx "$MODELS/Rfdetr/Copy of rfdetr_fp32.onnx" \
        "${names_arg[@]}" --out-dir "$ENGINES" ) 2>&1 | tee "$LOGS/export.log"
}

do_bench() {
    local val_orig=() val_corr=()
    if [ -n "$VAL_DIR" ]; then
        val_orig=(--val-dir "$VAL_DIR"); val_corr=(--val-dir "$VAL_DIR")
        [ -n "$VAL_ANN" ] && { val_orig+=(--val-ann "$VAL_ANN"); val_corr+=(--val-ann "$VAL_ANN"); }
    fi
    local names_arg=()
    [ -f "$RFDETR_NAMES" ] && names_arg=(--rfdetr-names "$RFDETR_NAMES")

    echo "=== ORIGINAL benchmark -> $HERE/results_original ==="
    ( cd "$ORIG_DIR" && python3 benchmark.py --mode headless \
        --yolov8-engine "$ENGINES/best_fp16.engine"   --yolov8-pt "$MODELS/Yolov8/best.pt" \
        --yolo11-engine "$ENGINES/yolo26n_fp16.engine" --yolo11-pt "$MODELS/Yolov11/yolo26n.pt" \
        --rfdetr-engine "$ENGINES/rfdetr_fp16.engine" --rfdetr-num-classes 10 \
        --source "$SOURCE" --num-frames "$NUM_FRAMES" --warmup "$WARMUP" \
        --conf "$CONF" --iou "$IOU" "${val_orig[@]}" \
        --out-dir "$HERE/results_original" ) 2>&1 | tee "$LOGS/original.log"

    echo "=== CORRECTED benchmark -> $HERE/results_corrected ==="
    ( cd "$HERE" && python3 benchmark.py --mode headless \
        --yolov8-engine "$ENGINES/best_fp16.engine"   --yolov8-pt "$MODELS/Yolov8/best.pt" \
        --yolo26-engine "$ENGINES/yolo26n_fp16.engine" --yolo26-pt "$MODELS/Yolov11/yolo26n.pt" \
        --rfdetr-engine "$ENGINES/rfdetr_fp16.engine" "${names_arg[@]}" --rfdetr-num-classes 10 \
        --source "$SOURCE" --num-frames "$NUM_FRAMES" --warmup "$WARMUP" \
        --conf "$CONF" --iou "$IOU" "${val_corr[@]}" \
        --out-dir "$HERE/results_corrected" ) 2>&1 | tee "$LOGS/corrected.log"

    echo
    echo "Compare:  $HERE/results_original/results.json   vs   $HERE/results_corrected/results.json"
}

case "$stage" in
    export) do_export;;
    bench)  do_bench;;
    all)    do_export || echo "[WARN] export reported problems — benchmarking whichever engines exist"
            do_bench;;
    *)      echo "usage: bash run_both.sh [export|bench|all]"; exit 2;;
esac
