#!/usr/bin/env bash
# ============================================================================
# setup.sh (corrected) — prepare a Jetson Orin Nano (JetPack 5.x / 6.x)
# Run as a NORMAL user (sudo is used only for apt):   bash setup.sh
#
# Principles: never replace JetPack-provided torch / tensorrt / OpenCV / numpy,
#             never hide a failure, tell the user exactly what is missing.
# ============================================================================
set -uo pipefail

FAILS=0
TORCH_OK=0
fail() { echo "  [FAIL] $*"; FAILS=$((FAILS + 1)); }
ok()   { echo "  [ OK ] $*"; }
warn() { echo "  [WARN] $*"; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

echo "============================================================"
echo " Jetson Orin Nano Benchmark — Environment Setup (corrected)"
echo "============================================================"

# ---------- Platform ----------
ARCH=$(uname -m)
if [ -f /etc/nv_tegra_release ]; then
    L4T=$(head -1 /etc/nv_tegra_release)
    echo "[INFO] L4T: $L4T"
    case "$L4T" in
        *"R35"*) echo "[INFO] => JetPack 5.x (Python 3.8, TensorRT 8.5)";;
        *"R36"*) echo "[INFO] => JetPack 6.x (Python 3.10, TensorRT 8.6 / 10.x)";;
        *)       warn "unrecognised L4T release";;
    esac
else
    warn "/etc/nv_tegra_release not found — not a Jetson (dry run only)"
fi
JP=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || true)
echo "[INFO] nvidia-jetpack: ${JP:-not installed}   arch: $ARCH   python: $(python3 --version 2>&1)"
[ "$ARCH" = "aarch64" ] || warn "not aarch64 — NVIDIA wheels will not match"

export PATH="/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# ---------- [1/6] system packages (only those NOT supplied by JetPack) ----------
echo; echo "[1/6] System packages"
$SUDO apt-get update -qq || warn "apt-get update failed"
for pkg in python3-pip python3-dev build-essential cmake pkg-config v4l-utils \
           libjpeg-dev libpng-dev unzip curl; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then continue; fi
    $SUDO apt-get install -y --no-install-recommends "$pkg" >/dev/null 2>&1 \
        && ok "installed $pkg" || fail "could not install $pkg"
done
# NOTE: libopencv-dev / python3-opencv are deliberately NOT installed — JetPack ships an OpenCV
# built with CUDA + GStreamer and Ubuntu's package would shadow it.

python3 -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || warn "pip upgrade failed"

# ---------- [2/6] torch (must be NVIDIA's CUDA build) ----------
echo; echo "[2/6] PyTorch (CUDA)"
if python3 - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("  torch import failed:", e); sys.exit(2)
print("  torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else 3)
PY
then ok "CUDA-enabled torch present"; TORCH_OK=1
else
    fail "CUDA-enabled torch missing."
    cat <<'EOF'
         Install NVIDIA's Jetson wheel (NOT from PyPI), then re-run this script:
           JetPack 5.x : https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/
           JetPack 6.x : https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/
                         (newer JetPack 6.x builds: https://pypi.jetson-ai-lab.dev)
         torchvision must be built to match (see the NVIDIA forum thread "PyTorch for Jetson").
EOF
fi

# ---------- [3/6] TensorRT python bindings (system package; do NOT pip install) ----------
echo; echo "[3/6] TensorRT"
if python3 -c "import tensorrt; print('  tensorrt', tensorrt.__version__)"; then
    ok "tensorrt python bindings present"
else
    fail "tensorrt python bindings missing. They come with JetPack:  sudo apt install python3-libnvinfer python3-libnvinfer-dev  (or nvidia-jetpack). Do NOT 'pip install tensorrt'."
fi
if command -v trtexec >/dev/null 2>&1 || [ -x /usr/src/tensorrt/bin/trtexec ]; then
    ok "trtexec found"
else
    warn "trtexec not found (optional; export_engines.py uses the Python API)"
fi

# ---------- [4/6] OpenCV (JetPack's) ----------
echo; echo "[4/6] OpenCV"
python3 - <<'PY'
import sys
try:
    import cv2
except Exception as e:
    print("  [FAIL] cv2 import failed:", e); sys.exit(2)
info = cv2.getBuildInformation()
def flag(key):
    for ln in info.splitlines():
        if ln.strip().startswith(key):
            return ln.split(":", 1)[1].strip()
    return "?"
gui = flag("GUI")
print("  cv2", cv2.__version__, "| GStreamer:", flag("GStreamer"), "| GUI:", gui)
if gui in ("", "NO", "?") or gui.startswith("NONE"):
    print("  [WARN] OpenCV has no GUI backend: --mode live (cv2.imshow) will not work")
if flag("GStreamer").startswith("NO"):
    print("  [WARN] OpenCV has no GStreamer (CSI cameras will not open). A pip 'opencv-python*' wheel "
          "is probably shadowing JetPack's build:  pip3 uninstall -y opencv-python opencv-python-headless")
PY
[ $? -eq 0 ] || fail "OpenCV not importable (JetPack provides it; try: sudo apt install python3-opencv only if JetPack's is missing)"

# ---------- [5/6] Python packages (no dependency pulling: keep JetPack's torch/numpy/cv2) ----------
echo; echo "[5/6] Python packages"
PIP_FLAGS=""
if [ "$(id -u)" -ne 0 ] && [ -z "${VIRTUAL_ENV:-}" ]; then PIP_FLAGS="--user"; fi

pipi() {  # pipi <label> <pip args...>
    local label="$1"; shift
    if python3 -m pip install --no-cache-dir $PIP_FLAGS "$@" >/tmp/pip_"${label}".log 2>&1; then
        ok "$label"
    else
        fail "$label — see /tmp/pip_${label}.log"
    fi
}

pipi numpy       "numpy<2"                                   # NVIDIA torch / TensorRT / pycuda need numpy 1.x
pipi support     tabulate matplotlib pyyaml psutil tqdm pycocotools scipy pillow requests \
                 polars py-cpuinfo onnx onnxslim
if [ "$TORCH_OK" -eq 1 ]; then
    pipi thop        --no-deps ultralytics-thop
    pipi ultralytics --no-deps ultralytics                   # --no-deps: do not pull PyPI torch / opencv
else
    warn "skipping ultralytics (needs the CUDA torch from step 2 first; otherwise pip may pull a CPU torch)"
fi
if python3 -c "import pycuda" >/dev/null 2>&1; then ok "pycuda already installed"
else pipi pycuda pycuda; fi                                  # compiles; needs nvcc on PATH (set above)

# ---------- [6/6] runtime tools ----------
echo; echo "[6/6] Runtime tools"
command -v tegrastats >/dev/null 2>&1 && ok "tegrastats found" || warn "tegrastats not found — power/temperature will be n/a"
command -v nvpmodel   >/dev/null 2>&1 && ok "nvpmodel found"   || warn "nvpmodel not found — power-mode sweep unavailable"
command -v jetson_clocks >/dev/null 2>&1 && ok "jetson_clocks found" || warn "jetson_clocks not found"
echo "  Power modes available on this device:"; (nvpmodel -q --verbose 2>/dev/null | head -5) || true
echo "  (list modes: grep -E '^< POWER_MODEL' /etc/nvpmodel.conf )"

# ---------- verification ----------
echo; echo "[Verify]"
python3 - <<'PY'
import importlib, sys
print("Python", sys.version.split()[0])
for name in ["torch", "torchvision", "tensorrt", "ultralytics", "onnx", "cv2", "pycuda",
             "pycocotools", "matplotlib", "tabulate", "numpy", "psutil", "yaml"]:
    try:
        m = importlib.import_module(name)
        print("  OK  %-12s %s" % (name, getattr(m, "__version__", "")))
    except Exception as e:
        print("  --  %-12s MISSING (%s)" % (name, str(e).splitlines()[0]))
try:
    import numpy, torch
    if int(numpy.__version__.split(".")[0]) >= 2:
        print("  WARN numpy >= 2 can break NVIDIA's torch/TensorRT/pycuda; run: pip3 install 'numpy<2'")
    print("  CUDA available:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
except Exception:
    pass
PY

echo
echo "============================================================"
if [ "$FAILS" -eq 0 ]; then
    echo " Setup complete. Next: python3 export_engines.py --help"
else
    echo " Setup finished with $FAILS problem(s) — fix the [FAIL] lines above before continuing."
fi
echo "============================================================"
exit "$FAILS"
