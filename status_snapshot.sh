#!/usr/bin/env bash
# status_snapshot.sh — write ONE new timestamped status file into ./status/ (run by cron every 5 min).
# Absolute paths only: cron starts with a minimal environment.
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ROOT=/home/spherenex/jetson_metrics
OUT_DIR="$ROOT/status"
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUT_DIR/status_$STAMP.md"
TMP="$OUT.tmp"                      # write to .tmp then rename, so a half-written file is never seen

{
echo "# Status snapshot — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo

echo "## Models (Models/)"
if [ -d "$ROOT/Models" ]; then
    find "$ROOT/Models" -type f \( -name '*.pt' -o -name '*.onnx' \) -printf '%P  (%s bytes)\n' | sort
else
    echo "Models/ not found"
fi
echo

echo "## Engines (corrected/engines/)"
if [ -d "$ROOT/corrected/engines" ]; then
    ls -l --time-style='+%F %T' "$ROOT/corrected/engines" | tail -n +2
    [ -f "$ROOT/corrected/engines/export_meta.json" ] && echo "export_meta.json present"
else
    echo "none built yet"
fi
echo

echo "## Results"
found=0
for d in "$ROOT"/corrected/results* ; do
    [ -d "$d" ] || continue
    found=1
    echo "- ${d#$ROOT/}: $(find "$d" -type f | wc -l) files, newest $(find "$d" -type f -printf '%TY-%Tm-%Td %TH:%TM\n' | sort | tail -1)"
done
[ "$found" -eq 0 ] && echo "no results yet"
echo

echo "## Running jobs"
pgrep -af 'export_engines|benchmark\.py|run_both|trtexec|setup\.sh' | grep -v pgrep || echo "none"
echo

echo "## Recent log output (corrected/logs/)"
if ls "$ROOT"/corrected/logs/*.log >/dev/null 2>&1; then
    for f in "$ROOT"/corrected/logs/*.log; do
        echo "### $(basename "$f") (modified $(date -r "$f" '+%F %T'))"
        tail -n 5 "$f"
    done
else
    echo "no logs yet"
fi
echo

echo "## Files changed in project (last 5 min)"
find "$ROOT" -path "$ROOT/.git" -prune -o -path "$OUT_DIR" -prune -o -type f -mmin -5 -printf '%P\n' | head -20
echo

echo "## Git"
git -C "$ROOT" status --short 2>&1 | head -20
echo

echo "## System"
echo "uptime: $(uptime -p)  |  load: $(cut -d' ' -f1-3 /proc/loadavg)"
free -m | awk 'NR==1||NR==2'
df -h "$ROOT" | tail -1 | awk '{print "disk: "$3" used / "$2" ("$5")"}'
nvpmodel -q 2>/dev/null | head -2 || true
for z in /sys/class/thermal/thermal_zone*; do
    t="$(cat "$z/temp" 2>/dev/null)"
    case "$t" in ''|*[!0-9-]*) continue;; esac        # skip zones that cannot be read
    echo "temp $(cat "$z/type" 2>/dev/null): $(( t / 1000 )) C"
done
echo '```'
timeout 3 tegrastats --interval 1000 2>/dev/null | head -1 || true
echo '```'
} > "$TMP" 2>&1

mv "$TMP" "$OUT"
