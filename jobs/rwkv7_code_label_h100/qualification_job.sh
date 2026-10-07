#!/usr/bin/env bash
# No paid execution is possible with the committed launch_enabled=false params.
set -euo pipefail
CODE=${SSDM_CODE_ROOT:-/opt/ssdm/m1}
PYTHON=${PYTHON:-python}
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
if [ "${1:-}" = --dry-run ] && [ "$#" = 2 ]; then
  cd "$(dirname "$0")/../.."
  exec "$PYTHON" -B -X utf8 -m jobs.rwkv7_code_label_h100.runner dry-run --out "$2"
fi
if [ "${1:-}" != --batch ] || [ "$#" != 1 ]; then
  echo 'usage: qualification_job.sh --dry-run OUT | --batch'
  exit 2
fi
cd "$CODE"
# Admission occurs before CUDA import, private rows or downloads.
"$PYTHON" -m jobs.rwkv7_code_label_h100.runner admit
OUT=${GMN_OUTPUT_DIR:-/output}
mkdir -p "$OUT"
cap() { "$PYTHON" -c 'import json,sys; print(json.load(open("jobs/rwkv7_code_label_h100/job_params.json"))["phase_caps_minutes"][sys.argv[1]])' "$1"; }
finish() {
  local rc=$?
  trap - EXIT
  timeout "$(cap export)m" "$PYTHON" -m jobs.rwkv7_code_label_h100.runner finalize --out "$OUT" --rc "$rc" || exit 6
  exit "$rc"
}
trap finish EXIT
timeout "$(cap setup)m" "$PYTHON" -m jobs.rwkv7_code_label_h100.runner setup --out "$OUT"
timeout "$(cap probe)m" "$PYTHON" -m jobs.rwkv7_code_label_h100.runner probe --out "$OUT"
python -m jobs.rwkv7_code_label_h100.preflight --probe "$OUT/probe.json" --out "$OUT/preflight.json"
timeout "$(cap screen)m" "$PYTHON" -m jobs.rwkv7_code_label_h100.runner screen --out "$OUT"
timeout "$(cap audit)m" "$PYTHON" -m jobs.rwkv7_code_label_h100.runner audit --out "$OUT"
