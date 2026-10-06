#!/usr/bin/env bash
# Single-attempt driver (no retry loop: retries spawned duplicate GPU jobs on a shared laptop, 2026-10-06).
# The pipeline is resumable: re-run this script by hand to continue from results.json.
cd "$(dirname "$0")/.."
OUT=${OUT:-receipts/2026-10-06-D-m1-pilot-0p8b}
PY=${PY:-.venv-fla/Scripts/python.exe}
LOCK=$OUT/.running.pid
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then echo "already running (pid $(cat "$LOCK"))"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
$PY -m convert.pipeline --config ${CONFIG:-configs/m1_pilot_0p8b.json} --out $OUT >> $OUT/stdout.log 2>&1
