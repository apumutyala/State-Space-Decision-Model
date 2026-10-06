#!/usr/bin/env bash
# resumable driver: retries the pipeline (e.g. after a transient CUDA OOM caused by another GPU tenant)
cd "$(dirname "$0")/.."
OUT=receipts/2026-10-06-D-m1-pilot-0p8b
for i in 1 2 3 4 5 6; do
  .venv-fla/Scripts/python.exe -m convert.pipeline --config configs/m1_pilot_0p8b.json --out $OUT >> $OUT/stdout.log 2>&1 && exit 0
  echo "attempt $i failed $(date)" >> $OUT/stdout.log; sleep 60
done
exit 1
