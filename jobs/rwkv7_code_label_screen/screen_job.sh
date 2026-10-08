#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} != --dry-run || $# != 2 ]]; then
  echo 'CPU preparation only: usage screen_job.sh --dry-run OUT' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"${PYTHON:-python}" -B -X utf8 -m jobs.rwkv7_code_label_screen.runner dry-run --out "$2"
