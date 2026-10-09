#!/usr/bin/env bash
# q9af_corpus_check: rebuild job-w3c2e's distillation mix (convert.mixdata, dev13-v1, configs/q9af_v2_h100.json) on a
# cpu-8 node and scan EVERY document / window against the pinned eval-hash index (DEV + TEST + held-out), the frozen
# catalogs and the eval attack templates. Spec: jobs/q9af_corpus_check/JOB_SPEC.md. Caps: job_params.json.
#
# Phases: setup (pins, data sha256, measured-rate preflight)   fetch (scripts/sf_fetch.py --stage q9af_corpus_check:
#   the 3 pinned parquet sources + the 9B tokenizer files, no weights)   C (jobs.q9af_corpus_check.corpus_check run:
#   rebuild + streaming scan + report).
# Exit codes (also in $GMN_RESULT_PATH): 0 census written (CLEAN or CONTAMINATED)   5 setup failed   7 phase timeout
#   9 report invalid   10 data sha256   11 fetch failed   12 rebuild / scan crashed   14 eval index refused
#   15 scanned docs do not tile the rebuilt stream   16 rebuild != the pinned w3c2e mix   1 finalize failed
# /output/Q9CC_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
# Q9CC_DRYRUN=1 (CPU, tests/test_q9af_corpus_check.py): this very script with the fetch replaced by
# tests/q9af_corpus_check_stub.py (tiny synthetic sources, a synthetic eval index, a whitespace tokenizer); the setup
# checks, the preflight, the rebuild + scan and finalize run for real. Refused inside the image.
set -Euo pipefail
MAIN_PID=$$
CODE=/opt/ssdm/m1
MANIFEST=/opt/ssdm/CONTEXT_MANIFEST.json
DATA=/data
WORK=/tmp/q9cc
DRY=""
if [ -n "${Q9CC_DRYRUN:-}" ]; then
  if [ -d /opt/ssdm/m1 ] || [ -z "${Q9CC_DRYRUN_ROOT:-}" ]; then echo "Q9CC_DRYRUN refused here"; exit 5; fi
  CODE=$(cd "$(dirname "$0")/../.." && pwd)
  MANIFEST=$Q9CC_DRYRUN_ROOT/CONTEXT_MANIFEST.json
  DATA=$Q9CC_DRYRUN_ROOT/data
  WORK=$Q9CC_DRYRUN_ROOT/work
  DRY="python -X utf8 -m tests.q9af_corpus_check_stub"
fi
PARAMS=$CODE/jobs/q9af_corpus_check/job_params.json
CFG=configs/q9af_v2_h100.json
CHECK_CFG=$CFG
INDEX=$DATA/doc-overlap/218affe13e367d07
DATA_SHA=$CODE/jobs/q9af_corpus_check/data.sha256
EXPECT=jobs/q9af_corpus_check/w3c2e_mix_pin.json
SEAMS=""
OUT=${GMN_OUTPUT_DIR:-/output}
if [ -n "$DRY" ]; then
  DATA_SHA=$Q9CC_DRYRUN_ROOT/data.sha256
  CHECK_CFG=$Q9CC_DRYRUN_ROOT/config.json
  SEAMS="--fake-tokenizer --index-sha $(cat "$Q9CC_DRYRUN_ROOT/index.sha")"
  OUT=$Q9CC_DRYRUN_ROOT/output
fi
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
CK=$WORK
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200
[ -z "$DRY" ] || HB_EVERY_S=1
WORKERS=${Q9CC_WORKERS:-6}
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs"
export SSDM_DATA_ROOT=$WORK/data SSDM_DATA=$DATA PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=true
mkdir -p "$SSDM_DATA_ROOT"
if [ -n "${HF_HUB_CACHE:-}" ] && [ -d "${HF_HUB_CACHE}" ]; then ln -sfn "$HF_HUB_CACHE" "$SSDM_DATA_ROOT/hf"; else mkdir -p "$SSDM_DATA_ROOT/hf"; fi
HF_DIR=$(readlink -f "$SSDM_DATA_ROOT/hf")
WATCH_EXTRA=$WORK
T0=$(date +%s)
EXPORT_FAIL=0
RC=""
GATES="{}"
exec > >(tee -a "$WORK/logs/run.log") 2>&1
cd "$CODE"
# shellcheck source=../common/lib.sh
. "$CODE/jobs/common/lib.sh"
# shellcheck source=../clef_baseline/watchdog.sh
. "$CODE/jobs/clef_baseline/watchdog.sh"
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  for f in report.json mix.json flagged.jsonl.gz; do cp -f "$WORK/out/check/$f" "$OUT/" 2> /dev/null || true; done
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/check/report.json" "$OUT" Q9CC_DONE jobs.q9af_corpus_check.corpus_check:valid_report
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup (exit 5 / 10)
say "setup ($CFG -> $CHECK_CFG${DRY:+; CPU DRY RUN}; $WORKERS scan workers)"
cp "$MANIFEST" "$WORK/out/" || fail 5 "no context manifest"
cp "$CODE/$CFG" "$WORK/out/config.used.json"
python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env"
import importlib.metadata as m, json, os
print(json.dumps({"cpus": os.cpu_count(), **{k: m.version(k) for k in ("numpy", "pyarrow")},
                  **{k: (m.version(k) if any(d.metadata["Name"] == k for d in m.distributions()) else None)
                     for k in ("transformers", "tokenizers")}}))
EOF
CAP_S=$(cap setup) && CAP_F=$(cap fetch) && CAP_C=$(cap C) || fail 5 "job_params"
# measured-rate preflight: every cap vs its projection, schedule, hard cap, the config is w3c2e's
python -m jobs.q9af_corpus_check.preflight --params "$PARAMS" --config "$CFG" --out "$WORK/out/preflight.json" \
  > "$WORK/logs/preflight.log" 2>&1; r=$?
gate "setup:preflight" $r
[ $r -eq 0 ] || fail 5 "preflight: a cap, the schedule or the config violates the margin rule"
( cd "$DATA" && sha256sum -c "$DATA_SHA" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows / eval index sha256"
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ fetch (exit 11)
say "fetch (pinned parquet sources + 9B tokenizer files)"
if [ -n "$DRY" ]; then
  $DRY fetch "$SSDM_DATA_ROOT" > "$WORK/logs/fetch.log" 2>&1; r=$?
else
  timeout "${CAP_F}m" python -I scripts/sf_fetch.py --stage q9af_corpus_check > "$WORK/logs/fetch.log" 2>&1; r=$?
fi
gate "fetch" $r
for d in FINEWEB_EDU_DIR ULTRACHAT_DIR FINEMATH_DIR; do [ -f "$SSDM_DATA_ROOT/DONE_$d" ] || fail 11 "fetch $d"; done
[ -n "$DRY" ] || [ -f "$SSDM_DATA_ROOT/DONE_Q9B_DIR" ] || fail 11 "fetch 9B tokenizer"
# shellcheck disable=SC1091
. "$SSDM_DATA_ROOT/paths.env"
export FINEWEB_EDU_DIR ULTRACHAT_DIR FINEMATH_DIR
export_phase fetch

# ------------------------------------------------------------------ C: rebuild + streaming scan of every window
say "phase C (rebuild w3c2e's mix + scan vs DEV/TEST/held-out hashes, frozen catalogs, attack templates)"
# shellcheck disable=SC2086  (SEAMS is a word list, empty outside the dry run)
timeout "${CAP_C}m" python -X utf8 -m jobs.q9af_corpus_check.corpus_check run --config "$CHECK_CFG" \
  --dev-rows "$DATA/slices/dev-rows.jsonl.gz" --index-dir "$INDEX" --mix "$WORK/mix" --out "$WORK/out/check" \
  --workers "$WORKERS" --expect "$EXPECT" $SEAMS > "$WORK/logs/check.log" 2>&1; r=$?
gate "C:check" $r
tail -1 "$WORK/logs/check.log"
export_phase C
case $r in 0) ;; 124) fail 7 "check timeout" ;; 14|15|16) fail $r "$(tail -1 "$WORK/logs/check.log")" ;; *) fail 12 "check crashed rc=$r" ;; esac
RC=0
exit 0
