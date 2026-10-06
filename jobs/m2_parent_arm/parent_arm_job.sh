#!/usr/bin/env bash
# jobs/m2_parent_arm: score the frozen m1 parent ONCE per gate on the frozen development cohort and cache the units
# (train.m2_gates arm cache) for every m2 cell on that parent. SF Compute batch job (chip h100). DRAFT, NOT SUBMITTED.
# Spec: jobs/m2_parent_arm/JOB_SPEC.md. Caps: jobs/m2_parent_arm/job_params.json. No training happens in this job.
# Env: PARENT=rwkv7|q9af   TOK_S=<measured prefill tok/s of this parent> (rwkv7 stage A probe / probe9b throughput)
# Exit codes: 0 cached + exports verified   2 bad PARENT / no TOK_S   5 setup   6 export verification   7 phase timeout
#   9 cache manifest missing/invalid   10 snapshot / records / cohort / pins mismatch, or download failed
#   15 projected past the cap from the MEASURED rate (nothing scored)   30/31/32 guard not ready   1 finalize failed
set -Euo pipefail
MAIN_PID=$$  # the watchdog polls it under set -u (PR #29 review B2: unset -> the watchdog died on its first iteration)
CODE=/opt/ssdm/m1
PARENT=${PARENT:-rwkv7}
case "$PARENT" in rwkv7) CELL=jobs/m2_pilot/m2_pilot_cell.json ;; q9af) CELL=jobs/m2_pilot/m2_pilot_cell_q9af.json ;;
  *) echo "PARENT must be rwkv7 or q9af"; exit 2 ;; esac
[ -n "${TOK_S:-}" ] || { echo "TOK_S (a MEASURED prefill rate) is required"; exit 2; }
PARAMS=$CODE/jobs/m2_parent_arm/job_params.json
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/m2-parent-arm
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=${HB_EVERY_S:-60} STALL_S=${STALL_S:-1200}
WATCH_EXTRA=$WORK
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}  # watchdog.sh needs CK and HF_DIR as well
HF_DIR=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs" "$CK" "$HF_DIR"
export SSDM_DATA=/data PYTHONUNBUFFERED=1 PYTHONUTF8=1
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
MARKER=M2_PARENT_ARM_DONE
VALIDATOR=jobs.m2_parent_arm.validate:valid_cache
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/arm/arm.json" "$OUT" "$MARKER" "$VALIDATOR"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

say "setup (PARENT=$PARENT)"
cp /opt/ssdm/CONTEXT_MANIFEST.json "$WORK/out/" || fail 5 "no context manifest"
# fla #640 guard: Hopper + Triton < 3.7.1 (or a non-Triton fla backward) -> exit 5 here, before any GPU phase.
python -I /opt/ssdm/m1/jobs/common/fla_guard.py --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
( cd /data && sha256sum -c "$CODE/jobs/rwkv7/records.sha256" ) > "$WORK/out/records-verify.txt" 2>&1 || fail 10 "records sha256"
( cd /data && sha256sum -c "$CODE/jobs/clef_baseline/data.sha256" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
python -X utf8 -m train.cohort verify > "$WORK/out/cohort-verify.json" 2>&1 || fail 10 "cohort does not rebuild to its committed sha256"
CAP_SETUP=$(cap setup) || fail 5 "job_params"
if [ "$PARENT" = rwkv7 ]; then
  SNAP=$(timeout ${CAP_SETUP}m python -c 'from huggingface_hub import snapshot_download as s; print(s("RWKV/RWKV7-G1k-13.3B-20260930", revision="ce3630b854cedbd673569f4600c6cf744bca9dba"))') || fail 10 "download"
  ( cd "$SNAP" && sha256sum -c "$CODE/jobs/rwkv7/snapshot.sha256" ) > "$WORK/out/snapshot-verify.txt" 2>&1 || fail 10 "snapshot sha256"
  GDN_ARGS=()
else
  SNAP=$(timeout ${CAP_SETUP}m python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a"))') || fail 10 "download"
  GDN_ARGS=(--gdn-ckpt /data/q9af/gdn_converted.pt)  # pinned by sha256 in the q9af cell config (checked in Python)
fi
python -X utf8 -m jobs.common.dev_subset rows --rows /data/slices/dev-rows.jsonl.gz --out "$WORK/index-rows.jsonl.gz" \
  --subset jobs/common/dev_subset_m1.json > "$WORK/out/index-subset.json" || fail 10 "DEV subset rows"
gate "setup" 0; export_phase setup

say "phase guard"
timeout $(cap guard)m python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir /data/suite-0.2 --kit-dir /opt/decision-index \
  --out-dir "$WORK/out/guard" > "$WORK/logs/guard.log" 2>&1; r=$?
gate "guard" $r
case $r in 0) ;; 124) fail 7 "guard timeout" ;; 30|31|32) fail $r "guard not ready" ;; *) fail 5 "guard rc=$r" ;; esac

say "phase preflight (projected minutes from the MEASURED rate TOK_S=$TOK_S)"
CAP_SCORE=$(( $(cap g12) + $(cap g13) ))
python -X utf8 -m jobs.m2_parent_arm.score_parent estimate --tok-s "$TOK_S" --cap-min "$CAP_SCORE" --fixed-min 0 \
  > "$WORK/out/preflight.json"; r=$?
gate "preflight" $r
[ $r -eq 0 ] || fail 15 "projected past the scoring cap at the measured rate; nothing scored"

for STAGE in g12 g13; do
  say "phase $STAGE (parent arm, cached)"
  timeout $(cap $STAGE)m python -X utf8 -m jobs.m2_parent_arm.score_parent --parent "$PARENT" --model-dir "$SNAP" \
    "${GDN_ARGS[@]}" --cell-config "$CELL" --records /data/m2/records.jsonl.gz \
    --guard-readiness "$WORK/out/guard/readiness.json" --index-rows "$WORK/index-rows.jsonl.gz" \
    --out-dir "$WORK/out/arm" --stage "$STAGE" > "$WORK/logs/$STAGE.log" 2>&1; r=$?
  gate "$STAGE" $r; export_phase "$STAGE"
  case $r in 0) ;; 124) fail 7 "$STAGE timeout" ;; 10) fail 10 "$STAGE identity" ;; *) fail 9 "$STAGE rc=$r" ;; esac
done
sample "$WORK/out/arm/arm.json" "arm"
RC=0
exit 0
