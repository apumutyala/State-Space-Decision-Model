#!/usr/bin/env bash
# jobs/m2_pilot: first paid m2 cell (cache-fork head on the decision-tree parent), SF Compute batch job (chip h100).
# DRAFT, NOT SUBMITTED. Spec: jobs/m2_pilot/JOB_SPEC.md. Caps: jobs/m2_pilot/job_params.json.
# Env: PARENT=rwkv7|q9af (q9af: /data/q9af/gdn_converted.pt, sha256-pinned in jobs/m2_pilot/m2_pilot_cell_q9af.json).
#      TOK_S=<MEASURED prefill tok/s of this parent>: the evals phase refuses (exit 15) if the child arm cannot fit.
# Needs the parent arm cache of the SAME parent + cohort in /data/parent_arm (jobs/m2_parent_arm, scored once).
# Exit codes: 0 measured + verdict + exports verified   2 bad PARENT   5 setup   6 export verification   7 phase timeout
#   9 report missing/invalid   10 snapshot / vendored code / records / cohort identity mismatch, or download failed
#   30 guard: kit/suite identity   31 guard: content index != pin (new sha256 recorded)   32 guard: harness manifests
#   33 held-out D1-B template found in the training records
#   4 TOK_S missing / not a number   20 train.run_cell returned an unexpected code
#   15 evals projected past their cap from the MEASURED rate (child arm not scored; re-price)
#   40 / 41 / 42  train.m2_gates: G12 / G13 / both REJECTED (report written and exported; promotion = reject)
#   21-29 train.run_cell stops (21 fork mismatch, 22 no T fits, 23 non-finite, 24 memory, 25 data, 26 too slow,
#         27 guard not ready, 28 cohort leak/mismatch, 29 q9af lineage / unpinned conversion)
#   1 finalize failed
set -Euo pipefail
MAIN_PID=$$  # the watchdog polls it under set -u (PR #29 review B2: unset -> the watchdog died on its first iteration)
CODE=/opt/ssdm/m1
PARENT=${PARENT:-rwkv7}
case "$PARENT" in rwkv7) CELL=jobs/m2_pilot/m2_pilot_cell.json; GDN_ARGS=() ;;
  q9af) CELL=jobs/m2_pilot/m2_pilot_cell_q9af.json; GDN_ARGS=(--gdn-ckpt /data/q9af/gdn_converted.pt) ;;
  *) echo "PARENT must be rwkv7 or q9af"; exit 2 ;; esac
PARAMS=$CODE/jobs/m2_pilot/job_params.json
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/m2-pilot
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200  # fixed in paid runs (not env-overridable)
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs" "$CK"
export SSDM_DATA=/data PYTHONUNBUFFERED=1 PYTHONUTF8=1
HF_DIR=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p "$HF_DIR"
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
MARKER=M2_PILOT_DONE
VALIDATOR=train.m2_gates:valid_report
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/report.json" "$OUT" "$MARKER" "$VALIDATOR"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT
[ -n "${TOK_S:-}" ] || fail 4 "TOK_S (a MEASURED prefill tok/s of this parent) is not set"
case "$TOK_S" in *[!0-9.]*|"") fail 4 "TOK_S=$TOK_S is not a number" ;; esac

# ------------------------------------------------------------------ setup: identities (exit 5 / 10)
say "setup (PARENT=$PARENT)"
cp /opt/ssdm/CONTEXT_MANIFEST.json "$WORK/out/" || fail 5 "no context manifest"
nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins"
import json, torch, transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
assert torch.cuda.is_available()
import decision_index  # noqa: F401
import third_party.rwkv7_hf as V
assert all(V.vendored_ok().values()), V.vendored_ok()
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "transformers": transformers.__version__, "vendored_rwkv7": V.REVISION}))
EOF
( cd /data && sha256sum -c "$CODE/jobs/rwkv7/records.sha256" ) > "$WORK/out/records-verify.txt" 2>&1 || fail 10 "records sha256"
python -X utf8 -m train.cohort verify > "$WORK/out/cohort-verify.json" 2>&1 || fail 10 "cohort does not rebuild to its committed sha256"
CAP_SETUP=$(cap setup) || fail 5 "job_params"
if [ "$PARENT" = rwkv7 ]; then
  SNAP=$(timeout ${CAP_SETUP}m python -c 'from huggingface_hub import snapshot_download as s; print(s("RWKV/RWKV7-G1k-13.3B-20260930", revision="ce3630b854cedbd673569f4600c6cf744bca9dba"))') || fail 10 "download"
  ( cd "$SNAP" && sha256sum -c "$CODE/jobs/rwkv7/snapshot.sha256" ) > "$WORK/out/snapshot-verify.txt" 2>&1 || fail 10 "snapshot sha256"
  python -c "import sys, third_party.rwkv7_hf as V; V.verify_snapshot_code(sys.argv[1])" "$SNAP" || fail 10 "vendored code != snapshot code"
else
  SNAP=$(timeout ${CAP_SETUP}m python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a"))') || fail 10 "download"
  ( cd "$SNAP" && sha256sum -c "$CODE/jobs/m2_pilot/q9af_snapshot.sha256" ) > "$WORK/out/snapshot-verify.txt" 2>&1 || fail 10 "q9af snapshot sha256"
fi
( cd /data && sha256sum -c "$CODE/jobs/clef_baseline/data.sha256" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
python -X utf8 -m jobs.common.dev_subset rows --rows /data/slices/dev-rows.jsonl.gz --out "$WORK/index-rows.jsonl.gz" \
  --subset jobs/common/dev_subset_m1.json > "$WORK/out/index-subset.json" || fail 10 "DEV subset rows"
python -c "import json,sys; from jobs.m2_parent_arm.validate import valid_cache; sys.exit(0 if valid_cache(json.load(open('/data/parent_arm/arm.json'))) else 1)" \
  || fail 10 "no valid parent arm cache in /data/parent_arm"
ARM_SHA=$(python -c "import json,sys; v=json.load(open(sys.argv[1]))['parent_arm_sha256'][sys.argv[2]]; print(v or '')" "$PARAMS" "$PARENT") || fail 5 "job_params"
[ -n "$ARM_SHA" ] || fail 10 "parent_arm_sha256.$PARENT is not pinned in job_params.json"
[ "$(sha256sum /data/parent_arm/arm.json | cut -d' ' -f1)" = "$ARM_SHA" ] || fail 10 "parent arm.json sha256 != pinned"
# everything the evals phase needs is checked HERE, before the paid cell phase (PR #29 review)
python -c "import train.m2_gates, train.heldout, jobs.m2_parent_arm.validate" || fail 5 "evals driver not importable"
CAP_E=$(cap evals) || fail 5 "job_params"
python -X utf8 -m jobs.m2_parent_arm.score_parent estimate --tok-s "$TOK_S" --mode child \
  --cap-min "$CAP_E" --fixed-min 0 > "$WORK/out/evals-preflight.json" || fail 15 "child evals projected past their cap"
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ guard readiness (exit 30 / 31 / 32): before ANY training
say "phase guard (rebuild the content-hash index from the pinned suite; refuse unless it is the pinned index)"
CAP_G=$(cap guard) || fail 5 "job_params"
timeout ${CAP_G}m python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir /data/suite-0.2 --kit-dir /opt/decision-index \
  --out-dir "$WORK/out/guard" > "$WORK/logs/guard.log" 2>&1; r=$?
gate "guard" $r; export_phase guard
case $r in 0) ;; 124) fail 7 "guard timeout" ;; 30|31|32) fail $r "guard not ready (see out/guard/readiness.json)" ;; *) fail 5 "guard rc=$r" ;; esac

# ------------------------------------------------------------------ held-out D1-B never in training rows (exit 33)
say "phase heldout (D1-B template hash check over the training records)"
CAP_H=$(cap heldout) || fail 5 "job_params"
timeout ${CAP_H}m python -X utf8 -m train.heldout --records /data/m2/records.jsonl.gz --out "$WORK/out/heldout.json" \
  > "$WORK/logs/heldout.log" 2>&1; r=$?
gate "heldout" $r
case $r in 0) ;; 124) fail 7 "heldout timeout" ;; 3) fail 33 "D1-B template in training records" ;; *) fail 5 "heldout rc=$r" ;; esac

# ------------------------------------------------------------------ cell: smoke -> ladder -> train (resumable)
say "phase cell (smoke -> ladder -> train; resumes from $CK)"
CAP_C=$(cap cell) || fail 5 "job_params"
mkdir -p "$CK/cell"
[ -f "$CK/cell/cell.json" ] && cp "$CK/cell/cell.json" "$WORK/out/cell.json"
timeout ${CAP_C}m python -m train.run_cell --config "$CELL" --model-dir "$SNAP" "${GDN_ARGS[@]}" \
  --records /data/m2/records.jsonl.gz --out "$WORK/out" --ckpt-dir "$CK/cell" --stage all \
  --guard-readiness "$WORK/out/guard/readiness.json" > "$WORK/logs/cell.log" 2>&1; r=$?
cp "$WORK/out/cell.json" "$CK/cell/cell.json" 2>/dev/null || true
gate "cell" $r; export_phase cell
case $r in 0) ;; 124) fail 7 "cell timeout" ;; 21|22|23|24|25|26|27|28|29) fail $r "cell stop" ;; *) fail 20 "cell unexpected rc=$r" ;; esac

# ------------------------------------------------------------------ evals: temperature, eval card, G12 + G13, promotion verdict
say "phase evals (calibration fit on the cohort calibration role; card on dev; G12 steer_dev; G13 steer_gate + retention)"
CAP_E=$(cap evals) || fail 5 "job_params"
timeout ${CAP_E}m python -X utf8 -m train.m2_gates --cell "$WORK/out/cell.json" --ckpt-dir "$CK/cell" --model-dir "$SNAP" \
  "${GDN_ARGS[@]}" --records /data/m2/records.jsonl.gz --guard-readiness "$WORK/out/guard/readiness.json" \
  --parent-arm /data/parent_arm --parent-arm-sha256 "$ARM_SHA" --index-rows "$WORK/index-rows.jsonl.gz" --out "$WORK/out/report.json" --stage all \
  > "$WORK/logs/evals.log" 2>&1; r=$?
gate "evals" $r; export_phase evals
case $r in 0) ;; 124) fail 7 "evals timeout" ;; 40|41|42) sample "$WORK/out/report.json" "report"; fail $r "promotion: reject" ;;
  10) fail 10 "evals identity (cohort / parent arm / checkpoint lineage)" ;; *) fail 9 "evals rc=$r" ;; esac
sample "$WORK/out/report.json" "report"
RC=0
exit 0
