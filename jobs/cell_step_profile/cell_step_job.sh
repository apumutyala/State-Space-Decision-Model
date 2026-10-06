#!/usr/bin/env bash
# jobs/cell_step_profile: the m2 first cell's train step under four pre-registered execution variants, run in the order
# baseline, ckpt_rows_lean, ckpt_rows, ckpt (rule v1-A2), one H100 SF batch job on the jobs/rwkv7 image (no extra
# packages). Spec + rule: jobs/cell_step_profile/JOB_SPEC.md, rule.py. NOT submitted by this branch.
#   setup (pins, CPU equivalence tests, snapshot) -> variant x4 (own process each, PYTORCH_CUDA_ALLOC_CONF set) -> decide
# Exit codes: 0 measured + decided + exports verified   5 setup (pins, equivalence tests)   6 export failed
#   9 decision missing/invalid   10 snapshot / vendored code mismatch or download failed   1 finalize failed.
#   A variant that crashes or times out is recorded in the gates and leaves its record partial or missing: it then
#   cannot qualify (rule.py), which is a verdict, not an exit code. Every verdict (ADOPT, CAP_T, NO_FIT) exits 0.
# The marker /output/CELL_STEP_PROFILE_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
# CELL_STEP_DRYRUN=1: CPU dry run of this exact script (tests/test_cell_step_job.py): code root = SSDM_CODE, no GPU
#   checks, no download (the harness uses the tiny vendored RWKV-7), same phases, same caps file, same finalize.
set -Euo pipefail
MAIN_PID=$$
DRY=${CELL_STEP_DRYRUN:-0}
CODE=${SSDM_CODE:-/opt/ssdm/m1}
PARAMS=${CELL_STEP_PARAMS:-$CODE/jobs/cell_step_profile/job_params.json}
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=${CELL_STEP_WORK:-/tmp/cell-step}
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=600
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs" "$CK"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # pre-registered for every variant (rule.ALLOC_CONF)
HF_DIR=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p "$HF_DIR"
T0=$(date +%s)
EXPORT_FAIL=0
RC=""
GATES="{}"
DL_PID=""
exec > >(tee -a "$WORK/logs/run.log") 2>&1
cd "$CODE"
# shellcheck source=../common/lib.sh
. "$CODE/jobs/common/lib.sh"
# shellcheck source=../clef_baseline/watchdog.sh
. "$CODE/jobs/clef_baseline/watchdog.sh"
MARKER=CELL_STEP_PROFILE_DONE
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" "${DL_PID:-}" 2> /dev/null || true
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/decision.json" "$OUT" "$MARKER" "jobs.cell_step_profile.rule:valid_decision"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup (exit 5 / 10)
say "setup (cell_step_profile, dryrun=$DRY)"
if [ "$DRY" = 1 ]; then
  echo '{"dryrun": true}' > "$WORK/out/CONTEXT_MANIFEST.json"
  echo "dryrun: no GPU" > "$WORK/out/nvidia-smi.txt"
else
  cp /opt/ssdm/CONTEXT_MANIFEST.json "$WORK/out/" || fail 5 "no context manifest"
  nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
  # weights download (warm hf_cache: seconds) overlaps the CPU equivalence tests
  ( timeout $(( $(cap setup) + $(cap download_wait) ))m python - > "$WORK/snap_path" 2> "$WORK/logs/download.log" <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("RWKV/RWKV7-G1k-13.3B-20260930", revision="ce3630b854cedbd673569f4600c6cf744bca9dba"))
EOF
  ) & DL_PID=$!
fi
DRY=$DRY python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins"
import json, os, torch, transformers
import third_party.rwkv7_hf as V
assert transformers.__version__ == "5.17.0", transformers.__version__
assert all(V.vendored_ok().values()), V.vendored_ok()
if os.environ["DRY"] != "1":
    assert torch.__version__.split("+")[0] == "2.13.0", torch.__version__
    assert torch.cuda.is_available()
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                  "vendored_rwkv7": V.REVISION, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}))
EOF
# the variants' equivalence to the original step (rule clause 4 rests on these when the baseline's step 0 OOMs): CPU
CUDA_VISIBLE_DEVICES="" timeout $(cap setup)m python -X utf8 -m pytest -q -p no:cacheprovider \
  tests/test_cell_perf.py tests/test_cell_step_rule.py > "$WORK/out/equivalence_tests.txt" 2>&1 \
  || fail 5 "CPU equivalence tests: $(tail -1 "$WORK/out/equivalence_tests.txt")"
say "equivalence tests: $(tail -1 "$WORK/out/equivalence_tests.txt")"
gate "setup" 0
export_phase setup

if [ "$DRY" = 1 ]; then
  SNAP=tiny
else
  say "wait for the snapshot, verify it"
  wait "$DL_PID"; r=$?; DL_PID=""
  [ $r -eq 0 ] || fail 10 "download rc=$r"
  SNAP=$(tail -1 "$WORK/snap_path")
  ( cd "$SNAP" && sha256sum -c "$CODE/jobs/rwkv7/snapshot.sha256" ) > "$WORK/out/snapshot-verify.txt" 2>&1 || fail 10 "snapshot sha256"
  python -c "import sys, third_party.rwkv7_hf as V; V.verify_snapshot_code(sys.argv[1])" "$SNAP" || fail 10 "vendored code != snapshot code"
fi
gate "snapshot" 0

# ------------------------------------------------------------------ variants (own process each; crash/timeout recorded)
for V in baseline ckpt_rows_lean ckpt_rows ckpt; do  # A2 order: the memory-only fallback last
  CAP_V=$(cap "$V") || fail 5 "job_params"
  say "variant $V (probe loss on step 22, 23 steps of train_mix_v2, K=10 ladder; cap ${CAP_V}m)"
  timeout ${CAP_V}m python -m jobs.cell_step_profile.cell_step variant --name "$V" --out "$WORK/out" --model-dir "$SNAP" \
    --deadline-s $(( CAP_V * 60 - 40 )) > "$WORK/logs/variant_$V.log" 2>&1; r=$?
  gate "variant_$V" $r; export_phase "variant_$V"
  [ $r -eq 0 ] || say "variant $V rc=$r: recorded; its measurement is partial or missing (cannot qualify)"
done

python -m jobs.cell_step_profile.cell_step decide --out "$WORK/out" > "$WORK/logs/decision.log" 2>&1 || fail 9 "decision"
say "decision: $(cat "$WORK/logs/decision.log")"
export_phase decide

# ------------------------------------------------------------------ profile (informational; never changes the exit code)
CAP_P=$(cap profile) || fail 5 "job_params"
say "profile (torch.profiler: decided variant + baseline on one mid-length step; cap ${CAP_P}m)"
timeout ${CAP_P}m python -m jobs.cell_step_profile.cell_step profile --out "$WORK/out" --model-dir "$SNAP" \
  --deadline-s $(( CAP_P * 60 - 30 )) > "$WORK/logs/profile.log" 2>&1; r=$?
gate "profile" $r; export_phase profile
[ $r -eq 0 ] || say "profile rc=$r: recorded (informational)"
RC=0
