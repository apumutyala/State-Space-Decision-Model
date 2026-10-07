#!/usr/bin/env bash
# jobs/fla_rwkv7: fla chunk_rwkv7 vs the vendored eager WKV on RWKV7-G1k-13.3B -- equivalence + throughput gate, one
# H100 SF batch job built from the jobs/rwkv7 image (build_context.sh copies its Dockerfile) plus fla-core 0.5.2
# installed at setup with a pinned wheel hash. Spec: jobs/fla_rwkv7/JOB_SPEC.md. NOT submitted by this branch.
#   setup -> fla guard -> (snapshot download in the background) -> kernel -> fork -> verify snapshot -> model ->
#   throughput -> decide (pre-registered rule, jobs/fla_rwkv7/fla_equivalence.py:decide)
# Exit codes: 0 measured + decided + exports verified   5 setup (pins, fla install, fla #640 guard)   6 export failed
#   7 phase timeout   9 decision missing/invalid   10 snapshot / vendored code mismatch or download failed
#   30 warmup (fla compile) crashed   31 kernel phase crashed   32 fork phase crashed   33 model phase crashed
#   34 throughput phase crashed. The optional fp32 phase (last, only with time left) never changes the exit code.
#   1 finalize failed. Every verdict (ADOPT ... REJECT_NUMERICS, INCONCLUSIVE) is a successful run (exit 0).
# The marker /output/FLA_RWKV7_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
# Mode (first argument): gate (default; the v1 rule above) | truth_v2 (RULE v2, JOB_SPEC.md "RULE v2"): the same setup /
#   warmup / kernel / fork / snapshot / model / throughput phases, unchanged; then truth_v2 (captured real inputs at
#   L0/15/30/45/60 vs fp64 truth; crash 35), train_v2 for chunked, fla, fla_safe_gate (the m2 cell's step, 3 + 20 steps;
#   a crash or timeout there is recorded in the gates and leaves that measurement missing -> INCONCLUSIVE, never an exit
#   code), decide_v2 -> decision_v2.json, marker FLA_RWKV7_TRUTH_V2_DONE. No fp32 tail in truth_v2 mode.
set -Euo pipefail
MAIN_PID=$$
CODE=/opt/ssdm/m1
MODE=${1:-gate}
case $MODE in
  gate) PARAMS=$CODE/jobs/fla_rwkv7/job_params.json ;;
  truth_v2) PARAMS=$CODE/jobs/fla_rwkv7/job_params_truth_v2.json ;;
  *) echo "unknown mode '$MODE' (gate | truth_v2)" >&2; exit 5 ;;
esac
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/fla-rwkv7
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=600
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs" "$CK"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 FLA_TILELANG=0
# one Triton cache for every phase process: kernels compiled + autotuned (fla cache_results=True) once in warmup are
# reused by kernel/fork/model/throughput (job-4idg4 lost its kernel phase to first-call compiles)
export TRITON_CACHE_DIR=$WORK/triton-cache
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
if [ "$MODE" = truth_v2 ]; then
  MARKER=FLA_RWKV7_TRUTH_V2_DONE VALIDATOR=jobs.fla_rwkv7.fla_equivalence:valid_decision_v2 DECISION=decision_v2.json
else
  MARKER=FLA_RWKV7_DONE VALIDATOR=jobs.fla_rwkv7.fla_equivalence:valid_decision DECISION=decision.json
fi
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" "${DL_PID:-}" 2> /dev/null || true
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/$DECISION" "$OUT" "$MARKER" "$VALIDATOR"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup (exit 5 / 10)
say "setup (fla_rwkv7, mode $MODE)"
cp /opt/ssdm/CONTEXT_MANIFEST.json "$WORK/out/" || fail 5 "no context manifest"
nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
# fla-core is not in the rwkv7 image: install the exact 0.5.2 wheel by hash, no deps (torch/triton/einops are pinned)
pip install --no-cache-dir --no-deps --require-hashes -r "$CODE/jobs/fla_rwkv7/requirements-fla.txt" \
  > "$WORK/logs/pip-fla.log" 2>&1 || fail 5 "fla-core install"
python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins"
import json, torch, triton, transformers, fla
assert transformers.__version__ == "5.17.0", transformers.__version__
assert torch.__version__.split("+")[0] == "2.13.0" and triton.__version__ == "3.7.1", (torch.__version__, triton.__version__)
assert fla.__version__ == "0.5.2", fla.__version__
assert torch.cuda.is_available()
import third_party.rwkv7_hf as V
assert all(V.vendored_ok().values()), V.vendored_ok()
from fla.ops.rwkv7 import chunk_rwkv7  # noqa: F401
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
                  "fla": fla.__version__, "gpu": torch.cuda.get_device_name(0),
                  "transformers": transformers.__version__, "vendored_rwkv7": V.REVISION}))
EOF
# fla #640 guard: Hopper + Triton < 3.7.1 (or a non-Triton fla backward) -> exit 5 here, before any GPU phase.
python -I /opt/ssdm/m1/jobs/common/fla_guard.py --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
gate "setup" 0
CAP_W=$(cap warmup) && CAP_K=$(cap kernel) && CAP_F=$(cap fork) && CAP_DL=$(cap download_wait) && CAP_M=$(cap model) && CAP_T=$(cap throughput) \
  || fail 5 "job_params"

# weights download overlaps the two weight-free phases (network vs GPU)
( timeout $(( CAP_W + CAP_K + CAP_F + CAP_DL ))m python - > "$WORK/snap_path" 2> "$WORK/logs/download.log" <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("RWKV/RWKV7-G1k-13.3B-20260930", revision="ce3630b854cedbd673569f4600c6cf744bca9dba"))
EOF
) & DL_PID=$!
export_phase setup

say "phase warmup (fla JIT compile + autotune, bf16, both variants, state/packed x fwd/bwd)"
timeout ${CAP_W}m python -m jobs.fla_rwkv7.fla_equivalence warmup --out "$WORK/out" > "$WORK/logs/warmup.log" 2>&1; r=$?
gate "warmup" $r; export_phase warmup
case $r in 0) ;; 124) fail 7 "warmup timeout" ;; *) fail 30 "warmup rc=$r" ;; esac

say "phase kernel (synthetic, real head shapes; fwd T=2048/1000, grads T=512/24; bf16)"
timeout ${CAP_K}m python -m jobs.fla_rwkv7.fla_equivalence kernel --out "$WORK/out" > "$WORK/logs/kernel.log" 2>&1; r=$?
gate "kernel" $r; export_phase kernel
case $r in 0) ;; 124) fail 7 "kernel timeout" ;; *) fail 31 "kernel rc=$r" ;; esac

say "phase fork (split / K-way fork with grads / packed / right padding)"
timeout ${CAP_F}m python -m jobs.fla_rwkv7.fla_equivalence fork --out "$WORK/out" > "$WORK/logs/fork.log" 2>&1; r=$?
gate "fork" $r; export_phase fork
case $r in 0) ;; 124) fail 7 "fork timeout" ;; *) fail 32 "fork rc=$r" ;; esac

say "wait for the snapshot, verify it"
wait "$DL_PID"; r=$?; DL_PID=""
[ $r -eq 0 ] || fail 10 "download rc=$r"
SNAP=$(tail -1 "$WORK/snap_path")
( cd "$SNAP" && sha256sum -c "$CODE/jobs/rwkv7/snapshot.sha256" ) > "$WORK/out/snapshot-verify.txt" 2>&1 || fail 10 "snapshot sha256"
python -c "import sys, third_party.rwkv7_hf as V; V.verify_snapshot_code(sys.argv[1])" "$SNAP" || fail 10 "vendored code != snapshot code"
gate "snapshot" 0

say "phase model (real weights: logits vs eager, continuation/fork, captured WKV inputs L0/30/60)"
timeout ${CAP_M}m python -m jobs.fla_rwkv7.fla_equivalence model --out "$WORK/out" --model-dir "$SNAP" \
  > "$WORK/logs/model.log" 2>&1; r=$?
gate "model" $r; export_phase model
case $r in 0) ;; 124) fail 7 "model timeout" ;; *) fail 33 "model rc=$r" ;; esac

say "phase throughput (prefill 2k/8k/16k, train fwd+bwd 2k, WKV-only attribution)"
timeout ${CAP_T}m python -m jobs.fla_rwkv7.fla_equivalence throughput --out "$WORK/out" --model-dir "$SNAP" \
  > "$WORK/logs/throughput.log" 2>&1; r=$?
gate "throughput" $r; export_phase throughput
case $r in 0) ;; 124) fail 7 "throughput timeout" ;; *) fail 34 "throughput rc=$r" ;; esac

if [ "$MODE" = gate ]; then
  # optional fp32 (only sets allow_fp32): runs only if it can finish before max_duration minus the reserve; a timeout or
  # crash here is recorded and does not change the exit code (missing fp32 -> allow_fp32 false)
  MAXD=$(python -c "import json,sys;p=json.load(open(sys.argv[1]));print(p['max_duration_minutes']-p['reserve_minutes'])" "$PARAMS")
  LEFT_S=$(( MAXD * 60 - ($(date +%s) - T0) - 30 ))
  if [ "$LEFT_S" -ge 90 ]; then
    say "phase fp32 (optional; ${LEFT_S}s left)"
    timeout ${LEFT_S}s python -m jobs.fla_rwkv7.fla_equivalence fp32 --out "$WORK/out" > "$WORK/logs/fp32.log" 2>&1; r=$?
    gate "fp32" $r; export_phase fp32
  else
    say "phase fp32 skipped (${LEFT_S}s left)"
  fi

  python -m jobs.fla_rwkv7.fla_equivalence decide --out "$WORK/out" > "$WORK/logs/decision.log" 2>&1 || fail 9 "decision"
  say "decision: $(cat "$WORK/logs/decision.log")"
  RC=0
fi

if [ "$MODE" = truth_v2 ]; then
  CAP_TV=$(cap truth_v2) || fail 5 "job_params"
  say "phase truth_v2 (captured real WKV inputs L0/15/30/45/60 vs fp64 token-serial truth)"
  timeout ${CAP_TV}m python -m jobs.fla_rwkv7.fla_equivalence truth_v2 --out "$WORK/out" --model-dir "$SNAP" \
    > "$WORK/logs/truth_v2.log" 2>&1; r=$?
  gate "truth_v2" $r; export_phase truth_v2
  case $r in 0) ;; 124) fail 7 "truth_v2 timeout" ;; *) fail 35 "truth_v2 rc=$r" ;; esac
  # the m2 cell's training step, one process per impl (clean peak memory); no step starts within 45 s of the cap
  for IMPL in chunked fla fla_safe_gate; do
    CAP_TR=$(cap "train_$IMPL") || fail 5 "job_params"
    say "phase train_v2 $IMPL (m2 cell step: LoRA r64, fork head, bf16, 1 x 8; 3 warmup + 20 timed steps)"
    timeout ${CAP_TR}m python -m jobs.fla_rwkv7.fla_equivalence train_v2 --out "$WORK/out" --model-dir "$SNAP" \
      --impl $IMPL --deadline-s $(( CAP_TR * 60 - 45 )) > "$WORK/logs/train_v2_$IMPL.log" 2>&1; r=$?
    gate "train_v2_$IMPL" $r; export_phase "train_v2_$IMPL"
    [ $r -eq 0 ] || say "train_v2 $IMPL rc=$r: recorded; its measurement is missing or partial (decide_v2)"
  done
  python -m jobs.fla_rwkv7.fla_equivalence decide_v2 --out "$WORK/out" > "$WORK/logs/decision_v2.log" 2>&1 \
    || fail 9 "decision_v2"
  say "decision_v2: $(cat "$WORK/logs/decision_v2.log")"
  RC=0
fi
