#!/usr/bin/env bash
# jobs/perf_step_gate RULE v2: the m2 cell's train step under four pre-registered execution variants, run in the order
# greedy (numerics reference), fast (= d1_fast, speed reference), fast_fla, fast_fla_nocompile (rule.py), one H100 SF
# batch job on the jobs/rwkv7 image + fla-core 0.5.2 installed at setup by wheel hash (jobs/fla_rwkv7/requirements-fla.txt,
# as jobs/fla_rwkv7/fla_job.sh). Spec + rule: jobs/perf_step_gate/JOB_SPEC.md, rule.py. NOT submitted here.
#   setup (pins, fla install + #640 guard, CPU equivalence tests, snapshot) -> greedy -> rate guard -> 3 variants -> decide
# Exit codes: 0 measured + decided + exports verified   5 setup   6 export failed   9 decision missing/invalid
#   10 snapshot / vendored code mismatch or download failed   15 rate guard (box > 1.5x job-tvuat's step time)
#   1 finalize failed. A variant that crashes or times out is recorded in the gates and cannot qualify (a verdict).
# The marker /output/PERF_STEP_GATE_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
# PERF_STEP_DRYRUN=1: CPU dry run of this exact script (tests/test_perf_step_gate.py): code root = SSDM_CODE, no GPU
#   checks, no download (tiny vendored RWKV-7), compile backend aot_eager, same phases, caps file and finalize.
set -Euo pipefail
MAIN_PID=$$
DRY=${PERF_STEP_DRYRUN:-0}
CODE=${SSDM_CODE:-/opt/ssdm/m1}
PARAMS=${PERF_STEP_PARAMS:-$CODE/jobs/perf_step_gate/job_params.json}
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=${PERF_STEP_WORK:-/tmp/perf-step}
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=900
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs" "$WORK/grads" "$WORK/inductor" "$CK"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # pre-registered for every variant (rule.ALLOC_CONF)
export TORCHINDUCTOR_CACHE_DIR=$WORK/inductor  # compile progress is file progress for the watchdog
[ "$DRY" = 1 ] && export SSDM_COMPILE_BACKEND=aot_eager PERF_STEP_DRYRUN=1
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
MARKER=PERF_STEP_GATE_DONE
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" "${DL_PID:-}" 2> /dev/null || true
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/decision.json" "$OUT" "$MARKER" "jobs.perf_step_gate.rule:valid_decision"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup (exit 5 / 10)
say "setup (perf_step_gate, dryrun=$DRY)"
if [ "$DRY" = 1 ]; then
  echo '{"dryrun": true}' > "$WORK/out/CONTEXT_MANIFEST.json"
  echo "dryrun: no GPU" > "$WORK/out/nvidia-smi.txt"
else
  cp /opt/ssdm/CONTEXT_MANIFEST.json "$WORK/out/" || fail 5 "no context manifest"
  nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
  # fla-core is not in the rwkv7 image: install the exact 0.5.2 wheel by hash, no deps (torch/triton/einops are pinned)
  pip install --no-cache-dir --no-deps --require-hashes -r "$CODE/jobs/fla_rwkv7/requirements-fla.txt"     > "$WORK/logs/pip-fla.log" 2>&1 || fail 5 "fla-core install"
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
    import triton, fla
    assert triton.__version__ == "3.7.1" and fla.__version__ == "0.5.2", (triton.__version__, fla.__version__)
    from fla.ops.rwkv7 import chunk_rwkv7  # noqa: F401
    print(json.dumps({"triton": triton.__version__, "fla": fla.__version__}), file=__import__("sys").stderr)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                  "vendored_rwkv7": V.REVISION, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}))
EOF
if [ "$DRY" != 1 ]; then  # fla #640 guard: Hopper + Triton < 3.7.1 (or a non-Triton fla backward) -> exit 5 here
  python -I "$CODE/jobs/common/fla_guard.py" --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1     || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
fi
CUDA_VISIBLE_DEVICES="" SSDM_COMPILE_BACKEND=aot_eager timeout $(cap setup)m python -X utf8 -m pytest -q -p no:cacheprovider \
  tests/test_cell_perf.py tests/test_perf_step_gate_rule.py > "$WORK/out/equivalence_tests.txt" 2>&1 \
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

run_variant() {
  local V=$1 CAP_V r
  CAP_V=$(cap "$V") || fail 5 "job_params"
  say "variant $V (grad step on mix step 6, 23 steps of train_mix_v2, ladder; cap ${CAP_V}m)"
  timeout ${CAP_V}m python -m jobs.perf_step_gate.gate variant --name "$V" --out "$WORK/out" --grads "$WORK/grads" \
    --model-dir "$SNAP" --deadline-s $(( CAP_V * 60 - 40 )) > "$WORK/logs/variant_$V.log" 2>&1; r=$?
  gate "variant_$V" $r; export_phase "variant_$V"
  [ $r -eq 0 ] || say "variant $V rc=$r: recorded; its measurement is partial or missing (cannot qualify)"
}

# ------------------------------------------------------------------ baseline, measured-rate guard, the other variants
run_variant greedy
python -m jobs.perf_step_gate.preflight --out "$WORK/out" > "$WORK/logs/preflight.log" 2>&1 \
  || fail 15 "rate guard: $(cat "$WORK/logs/preflight.log")"
say "rate guard: $(cat "$WORK/logs/preflight.log")"
for V in fast fast_fla fast_fla_nocompile; do
  run_variant "$V"
done

CAP_D=$(cap decide) || fail 5 "job_params"
timeout ${CAP_D}m python -m jobs.perf_step_gate.gate decide --out "$WORK/out" --grads "$WORK/grads" \
  > "$WORK/logs/decision.log" 2>&1 || fail 9 "decision"
say "decision: $(cat "$WORK/logs/decision.log")"
export_phase decide
RC=0
