#!/usr/bin/env bash
# jobs/rwkv7: the pre-registered fallback parent RWKV7-G1k-13.3B, run inside an SF Compute batch job (chip h100).
# Spec: jobs/rwkv7/JOB_SPEC.md. Env: JOB=readout (stage A) | lora (stage B; submitted with depends_on stage A).
#   readout: verify snapshot -> memory/prefill probe -> Index DEV subset (stock readout) -> score/ECE -> steer smoke
#   lora:    verify snapshot -> train.run_cell smoke (cache fork == recompute) -> T ladder -> LoRA pilot (resumable)
# Exit codes: 0 measured + decided + exports verified   2 bad JOB   5 setup   6 export verification   7 phase timeout
#   9 decision missing/invalid   10 snapshot / vendored code / DEV rows / records identity mismatch, or download failed
#   11 memory probe failed   12 Index kit run failed   13 scoring failed   14 steer smoke failed
#   21 fork mismatch   22 no T fits   23 non-finite loss   24 memory stop   25 too few pilot items   1 finalize failed
# The marker /output/RWKV7_<JOB>_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
set -Euo pipefail
MAIN_PID=$$
CODE=/opt/ssdm/m1
JOB=${JOB:-readout}
case "$JOB" in readout|lora) ;; *) echo "JOB must be readout or lora"; exit 2 ;; esac
PARAMS=$CODE/jobs/rwkv7/job_params_$JOB.json
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/rwkv7-$JOB
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200
mkdir -p "$OUT/phases" "$WORK/out/index" "$WORK/logs" "$CK"
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
MARKER=RWKV7_$(echo "$JOB" | tr a-z A-Z)_DONE
VALIDATOR=scripts.decide_rwkv7:valid_$JOB
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/decision.json" "$OUT" "$MARKER" "$VALIDATOR"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup: identities (exit 5 / 10)
say "setup ($JOB)"
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
( cd /data && sha256sum -c "$CODE/jobs/clef_baseline/data.sha256" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
CAP_SETUP=$(cap setup) || fail 5 "job_params"
SNAP=$(timeout ${CAP_SETUP}m python - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("RWKV/RWKV7-G1k-13.3B-20260930", revision="ce3630b854cedbd673569f4600c6cf744bca9dba"))
EOF
) || fail 10 "download"
( cd "$SNAP" && sha256sum -c "$CODE/jobs/rwkv7/snapshot.sha256" ) > "$WORK/out/snapshot-verify.txt" 2>&1 || fail 10 "snapshot sha256"
python -c "import sys, third_party.rwkv7_hf as V; V.verify_snapshot_code(sys.argv[1])" "$SNAP" || fail 10 "vendored code != snapshot code"
gate "setup" 0
export_phase setup

if [ "$JOB" = readout ]; then
  CAP_PR=$(cap probe) && CAP_I=$(cap index) && CAP_SC=$(cap score) && CAP_ST=$(cap steer) || fail 5 "job_params"
  SUBSET=jobs/common/dev_subset_m1.json
  python -X utf8 -m jobs.common.dev_subset rows --rows /data/slices/dev-rows.jsonl.gz --out "$WORK/subset-rows.jsonl.gz" \
    --subset "$SUBSET" > "$WORK/out/index/subset-rows.json" || fail 10 "subset rows"

  say "phase probe (memory / prefill)"
  timeout ${CAP_PR}m python -m jobs.rwkv7.probe_rwkv7 --model-dir "$SNAP" --out "$WORK/out/probe.json" \
    > "$WORK/logs/probe.log" 2>&1; r=$?
  gate "probe" $r; export_phase probe
  case $r in 0) ;; 124) fail 7 "probe timeout" ;; *) fail 11 "probe rc=$r" ;; esac

  say "phase index (stock readout, DEV subset)"
  timeout ${CAP_I}m python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family=rwkv7-hf --option model_dir="$SNAP" \
    --rows "$WORK/subset-rows.jsonl.gz" --out "$WORK/out/index/stock" > "$WORK/logs/index.log" 2>&1; r=$?
  gate "index" $r; export_phase index
  case $r in 0) ;; 124) fail 7 "index timeout" ;; *) fail 12 "index rc=$r" ;; esac

  say "phase score"
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset score --results "$WORK/out/index/stock/results.jsonl" \
    --subset "$SUBSET" > "$WORK/out/index/score.json" || fail 13 "score"
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset ece --results "$WORK/out/index/stock/results.jsonl" \
    --subset "$SUBSET" > "$WORK/out/index/ece.json" || fail 13 "ece"
  gate "score" 0

  say "phase steer smoke (safety_eval dev split, decide path, no D1-B)"
  timeout ${CAP_ST}m python -X utf8 -m safety_eval.run_smoke --backend decide --family rwkv7-hf --model "$SNAP" \
    --device cuda --n-route 40 --n-pay 24 --out "$WORK/out/steer" > "$WORK/logs/steer.log" 2>&1; r=$?
  gate "steer" $r; export_phase steer
  case $r in 0) ;; 124) fail 7 "steer timeout" ;; *) fail 14 "steer rc=$r" ;; esac

  python scripts/decide_rwkv7.py readout "$WORK/out" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1 \
    || fail 9 "decision"
else
  CAP_T=$(( $(cap smoke) + $(cap ladder) + $(cap train) )) || fail 5 "job_params"
  ( cd /data && sha256sum -c "$CODE/jobs/rwkv7/records.sha256" ) > "$WORK/out/records-verify.txt" 2>&1 || fail 10 "records sha256"
  say "phase cell (smoke -> ladder -> train; resumes from $CK)"
  mkdir -p "$CK/cell"
  [ -f "$CK/cell/cell.json" ] && cp "$CK/cell/cell.json" "$WORK/out/cell.json"   # finished stages are not re-run
  timeout ${CAP_T}m python -m train.run_cell --config jobs/rwkv7/pilot_cell.json --model-dir "$SNAP" \
    --records /data/m2/records.jsonl.gz --out "$WORK/out" --ckpt-dir "$CK/cell" --stage all \
    > "$WORK/logs/cell.log" 2>&1; r=$?
  cp "$WORK/out/cell.json" "$CK/cell/cell.json" 2>/dev/null || true
  gate "cell" $r; export_phase cell
  case $r in 0) ;; 124) fail 7 "cell timeout" ;; 21|22|23|24|25) fail $r "cell stop" ;; *) fail 2 "cell rc=$r" ;; esac
  python scripts/decide_rwkv7.py lora "$WORK/out/cell.json" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1 \
    || fail 9 "decision"
fi
sample "$WORK/out/decision.json" "decision"
RC=0
exit 0
