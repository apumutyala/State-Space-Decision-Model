#!/usr/bin/env bash
# jobs/rwkv7: the pre-registered fallback parent RWKV7-G1k-13.3B, run inside an SF Compute batch job (chip h100).
# Spec: jobs/rwkv7/JOB_SPEC.md. Env: JOB=readout (stage A) | lora (stage B; submitted with depends_on stage A)
#   | readout_g1v2 (C1: stock readout through the vendor-compliant jrt-rwkv-g1-v2 template; Index DEV subset only)
#   | readout_g1v3 (R3: stock readout under v2, v2f, v3, v3w, v3r, v3p, v1, v3c in one process; one model load).
#   readout: verify snapshot -> memory/prefill probe -> Index DEV subset (stock readout) -> score/ECE -> steer smoke
#   readout_g1v2: verify snapshot -> Index DEV subset (family rwkv7-hf-g1-v2, live rate guard) -> score/ECE
#   readout_g1v3: verify snapshot -> Index DEV subset x {v2, v2f, v3, v3w, v3r, v3p, v1, v3c} in one process
#                 (jobs.rwkv7.readout_variants, live rate guard over all variants) -> score/ECE per variant (index/<variant>/score.json, ece.json)
#   lora:    verify snapshot -> train.run_cell smoke (cache fork == recompute) -> T ladder -> LoRA pilot (resumable)
# Exit codes: 0 measured + decided + exports verified   2 bad JOB   5 setup   6 export verification   7 phase timeout
#   9 decision missing/invalid   10 snapshot / vendored code / DEV rows / records identity mismatch, or download failed
#   11 memory probe failed   12 Index kit run failed   13 scoring failed   14 steer smoke failed
#   15 readout too slow by MEASUREMENT (jobs/common/rate_guard.py: probe prefill preflight or the live row rate)
#   33 an eval attack template in the training rows (train.run_cell exit 3: train.attack_guard, pin attack_clean)
#   21 fork mismatch   22 no T fits   23 non-finite loss   24 memory stop   25 too few pilot items
#   26 training too slow by MEASUREMENT (ladder step_s, then the first 10 steps: cannot reach 2 x pass-rule window)
#   1 finalize failed
# The marker /output/RWKV7_<JOB>_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
set -Euo pipefail
MAIN_PID=$$
CODE=/opt/ssdm/m1
JOB=${JOB:-readout}
case "$JOB" in readout|lora|readout_g1v2|readout_g1v3) ;; *) echo "JOB must be readout, readout_g1v2, readout_g1v3 or lora"; exit 2 ;; esac
FAMILY=rwkv7-hf
[ "$JOB" = readout_g1v2 ] && FAMILY=rwkv7-hf-g1-v2
PARAMS=$CODE/jobs/rwkv7/job_params_$JOB.json
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/rwkv7-$JOB
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200
WATCH_EXTRA=$WORK  # every phase writes under $WORK (logs, index rows, cell.json, train_log): watch it
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
# fla #640 guard: Hopper + Triton < 3.7.1 (or a non-Triton fla backward) -> exit 5 here, before any GPU phase.
python -I /opt/ssdm/m1/jobs/common/fla_guard.py --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
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

if [ "$JOB" = readout_g1v3 ]; then
  CAP_I=$(cap index) && CAP_SC=$(cap score) || fail 5 "job_params"
  SUBSET=jobs/common/dev_subset_m1.json
  VARIANTS="v2 v2f v3 v3w v3r v3p v1 v3c"   # priority order: a cut-short run keeps the control and the highest-value arms
  python -X utf8 -m jobs.common.dev_subset rows --rows /data/slices/dev-rows.jsonl.gz --out "$WORK/subset-rows.jsonl.gz" \
    --subset "$SUBSET" > "$WORK/out/index/subset-rows.json" || fail 10 "subset rows"
  NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$WORK/out/index/subset-rows.json")
  say "phase index (variants $VARIANTS, DEV subset, one model load; live rate guard over all variants)"
  python -m jobs.common.rate_guard run --results "$WORK/out/index/v*/results.jsonl" --total $(( $(echo $VARIANTS | wc -w) * NROWS )) --exclude-startup \
    --cap-min "$CAP_I" --report "$WORK/out/index/rate.json" -- \
    python -X utf8 -m jobs.rwkv7.readout_variants --model-dir "$SNAP" --rows "$WORK/subset-rows.jsonl.gz" \
    --out-root "$WORK/out/index" --variants "$(echo $VARIANTS | tr ' ' ',')" > "$WORK/logs/index.log" 2>&1; r=$?
  gate "index" $r; export_phase index
  RI=$r

  # score every variant whose results are complete, also after a stopped index phase (a cut-short run keeps its arms)
  say "phase score (per complete variant)"
  SC1=$(( CAP_SC / 16 ))   # the score phase cap covers 8 variants x (score + ece)
  for v in $VARIANTS; do
    n=$(python -c "import sys;print(sum(1 for _ in open(sys.argv[1],'rb')))" "$WORK/out/index/$v/results.jsonl" 2>/dev/null || echo 0)
    [ "$n" -ge "$NROWS" ] || { say "variant $v incomplete ($n / $NROWS rows): not scored"; continue; }
    timeout ${SC1}m python -X utf8 -m jobs.common.dev_subset score --results "$WORK/out/index/$v/results.jsonl" \
      --subset "$SUBSET" > "$WORK/out/index/$v/score.json" || fail 13 "score $v"
    timeout ${SC1}m python -X utf8 -m jobs.common.dev_subset ece --results "$WORK/out/index/$v/results.jsonl" \
      --subset "$SUBSET" > "$WORK/out/index/$v/ece.json" || fail 13 "ece $v"
  done
  gate "score" 0; export_phase score
  case $RI in 0) ;; 124) fail 7 "index timeout" ;; 15) fail 15 "index: measured rate too slow" ;; *) fail 12 "index rc=$RI" ;; esac
  python scripts/decide_rwkv7.py "$JOB" "$WORK/out" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1 \
    || fail 9 "decision"
elif [ "$JOB" = readout ] || [ "$JOB" = readout_g1v2 ]; then
  CAP_I=$(cap index) && CAP_SC=$(cap score) || fail 5 "job_params"
  SUBSET=jobs/common/dev_subset_m1.json
  python -X utf8 -m jobs.common.dev_subset rows --rows /data/slices/dev-rows.jsonl.gz --out "$WORK/subset-rows.jsonl.gz" \
    --subset "$SUBSET" > "$WORK/out/index/subset-rows.json" || fail 10 "subset rows"

  if [ "$JOB" = readout ]; then
  CAP_PR=$(cap probe) && CAP_ST=$(cap steer) || fail 5 "job_params"
  say "phase probe (memory / prefill)"
  timeout ${CAP_PR}m python -m jobs.rwkv7.probe_rwkv7 --model-dir "$SNAP" --out "$WORK/out/probe.json" \
    > "$WORK/logs/probe.log" 2>&1; r=$?
  gate "probe" $r; export_phase probe
  case $r in 0) ;; 124) fail 7 "probe timeout" ;; *) fail 11 "probe rc=$r" ;; esac

  say "phase index (stock readout, DEV subset)"
  NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$WORK/out/index/subset-rows.json")
  # measured stop condition: the probe's MEASURED prefill rate must let the readout fit its cap (else exit 15)
  TOKS=$(python -c "import json,sys;p=json.load(open(sys.argv[1]))['prefill'];print(max(v['tok_s'] for v in p.values() if 'tok_s' in v))" \
    "$WORK/out/probe.json") || fail 11 "no measured prefill rate"
  # per-call latency MEASURED by the probe (small_call_s); the fitted job-mhc3h value if the probe could not time it
  CALLS=$(python -c "import json,sys;from jobs.common.readout_cost import CALL_S_FITTED_RWKV7 as F;print(json.load(open(sys.argv[1])).get('small_call_s', F))" \
    "$WORK/out/probe.json") || fail 11 "no call latency"
  # exact subset volume of the prefix-fork readout (jobs/common/readout_cost.py), not a proxy-token guess
  python -m jobs.common.rate_guard preflight --tok-s "$TOKS" --call-s "$CALLS" --volume jobs/common/readout_volume_rwkv7.json \
    --path fork --cap-min "$CAP_I" --out "$WORK/out/index/preflight.json" > "$WORK/logs/preflight.log" 2>&1; r=$?
  gate "index:preflight" $r
  [ $r -eq 0 ] || fail $r "readout projected past its cap from the measured prefill rate"
  else
  NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$WORK/out/index/subset-rows.json")
  say "phase index (family $FAMILY, DEV subset; live rate guard only)"
  fi
  python -m jobs.common.rate_guard run --results "$WORK/out/index/stock/results.jsonl" --total "$NROWS" \
    --cap-min "$CAP_I" --report "$WORK/out/index/rate.json" -- \
    python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family="$FAMILY" --option model_dir="$SNAP" \
    --rows "$WORK/subset-rows.jsonl.gz" --out "$WORK/out/index/stock" > "$WORK/logs/index.log" 2>&1; r=$?
  gate "index" $r; export_phase index
  case $r in 0) ;; 124) fail 7 "index timeout" ;; 15) fail 15 "index: measured rate too slow" ;; *) fail 12 "index rc=$r" ;; esac

  say "phase score"
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset score --results "$WORK/out/index/stock/results.jsonl" \
    --subset "$SUBSET" > "$WORK/out/index/score.json" || fail 13 "score"
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset ece --results "$WORK/out/index/stock/results.jsonl" \
    --subset "$SUBSET" > "$WORK/out/index/ece.json" || fail 13 "ece"
  gate "score" 0

  if [ "$JOB" = readout ]; then
  say "phase steer smoke (safety_eval dev split, decide path, no D1-B)"
  timeout ${CAP_ST}m python -X utf8 -m safety_eval.run_smoke --backend decide --family rwkv7-hf --model "$SNAP" \
    --device cuda --n-route 40 --n-pay 24 --out "$WORK/out/steer" > "$WORK/logs/steer.log" 2>&1; r=$?
  gate "steer" $r; export_phase steer
  case $r in 0) ;; 124) fail 7 "steer timeout" ;; *) fail 14 "steer rc=$r" ;; esac

  fi
  python scripts/decide_rwkv7.py "$JOB" "$WORK/out" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1 \
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
  case $r in 0) ;; 3) fail 33 "cell: eval attack template in the training rows (train.attack_guard)" ;; 124) fail 7 "cell timeout" ;; 21|22|23|24|25|26|34) fail $r "cell stop" ;; *) fail 2 "cell rc=$r" ;; esac
  python scripts/decide_rwkv7.py lora "$WORK/out/cell.json" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1 \
    || fail 9 "decision"
fi
sample "$WORK/out/decision.json" "decision"
RC=0
exit 0
