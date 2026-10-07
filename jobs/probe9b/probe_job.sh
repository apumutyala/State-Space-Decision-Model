#!/usr/bin/env bash
# probe9b: the $40 one-layer Qwen3.5-9B conversion probe, run inside an SF Compute batch job (submit_job, chip h100).
# Spec: jobs/probe9b/JOB_SPEC.md. Env: PROBE_CONFIG=full|lean (default full; lean only on the qual verdict
# re-price-lean-plan).
#
# Exit codes (also in $GMN_RESULT_PATH): 0 measured + decided + every export verified (a REJECTED layer is a valid
#   decision, not a failure)   2 numerics / pipeline crash   3 throughput stop   4 memory stop   5 setup failed
#   6 an export failed verification   7 a phase timed out   8 9B weights unavailable   9 decision missing/invalid
#   10 DEV-row / subset identity mismatch   12 Index kit run failed   13 subset scoring failed   1 finalize failed
#   15 Index readout too slow by MEASUREMENT (jobs/common/rate_guard.py: prefill preflight or the live row rate)
# /output/PROBE_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
set -Euo pipefail
MAIN_PID=$$
CODE=/opt/ssdm/m1
PARAMS=$CODE/jobs/probe9b/job_params.json
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/probe9b
CK=$WORK
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200
mkdir -p "$OUT/phases" "$WORK/out/index" "$WORK/logs"
export SSDM_DATA_ROOT=$WORK/data SSDM_DATA=/data PYTHONUNBUFFERED=1 PYTHONUTF8=1
mkdir -p "$SSDM_DATA_ROOT"
if [ -n "${HF_HUB_CACHE:-}" ] && [ -d "${HF_HUB_CACHE}" ]; then ln -sfn "$HF_HUB_CACHE" "$SSDM_DATA_ROOT/hf"; else mkdir -p "$SSDM_DATA_ROOT/hf"; fi
HF_DIR=$(readlink -f "$SSDM_DATA_ROOT/hf")  # the real cache dir (downloads count as progress)
WATCH_EXTRA=$WORK  # = CK; explicit so the static watch test sees it
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
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/decision.json" "$OUT" PROBE_DONE scripts.decide_probe9b:valid_decision
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

case "${PROBE_CONFIG:-full}" in
  full) CFG=configs/probe9b_h100.json ;;
  lean) CFG=configs/probe9b_h100_lean.json ;;
  *) fail 5 "PROBE_CONFIG must be full or lean" ;;
esac
SUBSET=jobs/common/dev_subset_m1.json

# ------------------------------------------------------------------ setup (exit 5 / 8 / 10)
say "setup ($CFG)"
cp /opt/ssdm/CONTEXT_MANIFEST.json "$WORK/out/" || fail 5 "no context manifest"
cp "$CFG" "$WORK/out/config.used.json"
nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins"
import json, torch, transformers, importlib.metadata as m
pins = {"transformers": "5.17.0", "flash-linear-attention": "0.5.2", "fla-core": "0.5.2"}
got = {k: m.version(k) for k in pins}
assert got == pins, got
assert torch.cuda.is_available()
import decision_index  # noqa: F401  (kit pinned in the image)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), **got}))
EOF
# fla #640 guard: Hopper + Triton < 3.7.1 (or a non-Triton fla backward) -> exit 5 here, before any GPU phase.
python -I /opt/ssdm/m1/jobs/common/fla_guard.py --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
CAP_SETUP=$(cap setup) && CAP_W=$(cap weights) && CAP_P=$(cap P) && CAP_IS=$(cap I_stock) && CAP_IC=$(cap I_conv) \
  && CAP_S=$(cap S) || fail 5 "job_params"
DEADLINE=$(python -c "import json;print(json.load(open('$PARAMS'))['pipeline_deadline_minutes'])") || fail 5 "job_params"
# preflight rate guard (jobs/probe9b/preflight.py): every phase cap >= 1.3 x its projection from the job-4j3vu MEASURED
# rates, the pre-probe pipeline work inside its deadline with the same margin, every stop floor <= 0.7 x the measured
# rate. A config/params pair that breaks this (e.g. an edited cap) stops here, before any GPU minute is spent.
python -m jobs.probe9b.preflight --params "$PARAMS" --config "$CFG" --out "$WORK/out/preflight_P.json"   > "$WORK/logs/preflight_P.log" 2>&1; r=$?
gate "setup:preflight" $r
[ $r -eq 0 ] || fail 5 "preflight: a cap or stop floor violates the measured-rate margin ($WORK/out/preflight_P.json)"
( cd /data && sha256sum -c "$CODE/jobs/clef_baseline/data.sha256" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
python -X utf8 -m jobs.common.dev_subset rows --rows /data/slices/dev-rows.jsonl.gz --out "$WORK/data/subset-rows.jsonl.gz" \
  --subset "$SUBSET" > "$WORK/out/index/subset-rows.json" 2> "$WORK/logs/subset.err" || fail 10 "subset rows"
timeout $(( CAP_SETUP + CAP_W ))m python -I scripts/sf_fetch.py --stage probe9b > "$WORK/logs/fetch.log" 2>&1
[ -f "$SSDM_DATA_ROOT/DONE_Q9B_DIR" ] || { gate "setup:weights" 8; export_phase setup; fail 8 "9B weights unavailable"; }
for d in WIKITEXT_DIR MMLU_DIR; do [ -f "$SSDM_DATA_ROOT/DONE_$d" ] || fail 5 "fetch $d"; done
# shellcheck disable=SC1091
. "$SSDM_DATA_ROOT/paths.env"
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ P: stage 1, KL importance, one layer, gate, probes
say "phase P"
DEADLINE_MINUTES=$DEADLINE timeout ${CAP_P}m python -m convert.pipeline --config "$CFG" --out "$WORK/out/pipeline" \
  > "$WORK/logs/pipeline.log" 2>&1; r=$?
gate "P:pipeline" $r
export_phase P
case $r in 0) ;; 2|3|4) fail $r "pipeline stop" ;; 124) fail 7 "pipeline timeout" ;; *) fail 2 "pipeline crashed rc=$r" ;; esac
LAYERS=$(python - "$WORK/out/pipeline/results.json" "$SSDM_DATA_ROOT/ckpt/pipeline" "$OUT" <<'EOF'
# converted layer list + the converted GDN exported on its own (reused by the full-8 job), with its sha256
import hashlib, json, os, sys, torch
R = json.load(open(sys.argv[1]))
conv = R["progressive"]["converted"]
if conv:
    sd = torch.load(os.path.join(sys.argv[2], "prog_1.pt"), map_location="cpu")
    p = os.path.join(sys.argv[3], "gdn_converted.pt")
    torch.save({i: sd[i] for i in conv}, p)
    open(p + ".sha256", "w").write(hashlib.sha256(open(p, "rb").read()).hexdigest() + "  gdn_converted.pt\n")
print(",".join(map(str, conv)))
EOF
) || fail 2 "reading pipeline results"
say "converted layers: [${LAYERS}]"

# ------------------------------------------------------------------ I: Index DEV subset, stock and converted arms
if [ -n "$LAYERS" ]; then
  Q9B=$(python -c "from convert.model import resolve_model_dir as r; import os; print(r(os.environ['Q9B_DIR']))")
  NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$WORK/out/index/subset-rows.json")
  # measured stop condition: the pipeline's MEASURED stock prefill rate (the MIN over lengths, conservative) must let
  # the readout fit its cap (else 15)
  TOKS=$(python -c "import json,sys;t=json.load(open(sys.argv[1]))['throughput']['stock'];print(min(v['tok_s'] for v in t.values() if isinstance(v, dict) and 'tok_s' in v))"     "$WORK/out/pipeline/results.json") || fail 2 "no measured prefill rate"
  # exact subset volume of the prefix-fork readout at the MEASURED prefill rate and the 9B's own MEASURED per-call
  # cost (probes.throughput.call_latency in the pipeline's throughput phase, stock arm, max(small_call_s, pack_call_s),
  # conservative: pack_call_s also contains its own 4,096 positions); the RWKV7 fitted
  # value only if that measurement failed. The live row-rate guard stays authoritative.
  CALLS=$(python -c "import json,sys;from jobs.common.readout_cost import CALL_S_FITTED_RWKV7 as F;c=json.load(open(sys.argv[1]))['throughput']['stock'].get('call_latency',{});v=[c[k] for k in ('small_call_s','pack_call_s') if k in c];print(max(v) if v else F)"     "$WORK/out/pipeline/results.json") || fail 2 "reading the measured call latency"
  python -m jobs.common.rate_guard preflight --tok-s "$TOKS" --call-s "$CALLS" --volume jobs/common/readout_volume_qwen3.5.json \
    --path fork --cap-min "$CAP_IS" --out "$WORK/out/index/preflight.json" > "$WORK/logs/preflight.log" 2>&1; r=$?
  gate "I:preflight" $r
  [ $r -eq 0 ] || fail $r "readout projected past its cap from the measured prefill rate"
  say "phase I (stock arm)"
  python -m jobs.common.rate_guard run --results "$WORK/out/index/stock/results.jsonl" --total "$NROWS" \
    --cap-min "$CAP_IS" --report "$WORK/out/index/rate_stock.json" -- \
    python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family=q9af-probe --option model_dir="$Q9B" --option converted=none \
    --rows "$WORK/data/subset-rows.jsonl.gz" --out "$WORK/out/index/stock" > "$WORK/logs/index_stock.log" 2>&1; r=$?
  gate "I:stock" $r
  export_phase I_stock
  case $r in 0) ;; 124) fail 7 "index stock timeout" ;; 15) fail 15 "index stock: measured rate too slow" ;; *) fail 12 "index stock rc=$r" ;; esac
  say "phase I (converted arm: layers $LAYERS)"
  python -m jobs.common.rate_guard run --results "$WORK/out/index/conv/results.jsonl" --total "$NROWS" \
    --cap-min "$CAP_IC" --report "$WORK/out/index/rate_conv.json" -- \
    python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family=q9af-probe --option model_dir="$Q9B" --option converted="$LAYERS" \
    --option gdn_ckpt="$OUT/gdn_converted.pt" \
    --rows "$WORK/data/subset-rows.jsonl.gz" --out "$WORK/out/index/conv" > "$WORK/logs/index_conv.log" 2>&1; r=$?
  gate "I:conv" $r
  export_phase I_conv
  case $r in 0) ;; 124) fail 7 "index conv timeout" ;; 15) fail 15 "index conv: measured rate too slow" ;; *) fail 12 "index conv rc=$r" ;; esac

  say "phase S (subset scoring, paired)"
  S() { timeout ${CAP_S}m python -X utf8 -m jobs.common.dev_subset "$@" --subset "$SUBSET"; }
  S compare --candidate "$WORK/out/index/conv/results.jsonl" --baseline "$WORK/out/index/stock/results.jsonl" \
    > "$WORK/out/index/compare_conv_vs_stock.json" 2> "$WORK/logs/score.err" || fail 13 "compare"
  for arm in stock conv; do
    S score --results "$WORK/out/index/$arm/results.jsonl" > "$WORK/out/index/score_$arm.json" 2>> "$WORK/logs/score.err" || fail 13 "score $arm"
    S ece --results "$WORK/out/index/$arm/results.jsonl" > "$WORK/out/index/ece_$arm.json" 2>> "$WORK/logs/score.err" || fail 13 "ece $arm"
  done
  gate "S:score" 0
  export_phase S
  CMP="$WORK/out/index/compare_conv_vs_stock.json"
else
  say "gate rejected the layer: no converted model, Index arms skipped (decision = quality-fail)"
  CMP=-
fi

# ------------------------------------------------------------------ D: pre-registered success rule
say "phase D"
python scripts/decide_probe9b.py "$WORK/out/pipeline/results.json" "$CMP" "$CFG" "$WORK/out/decision.json" \
  > "$WORK/logs/decision.log" 2>&1; r=$?
gate "D:decision" $r
[ $r -eq 0 ] || fail 9 "decision generation failed rc=$r"
sample "$WORK/out/decision.json" "decision"
RC=0
exit 0   # finish() re-validates decision.json via finalize.py before PROBE_DONE
