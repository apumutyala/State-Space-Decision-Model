#!/usr/bin/env bash
# full8: all 8 attention layers of Qwen3.5-9B -> Gated DeltaNet (Q9-AF), an SF Compute batch job (submit_job, h100,
# resume "checkpoint"). Spec: jobs/full8/JOB_SPEC.md. Caps: jobs/full8/job_params.json. Binding rule: jobs/full8/rule.json
# (pre-registered at be9eed8) via scripts/decide_full8.py.
#
# Resume: the pipeline's results.json and every GDN checkpoint live in $GMN_CHECKPOINT_DIR (= CK). After each finished
# layer (and stage 1) jobs/full8/phase_hook.sh POSTs /v1/checkpoint; a restarted attempt re-runs this script, which
# skips nothing by itself: convert.pipeline finds results.json and resumes after the last finished layer.
#
# Exit codes (also in $GMN_RESULT_PATH): 0 measured + decided + every export verified (a rejected layer, a partial
#   conversion or a failed bound is a valid decision, not a failure)   2 numerics / pipeline crash   3 throughput stop
#   4 memory stop   5 setup failed   6 an export failed verification   7 a phase timed out   8 9B weights unavailable
#   9 decision missing/invalid (incl. no-measurement: fewer than 8 layers attempted)   10 DEV-row / subset identity
#   12 Index kit run failed   13 subset scoring failed   15 Index readout too slow by MEASUREMENT   1 finalize failed
# /output/FULL8_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
set -Euo pipefail
MAIN_PID=$$
CODE=/opt/ssdm/m1
MANIFEST=/opt/ssdm/CONTEXT_MANIFEST.json
DATA=/data
WORK=/tmp/full8
DRY=""
if [ -n "${FULL8_DRYRUN:-}" ]; then
  # CPU dry run (tests/test_full8.py): this very script, with only the GPU / kit / private-data / weight-fetch steps
  # replaced by tests/full8_dryrun (tiny Qwen3.5 stand-in). Refused inside the job image.
  if [ -d /opt/ssdm/m1 ] || [ -z "${FULL8_DRYRUN_ROOT:-}" ]; then echo "FULL8_DRYRUN refused here"; exit 5; fi
  CODE=$(cd "$(dirname "$0")/../.." && pwd)
  MANIFEST=$FULL8_DRYRUN_ROOT/CONTEXT_MANIFEST.json
  DATA=$FULL8_DRYRUN_ROOT/data
  WORK=$FULL8_DRYRUN_ROOT/work
  DRY="python -X utf8 -m tests.full8_dryrun.stubs"
fi
PARAMS=$CODE/jobs/full8/job_params.json
RULE=$CODE/jobs/full8/rule.json
CFG=configs/full8_h100.json
SUBSET=jobs/common/dev_subset_m1.json
DATA_SHA=$CODE/jobs/clef_baseline/data.sha256
PIPE=convert.pipeline
KIT=decision_index
if [ -n "$DRY" ]; then DATA_SHA=$FULL8_DRYRUN_ROOT/data.sha256; PIPE=tests.full8_dryrun.pipeline_tiny; KIT=tests.full8_dryrun.kit; fi
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200
[ -z "$DRY" ] || HB_EVERY_S=1  # dry run only: no 60-s watchdog sleep outliving the script (it holds the console log)
mkdir -p "$OUT/phases" "$WORK/out/index" "$WORK/out/pipeline" "$WORK/logs" "$CK/pipeline"
export SSDM_DATA_ROOT=$WORK/data SSDM_DATA=$DATA PYTHONUNBUFFERED=1 PYTHONUTF8=1
mkdir -p "$SSDM_DATA_ROOT"
if [ -n "${HF_HUB_CACHE:-}" ] && [ -d "${HF_HUB_CACHE}" ]; then ln -sfn "$HF_HUB_CACHE" "$SSDM_DATA_ROOT/hf"; else mkdir -p "$SSDM_DATA_ROOT/hf"; fi
HF_DIR=$(readlink -f "$SSDM_DATA_ROOT/hf")
WATCH_EXTRA=$WORK
T0=$(date +%s)
EXPORT_FAIL=0
EXPORTED=0
RC=""
GATES="{}"
exec > >(tee -a "$WORK/logs/run.log") 2>&1
cd "$CODE"
# shellcheck source=../common/lib.sh
. "$CODE/jobs/common/lib.sh"
# shellcheck source=../clef_baseline/watchdog.sh
. "$CODE/jobs/clef_baseline/watchdog.sh"
sync_pipeline() {  # the pipeline's receipts (in CK) -> WORK/out, so every phase export carries them
  cp -f "$CK/pipeline/results.json" "$CK/pipeline/config.json" "$CK/pipeline/train_log.jsonl" "$WORK/out/pipeline/" 2> /dev/null || true
}
salvage() {  # a failed attempt still exports the layers it finished (CK keeps them for a restart as well)
  [ "$EXPORTED" = 1 ] && return 0
  [ -f "$CK/pipeline/results.json" ] && [ -n "${Q9B_DIR:-}" ] || return 0
  python -m jobs.full8.export_gdn --mode salvage --results "$CK/pipeline/results.json" --ckpt-dir "$CK/gdn" \
    --model-dir "$Q9B_DIR" --rule "$RULE" --out-dir "$OUT" --meta-out "$WORK/out/export_salvage.json" \
    > "$WORK/logs/export_salvage.log" 2>&1 || say "salvage export failed (CK still holds the layers)"
}
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  [ "$rc" = 0 ] || salvage
  sync_pipeline
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/decision.json" "$OUT" FULL8_DONE scripts.decide_full8:valid_decision
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup (exit 5 / 8 / 10)
STEPS_DONE=$(python -c "import json,sys;print(len(json.load(open(sys.argv[1])).get('progressive',{}).get('steps',[])))" "$CK/pipeline/results.json" 2> /dev/null || echo 0)
say "setup ($CFG; checkpoint dir $CK: $STEPS_DONE layer(s) already finished${DRY:+; CPU DRY RUN})"
cp "$MANIFEST" "$WORK/out/" || fail 5 "no context manifest"
cp "$CFG" "$WORK/out/config.used.json"
if [ -n "$DRY" ]; then
  $DRY setup "$WORK/out" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "dry-run setup"
else
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
fi
CAP_SETUP=$(cap setup) && CAP_W=$(cap weights) && CAP_P=$(cap P) && CAP_X=$(cap X) && CAP_IS=$(cap I_stock) \
  && CAP_IC=$(cap I_conv) && CAP_S=$(cap S) || fail 5 "job_params"
DEADLINE=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['pipeline_deadline_minutes'])" "$PARAMS") || fail 5 "job_params"
# measured-rate preflight (jobs/full8/preflight.py): every cap >= 1.3 x its projection from the job-5cmat receipts,
# the deadline / late-layer rule, the stop floors and the predicted peak. A broken pair stops here, before any GPU minute.
python -m jobs.full8.preflight --params "$PARAMS" --config "$CFG" --out "$WORK/out/preflight_P.json" > "$WORK/logs/preflight_P.log" 2>&1; r=$?
gate "setup:preflight" $r
[ $r -eq 0 ] || fail 5 "preflight: a cap, floor or deadline violates the measured-rate margin ($WORK/out/preflight_P.json)"
( cd "$DATA" && sha256sum -c "$DATA_SHA" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
python -X utf8 -m jobs.common.dev_subset rows --rows "$DATA/slices/dev-rows.jsonl.gz" --out "$WORK/data/subset-rows.jsonl.gz" \
  --subset "$SUBSET" > "$WORK/out/index/subset-rows.json" 2> "$WORK/logs/subset.err" || fail 10 "subset rows"
if [ -n "$DRY" ]; then
  $DRY fetch "$SSDM_DATA_ROOT" > "$WORK/logs/fetch.log" 2>&1
else
  timeout $(( CAP_SETUP + CAP_W ))m python -I scripts/sf_fetch.py --stage probe9b > "$WORK/logs/fetch.log" 2>&1
fi
[ -f "$SSDM_DATA_ROOT/DONE_Q9B_DIR" ] || { gate "setup:weights" 8; export_phase setup; fail 8 "9B weights unavailable"; }
for d in WIKITEXT_DIR MMLU_DIR; do [ -f "$SSDM_DATA_ROOT/DONE_$d" ] || fail 5 "fetch $d"; done
# shellcheck disable=SC1091
. "$SSDM_DATA_ROOT/paths.env"
Q9B=$(python -c "from convert.model import resolve_model_dir as r; import os; print(r(os.environ['Q9B_DIR']))") || fail 8 "9B snapshot dir"
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ P: stage 1, KL importance, 8 layers (gate each), probes
export FULL8_CK=$CK FULL8_OUT=$OUT FULL8_META=$META PHASE_HOOK="bash $CODE/jobs/full8/phase_hook.sh"
say "phase P (resuming after $STEPS_DONE finished layer(s))"
DEADLINE_MINUTES=$DEADLINE timeout ${CAP_P}m python -m $PIPE --config "$CFG" --out "$CK/pipeline" \
  >> "$WORK/logs/pipeline.log" 2>&1; r=$?
gate "P:pipeline" $r
sync_pipeline
export_phase P
case $r in 0) ;; 2|3|4) fail $r "pipeline stop" ;; 124) fail 7 "pipeline timeout" ;; *) fail 2 "pipeline crashed rc=$r" ;; esac

# ------------------------------------------------------------------ X: export the converted weights (verified)
say "phase X (export)"
timeout ${CAP_X}m python -m jobs.full8.export_gdn --mode final --results "$CK/pipeline/results.json" --ckpt-dir "$CK/gdn" \
  --model-dir "$Q9B" --rule "$RULE" --out-dir "$OUT" --meta-out "$WORK/out/export.json" > "$WORK/logs/export.log" 2>&1; r=$?
gate "X:export" $r
[ $r -eq 0 ] || { EXPORT_FAIL=1; export_phase X; fail 6 "converted-weight export failed verification"; }
EXPORTED=1
read -r LAYERS GDN_FILE < <(python -c "import json,sys;m=json.load(open(sys.argv[1]));print(','.join(map(str,m['converted'])) or '-', m['file'] or '-')" "$WORK/out/export.json") || fail 6 "export.json"
LAYERS=${LAYERS%$'\r'} GDN_FILE=${GDN_FILE%$'\r'}  # a CRLF-printing python (the Windows dry run) must not leak a CR
[ "$LAYERS" = "-" ] && LAYERS=""
say "converted layers: [${LAYERS}], exported as ${GDN_FILE}"
export_phase X

# ------------------------------------------------------------------ I: Index DEV subset, stock and converted arms
if [ -n "$LAYERS" ]; then
  NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$WORK/out/index/subset-rows.json")
  TOKS=$(python -c "import json,sys;t=json.load(open(sys.argv[1]))['throughput']['stock'];print(min(v['tok_s'] for v in t.values() if isinstance(v, dict) and 'tok_s' in v))" "$CK/pipeline/results.json") || fail 2 "no measured prefill rate"
  CALLS=$(python -c "import json,sys;from jobs.common.readout_cost import CALL_S_FITTED_RWKV7 as F;c=json.load(open(sys.argv[1]))['throughput']['stock'].get('call_latency',{});v=[c[k] for k in ('small_call_s','pack_call_s') if k in c];print(max(v) if v else F)" "$CK/pipeline/results.json") || fail 2 "reading the measured call latency"
  python -m jobs.common.rate_guard preflight --tok-s "$TOKS" --call-s "$CALLS" --volume jobs/common/readout_volume_qwen3.5.json \
    --path fork --cap-min "$CAP_IS" --out "$WORK/out/index/preflight.json" > "$WORK/logs/preflight.log" 2>&1; r=$?
  gate "I:preflight" $r
  [ $r -eq 0 ] || fail $r "readout projected past its cap from the measured prefill rate"
  for arm in stock conv; do  # a restarted attempt on the same disk never appends to a half-written arm
    [ -e "$WORK/out/index/$arm" ] && mv "$WORK/out/index/$arm" "$WORK/out/index/$arm.stale.$(date +%s)"
  done
  say "phase I (stock arm)"
  python -m jobs.common.rate_guard run --results "$WORK/out/index/stock/results.jsonl" --total "$NROWS" \
    --cap-min "$CAP_IS" --report "$WORK/out/index/rate_stock.json" -- \
    python -X utf8 -m $KIT run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family=q9af-probe --option model_dir="$Q9B" --option converted=none \
    --rows "$WORK/data/subset-rows.jsonl.gz" --out "$WORK/out/index/stock" > "$WORK/logs/index_stock.log" 2>&1; r=$?
  gate "I:stock" $r
  export_phase I_stock
  case $r in 0) ;; 124) fail 7 "index stock timeout" ;; 15) fail 15 "index stock: measured rate too slow" ;; *) fail 12 "index stock rc=$r" ;; esac
  # the converted arm is preflighted on ITS OWN measured prefill rate and call latency (review non-blocking #2; an
  # 8-GDN arm may be slower than stock), falling back to the stock numbers only if the converted ones are missing
  read -r TOKS_C CALLS_C < <(python - "$CK/pipeline/results.json" "$TOKS" "$CALLS" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))["throughput"]
arm = next((v for k, v in t.items() if k.startswith("converted_")), {}) or {}
rates = [v["tok_s"] for v in arm.values() if isinstance(v, dict) and "tok_s" in v]
c = arm.get("call_latency", {})
calls = [c[k] for k in ("small_call_s", "pack_call_s") if k in c]
print(min(rates) if rates else sys.argv[2], max(calls) if calls else sys.argv[3])
EOF
) || fail 2 "reading the converted arm's measured rate"
  TOKS_C=${TOKS_C%$'\r'} CALLS_C=${CALLS_C%$'\r'}
  python -m jobs.common.rate_guard preflight --tok-s "$TOKS_C" --call-s "$CALLS_C" --volume jobs/common/readout_volume_qwen3.5.json \
    --path fork --cap-min "$CAP_IC" --out "$WORK/out/index/preflight_conv.json" > "$WORK/logs/preflight_conv.log" 2>&1; r=$?
  gate "I:preflight_conv" $r
  [ $r -eq 0 ] || fail $r "converted readout projected past its cap from the converted arm's measured rate"
  say "phase I (converted arm: layers $LAYERS from $GDN_FILE)"
  python -m jobs.common.rate_guard run --results "$WORK/out/index/conv/results.jsonl" --total "$NROWS" \
    --cap-min "$CAP_IC" --report "$WORK/out/index/rate_conv.json" -- \
    python -X utf8 -m $KIT run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family=q9af-probe --option model_dir="$Q9B" --option converted="$LAYERS" \
    --option gdn_ckpt="$OUT/$GDN_FILE" \
    --rows "$WORK/data/subset-rows.jsonl.gz" --out "$WORK/out/index/conv" > "$WORK/logs/index_conv.log" 2>&1; r=$?
  gate "I:conv" $r
  export_phase I_conv
  case $r in 0) ;; 124) fail 7 "index conv timeout" ;; 15) fail 15 "index conv: measured rate too slow" ;; *) fail 12 "index conv rc=$r" ;; esac

  say "phase S (subset scoring, paired)"
  if [ -n "$DRY" ]; then S() { $DRY subset "$@" --subset "$SUBSET"; }
  else S() { timeout ${CAP_S}m python -X utf8 -m jobs.common.dev_subset "$@" --subset "$SUBSET"; }; fi
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
  say "the gate rejected every layer: nothing converted, Index arms skipped (decision = no-conversion)"
  CMP=-
fi

# ------------------------------------------------------------------ D: the pre-registered rule (jobs/full8/rule.json)
say "phase D"
python scripts/decide_full8.py "$CK/pipeline/results.json" "$CMP" "$RULE" "$WORK/out/decision.json" \
  > "$WORK/logs/decision.log" 2>&1; r=$?
gate "D:decision" $r
[ $r -eq 0 ] || fail 9 "no valid decision (rc=$r): $(head -c 300 "$WORK/out/decision.json" 2> /dev/null)"
cp "$WORK/out/decision.json" "$OUT/decision.json"
sample "$WORK/out/decision.json" "decision"
RC=0
exit 0   # finish() re-validates decision.json via finalize.py before FULL8_DONE
