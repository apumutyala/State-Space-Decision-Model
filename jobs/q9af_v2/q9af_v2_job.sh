#!/usr/bin/env bash
# q9af_v2: ALL 8 attention layers of Qwen3.5-9B -> Gated DeltaNet jointly (Q9-AF v2), an SF Compute batch job
# (submit_job, h100, one attempt). Spec: jobs/q9af_v2/JOB_SPEC.md. Caps: jobs/q9af_v2/job_params.json. Rule:
# jobs/q9af_v2/rule.py (held-out generic KL + stock Index proxy600, paired vs stock in this job).
#
# Phases: setup (pins, fla guard, measured-rate preflight, DEV rows + proxy600 identity, weights/data fetch)
#   D  convert.mixdata build (generic mixture, 13-gram DEV filter)   P  convert.joint (stock probes, init_cmp, stage 1,
#   stage 2 joint KL, final held-out / probes / MMLU / throughput)   X  export gdn_converted.pt (verified)
#   I  Index proxy600: stock arm, converted arm (same readout code path)   S  paired compare + scores   R  the rule.
# Exit codes (also in $GMN_RESULT_PATH): 0 measured + decided + exports verified (any verdict, FOLD included)
#   2 pipeline crash   3 throughput stop   4 memory stop   5 setup failed   6 export failed verification   7 phase
#   timeout   8 9B weights unavailable   9 no valid decision   10 DEV-row / proxy identity   11 mixture build failed
#   12 Index kit failed   13 scoring failed   15 Index readout too slow by MEASUREMENT   1 finalize failed
# /output/Q9AF2_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
# Mode (argv 1): none = job-w3c2e's job (job_params.json, configs/q9af_v2_h100.json, dev13-v1 decontam). "clean" = the
# clean rerun (job_params_clean.json + its config: full-eval-v1 decontam against the shipped eval-hash index, the same
# rule). Both modes export mix.json and the per-window manifest (windows.json.gz) after phase D.
# Q9AF2_DRYRUN=1 (CPU, tests/test_q9af_v2.py): this very script with only the fetch / GPU / kit steps replaced by
# tests/q9af_v2_stub.py; setup checks, the preflight, the rule and finalize run for real. Refused inside the image.
set -Euo pipefail
MAIN_PID=$$
CODE=/opt/ssdm/m1
MANIFEST=/opt/ssdm/CONTEXT_MANIFEST.json
DATA=/data
WORK=/tmp/q9af2
DRY=""
if [ -n "${Q9AF2_DRYRUN:-}" ]; then
  if [ -d /opt/ssdm/m1 ] || [ -z "${Q9AF2_DRYRUN_ROOT:-}" ]; then echo "Q9AF2_DRYRUN refused here"; exit 5; fi
  CODE=$(cd "$(dirname "$0")/../.." && pwd)
  MANIFEST=$Q9AF2_DRYRUN_ROOT/CONTEXT_MANIFEST.json
  DATA=$Q9AF2_DRYRUN_ROOT/data
  WORK=$Q9AF2_DRYRUN_ROOT/work
  DRY="python -X utf8 -m tests.q9af_v2_stub"
fi
MODE=${1:-}
PROXY=jobs/common/dev_proxy600.json
EVAL_INDEX=""
case "$MODE" in
  "") PARAMS=$CODE/jobs/q9af_v2/job_params.json; CFG=configs/q9af_v2_h100.json; DATA_SHA=$CODE/jobs/clef_baseline/data.sha256 ;;
  clean) PARAMS=$CODE/jobs/q9af_v2/job_params_clean.json; CFG=configs/q9af_v2_clean_h100.json
         DATA_SHA=$CODE/jobs/q9af_v2/data_clean.sha256; EVAL_INDEX=$DATA/doc-overlap/218affe13e367d07 ;;
  *) echo "unknown mode $MODE"; exit 5 ;;
esac
[ -z "$DRY" ] || DATA_SHA=$Q9AF2_DRYRUN_ROOT/data.sha256
OUT=${GMN_OUTPUT_DIR:-/output}
[ -z "$DRY" ] || OUT=$Q9AF2_DRYRUN_ROOT/output
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
CK=$WORK/ck
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200
[ -z "$DRY" ] || HB_EVERY_S=1
mkdir -p "$OUT/phases" "$WORK/out/index" "$WORK/out/pipeline" "$WORK/logs" "$CK/pipeline" "$CK/gdn"
export SSDM_DATA_ROOT=$WORK/data SSDM_DATA=$DATA PYTHONUNBUFFERED=1 PYTHONUTF8=1 Q9AF2_CK=$CK
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
sync_pipeline() {
  cp -f "$CK/pipeline/results.json" "$CK/pipeline/config.json" "$CK/pipeline/train_log.jsonl" "$WORK/out/pipeline/" 2> /dev/null || true
  cp -f "$WORK/mix/mix.json" "$WORK/out/mix.json" 2> /dev/null || true
  cp -f "$WORK/mix/windows.json.gz" "$WORK/out/windows.json.gz" 2> /dev/null || true
}
salvage() {  # a failed attempt still exports its last stage-2 checkpoint (never a Q9-AF input)
  [ "$EXPORTED" = 1 ] && return 0
  local src=""
  for f in "$CK/gdn/final.pt" "$CK/gdn/stage2_last.pt" "$CK/gdn/stage1.pt"; do [ -f "$f" ] && { src=$f; break; }; done
  [ -n "$src" ] && [ -n "${Q9B:-}" ] || return 0
  cp -f "$CK/gdn/"*.json "$WORK/out/" 2> /dev/null || true
  python -m jobs.q9af_v2.export --mode salvage --src "$src" --model-dir "$Q9B" --out-dir "$OUT" \
    --meta-out "$WORK/out/export_salvage.json" > "$WORK/logs/export_salvage.log" 2>&1 || say "salvage export failed"
}
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  [ "$rc" = 0 ] || salvage
  sync_pipeline
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/decision.json" "$OUT" Q9AF2_DONE jobs.q9af_v2.rule:valid_decision
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup (exit 5 / 8 / 10)
say "setup (${MODE:-w3c2e} mode: $CFG${DRY:+; CPU DRY RUN})"
cp "$MANIFEST" "$WORK/out/" || fail 5 "no context manifest"
cp "$CFG" "$WORK/out/config.used.json"
if [ -n "$DRY" ]; then
  $DRY setup "$WORK/out" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "dry-run setup"
else
  nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
  python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins"
import json, torch, transformers, importlib.metadata as m
pins = {"transformers": "5.17.0", "flash-linear-attention": "0.5.2", "fla-core": "0.5.2", "pyarrow": "25.0.1"}
got = {k: m.version(k) for k in pins}
assert got == pins, got
assert torch.cuda.is_available()
import decision_index  # noqa: F401  (kit pinned in the image)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), **got}))
EOF
  python -I /opt/ssdm/m1/jobs/common/fla_guard.py --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
fi
CAP_W=$(cap weights) && CAP_D=$(cap D) && CAP_P=$(cap P) && CAP_X=$(cap X) && CAP_IS=$(cap I_stock) \
  && CAP_IC=$(cap I_conv) && CAP_S=$(cap S) || fail 5 "job_params"
# measured-rate preflight: every cap vs its projection from the 5cmat / 2xe5w receipts, floors, peak, token budget
python -m jobs.q9af_v2.preflight --params "$PARAMS" --config "$CFG" --out "$WORK/out/preflight.json" \
  --volume-out "$WORK/out/index/volume_proxy600.json" > "$WORK/logs/preflight.log" 2>&1; r=$?
gate "setup:preflight" $r
[ $r -eq 0 ] || fail 5 "preflight: a cap, floor, peak or token budget violates the measured-rate margin"
( cd "$DATA" && sha256sum -c "$DATA_SHA" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
python -X utf8 -m jobs.common.dev_proxy check --proxy proxy600 > "$WORK/out/index/proxy-identity.json" 2> "$WORK/logs/proxy.err" || fail 10 "proxy600 identity"
python -X utf8 -m jobs.common.dev_subset rows --rows "$DATA/slices/dev-rows.jsonl.gz" --out "$WORK/data/proxy-rows.jsonl.gz" \
  --subset "$PROXY" > "$WORK/out/index/proxy-rows.json" 2> "$WORK/logs/subset.err" || fail 10 "proxy600 rows"
if [ -n "$DRY" ]; then
  $DRY fetch "$SSDM_DATA_ROOT" > "$WORK/logs/fetch.log" 2>&1
else
  timeout "${CAP_W}m" python -I scripts/sf_fetch.py --stage q9af_v2 > "$WORK/logs/fetch.log" 2>&1
fi
[ -f "$SSDM_DATA_ROOT/DONE_Q9B_DIR" ] || { gate "setup:weights" 8; export_phase setup; fail 8 "9B weights unavailable"; }
for d in WIKITEXT_DIR MMLU_DIR FINEWEB_EDU_DIR ULTRACHAT_DIR FINEMATH_DIR; do [ -f "$SSDM_DATA_ROOT/DONE_$d" ] || fail 5 "fetch $d"; done
# shellcheck disable=SC1091
. "$SSDM_DATA_ROOT/paths.env"
export FINEWEB_EDU_DIR ULTRACHAT_DIR FINEMATH_DIR
Q9B=$(python -c "from convert.model import resolve_model_dir as r; import os; print(r(os.environ['Q9B_DIR']))") || fail 8 "9B snapshot dir"
Q9B=${Q9B%$'\r'}
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ D: the generic distillation mixture
say "phase D (mixture build + ${EVAL_INDEX:+full-eval-v1 / }DEV 13-gram filter + window manifest)"
IDX_ARGS=()
[ -z "$EVAL_INDEX" ] || IDX_ARGS=(--eval-index-dir "$EVAL_INDEX")
if [ -n "$DRY" ]; then
  $DRY mix "$WORK/mix" "${IDX_ARGS[@]}" > "$WORK/logs/mix.log" 2>&1; r=$?
else
  timeout "${CAP_D}m" python -X utf8 -m convert.mixdata build --config "$CFG" --dev-rows "$DATA/slices/dev-rows.jsonl.gz" \
    "${IDX_ARGS[@]}" --windows-out "$WORK/mix/windows.json.gz" --out "$WORK/mix" > "$WORK/logs/mix.log" 2>&1; r=$?
fi
if [ $r -eq 0 ] && { [ ! -s "$WORK/mix/mix.json" ] || [ ! -s "$WORK/mix/windows.json.gz" ]; }; then
  r=11; say "mix.json / window manifest missing"
fi
gate "D:mix" $r
sync_pipeline
cp -f "$WORK/mix/mix.json" "$OUT/mix.json" 2> /dev/null || true
cp -f "$WORK/mix/windows.json.gz" "$OUT/windows.json.gz" 2> /dev/null || true
export_phase D
case $r in 0) ;; 124) fail 7 "mixture timeout" ;; *) fail 11 "mixture build rc=$r" ;; esac

# ------------------------------------------------------------------ P: stage 1 + stage 2 (all 8 jointly) + final evals
say "phase P"
if [ -n "$DRY" ]; then
  $DRY pipeline "$CK" > "$WORK/logs/pipeline.log" 2>&1; r=$?
else
  timeout "${CAP_P}m" python -X utf8 -m convert.joint --config "$CFG" --mix "$WORK/mix" --out "$CK/pipeline" \
    >> "$WORK/logs/pipeline.log" 2>&1; r=$?
fi
gate "P:pipeline" $r
sync_pipeline
export_phase P
case $r in 0) ;; 3|4) fail $r "pipeline stop" ;; 124) fail 7 "pipeline timeout" ;; *) fail 2 "pipeline crashed rc=$r" ;; esac

# ------------------------------------------------------------------ X: export (verified)
say "phase X (export)"
if [ -n "$DRY" ]; then
  $DRY export "$OUT" "$WORK/out/export.json" > "$WORK/logs/export.log" 2>&1; r=$?
else
  timeout "${CAP_X}m" python -m jobs.q9af_v2.export --mode final --src "$CK/gdn/final.pt" --model-dir "$Q9B" \
    --out-dir "$OUT" --meta-out "$WORK/out/export.json" > "$WORK/logs/export.log" 2>&1; r=$?
fi
gate "X:export" $r
[ $r -eq 0 ] || { EXPORT_FAIL=1; export_phase X; fail 6 "converted-weight export failed verification"; }
EXPORTED=1
cp -f "$WORK/out/export.json" "$OUT/export.json"
export_phase X
LAYERS=$(python -c "import json,sys;print(','.join(map(str,json.load(open(sys.argv[1]))['converted'])))" "$WORK/out/export.json") || fail 6 "export.json"
LAYERS=${LAYERS%$'\r'}

# ------------------------------------------------------------------ I: Index proxy600, stock and converted arms
NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$WORK/out/index/proxy-rows.json")
NROWS=${NROWS%$'\r'}
read -r TOKS CALLS TOKS_C CALLS_C < <(python - "$CK/pipeline/results.json" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))["throughput"]
def arm(a):
    rates = [v["tok_s"] for v in a.values() if isinstance(v, dict) and "tok_s" in v]
    c = a.get("call_latency", {})
    return min(rates), max(c.get("small_call_s", 0), c.get("pack_call_s", 0)) or 0.2406
s, c = arm(t["stock"]), arm(t["converted_8"])
print(s[0], s[1], c[0], c[1])
EOF
) || fail 2 "no measured prefill rate / call latency"
CALLS_C=${CALLS_C%$'\r'}
for arm in stock conv; do
  [ -e "$WORK/out/index/$arm" ] && mv "$WORK/out/index/$arm" "$WORK/out/index/$arm.stale.$(date +%s)"
done
python -m jobs.common.rate_guard preflight --tok-s "$TOKS" --call-s "$CALLS" --volume "$WORK/out/index/volume_proxy600.json" \
  --path fork --cap-min "$CAP_IS" --out "$WORK/out/index/preflight_stock.json" > "$WORK/logs/preflight_stock.log" 2>&1; r=$?
gate "I:preflight_stock" $r
[ $r -eq 0 ] || fail $r "stock readout projected past its cap"
say "phase I (stock arm, proxy600: $NROWS rows)"
if [ -n "$DRY" ]; then
  $DRY kit stock "$WORK/out/index/stock" > "$WORK/logs/index_stock.log" 2>&1; r=$?
else
  python -m jobs.common.rate_guard run --results "$WORK/out/index/stock/results.jsonl" --total "$NROWS" \
    --cap-min "$CAP_IS" --report "$WORK/out/index/rate_stock.json" -- \
    python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family=q9af-probe --option model_dir="$Q9B" --option converted=none \
    --rows "$WORK/data/proxy-rows.jsonl.gz" --out "$WORK/out/index/stock" > "$WORK/logs/index_stock.log" 2>&1; r=$?
fi
gate "I:stock" $r
export_phase I_stock
case $r in 0) ;; 124) fail 7 "index stock timeout" ;; 15) fail 15 "index stock: measured rate too slow" ;; *) fail 12 "index stock rc=$r" ;; esac
python -m jobs.common.rate_guard preflight --tok-s "$TOKS_C" --call-s "$CALLS_C" --volume "$WORK/out/index/volume_proxy600.json" \
  --path fork --cap-min "$CAP_IC" --out "$WORK/out/index/preflight_conv.json" > "$WORK/logs/preflight_conv.log" 2>&1; r=$?
gate "I:preflight_conv" $r
[ $r -eq 0 ] || fail $r "converted readout projected past its cap"
say "phase I (converted arm: layers $LAYERS, all GDN)"
if [ -n "$DRY" ]; then
  $DRY kit conv "$WORK/out/index/conv" > "$WORK/logs/index_conv.log" 2>&1; r=$?
else
  python -m jobs.common.rate_guard run --results "$WORK/out/index/conv/results.jsonl" --total "$NROWS" \
    --cap-min "$CAP_IC" --report "$WORK/out/index/rate_conv.json" -- \
    python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family=q9af-probe --option model_dir="$Q9B" --option converted="$LAYERS" \
    --option gdn_ckpt="$OUT/gdn_converted.pt" \
    --rows "$WORK/data/proxy-rows.jsonl.gz" --out "$WORK/out/index/conv" > "$WORK/logs/index_conv.log" 2>&1; r=$?
fi
gate "I:conv" $r
export_phase I_conv
case $r in 0) ;; 124) fail 7 "index conv timeout" ;; 15) fail 15 "index conv: measured rate too slow" ;; *) fail 12 "index conv rc=$r" ;; esac

# ------------------------------------------------------------------ S: paired proxy600 compare + per-arm scores
say "phase S (proxy600 scoring, paired)"
if [ -n "$DRY" ]; then S() { $DRY subset "$@"; }
else S() { timeout "${CAP_S}m" python -X utf8 -m jobs.common.dev_subset "$@" --subset "$PROXY"; }; fi
S compare --candidate "$WORK/out/index/conv/results.jsonl" --baseline "$WORK/out/index/stock/results.jsonl" \
  > "$WORK/out/index/compare_conv_vs_stock.json" 2> "$WORK/logs/score.err" || fail 13 "compare"
for arm in stock conv; do
  S score --results "$WORK/out/index/$arm/results.jsonl" > "$WORK/out/index/score_$arm.json" 2>> "$WORK/logs/score.err" || fail 13 "score $arm"
done
gate "S:score" 0
export_phase S

# ------------------------------------------------------------------ R: the pre-registered rule
say "phase R (rule)"
python -m jobs.q9af_v2.rule "$CK/pipeline/results.json" "$WORK/out/index/compare_conv_vs_stock.json" \
  "$WORK/out/export.json" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1; r=$?
gate "R:decision" $r
[ $r -eq 0 ] || fail 9 "no valid decision (rc=$r): $(head -c 300 "$WORK/out/decision.json" 2> /dev/null)"
cp "$WORK/out/decision.json" "$OUT/decision.json"
sample "$WORK/out/decision.json" "decision"
RC=0
exit 0   # finish() re-validates decision.json via finalize.py before Q9AF2_DONE
