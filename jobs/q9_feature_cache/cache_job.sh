#!/usr/bin/env bash
# jobs/q9_feature_cache: frozen Qwen3.5-9B feature cache for the proxy head loop (PREMORTEM blocker #1; rule:
# train/proxy_rule.py, pre-registered in e9db899). SF Compute batch job (chip h100). Spec: jobs/q9_feature_cache/JOB_SPEC.md.
# Env: Q9C_MODE=<part> (REQUIRED) selects jobs/q9_feature_cache/job_params_<part>.json: ONE arm and the splits it caches
#      (<arm>_dev = select+holdout, <arm>_train = train; arms stock, conv1, e5). One job = one part: the platform packages
#      the whole output directory as ONE tar artifact bounded by 5 GiB (max_artifact_bytes), so every part is sized to
#      <= max_output_gib (4.5 GiB), checked on the estimate before the model loads and on the bytes before success.
#      Measured rates come from that file ("measured", job-5cmat).
# Order: setup (pins, fla guard, DEV rows sha256, gdn sha256, split check, weights) -> guard (content index) -> heldout
#        (D1-B scan of the whole 150k set) -> one phase per arm (cache.py under the live weighted rate guard; its own
#        in-job preflight from the measured rates exits 15 before any GPU work if the arm cannot fit its cap) -> report.
# Exit codes: 0 every arm cached + every shard re-verified + selfcheck passed   2 numerics self-check   5 setup
#   6 export verification   7 phase timeout   9 report invalid   10 data identity (DEV rows, train set, gdn sha256)
#   12 cache.py unexpected code   15 projected / measured too slow   16 output over max_output_gib (estimate / bytes)   27 guard not ready   28 cohort / train-set leak
#   30 / 31 / 32 guard: kit-suite / content index / harness manifests   33 held-out D1-B fragment outside the pinned set
#   1 finalize failed
# TEST-ONLY overrides (a paid run sets none; every default IS the paid path; each one set is logged at the top):
#   Q9C_CODE (/opt/ssdm/m1)  Q9C_CONTEXT_MANIFEST (/opt/ssdm/CONTEXT_MANIFEST.json)  Q9C_KIT_DIR (/opt/decision-index)
#   Q9C_DATA (/data)  Q9C_WORK (/tmp/q9-feature-cache)  Q9C_MODEL_DIR (unset: fetched by scripts/sf_fetch.py --stage q9)
#   Q9C_CONFIG (jobs/q9_feature_cache/cache_config.json; a test points dev_splits at small subsets of the same format)
#   Q9C_DATA_PINS (jobs/q9_feature_cache/data.sha256: the DEV rows file = jobs/clef_baseline/data.sha256's pin)
#   Q9C_DRYRUN=1  CPU dry run on a tiny stand-in: device cpu / float32, synthetic toy training records with a
#                 synthetic-only guard instead of the 150k draw (no guard / heldout phases: they need the private suite),
#                 no fla guard / CUDA asserts. Every other step (splits, DEV rows, rate guard, preflight, self-check,
#                 shards, report, finalize) runs exactly as paid.
set -Euo pipefail
MAIN_PID=$$
CODE=${Q9C_CODE:-/opt/ssdm/m1}
CONTEXT_MANIFEST=${Q9C_CONTEXT_MANIFEST:-/opt/ssdm/CONTEXT_MANIFEST.json}
KIT_DIR=${Q9C_KIT_DIR:-/opt/decision-index}
DATA=${Q9C_DATA:-/data}
WORK=${Q9C_WORK:-/tmp/q9-feature-cache}
DRY=${Q9C_DRYRUN:-0}
MODE=${Q9C_MODE:-}
PARAMS=$CODE/jobs/q9_feature_cache/job_params_${MODE}.json
CFG=${Q9C_CONFIG:-jobs/q9_feature_cache/cache_config.json}
PINS=${Q9C_DATA_PINS:-$CODE/jobs/q9_feature_cache/data.sha256}
TRAIN_DIR=$DATA/m2-150k
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1500
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs" "$CK"
export SSDM_DATA=$DATA SSDM_DATA_ROOT=$WORK/data PYTHONUNBUFFERED=1 PYTHONUTF8=1
mkdir -p "$SSDM_DATA_ROOT"
HF_DIR=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p "$HF_DIR"
if [ -d "${HF_HUB_CACHE:-}" ]; then ln -sfn "$HF_HUB_CACHE" "$SSDM_DATA_ROOT/hf"; fi
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
OVERRIDES=$(env | grep -o '^Q9C_[A-Z0-9_]*' | grep -v '^Q9C_MODE$' | sort | tr '\n' ' ' || true)
[ -z "$OVERRIDES" ] || say "TEST OVERRIDES SET (never in a paid run): $OVERRIDES"
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/cache_result.json" "$OUT" Q9_CACHE_DONE jobs.q9_feature_cache.report:valid_result
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
pval() { python -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'): d=d[k]
print(' '.join(map(str, d)) if isinstance(d, list) else d)" "$PARAMS" "$1"; }
# the interpreter's own path for the child the rate guard spawns (a bare "python" there is resolved by the OS, not by
# this shell's PATH: on Windows dry runs that can be a different interpreter)
PY=$(python -c "import sys; print(sys.executable)")
hb
watchdog & WD_PID=$!
trap finish EXIT
case "$MODE" in stock_dev|stock_train|conv1_dev|conv1_train|e5_dev|e5_train) ;; *) fail 5 "Q9C_MODE must be one of stock_dev stock_train conv1_dev conv1_train e5_dev e5_train (got '$MODE')" ;; esac
[ -f "$PARAMS" ] || fail 5 "no $PARAMS"
ARMS=$(pval arms) && SPLITS=$(pval splits | tr ' ' ',') && MAXGIB=$(pval max_output_gib) && TOK_S=$(pval measured.stock_prefill_tok_s_min) && CALL_S=$(pval measured.call_s_fitted_5cmat) \
  || fail 5 "job_params"
say "part $MODE: arm [$ARMS] splits [$SPLITS] (<= $MAXGIB GiB) at TOK_S=$TOK_S CALL_S=$CALL_S"

# ------------------------------------------------------------------ setup (exit 5 / 10)
say "setup"
cp "$CONTEXT_MANIFEST" "$WORK/out/" || fail 5 "no context manifest"
cp "$PARAMS" "$CFG" "$WORK/out/"
if [ "$DRY" = 1 ]; then
  DEVICE=cpu DTYPE=float32
  gate "setup:fla_guard" 0
else
  DEVICE=cuda DTYPE=bfloat16
  nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
  python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins"
import json, torch, transformers, importlib.metadata as m
pins = {"transformers": "5.17.0", "flash-linear-attention": "0.5.2", "fla-core": "0.5.2"}
got = {k: m.version(k) for k in pins}
assert got == pins, got
assert torch.cuda.is_available()
import decision_index  # noqa: F401
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), **got}))
EOF
  python -I "$CODE/jobs/common/fla_guard.py" --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 \
    || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
fi
# static preflight (jobs/q9_feature_cache/preflight.py): every cap >= 1.3 x its projection from the MEASURED job-5cmat
# rates and the exact committed volume, schedule == max_duration, worst case <= hard cap; else exit 5 before any GPU work
python -m jobs.q9_feature_cache.preflight --mode "$MODE" --out "$WORK/out/preflight.json" > "$WORK/logs/preflight.log" 2>&1   || fail 5 "preflight: a cap violates the measured-rate margin ($WORK/out/preflight.json)"
( cd "$DATA" && sha256sum -c "$PINS" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
case ",$SPLITS," in *,train,*)  # the 150k set + the suite ship only with parts that draw training rows
  [ "$DRY" = 1 ] || ( cd "$DATA" && sha256sum -c "$CODE/jobs/q9_feature_cache/data_train.sha256" )     >> "$WORK/out/data-verify.txt" 2>&1 || fail 10 "training set / suite sha256" ;;
esac
python -X utf8 -m jobs.q9_feature_cache.splits check > "$WORK/out/splits-check.json" 2> "$WORK/logs/splits.err" \
  || fail 10 "SELECT / HOLDOUT split identity"
case " $ARMS " in *" conv1 "*)  # the converted GDN ships only with the conv1 part
  ( cd "$DATA" && sha256sum -c "$CODE/jobs/q9_feature_cache/gdn_converted.sha256" ) > "$WORK/out/gdn-verify.txt" 2>&1 \
    || fail 10 "gdn_converted.pt sha256" ;;
esac
if [ -n "${Q9C_MODEL_DIR:-}" ]; then
  Q9B=$Q9C_MODEL_DIR
else
  timeout 25m python -I scripts/sf_fetch.py --stage q9 > "$WORK/logs/fetch.log" 2>&1
  [ -f "$SSDM_DATA_ROOT/DONE_Q9B_DIR" ] || fail 5 "9B weights unavailable"
  # shellcheck disable=SC1091
  . "$SSDM_DATA_ROOT/paths.env"
  Q9B=$Q9B_DIR
fi
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ guard + heldout (parts that draw training rows)
if [ "${SPLITS#*train}" = "$SPLITS" ]; then
  say "part $MODE draws no training rows: guard / heldout not needed (DEV rows are never training data)"
  gate "guard:no_train_split" 0
  TRAIN_ARGS=()
elif [ "$DRY" = 1 ]; then
  say "DRY RUN: guard / heldout skipped (toy records + synthetic-only guard)"
  gate "guard:skipped_dryrun" 0
  TRAIN_ARGS=(--toy-train-records "$DATA/toy-train.jsonl.gz")
else
  say "phase guard"
  timeout "$(cap guard)m" python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir "$DATA/suite-0.2" --kit-dir "$KIT_DIR" \
    --out-dir "$WORK/out/guard" > "$WORK/logs/guard.log" 2>&1; r=$?
  gate "guard" $r
  case $r in 0) ;; 124) fail 7 "guard timeout" ;; 30|31|32) fail $r "guard not ready" ;; *) fail 5 "guard rc=$r" ;; esac
  say "phase heldout"
  timeout "$(cap heldout)m" python -X utf8 -m train.train_set heldout --config "$CFG" --train-dir "$TRAIN_DIR" \
    --out "$WORK/out/heldout.json" > "$WORK/logs/heldout.log" 2>&1; r=$?
  gate "heldout" $r
  case $r in 0) ;; 124) fail 7 "heldout timeout" ;; 3) fail 33 "held-out D1-B hits != pinned exclusions" ;; *) fail 5 "heldout rc=$r" ;; esac
  TRAIN_ARGS=(--train-dir "$TRAIN_DIR" --guard-readiness "$WORK/out/guard/readiness.json")
fi
export_phase guard

# ------------------------------------------------------------------ one phase per arm (exit 2 / 7 / 10 / 12 / 15 / 27 / 28)
for ARM in $ARMS; do
  say "phase arm_$ARM"
  CAP=$(cap "arm_$ARM") || fail 5 "no cap for arm_$ARM"
  GDN_ARGS=()
  [ "$ARM" = conv1 ] && GDN_ARGS=(--gdn-ckpt "$DATA/gdn/gdn_converted.pt")
  python -m jobs.common.rate_guard run-weighted --progress "$WORK/logs/progress_$ARM.jsonl" --cap-min "$CAP" \
    --report "$WORK/out/rate_$ARM.json" -- \
    "$PY" -X utf8 -m jobs.q9_feature_cache.cache run --arm "$ARM" --config "$CFG" --model-dir "$Q9B" \
    --dev-rows "$DATA/slices/dev-rows.jsonl.gz" --out "$OUT/cache" "${TRAIN_ARGS[@]}" "${GDN_ARGS[@]}" \
    --device "$DEVICE" --dtype "$DTYPE" --tok-s "$TOK_S" --call-s "$CALL_S" --cap-min "$CAP" \
    --progress "$WORK/logs/progress_$ARM.jsonl" --splits "$SPLITS" > "$WORK/logs/arm_$ARM.log" 2>&1; r=$?
  cp "$OUT/cache/$ARM/summary_${SPLITS//,/+}.json" "$WORK/out/summary_$ARM.json" 2> /dev/null || true
  gate "arm_$ARM" $r
  export_phase "arm_$ARM"
  case $r in 0) ;; 2|10|15|16|27|28) fail $r "arm $ARM: $(tail -1 "$WORK/logs/arm_$ARM.log")" ;; 124) fail 7 "arm $ARM timeout" ;;
    *) fail 12 "arm $ARM rc=$r" ;; esac
done

# ------------------------------------------------------------------ report (exit 9)
say "phase report"
timeout "$(cap report)m" python -X utf8 -m jobs.q9_feature_cache.report --cache "$OUT/cache" --arms "$ARMS" \
  --splits "$SPLITS" --output-dir "$OUT" --max-gib "$MAXGIB" \
  --result "$WORK/out/cache_result.json" > "$WORK/logs/report.log" 2>&1; r=$?
gate "report" $r
[ $r -eq 0 ] || fail 9 "report invalid (see out/cache_result.json)"
sample "$WORK/out/cache_result.json" "cache_result"
RC=0
exit 0   # finish() re-validates cache_result.json via finalize.py before Q9_CACHE_DONE
