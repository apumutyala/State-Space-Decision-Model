#!/usr/bin/env bash
# jobs/q9af_cell: the first Q9-AF decision cell (Clef-flash recipe analogue on the attention-free Qwen3.5-9B) and its
# parent arm. SF Compute batch job (chip h100). Spec: jobs/q9af_cell/JOB_SPEC.md. Rule: jobs/q9af_cell/rule.py (2d18afa).
# Env: Q9AFC_MODE (REQUIRED) = parent | cell_r64 | cell_r256 -> jobs/q9af_cell/job_params_<mode>.json (caps, pins).
#   parent     setup -> guard -> parent_g12 -> parent_g13 (freezes the full-DEV row set first) -> report
#   cell_r*    setup -> guard -> heldout -> frozen -> cell (smoke, ladder, train; resumable) -> evals (G12 + G13 vs the
#              pinned parent arm = lineage anchor) -> report (summarize + the rule verdict; LoRA/head weights exported)
# Exit codes: 0 measured + verdict recorded + exports verified (a FOLD / ESCALATE / gate REJECT is a measured outcome)
#   5 setup / preflight (a cap under 1.3 x its measured-rate projection)   6 export verification   7 phase timeout
#   9 report invalid   10 identity: data sha256, export pin (unpinned / partial without hybrid), parent arm pins
#   20 run_cell unexpected rc   21-29 run_cell stops (21 fork mismatch, 22 no T fits, 23 non-finite, 24 memory,
#   25 data, 26 too slow, 27 guard not ready, 28 cohort / train-set leak, 29 q9af lineage)   35 config / hooks
#   30 / 31 / 32 guard: kit-suite / content index / harness manifests   33 held-out D1-B   34 frozen-family hit
#   12 parent scoring unexpected rc   1 finalize failed
# TEST-ONLY overrides (a paid run sets none; every default IS the paid path; each one set is logged at the top):
#   Q9AFC_CODE Q9AFC_CONTEXT_MANIFEST Q9AFC_KIT_DIR Q9AFC_DATA Q9AFC_WORK Q9AFC_MODEL_DIR Q9AFC_CONFIG
#   Q9AFC_PARENT_DIR Q9AFC_PARENT_ARM_SHA256
#   Q9AFC_DRYRUN=1  CPU dry run on a tiny Qwen3.5 stand-in (q9af_from_stock-converted, byte tokenizer): device cpu /
#                   float32, toy training records (legacy cohort-less split, synthetic-only guard), no fla guard / CUDA
#                   asserts / weight fetch / guard / heldout phases (they need the private suite); the GPU scoring
#                   phases run jobs.q9af_cell.dry_gates (same output contracts, synthetic kit-format rows). Every other
#                   step (preflight, export pin check, frozen-family check, run_cell smoke/ladder/train with act_ckpt,
#                   checkpoint resume, summarize + the rule, finalize) runs exactly as paid.
set -Euo pipefail
MAIN_PID=$$
CODE=${Q9AFC_CODE:-/opt/ssdm/m1}
CONTEXT_MANIFEST=${Q9AFC_CONTEXT_MANIFEST:-/opt/ssdm/CONTEXT_MANIFEST.json}
KIT_DIR=${Q9AFC_KIT_DIR:-/opt/decision-index}
DATA=${Q9AFC_DATA:-/data}
WORK=${Q9AFC_WORK:-/tmp/q9af-cell}
DRY=${Q9AFC_DRYRUN:-0}
MODE=${Q9AFC_MODE:-}
PARAMS=$CODE/jobs/q9af_cell/job_params_${MODE}.json
TRAIN_DIR=$DATA/m2-150k
PARENT_DIR=${Q9AFC_PARENT_DIR:-$DATA/parent_arm}
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1500
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs" "$CK"
export SSDM_DATA=$DATA SSDM_DATA_ROOT=$WORK/data PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SSDM_PROGRESS_BEAT=$WORK/logs/progress.beat
mkdir -p "$SSDM_DATA_ROOT"
HF_DIR=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}
mkdir -p "$HF_DIR"
if [ -d "${HF_HUB_CACHE:-}" ]; then ln -sfn "$HF_HUB_CACHE" "$SSDM_DATA_ROOT/hf"; fi
T0=$(date +%s)
EXPORT_FAIL=0
RC=""
GATES="{}"
DECISION=$WORK/out/report/q9af_cell_result.json
VALIDATOR=jobs.q9af_cell.summarize:valid_result
exec > >(tee -a "$WORK/logs/run.log") 2>&1
cd "$CODE"
# shellcheck source=../common/lib.sh
. "$CODE/jobs/common/lib.sh"
# shellcheck source=../clef_baseline/watchdog.sh
. "$CODE/jobs/clef_baseline/watchdog.sh"
OVERRIDES=$(env | grep -o '^Q9AFC_[A-Z0-9_]*' | grep -v '^Q9AFC_MODE$' | sort | tr '\n' ' ' || true)
[ -z "$OVERRIDES" ] || say "TEST OVERRIDES SET (never in a paid run): $OVERRIDES"
if [ -n "${Q9AFC_KIT_DIR:-}" ]; then export PYTHONPATH="$KIT_DIR${PYTHONPATH:+:$PYTHONPATH}"; fi
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$DECISION" "$OUT" Q9AF_CELL_DONE "$VALIDATOR"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
pval() { python -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'): d=d[k]
print('' if d is None else d)" "$PARAMS" "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT
case "$MODE" in parent|cell_r64|cell_r256) ;; *) fail 5 "Q9AFC_MODE must be parent, cell_r64 or cell_r256 (got '$MODE')" ;; esac
[ -f "$PARAMS" ] || fail 5 "no $PARAMS"
CELL=${Q9AFC_CONFIG:-$CODE/$(pval cell_config)} || fail 5 "job_params"
[ -f "$CELL" ] || fail 5 "no cell config $CELL"
say "mode $MODE: cell config $CELL"

# ------------------------------------------------------------------ setup (exit 5 / 10 / 35)
say "setup"
cp "$CONTEXT_MANIFEST" "$WORK/out/" || fail 5 "no context manifest"
cp "$PARAMS" "$CELL" "$WORK/out/"
if [ "$DRY" = 1 ]; then
  DEVICE=cpu
  gate "setup:fla_guard_skipped_dryrun" 0
else
  DEVICE=cuda
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
# measured-rate preflight (jobs/q9af_cell/preflight.py): every cap >= 1.3 x its projection from job-5cmat's measured
# rates + the act-ckpt projection; schedule == max_duration <= 720; worst case <= hard cap; else exit 5, before any GPU work
python -m jobs.q9af_cell.preflight --mode "$MODE" --out "$WORK/out/preflight.json" > "$WORK/logs/preflight.log" 2>&1 \
  || fail 5 "preflight: a cap violates the measured-rate margin ($WORK/out/preflight.json)"
python -c "import json,sys; from jobs.q9af_cell import hooks; c=json.load(open(sys.argv[1])); assert hooks.apply(c)==c, 'unresolved head_from_proxy'" \
  "$CELL" > "$WORK/logs/hooks.log" 2>&1 || fail 35 "hooks: $(tail -1 "$WORK/logs/hooks.log")"
if [ "$DRY" != 1 ]; then
  ( cd "$DATA" && sha256sum -c "$CODE/jobs/clef_baseline/data.sha256" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
  ( cd "$DATA" && sha256sum -c "$CODE/jobs/rwkv7/records.sha256" ) >> "$WORK/out/data-verify.txt" 2>&1 || fail 10 "PR #16 records sha256"
  ( cd "$DATA" && sha256sum -c "$CODE/jobs/q9_feature_cache/data_train.sha256" ) >> "$WORK/out/data-verify.txt" 2>&1 \
    || fail 10 "150k set / suite sha256"
  ( cd "$DATA" && sha256sum -c "$CODE/jobs/q9af_cell/gdn.sha256" ) >> "$WORK/out/data-verify.txt" 2>&1 \
    || fail 10 "converted GDN export sha256 (gdn.sha256 PENDING until jobs.q9af_cell.pin fill)"
fi
GDN_FILE=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('file') or '')" "$DATA/gdn/export.json") \
  || fail 10 "no export.json"
GDN=$DATA/gdn/$GDN_FILE
# export pin: unpinned -> refuse; partial (gdn_partial.pt) -> refuse unless the config says hybrid (then labelled NOT
# attention-free in every report); a full export configured as hybrid -> refuse
python -m jobs.q9af_cell.pin check --config "$CELL" --export-json "$DATA/gdn/export.json" --gdn "$GDN" \
  --out "$WORK/out/label.json" > "$WORK/logs/pin.log" 2>&1 || fail 10 "export pin: $(tail -1 "$WORK/logs/pin.log")"
say "export: $(python -c "import json,sys; print(json.load(open(sys.argv[1]))['label'])" "$WORK/out/label.json")"
if [ -n "${Q9AFC_MODEL_DIR:-}" ]; then
  Q9B=$Q9AFC_MODEL_DIR
else
  timeout "$(cap setup)m" python -I scripts/sf_fetch.py --stage q9 > "$WORK/logs/fetch.log" 2>&1
  [ -f "$SSDM_DATA_ROOT/DONE_Q9B_DIR" ] || fail 5 "9B weights unavailable"
  # shellcheck disable=SC1091
  . "$SSDM_DATA_ROOT/paths.env"
  Q9B=$Q9B_DIR
  ( cd "$Q9B" && sha256sum -c "$CODE/jobs/m2_pilot/q9af_snapshot.sha256" ) > "$WORK/out/snapshot-verify.txt" 2>&1 \
    || fail 10 "9B snapshot sha256"
fi
TEACHER_ARGS=()  # KD cells only (config 'teacher'): the pinned Clef-flash export (train.run_cell.attach_teacher)
if python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('teacher') else 1)" "$CELL"; then
  [ "$MODE" != parent ] || fail 35 "the parent arm takes no teacher"
  if [ "$DRY" != 1 ]; then
    TSHA=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['teacher']['teacher_sha256'])" "$CELL")
    [ "$(sha256sum "$DATA/teacher/teacher-clef-flash.jsonl" | cut -d' ' -f1)" = "$TSHA" ] || fail 10 "teacher file sha256 != config"
  fi
  TEACHER_ARGS=(--teacher-dir "$DATA/teacher")
  say "KD cell: teacher $DATA/teacher"
fi
if [ "$MODE" != parent ]; then  # the pinned parent arm (comparator + lineage anchor) and its frozen full-DEV row set
  ARM_SHA=${Q9AFC_PARENT_ARM_SHA256:-$(pval parent_arm_sha256)}
  FROZEN_SHA=$(pval index_frozen_sha256)
  [ -n "$ARM_SHA" ] || fail 10 "parent_arm_sha256 is not pinned in $PARAMS (run the q9af-cell-parent job first)"
  [ "$(sha256sum "$PARENT_DIR/arm.json" | cut -d' ' -f1)" = "$ARM_SHA" ] || fail 10 "parent arm.json sha256 != pinned"
  if [ "$DRY" != 1 ]; then
    [ -n "$FROZEN_SHA" ] || fail 10 "index_frozen_sha256 is not pinned in $PARAMS"
    for f in units-g12.json units-g13.json; do
      [ "$(sha256sum "$PARENT_DIR/$f" | cut -d' ' -f1)" = "$(pval "parent_arm_files.$f")" ] || fail 10 "parent $f sha256 != pinned"
    done
  fi
fi
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ guard (+ heldout, frozen-family for training modes)
if [ "$DRY" = 1 ]; then
  say "DRY RUN: guard / heldout skipped (toy records + synthetic-only guard; they need the private suite)"
  gate "guard:skipped_dryrun" 0
  GUARD_ARGS=()
  DATA_ARGS=(--records "$DATA/toy-train.jsonl.gz")
else
  say "phase guard"
  timeout "$(cap guard)m" python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir "$DATA/suite-0.2" --kit-dir "$KIT_DIR" \
    --out-dir "$WORK/out/guard" > "$WORK/logs/guard.log" 2>&1; r=$?
  gate "guard" $r
  case $r in 0) ;; 124) fail 7 "guard timeout" ;; 30|31|32) fail $r "guard not ready" ;; *) fail 5 "guard rc=$r" ;; esac
  GUARD_ARGS=(--guard-readiness "$WORK/out/guard/readiness.json")
  DATA_ARGS=(--records "$DATA/m2/records.jsonl.gz")
  if [ "$MODE" != parent ]; then
    say "phase heldout"
    timeout "$(cap heldout)m" python -X utf8 -m train.train_set heldout --config "$CELL" --train-dir "$TRAIN_DIR" \
      --out "$WORK/out/heldout.json" > "$WORK/logs/heldout.log" 2>&1; r=$?
    gate "heldout" $r
    case $r in 0) ;; 124) fail 7 "heldout timeout" ;; 3) fail 33 "held-out D1-B or eval attack template hits != pinned exclusions" ;; *) fail 5 "heldout rc=$r" ;; esac
    DATA_ARGS+=(--train-dir "$TRAIN_DIR")
  fi
fi
export_phase guard
if [ "$MODE" != parent ]; then
  say "phase frozen (heldout-draw-v1 frozen-family check over every drawn training record)"
  if [ "$DRY" = 1 ]; then FZ_SRC=(--toy-records "$DATA/toy-train.jsonl.gz"); else FZ_SRC=(--train-dir "$TRAIN_DIR"); fi
  timeout "$(cap frozen)m" python -X utf8 -m jobs.q9af_cell.frozen --config "$CELL" "${FZ_SRC[@]}" \
    --out "$WORK/out/frozen.json" > "$WORK/logs/frozen.log" 2>&1; r=$?
  gate "frozen" $r
  case $r in 0) ;; 124) fail 7 "frozen timeout" ;; 34) fail 34 "a drawn record targets a frozen held-out family" ;;
    10) fail 28 "draw refused: $(tail -1 "$WORK/logs/frozen.log")" ;; *) fail 5 "frozen rc=$r" ;; esac
  export_phase frozen
fi

# ------------------------------------------------------------------ parent mode: score the parent arm once
if [ "$MODE" = parent ]; then
  mkdir -p "$WORK/out/report"
  if [ "$DRY" = 1 ]; then
    timeout "$(cap parent_g13)m" python -X utf8 -m jobs.q9af_cell.dry_gates parent --config "$CELL" --model-dir "$Q9B" \
      --gdn-ckpt "$GDN" --out "$OUT/parent" > "$WORK/logs/parent.log" 2>&1; r=$?
    gate "parent_dryrun" $r
    [ $r -eq 0 ] || fail 12 "dry parent rc=$r: $(tail -1 "$WORK/logs/parent.log")"
  else
    for ST in g12 g13; do
      say "phase parent_$ST"
      FZ=()
      [ "$ST" = g13 ] && FZ=(--index-rows "$DATA/slices/dev-rows.jsonl.gz" --index-frozen-out "$OUT/parent/parent_arm/index-frozen.json")
      timeout "$(cap "parent_$ST")m" python -X utf8 -m jobs.m2_parent_arm.score_parent --parent q9af --model-dir "$Q9B" \
        --gdn-ckpt "$GDN" --cell-config "$CELL" "${DATA_ARGS[@]}" "${GUARD_ARGS[@]}" --pool-dir "m2-150k=$TRAIN_DIR" \
        "${FZ[@]}" --out-dir "$OUT/parent/parent_arm" --stage "$ST" --device "$DEVICE" > "$WORK/logs/parent_$ST.log" 2>&1; r=$?
      gate "parent_$ST" $r
      export_phase "parent_$ST"
      case $r in 0) ;; 124) fail 7 "parent_$ST timeout" ;; 10) fail 10 "parent identity: $(tail -1 "$WORK/logs/parent_$ST.log")" ;;
        *) fail 12 "parent_$ST rc=$r" ;; esac
    done
  fi
  say "phase report (parent pins)"
  python - "$OUT/parent" "$WORK/out/label.json" "$DECISION" > "$WORK/logs/report.log" 2>&1 <<'EOF' || fail 9 "parent report"
import hashlib, json, sys
from pathlib import Path
d, label, out = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
arm = d / "parent_arm"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
pins = {"parent_arm_sha256": sha(arm / "arm.json"), "index_frozen_sha256": sha(arm / "index-frozen.json"),
        "parent_arm_files": {f: sha(arm / f) for f in ("units-g12.json", "units-g13.json")}}
from jobs.q9af_cell import rule as R
res = {"schema": "ssdm.q9af_cell.result.v1", "rule": R.RULE_ID, "variant": "parent", "verdict": "NO_MEASUREMENT",
       "why": ["parent arm: a measurement for the cells, no verdict of its own"], "label": label,
       "pins_for_cell_params": pins, "index": {}, "lora_rank": None}
out.write_text(json.dumps(res, indent=1))
print(json.dumps(pins))
EOF
  VALIDATOR=jobs.q9af_cell.summarize:valid_parent
  gate "report" 0
  sample "$DECISION" "parent_pins"
  RC=0
  exit 0
fi

# ------------------------------------------------------------------ cell: smoke -> ladder -> train (resumable; exit 20-29 / 35)
say "phase cell (smoke -> ladder -> train under act_ckpt; resumes from $CK/cell)"
mkdir -p "$CK/cell" "$WORK/out/cell"
[ -f "$CK/cell/cell.json" ] && cp "$CK/cell/cell.json" "$WORK/out/cell/cell.json"
timeout "$(cap cell)m" python -X utf8 -m train.run_cell --config "$CELL" --model-dir "$Q9B" --gdn-ckpt "$GDN" \
  "${DATA_ARGS[@]}" "${GUARD_ARGS[@]}" "${TEACHER_ARGS[@]}" --out "$WORK/out/cell" --ckpt-dir "$CK/cell" --stage all --device "$DEVICE" \
  > "$WORK/logs/cell.log" 2>&1; TRAIN_RC=$?
cp "$WORK/out/cell/cell.json" "$CK/cell/cell.json" 2> /dev/null || true
gate "cell" $TRAIN_RC
export_phase cell
case $TRAIN_RC in 0) ;; 124) fail 7 "cell timeout" ;; 21|22|23|24|25|26|27|28|29|35) fail $TRAIN_RC "cell stop: $(tail -1 "$WORK/logs/cell.log")" ;;
  *) fail 20 "cell unexpected rc=$TRAIN_RC" ;; esac

# ------------------------------------------------------------------ evals: G12 + G13 vs the parent arm (lineage anchor)
say "phase evals (temperature fit, dev card, G12 steer_dev, G13 steer_gate + cohort v3 retention + full DEV)"
mkdir -p "$WORK/out/gates"
if [ "$DRY" = 1 ]; then
  timeout "$(cap evals)m" python -X utf8 -m jobs.q9af_cell.dry_gates child --config "$CELL" --model-dir "$Q9B" \
    --gdn-ckpt "$GDN" --cell "$WORK/out/cell/cell.json" "${DATA_ARGS[@]}" --ckpt-dir "$CK/cell" \
    --parent-dir "$(dirname "$PARENT_DIR")" --out "$WORK/out/gates" > "$WORK/logs/evals.log" 2>&1; GATES_RC=$?
  INDEX_ARGS=(--index-rows "$(dirname "$PARENT_DIR")/dev-rows.jsonl.gz" --dry-run)
else
  timeout "$(cap evals)m" python -X utf8 -m train.m2_gates --cell "$WORK/out/cell/cell.json" --ckpt-dir "$CK/cell" \
    --model-dir "$Q9B" --gdn-ckpt "$GDN" "${DATA_ARGS[@]}" "${GUARD_ARGS[@]}" "${TEACHER_ARGS[@]}" --pool-dir "m2-150k=$TRAIN_DIR" \
    --parent-arm "$PARENT_DIR" --parent-arm-sha256 "$ARM_SHA" --anchor-arm "$PARENT_DIR" --anchor-arm-sha256 "$ARM_SHA" \
    --index-rows "$DATA/slices/dev-rows.jsonl.gz" --index-frozen "$PARENT_DIR/index-frozen.json" \
    --index-frozen-sha256 "$FROZEN_SHA" --out "$WORK/out/gates/report.json" --stage all --device "$DEVICE" \
    > "$WORK/logs/evals.log" 2>&1; GATES_RC=$?
  INDEX_ARGS=(--index-rows "$DATA/slices/dev-rows.jsonl.gz" --index-frozen "$PARENT_DIR/index-frozen.json"
    --index-frozen-sha256 "$FROZEN_SHA")
fi
gate "evals" $GATES_RC
export_phase evals
case $GATES_RC in 0|40|41|42|43|44|45) say "gates rc $GATES_RC (a measured outcome; the rule decides)" ;;
  124) fail 7 "evals timeout" ;; 10) fail 10 "evals identity: $(tail -1 "$WORK/logs/evals.log")" ;; *) fail 9 "evals rc=$GATES_RC" ;; esac

# ------------------------------------------------------------------ report: summarize + rule verdict; LoRA/head weights
say "phase report"
timeout "$(cap report)m" python -X utf8 -m jobs.q9af_cell.summarize cell --cell "$WORK/out/cell/cell.json" \
  --train-rc "$TRAIN_RC" --gates-report "$WORK/out/gates/report.json" --gates-rc "$GATES_RC" \
  --child-units "$WORK/out/gates/units-child-g13.json" --parent-arm "$PARENT_DIR" "${INDEX_ARGS[@]}" \
  --label "$WORK/out/label.json" --out-dir "$WORK/out/report" > "$WORK/logs/report.log" 2>&1; r=$?
gate "report" $r
[ $r -eq 0 ] || fail 9 "report invalid (see $WORK/logs/report.log)"
python -X utf8 -m jobs.q9af_cell.summarize export --ckpt-dir "$CK/cell" --out "$OUT/adapters" > "$WORK/logs/export.log" 2>&1 \
  || fail 6 "trained-weight export: $(tail -1 "$WORK/logs/export.log")"
cp "$WORK/out/report/report.json" "$WORK/out/report/q9af_cell_result.json" "$OUT/"
sample "$DECISION" "q9af_cell_result"
RC=0
exit 0   # finish() re-validates q9af_cell_result.json via finalize.py before Q9AF_CELL_DONE
