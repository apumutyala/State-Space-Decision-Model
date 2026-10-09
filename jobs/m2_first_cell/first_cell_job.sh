#!/usr/bin/env bash
# jobs/m2_first_cell: the first TRAINED attention-free decision model (RWKV7-G1k-13.3B + LoRA r64 + cache-fork head,
# Clef loss arm, PR #44 150k injection-augmented set), measured against its own stock readout on the 2,008-row Index
# DEV subset and the G12 development cohort. SF Compute batch job (chip h100). DRAFT, NOT SUBMITTED.
# Spec: jobs/m2_first_cell/JOB_SPEC.md. Caps: jobs/m2_first_cell/job_params.json. Config: m2_first_cell_rwkv7.json.
# Usage: first_cell_job.sh [MODE]. MODE base (default) = the files above, exactly as before modes existed. Any other
#   MODE (lowercase [a-z0-9_]) is generic, no per-mode code here: config m2_first_cell_rwkv7_<MODE>.json, caps
#   job_params_<MODE>.json, both required (exit 5 if missing or if the params' own "mode" / "cell_config" disagree).
#   Optional params "train_dir" (relative to $DATA, default m2-150k) moves the training set. A cell config with a
#   "teacher" (KD) block verifies $DATA/teacher before any training and passes --teacher-dir to run_cell / m2_gates.
#   Params with "eval_only": "fulldev" (mode fulldev_eval) are an EVAL-ONLY run of a SAVED checkpoint on the FULL Index
#   DEV slice: setup + guard as usual, then phase checkpoint (jobs.m2_first_cell.checkpoint_loader: the artifact from
#   $<checkpoint.url_env>, every byte sha256-pinned), fulldev (jobs.m2_first_cell.fulldev_eval score --arm child, all
#   12,104 rows, or params "rows": "smoke" = the pre-declared 250-row fulldev_smoke_rows.json; live rate guard, per-row
#   resume from $CK/fulldev), optional parent_full (params "fulldev_parent_arm"),
#   report (fulldev_eval report -> fulldev_eval_result.json). No heldout / parent g12+index / reprice / cell phase.
# Env: TOK_S=<MEASURED prefill tok/s>  CALL_S=<MEASURED per-call s>  (job-tkfaj probe: 3757 / 0.117)
# Order: setup (identities + every eval preflight) -> guard -> heldout -> parent arm (g12, index; live rate guard;
#        cached in $CK) -> reprice child evals at the measured speed -> cell (train; measured-rate stops) ->
#        child evals (g12 + index) -> report.
# Exit codes: 0 measured + result valid + exports verified (m2_gates 0 / 40 G12 reject / 43 pending / 44 partial /
#   45 underpowered are all MEASURED outcomes: reported, verdict "measured", not promotable unless 0, exit 0)
#   4 TOK_S / CALL_S missing or not a number   5 setup   6 export verification   7 phase timeout
#   9 report / arm / evals missing or invalid   10 snapshot / vendored code / records / train set / cohort / arm identity
#      (incl. a shipped parent arm seed that fails its pin or identity: jobs.m2_first_cell.arm_seed)
#   15 an eval phase projected (preflight at the env rates, or re-priced at the parent arm's measured speed BEFORE
#      training) or measured (live rate guard) past its cap
#   30 / 31 / 32 guard: kit-suite / content index / harness manifests   33 held-out D1-B fragment outside the pinned set
#   20 train.run_cell unexpected code   21-29 train.run_cell stops (21 fork mismatch, 22 no T fits, 23 non-finite,
#   24 memory, 25 data, 26 too slow for min_steps, 27 guard not ready, 28 cohort / train-set leak or mismatch,
#   35 head config, 36 fla requested but a training step fell back to "chunked" or no training call ran fla)
# fla: ONLY when the cell config sets "rwkv7_wkv" or "train_rwkv7_wkv" to an fla name, setup installs fla-core 0.5.2 by wheel hash, no deps
#   (jobs/fla_rwkv7/requirements-fla.txt, as jobs/perf_step_gate v2), checks torch 2.13.0 / triton 3.7.1 / fla 0.5.2 and
#   the chunk_rwkv7 import, then the #640 guard (fla_guard.py) must see the installed fla: any failure is exit 5. run_cell
#   trains under heads.rwkv7_wkv.strict and records fla calls vs fallbacks in cell.json "fla". Other cells: unchanged.
#   "train_rwkv7_wkv" (default = rwkv7_wkv) selects the WKV for TRAINING STEPS only; everything that scores (parent arm,
#   seed check, child evals / G12 / Index, dev cards, smoke) uses "rwkv7_wkv" (identity unchanged).
#   1 finalize failed
# Path overrides (TEST-ONLY; a paid run sets none of them, and every default below IS the paid-run path). They exist
# so tests/test_m2_first_cell_shell.py can EXECUTE this script end to end on CPU with a tiny stand-in model; they only
# relocate inputs/outputs, no check is skipped or relaxed (every sha256 / identity / rate / cap guard still runs on
# whatever the paths hold). Any M2FC_* variable that is set is logged at the top of run.log.
#   M2FC_CODE              code root                       (default /opt/ssdm/m1, the image's COPY code/m1)
#   M2FC_CONTEXT_MANIFEST  build-context manifest          (default /opt/ssdm/CONTEXT_MANIFEST.json)
#   M2FC_KIT_DIR           Decision Index kit clone        (default /opt/decision-index)
#   M2FC_DATA              data root (records, 150k set, suite, DEV rows, references; also SSDM_DATA)  (default /data)
#   M2FC_WORK              scratch dir                     (default /tmp/m2-first-cell)
#   M2FC_CELL              cell config (absolute, or relative to CODE, the working dir)  (default jobs/m2_first_cell/m2_first_cell_rwkv7.json)
# Already overridable by the platform contract: GMN_OUTPUT_DIR, GMN_RESULT_PATH, GMN_CHECKPOINT_DIR, GMN_METADATA_URL;
# the snapshot download honours the standard HF_HUB_CACHE / HF_HOME / HF_HUB_OFFLINE. nvidia-smi, the CUDA asserts,
# the snapshot sha256 pins and the TOK_S / CALL_S rates have NO override here (the test stubs them from outside).
set -Euo pipefail
MAIN_PID=$$
CODE=${M2FC_CODE:-/opt/ssdm/m1}
CONTEXT_MANIFEST=${M2FC_CONTEXT_MANIFEST:-/opt/ssdm/CONTEXT_MANIFEST.json}
KIT_DIR=${M2FC_KIT_DIR:-/opt/decision-index}
DATA=${M2FC_DATA:-/data}
MODE=${1:-base}
if [ "$MODE" = base ]; then
  CELL=${M2FC_CELL:-jobs/m2_first_cell/m2_first_cell_rwkv7.json}
  PARAMS=$CODE/jobs/m2_first_cell/job_params.json
else  # generic: the mode's two files; resolved and checked once the log and finalize trap exist (below)
  CELL=${M2FC_CELL:-jobs/m2_first_cell/m2_first_cell_rwkv7_$MODE.json}
  PARAMS=$CODE/jobs/m2_first_cell/job_params_$MODE.json
fi
TRAIN_DIR=$DATA/m2-150k
TEACHER_ARGS=()
EVAL_ONLY=none
STAGE=gates
RECORDS=$DATA/m2/records.jsonl.gz
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=${M2FC_WORK:-/tmp/m2-first-cell}
CK=${GMN_CHECKPOINT_DIR:-$WORK/ck}
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200  # fixed in paid runs (not env-overridable)
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs" "$CK"
export SSDM_DATA=$DATA PYTHONUNBUFFERED=1 PYTHONUTF8=1 SSDM_PROGRESS_BEAT=$WORK/logs/beat
# the allocator every job-tvuat variant ran with (jobs/cell_step_profile/cell_step_job.sh, rule.ALLOC_CONF): without it
# fragmentation can OOM a ladder rung the profile fit, and the ladder's OOM catch would silently lower T_max (PR #147)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
MARKER=M2_FIRST_CELL_DONE
VALIDATOR=jobs.m2_first_cell.summarize:valid_result
DECISION=$WORK/out/first_cell_result.json
OVERRIDES=$(env | grep -o '^M2FC_[A-Z0-9_]*' | sort | tr '\n' ' ' || true)
[ -z "$OVERRIDES" ] || say "TEST OVERRIDES SET (never in a paid run): $OVERRIDES"
finish() {
  local rc=${RC:-$?}
  trap - EXIT
  kill "${WD_PID:-}" 2> /dev/null || true
  rm -f "$OUT/$MARKER"
  # Terminal export and result validation share 60s + 30s KILL grace from the
  # existing seven-minute reserve. Export failure still reaches finalize (exit 6).
  export -f export_phase sample say
  export CODE OUT WORK T0 META RESULT GATES EXPORT_FAIL MARKER VALIDATOR DECISION
  export FINISH_RC=$rc
  timeout --kill-after=30s 60s bash -c '
    export_phase final
    python "$CODE/jobs/common/size_guard.py" --out "$OUT" --evict-dir "$WORK/evicted" || true  # results always upload
    exec python "$CODE/jobs/common/finalize.py" "$RESULT" "$FINISH_RC" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
      "$DECISION" "$OUT" "$MARKER" "$VALIDATOR"
  '
  local final=$?
  [ "$final" -eq 0 ] || rm -f "$OUT/$MARKER"
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
num() { case "$1" in *[!0-9.]*|"") return 1 ;; esac; }
hb
watchdog & WD_PID=$!
trap finish EXIT
if [ "$MODE" != base ]; then
  case "$MODE" in ""|*[!a-z0-9_]*) fail 5 "first-cell mode '$MODE' is not [a-z0-9_]+" ;; esac
  MODE_ENV=$(python -I - "$MODE" "$CELL" "$PARAMS" "${M2FC_CELL:-}" 2>&1 <<'EOF'
import json, re, sys
mode, cell, params, override = sys.argv[1:5]
def load(f, what):
    try:
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError:
        sys.exit(f"mode {mode}: no {what} {f}")
    except ValueError as e:
        sys.exit(f"mode {mode}: {what} {f} is not JSON ({e})")
p, c = load(params, "job params"), load(cell, "cell config")
if p.get("mode", mode) != mode:
    sys.exit(f"mode {mode}: {params} says mode {p['mode']!r}")
if not override and p.get("cell_config", cell) != cell:
    sys.exit(f"mode {mode}: {params} cell_config {p['cell_config']!r} != {cell}")
td = p.get("train_dir", "m2-150k")
if not isinstance(td, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*", td) or ".." in td.split("/"):
    sys.exit(f"mode {mode}: train_dir {td!r} must be a relative path under $DATA")
ev = p.get("eval_only", "none")
if ev not in ("none", "fulldev"):
    sys.exit(f"mode {mode}: eval_only {ev!r} must be none or fulldev")
if ev != "none":
    sys.path.insert(0, ".")
    from jobs.m2_first_cell.checkpoint_loader import CheckpointError, checkpoint_block
    try:
        checkpoint_block(p)
    except CheckpointError as e:
        sys.exit(f"mode {mode}: {e}")
st = p.get("stage", "gates")
if st not in ("gates", "screen") or (st == "screen" and ev != "none"):
    sys.exit(f"mode {mode}: stage {st!r} must be gates or screen (screen = a training mode's evals, not eval_only)")
print(td, 1 if c.get("teacher") else 0, ev, st)
EOF
  ) || fail 5 "$MODE_ENV"
  read -r TRAIN_SUB TEACHER EVAL_ONLY STAGE <<< "$MODE_ENV"
  TRAIN_DIR=$DATA/$TRAIN_SUB
  [ "$TEACHER" = 1 ] && TEACHER_ARGS=(--teacher-dir "$DATA/teacher")
  say "mode $MODE: cell $CELL, params $PARAMS, train dir $TRAIN_DIR, KD teacher ${TEACHER_ARGS[*]:-none}, eval-only $EVAL_ONLY"
  [ "$EVAL_ONLY" = fulldev ] && VALIDATOR=jobs.m2_first_cell.fulldev_eval:valid_result DECISION=$WORK/out/fulldev_eval_result.json
  [ "$STAGE" = screen ] && VALIDATOR=jobs.m2_first_cell.screen:valid_result DECISION=$WORK/out/screen.json
fi
[ -n "${TOK_S:-}" ] || fail 4 "TOK_S (a MEASURED prefill tok/s) is not set"
num "$TOK_S" || fail 4 "TOK_S=$TOK_S is not a number"
[ -n "${CALL_S:-}" ] || fail 4 "CALL_S (a MEASURED per-call latency, s) is not set"
num "$CALL_S" || fail 4 "CALL_S=$CALL_S is not a number"

# ------------------------------------------------------------------ setup: identities + eval preflights (exit 5 / 10 / 15)
say "setup"
cp "$CONTEXT_MANIFEST" "$WORK/out/" || fail 5 "no context manifest"
nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
# WKV = the fla name if EITHER rwkv7_wkv (scoring) or train_rwkv7_wkv (training steps only) is fla, else rwkv7_wkv
WKV=$(python -I -c 'import json,sys; c=json.load(open(sys.argv[1])); s=c.get("rwkv7_wkv") or "chunked"; t=c.get("train_rwkv7_wkv") or s; print(t if t.startswith("fla") else s)' "$CELL")   || fail 5 "cell config rwkv7_wkv"
case "$WKV" in
  fla|fla_safe_gate)  # fla-core is not in the rwkv7 image: the exact 0.5.2 wheel by hash, no deps (perf_step_gate v2)
    say "rwkv7_wkv=$WKV: install fla-core 0.5.2 by hash"
    pip install --no-cache-dir --no-deps --require-hashes -r "$CODE/jobs/fla_rwkv7/requirements-fla.txt"       > "$WORK/logs/pip-fla.log" 2>&1 || fail 5 "fla-core install: $(tail -1 "$WORK/logs/pip-fla.log")"
    python - > "$WORK/out/fla-versions.json" 2> "$WORK/logs/fla-versions.log" <<'EOF' || fail 5 "fla version pins: $(tail -1 "$WORK/logs/fla-versions.log")"
import json, torch, triton, fla
assert torch.__version__.split("+")[0] == "2.13.0", torch.__version__
assert triton.__version__ == "3.7.1" and fla.__version__ == "0.5.2", (triton.__version__, fla.__version__)
from fla.ops.rwkv7 import chunk_rwkv7  # noqa: F401
print(json.dumps({"torch": torch.__version__, "triton": triton.__version__, "fla": fla.__version__}))
EOF
    ;;
esac
CELL_PATH=$CELL python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins / WKV selection"
import json, torch, transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
assert torch.cuda.is_available()
import decision_index  # noqa: F401
import third_party.rwkv7_hf as V
import os
from heads.rwkv7_wkv import FLA_NAMES, GPU_VERIFIED, GPU_VERIFIED_NAMES, LEAN, VENDORED
assert all(V.vendored_ok().values()), V.vendored_ok()
c = json.load(open(os.environ["CELL_PATH"]))
wkv = c.get("rwkv7_wkv") or "chunked"
train_wkv = c.get("train_rwkv7_wkv") or wkv  # training steps only (scoring uses rwkv7_wkv)
for w in (wkv, train_wkv):
    assert w in VENDORED + LEAN + FLA_NAMES, w
    assert w not in FLA_NAMES or (GPU_VERIFIED and w in GPU_VERIFIED_NAMES), f"wkv={w} is not GPU-verified"
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "transformers": transformers.__version__, "vendored_rwkv7": V.REVISION,
                  "wkv": wkv, "fla_gpu_verified": (wkv in GPU_VERIFIED_NAMES) if wkv in FLA_NAMES else None,
                  **({"train_wkv": train_wkv} if train_wkv != wkv else {})}))
EOF
python -I "$CODE/jobs/common/fla_guard.py" --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
case "$WKV" in fla|fla_safe_gate)  # the guard must have checked the installed fla (not the "not installed" no-op)
  python -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("fla_installed") else 1)' "$WORK/out/fla_guard.json"     || fail 5 "fla #640 guard did not see fla-core installed" ;;
esac
( cd "$DATA" && sha256sum -c "$CODE/jobs/rwkv7/records.sha256" ) > "$WORK/out/records-verify.txt" 2>&1 || fail 10 "PR #16 records sha256"
python -X utf8 -m train.train_set verify --config "$CELL" --train-dir "$TRAIN_DIR" --out "$WORK/out/train-set-verify.json" \
  > "$WORK/logs/train-set.log" 2>&1 || fail 10 "150k train set != pointer"
if [ ${#TEACHER_ARGS[@]} -gt 0 ]; then
  # KD arm (the cell config's "teacher" block): the context pins the teacher files; verify semantic identity before
  # loading weights. run_cell and m2_gates also enforce the pinned teacher identity, training draw and content mapping.
  python -X utf8 -c 'import json,sys; from jobs.q9af_cell.hooks import kd_arm; from train.teacher import load_teacher; c=json.load(open(sys.argv[1])); assert kd_arm(c) is not None; print(json.dumps(load_teacher(sys.argv[2]).summary()))' \
    "$CELL" "$DATA/teacher" > "$WORK/out/teacher-verify.json" 2>&1 || fail 10 "Clef KD teacher identity / contamination"
fi
python -X utf8 -m train.cohort verify > "$WORK/out/cohort-verify.json" 2>&1 || fail 10 "cohort does not rebuild to its committed sha256"
CAP_SETUP=$(cap setup) || fail 5 "job_params"
SNAP=$(timeout --kill-after=30s ${CAP_SETUP}m python -c 'from huggingface_hub import snapshot_download as s; print(s("RWKV/RWKV7-G1k-13.3B-20260930", revision="ce3630b854cedbd673569f4600c6cf744bca9dba"))'); r=$?
case $r in 0) ;; 124|137) fail 7 "download timeout" ;; *) fail 10 "download" ;; esac
( cd "$SNAP" && sha256sum -c "$CODE/jobs/rwkv7/snapshot.sha256" ) > "$WORK/out/snapshot-verify.txt" 2>&1 || fail 10 "snapshot sha256"
python -c "import sys, third_party.rwkv7_hf as V; V.verify_snapshot_code(sys.argv[1])" "$SNAP" || fail 10 "vendored code != snapshot code"
( cd "$DATA" && sha256sum -c "$CODE/jobs/clef_baseline/data.sha256" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
python -X utf8 -m jobs.common.dev_subset rows --rows "$DATA/slices/dev-rows.jsonl.gz" --out "$WORK/index-rows.jsonl.gz" \
  --subset jobs/common/dev_subset_m1.json > "$WORK/out/index-subset.json" || fail 10 "DEV subset rows"
REFS=()
for NAME in rwkv7_stock_tkfaj clef_flash; do  # optional comparison files, used only if pinned AND present
  PIN=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['reference_results_sha256'].get(sys.argv[2]) or '')" "$PARAMS" "$NAME")
  F=$DATA/reference/$NAME.results.jsonl
  if [ -n "$PIN" ]; then
    [ -f "$F" ] && [ "$(sha256sum "$F" | cut -d' ' -f1)" = "$PIN" ] || fail 10 "reference $NAME missing or sha256 != pin"
    REFS+=(--reference "$NAME=$F")
  fi
done
python -c "import train.m2_gates, train.heldout, train.train_set, jobs.m2_parent_arm.validate, jobs.m2_first_cell.summarize, jobs.m2_first_cell.readout_diag, jobs.m2_first_cell.fulldev_eval, jobs.m2_first_cell.checkpoint_loader" || fail 5 "drivers not importable"
if [ "$EVAL_ONLY" = fulldev ]; then  # full-DEV scoring priced from the committed row lengths x D1's measured child rate
  python -X utf8 -m jobs.m2_first_cell.fulldev_eval estimate --params "$PARAMS" > "$WORK/out/preflight-fulldev.json" \
    || fail 15 "fulldev projected past its cap at the measured D1 child rate"
else
# params "rows" (base / training modes): "subset" (default) = the child's Index stage on all 2,008 subset rows; "proxy600"
# = a SCREEN on the pre-declared 644-row proxy (jobs/common/dev_proxy600.json, ids sha pinned in jobs.common.dev_proxy;
# the parent arm stays verified on the full subset and is paired on the proxy rows). Winners get the full subset.
IXROWS=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('rows', 'subset'))" "$PARAMS") || fail 5 "job_params rows"
INDEX_PROXY=()
case $IXROWS in
  subset) ;;
  proxy600) python -X utf8 -m jobs.common.dev_proxy check --proxy proxy600 > "$WORK/out/index-proxy.json" || fail 10 "index proxy600 identity"
            INDEX_PROXY=(--index-proxy proxy600) ;;
  *) fail 5 "job_params rows $IXROWS: subset or proxy600" ;;
esac
FIX_P=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['evals_fixed_minutes']['parent'])" "$PARAMS") || fail 5 "job_params"
FIX_C=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['evals_fixed_minutes']['child'])" "$PARAMS") || fail 5 "job_params"
# every eval phase is priced HERE from the measured rates, before the paid training phase
EVAL_PRICES=("g12 parent parent_g12 $FIX_P" "index parent parent_index $FIX_P" "g12,index child evals $FIX_C")
if [ "$STAGE" = screen ]; then  # params "stage": "screen": the child evals are ONE proxy600 Index pass (no G12 / fit / card)
  EVAL_PRICES=("${EVAL_PRICES[@]:0:2}")
  python -X utf8 -m jobs.common.dev_proxy check --proxy proxy600 > "$WORK/out/index-proxy.json" || fail 10 "index proxy600 identity"
  python -X utf8 -m jobs.m2_first_cell.screen estimate --params "$PARAMS" > "$WORK/out/preflight-screen.json" \
    || fail 15 "screen projected past its cap at D1's measured proxy rate"
  # the reference cell's child predictions (sha-pinned artifact chain), checked to cover every proxy row BEFORE training
  python -X utf8 -m jobs.m2_first_cell.screen reference --params "$PARAMS" --cell-config "$CELL" --data "$DATA" --work "$WORK/reference" \
    --out "$WORK/out/screen-reference.json" > "$WORK/logs/screen-reference.log" 2>&1 \
    || fail 10 "screen reference: $(tail -1 "$WORK/logs/screen-reference.log")"
fi
for P in "${EVAL_PRICES[@]}"; do
  set -- $P
  python -X utf8 -m jobs.m2_parent_arm.score_parent estimate --tok-s "$TOK_S" --call-s "$CALL_S" --stages "$1" \
    --mode "$2" --cap-min "$(cap "$3")" --fixed-min "$4" > "$WORK/out/preflight-$3.json" || fail 15 "$3 projected past its cap at TOK_S=$TOK_S CALL_S=$CALL_S"
done
fi
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ guard readiness (exit 30 / 31 / 32): before ANY training
say "phase guard"
timeout --kill-after=30s $(cap guard)m python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir "$DATA/suite-0.2" --kit-dir "$KIT_DIR" \
  --out-dir "$WORK/out/guard" > "$WORK/logs/guard.log" 2>&1; r=$?
gate "guard" $r; export_phase guard
case $r in 0) ;; 124|137) fail 7 "guard timeout" ;; 30|31|32) fail $r "guard not ready (see out/guard/readiness.json)" ;; *) fail 5 "guard rc=$r" ;; esac

if [ "$EVAL_ONLY" = fulldev ]; then  # ---------------------------- eval-only: a SAVED checkpoint on the FULL Index DEV
  say "phase checkpoint (shared loader: jobs.m2_first_cell.checkpoint_loader)"
  timeout --kill-after=30s $(cap checkpoint)m python -X utf8 -m jobs.m2_first_cell.checkpoint_loader install --params "$PARAMS" \
    --cell-src "$DATA/checkpoint/cell.json" --cell-config "$CELL" --ck-dir "$CK/cell" --work "$WORK/art" \
    --cell-out "$WORK/out/cell.json" --out "$WORK/out/checkpoint.json" > "$WORK/logs/checkpoint.log" 2>&1; r=$?
  gate "checkpoint" $r; export_phase checkpoint
  case $r in 0) ;; 124|137) fail 7 "checkpoint timeout" ;; 10) fail 10 "checkpoint identity: $(tail -1 "$WORK/logs/checkpoint.log")" ;;
    *) fail 5 "checkpoint rc=$r: $(tail -1 "$WORK/logs/checkpoint.log")" ;; esac
  CKPT=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['path'])" "$WORK/out/checkpoint.json") || fail 5 "checkpoint.json"
  read -r MAXIN FD_PARENT ROWS_SEL <<< "$(python -c "import json,sys; p=json.load(open(sys.argv[1])); print(int(p['max_input_tokens']), 1 if p.get('fulldev_parent_arm') else 0, p.get('rows', 'full'))" "$PARAMS")" \
    || fail 5 "job_params max_input_tokens / fulldev_parent_arm / rows"
  FD_ARMS=(child); [ "$FD_PARENT" = 1 ] && FD_ARMS+=(parent)
  for ARMNAME in "${FD_ARMS[@]}"; do
    PH=fulldev; [ "$ARMNAME" = parent ] && PH=parent_full
    if [ "$ARMNAME" = child ]; then
      ARGS=(--cell "$WORK/out/cell.json" --ckpt-dir "$CK/cell" --ckpt "$CKPT" --records "$RECORDS" --train-dir "$TRAIN_DIR"
            "${TEACHER_ARGS[@]}" --guard-readiness "$WORK/out/guard/readiness.json")
    else
      ARGS=(--cell-config "$CELL")
    fi
    say "phase $PH ($ARMNAME arm, DEV rows: $ROWS_SEL, max_input_tokens $MAXIN; resumes from $CK/fulldev)"
    python -X utf8 -m jobs.common.rate_guard run-weighted --progress "$WORK/logs/progress-$PH.jsonl" --cap-min "$(cap $PH)" \
      --report "$WORK/out/rate-$PH.json" -- \
      timeout --kill-after=30s $(cap $PH)m python -X utf8 -m jobs.m2_first_cell.fulldev_eval score --arm "$ARMNAME" "${ARGS[@]}" \
      --model-dir "$SNAP" --rows "$DATA/slices/dev-rows.jsonl.gz" --max-input-tokens "$MAXIN" --rows-select "$ROWS_SEL" --resume-dir "$CK/fulldev" \
      --out-dir "$WORK/out/fulldev" --progress "$WORK/logs/progress-$PH.jsonl" > "$WORK/logs/$PH.log" 2>&1; r=$?
    gate "$PH" $r; export_phase "$PH"
    case $r in 0) ;; 124|137) fail 7 "$PH timeout" ;; 15) fail 15 "$PH: live rate past its cap" ;;
      10) fail 10 "$PH identity (rows / checkpoint lineage / cohort)" ;; *) fail 9 "$PH rc=$r" ;; esac
  done
  say "phase report"
  timeout --kill-after=30s $(cap report)m python -X utf8 -m jobs.m2_first_cell.fulldev_eval report --out-dir "$WORK/out/fulldev" \
    --params "$PARAMS" --elapsed-min "$(( ($(date +%s) - T0) / 60 ))" --out "$DECISION" > "$WORK/logs/report.log" 2>&1; r=$?
  gate "report" $r
  [ $r -eq 0 ] || fail 9 "fulldev_eval result missing or invalid (rc=$r)"
  sample "$DECISION" "fulldev-eval"
  RC=0
  exit 0
fi

# ------------------------------------------------------------------ held-out D1-B over the WHOLE training set (exit 33)
say "phase heldout (hits must equal the pinned exclusions)"
timeout --kill-after=30s $(cap heldout)m python -X utf8 -m train.train_set heldout --config "$CELL" --train-dir "$TRAIN_DIR" \
  --out "$WORK/out/heldout.json" > "$WORK/logs/heldout.log" 2>&1; r=$?
gate "heldout" $r; export_phase heldout
case $r in 0) ;; 124|137) fail 7 "heldout timeout" ;; 3) fail 33 "held-out D1-B fragment or eval attack template (train.attack_guard) outside the pinned exclusions" ;; *) fail 5 "heldout rc=$r" ;; esac

# ------------------------------------------------------------------ parent arm (stock readout) BEFORE training (review fix 1)
# Its live weighted rate guard measures this box's scoring speed before any training spend; then the child evals are
# re-priced from that measured speed (jobs.m2_first_cell.reprice, exit 15). Cached in $CK across restarts.
ARM=$CK/parent_arm
# A RESUBMITTED job (empty checkpoint slot) seeds the cache from a pinned earlier export shipped as data/parent_arm_seed/
# (job_params parent_arm_seed; LESSONS L-34): installed atomically (temp dir, checked, renamed), then verified BEFORE
# any training (seed_verify: full units identity via read_arm_cache, guard, the parent's revision on this box and a
# short re-score probe whose speed feeds the re-price). A restart's own cache is left as is (a seeded one is
# re-verified using provenance inside the atomic cache install); nothing shipped = score the parent.
python -X utf8 -m jobs.m2_first_cell.arm_seed seed --src "$DATA/parent_arm_seed" --dst "$ARM" --params "$PARAMS" \
  --cell "$CELL" --out "$WORK/out/arm-seed.json" > "$WORK/logs/arm-seed.log" 2>&1; r=$?
case $r in 0) ;; 10) fail 10 "parent arm seed refused: $(tail -1 "$WORK/logs/arm-seed.log")" ;; *) fail 5 "arm seed rc=$r" ;; esac
mkdir -p "$ARM"
ARM_SRC=restart
PROBE=()
if python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('verify_seed') else 1)" "$WORK/out/arm-seed.json"; then
  ARM_SRC="seeded from data/parent_arm_seed"
  SEED_SHA=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['parent_arm_seed']['arm_sha256'])" "$PARAMS") \
    || fail 5 "job_params parent_arm_seed"
  PROBE_N=$(python -c "import json,sys; print(int(json.load(open(sys.argv[1]))['parent_arm_seed']['probe_records']))" "$PARAMS") \
    || fail 5 "job_params parent_arm_seed"
  say "phase seed_verify (seeded parent arm: units identity, guard, revision + $PROBE_N-record re-score on this box)"
  timeout --kill-after=30s $(cap parent_g12)m python -X utf8 -m jobs.m2_parent_arm.score_parent --verify-cache --parent rwkv7 \
    --model-dir "$SNAP" --cell-config "$CELL" --records "$RECORDS" --guard-readiness "$WORK/out/guard/readiness.json" \
    --index-rows "$WORK/index-rows.jsonl.gz" --out-dir "$ARM" --arm-sha256 "$SEED_SHA" --probe-n "$PROBE_N" \
    --tok-s "$TOK_S" --call-s "$CALL_S" --verify-out "$WORK/out/seed-verify.json" > "$WORK/logs/seed-verify.log" 2>&1; r=$?
  mkdir -p "$WORK/out/parent_arm"; cp "$ARM"/*.json "$WORK/out/parent_arm/" 2>/dev/null || true  # comparator units
  case $r in 0) ;; 124|137) fail 7 "seed_verify timeout" ;;
    10) fail 10 "seeded parent arm failed verification: $(tail -1 "$WORK/logs/seed-verify.log")" ;;
    *) fail 5 "seed_verify rc=$r" ;; esac
  PROBE=(--probe "$WORK/out/seed-verify.json")
fi
for PSTAGE in g12 index; do  # not STAGE: that is the params stage (gates|screen), read below
  if python -c "import json,sys,hashlib,pathlib; d=pathlib.Path(sys.argv[1]); e=json.load(open(d/'arm.json'))['stages'][sys.argv[2]]; sys.exit(0 if hashlib.sha256((d/e['file']).read_bytes()).hexdigest()==e['sha256'] else 1)" "$ARM" "$PSTAGE" 2>/dev/null; then
    say "parent $PSTAGE: cached ($ARM_SRC)"; gate "parent_$PSTAGE" 0; continue
  fi
  say "phase parent_$PSTAGE"
  python -X utf8 -m jobs.common.rate_guard run-weighted --progress "$WORK/logs/parent-$PSTAGE.progress.jsonl" \
    --cap-min "$(cap parent_$PSTAGE)" --report "$ARM/rate-parent-$PSTAGE.json" -- \
    timeout --kill-after=30s $(cap parent_$PSTAGE)m python -X utf8 -m jobs.m2_parent_arm.score_parent --parent rwkv7 --model-dir "$SNAP" --cell-config "$CELL" \
    --records "$RECORDS" --guard-readiness "$WORK/out/guard/readiness.json" --index-rows "$WORK/index-rows.jsonl.gz" \
    --out-dir "$ARM" --stage "$PSTAGE" --progress "$WORK/logs/parent-$PSTAGE.progress.jsonl" \
    > "$WORK/logs/parent-$PSTAGE.log" 2>&1; r=$?
  gate "parent_$PSTAGE" $r
  mkdir -p "$WORK/out/parent_arm"; cp "$ARM"/*.json "$WORK/out/parent_arm/" 2>/dev/null || true
  export_phase "parent_$PSTAGE"
  case $r in 0) ;; 124|137) fail 7 "parent $PSTAGE timeout" ;; 15) fail 15 "parent $PSTAGE: live rate past its cap" ;;
    10) fail 10 "parent $PSTAGE identity" ;; *) fail 9 "parent $PSTAGE rc=$r" ;; esac
done
ARM_SHA=$(sha256sum "$ARM/arm.json" | cut -d' ' -f1)
# Preserve the comparator even when every stage came from the checkpoint cache, including ordinary restarts.
mkdir -p "$WORK/out/parent_arm" && cp "$ARM"/*.json "$WORK/out/parent_arm/" || fail 6 "parent arm export copy"
if [ ${#PROBE[@]} -eq 0 ]; then
  # A cache from this job also needs full identity validation BEFORE training, not just per-file hashes.
  timeout --kill-after=30s $(cap parent_g12)m python -X utf8 -m jobs.m2_parent_arm.score_parent --verify-cache --parent rwkv7 \
    --model-dir "$SNAP" --cell-config "$CELL" --records "$RECORDS" --guard-readiness "$WORK/out/guard/readiness.json" \
    --index-rows "$WORK/index-rows.jsonl.gz" --out-dir "$ARM" --arm-sha256 "$ARM_SHA" \
    --verify-out "$WORK/out/parent-cache-verify.json" > "$WORK/logs/parent-cache-verify.log" 2>&1; r=$?
  case $r in 0) ;; 124|137) fail 7 "parent cache verification timeout" ;;
    10) fail 10 "parent cache failed verification: $(tail -1 "$WORK/logs/parent-cache-verify.log")" ;;
    *) fail 5 "parent cache verification rc=$r" ;; esac
fi
if [ "$STAGE" = screen ]; then
  say "stage screen: child evals priced in setup (D1 measured proxy rate); no re-price"
else
say "re-price the child evals from the measured parent scoring speed"
python -X utf8 -m jobs.m2_first_cell.reprice --arm-dir "$ARM" --params "$PARAMS" --tok-s "$TOK_S" --call-s "$CALL_S" \
  --out "$WORK/out/reprice.json" "${PROBE[@]}" > "$WORK/logs/reprice.log" 2>&1; r=$?
gate "reprice" $r
case $r in 0) ;; 15) fail 15 "child evals re-priced past their cap at the box's measured speed (before training)" ;;
  *) fail 5 "reprice rc=$r" ;; esac
fi

# ------------------------------------------------------------------ cell: smoke -> ladder -> train (resumable; exit 26 = too slow)
say "phase cell"
mkdir -p "$CK/cell"
[ -f "$CK/cell/cell.json" ] && cp "$CK/cell/cell.json" "$WORK/out/cell.json"
timeout --kill-after=30s $(cap cell)m python -m train.run_cell --config "$CELL" --model-dir "$SNAP" --records "$RECORDS" \
  --train-dir "$TRAIN_DIR" "${TEACHER_ARGS[@]}" --out "$WORK/out" --ckpt-dir "$CK/cell" --stage all \
  --guard-readiness "$WORK/out/guard/readiness.json" > "$WORK/logs/cell.log" 2>&1; r=$?
cp "$WORK/out/cell.json" "$CK/cell/cell.json" 2>/dev/null || true
gate "cell" $r; export_phase cell
case $r in 0) ;; 124|137) fail 7 "cell timeout" ;; 21|22|23|24|25|26|27|28|29|35|36) fail $r "cell stop" ;; *) fail 20 "cell unexpected rc=$r" ;; esac

# ------------------------------------------------------------------ stage screen: ONE proxy600 Index pass, then screen.json
if [ "$STAGE" = screen ]; then
  say "phase screen (child untempered arm on the 644 proxy600 rows; paired vs the parent arm and the reference cell)"
  python -X utf8 -m jobs.common.rate_guard run-weighted --progress "$WORK/logs/progress-screen.jsonl" --cap-min "$(cap screen)" \
    --report "$WORK/out/rate-screen.json" -- \
    timeout --kill-after=30s $(cap screen)m python -X utf8 -m jobs.m2_first_cell.screen score --cell "$WORK/out/cell.json" \
    --ckpt-dir "$CK/cell" --model-dir "$SNAP" --records "$RECORDS" --train-dir "$TRAIN_DIR" "${TEACHER_ARGS[@]}" \
    --guard-readiness "$WORK/out/guard/readiness.json" --index-rows "$WORK/index-rows.jsonl.gz" --resume-dir "$CK/screen" \
    --parent-arm "$ARM" --parent-arm-sha256 "$ARM_SHA" --reference "$WORK/out/screen-reference.json" \
    --out-dir "$WORK/out/screen" --progress "$WORK/logs/progress-screen.jsonl" > "$WORK/logs/screen.log" 2>&1; r=$?
  gate "screen" $r; export_phase screen
  case $r in 0) ;; 124|137) fail 7 "screen timeout" ;; 15) fail 15 "screen: live rate past the cap" ;;
    10) fail 10 "screen identity (proxy / parent arm / checkpoint lineage / plan check)" ;; *) fail 9 "screen rc=$r" ;; esac
  say "phase report (screen.json)"
  timeout --kill-after=30s $(cap report)m python -X utf8 -m jobs.m2_first_cell.screen report --params "$PARAMS" \
    --index-rows "$WORK/index-rows.jsonl.gz" --out-dir "$WORK/out/screen" --parent-arm "$ARM" --parent-arm-sha256 "$ARM_SHA" \
    --reference "$WORK/out/screen-reference.json" --elapsed-min "$(( ($(date +%s) - T0) / 60 ))" --out "$DECISION" \
    > "$WORK/logs/report.log" 2>&1; r=$?
  gate "report" $r
  [ $r -eq 0 ] || fail 9 "screen result missing or invalid (rc=$r): $(tail -1 "$WORK/logs/report.log")"
else
# ------------------------------------------------------------------ child evals: temperature fit, dev card, G12 + Index DEV subset
say "phase evals (calibration fit; dev card; G12 dev + steer_dev; Index DEV subset + readout diagnostics)"
python -X utf8 -m jobs.common.rate_guard run-weighted --progress "$WORK/logs/progress.jsonl" --cap-min "$(cap evals)" \
  --report "$WORK/out/rate-evals.json" -- \
  timeout --kill-after=30s $(cap evals)m python -X utf8 -m train.m2_gates --cell "$WORK/out/cell.json" --ckpt-dir "$CK/cell" --model-dir "$SNAP" \
  --records "$RECORDS" --train-dir "$TRAIN_DIR" "${TEACHER_ARGS[@]}" --guard-readiness "$WORK/out/guard/readiness.json" \
  --parent-arm "$ARM" --parent-arm-sha256 "$ARM_SHA" --anchor-arm "$ARM" --anchor-arm-sha256 "$ARM_SHA" \
  --index-rows "$WORK/index-rows.jsonl.gz" "${INDEX_PROXY[@]}" \
  --out "$WORK/out/report.json" --stage g12+index --progress-dir "$WORK/logs" > "$WORK/logs/evals.log" 2>&1; r=$?
gate "evals" $r; export_phase evals
# Trained + measured (the report is written; the checkpoint and every eval export follow), m2_gates (PR #102, Codex
# #36): 0 promotable | 40 G12 rejected | 43 margins pending | 44 partial (this cell runs G12 + the index measurement,
# not G13, so an accepting run is 44) | 45 underpowered. 43 / 44 / 45 are NOT promotable; summarize records why.
# The anchor (lineage comparator) is the in-job parent arm itself, pinned by the same sha256.
case $r in 0|40|43|44|45) ;; 124|137) fail 7 "evals timeout" ;; 15) fail 15 "evals: live rate past the cap" ;;
  10) fail 10 "evals identity (cohort / parent arm / checkpoint lineage)" ;; *) fail 9 "evals rc=$r" ;; esac

# ------------------------------------------------------------------ report: result + checkpoint export
say "phase report"
timeout --kill-after=30s $(cap report)m python -X utf8 -m jobs.m2_first_cell.summarize --cell "$WORK/out/cell.json" \
  --report "$WORK/out/report.json" --parent-arm "$ARM" --index-rows "$WORK/index-rows.jsonl.gz" --params "$PARAMS" \
  --elapsed-min "$(( ($(date +%s) - T0) / 60 ))" --out "$WORK/out/first_cell_result.json" "${REFS[@]}" \
  > "$WORK/logs/report.log" 2>&1; r=$?
gate "report" $r
[ $r -eq 0 ] || fail 9 "first-cell result missing or invalid (rc=$r)"
fi
LATEST=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['path'])" "$CK/cell/latest.json") || fail 9 "no checkpoint"
# INFERENCE export only (train.ckpt_export: trainable LoRA + head as bf16 safetensors + JSON sidecar + sha256; r256
# ~2.3 GB, r64 ~0.6 GB). The full .pt (Adam + scheduler + RNG; ~14 GB at r256) stays in $CK: job-igcu4 copied it to
# /output, went over the store's 5 GiB single-PUT cap, and lost the whole result.
mkdir -p "$OUT/ckpt"
# best_ckpt cells (train/trainer.py, D10 anti-collapse): latest.json's "best" pointer names the SCORED checkpoint; it is
# exported too (ckpt_export --best) and the export's latest.json keeps the pointer (naming the exported file)
BEST=$(python -c "import json,sys; print((json.load(open(sys.argv[1])).get('best') or {}).get('path') or '')" "$CK/cell/latest.json")
BEST_ARGS=()
[ -n "$BEST" ] && [ "$BEST" != "$LATEST" ] && BEST_ARGS=(--best "$CK/cell/$BEST")
python -X utf8 -m train.ckpt_export inference --ckpt "$CK/cell/$LATEST" "${BEST_ARGS[@]}" --out-dir "$OUT/ckpt" > "$WORK/logs/ckpt-export.log" 2>&1 \
  || { EXPORT_FAIL=1; say "checkpoint export failed: $(tail -1 "$WORK/logs/ckpt-export.log")"; }
python -X utf8 jobs/common/size_guard.py --out "$OUT" --evict-dir "$WORK/evicted" || true
sample "$DECISION" "first-cell"
RC=0
exit 0
