#!/usr/bin/env bash
# q9af_v3: Q9-AF v3 = Qwen3.5-9B -> fully attention-free (all 8 attention layers -> GDN) with the 2025-26 recipe
# upgrades (jobs/q9af_v3/README.md: plan + the pre-registered gate; JOB_SPEC.md: phases, caps, pricing). Two SF Compute
# batch jobs (h100, one attempt each) chained by a sha256-pinned artifact, gated by a cheap first job:
#   mode S (job_params_S.json, the cheap first job): setup | D small clean mix | P per-layer sensitivity table (each
#          attention layer ALONE converted to a widened GDN, recall-heavy KD, MQAR + NIAH + KL; 2 base-width controls)
#          | R jobs.q9af_v3.rule sens (GO / PARTIAL / NO_GO)
#   mode A (job_params_A.json): setup | D mix (v2 clean, full-eval-v1) | G teacher generation (incl. retrieval / tool /
#          knowledge instructions) + 32k block split + on-policy prompts | I_stock proxy600 (v3w) | per stage k = 1..4:
#          P stage k (convert.v3 stageA --until-stage k: stage 1 + recall probe + order on k = 1) -> I proxy600 of the
#          converted-so-far model -> the stage gate (rule gate: recall + proxy600; STOP ends the conversion) | X stage-A
#          export (/output/stageA/*, the artifact job B pins; a STOPPED hybrid only as gdn_salvage.pt) | R record
#   mode B (job_params_B.json): setup (stage-A pin present, v3w readout identity) | F fetch stage A (+ v2's export when
#          pinned) | P 32k long-context + on-policy (convert.v3 stageB) | X final export (gdn_converted.pt) |
#          L proxy600 arms through the generic v3w readout: stock, v3, v2 (if pinned) | S paired compares |
#          R the gate (jobs.q9af_v3.rule decide: PROMOTE iff CI lo of v3 - reference > 0)
# Exit codes (also in $GMN_RESULT_PATH): 0 measured (+ decided) and exports verified   2 pipeline crash   3 throughput
#   stop   4 memory stop   5 setup failed   6 export failed verification   7 phase timeout   8 9B weights unavailable
#   9 no valid decision / stage-A record   10 DEV-row / proxy / readout / artifact identity   11 mixture build failed
#   12 Index kit failed   13 scoring failed   15 Index readout too slow by MEASUREMENT   1 finalize failed
# /output/Q9AF3_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
# Q9AF3_DRYRUN=1 (CPU, tests/test_q9af_v3.py): this very script with only the fetch / GPU / kit steps replaced by
# tests/q9af_v3_stub.py; setup checks, the preflight, the artifact pins, the rule and finalize run for real.
set -Euo pipefail
MAIN_PID=$$
CODE=/opt/ssdm/m1
MANIFEST=/opt/ssdm/CONTEXT_MANIFEST.json
DATA=/data
WORK=/tmp/q9af3
DRY=""
if [ -n "${Q9AF3_DRYRUN:-}" ]; then
  if [ -d /opt/ssdm/m1 ] || [ -z "${Q9AF3_DRYRUN_ROOT:-}" ]; then echo "Q9AF3_DRYRUN refused here"; exit 5; fi
  CODE=$(cd "$(dirname "$0")/../.." && pwd)
  MANIFEST=$Q9AF3_DRYRUN_ROOT/CONTEXT_MANIFEST.json
  DATA=$Q9AF3_DRYRUN_ROOT/data
  WORK=$Q9AF3_DRYRUN_ROOT/work
  DRY="python -X utf8 -m tests.q9af_v3_stub"
fi
MODE=${1:-}
CFG=configs/q9af_v3_h100.json
PROXY=jobs/common/dev_proxy600.json
V3W_VOLUME=jobs/q9af_v3/readout_volume_v3w_proxy600.json
case "$MODE" in
  A) PARAMS=$CODE/jobs/q9af_v3/job_params_A.json; DATA_SHA=$CODE/jobs/q9af_v3/data_A.sha256; STAGE=q9af_v2
     VALIDATOR=jobs.q9af_v3.rule:valid_stage_a ;;
  S) PARAMS=$CODE/jobs/q9af_v3/job_params_S.json; DATA_SHA=$CODE/jobs/q9af_v3/data_S.sha256; STAGE=q9af_v2
     VALIDATOR=jobs.q9af_v3.rule:valid_sensitivity; CFG=configs/q9af_v3_sens_h100.json ;;
  B) PARAMS=$CODE/jobs/q9af_v3/job_params_B.json; DATA_SHA=$CODE/jobs/q9af_v3/data_B.sha256; STAGE=q9
     VALIDATOR=jobs.q9af_v3.rule:valid_decision ;;
  *) echo "usage: q9af_v3_job.sh S|A|B (got '$MODE')"; exit 5 ;;
esac
[ -z "$DRY" ] || { DATA_SHA=$Q9AF3_DRYRUN_ROOT/data_$MODE.sha256; PARAMS=${Q9AF3_DRYRUN_PARAMS:-$PARAMS}; }
OUT=${GMN_OUTPUT_DIR:-/output}
[ -z "$DRY" ] || OUT=$Q9AF3_DRYRUN_ROOT/output
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
CK=$WORK/ck
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200
[ -z "$DRY" ] || HB_EVERY_S=1
mkdir -p "$OUT/phases" "$WORK/out/index" "$WORK/out/pipeline" "$WORK/logs" "$CK/pipeline" "$CK/gdn" "$WORK/gen" "$WORK/a"
export SSDM_DATA_ROOT=$WORK/data SSDM_DATA=$DATA PYTHONUNBUFFERED=1 PYTHONUTF8=1 Q9AF3_CK=$CK
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
  cp -f "$WORK/gen/results.json" "$WORK/out/gen.json" 2> /dev/null || true
  cp -f "$WORK/gen/split.json" "$WORK/out/split.json" 2> /dev/null || true
  cp -f "$WORK/mix/mix.json" "$WORK/out/mix.json" 2> /dev/null || true
}
salvage() {  # a failed attempt still exports its newest all-8 GDN checkpoint (never a Q9-AF input)
  [ "$EXPORTED" = 1 ] && return 0
  local src=""
  for f in "$CK/gdn/final.pt" "$CK/gdn/stageB_last.pt" "$CK/gdn/stageA.pt" "$CK/gdn/stage4.pt" "$CK/gdn/stage3.pt" \
           "$CK/gdn/stage2.pt" "$CK/gdn/stage1.pt"; do [ -f "$f" ] && { src=$f; break; }; done
  [ -n "$src" ] && [ -n "${Q9B:-}" ] || return 0
  cp -f "$CK/gdn/"*.json "$WORK/out/" 2> /dev/null || true
  python -m jobs.q9af_v3.export --mode salvage --stage "$MODE" --src "$src" --model-dir "$Q9B" --out-dir "$OUT" \
    --meta-out "$WORK/out/export_salvage.json" > "$WORK/logs/export_salvage.log" 2>&1 || say "salvage export failed"
}
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  [ "$rc" = 0 ] || salvage
  sync_pipeline
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$WORK/out/decision.json" "$OUT" Q9AF3_DONE "$VALIDATOR"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup (exit 5 / 8 / 10)
say "setup (mode $MODE: $CFG${DRY:+; CPU DRY RUN})"
cp "$MANIFEST" "$WORK/out/" || fail 5 "no context manifest"
cp "$CFG" "$WORK/out/config.used.json"
cp "$PARAMS" "$WORK/out/params.used.json"
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
# measured-rate preflight: every cap vs its projection, floors, peak, data budgets, schedule, hard cap
python -m jobs.q9af_v3.preflight --params "$PARAMS" --config "$CFG" --out "$WORK/out/preflight.json" > "$WORK/logs/preflight.log" 2>&1; r=$?
gate "setup:preflight" $r
[ $r -eq 0 ] || fail 5 "preflight: a cap, floor, peak or data budget violates the measured-rate margin"
( cd "$DATA" && sha256sum -c "$DATA_SHA" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "shipped data sha256"
if [ "$MODE" = B ]; then
  python -X utf8 -m jobs.q9af_v3.artifact status --params "$PARAMS" > "$WORK/out/artifact-status.json" 2> "$WORK/logs/artifact.err" || fail 5 "artifact status"
  REF=$(python -c "import json,sys; from jobs.q9af_v3.rule import expected_reference as e; print(e(json.load(open(sys.argv[1]))))" "$PARAMS") || fail 5 "reference"
  REF=${REF%$'\r'}
  python -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['stageA']['pinned'] else 1)" "$WORK/out/artifact-status.json" \
    || fail 10 "stage-A artifact is not pinned in job_params_B (python -m jobs.q9af_v3.artifact fill --which stageA)"
  say "pre-registered reference arm: $REF"
fi
if [ "$MODE" != S ]; then
  python -X utf8 -m jobs.common.dev_proxy check --proxy proxy600 > "$WORK/out/index/proxy-identity.json" 2> "$WORK/logs/proxy.err" || fail 10 "proxy600 identity"
  python -X utf8 -m jobs.common.dev_subset rows --rows "$DATA/slices/dev-rows.jsonl.gz" --out "$WORK/data/proxy-rows.jsonl.gz" \
    --subset "$PROXY" > "$WORK/out/index/proxy-rows.json" 2> "$WORK/logs/subset.err" || fail 10 "proxy600 rows"
  TOKROOT=$DATA/tokenizers
  V3W_IDENTITY=$(python -c "from jobs.q9af_v3.rule import V3W_IDENTITY as v; print(v)")  # A + B
  V3W_IDENTITY=${V3W_IDENTITY%$'\r'}
  if [ -n "$DRY" ]; then
    $DRY v3wid "$WORK/out/v3w_identity.json" > "$WORK/logs/v3w.log" 2>&1 || fail 10 "dry-run v3w identity"
  else
    python -X utf8 - "$TOKROOT" "$V3W_IDENTITY" > "$WORK/out/v3w_identity.json" 2> "$WORK/logs/v3w.log" <<'EOF' || fail 10 "v3w readout identity"
import json, sys
from decide.v3w_generic import make_v3w
v = make_v3w("qwen3.5-9b", sys.argv[1])
got = v.identity()["sha256"]
print(json.dumps({"identity_sha256": got, "want": sys.argv[2], "codes": len(v.codes), "template": v.template_id}, indent=1))
sys.exit(0 if got == sys.argv[2] and len(v.codes) == 255 else 1)
EOF
  fi
fi
if [ -n "$DRY" ]; then
  $DRY fetch "$SSDM_DATA_ROOT" > "$WORK/logs/fetch.log" 2>&1
else
  timeout "$(cap weights)m" python -I scripts/sf_fetch.py --stage "$STAGE" > "$WORK/logs/fetch.log" 2>&1
fi
[ -f "$SSDM_DATA_ROOT/DONE_Q9B_DIR" ] || { gate "setup:weights" 8; export_phase setup; fail 8 "9B weights unavailable"; }
if [ "$MODE" != B ]; then
  for d in FINEWEB_EDU_DIR ULTRACHAT_DIR FINEMATH_DIR WIKITEXT_DIR; do [ -f "$SSDM_DATA_ROOT/DONE_$d" ] || fail 5 "fetch $d"; done
fi
# shellcheck disable=SC1091
. "$SSDM_DATA_ROOT/paths.env"
export FINEWEB_EDU_DIR ULTRACHAT_DIR FINEMATH_DIR 2> /dev/null || true
Q9B=$(python -c "from convert.model import resolve_model_dir as r; import os; print(r(os.environ['Q9B_DIR']))") || fail 8 "9B snapshot dir"
Q9B=${Q9B%$'\r'}
gate "setup" 0
export_phase setup

run_py() {  # run_py <phase> <cap> <stub cmd...> -- <real cmd...>: the GPU step (stubbed in the dry run)
  local ph=$1 capm=$2 r stub=() real=()
  shift 2
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do stub+=("$1"); shift; done
  shift
  real=("$@")
  if [ -n "$DRY" ]; then
    $DRY "${stub[@]}" >> "$WORK/logs/$ph.log" 2>&1; r=$?
  else
    timeout "${capm}m" "${real[@]}" >> "$WORK/logs/$ph.log" 2>&1; r=$?
  fi
  return $r
}
pipe_rc() {  # pipe_rc <phase> <rc>
  case $2 in 0) ;; 3|4) fail "$2" "$1 stop" ;; 124) fail 7 "$1 timeout" ;; *) fail 2 "$1 crashed rc=$2" ;; esac
}

IDX=$WORK/out/index
V3WOPT=()
[ "$MODE" = S ] || V3WOPT=(--option family=q9af-probe-v3w --option model_dir="$Q9B" --option tokenizers_root="$TOKROOT"
                          --option v3w_identity="$V3W_IDENTITY")
_arm() {  # _arm <phase> <dir> <tok_s> <call_s> <cap> [kit options...]: rate preflight + one guarded kit run (proxy600)
  local phase=$1 dir=$2 toks=$3 calls=$4 capm=$5 r
  shift 5
  python -m jobs.common.rate_guard preflight --tok-s "$toks" --call-s "$calls" --volume "$V3W_VOLUME" --path fork \
    --cap-min "$capm" --out "$IDX/preflight_$dir.json" > "$WORK/logs/preflight_$dir.log" 2>&1; r=$?
  gate "$phase:preflight" $r
  [ $r -eq 0 ] || fail $r "$dir readout projected past its cap"
  [ -e "$IDX/$dir" ] && mv "$IDX/$dir" "$IDX/$dir.stale.$(date +%s)"
  NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$IDX/proxy-rows.json")
  NROWS=${NROWS%$'\r'}
  say "phase $phase ($dir; proxy600: $NROWS rows)"
  if [ -n "$DRY" ]; then
    $DRY kit "$dir" "$IDX/$dir" > "$WORK/logs/index_$dir.log" 2>&1; r=$?
  else
    python -m jobs.common.rate_guard run --results "$IDX/$dir/results.jsonl" --total "$NROWS" --cap-min "$capm" \
      --check-after-min 3 --exclude-startup --report "$IDX/rate_$dir.json" -- \
      python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine "${V3WOPT[@]}" "$@" \
      --rows "$WORK/data/proxy-rows.jsonl.gz" --out "$IDX/$dir" > "$WORK/logs/index_$dir.log" 2>&1; r=$?
  fi
  gate "$phase" $r
  export_phase "$phase"
  case $r in 0) ;; 124) fail 7 "$phase timeout" ;; 15) fail 15 "$phase: measured rate too slow" ;; *) fail 12 "$phase rc=$r" ;; esac
}
if [ -n "$DRY" ]; then S() { $DRY subset "$@"; }
else S() { timeout "$(cap S)m" python -X utf8 -m jobs.common.dev_subset "$@" --subset "$PROXY"; }; fi

if [ "$MODE" != B ]; then
  # ---------------------------------------------------------------- D: the v2 clean mixture (full-eval-v1)
  say "phase D (mixture build: full-eval-v1 decontam + window manifest)"
  run_py D "$(cap D)" mix "$WORK/mix" -- python -X utf8 -m convert.mixdata build --config "$CFG" \
    --dev-rows "$DATA/slices/dev-rows.jsonl.gz" --eval-index-dir "$DATA/doc-overlap/218affe13e367d07" \
    --windows-out "$WORK/mix/windows.json.gz" --out "$WORK/mix"; r=$?
  if [ $r -eq 0 ] && [ ! -s "$WORK/mix/mix.json" ]; then r=11; say "mix.json missing"; fi
  gate "D:mix" $r
  sync_pipeline
  cp -f "$WORK/mix/mix.json" "$OUT/mix.json" 2> /dev/null || true
  export_phase D
  case $r in 0) ;; 124) fail 7 "mixture timeout" ;; *) fail 11 "mixture build rc=$r" ;; esac
fi
if [ "$MODE" = S ]; then
  # ---------------------------------------------------------------- P: the per-layer sensitivity table
  say "phase P (per-layer sensitivity: widened GDN, recall-heavy KD, MQAR + NIAH)"
  run_py P "$(cap P)" sens "$CK" -- python -X utf8 -m convert.v3 sens --config "$CFG" --mix "$WORK/mix" --out "$CK/pipeline"; r=$?
  gate "P:sens" $r
  sync_pipeline
  export_phase P
  pipe_rc P $r
  python -m jobs.q9af_v3.rule sens "$CK/pipeline/results.json" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1; r=$?
  gate "R:sens" $r
  [ $r -eq 0 ] || fail 9 "incomplete sensitivity table: $(head -c 300 "$WORK/out/decision.json" 2> /dev/null)"
  cp "$WORK/out/decision.json" "$OUT/decision.json"
  sample "$WORK/out/decision.json" "decision"
  RC=0
  exit 0
fi
if [ "$MODE" = A ]; then
  # ---------------------------------------------------------------- G: teacher-generated data + split + prompts
  say "phase G (teacher generation: stock Qwen3.5-9B, 32k block split, on-policy prompts)"
  run_py G "$(cap G)" gen "$WORK/gen" -- python -X utf8 -m convert.v3 gen --config "$CFG" --mix "$WORK/mix" \
    --eval-index-dir "$DATA/doc-overlap/218affe13e367d07" --out "$WORK/gen"; r=$?
  gate "G:gen" $r
  sync_pipeline
  export_phase G
  pipe_rc G $r
  # ---------------------------------------------------------------- I_stock: proxy600 of the stock model (v3w)
  read -r TOKS CALLS < <(python -c "import json,sys; m=json.load(open(sys.argv[1]))['measured']; print(m['stock_prefill_tok_s_min'], max(m['pack_call_s'], m['small_call_s']))" "$PARAMS")
  CALLS=${CALLS%$'\r'}
  _arm I_stock v3w_stock "$TOKS" "$CALLS" "$(cap I_stock)" --option converted=none
  # ---------------------------------------------------------------- P_k -> I_k -> gate_k, k = 1..4
  P_END=$(( $(date +%s) + 60 * $(cap P) ))
  ARM_CAP=$(( $(cap I_stages) / 4 ))
  echo "[]" > "$WORK/out/gates.json"
  STOPPED=0
  for k in 1 2 3 4; do
    left=$(( (P_END - $(date +%s)) / 60 ))
    [ "$left" -gt 0 ] || fail 7 "P budget exhausted before stage $k"
    say "phase P stage $k (${left} min of P left)"
    run_py P "$left" stagea "$CK" "$k" -- python -X utf8 -m convert.v3 stageA --config "$CFG" --mix "$WORK/mix" \
      --gen "$WORK/gen" --out "$CK/pipeline" --until-stage "$k"; r=$?
    gate "P:stage$k" $r
    sync_pipeline
    export_phase "P$k"
    pipe_rc "P$k" $r
    CONV=$(python -c "import json,sys;print(','.join(map(str,json.load(open(sys.argv[1]))['stages']['stage'+sys.argv[2]]['converted'])))" "$CK/pipeline/results.json" "$k") || fail 2 "stage $k record"
    CONV=${CONV%$'\r'}
    _arm "I_stage$k" "v3w_stage$k" "$TOKS" "$CALLS" "$ARM_CAP" --option converted="$CONV" --option gdn_ckpt="$CK/gdn/stage$k.pt"
    S compare --candidate "$IDX/v3w_stage$k/results.jsonl" --baseline "$IDX/v3w_stock/results.jsonl" \
      > "$IDX/compare_stage${k}_vs_stock.json" 2>> "$WORK/logs/score.err" || fail 13 "compare stage $k"
    python -m jobs.q9af_v3.rule gate "$CK/pipeline/results.json" "$k" "$IDX/compare_stage${k}_vs_stock.json" \
      "$WORK/out/gate_stage$k.json" > "$WORK/logs/gate_stage$k.log" 2>&1; r=$?
    python -c "import json,sys; g=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); a.append(g); json.dump(a, open(sys.argv[2],'w'), indent=1)" \
      "$WORK/out/gate_stage$k.json" "$WORK/out/gates.json" || fail 9 "gate record"
    gate "G:stage$k" $r
    export_phase "gate$k"
    if [ $r -ne 0 ]; then STOPPED=$k; say "STOP rule after stage $k: $(head -c 400 "$WORK/out/gate_stage$k.json")"; break; fi
  done
  # ---------------------------------------------------------------- X: the stage-A artifact (job B pins it)
  mkdir -p "$OUT/stageA"
  if [ "$STOPPED" != 0 ]; then
    say "phase X (STOPPED at stage $STOPPED: hybrid salvage export only, nothing to chain)"
    run_py X "$(cap X)" export "$OUT" "$WORK/out/export.json" -- python -m jobs.q9af_v3.export --stage A --mode salvage \
      --layers "$CONV" --src "$CK/gdn/stage$STOPPED.pt" --model-dir "$Q9B" --out-dir "$OUT" --meta-out "$WORK/out/export.json"; r=$?
    gate "X:export" $r
    [ $r -eq 0 ] || { EXPORT_FAIL=1; export_phase X; fail 6 "salvage export failed verification"; }
    EXPORTED=1
    export_phase X
    python -m jobs.q9af_v3.rule stagea "$CK/pipeline/results.json" "$WORK/gen/results.json" "$WORK/out/export.json" \
      "$WORK/out/gates.json" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1; r=$?
    gate "R:stageA" $r
    [ $r -eq 0 ] || fail 9 "stage-A record: $(head -c 300 "$WORK/out/decision.json" 2> /dev/null)"
    cp "$WORK/out/decision.json" "$OUT/decision.json"
    RC=0
    exit 0
  fi
  say "phase X (stage-A export)"
  run_py X "$(cap X)" export "$OUT/stageA" "$WORK/out/export.json" -- python -m jobs.q9af_v3.export --stage A \
    --src "$CK/gdn/stageA.pt" --model-dir "$Q9B" --out-dir "$OUT/stageA" --meta-out "$WORK/out/export.json"; r=$?
  gate "X:export" $r
  [ $r -eq 0 ] || { EXPORT_FAIL=1; export_phase X; fail 6 "stage-A export failed verification"; }
  EXPORTED=1
  cp -f "$WORK/out/export.json" "$OUT/stageA/export.json"
  for f in lc.u32 onpolicy.npz gen_heldout.npz split.json; do cp -f "$WORK/gen/$f" "$OUT/stageA/$f" || fail 6 "stage-A member $f"; done
  for s in web chat math; do cp -f "$WORK/mix/heldout_$s.u32" "$OUT/stageA/" || fail 6 "stage-A member heldout_$s.u32"; done
  cp -f "$WORK/mix/mix.json" "$OUT/stageA/mix.json" && cp -f "$CK/pipeline/results.json" "$OUT/stageA/results_A.json" \
    && cp -f "$WORK/gen/results.json" "$OUT/stageA/gen.json" || fail 6 "stage-A members"
  ( cd "$OUT/stageA" && sha256sum ./* > MEMBERS.sha256 ) || fail 6 "stage-A member hashes"
  export_phase X
  # ---------------------------------------------------------------- R: the stage-A record (no verdict)
  python -m jobs.q9af_v3.rule stagea "$CK/pipeline/results.json" "$WORK/gen/results.json" "$WORK/out/export.json" \
    "$WORK/out/gates.json" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1; r=$?
  gate "R:stageA" $r
  [ $r -eq 0 ] || fail 9 "incomplete stage A: $(head -c 300 "$WORK/out/decision.json" 2> /dev/null)"
  cp "$WORK/out/decision.json" "$OUT/decision.json"
  sample "$WORK/out/decision.json" "decision"
  RC=0
  exit 0
fi

# ==================================================================== mode B
# ------------------------------------------------------------------ F: fetch the pinned stage-A (+ v2) artifacts
say "phase F (stage-A artifact${REF:+; reference $REF})"
for which in stageA v2; do
  [ "$which" = v2 ] && [ "$REF" != v2 ] && continue
  dest=$WORK/a; [ "$which" = v2 ] && dest=$WORK/v2
  if [ -n "$DRY" ]; then
    $DRY fetch_artifact "$which" "$dest" > "$WORK/logs/fetch_$which.log" 2>&1; r=$?
  else
    timeout "$(cap F)m" python -X utf8 -m jobs.q9af_v3.artifact fetch --params "$PARAMS" --which "$which" --dest "$dest" \
      --work "$WORK" --out "$WORK/out/artifact_$which.json" > "$WORK/logs/fetch_$which.log" 2>&1; r=$?
  fi
  gate "F:$which" $r
  case $r in 0) ;; 124) fail 7 "fetch $which timeout" ;; 5) fail 5 "fetch $which: unpinned / no URL" ;; *) fail 10 "artifact $which identity rc=$r" ;; esac
done
export_phase F
# ------------------------------------------------------------------ P: 32k long context + on-policy
say "phase P (stage B: long context + on-policy)"
run_py P "$(cap P)" stageb "$CK" -- python -X utf8 -m convert.v3 stageB --config "$CFG" --a "$WORK/a" --out "$CK/pipeline"; r=$?
gate "P:stageB" $r
sync_pipeline
export_phase P
pipe_rc P $r
# ------------------------------------------------------------------ X: export (verified)
say "phase X (export)"
run_py X "$(cap X)" export "$OUT" "$WORK/out/export.json" -- python -m jobs.q9af_v3.export --stage B --mode final \
  --src "$CK/gdn/final.pt" --model-dir "$Q9B" --out-dir "$OUT" --meta-out "$WORK/out/export.json"; r=$?
gate "X:export" $r
[ $r -eq 0 ] || { EXPORT_FAIL=1; export_phase X; fail 6 "converted-weight export failed verification"; }
EXPORTED=1
cp -f "$WORK/out/export.json" "$OUT/export.json"
export_phase X
LAYERS=$(python -c "import json,sys;print(','.join(map(str,json.load(open(sys.argv[1]))['converted'])))" "$WORK/out/export.json") || fail 6 "export.json"
LAYERS=${LAYERS%$'\r'}
# ------------------------------------------------------------------ L: proxy600 arms, generic v3w readout (same path)
IDX=$WORK/out/index
NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$IDX/proxy-rows.json")
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
_arm L_stock v3w_stock "$TOKS" "$CALLS" "$(cap L_stock)" --option converted=none
_arm L_v3 v3w_v3 "$TOKS_C" "$CALLS_C" "$(cap L_v3)" --option converted="$LAYERS" --option gdn_ckpt="$OUT/gdn_converted.pt"
if [ "$REF" = v2 ]; then
  _arm L_v2 v3w_v2 "$TOKS_C" "$CALLS_C" "$(cap L_v2)" --option converted="$LAYERS" --option gdn_ckpt="$WORK/v2/gdn_converted.pt"
fi
# ------------------------------------------------------------------ S: paired compares + per-arm scores (frozen scorer)
say "phase S (proxy600 scoring, paired; reference $REF)"
REFDIR=v3w_stock; [ "$REF" = v2 ] && REFDIR=v3w_v2
S compare --candidate "$IDX/v3w_v3/results.jsonl" --baseline "$IDX/$REFDIR/results.jsonl" \
  > "$IDX/compare_v3_vs_ref.json" 2> "$WORK/logs/score.err" || fail 13 "compare v3 vs $REF"
S compare --candidate "$IDX/v3w_v3/results.jsonl" --baseline "$IDX/v3w_stock/results.jsonl" \
  > "$IDX/compare_v3_vs_stock.json" 2>> "$WORK/logs/score.err" || fail 13 "compare v3 vs stock"
CMP_V2=-
if [ "$REF" = v2 ]; then
  S compare --candidate "$IDX/v3w_v2/results.jsonl" --baseline "$IDX/v3w_stock/results.jsonl" \
    > "$IDX/compare_v2_vs_stock.json" 2>> "$WORK/logs/score.err" || fail 13 "compare v2 vs stock"
  CMP_V2=$IDX/compare_v2_vs_stock.json
fi
for arm in v3w_stock v3w_v3 v3w_v2; do
  [ -d "$IDX/$arm" ] || continue
  S score --results "$IDX/$arm/results.jsonl" > "$IDX/score_$arm.json" 2>> "$WORK/logs/score.err" || fail 13 "score $arm"
done
gate "S:score" 0
export_phase S
# ------------------------------------------------------------------ R: the pre-registered gate
say "phase R (rule)"
V2E=-; [ "$REF" = v2 ] && V2E=$WORK/v2/export.json
python -m jobs.q9af_v3.rule decide "$PARAMS" "$CK/pipeline/results.json" "$WORK/out/export.json" "$V2E" \
  "$IDX/compare_v3_vs_ref.json" "$IDX/compare_v3_vs_stock.json" "$CMP_V2" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1; r=$?
gate "R:decision" $r
[ $r -eq 0 ] || fail 9 "no valid decision (rc=$r): $(head -c 300 "$WORK/out/decision.json" 2> /dev/null)"
cp "$WORK/out/decision.json" "$OUT/decision.json"
sample "$WORK/out/decision.json" "decision"
RC=0
exit 0   # finish() re-validates decision.json via finalize.py before Q9AF3_DONE
