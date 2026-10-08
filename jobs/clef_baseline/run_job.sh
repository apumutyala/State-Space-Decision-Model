#!/usr/bin/env bash
# Clef baseline job entry point (see JOB_SPEC.md). Env: RELEASE=clef-flash|clef, JOB=A|B|I|T.
#   A: probes + Index DEV (decide + native arm) + score/ECE/latency + safety_eval (no GCG)
#   B: safety_eval GCG-lite (HP_GATE; 30 steer + 15 hijack), resumable from $GMN_CHECKPOINT_DIR
#   I: Index only (JOB_SPEC 4b): probes (no gradient probe) + Index on ROWS=subset (the 2,008-row m1 DEV subset) or
#      ROWS=full (12,104 DEV rows) + score / composite / ECE / latency (+ paired compare against Clef-flash when the
#      job-sgihj decide results ship as $DATA/reference/clef_flash.decide.results.jsonl). DEV only: TEST is never
#      loaded (no SSDM_UNSEAL_TEST), and the full 150,317-row board run is not this job (it would unseal TEST).
#   T: teacher export (JOB_SPEC 11): contamination guard + select on CPU first, then Clef per-option probabilities on
#      the m2 TRAINING draw (jobs/clef_baseline/teacher_scope.json). Never DEV/TEST/held-out rows.
# Exit codes: 0 ok | 2 bad env | 5 setup: fla #640 guard refused the image (before any download or GPU phase) |
#             10 download/verify/identity | 11 parity or fast-vs-reference GDN kernel check | 12 Index run |
#             13 scoring | 14 safety (A; includes the PREREG D1-B k=8 gate, runner exit 32) | 15 GCG (B) |
#             16 report | 17 a results file is truncated, malformed or missing rows (composite / scoring refused it;
#             job-sgihj class) | 18 too slow: the live rate guard projected the phase past its cap, or the cap hit |
#             19 teacher export failed | 20 contamination guard refused (T: DEV/TEST/cohort/held-out hit, or the
#             guard is not ready)
# I extras (JOB_SPEC 4b/4c): scores are edition 0.2.1 (as job-sgihj) plus a 0.3 report; the headline Clef number is
# the native composite (Clef's own 16,384-token truncation); phase sens re-reads the over-length rows
# (overlength_rows.json) at max_length 32768; ROWS=overlength (Clef-flash) runs only that arm, substituted into the
# pinned job-sgihj decide results. RELEASE=clef I jobs require that reference (exit 10 if absent).
#
# Heartbeat semantics (platform health check: heartbeat on $OUT/heartbeat, interval 600 s, fail_threshold 2,
# on_fail kill). A background watchdog touches $OUT/heartbeat every HB_EVERY_S (60 s) ONLY while BOTH hold:
#   (1) this script (the job's main process, $MAIN_PID) is still alive, and
#   (2) the worker made progress: some file under $OUT, $CK or the HF download dir changed within the last
#       STALL_S (1200 s). Every stage writes at that cadence or faster: the kit appends one results row per
#       request, metrics.run logs every 200 variants, GCG logs every 10 steps (~40 s), probes log every
#       1,000 rows, snapshot_download writes blobs, the teacher export appends one row per record.
# So a hung worker stops the heartbeat after STALL_S and the platform kills it within STALL_S + 2 x 600 s (<= 40 min)
# instead of billing to max_duration; a dead main process stops it immediately. No progress, no heartbeat.
#
# CPU dry run (tests/clef/test_clef_job_sh.py executes THIS script end to end): CLEF_DRY_RUN=1 with
# CLEF_STANDIN=<module:function> (a tiny stand-in model; refused without CLEF_DRY_RUN), CLEF_APP / CLEF_DATA (code and
# data roots, default /app and /data), CLEF_SUBSET (subset file), CLEF_TEACHER_N (draw size), CLEF_N_PARITY,
# CLEF_OVERLENGTH (over-length id list); every one of these overrides is refused (exit 2) outside a dry run. The dry
# run skips only what needs a GPU or the network (fla_guard, snapshot download + weight hashes, guard_ready's suite
# rebuild); every other phase, file check, exit-code mapping and the rate guard run unchanged. result.json says
# "dry_run": true. CLEF_DRY_FAULT=truncate_decide|drop_decide_row (dry run only) cuts the decide results mid-line or
# drops one complete row before scoring, to prove the exit-17 path end to end (strict reader on the merged harness).
set -uo pipefail
MAIN_PID=$$
HB_EVERY_S="${HB_EVERY_S:-60}"
STALL_S="${STALL_S:-1200}"
RELEASE="${RELEASE:?RELEASE must be clef-flash or clef}"
JOB="${JOB:?JOB must be A, B, I or T}"
ROWS="${ROWS:-full}"
DRY="${CLEF_DRY_RUN:-0}"
APP="${CLEF_APP:-/app}"
DATA="${CLEF_DATA:-/data}"
# CLEF_GDN_PATH (JOB_SPEC 4d): fast (default; fla / causal-conv1d kernels, unchanged) | reference (every Index phase --
# S2 decide, S2c 32k arm, S2b native -- on transformers' torch reference GDN, the arm kernel_path compares against;
# kernel_path still runs and is reported, the parity gate still gates; JOB=I subset/full only)
GDN_PATH="${CLEF_GDN_PATH:-fast}"
case "$GDN_PATH" in fast|reference) ;; *) echo "CLEF_GDN_PATH must be fast or reference"; exit 2 ;; esac
case "$RELEASE" in
  clef-flash) REPO=Cloudflare/clef-flash; REV=17f0b0ad64efb65d273590632833508766b2aae6 ;;
  clef)       REPO=Cloudflare/clef;       REV=2f3de3dd85f379784083b0814d997ab627200f0c ;;
  *) echo "unknown RELEASE $RELEASE"; exit 2 ;;
esac
case "$JOB" in A|B|T) ;; I) case "$ROWS" in subset|full|overlength) ;; *) echo "ROWS must be subset, full or overlength"; exit 2 ;; esac ;;
  *) echo "unknown JOB $JOB"; exit 2 ;; esac
# ROWS=overlength (Clef-flash only): the 51-row max_length-32768 sensitivity arm substituted into job-sgihj's decide
# results. The Clef-27B jobs run the same arm in-job (phase sens) on their own results.
[ "$ROWS" = "overlength" ] && [ "$RELEASE" != "clef-flash" ] && { echo "ROWS=overlength is the Clef-flash arm"; exit 2; }
[ "$GDN_PATH" = "reference" ] && { [ "$JOB" != "I" ] || [ "$ROWS" = "overlength" ]; } && { echo "CLEF_GDN_PATH=reference is for JOB=I subset/full"; exit 2; }
GDNOPT=(); [ "$GDN_PATH" = "reference" ] && GDNOPT=(--option gdn_path=reference)  # fast: the command lines are unchanged
if [ "$DRY" = "1" ]; then
  [ -n "${CLEF_STANDIN:-}" ] || { echo "CLEF_DRY_RUN=1 needs CLEF_STANDIN"; exit 2; }
  DEVOPT=(--option device=cpu --option dtype=float32); DEVARG=(--device cpu)
else
  for v in CLEF_STANDIN CLEF_DRY_FAULT CLEF_TEACHER_N CLEF_SUBSET CLEF_N_PARITY CLEF_OVERLENGTH; do
    [ -z "${!v:-}" ] || { echo "$v is for CLEF_DRY_RUN=1 only"; exit 2; }
  done
  DEVOPT=(); DEVARG=()
fi
# Index edition: 0.2.1, explicitly, so every number stays comparable to job-sgihj whatever the harness default
# becomes; a 0.3 score is reported alongside (--edition 0.3 --omit-gsm8k03: these runs never read the rebuilt GSM8K).
ED=(--edition 0.2.1); ED03=(--edition 0.3 --omit-gsm8k03)
SENS_MAX=32768
cd "$APP" || exit 2
case "$JOB" in I) TAG="${RELEASE}-index-${ROWS}" ;; T) TAG="${RELEASE}-teacher" ;; *) TAG="${RELEASE}-job${JOB}" ;; esac
OUT="${GMN_OUTPUT_DIR:-/output}/${TAG}"
CK="${GMN_CHECKPOINT_DIR:-/tmp/ck}/${TAG}"
mkdir -p "$OUT/env" "$OUT/index" "$CK"
export SSDM_DATA="$DATA" PYTHONUTF8=1 CLEF_DRY_RUN="$DRY"
SSDM_GIT_SHA=$(python -c "import json; print(json.load(open('$APP/CONTEXT_MANIFEST.json'))['branch_commit'])" 2>/dev/null || echo unknown)
export SSDM_GIT_SHA
HF_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
mkdir -p "$HF_DIR"
# job parameters (phase caps for the rate guard); JOB_PARAMS overrides, else the file for this job/release
case "$JOB$ROWS" in
  Isubset) PFILE=jobs/clef_baseline/job_params_index_subset.json ;;
  Ifull)   PFILE=jobs/clef_baseline/job_params_index_full.json
           [ "$GDN_PATH" = "reference" ] && PFILE=jobs/clef_baseline/job_params_index_full_reference.json
           [ "$GDN_PATH" = "reference" ] && [ "$RELEASE" = "clef-flash" ] && PFILE=jobs/clef_baseline/job_params_index_full_reference_flash.json ;;
  Ioverlength) PFILE=jobs/clef_baseline/job_params_overlength_flash.json ;;
  T*)      PFILE=jobs/clef_baseline/job_params_teacher_${RELEASE//-/_}.json ;;
  *)       PFILE="" ;;
esac
PFILE="${JOB_PARAMS:-$PFILE}"
cap() {  # cap <phase> <default>: phase cap minutes from the params file
  if [ -n "$PFILE" ] && [ -f "$PFILE" ]; then
    python -c "import json,sys; print(json.load(open(sys.argv[1]))['phase_caps_minutes'][sys.argv[2]])" "$PFILE" "$1" 2>/dev/null || echo "$2"
  else echo "$2"; fi
}
# shellcheck source=watchdog.sh
. "$APP/jobs/clef_baseline/watchdog.sh"
hb
start_watchdog
stage() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$OUT/stages.log"; hb; \
  curl -fsS -X POST "${GMN_METADATA_URL:-http://127.0.0.1:9}/v1/sample" -H 'content-type: application/json' \
    -d "{\"path\":\"$OUT/stages.log\",\"kind\":\"text\",\"label\":\"$*\"}" >/dev/null 2>&1 || true; }
fail() { echo "{\"ok\": false, \"exit\": $1, \"stage\": \"$2\", \"release\": \"$RELEASE\", \"job\": \"$JOB\", \"rows\": \"$ROWS\", \"dry_run\": $([ "$DRY" = 1 ] && echo true || echo false)}" > "${GMN_RESULT_PATH:-$OUT/result.json}"; exit "$1"; }
# scoring / composite exit 3 or 4 = a results file the strict reader refused (harness/jsonl.py) -> 17
score_rc() { case "$1" in 0) ;; 3|4) fail 17 "$2" ;; *) fail 13 "$2" ;; esac; }
# rate-guarded run: guarded <phase> <cap-min> <results path under $OUT> <total> <check-after-min> <fail-code> -- cmd...
guarded() {
  local phase=$1 capm=$2 rel=$3 total=$4 after=$5 code=$6; shift 7
  # the guard execs its child without a shell: give it this shell's python by absolute path (on Windows dry runs a bare
  # "python" can resolve to the interpreter's own directory instead of PATH)
  [ "$1" = "python" ] && set -- "$(command -v python)" "${@:2}"
  python -X utf8 -m jobs.common.rate_guard run --results "$OUT/$rel" --total "$total" --cap-min "$capm" \
      --check-after-min "$after" --exclude-startup --report "$OUT/env/rate_guard_${phase}.json" -- "$@"
  local rc=$?
  case $rc in 0) ;; 15|124) fail 18 "${phase}_rate_guard" ;; *) fail "$code" "$phase" ;; esac
}

stage "S0 env + data ($JOB${ROWS:+ $ROWS}, dry_run=$DRY)"
( nvidia-smi; nvcc --version; python -VV; pip freeze; python -c "import json; from baselines.clef_backend import kernel_stack; print(json.dumps(kernel_stack()))"; git -C "$APP" rev-parse HEAD 2>/dev/null; cat "$APP/CONTEXT_MANIFEST.json" ) > "$OUT/env/environment.txt" 2>&1
DEV_ROWS="$DATA/slices/dev-rows.jsonl.gz"
if [ "$JOB" != "T" ]; then
  WANT=$(cut -d' ' -f1 "$APP/jobs/clef_baseline/data.sha256")
  GOT=$(sha256sum "$DEV_ROWS" 2>/dev/null | cut -d' ' -f1)
  echo "$GOT $DEV_ROWS (want $WANT)" > "$OUT/env/data-verify.txt"
  [ "$GOT" = "$WANT" ] && echo OK >> "$OUT/env/data-verify.txt" || fail 10 data_verify
fi
# The paired Clef - Clef-flash comparison (and the Clef-flash sensitivity base) needs job-sgihj's decide results,
# pinned by sha256. Required in production before any download; a dry run records its absence in result.json.
REF="$DATA/reference/clef_flash.decide.results.jsonl"
REF_SHA=463f21ff492fc792da4d40ecc501cffab0498d2458715f8effd3270614cbd0aa
if [ "$JOB" = "I" ] && { [ "$RELEASE" = "clef" ] || [ "$ROWS" = "overlength" ]; }; then
  if [ -f "$REF" ]; then
    [ "$DRY" = "1" ] || [ "$(sha256sum "$REF" | cut -d' ' -f1)" = "$REF_SHA" ] || fail 10 reference_sha256
  elif [ "$DRY" != "1" ]; then
    fail 10 reference_missing
  fi
fi

if [ "$JOB" = "T" ]; then
  # Contamination guard + selection BEFORE any GPU phase or weight download (exit 20 = refused).
  stage "T0 guard readiness + teacher selection (CPU)"
  TEACH="$OUT/teacher"; mkdir -p "$TEACH"
  if [ "$DRY" = "1" ]; then
    GUARD=(--guard-standin --skip-file-hash --allow-n-mismatch)
  else
    python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir "$DATA/suite-0.2" --kit-dir /opt/decision-index \
        --out-dir "$OUT/guard" > "$OUT/guard.log" 2>&1 || fail 20 guard_ready
    GUARD=(--guard-readiness "$OUT/guard/readiness.json")
  fi
  N_ARG=(); [ "$DRY" = "1" ] && [ -n "${CLEF_TEACHER_N:-}" ] && N_ARG=(--n "$CLEF_TEACHER_N")  # dry run only
  timeout "$(cap guard 6)m" python -X utf8 -m baselines.clef_teacher select --train-dir "$DATA/m2-150k" \
      --out "$TEACH" "${GUARD[@]}" "${N_ARG[@]}" > "$OUT/teacher-select.log" 2>&1
  rc=$?; case $rc in 0) ;; 20) fail 20 teacher_select_contamination ;; 124) fail 18 teacher_select_timeout ;; *) fail 10 teacher_select ;; esac
  N_SEL=$(grep -c . "$TEACH/selected-ids.txt")
fi

if [ "$DRY" = "1" ]; then
  echo '{"skipped": "CLEF_DRY_RUN=1 (no GPU)"}' > "$OUT/env/fla_guard.json"
  SNAP="$OUT/standin-snapshot"; mkdir -p "$SNAP"
  echo '{"ok": null, "skipped": "CLEF_DRY_RUN=1 (stand-in model, no weights)"}' > "$OUT/env/snapshot-verify.json"
else
  # fla #640 (jobs/common/fla_guard.py): refuse the image (exit 5) unless torch 2.13 / triton >= 3.7.1 with fla's Triton
  # backward pinned and a tiny gated fwd+bwd runs on this GPU. Before the weights download and every GPU phase.
python -I /app/jobs/common/fla_guard.py --out "$OUT/env/fla_guard.json" > "$OUT/env/fla_guard.log" 2>&1 || fail 5 fla_guard
  stage "S0b download $REPO@$REV"
  SNAP=$(python - <<EOF
from huggingface_hub import snapshot_download
print(snapshot_download("$REPO", revision="$REV"))
EOF
  ) || fail 10 download
  python -X utf8 -c "import json,sys; from baselines.clef_backend import verify_snapshot as v; r=v('$SNAP','$RELEASE'); json.dump(r, open('$OUT/env/snapshot-verify.json','w'), indent=1); sys.exit(0 if r['ok'] else 1)" || fail 10 verify
fi

if [ "$JOB" = "A" ] || [ "$JOB" = "I" ]; then
  IDX="$OUT/index"
  if [ "$JOB" = "I" ] && [ "$ROWS" = "subset" ]; then
    SUBSET="${CLEF_SUBSET:-jobs/common/dev_subset_m1.json}"
    python -X utf8 -m jobs.common.dev_subset rows --rows "$DEV_ROWS" --out "$OUT/index/rows.jsonl.gz" \
        --subset "$SUBSET" > "$OUT/index/rows.json" 2>&1 || fail 10 subset_rows
    ROWS_FILE="$OUT/index/rows.jsonl.gz"
  else
    ROWS_FILE="$DEV_ROWS"
  fi
  # S2c rows: the committed over-length list (rows Clef's shipped max_length 16,384 cuts; 51 on DEV) within this job's rows
  OVERLEN="${CLEF_OVERLENGTH:-jobs/clef_baseline/overlength_rows.json}"
  python -X utf8 - "$OVERLEN" "$ROWS_FILE" "$OUT/index/overlength-rows.jsonl.gz" <<'EOF' || fail 10 overlength_rows
import gzip, json, sys
ids = {r["run_id"] for r in json.load(open(sys.argv[1], encoding="utf-8"))["rows"]}
with gzip.open(sys.argv[2], "rt", encoding="utf-8", newline="\n") as fi, \
     gzip.open(sys.argv[3], "wt", encoding="utf-8", newline="\n") as fo:
    for line in fi:
        if line.strip() and json.loads(line)["_evaluation"]["run_id"] in ids:
            fo.write(line)
EOF
  N_SENS=$(python -c "import gzip,sys; print(sum(1 for x in gzip.open(sys.argv[1],'rt',encoding='utf-8') if x.strip()))" "$OUT/index/overlength-rows.jsonl.gz")
  if [ "$ROWS" = "overlength" ]; then
    SC=(python -X utf8 -I -m harness); SL=(--slice dev)
    stage "S2c over-length sensitivity: $N_SENS rows at max_length $SENS_MAX (decide path)"
    timeout "$(cap sens 10)m" python -X utf8 -m decision_index run --engine baselines.clef_index_engine:ClefIndexEngine \
        --option model_dir="$SNAP" --option release="$RELEASE" --option path=decide --option max_length="$SENS_MAX" \
        "${DEVOPT[@]}" --rows "$OUT/index/overlength-rows.jsonl.gz" --out "$IDX/decide_32k" > "$IDX/decide_32k.log" 2>&1
    rc=$?; case $rc in 0) ;; 124) fail 18 sens_timeout ;; *) fail 12 sens ;; esac
    stage "S3 score: job-sgihj decide results with the $N_SENS rows substituted (edition 0.2.1)"
    python -X utf8 jobs/clef_baseline/summarize.py substitute --base "$REF" --rows "$IDX/decide_32k/results.jsonl" \
        --expect "$N_SENS" --out "$IDX/decide_32k_substituted/results.jsonl"; score_rc $? substitute_32k
    # read-only input (the pinned job-sgihj reference), not a phase output: hence --results=PATH
    "${SC[@]}" score --results="$REF" "${SL[@]}" "${ED[@]}" > "$IDX/score_reference_decide.json"; score_rc $? score_reference
    "${SC[@]}" score --results "$IDX/decide_32k_substituted/results.jsonl" "${SL[@]}" "${ED[@]}" > "$IDX/score_decide_32k_substituted.json"; score_rc $? score_32k
    "${SC[@]}" compare --candidate "$IDX/decide_32k_substituted/results.jsonl" --baseline "$REF" "${SL[@]}" "${ED[@]}" > "$IDX/compare_32k_vs_16k.json"; score_rc $? compare_32k
  else
  N_ROWS=$(python -c "import gzip,sys; print(sum(1 for x in gzip.open(sys.argv[1],'rt',encoding='utf-8') if x.strip()))" "$ROWS_FILE")
  GRAD=(); [ "$JOB" = "A" ] && GRAD=(--grad-probe)
  stage "S1 probes (parity, kernel_path, encodings, G8, G2b${GRAD:+, memory}) on $N_ROWS rows (gdn_path=$GDN_PATH)"
  timeout "$(cap probes 30)m" python -X utf8 -m baselines.clef_probes --model-dir "$SNAP" --release "$RELEASE" \
      --dev-rows "$ROWS_FILE" --out "$OUT/probes.json" --n-parity "${CLEF_N_PARITY:-64}" "${GRAD[@]}" "${DEVARG[@]}" \
      --gdn-path "$GDN_PATH" > "$OUT/probes.log" 2>&1
  rc=$?; case $rc in 0) ;; 11) fail 11 probes ;; 124) fail 18 probes_timeout ;; *) fail 10 probes ;; esac

  CAPI=$(cap index "$([ "$RELEASE" = clef ] && echo 141 || echo 60)")
  # measured-rate preflight: the S1 parity rows' decide timing on THIS box vs the job-sgihj rate model -> projected
  # Index minutes; past 95 % of the cap = stop now (18), before the long phase. Advisory in the CPU dry run.
  ADV=(); [ "$DRY" = "1" ] && ADV=(--advisory)
  VOL=dev_full; [ "$ROWS" = subset ] && VOL=dev_subset
  python -X utf8 -m jobs.clef_baseline.price preflight --probes "$OUT/probes.json" --volume "$VOL" \
      --cap-min "$CAPI" --out "$OUT/env/preflight_index.json" --gdn-path "$GDN_PATH" "${ADV[@]}" \
      > "$OUT/env/preflight_index.log" 2>&1
  rc=$?; case $rc in 0) ;; 15) fail 18 index_preflight ;; *) fail 13 index_preflight ;; esac
  stage "S2 Index DEV ($ROWS: $N_ROWS rows), decide path, batch 1, sequential; live rate guard"
  AFTER=$([ "$ROWS" = subset ] && echo 3 || echo 8)
  guarded index "$CAPI" index/decide/results.jsonl "$N_ROWS" "$AFTER" 12 -- \
    python -X utf8 -m decision_index run --engine baselines.clef_index_engine:ClefIndexEngine \
      --option model_dir="$SNAP" --option release="$RELEASE" --option path=decide "${DEVOPT[@]}" "${GDNOPT[@]}" \
      --rows "$ROWS_FILE" --out "$OUT/index/decide"
  if [ "$N_SENS" -gt 0 ]; then
    stage "S2c over-length sensitivity: $N_SENS rows at max_length $SENS_MAX (decide path; declared, JOB_SPEC 4c)"
    SENS_GDN="$GDN_PATH"
    timeout "$(cap sens 15)m" python -X utf8 -m decision_index run --engine baselines.clef_index_engine:ClefIndexEngine \
        --option model_dir="$SNAP" --option release="$RELEASE" --option path=decide --option max_length="$SENS_MAX" \
        "${DEVOPT[@]}" "${GDNOPT[@]}" --rows "$OUT/index/overlength-rows.jsonl.gz" --out "$IDX/decide_32k" > "$IDX/decide_32k.log" 2>&1
    rc=$?
    if [ "$rc" != 0 ] && [ "$rc" != 124 ] && [ "$GDN_PATH" = "reference" ] && grep -qi "out of memory" "$IDX/decide_32k.log"; then
      # JOB_SPEC 4d: the 32k arm only, re-run on the FAST path and LABELLED (env/sens_gdn_path.json -> result.json
      # gdn.sens). The reference attempt is kept as evidence (moved, never deleted); the 16k Index stays on the reference.
      mv "$IDX/decide_32k" "$IDX/decide_32k_reference_oom"; mv "$IDX/decide_32k.log" "$IDX/decide_32k_reference_oom.log"
      SENS_GDN=fast
      stage "S2c 32k arm: reference GDN out of memory; re-running the $N_SENS rows on the FAST path (labelled)"
      timeout "$(cap sens 15)m" python -X utf8 -m decision_index run --engine baselines.clef_index_engine:ClefIndexEngine \
          --option model_dir="$SNAP" --option release="$RELEASE" --option path=decide --option max_length="$SENS_MAX" \
          "${DEVOPT[@]}" --rows "$OUT/index/overlength-rows.jsonl.gz" --out "$IDX/decide_32k" > "$IDX/decide_32k.log" 2>&1
      rc=$?
    fi
    FB=null; [ "$SENS_GDN" != "$GDN_PATH" ] && FB='"reference out of memory at 32k"'
    echo "{\"gdn_path_requested\": \"$GDN_PATH\", \"gdn_path\": \"$SENS_GDN\", \"fallback\": $FB}" > "$OUT/env/sens_gdn_path.json"
    case $rc in 0) ;; 124) fail 18 sens_timeout ;; *) fail 12 sens ;; esac
  fi
  python -X utf8 - "$OUT/probes.json" "$ROWS_FILE" "$OUT/index/native-rows.jsonl.gz" <<'EOF' || fail 12 index_native_rows
import gzip, json, sys
p = json.load(open(sys.argv[1]))
keep = set(p["native_arm_run_ids"]) | set(p["parity_sample_run_ids"])
with gzip.open(sys.argv[2], "rt", encoding="utf-8", newline="\n") as fi, \
     gzip.open(sys.argv[3], "wt", encoding="utf-8", newline="\n") as fo:
    for line in fi:
        if line.strip() and json.loads(line)["_evaluation"]["run_id"] in keep:
            fo.write(line)
EOF
  N_NAT=$(python -c "import gzip,sys; print(sum(1 for x in gzip.open(sys.argv[1],'rt',encoding='utf-8') if x.strip()))" "$OUT/index/native-rows.jsonl.gz")
  stage "S2b Index DEV, Clef native systemone on differing/truncated rows + parity sample ($N_NAT rows)"
  timeout "$(cap native 15)m" python -X utf8 -m decision_index run --engine baselines.clef_index_engine:ClefIndexEngine \
      --option model_dir="$SNAP" --option release="$RELEASE" --option path=native --option verify=false "${DEVOPT[@]}" "${GDNOPT[@]}" \
      --rows "$OUT/index/native-rows.jsonl.gz" --out "$OUT/index/native" > "$OUT/index/native.log" 2>&1
  rc=$?; case $rc in 0) ;; 124) fail 18 index_native_timeout ;; *) fail 12 index_native ;; esac
  if [ "$DRY" = "1" ]; then  # fault injection, dry run only: a partial write / a lost row must end in exit 17
    case "${CLEF_DRY_FAULT:-}" in
      truncate_decide) python -c "import sys; p=sys.argv[1]; b=open(p,'rb').read(); open(p,'wb').write(b[:len(b)-len(b.rstrip(b'\n').rsplit(b'\n',1)[-1])//2-1])" "$IDX/decide/results.jsonl" ;;
      drop_decide_row) python -c "import sys; p=sys.argv[1]; b=open(p,'rb').read(); open(p,'wb').write(b[b.index(b'\n')+1:])" "$IDX/decide/results.jsonl" ;;
    esac
  fi

  stage "S3 score (decide first; edition 0.2.1 + a 0.3 report) + composite (headline) + 32k sensitivity + ECE + latency"
  if [ "$JOB" = "I" ] && [ "$ROWS" = "subset" ]; then
    SC=(python -X utf8 -m jobs.common.dev_subset); SL=(--subset "$SUBSET"); ECE_ED=()
  else
    SC=(python -X utf8 -I -m harness); SL=(--slice dev); ECE_ED=("${ED[@]}")
  fi
  "${SC[@]}" score --results "$IDX/decide/results.jsonl" "${SL[@]}" "${ED[@]}" > "$IDX/score_decide.json"; score_rc $? score_decide
  python -X utf8 jobs/clef_baseline/summarize.py composite --decide "$IDX/decide/results.jsonl" \
      --native "$IDX/native/results.jsonl" --probes "$OUT/probes.json" --out "$IDX/native_composite/results.jsonl"
  rc=$?; case $rc in 0) ;; 3|4) fail 17 composite ;; *) fail 12 composite ;; esac
  "${SC[@]}" score --results "$IDX/native_composite/results.jsonl" "${SL[@]}" "${ED[@]}" > "$IDX/score_native_composite.json"; score_rc $? score_native
  "${SC[@]}" compare --candidate "$IDX/decide/results.jsonl" --baseline "$IDX/native_composite/results.jsonl" "${SL[@]}" "${ED[@]}" > "$IDX/compare_decide_vs_native.json"; score_rc $? compare
  "${SC[@]}" ece --results "$IDX/decide/results.jsonl" "${SL[@]}" "${ECE_ED[@]}" > "$IDX/ece_decide.json"; score_rc $? ece
  # 0.3 report (not the comparison basis: job-sgihj and every receipt so far are 0.2.1)
  "${SC[@]}" score --results "$IDX/decide/results.jsonl" "${SL[@]}" "${ED03[@]}" > "$IDX/score_decide_03.json"; score_rc $? score_decide_03
  "${SC[@]}" score --results "$IDX/native_composite/results.jsonl" "${SL[@]}" "${ED03[@]}" > "$IDX/score_native_composite_03.json"; score_rc $? score_native_03
  if [ "$N_SENS" -gt 0 ]; then
    python -X utf8 jobs/clef_baseline/summarize.py substitute --base "$IDX/decide/results.jsonl" \
        --rows "$IDX/decide_32k/results.jsonl" --expect "$N_SENS" --out "$IDX/decide_32k_substituted/results.jsonl"; score_rc $? substitute_32k
    "${SC[@]}" score --results "$IDX/decide_32k_substituted/results.jsonl" "${SL[@]}" "${ED[@]}" > "$IDX/score_decide_32k_substituted.json"; score_rc $? score_32k
  fi
  if [ "$ROWS" = "full" ] && [ -f "$APP/jobs/common/dev_subset_m1.json" ] && [ "$DRY" != "1" ]; then
    python -X utf8 -m jobs.common.dev_subset score --results "$IDX/decide/results.jsonl" "${ED[@]}" > "$IDX/score_decide_subset.json"; score_rc $? score_subset
  fi
  if [ -f "$REF" ] && [ "$RELEASE" = "clef" ]; then  # paired on identical rows: Clef - Clef-flash (job-sgihj); required in production
    "${SC[@]}" compare --candidate "$IDX/decide/results.jsonl" --baseline "$REF" "${SL[@]}" "${ED[@]}" > "$IDX/compare_clef_vs_flash.json"; score_rc $? compare_vs_flash
    "${SC[@]}" compare --candidate "$IDX/native_composite/results.jsonl" --baseline "$REF" "${SL[@]}" "${ED[@]}" > "$IDX/compare_clef_composite_vs_flash.json"; score_rc $? compare_composite_vs_flash
  fi
  fi
fi

if [ "$JOB" = "A" ]; then
  stage "S4 safety_eval (test split, --final, no GCG)"
  python -X utf8 -m baselines.run_clef_safety --model-dir "$SNAP" --release "$RELEASE" --split test --final \
      --skip-gcg --out "$OUT/safety" "${DEVARG[@]}" > "$OUT/safety.log" 2>&1 || fail 14 safety
elif [ "$JOB" = "B" ]; then
  stage "S4b safety_eval GCG-lite (HP_GATE, 30 steer + 15 hijack), resumable"
  python -X utf8 -m baselines.run_clef_safety --model-dir "$SNAP" --release "$RELEASE" --split test --final \
      --gcg-only --gcg-steer 30 --gcg-hijack 15 --gcg-hp gate --out "$CK/safety" "${DEVARG[@]}" > "$OUT/gcg.log" 2>&1 || fail 15 gcg
  cp -r "$CK/safety" "$OUT/safety"
elif [ "$JOB" = "T" ]; then
  stage "T1 teacher export ($N_SEL training records, decide path, batch 1; live rate guard)"
  DT=(); [ "$DRY" = "1" ] && DT=(--dtype float32 --no-verify)
  guarded export "$(cap export 230)" "teacher/teacher-${RELEASE}.jsonl" "$N_SEL" 8 19 -- \
    python -X utf8 -m baselines.clef_teacher export --model-dir "$SNAP" --release "$RELEASE" \
      --train-dir "$DATA/m2-150k" --out "$TEACH" "${DEVARG[@]}" "${DT[@]}"
fi

stage "S5 report"
python -X utf8 jobs/clef_baseline/summarize.py report --out-dir "$OUT" --release "$RELEASE" --job "$JOB" \
    --rows "$ROWS" --dry-run "$DRY" --gdn-path "$GDN_PATH" > "$OUT/result.json"
rc=$?; case $rc in 0) ;; 3|4) fail 17 report ;; *) fail 16 report ;; esac
cp "$OUT/result.json" "${GMN_RESULT_PATH:-$OUT/result.json}" 2>/dev/null || true
( cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name heartbeat ! -name watchdog.log -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
stage "done"
exit 0
