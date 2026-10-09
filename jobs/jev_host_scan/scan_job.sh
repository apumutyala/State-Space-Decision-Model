#!/usr/bin/env bash
# jobs/jev_host_scan: score candidate small decision-model hosts on the 231 public JevBench items (v1.2, pinned
# b6b8fff7, EVAL-ONLY) through jevkit: open-split Intelligence, Calibration, raw + adjusted p50/p95 latency (one request
# at a time), input tokens per decision, a projected Cost axis under each candidate's DECLARED price class, Composite
# A/B/C and Capability with bootstrap CIs. Spec: jobs/jev_host_scan/JOB_SPEC.md. Candidates, readouts, caps and price
# classes: jobs/jev_host_scan/candidates.json (run order = priority).
#   setup: pins + fla guard -> the pinned public items (sha256) -> per-candidate schedule
#   scan:  per candidate, serially (one model on the GPU at a time): pinned snapshot_download (download cap) -> sha256 of
#          every pinned file + vendored-code pins -> host_scan run under jobs.common.rate_guard (the candidate's cap;
#          a slow model is stopped by its MEASURED rate, never eats the job) -> jevkit score per arm -> export (partial
#          results survive) -> the snapshot is deleted. A candidate that fails is recorded and the scan moves on; a
#          candidate whose cap no longer fits the scan phase is skipped (lowest priority last).
#   summarize: decision.json (measurement only: per candidate x readout axes, CIs, H2O-Lightning identity check)
# Exit codes: 0 measured   5 setup   6 export verification   7 scan phase over its cap   9 decision invalid
#   10 public items identity   13 summarize failed   1 finalize
# The marker /output/JEV_HOST_SCAN_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
set -Euo pipefail
MAIN_PID=$$  # jobs/clef_baseline/watchdog.sh beats only while this PID lives
CODE=/opt/ssdm/m1
PARAMS=$CODE/jobs/jev_host_scan/job_params.json
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/jev-host-scan
CK=$WORK/ck
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1500
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out/cand" "$WORK/logs" "$CK"
export SSDM_DATA=/data PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_ENABLE_HF_TRANSFER=0
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
MARKER=JEV_HOST_SCAN_DONE
VALIDATOR=jobs.jev_host_scan.host_scan:valid_scan
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
status() {  # status <dir> <state> <json extras>
  python - "$1/status.json" "$2" "$3" <<'EOF'
import json, sys
d = json.loads(sys.argv[3]); d["state"] = sys.argv[2]
json.dump(d, open(sys.argv[1], "w"), indent=1)
EOF
}
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup: identities (exit 5 / 10)
say "setup"
cp /opt/ssdm/CONTEXT_MANIFEST.json "$WORK/out/" || fail 5 "no context manifest"
nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
df -h "$HF_DIR" /tmp > "$WORK/out/disk.txt" 2>&1 || true
python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins"
import json, torch, transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
assert torch.cuda.is_available()
import fla  # noqa: F401  (Qwen3.5 Gated DeltaNet kernels; transformers falls back to torch without it)
from importlib import metadata
import third_party.d1_3b as D, third_party.h2o_lightning_shim as H, third_party.rwkv7_hf as V
assert all(D.vendored_ok().values()) and all(H.vendored_ok().values()) and all(V.vendored_ok().values())
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "transformers": transformers.__version__, "fla": metadata.version("flash-linear-attention"),
                  "gpu_mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}))
EOF
python -I /opt/ssdm/m1/jobs/common/fla_guard.py --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
python -X utf8 -m jevkit items > "$WORK/out/items-verify.json" 2> "$WORK/logs/items.log" || fail 10 "public items sha256"
NITEMS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['items']['n'])" "$CODE/jobs/jev_host_scan/candidates.json") \
  || fail 5 "candidates"
CAP_SCAN=$(cap scan) && CAP_SUM=$(cap summarize) && SCORE_MIN=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['score_cap_minutes'])" "$PARAMS") \
  || fail 5 "job_params"
python -X utf8 -m jobs.jev_host_scan.host_scan plan > "$WORK/out/plan.txt" || fail 5 "plan"
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ scan (exit 7); per-candidate failures are recorded
SCAN_END=$(( $(date +%s) + 60 * CAP_SCAN ))
say "phase scan ($(wc -l < "$WORK/out/plan.txt") candidates, $NITEMS items each arm, scan cap ${CAP_SCAN}m)"
exec 3< "$WORK/out/plan.txt"
while read -r -u 3 ID REPO REV CAP DLCAP NARMS PIN PATS; do
  D=$WORK/out/cand/$ID
  mkdir -p "$D"
  LEFT=$(( (SCAN_END - $(date +%s)) / 60 ))
  NEED=$(( CAP + DLCAP + SCORE_MIN * NARMS ))
  if [ "$LEFT" -lt "$NEED" ]; then
    say "skip $ID: ${LEFT}m left in the scan phase < ${NEED}m (caps)"
    status "$D" skipped_no_time "{\"left_min\": $LEFT, \"need_min\": $NEED}"
    continue
  fi
  say "candidate $ID ($REPO@$REV): download (cap ${DLCAP}m)"
  TD=$(date +%s)
  SNAP=$(timeout "${DLCAP}m" python - "$REPO" "$REV" "$PATS" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=16, allow_patterns=sys.argv[3].split(",")))
EOF
  ); r=$?
  if [ "$r" != 0 ]; then
    status "$D" download_failed "{\"rc\": $r}"
    export_phase "cand-$ID"
    continue
  fi
  SNAP=$(echo "$SNAP" | tail -1)
  DL_S=$(( $(date +%s) - TD ))
  if ! python -X utf8 -m jobs.jev_host_scan.host_scan verify --cand "$ID" --model-dir "$SNAP" > "$D/verify.json" 2> "$D/verify.err" < /dev/null; then
    say "verify failed for $ID: $(tail -1 "$D/verify.err")"
    status "$D" verify_failed "{\"download_s\": $DL_S}"
    rm -rf "${SNAP%/snapshots/*}"
    export_phase "cand-$ID"
    continue
  fi
  say "candidate $ID: run $NARMS arm(s) x $NITEMS items, serial, rate-guarded (cap ${CAP}m)"
  TR=$(date +%s)
  python -m jobs.common.rate_guard run --results "$D/*/preds.jsonl" --total $(( NARMS * NITEMS )) --exclude-startup \
    --cap-min "$CAP" --check-after-min 3 --report "$D/rate.json" -- \
    python -X utf8 -m jobs.jev_host_scan.host_scan run --cand "$ID" --model-dir "$SNAP" --out "$D" \
    > "$WORK/logs/run-$ID.log" 2>&1 < /dev/null; r=$?
  RUN_S=$(( $(date +%s) - TR ))
  python -X utf8 -m jobs.jev_host_scan.host_scan price --cand "$ID" --out "$D/price.json" < /dev/null || true
  for P in "$D"/*/preds.jsonl; do
    [ -s "$P" ] || continue
    A=$(dirname "$P")
    timeout "${SCORE_MIN}m" python -X utf8 -m jevkit score --public --pred "$P" --price "$D/price.json" \
      --out "$A/report.json" > "$A/score.log" 2>&1 < /dev/null || say "score failed: $A ($(tail -1 "$A/score.log"))"
  done
  case $r in 0) ST=done ;; 15) ST=rate_stopped ;; 124) ST=cap_timeout ;; 3) ST=load_failed ;; 4) ST=arm_failed ;; *) ST=failed ;; esac
  status "$D" "$ST" "{\"run_rc\": $r, \"download_s\": $DL_S, \"run_s\": $RUN_S, \"snapshot\": \"$SNAP\"}"
  say "candidate $ID: $ST (rc $r, download ${DL_S}s, run ${RUN_S}s)"
  cp "$WORK/logs/run-$ID.log" "$D/" 2> /dev/null || true
  rm -rf "${SNAP%/snapshots/*}"
  export_phase "cand-$ID"
  if [ "$(date +%s)" -gt "$SCAN_END" ]; then
    gate "scan" 7
    fail 7 "scan phase over its cap after $ID"
  fi
done
exec 3<&-
gate "scan" 0

# ------------------------------------------------------------------ summarize (exit 13 / 9)
say "phase summarize"
timeout "${CAP_SUM}m" python -X utf8 -m jobs.jev_host_scan.host_scan summarize --root "$WORK/out" \
  --out "$WORK/out/decision.json" > "$WORK/logs/summarize.log" 2>&1 || fail 13 "summarize: $(tail -1 "$WORK/logs/summarize.log")"
gate "summarize" 0
sample "$WORK/out/decision.json" "decision"
RC=0
exit 0
