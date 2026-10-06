#!/usr/bin/env bash
# Rung-2 HARDWARE QUALIFICATION, run inside an SF Compute batch job (submit_job, chip h100). Scope: numerics,
# throughput, memory, artifact recovery. No curve, no capability claim. Spec: jobs/qual/JOB_SPEC.md.
#
# Exit codes (also in $GMN_RESULT_PATH): 0 all required gates passed and every export verified
#   2 numerics gate failed (tests or qual numerics sanity)   3 throughput stop   4 memory stop
#   5 setup failed   6 an export failed verification   7 a required phase timed out   8 9B weights unavailable
#   9 decision.json missing/invalid   1 finalize itself failed
# /output/QUAL_PASS is written ONLY by jobs/qual/finalize.py, and only when its final code is 0.
# Phase caps come from jobs/qual/job_params.json (checked against max_duration_minutes by tests/test_qual_job.py).
set -Euo pipefail
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/qual
META=${GMN_METADATA_URL:-http://169.254.42.1}
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs"
export SSDM_DATA_ROOT=$WORK/data PYTHONUNBUFFERED=1
mkdir -p "$SSDM_DATA_ROOT"
# org HF cache (hf_cache: true) when mounted; code expects $SSDM_DATA_ROOT/hf/<hub layout>
if [ -n "${HF_HUB_CACHE:-}" ] && [ -d "${HF_HUB_CACHE}" ]; then ln -sfn "$HF_HUB_CACHE" "$SSDM_DATA_ROOT/hf"; else mkdir -p "$SSDM_DATA_ROOT/hf"; fi
T0=$(date +%s)
EXPORT_FAIL=0
RC=""
GATES="{}"
exec > >(tee -a "$WORK/logs/run.log") 2>&1
say() { echo "[$(date -u +%FT%TZ) +$(( ($(date +%s) - T0) / 60 ))m] $*"; }
gate() {  # gate <name> <rc> : record a required check
  GATES=$(python - "$GATES" "$1" "$2" <<'EOF'
import json, sys
g = json.loads(sys.argv[1]); g[sys.argv[2]] = int(sys.argv[3]); print(json.dumps(g))
EOF
)
}
sample() { curl -s -m 5 -X POST -H 'Metadata-Flavor: givemeanode' -H 'Content-Type: application/json' \
  -d "{\"path\": \"$1\", \"kind\": \"json\", \"label\": \"$2\"}" "$META/v1/sample" > /dev/null 2>&1 || true; }
export_phase() {  # tar the phase output, verify by re-reading + hash, record success or failure in MANIFEST.jsonl
  local ph=$1 f="$OUT/phases/$1.tgz" ok=false sha="" bytes=0
  if tar czf "$f" -C "$WORK" out logs 2>/dev/null && tar tzf "$f" > /dev/null 2>&1; then
    sha=$(sha256sum "$f" | cut -d' ' -f1); bytes=$(stat -c %s "$f")
    [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ] && [ "$bytes" -gt 0 ] && ok=true
  fi
  echo "{\"phase\": \"$ph\", \"file\": \"phases/$ph.tgz\", \"sha256\": \"$sha\", \"bytes\": $bytes, \"ok\": $ok, \"t\": \"$(date -u +%FT%TZ)\"}" >> "$OUT/MANIFEST.jsonl"
  [ "$ok" = true ] || EXPORT_FAIL=1
  sample "$OUT/MANIFEST.jsonl" "export:$ph"
  say "export $ph ok=$ok sha256=$sha"
}
finish() {
  local rc=${RC:-$?}
  kill $HB_PID 2> /dev/null || true
  export_phase final
  # the only writer of QUAL_PASS: non-zero on any failed gate, unverified export (6) or invalid decision (9)
  python /opt/ssdm/m1/jobs/qual/finalize.py "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL"     "$(( ($(date +%s) - T0) / 60 ))" "$WORK/out/decision.json" "$OUT"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
trap finish EXIT
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
( while true; do date -u +%FT%TZ > "$OUT/heartbeat"; sleep 60; done ) & HB_PID=$!
cap() { python -c "import json,sys;print(json.load(open('/opt/ssdm/m1/jobs/qual/job_params.json'))['phase_caps_minutes'][sys.argv[1]])" "$1"; }

# ------------------------------------------------------------------ setup (fail fast: exit 5)
say "setup"
cp /opt/ssdm/CONTEXT_MANIFEST.json "$WORK/out/" || fail 5 "no context manifest"
nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins"
import json, torch, transformers, importlib.metadata as m
pins = {"transformers": "5.17.0", "flash-linear-attention": "0.5.2", "fla-core": "0.5.2"}
got = {k: m.version(k) for k in pins}
assert got == pins, got
assert torch.cuda.is_available()
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), **got}))
EOF
cd /opt/ssdm/m1
CAP_A=$(cap A_per_test) && CAP_B=$(cap B) && CAP_CW=$(cap C_weight_wait) && CAP_C=$(cap C) || fail 5 "job_params"
python -I scripts/sf_fetch.py --stage first > "$WORK/logs/fetch_first.log" 2>&1 || fail 5 "fetch first"
ls "$SSDM_DATA_ROOT"/FAILED_* > /dev/null 2>&1 && fail 5 "fetch first: $(cat "$SSDM_DATA_ROOT"/FAILED_*)"
( python -I scripts/sf_fetch.py --stage rest > "$WORK/logs/fetch_rest.log" 2>&1 ) &   # 9B in the background
export_phase setup

# ------------------------------------------------------------------ A: numerics gates (exit 2)
say "phase A"
mkdir -p "$WORK/out/tests"
for t in test_gdn test_init_shapes test_progressive_resume test_guard test_qual_job; do
  timeout ${CAP_A}m python -m tests.$t > "$WORK/out/tests/$t.json" 2> "$WORK/logs/$t.err"; r=$?
  gate "A:$t" $r
  say "A $t rc=$r"
  [ $r -eq 0 ] || { export_phase A; fail 2 "numerics gate $t"; }
done
export_phase A

# ------------------------------------------------------------------ B: 0.8B throughput/memory/numerics
say "phase B"
timeout ${CAP_B}m python -m convert.qual --config configs/qual_h100.json --model 0p8b --out "$WORK/out" \
  > "$WORK/logs/qual_0p8b.log" 2>&1; r=$?
gate "B:qual_0p8b" $r
export_phase B
case $r in 0) ;; 2|3|4) fail $r "qual 0p8b stop" ;; 124) fail 7 "qual 0p8b timeout" ;; *) fail 2 "qual 0p8b crashed rc=$r" ;; esac

# ------------------------------------------------------------------ C: 9B throughput/memory probe
say "phase C"
W_END=$(( $(date +%s) + CAP_CW*60 ))
while [ ! -f "$SSDM_DATA_ROOT/DONE_Q9B_DIR" ] && [ $(date +%s) -lt $W_END ]; do
  [ -f "$SSDM_DATA_ROOT/FAILED_Q9B_DIR" ] && break; sleep 20; done
[ -f "$SSDM_DATA_ROOT/DONE_Q9B_DIR" ] || { gate "C:weights" 8; export_phase C; fail 8 "9B weights unavailable"; }
timeout ${CAP_C}m python -m convert.qual --config configs/qual_h100.json --model 9b --out "$WORK/out" \
  > "$WORK/logs/qual_9b.log" 2>&1; r=$?
gate "C:qual_9b" $r
export_phase C
case $r in 0) ;; 2|3|4) fail $r "qual 9b stop" ;; 124) fail 7 "qual 9b timeout" ;; *) fail 2 "qual 9b crashed rc=$r" ;; esac

# ------------------------------------------------------------------ D: pre-registered decision
say "phase D"
python scripts/decide_9b.py "$WORK/out/qual_9b.json" configs/qual_h100.json "$WORK/out/decision.json"   > "$WORK/logs/decision.log" 2>&1; r=$?
gate "D:decision" $r
[ $r -eq 0 ] || fail 9 "decision generation failed rc=$r"
RC=0
exit 0   # finish() re-validates decision.json via finalize.py before QUAL_PASS
