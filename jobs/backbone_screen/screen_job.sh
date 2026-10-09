#!/usr/bin/env bash
# jobs/backbone_screen: STOCK zero-shot screen of an attention-free backbone candidate on the pre-declared proxy600 rows
# (644 rows of the 2,008-row Index DEV subset, jobs/common/dev_proxy600.json), frozen scorer, paired against the stock
# RWKV7-G1k v3w readout (job-fgu2c / art-ivu28, the same 644 rows). Spec: jobs/backbone_screen/JOB_SPEC.md.
# Env: MODE=qrwkv6_32b (recursal/QRWKV6-32B-Instruct-Preview-v0.1) | qrwkv_qwq32b (featherless-ai/QRWKV-QwQ-32B)
#   | rwkv_qwen3_30b_a3b (OpenMOSE/RWKV-Qwen3-30B-A3B-Instruct-cxa07c)
#   setup: pins + fla guard -> DEV rows + parent sha256 -> pinned snapshot_download -> sha256 of every file -> vendored
#          code == snapshot code -> proxy600 rows
#   index: decision_index run (ReadoutIndexEngine, family from params; fork == recompute canary self-check before row
#          one), live rate guard
#   score: proxy600 slice index + CI, ECE, paired compare vs stock RWKV7 v3w -> decision.json (measurement, no gate)
# Exit codes: 0 measured   2 bad MODE   5 setup   6 export verification   7 phase timeout   9 decision invalid
#   10 snapshot / vendored code / DEV rows / parent identity, or download failed   12 Index kit run failed
#   13 scoring failed   15 readout too slow by MEASUREMENT (jobs/common/rate_guard.py live row rate)   1 finalize
# The marker /output/BACKBONE_SCREEN_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
set -Euo pipefail
MAIN_PID=$$  # jobs/clef_baseline/watchdog.sh beats only while this PID lives
CODE=/opt/ssdm/m1
MODE=${MODE:-}
case "$MODE" in qrwkv6_32b|qrwkv_qwq32b|rwkv_qwen3_30b_a3b) ;; *) echo "MODE must be qrwkv6_32b, qrwkv_qwq32b or rwkv_qwen3_30b_a3b"; exit 2 ;; esac
PARAMS=$CODE/jobs/backbone_screen/job_params_$MODE.json
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/backbone-$MODE
CK=$WORK/ck
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1200
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out/index" "$WORK/logs" "$CK"
export SSDM_DATA=/data PYTHONUNBUFFERED=1 PYTHONUTF8=1
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
MARKER=BACKBONE_SCREEN_DONE
VALIDATOR=jobs.backbone_screen.summarize:valid_screen
P() { python -c "import json,sys;print(json.load(open(sys.argv[1]))['screen'][sys.argv[2]])" "$PARAMS" "$1"; }
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
hb
watchdog & WD_PID=$!
trap finish EXIT

# ------------------------------------------------------------------ setup: identities (exit 5 / 10)
REPO=$(P repo) && REV=$(P revision) && FAMILY=$(P family) && PIN=$CODE/$(P snapshot_pin) && CAP_IN=$(P input_cap) \
  && PKG=$(P code_pkg) && KERNEL=$(P kernel) || fail 5 "job_params screen block"
export PKG KERNEL
PARENT=/data/$(P parent_results) && PARENT_SHA=$(P parent_sha256) || fail 5 "job_params parent"
say "setup ($MODE: $REPO@$REV, family $FAMILY)"
cp /opt/ssdm/CONTEXT_MANIFEST.json "$WORK/out/" || fail 5 "no context manifest"
nvidia-smi > "$WORK/out/nvidia-smi.txt" || fail 5 "nvidia-smi"
df -h "$HF_DIR" /tmp > "$WORK/out/disk.txt" 2>&1 || true
python - > "$WORK/out/env.json" <<'EOF' || fail 5 "env pins"
import json, torch, transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
assert torch.cuda.is_available()
import decision_index  # noqa: F401
import fla  # noqa: F401
import importlib, os
mod, _, fn = os.environ["KERNEL"].rpartition(".")
getattr(importlib.import_module(mod), fn)  # the vendored model's recurrent kernel (fla 0.5.2)
V = importlib.import_module(os.environ["PKG"])
assert all(V.vendored_ok().values()), V.vendored_ok()
from importlib import metadata
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "transformers": transformers.__version__, "fla": metadata.version("flash-linear-attention"),
                  "gpu_mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}))
EOF
python -I /opt/ssdm/m1/jobs/common/fla_guard.py --out "$WORK/out/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
( cd /data && sha256sum -c "$CODE/jobs/clef_baseline/data.sha256" ) > "$WORK/out/data-verify.txt" 2>&1 || fail 10 "DEV rows sha256"
echo "$PARENT_SHA  $PARENT" | sha256sum -c - >> "$WORK/out/data-verify.txt" 2>&1 || fail 10 "parent results sha256"
CAP_SETUP=$(cap setup) || fail 5 "job_params"
SNAP=$(timeout ${CAP_SETUP}m python - "$REPO" "$REV" "$PARAMS" <<'EOF'
import json, sys
from huggingface_hub import snapshot_download
pats = json.load(open(sys.argv[3]))["screen"]["allow_patterns"]
print(snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=16, allow_patterns=pats))
EOF
) || fail 10 "download"
SNAP=$(echo "$SNAP" | tail -1)
say "snapshot $SNAP; verifying sha256 of every pinned file"
# every pinned file, 8 at a time (65 GiB of shards); a missing or changed file fails the line
( cd "$SNAP" && xargs -P 8 -n 2 sh -c 'echo "$0  $1" | sha256sum -c -' < "$PIN" ) > "$WORK/out/snapshot-verify.txt" 2>&1 \
  || fail 10 "snapshot sha256"
[ "$(grep -c ': OK$' "$WORK/out/snapshot-verify.txt")" -eq "$(grep -c . "$PIN")" ] || fail 10 "snapshot sha256 count"
python -c "import importlib, os, sys; V = importlib.import_module(os.environ['PKG']); V.verify_snapshot_code(*([sys.argv[1], sys.argv[2]] if hasattr(V, 'REPOS') else [sys.argv[1]]))" "$SNAP" "$REPO" \
  || fail 10 "vendored code != snapshot code"
SUBSET=jobs/common/dev_proxy600.json
python -X utf8 -m jobs.common.dev_proxy check --proxy proxy600 > "$WORK/out/index/proxy-check.json" || fail 10 "proxy600 identity"
python -X utf8 -m jobs.common.dev_subset rows --rows /data/slices/dev-rows.jsonl.gz --out "$WORK/proxy-rows.jsonl.gz" \
  --subset "$SUBSET" > "$WORK/out/index/proxy-rows.json" || fail 10 "proxy rows"
NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$WORK/out/index/proxy-rows.json")
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ index (exit 7 / 12 / 15)
CAP_I=$(cap index) && CAP_SC=$(cap score) || fail 5 "job_params"
say "phase index ($NROWS proxy600 rows, family $FAMILY, input cap $CAP_IN; canary fork self-check first; live rate guard)"
python -m jobs.common.rate_guard run --results "$WORK/out/index/stock/results.jsonl" --total "$NROWS" --exclude-startup \
  --cap-min "$CAP_I" --report "$WORK/out/index/rate.json" -- \
  python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine \
  --option family="$FAMILY" --option model_dir="$SNAP" --option repo="$REPO" --option input_cap="$CAP_IN" \
  --option selfcheck_out="$WORK/out/index/selfcheck.json" \
  --rows "$WORK/proxy-rows.jsonl.gz" --out "$WORK/out/index/stock" > "$WORK/logs/index.log" 2>&1; r=$?
gate "index" $r; export_phase index
case $r in 0) ;; 124) fail 7 "index timeout" ;; 15) fail 15 "index: measured rate too slow" ;; *) fail 12 "index rc=$r" ;; esac

# ------------------------------------------------------------------ score (exit 13)
say "phase score (proxy600 slice index, ECE, paired vs stock RWKV7 v3w)"
R=$WORK/out/index/stock/results.jsonl
timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset score --results "$R" --subset "$SUBSET" \
  > "$WORK/out/index/score.json" || fail 13 "score"
timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset ece --results "$R" --subset "$SUBSET" \
  > "$WORK/out/index/ece.json" || fail 13 "ece"
timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset compare --candidate "$R" --baseline "$PARENT" --subset "$SUBSET" \
  > "$WORK/out/index/compare_stock_v3w.json" || fail 13 "compare"
gate "score" 0; export_phase score
python -X utf8 -m jobs.backbone_screen.summarize "$PARAMS" "$WORK/out" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1 \
  || fail 9 "decision"
sample "$WORK/out/decision.json" "decision"
RC=0
exit 0
