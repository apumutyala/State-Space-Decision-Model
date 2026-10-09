#!/usr/bin/env bash
# jobs/backbone_screen: STOCK zero-shot screen of an attention-free backbone candidate on the pre-declared proxy600 rows
# (644 rows of the 2,008-row Index DEV subset, jobs/common/dev_proxy600.json), frozen scorer, paired against the stock
# RWKV7-G1k v3w readout (job-fgu2c / art-ivu28, the same 644 rows). Spec: jobs/backbone_screen/JOB_SPEC.md.
# Env: MODE=qrwkv6_32b (recursal/QRWKV6-32B-Instruct-Preview-v0.1) | qrwkv_qwq32b (featherless-ai/QRWKV-QwQ-32B)
#   | rwkv_qwen3_30b_a3b (OpenMOSE/RWKV-Qwen3-30B-A3B-Instruct-cxa07c)
#   | qrwkv6_32b_v3w (readout-artifact check: QRWKV6-32B, arm a = the scout's chatml readout, asserted to reproduce
#     job-6c27k's 13.56, then arm b = the v3w-generic readout on the same rows; paired diff b - a)
#   | qwen36_a3b (STOCK Qwen/Qwen3.6-35B-A3B, hybrid 30 GDN + 10 attention layers, 256-expert MoE; NOT attention-free:
#     a backbone-capability scout; no vendored code, stock transformers 5.17 qwen3_5_moe). m0 readout WITHOUT the prefix
#     fork (job-gen87), screen.primary_subset (a stratified half of proxy600) first, then the rest until the index cap;
#     cost-weighted rate guard on the half; the half is the headline, all 644 scored too if complete)
#   setup: pins + fla guard -> DEV rows + parent sha256 -> pinned snapshot_download -> sha256 of every file -> vendored
#          code == snapshot code -> proxy600 rows
#   index: decision_index run (ReadoutIndexEngine, family from params; fork == recompute canary self-check before row
#          one), live rate guard
#   score: proxy600 slice index + CI, ECE, paired compare vs stock RWKV7 v3w -> decision.json (measurement, no gate)
# Exit codes: 0 measured   2 bad MODE   5 setup   6 export verification   7 phase timeout   9 decision invalid
#   10 snapshot / vendored code / DEV rows / parent identity, or download failed   12 Index kit run failed
#   13 scoring failed   15 readout too slow by MEASUREMENT (jobs/common/rate_guard.py live row rate)   1 finalize
#   16 arm a (two-arm modes) does not reproduce the expected stock index (arm b is not run)
# The marker /output/BACKBONE_SCREEN_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
set -Euo pipefail
MAIN_PID=$$  # jobs/clef_baseline/watchdog.sh beats only while this PID lives
CODE=/opt/ssdm/m1
MODE=${MODE:-}
case "$MODE" in qrwkv6_32b|qrwkv_qwq32b|rwkv_qwen3_30b_a3b|qrwkv6_32b_v3w|qwen36_a3b) ;; *) echo "MODE must be qrwkv6_32b, qrwkv_qwq32b, rwkv_qwen3_30b_a3b, qrwkv6_32b_v3w or qwen36_a3b"; exit 2 ;; esac
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
if os.environ["PKG"]:  # vendored model code (attention-free conversions); "" = a stock transformers architecture
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
if [ -n "$PKG" ]; then
  python -c "import importlib, os, sys; V = importlib.import_module(os.environ['PKG']); V.verify_snapshot_code(*([sys.argv[1], sys.argv[2]] if hasattr(V, 'REPOS') else [sys.argv[1]]))" "$SNAP" "$REPO" \
    || fail 10 "vendored code != snapshot code"
fi
SUBSET=jobs/common/dev_proxy600.json
python -X utf8 -m jobs.common.dev_proxy check --proxy proxy600 > "$WORK/out/index/proxy-check.json" || fail 10 "proxy600 identity"
python -X utf8 -m jobs.common.dev_subset rows --rows /data/slices/dev-rows.jsonl.gz --out "$WORK/proxy-rows.jsonl.gz" \
  --subset "$SUBSET" > "$WORK/out/index/proxy-rows.json" || fail 10 "proxy rows"
NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$WORK/out/index/proxy-rows.json")
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ index (exit 7 / 12 / 15)
CAP_I=$(cap index) && CAP_SC=$(cap score) || fail 5 "job_params"
ONE_LOAD=$(python -c "import json,sys;print(int(bool(json.load(open(sys.argv[1]))['screen'].get('one_load_arms'))))" "$PARAMS") \
  || fail 5 "job_params one_load_arms"
# primary_subset (MODE qwen36_a3b): rows run primary part first, then the rest until the index cap; cost_weights feed the
# rate guard's projection of the primary part; engine_options are passed to the engine as --option k=v
PRIMARY=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['screen'].get('primary_subset') or '')" "$PARAMS") \
  && COSTW=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['screen'].get('cost_weights') or '')" "$PARAMS") \
  || fail 5 "job_params primary_subset"
EOPTS=()
while read -r kv; do [ -n "$kv" ] && EOPTS+=(--option "$kv"); done < <(python -c \
  "import json,sys;[print(f'{k}={v}') for k,v in (json.load(open(sys.argv[1]))['screen'].get('engine_options') or {}).items()]" "$PARAMS")
if [ "$ONE_LOAD" = 1 ]; then
  # both arms in ONE process (one checkpoint load): arm a -> index/stock, arm b -> index/v3w; rate guard over both
  say "phase index (one load: arm a $FAMILY then arm b; $NROWS proxy600 rows each, input cap $CAP_IN; canary self-check per arm; live rate guard)"
  python -m jobs.common.rate_guard run --results "$WORK/out/index/stock/results.jsonl,$WORK/out/index/v3w/results.jsonl" \
    --total $(( 2 * NROWS )) --exclude-startup --cap-min "$CAP_I" --report "$WORK/out/index/rate.json" -- \
    python -X utf8 -m jobs.backbone_screen.arms "$PARAMS" --model-dir "$SNAP" --repo "$REPO" \
    --rows "$WORK/proxy-rows.jsonl.gz" --out-root "$WORK/out/index" --selfcheck-dir "$WORK/out/index" \
    > "$WORK/logs/index.log" 2>&1; r=$?
elif [ -n "$PRIMARY" ]; then
  python -X utf8 -m jobs.backbone_screen.half_subset order --rows "$WORK/proxy-rows.jsonl.gz" --first "$PRIMARY" \
    --out "$WORK/ordered-rows.jsonl.gz" --cost "$COSTW" --weights-out "$WORK/weights.json" \
    > "$WORK/out/index/order.json" || fail 10 "row order"
  NPRIM=$(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))['run_ids']))" "$PRIMARY")
  say "phase index ($FAMILY: primary $NPRIM rows first, then the rest of $NROWS until the cap; input cap $CAP_IN; canary self-check; cost-weighted rate guard on the primary rows)"
  python -m jobs.common.rate_guard run --results "$WORK/out/index/stock/results.jsonl" --total "$NPRIM" \
    --weights "$WORK/weights.json" --exclude-startup --cap-min "$CAP_I" --report "$WORK/out/index/rate.json" -- \
    python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family="$FAMILY" --option model_dir="$SNAP" --option repo="$REPO" --option input_cap="$CAP_IN" "${EOPTS[@]}" \
    --option selfcheck_out="$WORK/out/index/selfcheck.json" \
    --rows "$WORK/ordered-rows.jsonl.gz" --out "$WORK/out/index/stock" > "$WORK/logs/index.log" 2>&1; r=$?
else
  say "phase index ($NROWS proxy600 rows, family $FAMILY, input cap $CAP_IN; canary fork self-check first; live rate guard)"
  python -m jobs.common.rate_guard run --results "$WORK/out/index/stock/results.jsonl" --total "$NROWS" --exclude-startup \
    --cap-min "$CAP_I" --report "$WORK/out/index/rate.json" -- \
    python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine \
    --option family="$FAMILY" --option model_dir="$SNAP" --option repo="$REPO" --option input_cap="$CAP_IN" "${EOPTS[@]}" \
    --option selfcheck_out="$WORK/out/index/selfcheck.json" \
    --rows "$WORK/proxy-rows.jsonl.gz" --out "$WORK/out/index/stock" > "$WORK/logs/index.log" 2>&1; r=$?
fi
gate "index" $r; export_phase index
RI=$r
R=$WORK/out/index/stock/results.jsonl
FDONE=0
if [ -n "$PRIMARY" ]; then
  # complete lines only (a cap kill may cut the last one) -> results.scored.jsonl; which parts finished
  python -X utf8 -m jobs.backbone_screen.half_subset parts --results "$R" --out "$WORK/out/index/stock/results.scored.jsonl" \
    --primary "$PRIMARY" --full "$SUBSET" > "$WORK/out/index/parts.json" || fail 12 "parts"
  say "parts: $(cat "$WORK/out/index/parts.json") (index rc $RI)"
  PDONE=$(python -c "import json,sys;print(int(json.load(open(sys.argv[1]))['primary_done']))" "$WORK/out/index/parts.json")
  FDONE=$(python -c "import json,sys;print(int(json.load(open(sys.argv[1]))['full_done']))" "$WORK/out/index/parts.json")
  # the cap ending the run after the primary rows is the plan; before them it is a failure
  if [ "$PDONE" != 1 ]; then
    case $RI in 124) fail 7 "index timeout before the primary rows finished" ;; 15) fail 15 "index: measured rate too slow for the primary rows" ;; *) fail 12 "index rc=$RI, primary rows incomplete" ;; esac
  fi
  R=$WORK/out/index/stock/results.scored.jsonl
else
  NA=$(python -c "import sys;print(sum(1 for _ in open(sys.argv[1],'rb')))" "$R" 2>/dev/null || echo 0)
  # a one-load run cut short in arm b still scores a complete arm a (below) before it fails
  if [ "$RI" != 0 ] && { [ "$ONE_LOAD" != 1 ] || [ "$NA" -lt "$NROWS" ]; }; then
    case $RI in 124) fail 7 "index timeout" ;; 15) fail 15 "index: measured rate too slow" ;; *) fail 12 "index rc=$RI" ;; esac
  fi
fi

# ------------------------------------------------------------------ score (exit 13)
SC_SUBSET=${PRIMARY:-$SUBSET}
say "phase score (slice index on $SC_SUBSET, ECE, paired vs stock RWKV7 v3w; all proxy600 rows too if complete: $FDONE)"
timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset score --results "$R" --subset "$SC_SUBSET" \
  > "$WORK/out/index/score.json" || fail 13 "score"
timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset ece --results "$R" --subset "$SC_SUBSET" \
  > "$WORK/out/index/ece.json" || fail 13 "ece"
timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset compare --candidate "$R" --baseline "$PARENT" --subset "$SC_SUBSET" \
  > "$WORK/out/index/compare_stock_v3w.json" || fail 13 "compare"
if [ "$FDONE" = 1 ]; then
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset score --results "$R" --subset "$SUBSET" \
    > "$WORK/out/index/score_full.json" || fail 13 "score_full"
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset ece --results "$R" --subset "$SUBSET" \
    > "$WORK/out/index/ece_full.json" || fail 13 "ece_full"
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset compare --candidate "$R" --baseline "$PARENT" --subset "$SUBSET" \
    > "$WORK/out/index/compare_full_stock_v3w.json" || fail 13 "compare_full"
fi
gate "score" 0; export_phase score
if [ "$RI" != 0 ] && [ -z "$PRIMARY" ]; then  # one-load mode, arm a scored above, arm b incomplete
  case $RI in 124) fail 7 "index timeout (arm a scored)" ;; 15) fail 15 "index: measured rate too slow (arm a scored)" ;; *) fail 12 "index rc=$RI (arm a scored)" ;; esac
fi

# ------------------------------------------------------------------ arm b (two-arm modes; exit 16 / 7 / 12 / 13 / 15)
ARM_B=$(python -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['screen'].get('arm_b') or ''))" "$PARAMS")
if [ "$ARM_B" != '""' ]; then
  python -X utf8 -m jobs.backbone_screen.summarize check-arm-a "$PARAMS" "$WORK/out/index/score.json"     > "$WORK/out/index/arm_a_check.json" || fail 16 "arm a does not reproduce: $(cat "$WORK/out/index/arm_a_check.json")"
  say "arm a check: $(cat "$WORK/out/index/arm_a_check.json")"
  B=$(python -c "import json,sys;b=json.loads(sys.argv[1]);print(b['family'], b['backbone_id'], b['v3w_sha256'])" "$ARM_B")
  read -r FAMILY_B BB_B V3W_SHA <<< "$B"
  if [ "$ONE_LOAD" != 1 ]; then  # one-load mode: arm b already ran in the index phase (index/v3w)
  CAP_IB=$(cap index_b) || fail 5 "job_params"
  say "phase index_b ($NROWS rows, family $FAMILY_B, backbone $BB_B, v3w $V3W_SHA; canary fork == recompute first)"
  python -m jobs.common.rate_guard run --results "$WORK/out/index/v3w/results.jsonl" --total "$NROWS" --exclude-startup     --cap-min "$CAP_IB" --report "$WORK/out/index/rate_b.json" --     python -X utf8 -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine     --option family="$FAMILY_B" --option model_dir="$SNAP" --option repo="$REPO" --option input_cap="$CAP_IN"     --option backbone_id="$BB_B" --option v3w_sha256="$V3W_SHA"     --option selfcheck_out="$WORK/out/index/selfcheck_b.json"     --rows "$WORK/proxy-rows.jsonl.gz" --out "$WORK/out/index/v3w" > "$WORK/logs/index_b.log" 2>&1; r=$?
  gate "index_b" $r; export_phase index_b
  case $r in 0) ;; 124) fail 7 "index_b timeout" ;; 15) fail 15 "index_b: measured rate too slow" ;; *) fail 12 "index_b rc=$r" ;; esac
  fi
  say "phase score_b (v3w arm: slice index, ECE, paired vs stock RWKV7 v3w, paired vs arm a)"
  RB=$WORK/out/index/v3w/results.jsonl
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset score --results "$RB" --subset "$SUBSET"     > "$WORK/out/index/score_b.json" || fail 13 "score_b"
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset ece --results "$RB" --subset "$SUBSET"     > "$WORK/out/index/ece_b.json" || fail 13 "ece_b"
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset compare --candidate "$RB" --baseline "$PARENT" --subset "$SUBSET"     > "$WORK/out/index/compare_b_stock_v3w.json" || fail 13 "compare_b"
  timeout ${CAP_SC}m python -X utf8 -m jobs.common.dev_subset compare --candidate "$RB" --baseline "$R" --subset "$SUBSET"     > "$WORK/out/index/compare_b_vs_a.json" || fail 13 "compare_b_vs_a"
  gate "score_b" 0; export_phase score_b
fi
python -X utf8 -m jobs.backbone_screen.summarize "$PARAMS" "$WORK/out" "$WORK/out/decision.json" > "$WORK/logs/decision.log" 2>&1 \
  || fail 9 "decision"
sample "$WORK/out/decision.json" "decision"
RC=0
exit 0
