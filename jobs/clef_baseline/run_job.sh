#!/usr/bin/env bash
# Clef baseline job entry point (see JOB_SPEC.md). Env: RELEASE=clef-flash|clef, JOB=A|B.
#   A: probes + Index DEV (decide + native arm) + score/ECE/latency + safety_eval (no GCG)
#   B: safety_eval GCG-lite (HP_GATE; 30 steer + 15 hijack), resumable from $GMN_CHECKPOINT_DIR
# Exit codes: 0 ok | 2 bad env | 10 download/verify | 11 parity | 12 Index run | 13 scoring |
#             14 safety (A) | 15 GCG (B) | 16 report
set -uo pipefail
RELEASE="${RELEASE:?RELEASE must be clef-flash or clef}"
JOB="${JOB:?JOB must be A or B}"
case "$RELEASE" in
  clef-flash) REPO=Cloudflare/clef-flash; REV=17f0b0ad64efb65d273590632833508766b2aae6 ;;
  clef)       REPO=Cloudflare/clef;       REV=2f3de3dd85f379784083b0814d997ab627200f0c ;;
  *) echo "unknown RELEASE $RELEASE"; exit 2 ;;
esac
cd /app
OUT="${GMN_OUTPUT_DIR:-/output}/${RELEASE}-job${JOB}"
CK="${GMN_CHECKPOINT_DIR:-/tmp/ck}/${RELEASE}-job${JOB}"
mkdir -p "$OUT/env" "$OUT/index" "$CK"
export SSDM_DATA=/data PYTHONUTF8=1
hb() { date -u +%FT%TZ > "$OUT/heartbeat"; }
stage() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$OUT/stages.log"; hb; \
  curl -fsS -X POST "${GMN_METADATA_URL:-http://127.0.0.1:9}/v1/sample" -H 'content-type: application/json' \
    -d "{\"path\":\"$OUT/stages.log\",\"kind\":\"text\",\"label\":\"$*\"}" >/dev/null 2>&1 || true; }
fail() { echo "{\"ok\": false, \"exit\": $1, \"stage\": \"$2\", \"release\": \"$RELEASE\", \"job\": \"$JOB\"}" > "${GMN_RESULT_PATH:-$OUT/result.json}"; exit "$1"; }

stage "S0 env + download $REPO@$REV"
( nvidia-smi; python -VV; pip freeze; git -C /app rev-parse HEAD 2>/dev/null; cat /app/CONTEXT_MANIFEST.json ) > "$OUT/env/environment.txt" 2>&1
sha256sum -c /app/jobs/clef_baseline/data.sha256 > "$OUT/env/data-verify.txt" 2>&1 || fail 10 data_verify
SNAP=$(python - <<EOF
from huggingface_hub import snapshot_download
print(snapshot_download("$REPO", revision="$REV"))
EOF
) || fail 10 download
python -X utf8 -c "import json,sys; from baselines.clef_backend import verify_snapshot as v; r=v('$SNAP','$RELEASE'); json.dump(r, open('$OUT/env/snapshot-verify.json','w'), indent=1); sys.exit(0 if r['ok'] else 1)" || fail 10 verify

if [ "$JOB" = "A" ]; then
  stage "S1 probes (parity, encodings, G8, G2b, memory)"
  python -X utf8 -m baselines.clef_probes --model-dir "$SNAP" --release "$RELEASE" \
      --dev-rows /data/slices/dev-rows.jsonl.gz --out "$OUT/probes.json" --n-parity 64 --grad-probe \
      > "$OUT/probes.log" 2>&1
  rc=$?; [ $rc -eq 0 ] || fail $([ $rc -eq 11 ] && echo 11 || echo 10) probes

  stage "S2 Index DEV, decide path (12,104 rows, batch 1, sequential)"
  python -X utf8 -m decision_index run --engine baselines.clef_index_engine:ClefIndexEngine \
      --option model_dir="$SNAP" --option release="$RELEASE" --option path=decide \
      --rows /data/slices/dev-rows.jsonl.gz --out "$OUT/index/decide" > "$OUT/index/decide.log" 2>&1 || fail 12 index_decide
  python -X utf8 - <<EOF || fail 12 index_native_rows
import gzip, json
p = json.load(open("$OUT/probes.json"))
keep = set(p["native_arm_run_ids"]) | set(p["parity_sample_run_ids"])
with gzip.open("/data/slices/dev-rows.jsonl.gz", "rt", encoding="utf-8") as fi, \
     gzip.open("$OUT/index/native-rows.jsonl.gz", "wt", encoding="utf-8") as fo:
    for line in fi:
        if json.loads(line)["_evaluation"]["run_id"] in keep:
            fo.write(line)
EOF
  stage "S2b Index DEV, Clef native systemone on differing/truncated rows + parity sample"
  python -X utf8 -m decision_index run --engine baselines.clef_index_engine:ClefIndexEngine \
      --option model_dir="$SNAP" --option release="$RELEASE" --option path=native --option verify=false \
      --rows "$OUT/index/native-rows.jsonl.gz" --out "$OUT/index/native" > "$OUT/index/native.log" 2>&1 || fail 12 index_native
  python -X utf8 jobs/clef_baseline/summarize.py composite --decide "$OUT/index/decide/results.jsonl" \
      --native "$OUT/index/native/results.jsonl" --out "$OUT/index/native_composite/results.jsonl" || fail 12 composite

  stage "S3 harness score + ECE"
  python -X utf8 -I -m harness score --results "$OUT/index/decide/results.jsonl" --slice dev > "$OUT/index/score_decide.json" || fail 13 score_decide
  python -X utf8 -I -m harness score --results "$OUT/index/native_composite/results.jsonl" --slice dev > "$OUT/index/score_native_composite.json" || fail 13 score_native
  python -X utf8 -I -m harness compare --candidate "$OUT/index/decide/results.jsonl" --baseline "$OUT/index/native_composite/results.jsonl" --slice dev > "$OUT/index/compare_decide_vs_native.json" || fail 13 compare
  python -X utf8 -I -m harness ece --results "$OUT/index/decide/results.jsonl" --slice dev > "$OUT/index/ece_decide.json" || fail 13 ece

  stage "S4 safety_eval (test split, --final, no GCG)"
  python -X utf8 -m baselines.run_clef_safety --model-dir "$SNAP" --release "$RELEASE" --split test --final \
      --skip-gcg --out "$OUT/safety" > "$OUT/safety.log" 2>&1 || fail 14 safety
else
  stage "S4b safety_eval GCG-lite (HP_GATE, 30 steer + 15 hijack), resumable"
  python -X utf8 -m baselines.run_clef_safety --model-dir "$SNAP" --release "$RELEASE" --split test --final \
      --gcg-only --gcg-steer 30 --gcg-hijack 15 --gcg-hp gate --out "$CK/safety" > "$OUT/gcg.log" 2>&1 || fail 15 gcg
  cp -r "$CK/safety" "$OUT/safety"
fi

stage "S5 report"
python -X utf8 jobs/clef_baseline/summarize.py report --out-dir "$OUT" --release "$RELEASE" --job "$JOB" \
    > "$OUT/result.json" || fail 16 report
cp "$OUT/result.json" "${GMN_RESULT_PATH:-$OUT/result.json}" 2>/dev/null || true
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
stage "done"
exit 0
