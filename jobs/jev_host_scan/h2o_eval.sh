#!/usr/bin/env bash
# jobs/jev_host_scan mode h2o_eval: EV-PREREG-H2O-HOST eval-only arms, H2O-Lightning-4B at the PINNED revision
# (job_params_h2o_eval.json "h2o_eval".revision; any other sha fails closed) on the 231 public JevBench items, every arm
# paired in this one job. Spec: jobs/jev_host_scan/JOB_SPEC.md (section "Mode h2o_eval"); code: h2o_eval.py.
#   setup:     env pins -> public items sha256 -> the held-out jev2a slice (ids sha256 == config, else exit 10) ->
#              snapshot_download(revision=pinned) -> revision + file pins (exit 10) -> LICENSE / NOTICE / README copied
#   run:       slice raw log-probs + T fits, control (the revision's own shim), c_symfork, G1 perturbation probe;
#              under jobs.common.rate_guard (cap = the run phase)
#   summarize: derived arms (a_T, a_Ttype, a_Tfam, b_native, b_shimcal) -> jevkit score per arm -> decision.json
# Exit codes: 0 measured   5 setup   6 export verification   7 run failed (no control)   9 decision invalid
#   10 identity (items / slice / revision / pinned files)   13 summarize failed   1 finalize
# The marker /output/H2O_EVAL_DONE is written ONLY by jobs/common/finalize.py, and only when its final code is 0.
set -Euo pipefail
MAIN_PID=$$  # jobs/clef_baseline/watchdog.sh beats only while this PID lives
CODE=/opt/ssdm/m1
export JEV_SCAN_PARAMS=jobs/jev_host_scan/job_params_h2o_eval.json
PARAMS=$CODE/$JEV_SCAN_PARAMS
OUT=${GMN_OUTPUT_DIR:-/output}
RESULT=${GMN_RESULT_PATH:-$OUT/result.json}
WORK=/tmp/jev-h2o-eval
META=${GMN_METADATA_URL:-http://169.254.42.1}
HB_EVERY_S=60 STALL_S=1500
WATCH_EXTRA=$WORK
mkdir -p "$OUT/phases" "$WORK/out" "$WORK/logs"
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
MARKER=H2O_EVAL_DONE
VALIDATOR=jobs.jev_host_scan.h2o_eval:valid
D=$WORK/out
finish() {
  local rc=${RC:-$?}
  kill "${WD_PID:-}" 2> /dev/null || true
  export_phase final
  python "$CODE/jobs/common/finalize.py" "$RESULT" "$rc" "$GATES" "$EXPORT_FAIL" "$(( ($(date +%s) - T0) / 60 ))" \
    "$D/decision.json" "$OUT" "$MARKER" "$VALIDATOR"
  local final=$?
  say "EXIT $final (runner rc $rc)"
  exit "$final"
}
fail() { RC=$1; say "FAIL($1): $2"; exit "$1"; }
hb
watchdog & WD_PID=$!
trap finish EXIT
jp() {  # jp <key> [<key> ...]: a job_params value (a list joined by commas)
  python - "$PARAMS" "$@" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    d = d[k]
print(",".join(d) if isinstance(d, list) else d)
EOF
}

# ------------------------------------------------------------------ setup (exit 5 / 10)
say "setup"
cp /opt/ssdm/CONTEXT_MANIFEST.json "$D/" || fail 5 "no context manifest"
nvidia-smi > "$D/nvidia-smi.txt" || fail 5 "nvidia-smi"
python - > "$D/env.json" <<'EOF' || fail 5 "env pins"
import json, torch, transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
assert torch.cuda.is_available()
import fla  # noqa: F401  (Qwen3.5 Gated DeltaNet kernels)
from importlib import metadata
import huggingface_hub
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "transformers": transformers.__version__, "fla": metadata.version("flash-linear-attention"),
                  "huggingface_hub": huggingface_hub.__version__}))
EOF
python -I /opt/ssdm/m1/jobs/common/fla_guard.py --out "$D/fla_guard.json" > "$WORK/logs/fla_guard.log" 2>&1 || fail 5 "fla #640 guard: $(tail -1 "$WORK/logs/fla_guard.log")"
python -X utf8 -m jevkit items > "$D/items-verify.json" 2> "$WORK/logs/items.log" || fail 10 "public items sha256"
REC=/data/jev2a/records.jsonl.gz
python -X utf8 -m jobs.jev_host_scan.h2o_eval slice --records "$REC" --out "$D/slice.json" > "$D/slice-verify.json" \
  2> "$WORK/logs/slice.log" || fail 10 "held-out slice ids sha256 != config: $(cat "$D/slice-verify.json") $(tail -1 "$WORK/logs/slice.log")"
python - "$D/slice.json" "$CODE/$(jp h2o_eval slice ids_file)" <<'EOF' || fail 10 "slice ids != committed ids file"
import json, sys
s = json.load(open(sys.argv[1]))
want = open(sys.argv[2], encoding="utf-8").read().split()
assert sorted(i for v in s["families"].values() for i in v) == want, "slice ids differ from the committed file"
EOF
REPO=$(jp h2o_eval repo) REV=$(jp h2o_eval revision) PATS=$(jp h2o_eval allow_patterns)
DLCAP=$(jp download_cap_minutes)
say "download $REPO@$REV (cap ${DLCAP}m)"
TD=$(date +%s)
SNAP=$(timeout "${DLCAP}m" python - "$REPO" "$REV" "$PATS" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=16, allow_patterns=sys.argv[3].split(",")))
EOF
) || fail 5 "download"
SNAP=$(echo "$SNAP" | tail -1)
echo "{\"download_s\": $(( $(date +%s) - TD )), \"snapshot\": \"$SNAP\"}" > "$D/download.json"
python -X utf8 -m jobs.jev_host_scan.h2o_eval verify --model-dir "$SNAP" > "$D/verify.json" 2> "$WORK/logs/verify.err" \
  || fail 10 "pinned revision / files: $(tail -1 "$WORK/logs/verify.err")"
python - "$REPO" "$REV" > "$D/hf_revision.json" <<'EOF' || fail 10 "HF revision sha != pinned"
import json, sys
from huggingface_hub import HfApi
info = HfApi().model_info(sys.argv[1], revision=sys.argv[2])
print(json.dumps({"repo": sys.argv[1], "requested": sys.argv[2], "resolved_sha": info.sha,
                  "last_modified": str(getattr(info, "last_modified", None)),
                  "license": getattr(getattr(info, "card_data", None), "license", None)}))
assert info.sha == sys.argv[2], (info.sha, sys.argv[2])
EOF
mkdir -p "$D/license"
for f in LICENSE NOTICE NOTICE.md README.md; do
  [ -f "$SNAP/$f" ] && cp -L "$SNAP/$f" "$D/license/$f"
done
[ -f "$D/license/LICENSE" ] || fail 10 "no LICENSE in the snapshot"
ls "$D/license" > "$D/license/FILES.txt"
python - "$PARAMS" > "$D/price.json" <<'EOF' || fail 5 "price"
import json, sys
print(json.dumps({"price": json.load(open(sys.argv[1]))["h2o_eval"]["price_class"], "price_class": "qwen35_4b"}))
EOF
gate "setup" 0
export_phase setup

# ------------------------------------------------------------------ run (exit 7 only when the control is missing)
CAP_RUN=$(cap run)
say "phase run (cap ${CAP_RUN}m)"
python -m jobs.common.rate_guard run --results "$D/progress.txt" --total 1462 --exclude-startup \
  --cap-min "$CAP_RUN" --check-after-min 4 --report "$D/rate.json" -- \
  python -X utf8 -m jobs.jev_host_scan.h2o_eval run --model-dir "$SNAP" --slice "$D/slice.json" --records "$REC" --out "$D" \
  > "$WORK/logs/run-h2o.log" 2>&1 < /dev/null; r=$?
say "run rc $r"
gate "run" "$r"
[ -s "$D/arms/h2o/preds.jsonl" ] && [ -s "$D/raw_public.jsonl" ] && [ -s "$D/temperature_fit.json" ] \
  || { export_phase run; fail 7 "control / slice fit missing (rc $r): $(tail -3 "$WORK/logs/run-h2o.log")"; }
export_phase run

# ------------------------------------------------------------------ summarize (exit 13 / 9)
CAP_SUM=$(cap summarize) SCORE_MIN=$(jp score_cap_minutes)
say "phase summarize (cap ${CAP_SUM}m)"
SUM_END=$(( $(date +%s) + 60 * CAP_SUM ))
python -X utf8 -m jobs.jev_host_scan.h2o_eval derive --out "$D" --model-dir "$SNAP" > "$WORK/logs/derive.log" 2>&1 \
  || fail 13 "derive: $(tail -1 "$WORK/logs/derive.log")"
for P in "$D"/arms/*/preds.jsonl; do
  A=$(dirname "$P")
  timeout "${SCORE_MIN}m" python -X utf8 -m jevkit score --public --pred "$P" --price "$D/price.json" \
    --out "$A/report.json" > "$A/score.log" 2>&1 < /dev/null || say "score failed: $A ($(tail -1 "$A/score.log"))"
done
LEFT=$(( SUM_END - $(date +%s) ))
[ "$LEFT" -gt 30 ] || fail 13 "no time left to summarize"
timeout "$LEFT" python -X utf8 -m jobs.jev_host_scan.h2o_eval summarize --root "$D" --out "$D/decision.json" \
  > "$WORK/logs/summarize.log" 2>&1 || fail 13 "summarize: $(tail -1 "$WORK/logs/summarize.log")"
gate "summarize" 0
sample "$D/decision.json" "decision"
RC=0
exit 0
