# shellcheck shell=bash
# jobs/q9af_v2 mode clefflash (README_clefflash.md): sourced by q9af_v2_job.sh, which provides CODE DATA WORK OUT CK Q9B
# PARAMS PROXY DRY and the lib.sh helpers (say gate fail export_phase cap sample).
#   clef_setup  (end of setup, before any GPU phase): Clef stack in /opt/clefenv, the Clef-flash snapshot against its LFS
#               pins (baselines.clef_backend.verify_snapshot), the v3w readout identity built from the shipped 9B tokenizer
#   clef_arms   (after X): proxy600, four arms, each a decision_index kit run under the live rate guard
#       C_stock  Clef-flash's OFFICIAL path: baselines.clef_index_engine:ClefIndexEngine (joint head, decide, ssdm
#                encoding, gdn_path=reference), unmodified, in /opt/clefenv (transformers 5.10.2, Clef's model card)
#       C_conv   the same engine + the 8 converted GDNs in place of the attention layers (jobs.q9af_v2.clef_af)
#       L_stock  LM-logit v3w readout (family q9af-probe-v3w: make_v3w qwen3.5-9b + V3WGenericBackend) on Clef-flash
#       L_conv   the same readout on the converted backbone: the fully attention-free system
#     S  paired compares (A: C_conv vs C_stock, B: L_conv vs L_stock) + the four scores, editions 0.2.1 and 0.3
#     R  jobs.q9af_v2.diag (no keep/discard rule: complete measurement -> verdict DIAGNOSTIC, else exit 9)
# Exit codes as the runner: 5 setup, 8 snapshot pins, 10 readout identity, 7 phase timeout, 12 kit, 13 scoring,
# 15 rate guard, 9 incomplete measurement.

V3W_IDENTITY=2e4907ee4302d9d196052fc04c5618e88fc8058446644715345eaad93a3a2543
V3W_VOLUME=jobs/q9af_v2/readout_volume_v3w_proxy600.json
TOKROOT=$DATA/tokenizers
CLEFPY=/opt/clefenv/bin/python
[ -z "$DRY" ] || CLEFPY=$(command -v python)
IDX=$WORK/out/index

clef_setup() {
  mkdir -p "$WORK/out/clef"
  if [ -n "$DRY" ]; then
    $DRY clefsetup "$WORK/out/clef" > "$WORK/logs/clef_setup.log" 2>&1 || fail 5 "dry-run clef setup"
    return 0
  fi
  "$CLEFPY" -c "import json, transformers, torch, torchvision; from baselines.clef_backend import kernel_stack; v = (transformers.__version__, torch.__version__.split('+')[0], torchvision.__version__.split('+')[0]); assert v == ('5.10.2', '2.13.0', '0.28.0'), v; print(json.dumps(kernel_stack()))" \
    > "$WORK/out/clef/clefenv.json" 2> "$WORK/logs/clef_setup.log" || fail 5 "clefenv: $(tail -1 "$WORK/logs/clef_setup.log")"
  "$CLEFPY" -X utf8 -c "import json, sys; from baselines.clef_backend import verify_snapshot as v; r = v(sys.argv[1], 'clef-flash'); json.dump(r, open(sys.argv[2], 'w'), indent=1); sys.exit(0 if r['ok'] else 1)" \
    "$Q9B" "$WORK/out/clef/snapshot-verify.json" >> "$WORK/logs/clef_setup.log" 2>&1 || fail 8 "Clef-flash snapshot does not match its pins"
  python -X utf8 - "$TOKROOT" "$V3W_IDENTITY" "${Q9B_TOK_DIR:-}" > "$WORK/out/clef/v3w_identity.json" 2>> "$WORK/logs/clef_setup.log" <<'EOF' || fail 10 "v3w readout identity"
import glob, hashlib, json, os, sys
from decide.v3w_generic import make_v3w
v = make_v3w("qwen3.5-9b", sys.argv[1])
got = v.identity()["sha256"]
fetched = sorted(glob.glob(os.path.join(sys.argv[3], "*", "tokenizer.json")) + glob.glob(os.path.join(sys.argv[3], "tokenizer.json")))
shipped = os.path.join(sys.argv[1], v.backbone.tokenizer_dir, "tokenizer.json")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
out = {"identity_sha256": got, "want": sys.argv[2], "codes": len(v.codes), "template": v.template_id,
       "shipped_tokenizer_sha256": sha(shipped), "fetched_9b_tokenizer_sha256": [sha(p) for p in fetched]}
print(json.dumps(out, indent=1))
sys.exit(0 if got == sys.argv[2] and len(v.codes) == 255 else 1)
EOF
}

_kit_rc() {  # _kit_rc <phase> <rc>
  case $2 in 0) ;; 124) fail 7 "$1 timeout" ;; 15) fail 15 "$1: measured rate too slow" ;; *) fail 12 "$1 rc=$2" ;; esac
}

_arm() {  # _arm <phase> <dir> <cap> <python> <engine> [kit options...]: one guarded kit run over the proxy600 rows
  local phase=$1 dir=$2 capm=$3 py=$4 engine=$5 r
  shift 5
  [ -e "$IDX/$dir" ] && mv "$IDX/$dir" "$IDX/$dir.stale.$(date +%s)"
  say "phase $phase ($engine $*; proxy600: $NROWS rows)"
  if [ -n "$DRY" ]; then
    $DRY kit "$dir" "$IDX/$dir" > "$WORK/logs/index_$dir.log" 2>&1; r=$?
  else
    python -m jobs.common.rate_guard run --results "$IDX/$dir/results.jsonl" --total "$NROWS" --cap-min "$capm" \
      --check-after-min 3 --exclude-startup --report "$IDX/rate_$dir.json" -- \
      "$py" -X utf8 -m decision_index run --engine "$engine" "$@" \
      --rows "$WORK/data/proxy-rows.jsonl.gz" --out "$IDX/$dir" > "$WORK/logs/index_$dir.log" 2>&1; r=$?
  fi
  gate "$phase" $r
  export_phase "$phase"
  _kit_rc "$phase" $r
}

clef_arms() {
  local r ed sfx arm pair
  NROWS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rows'])" "$IDX/proxy-rows.json")
  NROWS=${NROWS%$'\r'}
  CLEFOPT=(--option model_dir="$Q9B" --option release=clef-flash --option path=decide --option gdn_path=reference)
  CONVOPT=(--option gdn_ckpt="$OUT/gdn_converted.pt" --option converted="$LAYERS")
  V3WOPT=(--option family=q9af-probe-v3w --option model_dir="$Q9B" --option tokenizers_root="$TOKROOT"
          --option v3w_identity="$V3W_IDENTITY")
  # ---- pair A: Clef's joint head (primary)
  _arm C_stock clef_stock "$(cap C_stock)" "$CLEFPY" baselines.clef_index_engine:ClefIndexEngine "${CLEFOPT[@]}"
  _arm C_conv clef_conv "$(cap C_conv)" "$CLEFPY" jobs.q9af_v2.clef_af:ClefAFIndexEngine "${CLEFOPT[@]}" "${CONVOPT[@]}"
  # ---- pair B: the LM-logit v3w readout (rate preflight on the exact v3w fork volume at this job's measured rates)
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
  for arm in stock conv; do
    if [ "$arm" = stock ]; then set -- "$TOKS" "$CALLS" none; else set -- "$TOKS_C" "$CALLS_C" "$LAYERS"; fi
    python -m jobs.common.rate_guard preflight --tok-s "$1" --call-s "$2" --volume "$V3W_VOLUME" --path fork \
      --cap-min "$(cap "L_$arm")" --out "$IDX/preflight_v3w_$arm.json" > "$WORK/logs/preflight_v3w_$arm.log" 2>&1; r=$?
    gate "L:preflight_$arm" $r
    [ $r -eq 0 ] || fail $r "v3w $arm readout projected past its cap"
    if [ "$arm" = stock ]; then
      _arm L_stock v3w_stock "$(cap L_stock)" "$(command -v python)" jobs.common.readout_engine:ReadoutIndexEngine \
        "${V3WOPT[@]}" --option converted=none
    else
      _arm L_conv v3w_conv "$(cap L_conv)" "$(command -v python)" jobs.common.readout_engine:ReadoutIndexEngine \
        "${V3WOPT[@]}" "${CONVOPT[@]}"
    fi
  done
  # ---- S: paired compares + the four scores, edition 0.2.1 (primary) and 0.3 (rebuilt GSM8K omitted, as the Clef jobs)
  say "phase S (proxy600 scoring: pairs A and B, editions 0.2.1 + 0.3)"
  if [ -n "$DRY" ]; then S() { $DRY subset "$@"; }
  else S() { timeout "$(cap S)m" python -X utf8 -m jobs.common.dev_subset "$@" --subset "$PROXY"; }; fi
  for ed in 0.2.1 0.3; do
    if [ "$ed" = 0.2.1 ]; then sfx=""; EDO=(--edition 0.2.1); else sfx=_03; EDO=(--edition 0.3 --omit-gsm8k03); fi
    for pair in clef v3w; do
      S compare --candidate "$IDX/${pair}_conv/results.jsonl" --baseline "$IDX/${pair}_stock/results.jsonl" "${EDO[@]}" \
        > "$IDX/compare_${pair}_conv_vs_stock$sfx.json" 2>> "$WORK/logs/score.err" || fail 13 "compare $pair $ed"
      for arm in stock conv; do
        S score --results "$IDX/${pair}_$arm/results.jsonl" "${EDO[@]}" > "$IDX/score_${pair}_$arm$sfx.json" \
          2>> "$WORK/logs/score.err" || fail 13 "score ${pair}_$arm $ed"
      done
    done
  done
  gate "S:score" 0
  export_phase S
  # ---- R: the diagnostic report (no rule)
  say "phase R (diagnostic report)"
  python -m jobs.q9af_v2.diag "$CK/pipeline/results.json" "$IDX" "$WORK/out/export.json" "$WORK/out/decision.json" \
    > "$WORK/logs/decision.log" 2>&1; r=$?
  gate "R:diagnostic" $r
  [ $r -eq 0 ] || fail 9 "incomplete measurement (rc=$r): $(head -c 300 "$WORK/out/decision.json" 2> /dev/null)"
  cp "$WORK/out/decision.json" "$OUT/decision.json"
  sample "$WORK/out/decision.json" "decision"
}
