#!/bin/bash
# JEV3-LM CPU job: the CPU suites TESTS (pytest; KEXPR deselects) on the branch tree (/opt/tree), data under /data;
# SSDM_INV_HEAD plugs the canonicalised shim into the evidence harness (tests/guarantees/test_reorder_invariance.py).
set -u
OUT=/output
mkdir -p "$OUT"
(while true; do date -u +%s > "$OUT/heartbeat"; sleep 60; done) &
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }
cp /opt/job/CONTEXT_MANIFEST.json "$OUT/" 2>/dev/null
python -c "import sys, torch, transformers; print(sys.version, torch.__version__, transformers.__version__)" > "$OUT/env.txt" 2>&1
QWEN_TOK=/data/tokenizers/Qwen--Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export SSDM_BPE_TOKENIZER="$QWEN_TOK" SSDM_INV_HEAD="${SSDM_INV_HEAD:-jobs.sym_eval.canon_plug:score_fn}"
JEVPUB=$(ls -d /data/jevbench/* 2>/dev/null | head -1)
D_RC=0
if [ "${DRAW:-0}" = 1 ]; then
  log "draw build (public dir $JEVPUB)"
  timeout 1200 python -X utf8 -I scripts/jev3_typed_draw.py --data /data --out-dir "$OUT/draw" --public-dir "$JEVPUB" \
    > "$OUT/draw.log" 2>&1
  D_RC=$?
  log "draw rc=$D_RC"
fi
P_RC=0
if [ -n "${PREFLIGHT_PARAMS:-}" ]; then
  log "preflight $PREFLIGHT_PARAMS"
  timeout 1500 python -X utf8 -I scripts/cell_preflight.py --params "$PREFLIGHT_PARAMS" --data /data --tokenizer "$QWEN_TOK" \
    --out "$OUT/preflight.json" > "$OUT/preflight.log" 2>&1
  P_RC=$?
  log "preflight rc=$P_RC $(tail -c 600 "$OUT/preflight.log")"
fi
TESTS="${TESTS:-tests/test_jev3_lm.py tests/guarantees/test_reorder_invariance.py tests/test_ev_score_inv.py tests/test_jev2_h2o.py tests/test_jev3_sym.py tests/test_sym_invariance.py tests/test_m2_screen.py tests/test_m2_checkpoint_loader.py tests/test_backend_row_cache.py}"
KEXPR="${KEXPR:-not test_cell_rescore_section}"
log "tests: $TESTS (-k $KEXPR)"
T_RC=0
if [ "$TESTS" != none ]; then
timeout 1800 python -X utf8 -m pytest -v -rs -p no:cacheprovider --junitxml="$OUT/junit.xml" -k "$KEXPR" $TESTS \
  > "$OUT/tests.log" 2>&1
T_RC=$?
fi
log "tests rc=$T_RC"
python - "$OUT" "$D_RC" "$T_RC" "$P_RC" <<'EOF'
import hashlib, json, os, sys
import xml.etree.ElementTree as ET
out, d_rc, t_rc, p_rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
s = {"draw_rc": d_rc, "tests_rc": t_rc, "preflight_rc": p_rc, "passed": 0, "failed": 0, "skipped": 0, "failures": [], "skips": []}
f = os.path.join(out, "junit.xml")
if os.path.isfile(f):
    for tc in ET.parse(f).getroot().iter("testcase"):
        n = f"{tc.get('classname')}::{tc.get('name')}"
        bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        if bad is not None:
            s["failed"] += 1; s["failures"].append({"test": n, "message": (bad.get("message") or "")[:600]})
        elif tc.find("skipped") is not None:
            s["skipped"] += 1; s["skips"].append({"test": n, "reason": (tc.find("skipped").get("message") or "")[:300]})
        else:
            s["passed"] += 1
open(os.path.join(out, "summary.json"), "w").write(json.dumps(s, indent=1))
with open(os.path.join(out, "MANIFEST.jsonl"), "w") as m:
    for root, _, fs in os.walk(out):
        for x in sorted(fs):
            p = os.path.join(root, x)
            if x in ("MANIFEST.jsonl", "heartbeat"):
                continue
            m.write(json.dumps({"path": os.path.relpath(p, out), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                                "bytes": os.path.getsize(p)}) + "\n")
print(json.dumps({k: s[k] for k in ("draw_rc", "tests_rc", "passed", "failed", "skipped")}))
EOF
touch "$OUT/JEV3_LM_CPU_DONE"
log "done"
