#!/bin/bash
# CPU test suites for stack/qwen35 shas (nupi with the 5e159437 golden, g16-sym, sym-invariant; shas in
# CONTEXT_MANIFEST.json), SUITES picks them (nupi g16 sym symall). Trees are git archives under /opt/trees/<name>;
# Qwen3.5-4B tokenizer files (no weights) under /opt/tokenizers. Writes junit + logs to /output.
set -u
OUT=/output
mkdir -p "$OUT"
(while true; do date -u +%s > "$OUT/heartbeat"; sleep 60; done) &
HB=$!
T=/opt/trees
J=/opt/job
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }
cp "$J/CONTEXT_MANIFEST.json" "$OUT/" 2>/dev/null
python -c "import sys, torch, transformers; print(sys.version); print('torch', torch.__version__, 'transformers', transformers.__version__)" > "$OUT/env.txt" 2>&1
pip freeze > "$OUT/pip-freeze.txt" 2>&1
echo "GMN_CPU_LIMIT=${GMN_CPU_LIMIT:-} OMP_NUM_THREADS=${OMP_NUM_THREADS:-}" >> "$OUT/env.txt"
PYT="python -X utf8 -m pytest -v -rs -p no:cacheprovider"

SUITES="${SUITES:-nupi g16 sym}"
G_RC=0; N_RC=0
QWEN_TOK=/opt/tokenizers/Qwen--Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export SSDM_TOKENIZERS="${SSDM_TOKENIZERS:-/opt/tokenizers}" SSDM_BPE_TOKENIZER="$QWEN_TOK"
ls -la "$QWEN_TOK" >> "$OUT/env.txt" 2>&1
# evidence lane's adversarial suites (claude/ev-sym-check), overlaid unchanged on the g16 and sym trees
for tr in g16 sym g16old g16v2 symold; do
  [ -d "$T/$tr" ] || continue
  mkdir -p "$T/$tr/tests/guarantees"
  cp -n "$T/ev/tests/guarantees/"test_*_adversarial.py "$T/$tr/tests/guarantees/" 2>/dev/null
done
rm -f "$T/sym/tests/guarantees/test_g16_sym_adversarial.py" "$T/symold/tests/guarantees/test_g16_sym_adversarial.py"
SYM_TESTS="tests/test_sym_invariance.py tests/test_sym_perturb_eval.py tests/test_sym_g15_bpe.py tests/test_policy_actions.py"
if [[ " $SUITES " == *" nupi "* ]]; then
# ---------------------------------------------------------------- 1. nupi + base golden (5e159437)
log "golden: base 5e159437"
(cd "$T/base" && PYTHONPATH="$T/base" timeout 600 python -X utf8 "$J/golden.py" > "$OUT/golden_base_losses.json" 2> "$OUT/golden.stderr")
G_RC=$?
log "golden rc=$G_RC $(cat "$OUT/golden_base_losses.json" 2>/dev/null)"
if [ "$G_RC" = 0 ]; then
  python "$J/patch_golden.py" "$T/nupi/tests/test_nupi.py" "$OUT/golden_base_losses.json" >> "$OUT/run.log" 2>&1 || G_RC=97
fi
NUPI_TESTS=$(cd "$T/nupi" && grep -lE "nupi|pi_control|train\.constraints|train/constraints" tests -r --include='test_*.py' | sort | tr '\n' ' ')
log "nupi tests: $NUPI_TESTS"
(cd "$T/nupi" && timeout 1200 $PYT --junitxml="$OUT/junit-nupi.xml" $NUPI_TESTS > "$OUT/nupi.log" 2>&1)
N_RC=$?
log "nupi rc=$N_RC"
fi

G16_RC=0; H_RC=0; S_RC=0; A_RC=0
if [[ " $SUITES " == *" g16 "* ]]; then
# ---------------------------------------------------------------- 2. g16-sym (on the sym tree it is rebased onto)
(cd "$T/g16" && timeout 1200 $PYT --junitxml="$OUT/junit-g16.xml" tests/controlsafe/test_g16_sym.py \
   tests/guarantees/test_g16_sym_adversarial.py tests/guarantees/test_sym_fork_adversarial.py $SYM_TESTS > "$OUT/g16.log" 2>&1)
G16_RC=$?
log "g16 rc=$G16_RC"
(cd "$T/g16" && SSDM_INV_HEAD=controlsafe.g16_sym_plug:score_fn timeout 600 $PYT --junitxml="$OUT/junit-g16-harness.xml" \
   tests/guarantees/test_reorder_invariance.py > "$OUT/g16-harness.log" 2>&1)
H_RC=$?
log "g16 harness rc=$H_RC"

fi
if [[ " $SUITES " == *" sym "* ]]; then
# ---------------------------------------------------------------- 3. sym-invariant
(cd "$T/sym" && timeout 1200 $PYT --junitxml="$OUT/junit-sym.xml" $SYM_TESTS tests/guarantees/test_sym_fork_adversarial.py \
   > "$OUT/sym.log" 2>&1)
S_RC=$?
log "sym rc=$S_RC"
fi
R_RC=0; RH_RC=0
if [[ " $SUITES " == *" rsc "* ]]; then
# ---------------------------------------------------------------- 3c. stack/rwkv7/sym-contractive (D21 A + B, OFF = d20 parent)
(cd "$T/d20" && PYTHONPATH="$T/d20" timeout 600 python -X utf8 "$T/rsc/tests/rsc_off_golden.py" > "$OUT/rsc_parent_golden.json" 2> "$OUT/rsc_parent_golden.stderr")
log "rsc parent golden rc=$? $(wc -c < "$OUT/rsc_parent_golden.json") bytes"
(cd "$T/rsc" && RSC_PARENT_GOLDEN="$OUT/rsc_parent_golden.json" timeout 1500 $PYT --junitxml="$OUT/junit-rsc.xml"    tests/test_sym_contractive.py tests/test_joint_record_head.py tests/test_g1v3w_train_rows.py tests/test_equivariance.py    tests/test_heads.py tests/test_policy_actions.py tests/test_conformance.py tests/test_m2_gates.py > "$OUT/rsc.log" 2>&1)
R_RC=$?
log "rsc rc=$R_RC"
(cd "$T/rsc" && SSDM_INV_HEAD=heads.inv_adapters:rwkv7_sym_fork timeout 900 $PYT --junitxml="$OUT/junit-rsc-harness.xml"    tests/guarantees/test_reorder_invariance.py > "$OUT/rsc-harness.log" 2>&1)
RH_RC=$?
log "rsc harness rc=$RH_RC"
fi
J2_RC=0
if [[ " $SUITES " == *" jev2 "* ]]; then
# ---------------------------------------------------------------- 3a. jev2-4b (A-PREREG-JEV2-4B package checks)
(cd "$T/jev2" && timeout 900 $PYT --junitxml="$OUT/junit-jev2.xml" tests/test_jev2_4b.py tests/test_backend_row_cache.py \
   > "$OUT/jev2.log" 2>&1)
J2_RC=$?
log "jev2 rc=$J2_RC"
fi
B_RC=0
if [[ " $SUITES " == *" baseline "* ]]; then
# ---------------------------------------------------------------- 3b. pre-fix shas, same tests (informational: env vs change)
for tr in g16old g16v2; do
  [ -d "$T/$tr" ] || continue
  (cd "$T/$tr" && timeout 900 $PYT --junitxml="$OUT/junit-baseline-$tr.xml" tests/controlsafe/test_g16_sym.py \
     tests/guarantees/test_g16_sym_adversarial.py tests/guarantees/test_sym_fork_adversarial.py > "$OUT/baseline-$tr.log" 2>&1)
done
(cd "$T/symold" && timeout 900 $PYT --junitxml="$OUT/junit-baseline-sym.xml" tests/test_sym_invariance.py \
   tests/test_sym_perturb_eval.py tests/guarantees/test_sym_fork_adversarial.py > "$OUT/baseline-sym.log" 2>&1)
log "baseline done"
fi
if [[ " $SUITES " == *" symall "* ]]; then
# ---------------------------------------------------------------- 4. the whole tests/ tree of sym (informational)
(cd "$T/sym" && timeout 1800 $PYT --junitxml="$OUT/junit-symall.xml" tests > "$OUT/symall.log" 2>&1)
A_RC=$?
log "symall rc=$A_RC (informational, not gating)"
fi

# ---------------------------------------------------------------- summary
python - "$OUT" "$G_RC" "$N_RC" "$G16_RC" "$H_RC" "$S_RC" "$J2_RC" "$R_RC" "$RH_RC" <<'EOF' >> "$OUT/run.log" 2>&1
import hashlib, json, os, sys, xml.etree.ElementTree as ET
out = sys.argv[1]
rcs = dict(zip(["golden", "nupi", "g16", "g16_harness", "sym", "jev2", "rsc", "rsc_harness"], map(int, sys.argv[2:])))
suites = {}
want = os.environ.get("SUITES", "nupi g16 sym").split()
for name in [n for n in ["nupi", "g16", "g16-harness", "sym", "jev2", "rsc", "rsc-harness", "baseline-g16old", "baseline-g16v2", "baseline-sym", "symall"]
             if os.path.isfile(os.path.join(out, f"junit-{n}.xml")) or n.split("-")[0] in want]:
    f = os.path.join(out, f"junit-{name}.xml")
    s = {"passed": 0, "failed": 0, "skipped": 0, "failures": [], "skips": []}
    if os.path.isfile(f):
        for tc in ET.parse(f).getroot().iter("testcase"):
            n = f"{tc.get('classname')}::{tc.get('name')}"
            bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            if bad is not None:
                s["failed"] += 1; s["failures"].append({"test": n, "message": (bad.get("message") or "")[:400]})
            elif tc.find("skipped") is not None and "xfail" in (tc.find("skipped").get("type") or "") + (tc.find("skipped").get("message") or ""):
                s.setdefault("xfailed", 0); s["xfailed"] += 1
                s.setdefault("xfails", []).append({"test": n, "reason": (tc.find("skipped").get("message") or "")[:300]})
            elif tc.find("skipped") is not None:
                sk = tc.find("skipped"); s["skipped"] += 1
                s["skips"].append({"test": n, "reason": (sk.get("message") or "")[:300]})
            else:
                s["passed"] += 1
    else:
        s["missing_junit"] = True
    suites[name] = s
res = {"ok": all(v == 0 for v in rcs.values()) and all(s["failed"] == 0 and not s.get("missing_junit")
                                                      for k, s in suites.items()
                                                      if k != "symall" and not k.startswith("baseline")),
       "rc": rcs, "suites": suites}
json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
rp = os.environ.get("GMN_RESULT_PATH")
if rp:
    json.dump({"ok": res["ok"], "counts": {k: {x: v[x] for x in ("passed", "failed", "skipped")} for k, v in suites.items()}}, open(rp, "w"))
with open(os.path.join(out, "MANIFEST.jsonl"), "w") as m:
    for fn in sorted(os.listdir(out)):
        p = os.path.join(out, fn)
        if os.path.isfile(p) and fn not in ("MANIFEST.jsonl", "heartbeat"):
            m.write(json.dumps({"file": fn, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}) + "\n")
print(json.dumps({k: {x: v[x] for x in ("passed", "failed", "skipped")} for k, v in suites.items()}), rcs)
EOF
touch "$OUT/CPU_TESTS_DONE"
log "done"
kill $HB 2>/dev/null
exit 0
