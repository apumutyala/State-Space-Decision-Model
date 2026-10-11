#!/bin/bash
# CPU test suites for three stack/qwen35 shas (nupi c3f3a3f9 with the 5e159437 golden, g16-sym 2d913b3e,
# sym-invariant 42388c2c). Trees are git archives under /opt/trees/<name>. Writes junit + logs to /output.
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

# ---------------------------------------------------------------- 1. nupi (c3f3a3f9) + base golden (5e159437)
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

SUITES="${SUITES:-nupi g16 sym}"
G16_RC=0; H_RC=0; S_RC=0
if [[ " $SUITES " == *" g16 "* ]]; then
# ---------------------------------------------------------------- 2. g16-sym (2d913b3e)
(cd "$T/g16" && timeout 900 $PYT --junitxml="$OUT/junit-g16.xml" tests/controlsafe/test_g16_sym.py > "$OUT/g16.log" 2>&1)
G16_RC=$?
log "g16 rc=$G16_RC"
(cd "$T/g16" && SSDM_INV_HEAD=controlsafe.g16_sym_plug:score_fn timeout 600 $PYT --junitxml="$OUT/junit-g16-harness.xml" \
   tests/guarantees/test_reorder_invariance.py > "$OUT/g16-harness.log" 2>&1)
H_RC=$?
log "g16 harness rc=$H_RC"

fi
if [[ " $SUITES " == *" sym "* ]]; then
# ---------------------------------------------------------------- 3. sym-invariant (42388c2c), control
(cd "$T/sym" && timeout 900 $PYT --junitxml="$OUT/junit-sym.xml" tests/test_sym_invariance.py > "$OUT/sym.log" 2>&1)
S_RC=$?
log "sym rc=$S_RC"
fi

# ---------------------------------------------------------------- summary
python - "$OUT" "$G_RC" "$N_RC" "$G16_RC" "$H_RC" "$S_RC" <<'EOF' >> "$OUT/run.log" 2>&1
import hashlib, json, os, sys, xml.etree.ElementTree as ET
out = sys.argv[1]
rcs = dict(zip(["golden", "nupi", "g16", "g16_harness", "sym"], map(int, sys.argv[2:])))
suites = {}
for name in [n for n in ["nupi", "g16", "g16-harness", "sym"] if os.path.isfile(os.path.join(out, f"junit-{n}.xml")) or n == "nupi"]:
    f = os.path.join(out, f"junit-{name}.xml")
    s = {"passed": 0, "failed": 0, "skipped": 0, "failures": [], "skips": []}
    if os.path.isfile(f):
        for tc in ET.parse(f).getroot().iter("testcase"):
            n = f"{tc.get('classname')}::{tc.get('name')}"
            bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            if bad is not None:
                s["failed"] += 1; s["failures"].append({"test": n, "message": (bad.get("message") or "")[:400]})
            elif tc.find("skipped") is not None:
                sk = tc.find("skipped"); s["skipped"] += 1
                s["skips"].append({"test": n, "reason": (sk.get("message") or "")[:300]})
            else:
                s["passed"] += 1
    else:
        s["missing_junit"] = True
    suites[name] = s
res = {"ok": all(v == 0 for v in rcs.values()) and all(s["failed"] == 0 and not s.get("missing_junit") for s in suites.values()),
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
