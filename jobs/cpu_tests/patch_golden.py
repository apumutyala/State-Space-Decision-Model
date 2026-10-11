"""Fill tests/test_nupi.py's BASE_LOSSES_HEX (None at c3f3a3f9) with the golden list, in the job's copy only.
No-op when the tree already carries the recorded golden (e.g. nupi 54d34df9: ``BASE_LOSSES_HEX = [...]``)."""
import json
import re
import sys

test, golden = sys.argv[1], sys.argv[2]
hexes = json.load(open(golden))
assert isinstance(hexes, list) and len(hexes) == 12 and all(isinstance(h, str) for h in hexes), hexes
src = open(test, encoding="utf-8").read()
old = "BASE_LOSSES_HEX = None  # filled in below by the recorded receipt"
if src.count(old) == 1:
    open(test, "w", encoding="utf-8", newline="").write(src.replace(old, "BASE_LOSSES_HEX = " + json.dumps(hexes)))
    print("patched", test, hexes)
elif re.search(r"^BASE_LOSSES_HEX\s*=\s*[\[(]", src, re.M):
    same = [h for h in hexes if h in src]
    print(f"golden already present in {test}; no-op ({len(same)}/12 of the in-job golden hexes appear verbatim)")
else:
    sys.exit(f"BASE_LOSSES_HEX: neither the None placeholder nor a recorded golden found in {test}")
