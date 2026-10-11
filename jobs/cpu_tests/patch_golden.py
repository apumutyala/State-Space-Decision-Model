"""Fill tests/test_nupi.py's BASE_LOSSES_HEX (None at c3f3a3f9) with the golden list, in the job's copy only."""
import json
import sys

test, golden = sys.argv[1], sys.argv[2]
hexes = json.load(open(golden))
assert isinstance(hexes, list) and len(hexes) == 12 and all(isinstance(h, str) for h in hexes), hexes
src = open(test, encoding="utf-8").read()
old = "BASE_LOSSES_HEX = None  # filled in below by the recorded receipt"
assert src.count(old) == 1, "BASE_LOSSES_HEX line not found exactly once"
open(test, "w", encoding="utf-8", newline="").write(src.replace(old, "BASE_LOSSES_HEX = " + json.dumps(hexes)))
print("patched", test, hexes)
