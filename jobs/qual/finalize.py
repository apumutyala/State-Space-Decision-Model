"""Final gate of the qualification job: decides the exit code and whether /output/QUAL_PASS may exist.

python jobs/qual/finalize.py <result.json> <rc> <gates_json> <export_fail 0|1> <elapsed_min> <decision.json> <out_dir>
Prints and exits with the final code. QUAL_PASS is written here and only here, and only when the final code is 0:
a run that reached exit 0 must also have every export verified (else 6) and a VALID decision.json (else 9).
Any exception in this script exits non-zero (Python's 1), so QUAL_PASS is never written by accident.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from scripts.decide_9b import valid_decision  # noqa: E402


def final_code(rc, export_fail, decision_path):
    if rc != 0:
        return rc
    if export_fail:
        return 6
    try:
        ok = valid_decision(json.load(open(decision_path)))
    except Exception:  # noqa: BLE001  missing / unreadable / malformed
        ok = False
    return 0 if ok else 9


def main(argv):
    res, rc, gates, ef, mins, dec, out_dir = argv
    rc, ef = int(rc), int(ef)
    code = final_code(rc, ef, dec)
    decision = None
    if os.path.exists(dec):
        try:
            decision = json.load(open(dec))
        except Exception:  # noqa: BLE001
            decision = "unreadable"
    pass_path = os.path.join(out_dir, "QUAL_PASS")
    if os.path.exists(pass_path):
        os.remove(pass_path)  # never inherit a stale pass marker
    out = {"ok": code == 0, "exit_code": code, "runner_rc": rc, "gates": json.loads(gates),
           "exports_verified": ef == 0, "elapsed_min": int(mins), "decision": decision}
    json.dump(out, open(res, "w"), indent=1)
    if code == 0:
        open(pass_path, "w").write(json.dumps({"exit_code": 0}))
    print(json.dumps(out))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
