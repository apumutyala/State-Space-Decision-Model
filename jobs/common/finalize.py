"""Final gate of an m1 batch job (jobs/probe9b, jobs/rwkv7): decides the exit code and whether the PASS marker exists.

python jobs/common/finalize.py <result.json> <rc> <gates_json> <export_fail 0|1> <elapsed_min> <decision.json> <out_dir> \
    <marker> <validator module:function>

Same contract as jobs/qual/finalize.py: the marker is written here and only here, and only when the final code is 0; a
run that reached 0 must also have every export verified (else 6) and a decision the validator accepts (else 9). A stale
marker is deleted first. Any exception exits non-zero (Python's 1), so the marker is never written by accident.
"""
from __future__ import annotations

import importlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def final_code(rc, export_fail, decision_path, validator):
    if rc != 0:
        return rc
    if export_fail:
        return 6
    try:
        ok = validator(json.load(open(decision_path)))
    except Exception:  # noqa: BLE001  missing / unreadable / malformed
        ok = False
    return 0 if ok else 9


def load_validator(spec):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def main(argv):
    res, rc, gates, ef, mins, dec, out_dir, marker, vspec = argv
    rc, ef = int(rc), int(ef)
    code = final_code(rc, ef, dec, load_validator(vspec))
    decision = None
    if os.path.exists(dec):
        try:
            decision = json.load(open(dec))
        except Exception:  # noqa: BLE001
            decision = "unreadable"
    pass_path = os.path.join(out_dir, marker)
    if os.path.exists(pass_path):
        os.remove(pass_path)
    out = {"ok": code == 0, "exit_code": code, "runner_rc": rc, "gates": json.loads(gates),
           "exports_verified": ef == 0, "elapsed_min": int(mins), "decision": decision,
           "verdict": decision.get("verdict") if isinstance(decision, dict) else None}
    sg = os.path.join(out_dir, "SIZE_GUARD.json")  # jobs/common/size_guard.py moved files out of the upload
    if os.path.exists(sg):
        try:
            g = json.load(open(sg))
            out["size_guard"] = {k: g.get(k) for k in ("fits", "ckpt_evicted", "evicted", "left_in_place")}
        except Exception:  # noqa: BLE001
            out["size_guard"] = "unreadable"
    json.dump(out, open(res, "w"), indent=1)
    if code == 0:
        open(pass_path, "w").write(json.dumps({"exit_code": 0}))
    print(json.dumps({k: out[k] for k in ("ok", "exit_code", "runner_rc", "verdict")}))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
