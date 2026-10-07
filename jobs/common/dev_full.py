"""Frozen FULL Index DEV expected-row set for G13 (issue #36 review, adopted 2026-10-07).

``g13_capability`` / ``g13_ece`` are evaluated on the FULL Index DEV slice (12,104 rows, ``harness/manifests/
dev-manifest.jsonl.gz``) minus the rows the decision engine cannot serve by construction, frozen by id hash BEFORE any
arm is scored. The 2,008-row ``dev_subset_m1`` remains a screen only.

What is excluded, and why it is not after-the-fact. A row is excluded iff ``decide.Decider`` refuses it BEFORE any model
forward: request validation, the interface limits (``max_input_tokens`` 16,384 = the Clef ``encode_record`` default,
"no truncation"; questions / options caps) and the backend's ``plan()`` (tokenisation only). That outcome depends only
on the row, the tokenizer + prompt template and the adapter limits, never on model outputs, so it is the same for every
arm on one parent lineage and can be fixed before scoring. On the 2,008-row subset (job-5cmat) this is exactly the 7
BRIGHT / ToolRet rows (longest model input up to 28k tokens > 16,384). Rows that fail AT scoring time (e.g. the
POP909-CL CUDA OOM, ``decide.backends.state_fork.KV_STATE_TOKEN_BUDGET``) are NEVER excluded: they are bugs to fix, and
until fixed they make G13 non-evaluable (fail-closed).

Exactness: ``freeze`` runs the real ``Decider`` over a :class:`PlanOnly` wrapper of the real backend whose ``score()``
raises a sentinel, so a row reaches the sentinel iff the real decider would start scoring it.

    python -X utf8 -m jobs.common.dev_full verify --frozen <frozen.json> [--sha256 <pin>]

``freeze`` itself runs inside the parent-arm job (``jobs.m2_parent_arm.score_parent --index-frozen-out``) with the
parent's own backend, before that job scores anything; its file sha256 is then pinned in ``jobs/m2_pilot/
job_params.json`` (``index_frozen_sha256``) and every child run verifies it (``train.m2_gates --index-frozen``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from jobs.common.dev_subset import ids_sha256, manifest_hashes, read_manifest

SCHEMA = "ssdm.index.dev_full_frozen.v1"
RULE = ("excluded iff decide.Decider refuses the row before any model forward (status exhausted/unsupported: "
        "max_input_tokens 16,384 with no truncation, questions/options caps, tokenisation refusals); computed with the "
        "comparator lineage's backend plan() before any arm is scored; scoring-time failures are never excluded")


class FrozenMismatch(ValueError):
    """The frozen manifest, its pin, or the rows do not agree."""


class _Reached(Exception):
    """Sentinel: the decider got past every pre-forward check and called score()."""


class PlanOnly:
    """A backend wrapper that plans exactly like ``backend`` and stops at ``score()``."""

    name = "plan_only"

    def __init__(self, backend):
        self.backend = backend
        self.reached = False

    def revision(self):
        return self.backend.revision()

    def plan(self, request):
        return self.backend.plan(request)

    def score(self, request, check_deadline):
        self.reached = True
        raise _Reached()


EXCLUDABLE_STATUSES = ("exhausted", "unsupported")  # allow-list: anything else is a bug, never an exclusion


def classify(decider, rows: Iterable[dict], model: str = "ssdm-freeze") -> dict[str, dict[str, Any]]:
    """run_id -> {"supported": bool, "status", "code", "message"} for every row, through ``decider``'s own limits.
    Only ``EXCLUDABLE_STATUSES`` may be excluded; any other pre-forward outcome (``internal_error`` from a ``plan()``
    bug, ``invalid_request``, ...) raises ``FrozenMismatch``."""
    from decide import Decider

    po = PlanOnly(decider.backend)
    d = Decider(po, enable_v0_1_deltas=getattr(decider, "enable_v0_1_deltas", False),
                backend_max_input_tokens=getattr(decider, "backend_max_input_tokens", None))
    out = {}
    for r in rows:
        po.reached = False
        resp = d.decide({"model": model, "state": r["state"], "questions": r["questions"]})
        ss = resp.get("ssdm") or {}
        err = ss.get("error") or {}
        if po.reached:
            out[r["_evaluation"]["run_id"]] = {"supported": True}
        elif ss.get("status") not in EXCLUDABLE_STATUSES:
            raise FrozenMismatch(f"row {r['_evaluation']['run_id']}: pre-forward outcome {ss.get('status')}/"
                                 f"{err.get('code')} is not an excludable structural refusal (fix the bug)")
        else:
            out[r["_evaluation"]["run_id"]] = {"supported": False, "status": ss.get("status"), "code": err.get("code"),
                                               "message": err.get("message")}
    return out


def _dev_ids(expected_ids=None) -> set[str]:
    return set(expected_ids) if expected_ids is not None else {e["run_id"] for e in read_manifest()}


def freeze(rows: list[dict], decider, planner: dict | None = None, expected_ids=None) -> dict[str, Any]:
    """The frozen manifest over the FULL DEV rows (refuses a partial or foreign row set). ``expected_ids``: tests only
    (default: the committed DEV manifest's run_ids)."""
    want = _dev_ids(expected_ids)
    got = [r["_evaluation"]["run_id"] for r in rows]
    if len(set(got)) != len(got):
        raise FrozenMismatch("duplicate run_ids in the DEV rows")
    if set(got) != want:
        raise FrozenMismatch(f"rows are not the full DEV slice: {len(set(got) - want)} foreign, "
                             f"{len(want - set(got))} missing of {len(want)}")
    cls = classify(decider, rows)
    excl = sorted(i for i, c in cls.items() if not c["supported"])
    incl = sorted(i for i, c in cls.items() if c["supported"])
    out = {"schema": SCHEMA, "rule": RULE, **manifest_hashes(), "n_rows": len(rows), "n_excluded": len(excl),
           "n_included": len(incl), "excluded_run_ids_sha256": ids_sha256(excl), "included_run_ids_sha256": ids_sha256(incl),
           "excluded": [{"run_id": i, **{k: cls[i][k] for k in ("status", "code", "message")}} for i in excl],
           "planner": planner or {"revision_id": decider.revision().get("revision_id")}}
    return out


def check_backend_accepts(decider, rows: list[dict]) -> dict[str, Any]:
    """$0 CPU plan-only check (tokenisation only, no forward): ``decider`` (e.g. the CHILD's backend) must accept
    every row of the frozen expected set. A row the parent accepted but the child refuses would make G13 permanently
    non-evaluable after a paid gate, so it is refused here, before scoring."""
    cls = classify(decider, rows)
    bad = sorted(i for i, c in cls.items() if not c["supported"])
    if bad:
        raise FrozenMismatch(f"{len(bad)} frozen expected rows are refused by this backend before any forward "
                             f"(e.g. {bad[:3]}: {cls[bad[0]].get('code')})")
    return {"n_rows": len(cls), "all_supported": True}


def write_frozen(path: str | Path, frozen: dict) -> str:
    b = json.dumps(frozen, indent=1, sort_keys=True).encode("utf-8")
    Path(path).write_bytes(b)  # bytes: the pinned sha256 is of the file, on every OS
    return hashlib.sha256(b).hexdigest()


def load_frozen(path: str | Path, sha256: str | None) -> dict:
    """Read + verify: file sha256 == the pin, the excluded hash recomputes, and it was built on the committed DEV
    manifest. A missing pin is refused (the set must be pre-declared)."""
    if not sha256:
        raise FrozenMismatch("no pinned sha256 for the frozen DEV row set (index_frozen_sha256)")
    b = Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha256:
        raise FrozenMismatch("frozen DEV manifest sha256 != the pinned one")
    d = json.loads(b.decode("utf-8"))
    if d.get("schema") != SCHEMA:
        raise FrozenMismatch(f"frozen manifest schema {d.get('schema')!r} != {SCHEMA}")
    if ids_sha256([e["run_id"] for e in d["excluded"]]) != d["excluded_run_ids_sha256"]:
        raise FrozenMismatch("excluded run_ids do not match excluded_run_ids_sha256")
    if manifest_hashes()["dev_manifest_sha256_uncompressed"] != d["dev_manifest_sha256_uncompressed"]:
        raise FrozenMismatch("frozen set built on another DEV manifest")
    return d


def select_rows(rows: list[dict], frozen: dict, expected_ids=None) -> list[dict]:
    """The G13 expected rows: full DEV minus the frozen exclusions; refuses rows that do not reproduce the hashes."""
    ex = {e["run_id"] for e in frozen["excluded"]}
    ids = [r["_evaluation"]["run_id"] for r in rows]
    want = _dev_ids(expected_ids)
    if set(ids) != want or len(ids) != len(want):
        raise FrozenMismatch(f"Index rows are not the full DEV slice ({len(set(ids) & want)} of {len(want)})")
    keep = [r for r in rows if r["_evaluation"]["run_id"] not in ex]
    if ids_sha256([r["_evaluation"]["run_id"] for r in keep]) != frozen["included_run_ids_sha256"]:
        raise FrozenMismatch("included rows do not match included_run_ids_sha256")
    return keep


def identity(frozen: dict) -> dict[str, Any]:
    return {k: frozen[k] for k in ("schema", "dev_manifest_sha256_uncompressed", "excluded_run_ids_sha256",
                                   "included_run_ids_sha256", "n_excluded", "n_included")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.common.dev_full")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--frozen", required=True)
    v.add_argument("--sha256", default=None)
    a = ap.parse_args(argv)
    try:
        sha = a.sha256 or hashlib.sha256(Path(a.frozen).read_bytes()).hexdigest()
        d = load_frozen(a.frozen, sha)
    except FrozenMismatch as e:
        print(f"frozen: {e}", file=sys.stderr)
        return 10
    print(json.dumps({**identity(d), "sha256": sha, "excluded": [e["run_id"] for e in d["excluded"]]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
