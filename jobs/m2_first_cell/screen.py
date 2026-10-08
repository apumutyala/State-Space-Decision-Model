"""Stage ``screen``: a trained cell's Index SCREEN on the pre-declared proxy600 rows (job_params ``"stage": "screen"``).

Replaces the evals phase (temperature fit, dev card, G12, 2,008-row Index) of a training mode with ONE scoring pass:
the child's untempered arm decider (exactly what ``train.m2_gates`` scores the Index stage with) on the 644 proxy rows
(``jobs/common/dev_proxy600.json``, sha pinned in ``jobs.common.dev_proxy``), at the trained input cap. The report
pairs it, with the frozen scorer (``harness.score.decompose`` + ``harness.bootstrap.compare``, case clusters), with
  * the in-job parent arm (stock readout; its ``units-index.json`` over the 2,008 subset, pinned by the arm.json
    sha256 the job verified before training), restricted to the proxy rows, and
  * a REFERENCE cell's child predictions (the current best; job_params ``screen_reference``, default D1 = job-c7pq2
    art-bm3e6): its ``units-child-index.json``, sha-pinned through the artifact chain (artifact tar sha256 -> member
    tgz sha256 -> inner file sha256), restricted to the proxy rows; a reference that does not cover EVERY proxy row
    fails closed (exit 10).
Verdict (pre-declared, ``analysis/proxy_validate.py``: the proxy's 90th-percentile paired-gap error on 1-5 point
pairs is 2.5): DROP if child - reference < -2.5 Index points on the proxy (point estimate), else PROMOTE (to the full
subset + full DEV). Both paired differences are recorded either way. A SCREEN, never a promotion.

    python -X utf8 -m jobs.m2_first_cell.screen estimate  --params P
    python -X utf8 -m jobs.m2_first_cell.screen reference --params P --data $DATA --work W --out ref.json
    python -X utf8 -m jobs.m2_first_cell.screen score     --cell C --ckpt-dir D --model-dir M --records R --train-dir T
        --guard-readiness G --index-rows <subset rows> --resume-dir K --out-dir O [--progress p.jsonl]
    python -X utf8 -m jobs.m2_first_cell.screen report    --params P --index-rows <subset rows> --out-dir O
        --parent-arm A --parent-arm-sha256 S --reference ref.json --elapsed-min N --out screen.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "m2-screen/v1"
EDITION = "0.2.1"
PROXY = "proxy600"
DROP_MARGIN = 2.5  # Index points (proxy p90 paired-gap error, receipts/2026-10-08-B-dev-proxy600)
#: D1's measured child Index wall on the proxy rows (job-c7pq2 units-child-index.json, sum total_wall_ms over the 644
#: proxy ids at the trained input cap): 25.97 min. Fixed: model load + 150k lineage rebuild + plan (no temperature
#: fit, no card): budgeted 15 min (planning number, not measured separately).
D1_PROXY_CHILD_MIN = 25.97
DEFAULT_FIXED_MIN = 15
DEFAULT_REFERENCE = {
    "_note": "D1 = job-c7pq2 (m2_first_cell base), artifact art-bm3e6; the child's Index units from the evals phase "
             "(phases/final.tgz out/units-child-index.json, == x_evals). Chain verified locally by agent B 2026-10-08.",
    "artifact": "art-bm3e6", "job": "job-c7pq2",
    "artifact_sha256": "a405907eb85f3b6bc25f287ef37066dcd192b0b2897e4ec45f336a6fe3580a18",
    "member": "./phases/final.tgz",
    "member_sha256": "c49988af37c60754de5abc5cff686ad7dcb1152a0848b341af53dfb02b3414e7",
    "inner_member": "out/units-child-index.json",
    "units_sha256": "545dbb946c16b2e9975d2bb7a19085edc9fc58b69b2d668d7ace153ded55a6b7",
    "shipped": "screen_reference/units-child-index.json",
    "url_env": "REF_URL"}
REF_KEYS = ("artifact", "artifact_sha256", "member", "member_sha256", "inner_member", "units_sha256", "shipped",
            "url_env")


class ScreenError(RuntimeError):
    def __init__(self, msg: str, code: int = 10):
        super().__init__(msg)
        self.code = code


def is_screen(params: dict) -> bool:
    return params.get("stage", "gates") == "screen"


def reference_block(params: dict) -> dict:
    ref = params.get("screen_reference") or DEFAULT_REFERENCE
    miss = [k for k in REF_KEYS if not isinstance(ref.get(k), str) or not ref[k]]
    if miss:
        raise ScreenError(f"screen_reference missing {miss}", 5)
    for k in ("artifact_sha256", "member_sha256", "units_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", ref[k]):
            raise ScreenError(f"screen_reference.{k} is not a sha256", 5)
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", ref["url_env"]) or ".." in ref["shipped"].split("/") \
            or ref["shipped"].startswith("/"):
        raise ScreenError("screen_reference url_env / shipped malformed", 5)
    return ref


# ------------------------------------------------------------------------------------------------ CPU: pricing
def estimate(params: dict) -> dict[str, Any]:
    fixed = float(params.get("screen_fixed_minutes", DEFAULT_FIXED_MIN))
    minutes = fixed + D1_PROXY_CHILD_MIN
    cap = params["phase_caps_minutes"]["screen"]
    usd = params["batch_usd_per_min"]
    return {"rows": PROXY, "child_minutes_d1_rate": D1_PROXY_CHILD_MIN, "fixed_minutes": fixed,
            "minutes": round(minutes, 1), "cap": cap, "ok": minutes <= 0.95 * cap,
            "usd_batch": round(minutes * usd, 2), "usd_quoted": round(minutes * 0.063, 2),
            "source": "D1 job-c7pq2 per-row total_wall_ms on the proxy rows (untempered arm, trained cap)"}


# ------------------------------------------------------------------------------------------------ reference
def locate_reference(ref: dict, data: str | Path, work: str | Path, url: str | None = None) -> dict[str, Any]:
    """The reference units file: the shipped copy ($DATA/<shipped>) if its sha256 is the pin, else the artifact from
    $<url_env> (artifact sha256 -> member tgz sha256 -> inner file sha256, one named member each, never extractall)."""
    from jobs.m2_first_cell.checkpoint_loader import CheckpointError, _extract, download, sha256_file

    shipped = Path(data) / ref["shipped"]
    if shipped.is_file() and sha256_file(shipped) == ref["units_sha256"]:
        return {"path": str(shipped), "how": "shipped", **{k: ref[k] for k in REF_KEYS if k != "url_env"}}
    url = url or os.environ.get(ref["url_env"], "")
    if not url:
        raise ScreenError(f"reference {ref['artifact']}: no shipped {ref['shipped']} with the pinned sha256 and "
                          f"${ref['url_env']} is not set")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    try:
        got = download(url, work / "reference.tar")
        if got != ref["artifact_sha256"]:
            raise ScreenError(f"reference artifact {ref['artifact']} sha256 {got} != screen_reference.artifact_sha256")
        _extract(work / "reference.tar", ref["member"], work / "reference-member.tgz", ref["member_sha256"],
                 ref["member"])
    except CheckpointError as e:
        raise ScreenError(str(e), e.code) from e
    out = work / "reference-units.json"
    with tarfile.open(work / "reference-member.tgz", "r:gz") as t:
        names = {ref["inner_member"], "./" + ref["inner_member"]}
        m = next((x for x in t.getmembers() if x.name in names), None)
        if m is None or not m.isfile():
            raise ScreenError(f"reference member has no regular file {ref['inner_member']}")
        b = t.extractfile(m).read()
    if hashlib.sha256(b).hexdigest() != ref["units_sha256"]:
        raise ScreenError("reference units sha256 != screen_reference.units_sha256")
    out.write_bytes(b)
    return {"path": str(out), "how": "downloaded", **{k: ref[k] for k in REF_KEYS if k != "url_env"}}


def restrict(results: dict, ids: set[str], what: str) -> dict:
    miss = sorted(ids - set(results))
    if miss:
        raise ScreenError(f"{what} predictions miss {len(miss)} of the {len(ids)} proxy rows, e.g. {miss[:3]}")
    return {k: results[k] for k in ids}


def load_parent_units(arm_dir: str | Path, arm_sha256: str) -> dict:
    d = Path(arm_dir)
    man_b = (d / "arm.json").read_bytes()
    if hashlib.sha256(man_b).hexdigest() != arm_sha256:
        raise ScreenError("parent arm.json sha256 != the pinned one")
    ent = json.loads(man_b)["stages"]["index"]
    text = (d / ent["file"]).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != ent["sha256"]:
        raise ScreenError(f"parent {ent['file']} sha256 drifted from arm.json")
    return json.loads(text)


def proxy_rows(index_rows_path: str | Path) -> tuple[list[dict], dict]:
    from harness.suite import read_rows
    from jobs.common.dev_proxy import load_proxy

    prox = load_proxy(PROXY)
    ids = set(prox["run_ids"])
    rows = [r for r in read_rows(index_rows_path) if r["_evaluation"]["run_id"] in ids]
    if len(rows) != len(ids):
        raise ScreenError(f"index rows hold {len(rows)} of the {len(ids)} proxy rows")
    return rows, prox


# ------------------------------------------------------------------------------------------------ GPU: child
def score(a) -> int:
    from jobs.m2_first_cell.fulldev_eval import _read_partial, load_child
    from jobs.m2_pilot.guard_ready import GuardNotReady
    from train import m2_gates as G
    from train.cohort import CohortLeak

    try:
        rows, prox = proxy_rows(a.index_rows)
        parent = load_parent_units(a.parent_arm, a.parent_arm_sha256)
        decider, _, ident = load_child(a, temperature=False)
        pu = {"index_results": restrict(parent["index_results"], set(prox["run_ids"]), "parent arm")}
        plan = G.index_plan_check(decider.decide, rows, pu)  # child refuses exactly the rows the parent refused
    except (ValueError, KeyError, GuardNotReady, CohortLeak, ScreenError, G.IdentityMismatch) as e:
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity-child.json").write_text(json.dumps({**ident, "plan_check": plan, "proxy": prox["ids_sha256"]},
                                                        indent=1, default=str), encoding="utf-8")
    resume = Path(a.resume_dir) / "screen-child.partial.jsonl"
    resume.parent.mkdir(parents=True, exist_ok=True)
    done = _read_partial(resume)
    todo = [r for r in rows if r["_evaluation"]["run_id"] not in done]
    prog = G.Progress(a.progress)
    prog.plan([G.unit_weight(r) for r in todo])
    decide = G.beat_wrap(prog.wrap(decider.decide))
    t0 = time.perf_counter()
    with open(resume, "a", encoding="utf-8", newline="\n") as f:
        for r in todo:
            res = G.index_predict(decide, [r])
            f.write(json.dumps(res[r["_evaluation"]["run_id"]], separators=(",", ":")) + "\n")
            f.flush()
    done = _read_partial(resume)
    ids = [r["_evaluation"]["run_id"] for r in rows]
    if set(done) != set(ids):
        print(f"{len(set(ids) - set(done))} proxy rows have no result", file=sys.stderr)
        return 9
    with open(out / "child.results.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for rid in ids:
            f.write(json.dumps(done[rid], separators=(",", ":")) + "\n")
    (out / "timing-child.json").write_text(json.dumps(
        {"seconds_this_attempt": time.perf_counter() - t0, "rows_this_attempt": len(todo),
         "sum_row_wall_s": sum(float(done[i].get("total_wall_ms") or 0) for i in ids) / 1000}, indent=1),
        encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------ CPU: report
def pair(rows: list[dict], cand: dict, base: dict, B: int, seed: int = 0) -> dict:
    from harness import bootstrap, score as S

    c = bootstrap.compare(S.decompose(rows, cand, edition=EDITION), S.decompose(rows, base, edition=EDITION), B=B,
                          seed=seed)
    return {"candidate": c["candidate"]["index"], "baseline": c["baseline"]["index"], "diff": c["diff"]["index"],
            "areas_diff": c["diff"]["areas"]}


def verdict(d_ref: float) -> str:
    return "DROP" if d_ref < -DROP_MARGIN else "PROMOTE"


def build_report(rows: list[dict], prox: dict, child: dict, parent: dict, ref: dict, ref_ident: dict, params: dict,
                 elapsed_min: float, B: int = 2000) -> dict[str, Any]:
    ids = {r["_evaluation"]["run_id"] for r in rows}
    child = restrict(child, ids, "child")
    parent = restrict(parent, ids, "parent arm")
    ref = restrict(ref, ids, f"reference {ref_ident.get('artifact')}")
    vp, vr = pair(rows, child, parent, B), pair(rows, child, ref, B)
    usd = params["batch_usd_per_min"]
    return {
        "schema": SCHEMA, "edition": EDITION, "stage": "screen",
        "proxy": {k: prox[k] for k in ("ids_sha256", "n_target", "design", "seed", "base_subset_ids_sha256")}
        | {"rows": len(rows)},
        "scorer": "harness.score.decompose + harness.bootstrap.compare (case clusters, B=%d, seed 0) on the proxy rows" % B,
        "child": vp["candidate"], "parent_stock_readout": vp["baseline"], "reference": vr["baseline"],
        "reference_identity": {k: v for k, v in ref_ident.items() if k != "path"},
        "child_minus_parent": vp["diff"], "child_minus_reference": vr["diff"],
        "areas": {"child_minus_parent": vp["areas_diff"], "child_minus_reference": vr["areas_diff"]},
        "status": {"child": _count(r.get("status") for r in child.values()),
                   "parent": _count(r.get("status") for r in parent.values()),
                   "reference": _count(r.get("status") for r in ref.values())},
        "rule": f"DROP if child - reference < -{DROP_MARGIN} Index points on the proxy (point), else PROMOTE to the "
                "full subset + full DEV; a screen, never a promotion",
        "verdict": verdict(vr["diff"]["point"]),
        "cost": {"elapsed_min": elapsed_min, "runtime_usd": round(elapsed_min * usd, 2), "usd_per_min": usd,
                 "build_minutes_not_included": True},
        "measured": True}


def _count(xs) -> dict:
    c: dict = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def _results(path: Path) -> dict:
    return {r["run_id"]: r for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}


def report(a) -> int:
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    try:
        rows, prox = proxy_rows(a.index_rows)
        parent = load_parent_units(a.parent_arm, a.parent_arm_sha256)["index_results"]
        ref_ident = json.loads(Path(a.reference).read_text(encoding="utf-8"))
        rp = Path(ref_ident["path"])
        if hashlib.sha256(rp.read_bytes()).hexdigest() != ref_ident["units_sha256"]:
            raise ScreenError("reference units sha256 changed since it was located")
        ref = json.loads(rp.read_text(encoding="utf-8"))["index_results"]
        child = _results(Path(a.out_dir) / "child.results.jsonl")
        out = build_report(rows, prox, child, parent, ref, ref_ident, params, a.elapsed_min, a.B)
    except ScreenError as e:
        print(f"screen: {e}", file=sys.stderr)
        return e.code
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": out["verdict"], "child_minus_reference": out["child_minus_reference"]["point"],
                      "child_minus_parent": out["child_minus_parent"]["point"]}))
    return 0


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: a complete screen measurement."""
    try:
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        return (d.get("schema") == SCHEMA and d.get("measured") is True and d.get("verdict") in ("DROP", "PROMOTE")
                and all(fin(d[k][f]) for k in ("child", "parent_stock_readout", "reference", "child_minus_parent",
                                               "child_minus_reference") for f in ("point", "lo", "hi"))
                and sum(d["status"]["child"].values()) == d["proxy"]["rows"]
                and d["verdict"] == verdict(d["child_minus_reference"]["point"]))
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.screen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("estimate")
    s.add_argument("--params", required=True)
    s = sub.add_parser("reference")
    s.add_argument("--params", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("score")
    for k in ("--cell", "--ckpt-dir", "--model-dir", "--records", "--guard-readiness", "--index-rows", "--resume-dir",
              "--out-dir", "--parent-arm", "--parent-arm-sha256"):
        s.add_argument(k, required=True)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--train-dir", default=None)
    s.add_argument("--teacher-dir", default=None)
    s.add_argument("--pool-dir", action="append", default=None)
    s.add_argument("--progress", default=None)
    s.add_argument("--device", default="cuda")
    s = sub.add_parser("report")
    for k in ("--params", "--index-rows", "--out-dir", "--parent-arm", "--parent-arm-sha256", "--reference", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--elapsed-min", type=float, required=True)
    s.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        e = estimate(json.loads(Path(a.params).read_text(encoding="utf-8")))
        print(json.dumps(e, indent=1))
        return 0 if e["ok"] else 15
    if a.cmd == "reference":
        try:
            ref = reference_block(json.loads(Path(a.params).read_text(encoding="utf-8")))
            rec = locate_reference(ref, a.data, a.work)
            proxy = rows_cover(rec["path"])
        except ScreenError as e:
            print(f"screen reference: {e}", file=sys.stderr)
            return e.code
        rec["proxy_rows_covered"] = proxy
        Path(a.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(json.dumps(rec))
        return 0
    return {"score": score, "report": report}[a.cmd](a)


def rows_cover(units_path: str | Path) -> int:
    """Fail closed (setup, before training) unless the reference has a prediction for EVERY proxy row."""
    from jobs.common.dev_proxy import proxy_ids

    ids = proxy_ids(PROXY)
    res = json.loads(Path(units_path).read_text(encoding="utf-8")).get("index_results") or {}
    restrict(res, ids, "reference")
    return len(ids)


if __name__ == "__main__":
    sys.exit(main())
