"""Stage ``screen``: a trained cell's Index SCREEN on the pre-declared proxy600 rows (job_params ``"stage": "screen"``).

Replaces the evals phase (temperature fit, dev card, G12, 2,008-row Index) of a training mode with ONE scoring pass:
the child's untempered arm decider (exactly what ``train.m2_gates`` scores the Index stage with) on the 644 proxy rows
(``jobs/common/dev_proxy600.json``, sha pinned in ``jobs.common.dev_proxy``), at the trained input cap, under ONE
readout (``screen_readout``). The report pairs it, with the frozen scorer (``harness.score.decompose`` +
``harness.bootstrap.compare``, case clusters), with
  * the PARENT (stock readout of the same weights) under the SAME readout: the in-job parent arm when its prompt
    template is the screen's (its ``units-index.json``, pinned by the arm.json sha256 the job verified before
    training), else a pinned stock-family results file (``screen_parent``; v3w default ``stock_v3w`` = job-fgu2c
    art-ivu28), and
  * a REFERENCE cell's child predictions under the SAME readout (the current best; ``screen_reference``; defaults by
    readout: v1 -> ``d1_v1`` = job-c7pq2 art-bm3e6, v3w -> ``d1_v3w`` = job-39gnn art-zskmx).
Pinned files are located through the artifact chain (artifact tar sha256 -> member tgz sha256 -> inner file sha256):
the shipped copy ($DATA/<shipped>) when its sha is the pin, else downloaded from $<url_env>. Fail closed (exit 10):
a reference or parent whose readout template differs from the screen's, or that does not cover EVERY proxy row; a
readout this code cannot render (exit 5).

knobs (job_params, all optional; ``stage`` must be ``"screen"``):
  screen_readout    "native" (default: the cell config's template + layout) | one of SCREEN_READOUTS (v1, v2f, v3w)
  screen_reference  a REFERENCES preset name (d1_v1, d1_v3w) | a full block | absent = the readout's default preset
  screen_parent     "in_job" | a PARENTS preset name (stock_v3w) | a full block | absent = in_job when the in-job
                    arm's template is the screen's, else the readout's default preset
Verdict (pre-declared, ``analysis/proxy_validate.py``: the proxy's 90th-percentile paired-gap error on 1-5 point
pairs is 2.5): DROP if child - reference < -2.5 Index points on the proxy (point estimate), else PROMOTE (to the full
subset + full DEV). Both paired differences are recorded either way. A SCREEN, never a promotion.

    python -X utf8 -m jobs.m2_first_cell.screen estimate  --params P
    python -X utf8 -m jobs.m2_first_cell.screen reference --params P --cell-config C --data $DATA --work W --out ref.json
    python -X utf8 -m jobs.m2_first_cell.screen score     --cell C --ckpt-dir D --model-dir M --records R --train-dir T
        --guard-readiness G --index-rows <subset rows> --resume-dir K --parent-arm A --parent-arm-sha256 S
        --reference ref.json --out-dir O [--progress p.jsonl]
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
SCHEMA = "m2-screen/v2"
EDITION = "0.2.1"
PROXY = "proxy600"
DROP_MARGIN = 2.5  # Index points (proxy p90 paired-gap error, receipts/2026-10-08-B-dev-proxy600)
#: D1's measured child Index wall on the proxy rows (job-c7pq2 units-child-index.json, sum total_wall_ms over the 644
#: proxy ids at the trained input cap): 25.97 min. Fixed: model load + 150k lineage rebuild + plan (no temperature
#: fit, no card): budgeted 15 min (planning number, not measured separately).
D1_PROXY_CHILD_MIN = 25.97
DEFAULT_FIXED_MIN = 15
#: screen readout name -> (template id, heads.layout layout), as jobs/m2_first_cell/readout_ab READOUTS / READOUT_LAYOUT
SCREEN_READOUTS = {"v1": ("jrt-rwkv-g1-v1", "ordered"), "v2f": ("jrt-rwkv-g1-v2f", "g1v2"),
                   "v3w": ("jrt-rwkv-g1-v3w", "g1v2")}
REFERENCES = {
    "d1_v1": {
        "_note": "D1 = job-c7pq2 (m2_first_cell base, template v1), artifact art-bm3e6; the child's in-job Index units "
                 "(phases/final.tgz out/units-child-index.json, == x_evals). Chain verified locally by agent B.",
        "artifact": "art-bm3e6", "job": "job-c7pq2", "readout_template": "jrt-rwkv-g1-v1",
        "artifact_sha256": "a405907eb85f3b6bc25f287ef37066dcd192b0b2897e4ec45f336a6fe3580a18",
        "member": "./phases/final.tgz",
        "member_sha256": "c49988af37c60754de5abc5cff686ad7dcb1152a0848b341af53dfb02b3414e7",
        "inner_member": "out/units-child-index.json",
        "units_sha256": "545dbb946c16b2e9975d2bb7a19085edc9fc58b69b2d668d7ace153ded55a6b7",
        "shipped": "screen_reference/units-child-index.json", "url_env": "REF_URL"},
    "d1_v3w": {
        "_note": "D1 + the v3w readout = job-39gnn (mode readout_ab_d1), artifact art-zskmx; the child's untempered v3w "
                 "results (phases/final.tgz out/readout_ab/child.v3w.results.jsonl; 2,008 rows, 38.67 [35.50, 41.73], "
                 "independently reproduced). Chain verified locally by agent B.",
        "artifact": "art-zskmx", "job": "job-39gnn", "readout_template": "jrt-rwkv-g1-v3w",
        "artifact_sha256": "dd017eb0e4f1ba31cf1f464cc01b533cf9e7dcdfa341af6725184b2302a12632",
        "member": "./phases/final.tgz",
        "member_sha256": "1ad5f899f8dca73a240b261c353178a9109b72c082c56e2bef095b471a517fb3",
        "inner_member": "out/readout_ab/child.v3w.results.jsonl",
        "units_sha256": "941af29e4e6d427b300703d3a7d29bd959e8c16af82292eb04e87b73b4474f02",
        "shipped": "screen_reference/art-zskmx/child.v3w.results.jsonl", "url_env": "REF_URL"}}
PARENTS = {
    "stock_v3w": {
        "_note": "the stock v3w readout family (rwkv7-hf-g1-v3w, prompt=jrt-rwkv-g1-v3w) of the same RWKV7 snapshot, "
                 "job-fgu2c (R3), artifact art-ivu28; phases/final.tgz out/index/v3w/results.jsonl (31.07). Same 8 "
                 "over-cap rows refused as the D1+v3w child. Chain verified locally by agent B.",
        "artifact": "art-ivu28", "job": "job-fgu2c", "readout_template": "jrt-rwkv-g1-v3w",
        "artifact_sha256": "aeef838bd12eea2d79c45204033a393651a8b2e60fe3f6f75940df1973c35c39",
        "member": "./phases/final.tgz",
        "member_sha256": "6cde2da7c02a95576442b437e5f71d77e09b75da1249b1e04f808ef0b43754ae",
        "inner_member": "out/index/v3w/results.jsonl",
        "units_sha256": "a52a68ebf537df75818f4e128f8eb3e354bdbc0fb5625f1a9ce53621574850f1",
        "shipped": "screen_parent/art-ivu28/v3w.results.jsonl", "url_env": "PARENT_URL"}}
DEFAULT_REFERENCE_BY_TEMPLATE = {"jrt-rwkv-g1-v1": "d1_v1", "jrt-rwkv-g1-v3w": "d1_v3w"}
DEFAULT_PARENT_BY_TEMPLATE = {"jrt-rwkv-g1-v3w": "stock_v3w"}
DEFAULT_REFERENCE = REFERENCES["d1_v1"]  # back-compat name
REF_KEYS = ("artifact", "artifact_sha256", "member", "member_sha256", "inner_member", "units_sha256", "shipped",
            "url_env", "readout_template")


class ScreenError(RuntimeError):
    def __init__(self, msg: str, code: int = 10):
        super().__init__(msg)
        self.code = code


def is_screen(params: dict) -> bool:
    return params.get("stage", "gates") == "screen"


# ------------------------------------------------------------------------------------------------ readout
def screen_readout(params: dict, cfg: dict) -> dict[str, Any]:
    """{name, template, layout, native}: the ONE readout every arm of this screen is scored / pinned under."""
    name = params.get("screen_readout") or "native"
    if name == "native":
        tpl, lay = cfg.get("template"), cfg.get("layout", "ordered")
        if not tpl:
            raise ScreenError("cell config has no template", 5)
    elif name in SCREEN_READOUTS:
        tpl, lay = SCREEN_READOUTS[name]
    else:
        raise ScreenError(f"screen_readout {name!r}: native or one of {sorted(SCREEN_READOUTS)}", 5)
    return {"name": name, "template": tpl, "layout": lay,
            "native": tpl == cfg.get("template") and lay == cfg.get("layout", "ordered")}


def template_available(template: str) -> bool:
    """This code can render ``template`` for the trained head (``heads.layout.resolve_template`` where the readout
    code is present, else ``decide.prompt.TEMPLATES``)."""
    try:
        from heads.layout import resolve_template
    except ImportError:
        resolve_template = None
    try:
        if resolve_template is not None:
            resolve_template(template)
            return True
        from decide.prompt import TEMPLATES

        return template in TEMPLATES
    except KeyError:
        return False


def _block(spec, presets: dict, default_by_tpl: dict, template: str, what: str):
    if spec is None:
        if template not in default_by_tpl:
            raise ScreenError(f"no default {what} for readout {template}: set {what} in job_params", 5)
        spec = default_by_tpl[template]
    if isinstance(spec, str):
        if spec not in presets:
            raise ScreenError(f"{what} preset {spec!r}: one of {sorted(presets)}", 5)
        blk = dict(presets[spec], preset=spec)
    elif isinstance(spec, dict):
        blk = dict(spec)
    else:
        raise ScreenError(f"{what} must be a preset name or a block", 5)
    miss = [k for k in REF_KEYS if not isinstance(blk.get(k), str) or not blk[k]]
    if miss:
        raise ScreenError(f"{what} missing {miss}", 5)
    for k in ("artifact_sha256", "member_sha256", "units_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", blk[k]):
            raise ScreenError(f"{what}.{k} is not a sha256", 5)
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", blk["url_env"]) or ".." in blk["shipped"].split("/") \
            or blk["shipped"].startswith("/"):
        raise ScreenError(f"{what} url_env / shipped malformed", 5)
    if blk["readout_template"] != template:  # fair pairing: every arm under the screen's ONE readout
        raise ScreenError(f"{what} {blk['artifact']} readout {blk['readout_template']} != the screen readout {template}")
    return blk


def reference_block(params: dict, template: str = "jrt-rwkv-g1-v1") -> dict:
    return _block(params.get("screen_reference"), REFERENCES, DEFAULT_REFERENCE_BY_TEMPLATE, template,
                  "screen_reference")


def parent_block(params: dict, template: str, in_job_template: str | None) -> dict | None:
    """None = the in-job parent arm (its prompt template must be the screen's); else a pinned stock results block."""
    spec = params.get("screen_parent")
    if spec is None:
        if in_job_template == template or template not in DEFAULT_PARENT_BY_TEMPLATE:
            spec = "in_job"  # its readout is checked again when the arm is read (parent_predictions)
    if spec == "in_job":
        if in_job_template is not None and in_job_template != template:
            raise ScreenError(f"in-job parent arm readout {in_job_template} != the screen readout {template}")
        return None
    return _block(spec, PARENTS, DEFAULT_PARENT_BY_TEMPLATE, template, "screen_parent")


def arm_template(arm_identity: dict) -> str | None:
    """The prompt template id in a parent arm's revision (``...;prompt=<template>.<hash>;...``)."""
    m = re.search(r";prompt=([a-z0-9-]+)\.[0-9a-f]+;", str(arm_identity.get("revision", "")))
    return m.group(1) if m else None


# ------------------------------------------------------------------------------------------------ CPU: pricing
def _quote_usd_per_min() -> float:
    from jobs.common.rates import QUOTE_USD_PER_MIN

    return QUOTE_USD_PER_MIN


def estimate(params: dict) -> dict[str, Any]:
    fixed = float(params.get("screen_fixed_minutes", DEFAULT_FIXED_MIN))
    minutes = fixed + D1_PROXY_CHILD_MIN
    cap = params["phase_caps_minutes"]["screen"]
    usd = params["batch_usd_per_min"]
    return {"rows": PROXY, "child_minutes_d1_rate": D1_PROXY_CHILD_MIN, "fixed_minutes": fixed,
            "minutes": round(minutes, 1), "cap": cap, "ok": minutes <= 0.95 * cap,
            "usd_batch": round(minutes * usd, 2), "usd_quoted": round(minutes * _quote_usd_per_min(), 2),
            "source": "D1 job-c7pq2 per-row total_wall_ms on the proxy rows (untempered arm, trained cap; the v1 "
                      "readout: other readouts' prompt lengths differ)"}


# ------------------------------------------------------------------------------------------------ pinned files
def locate_reference(ref: dict, data: str | Path, work: str | Path, url: str | None = None) -> dict[str, Any]:
    """A pinned file: the shipped copy ($DATA/<shipped>) if its sha256 is the pin, else the artifact from $<url_env>
    (artifact sha256 -> member tgz sha256 -> inner file sha256, one named member each, never extractall)."""
    from jobs.m2_first_cell.checkpoint_loader import CheckpointError, _extract, download, sha256_file

    keep = {k: v for k, v in ref.items() if k != "url_env" and not k.startswith("_")}
    shipped = Path(data) / ref["shipped"]
    if shipped.is_file() and sha256_file(shipped) == ref["units_sha256"]:
        return {"path": str(shipped), "how": "shipped", **keep}
    url = url or os.environ.get(ref["url_env"], "")
    if not url:
        raise ScreenError(f"{ref['artifact']}: no shipped {ref['shipped']} with the pinned sha256 and "
                          f"${ref['url_env']} is not set")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", ref["artifact"])
    try:
        got = download(url, work / f"{tag}.tar")
        if got != ref["artifact_sha256"]:
            raise ScreenError(f"artifact {ref['artifact']} sha256 {got} != the pinned artifact_sha256")
        _extract(work / f"{tag}.tar", ref["member"], work / f"{tag}-member.tgz", ref["member_sha256"], ref["member"])
    except CheckpointError as e:
        raise ScreenError(str(e), e.code) from e
    out = work / f"{tag}-{Path(ref['inner_member']).name}"
    with tarfile.open(work / f"{tag}-member.tgz", "r:gz") as t:
        names = {ref["inner_member"], "./" + ref["inner_member"]}
        m = next((x for x in t.getmembers() if x.name in names), None)
        if m is None or not m.isfile():
            raise ScreenError(f"{ref['artifact']} member has no regular file {ref['inner_member']}")
        b = t.extractfile(m).read()
    if hashlib.sha256(b).hexdigest() != ref["units_sha256"]:
        raise ScreenError(f"{ref['artifact']} {ref['inner_member']} sha256 != the pinned units_sha256")
    out.write_bytes(b)
    return {"path": str(out), "how": "downloaded", **keep}


def read_predictions(path: str | Path) -> dict[str, dict]:
    """{run_id: result}: a units ``.json`` (``index_results``) or a results ``.jsonl`` (one row per line)."""
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))["index_results"]
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["run_id"] in out:
                raise ScreenError(f"{p.name}: duplicate run_id {r['run_id']}")
            out[r["run_id"]] = r
    return out


def pinned_predictions(rec: dict) -> dict[str, dict]:
    p = Path(rec["path"])
    if hashlib.sha256(p.read_bytes()).hexdigest() != rec["units_sha256"]:
        raise ScreenError(f"{rec['artifact']} file sha256 changed since it was located")
    return read_predictions(p)


def restrict(results: dict, ids: set[str], what: str) -> dict:
    miss = sorted(ids - set(results))
    if miss:
        raise ScreenError(f"{what} predictions miss {len(miss)} of the {len(ids)} proxy rows, e.g. {miss[:3]}")
    return {k: results[k] for k in ids}


def load_parent_units(arm_dir: str | Path, arm_sha256: str) -> tuple[dict, str | None]:
    """(in-job parent arm index units, its prompt template), pinned by the arm.json sha256."""
    d = Path(arm_dir)
    man_b = (d / "arm.json").read_bytes()
    if hashlib.sha256(man_b).hexdigest() != arm_sha256:
        raise ScreenError("parent arm.json sha256 != the pinned one")
    ent = json.loads(man_b)["stages"]["index"]
    text = (d / ent["file"]).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != ent["sha256"]:
        raise ScreenError(f"parent {ent['file']} sha256 drifted from arm.json")
    return json.loads(text), arm_template(ent.get("arm") or {})


def parent_predictions(ref_rec: dict, arm_dir, arm_sha256) -> tuple[dict, dict]:
    """(parent results, identity) for the screen's readout (fail closed on a readout mismatch)."""
    tpl = ref_rec["readout"]["template"]
    if ref_rec["parent"].get("source") == "in_job":
        units, t = load_parent_units(arm_dir, arm_sha256)
        if t != tpl:
            raise ScreenError(f"in-job parent arm readout {t} != the screen readout {tpl}")
        return units["index_results"], {"source": "in_job", "readout_template": t, "arm_sha256": arm_sha256}
    rec = ref_rec["parent"]
    if rec["readout_template"] != tpl:
        raise ScreenError(f"parent {rec['artifact']} readout {rec['readout_template']} != the screen readout {tpl}")
    return pinned_predictions(rec), {k: v for k, v in rec.items() if k != "path"}


def proxy_rows(index_rows_path: str | Path) -> tuple[list[dict], dict]:
    from harness.suite import read_rows
    from jobs.common.dev_proxy import load_proxy

    prox = load_proxy(PROXY)
    ids = set(prox["run_ids"])
    rows = [r for r in read_rows(index_rows_path) if r["_evaluation"]["run_id"] in ids]
    if len(rows) != len(ids):
        raise ScreenError(f"index rows hold {len(rows)} of the {len(ids)} proxy rows")
    return rows, prox


def locate_all(params: dict, cfg: dict, data, work, in_job_template: str | None = None) -> dict[str, Any]:
    """Setup (before training): the readout, the reference and the parent source, every pinned file located, its
    readout checked and its coverage of EVERY proxy row checked (fail closed)."""
    from jobs.common.dev_proxy import proxy_ids

    ro = screen_readout(params, cfg)
    if not template_available(ro["template"]):
        raise ScreenError(f"screen_readout {ro['name']} ({ro['template']}) cannot be rendered by this code (the readout "
                          "templates are not on this branch)", 5)
    ids = proxy_ids(PROXY)
    ref = locate_reference(reference_block(params, ro["template"]), data, work)
    restrict(read_predictions(ref["path"]), ids, f"reference {ref['artifact']}")
    pb = parent_block(params, ro["template"], in_job_template)
    if pb is None:
        par = {"source": "in_job", "readout_template_required": ro["template"]}
    else:
        par = {"source": "pinned", **locate_reference(pb, data, work)}
        restrict(read_predictions(par["path"]), ids, f"parent {par['artifact']}")
    return {"readout": ro, "reference": ref, "parent": par, "proxy_rows_covered": len(ids)}


# ------------------------------------------------------------------------------------------------ GPU: child
def score(a) -> int:
    from jobs.m2_first_cell.fulldev_eval import _read_partial, load_child
    from jobs.m2_pilot.guard_ready import GuardNotReady
    from train import m2_gates as G
    from train.cohort import CohortLeak

    try:
        rec = json.loads(Path(a.reference).read_text(encoding="utf-8"))
        ro = rec["readout"]
        rows, prox = proxy_rows(a.index_rows)
        parent, _ = parent_predictions(rec, a.parent_arm, a.parent_arm_sha256)
        decider, _, ident = load_child(a, temperature=False, readout=(ro["template"], ro["layout"]))
        pu = {"index_results": restrict(parent, set(prox["run_ids"]), "parent")}
        plan = G.index_plan_check(decider.decide, rows, pu)  # child refuses exactly the rows the parent refused
    except (ValueError, KeyError, GuardNotReady, CohortLeak, ScreenError, G.IdentityMismatch) as e:
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity-child.json").write_text(json.dumps({**ident, "readout": ro, "plan_check": plan,
                                                         "proxy": prox["ids_sha256"]}, indent=1, default=str),
                                             encoding="utf-8")
    resume = Path(a.resume_dir) / f"screen-child.{ro['template']}.partial.jsonl"
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
                 elapsed_min: float, B: int = 2000, readout: dict | None = None,
                 parent_ident: dict | None = None) -> dict[str, Any]:
    ids = {r["_evaluation"]["run_id"] for r in rows}
    child = restrict(child, ids, "child")
    parent = restrict(parent, ids, "parent")
    ref = restrict(ref, ids, f"reference {ref_ident.get('artifact')}")
    vp, vr = pair(rows, child, parent, B), pair(rows, child, ref, B)
    usd = params["batch_usd_per_min"]
    return {
        "schema": SCHEMA, "edition": EDITION, "stage": "screen", "readout": readout,
        "proxy": {k: prox[k] for k in ("ids_sha256", "n_target", "design", "seed", "base_subset_ids_sha256")}
        | {"rows": len(rows)},
        "scorer": "harness.score.decompose + harness.bootstrap.compare (case clusters, B=%d, seed 0) on the proxy rows" % B,
        "child": vp["candidate"], "parent_stock_readout": vp["baseline"], "reference": vr["baseline"],
        "reference_identity": {k: v for k, v in ref_ident.items() if k != "path"},
        "parent_identity": parent_ident,
        "child_minus_parent": vp["diff"], "child_minus_reference": vr["diff"],
        "areas": {"child_minus_parent": vp["areas_diff"], "child_minus_reference": vr["areas_diff"]},
        "status": {"child": _count(r.get("status") for r in child.values()),
                   "parent": _count(r.get("status") for r in parent.values()),
                   "reference": _count(r.get("status") for r in ref.values())},
        "rule": f"DROP if child - reference < -{DROP_MARGIN} Index points on the proxy (point), else PROMOTE to the "
                "full subset + full DEV; every arm under the same readout; a screen, never a promotion",
        "verdict": verdict(vr["diff"]["point"]),
        "warnings": collapse_warnings(rows, child, parent, ref),
        "cost": {"elapsed_min": elapsed_min, "runtime_usd": round(elapsed_min * usd, 2), "usd_per_min": usd,
                 "build_minutes_not_included": True},
        "measured": True}


def collapse_warnings(rows: list[dict], child: dict, parent: dict, ref: dict) -> dict:
    """NON-GATING per-family label-collapse warnings (``jobs.common.label_collapse``) on the proxy rows: the child's
    at the top level, the parent's and the reference's beside it for comparison. Never changes the verdict."""
    from jobs.common.label_collapse import safe_detect

    out = safe_detect(rows, child)
    out["parent"] = safe_detect(rows, parent)
    out["reference"] = safe_detect(rows, ref)
    return out


def _count(xs) -> dict:
    c: dict = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def report(a) -> int:
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    try:
        rows, prox = proxy_rows(a.index_rows)
        rec = json.loads(Path(a.reference).read_text(encoding="utf-8"))
        parent, pid = parent_predictions(rec, a.parent_arm, a.parent_arm_sha256)
        ref_ident = rec["reference"]
        if ref_ident["readout_template"] != rec["readout"]["template"]:
            raise ScreenError("reference readout != the screen readout")
        ref = pinned_predictions(ref_ident)
        child = read_predictions(Path(a.out_dir) / "child.results.jsonl")
        out = build_report(rows, prox, child, parent, ref, ref_ident, params, a.elapsed_min, a.B,
                           readout=rec["readout"], parent_ident=pid)
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
        tpl = (d.get("readout") or {}).get("template")
        return (d.get("schema") == SCHEMA and d.get("measured") is True and d.get("verdict") in ("DROP", "PROMOTE")
                and all(fin(d[k][f]) for k in ("child", "parent_stock_readout", "reference", "child_minus_parent",
                                               "child_minus_reference") for f in ("point", "lo", "hi"))
                and sum(d["status"]["child"].values()) == d["proxy"]["rows"]
                and d["verdict"] == verdict(d["child_minus_reference"]["point"])
                and bool(tpl) and d["reference_identity"].get("readout_template") == tpl
                and (d.get("parent_identity") or {}).get("readout_template") == tpl)
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.screen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("estimate")
    s.add_argument("--params", required=True)
    s = sub.add_parser("reference")
    s.add_argument("--params", required=True)
    s.add_argument("--cell-config", required=True, help="the mode's cell config (the native readout)")
    s.add_argument("--data", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("score")
    for k in ("--cell", "--ckpt-dir", "--model-dir", "--records", "--guard-readiness", "--index-rows", "--resume-dir",
              "--out-dir", "--parent-arm", "--parent-arm-sha256", "--reference"):
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
            params = json.loads(Path(a.params).read_text(encoding="utf-8"))
            cfg = json.loads(Path(a.cell_config).read_text(encoding="utf-8"))
            rec = locate_all(params, cfg, a.data, a.work)
        except ScreenError as e:
            print(f"screen reference: {e}", file=sys.stderr)
            return e.code
        Path(a.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(json.dumps({"readout": rec["readout"]["template"], "reference": rec["reference"]["artifact"],
                          "parent": rec["parent"].get("artifact", "in_job")}))
        return 0
    return {"score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
