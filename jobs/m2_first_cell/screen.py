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
  screen_parent "none" (A-PREREG-SIZECLIFF): no parent arm at all (job_params "parent_arm": "none"); the report pairs
                    the child with the reference and the comparators only (child_minus_parent null)
  screen_comparators  [{name, shipped, units_sha256, readout_template, source}]: extra pinned per-row results
                    (e.g. Clef-flash / Clef 27B decide_32k_substituted restricted to the proxy rows), shipped in the
                    context, covering every proxy row; paired child - comparator in the report (``comparators``). They
                    are CROSS-readout (each system under its own native readout, labelled) and never enter the verdict
  jevkit            {price, lean_template (optional)}: after the proxy rows, the same child decider (and, if given,
                    a second decider on the SAME model under the lean template) answers the 231 pinned public
                    JevBench items serially (jevkit.systemone, 2 warm-up requests first); the report scores each
                    with jevkit (open-split Intelligence, Calibration, Speed, Cost at ``price``, tokens per
                    decision) and pairs lean - native (paired stratified bootstrap of I and C)
  screen_reference_match  "exact" (default) | "family": the reference may be ANOTHER backbone scored under the same
                    readout FAMILY (``readout_family``: jrt-rwkv-g1-v3w and every generic jrt-v3w-generic:<family> are
                    "v3w"), e.g. a Q9-AF v2 cell (generic v3w on the Qwen tokenizer) screened against D1+v3w (RWKV7).
                    The paired comparison is row-for-row on the same proxy rows either way; the parent is always
                    exact (same weights, same template). Recorded in the report (``readout.reference_match``).
For a cell whose template is a generic v3w template, ``screen_readout`` "v3w" means that template (layout g1v2).
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
#: "cross" (A-PREREG-SIZECLIFF, shim readout): the reference is scored under ANOTHER readout family; the pairing is
#: row-for-row on the same proxy rows and labelled cross-readout in the report (readout.reference_match)
REFERENCE_MATCH = ("exact", "family", "cross")


def _is_generic(template) -> bool:
    return isinstance(template, str) and template.startswith("jrt-v3w-generic:")


def _is_shim(template) -> bool:
    return isinstance(template, str) and template.startswith("jrt-h2o-shim:")


def readout_family(template: str | None) -> str | None:
    """The readout family of a prompt template: "v3w" for jrt-rwkv-g1-v3w and every generic jrt-v3w-generic:<family>
    (the same v3w body, code key and word-id rule on another backbone's chat format and tokenizer), "h2o_shim" for
    every jrt-h2o-shim:<family> (decide.shim_format), else the id."""
    if _is_shim(template):
        return "h2o_shim"
    return "v3w" if template == "jrt-rwkv-g1-v3w" or _is_generic(template) else template


def same_readout(a: str | None, b: str | None, match: str = "exact") -> bool:
    if match not in REFERENCE_MATCH:
        raise ScreenError(f"screen_reference_match {match!r}: one of {REFERENCE_MATCH}", 5)
    if match == "cross":
        return a is not None and b is not None
    return a == b if match == "exact" else (a is not None and readout_family(a) == readout_family(b))
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
    elif name == "v3w" and _is_generic(cfg.get("template")):  # the cell's own generic v3w template
        tpl, lay = cfg["template"], "g1v2"
    elif name in SCREEN_READOUTS:
        tpl, lay = SCREEN_READOUTS[name]
    else:
        raise ScreenError(f"screen_readout {name!r}: native or one of {sorted(SCREEN_READOUTS)}", 5)
    match = params.get("screen_reference_match", "exact")
    if match not in REFERENCE_MATCH:
        raise ScreenError(f"screen_reference_match {match!r}: one of {REFERENCE_MATCH}", 5)
    return {"name": name, "template": tpl, "layout": lay,
            "native": tpl == cfg.get("template") and lay == cfg.get("layout", "ordered"),
            "reference_match": match, "family": readout_family(tpl)}


def template_available(template: str) -> bool:
    """This code can render ``template`` for the trained head (``heads.layout.resolve_template`` where the readout
    code is present, else ``decide.prompt.TEMPLATES``). A generic v3w template: registered in
    ``decide.v3w_generic.TEMPLATE_BACKBONE`` with a pinned spec identity (bound to the snapshot when scored)."""
    if _is_shim(template):
        try:
            from decide.shim_format import PINNED_SPEC_SHA256 as SHIM_PINS
            from decide.shim_format import is_shim_template
        except ImportError:
            return False
        return is_shim_template(template) and template in SHIM_PINS
    if _is_generic(template):
        try:
            from decide.v3w_generic import PINNED_SPEC_SHA256, TEMPLATE_BACKBONE
        except ImportError:
            return False
        return template in TEMPLATE_BACKBONE and template in PINNED_SPEC_SHA256
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


def _block(spec, presets: dict, default_by_tpl: dict, template: str, what: str, match: str = "exact"):
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
    if not same_readout(blk["readout_template"], template, match):  # fair pairing: every arm under ONE readout (family)
        raise ScreenError(f"{what} {blk['artifact']} readout {blk['readout_template']} != the screen readout {template}"
                          + (" (family)" if match == "family" else ""))
    return blk


def reference_block(params: dict, template: str = "jrt-rwkv-g1-v1") -> dict:
    return _block(params.get("screen_reference"), REFERENCES, DEFAULT_REFERENCE_BY_TEMPLATE, template,
                  "screen_reference", params.get("screen_reference_match", "exact"))


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
    m = re.search(r";prompt=([A-Za-z0-9:._-]+?)\.[0-9a-f]{8};", str(arm_identity.get("revision", "")))
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
    if ref_rec["parent"].get("source") == "none":  # no parent arm in this job (screen_parent "none")
        return None, {"source": "none", "readout_template": tpl}
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
    if params.get("screen_parent") == "none":
        if params.get("parent_arm") != "none":
            raise ScreenError('screen_parent "none" needs job_params "parent_arm": "none"', 5)
        par = {"source": "none", "readout_template": ro["template"]}
    else:
        pb = parent_block(params, ro["template"], in_job_template)
        if pb is None:
            par = {"source": "in_job", "readout_template_required": ro["template"]}
        else:
            par = {"source": "pinned", **locate_reference(pb, data, work)}
            restrict(read_predictions(par["path"]), ids, f"parent {par['artifact']}")
    rec = {"readout": ro, "reference": ref, "parent": par, "proxy_rows_covered": len(ids)}
    comps = locate_comparators(params, data, ids)
    if comps:
        rec["comparators"] = comps
    jk = jevkit_block(params, ro["template"])
    if jk:
        rec["jevkit"] = jk
    return rec


COMP_KEYS = ("name", "shipped", "units_sha256", "readout_template", "source")


def locate_comparators(params: dict, data, ids: set[str]) -> list[dict]:
    """``screen_comparators``: each shipped file with its pinned sha256, covering every proxy row (fail closed)."""
    out = []
    for c in params.get("screen_comparators") or []:
        miss = [k for k in COMP_KEYS if not isinstance(c.get(k), str) or not c[k]]
        if (miss or not re.fullmatch(r"[0-9a-f]{64}", c["units_sha256"]) or ".." in c["shipped"].split("/")
                or c["shipped"].startswith("/") or not re.fullmatch(r"[a-z0-9_]+", c["name"])):
            raise ScreenError(f"screen_comparators entry {c.get('name')!r} malformed (missing {miss})", 5)
        path = Path(data) / c["shipped"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != c["units_sha256"]:
            raise ScreenError(f"comparator {c['name']}: no shipped {c['shipped']} with the pinned sha256")
        restrict(read_predictions(path), ids, f"comparator {c['name']}")
        out.append({**{k: c[k] for k in COMP_KEYS}, "path": str(path), "cross_readout": True})
    if len({c["name"] for c in out}) != len(out):
        raise ScreenError("screen_comparators names must be unique", 5)
    return out


def jevkit_block(params: dict, template: str) -> dict | None:
    """``jevkit``: the price spec (jevkit.meter.parse_price, validated here) and an optional lean template, which must
    be a registered generic v3w template with a pinned identity, scored beside a generic v3w screen readout."""
    jk = params.get("jevkit")
    if not jk:
        return None
    from jevkit.meter import PriceError, parse_price

    try:
        price = parse_price(jk["price"]).to_dict()
    except (KeyError, PriceError, ValueError) as e:
        raise ScreenError(f"jevkit.price: {e}", 5) from e
    lean = jk.get("lean_template")
    extra = dict(jk.get("extra_templates") or {})
    for name, t in [("lean", lean)] + sorted(extra.items()):
        if t is None:
            continue
        if not (_is_generic(t) or _is_shim(t)) or not template_available(t) or name in ("native", "native_shimcal"):
            raise ScreenError(f"jevkit {name} template {t!r}: a pinned generic v3w / shim template", 5)
    if lean is not None and readout_family(lean) != readout_family(template):
        raise ScreenError(f"jevkit.lean_template {lean!r} is not in the screen readout's family ({template})", 5)
    return {"price": jk["price"], "price_parsed": price, "lean_template": lean, "extra_templates": extra,
            "shim_calibration": bool(jk.get("shim_calibration")),
            "items": "jevkit.items.load_public (231 pinned public JevBench items; eval-only)"}


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
        if parent is None:  # no parent arm: nothing to pair the refusal plan with (recorded)
            plan = {"skipped": "screen_parent none: no parent arm in this job"}
        else:
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
    if rec.get("jevkit"):
        return jevkit_answer(rec["jevkit"], decider, a.model_dir, out / "jevkit")
    return 0


JEVKIT_WARMUP = 2  # jevkit.meter.measure's warm-up requests (untimed, not recorded)


def jevkit_answer(jk: dict, decider, model_dir: str, out_dir: Path) -> int:
    """The child's answers to the pinned public JevBench items: ``native`` (the screen's decider) and, with a lean
    template, ``lean`` (a second decider over the SAME model object and encoder, the lean template bound to this
    snapshot's tokenizer files and checked against its pinned identity). Serial, one request per item, timed."""
    from decide import Decider
    from decide.shim_format import bind_template as bind_shim
    from decide.shim_format import is_shim_template
    from decide.v3w_generic import bind_template
    from heads.decide_backend import DecisionModelBackend
    from jevkit.items import load_public
    from jevkit.systemone import SystemOneAdapter, run_items

    try:
        items = load_public()
    except Exception as e:  # noqa: BLE001 - a missing / changed pinned file is an identity failure
        print(f"identity: jevkit items: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    arms = {"native": decider}
    more = ([("lean", jk["lean_template"])] if jk.get("lean_template") else []) + sorted(
        (jk.get("extra_templates") or {}).items())
    be = decider.backend
    for name, tpl in more:  # score-only arms: the SAME model object and encoder under another pinned template
        try:
            (bind_shim if is_shim_template(tpl) else bind_template)(tpl, model_dir)
        except (KeyError, ValueError) as e:
            print(f"identity: {name} template: {e}", file=sys.stderr)
            return 10
        arms[name] = Decider(DecisionModelBackend(be.model, be.encode, template=tpl, layout="g1v2",
                                                  checkpoint=be.checkpoint))
    out_dir.mkdir(parents=True, exist_ok=True)
    ident = {}
    for name, dec in arms.items():
        ad = SystemOneAdapter(dec, kind="decider", model_name=f"child-{name}")
        for it in items[:JEVKIT_WARMUP]:
            ad.answer_item(it)
        t0 = time.perf_counter()
        recs = run_items(ad, items)
        with open(out_dir / f"{name}.preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
        ident[name] = {"decider": dec.revision(), "seconds": round(time.perf_counter() - t0, 2), "n": len(recs),
                       "status": _count(r["status"] for r in recs)}
    (out_dir / "identity.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
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
    parent = None if parent is None else restrict(parent, ids, "parent")
    ref = restrict(ref, ids, f"reference {ref_ident.get('artifact')}")
    nopar = {"candidate": None, "baseline": None, "diff": None, "areas_diff": None}
    vp, vr = (nopar if parent is None else pair(rows, child, parent, B)), pair(rows, child, ref, B)
    usd = params["batch_usd_per_min"]
    return {
        "schema": SCHEMA, "edition": EDITION, "stage": "screen", "readout": readout,
        "proxy": {k: prox[k] for k in ("ids_sha256", "n_target", "design", "seed", "base_subset_ids_sha256")}
        | {"rows": len(rows)},
        "scorer": "harness.score.decompose + harness.bootstrap.compare (case clusters, B=%d, seed 0) on the proxy rows" % B,
        "child": vr["candidate"], "parent_stock_readout": vp["baseline"], "reference": vr["baseline"],
        "reference_identity": {k: v for k, v in ref_ident.items() if k != "path"},
        "parent_identity": parent_ident,
        "child_minus_parent": vp["diff"], "child_minus_reference": vr["diff"],
        "areas": {"child_minus_parent": vp["areas_diff"], "child_minus_reference": vr["areas_diff"]},
        "status": {"child": _count(r.get("status") for r in child.values()),
                   "parent": None if parent is None else _count(r.get("status") for r in parent.values()),
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
    out["parent"] = None if parent is None else safe_detect(rows, parent)
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
        if not same_readout(ref_ident["readout_template"], rec["readout"]["template"],
                            rec["readout"].get("reference_match", "exact")):
            raise ScreenError("reference readout != the screen readout")
        ref = pinned_predictions(ref_ident)
        child = read_predictions(Path(a.out_dir) / "child.results.jsonl")
        out = build_report(rows, prox, child, parent, ref, ref_ident, params, a.elapsed_min, a.B,
                           readout=rec["readout"], parent_ident=pid)
        if rec.get("comparators"):
            out["comparators"] = comparator_report(rows, child, rec["comparators"], a.B)
        if rec.get("jevkit"):
            out["jevkit"] = jevkit_report(rec["jevkit"], Path(a.out_dir) / "jevkit")
    except ScreenError as e:
        print(f"screen: {e}", file=sys.stderr)
        return e.code
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": out["verdict"], "child_minus_reference": out["child_minus_reference"]["point"],
                      "child_minus_parent": (out["child_minus_parent"] or {}).get("point"),
                      "comparators": {k: v["child_minus_comparator"]["point"]
                                      for k, v in (out.get("comparators") or {}).items()},
                      "jevkit": {k: v.get("summary") for k, v in (out.get("jevkit") or {}).get("arms", {}).items()}}))
    return 0


def shim_calibrate(src: Path, dst: Path, items) -> None:
    """The H2O shim's serving post-processing (vendored ``format_answer``: temperature 0.8 on every type, the yes/no
    commit floor 0.801) applied to the native answers' probabilities: a score-only variant, same decisions' ranking."""
    import math

    import third_party.h2o_lightning_shim as H

    mod, c = H.load_contract()
    by_id = {it.id: it for it in items}
    lg = lambda p: math.log(max(float(p), 1e-12))  # noqa: E731
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            a, it = r.get("answer"), by_id.get(r.get("id"))
            if r.get("status") == "ok" and isinstance(a, dict) and it is not None:
                q = it.question
                if q["type"] == "noul":
                    py = float(a["noul"])
                    r["answer"] = mod.format_answer("noul", ["true", "false"], ["", ""], [lg(py), lg(1 - py)],
                                                    c.temperature, c.noul_floor)
                else:
                    names = list(a["probabilities"])
                    descs = [str(x) for x in (q.get("criteria") or [])] if q["type"] == "score" else names
                    r["answer"] = mod.format_answer(q["type"], names, descs, [lg(a["probabilities"][n]) for n in names],
                                                    c.temperature, c.noul_floor)
            g.write(json.dumps(r, separators=(",", ":")) + "\n")


def comparator_report(rows: list[dict], child: dict, comps: list[dict], B: int) -> dict:
    """child - comparator on the proxy rows (frozen scorer), per pinned comparator; cross-readout, never the verdict."""
    ids = {r["_evaluation"]["run_id"] for r in rows}
    out = {}
    for c in comps:
        p = Path(c["path"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != c["units_sha256"]:
            raise ScreenError(f"comparator {c['name']} file sha256 changed since it was located")
        v = pair(rows, restrict(child, ids, "child"), restrict(read_predictions(p), ids, c["name"]), B)
        out[c["name"]] = {"comparator": v["baseline"], "child": v["candidate"], "child_minus_comparator": v["diff"],
                          "source": c["source"], "readout_template": c["readout_template"], "cross_readout": True,
                          "units_sha256": c["units_sha256"]}
    return out


def jevkit_report(jk: dict, d: Path) -> dict:
    """jevkit scores of each answered arm (``jevkit score --public``: self-hosted lane, the declared price) and the
    paired lean - native comparison (the same stratified resamples for both arms; I and C differences)."""
    import numpy as np

    from jevkit import cli as JC
    from jevkit import scoring as JS
    from jevkit.items import load_public
    from jevkit.systemone import preds_to_answers

    items = load_public()
    arms, tables, axes = {}, {}, {}
    names = ["native"] + (["lean"] if jk.get("lean_template") else []) + sorted(jk.get("extra_templates") or {})
    if jk.get("shim_calibration"):  # score-only: the shim's own post-processing on the native answers (no GPU)
        shim_calibrate(d / "native.preds.jsonl", d / "native_shimcal.preds.jsonl", items)
        names.append("native_shimcal")
    for name in names:
        pf = d / f"{name}.preds.jsonl"
        if not pf.is_file():
            raise ScreenError(f"jevkit {name} predictions missing ({pf})", 9)
        rep_path = d / f"{name}.jevkit.json"
        if JC.main(["score", "--public", "--pred", str(pf), "--price", jk["price"], "--out", str(rep_path)]) != 0:
            raise ScreenError(f"jevkit score {name} failed", 9)
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        preds = [json.loads(x) for x in pf.read_text(encoding="utf-8").splitlines() if x.strip()]
        toks = [(p.get("usage") or {}).get("input_tokens") for p in preds]
        toks = [t for t in toks if isinstance(t, (int, float)) and not isinstance(t, bool)]
        ci = rep.get("ci95") or {}
        arms[name] = {"report": rep_path.name, "report_sha256": hashlib.sha256(rep_path.read_bytes()).hexdigest(),
                      "summary": {"I": rep["axes"]["intelligence"], "C": rep["axes"]["calibration"],
                                  "S": rep["axes"]["speed"], "K": rep["axes"]["cost"],
                                  "capability": rep["capability"], "composite": rep["scores"],
                                  "intelligence_basis": rep["intelligence"].get("basis"),
                                  "I_ci95": ci.get("I"), "C_ci95": ci.get("C"),
                                  "usd_per_1000": (rep.get("cost") or {}).get("usd_per_1000"),
                                  "input_tokens_per_decision": {"mean": float(np.mean(toks)) if toks else None,
                                                                "p50": float(np.median(toks)) if toks else None,
                                                                "n": len(toks)},
                                  "speed": rep.get("speed"),
                                  "status": _count(p.get("status") for p in preds)}}
        tables[name] = JS.build_table(items, preds_to_answers(preds))
        axes[name] = (rep["axes"]["speed"], rep["axes"]["cost"])
    out = {"price": jk["price_parsed"], "lean_template": jk.get("lean_template"),
           "extra_templates": jk.get("extra_templates") or {}, "arms": arms,
           "note": "open split only (the 231 pinned v1.2 public items, not the v1.6.1 P300 draw): Intelligence basis "
                   "open-only, no gap penalty; latency in-process on this H100 (self-hosted x2 + 0.15 s lane)"}
    if "lean" in tables:
        rng = np.random.default_rng(16)
        st = JS.strata(tables["native"])
        di, dc = [], []
        for _ in range(1000):  # the same stratified resample for both arms: paired I / C differences
            idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in st])
            ln = JS.aggregate(tables["lean"], idx, detail=False)
            nt = JS.aggregate(tables["native"], idx, detail=False)
            di.append(ln["intelligence"]["I"] - nt["intelligence"]["I"])
            dc.append(ln["calibration"]["score"] - nt["calibration"]["score"])
        sl, sn = arms["lean"]["summary"], arms["native"]["summary"]
        tl, tn = sl["input_tokens_per_decision"]["mean"], sn["input_tokens_per_decision"]["mean"]
        drop = (tn - tl) / tn if tl is not None and tn else None
        out["lean_minus_native"] = {
            "I": sl["I"] - sn["I"], "I_ci95": [float(np.percentile(di, 2.5)), float(np.percentile(di, 97.5))],
            "C": sl["C"] - sn["C"], "C_ci95": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
            "tokens_mean_drop_frac": drop,
            "composite_pairs": JS.bootstrap({"lean": tables["lean"], "native": tables["native"]}, axes).get("_pairs"),
            "rule": "A-PREREG-SIZECLIFF: lean adopted iff jevkit-I drops <= 1 point (paired) while tokens drop >= 25%",
            "adopt": bool(sl["I"] - sn["I"] >= -1.0 and drop is not None and drop >= 0.25)}
    return out


def _no_parent(d: dict) -> bool:
    """A screen run without a parent arm (screen_parent "none"): its parent fields are null by construction."""
    return (d.get("parent_identity") or {}).get("source") == "none" and d.get("child_minus_parent") is None


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: a complete screen measurement."""
    try:
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        tpl = (d.get("readout") or {}).get("template")
        return (d.get("schema") == SCHEMA and d.get("measured") is True and d.get("verdict") in ("DROP", "PROMOTE")
                and all(fin(d[k][f]) for k in ("child", "reference", "child_minus_reference")
                        + (() if _no_parent(d) else ("parent_stock_readout", "child_minus_parent"))
                        for f in ("point", "lo", "hi"))
                and sum(d["status"]["child"].values()) == d["proxy"]["rows"]
                and d["verdict"] == verdict(d["child_minus_reference"]["point"])
                and bool(tpl) and same_readout(d["reference_identity"].get("readout_template"), tpl,
                                               (d.get("readout") or {}).get("reference_match", "exact"))
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
    s.add_argument("--gdn-ckpt", default=None, help="parent q9af: the converted GDN export (pinned by the cell config)")
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
                          "parent": rec["parent"].get("artifact") or rec["parent"]["source"]}))
        return 0
    return {"score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
