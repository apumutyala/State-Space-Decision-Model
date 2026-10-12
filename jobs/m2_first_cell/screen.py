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
    stock = jk.get("stock_shim")
    if stock and not _is_shim(template):
        raise ScreenError("jevkit.stock_shim needs a shim screen readout (the stock arm reads the same shim prompt)", 5)
    tfit = jk.get("tfit")  # JEV2-4B: post-hoc scalar temperature fitted on the CALIB draw (never eval items)
    if tfit is not None and not (isinstance(tfit, dict) and tfit.get("records")):
        raise ScreenError("jevkit.tfit: {records: <path under --data>, lo, hi}", 5)
    pops = jk.get("perturb_ops")  # JEV2-4B G1: jevkit perturb operators answered by the native arm
    if pops is not None:
        from jevkit.perturb import OPERATORS
        if not pops or any(o not in OPERATORS for o in pops):
            raise ScreenError(f"jevkit.perturb_ops {pops!r}: names from jevkit.perturb.OPERATORS", 5)
    extra_keys = (({"tfit": tfit} if tfit is not None else {}) | ({"perturb_ops": list(pops)} if pops else {})
                  | ({"covered_families": sorted(jk["covered_families"])} if jk.get("covered_families") else {})
                  | ({"assert_no_truncation": True} if jk.get("assert_no_truncation") else {})
                  | ({"slice_floor": True} if jk.get("slice_floor") else {})
                  | ({"by_family": True} if jk.get("by_family") else {})
                  | ({"slice_first": True} if jk.get("slice_first") else {})
                  # JEV3-SYM: bitwise G1 KEEP gate, the 8-op flip battery, the H2O target reference, slice/train disjointness
                  | {k: jk[k] for k in ("g1_max_dp", "g1_keep_gate", "flip_ops_all", "flip_reference", "target_ref",
                                        "slice_state_disjoint", "g1_per_arm", "score_inv", "canon_stock",
                                        "letter_prior") if k in jk})
    if jk.get("slice_floor") and not (stock and tfit):
        raise ScreenError("jevkit.slice_floor needs stock_shim (step 0 = the untrained host) and tfit (the slice)", 5)
    si = jk.get("score_inv")  # JEV3-TYPED KEEP gate 2: the evidence lane's score-item invariance op (vendored, pinned)
    if si is not None:
        if not (isinstance(si, dict) and set(si) == {"shuffle_seeds", "max_dp", "keep_gate", "scripts_sha256"}
                and stock and si["max_dp"] == 0 and list(si["shuffle_seeds"]) == [0, 1, 2]):
            raise ScreenError("jevkit.score_inv: {shuffle_seeds [0,1,2], max_dp 0, keep_gate, scripts_sha256}; "
                              "needs stock_shim (the control)", 5)
        bad = score_inv_scripts_check(si["scripts_sha256"])
        if bad:
            raise ScreenError(f"jevkit.score_inv: vendored scripts differ from the pin: {bad}", 5)
    return {**extra_keys, "price": jk["price"], "price_parsed": price, "lean_template": lean, "extra_templates": extra,
            "shim_calibration": bool(jk.get("shim_calibration")), "stock_shim": bool(stock),
            "paired_primary": jk.get("paired_primary"),
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
    jk0 = rec.get("jevkit") or {}
    if jk0.get("slice_first"):  # JEV3-H2O: tfit + the step-0 floor on the held-out slice BEFORE any screen row is scored
        rc = slice_first(jk0, decider, a.model_dir, out / "jevkit", getattr(a, "data", None))
        if rc:
            return rc
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
        return jevkit_answer(rec["jevkit"], decider, a.model_dir, out / "jevkit", data_dir=a.data)
    return 0


JEVKIT_WARMUP = 2  # jevkit.meter.measure's warm-up requests (untimed, not recorded)


def jevkit_answer(jk: dict, decider, model_dir: str, out_dir: Path, data_dir: str | None = None) -> int:
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
    if jk.get("perturb_ops"):  # JEV2-4B G1: jobs.sym_eval.perturb_eval (the A-PREREG-JEV1-4B-SYM eval path) in process
        from jobs.sym_eval.perturb_eval import run as perturb_run
        t0 = time.perf_counter()
        g1dir = out_dir / "g1" / "native" if jk.get("g1_per_arm") else out_dir / "g1"
        g1 = perturb_run(decider, items, g1dir, ops=jk["perturb_ops"], seeds=(0,),
                         row_fn=lambda req: decider.backend._rows(req)[0], max_dp=float(jk.get("g1_max_dp", 1e-3)))
        ident["g1"] = {"g1_pass": g1["g1_pass"], "rows_all_identical": g1.get("rows_all_identical"),
                       "n_perturbed": g1["n_perturbed"], "seconds": round(time.perf_counter() - t0, 2)}
    if jk.get("flip_ops_all"):  # JEV3-SYM: every jevkit perturb operator (8) on the native arm (report only)
        from jevkit.perturb import OPERATORS
        from jobs.sym_eval.perturb_eval import run as perturb_run
        t0 = time.perf_counter()
        f8 = perturb_run(decider, items, out_dir / "flip8", ops=list(OPERATORS), seeds=(0,), max_dp=1.0)
        ident["flip8"] = {"n_perturbed": f8["n_perturbed"], "seconds": round(time.perf_counter() - t0, 2)}
    if jk.get("target_ref"):  # JEV3-SYM: H2O's EXISTING preds (job-zbtbj), copied into the report dir if pinned
        tr = jk["target_ref"]
        src = Path(data_dir) / tr["preds"] if data_dir else None
        pinned = len(tr.get("sha256") or "") == 64
        if src is not None and src.is_file() and pinned \
                and hashlib.sha256(src.read_bytes()).hexdigest() == tr["sha256"]:
            (out_dir / "target_ref.preds.jsonl").write_bytes(src.read_bytes())
            ident["target_ref"] = {"preds": tr["preds"], "sha256": tr["sha256"], "status": "copied"}
        elif pinned:  # a pinned target that is missing / changed fails closed
            print(f"identity: target_ref {tr.get('preds')} missing or sha256 != {tr['sha256']}", file=sys.stderr)
            return 10
        else:
            ident["target_ref"] = {"preds": tr.get("preds"), "status": "UNAVAILABLE (not pinned)"}
    first = None
    if jk.get("slice_first"):  # tfit.json / calib_probs.jsonl / slice_floor.json were written before the screen
        first = json.loads((out_dir / "slice_first.json").read_text(encoding="utf-8"))
        ident.update(first)
    if jk.get("tfit") and not data_dir and first is None:  # job-9fy9p: screen score had no --data (AttributeError after the proxy)
        print("identity: jevkit.tfit needs screen score --data (the data root)", file=sys.stderr)
        return 10
    if jk.get("tfit") and first is None:  # JEV2-4B: scalar T on the CALIB draw's native head log-probs (fit once, frozen before scoring)
        try:
            ident["tfit"] = fit_native_temperature(decider.backend, Path(data_dir) / jk["tfit"]["records"],
                                                   out_dir / "tfit.json", float(jk["tfit"].get("lo", 0.25)),
                                                   float(jk["tfit"].get("hi", 4.0)), jk["tfit"].get("ids_sha256"))
        except (OSError, ValueError, KeyError) as e:
            print(f"identity: tfit: {type(e).__name__}: {e}", file=sys.stderr)
            return 10
    if jk.get("stock_shim"):  # A-PREREG-JEV1-4B: the STOCK model under the H2O shim readout, in the same process
        eng = StockShimEngine(decider.backend, model_dir)
        ad = SystemOneAdapter(eng, kind="request", model_name="stock-shim")
        for it in items[:JEVKIT_WARMUP]:
            ad.answer_item(it)
        t0 = time.perf_counter()
        with eng.lora_off():
            recs = run_items(ad, items)
        with open(out_dir / "stock_shim.preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
        ident["stock_shim"] = {**eng.identity(), "seconds": round(time.perf_counter() - t0, 2), "n": len(recs),
                               "status": _count(r["status"] for r in recs)}
        if jk.get("g1_per_arm") and jk.get("perturb_ops"):  # JEV3-SYM: the control's reorder/relabel preds (report)
            from jobs.sym_eval.perturb_eval import run as perturb_run
            t0 = time.perf_counter()
            with eng.lora_off():
                g1s = perturb_run(eng, items, out_dir / "g1" / "stock_shim", ops=jk["perturb_ops"], seeds=(0,),
                                  max_dp=1.0, kind="request")
            ident["g1_stock_shim"] = {"n_perturbed": g1s["n_perturbed"], "seconds": round(time.perf_counter() - t0, 2)}
        if jk.get("canon_stock"):  # JEV3-LM secondary: the canonical layout on the UNTRAINED host (LoRA off, scale 1)
            t0 = time.perf_counter()
            recs = canon_stock_answers(decider, eng, items)
            with open(out_dir / "canon_stock.preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
            ident["canon_stock"] = {"seconds": round(time.perf_counter() - t0, 2), "n": len(recs),
                                    "status": _count(r["status"] for r in recs),
                                    "rule": "the native decider (canonical layout) with every LoRA off and the "
                                            "lm_label scale at its init 1.0 = the untrained host read canonically"}
        if jk.get("score_inv"):  # JEV3-TYPED KEEP gate 2 (trained arm) + its control (stock shim): ev_score_* scripts
            t0 = time.perf_counter()
            ident["score_inv"] = score_inv_run(jk["score_inv"], decider, eng, out_dir / "score_inv")
            ident["score_inv"]["seconds"] = round(time.perf_counter() - t0, 2)
        if jk.get("slice_floor") and jk.get("tfit") and first is None:  # A-PREREG-JEV2-H2O: step 0 on the slice
            t0 = time.perf_counter()
            try:
                ident["slice_floor"] = slice_floor(eng, Path(data_dir) / jk["tfit"]["records"], out_dir / "calib_probs.jsonl",
                                                   out_dir / "slice_floor.json", jk["tfit"].get("ids_sha256"))
            except (OSError, ValueError, KeyError) as e:  # recorded; the report then calls the floor UNEVALUATED
                ident["slice_floor"] = {"error": f"{type(e).__name__}: {e}"}
            ident["slice_floor"]["seconds"] = round(time.perf_counter() - t0, 2)
    (out_dir / "identity.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
    return 0


class StockShimEngine:
    """STOCK Qwen3.5 + the H2O shim readout (A-PREREG-JEV1-4B's paired baseline), on the screen's already-loaded model:
    every LoRA adapter is switched off (``LoRALinear.scaling`` = 0, so each layer is exactly its base nn.Linear; restored
    on exit), the vendored shim's own prompt (``Contract.question_prompt``) is tokenized whole-string, ONE forward, and
    the label log-probabilities at the answer slot come from the tied LM head (Qwen3.5 small: lm_head = embed_tokens,
    checked on the snapshot's config.json); then the shim's ``format_answer`` (temperature 0.8, yes/no floor 0.801).
    This is the host scan's ShimEngine (jobs/jev_host_scan) with the logits restricted to the label rows: the softmax
    over labels cancels the full-vocabulary normaliser, so the answers are the same."""

    def __init__(self, backend, model_dir: str):
        import third_party.h2o_lightning_shim as H
        from decide.shim_format import spec_for_template

        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        if not (cfg.get("tie_word_embeddings") or (cfg.get("text_config") or {}).get("tie_word_embeddings")):
            raise ScreenError("stock shim: the snapshot's LM head is not tied to embed_tokens (refusing)", 10)
        self.mod, self.c = H.load_contract()
        self.bb = backend.model.backbone
        self.spec = spec_for_template(backend.template.id)  # the bound shim spec: its tokenizer and label ids
        if self.spec.lean:
            raise ScreenError("stock shim: the cell's template must be the full shim prompt", 5)
        self.W = self.bb.model.get_input_embeddings().weight

    def identity(self) -> dict:
        import third_party.h2o_lightning_shim as H

        return {"arm": "stock_shim", "shim": f"{H.REPO}@{H.TAG}", "temperature": self.c.temperature,
                "noul_floor": self.c.noul_floor, "template_identity": self.spec.identity(),
                "lm_head": "tied embed_tokens, fp32 label rows", "lora": "off (scaling 0)"}

    def lora_off(self):
        import contextlib

        from train.lora import LoRALinear

        @contextlib.contextmanager
        def ctx():
            mods = [m for m in self.bb.model.modules() if isinstance(m, LoRALinear)]
            saved = [m.scaling for m in mods]
            try:
                for m in mods:
                    m.scaling = 0.0
                yield len(mods)
            finally:
                for m, v in zip(mods, saved):
                    m.scaling = v
        return ctx()

    def __call__(self, request: dict) -> dict:
        import torch

        state_txt = self.mod.render_state(request["state"], self.c.state_compact)
        answers, ntok = {}, 0
        for qid, q in request["questions"].items():
            text, names, descs, labels = self.c.question_prompt(state_txt, q, qid)
            ids = self.spec.ids(text)
            with torch.inference_mode():
                out = self.bb.model(input_ids=torch.tensor([ids], dtype=torch.long, device=self.W.device), use_cache=False)
                h = out.last_hidden_state[0, -1].float()  # after the final norm: the LM head's input
                lab = torch.tensor([self.spec.label_ids[x] for x in labels], device=self.W.device)
                lp = (self.W.index_select(0, lab).float() @ h.to(self.W.device)).tolist()
            thr = None
            if q.get("type", "choice") == "noul":
                t = q.get("threshold", q.get("noul_threshold"))
                if isinstance(t, (int, float)) and 0.0 <= float(t) <= 1.0:
                    thr = float(t)
            answers[qid] = self.mod.format_answer(q.get("type", "choice"), names, descs, lp, self.c.temperature,
                                                  self.c.noul_floor, thr)
            ntok += len(ids)
        return {"model": "stock-shim", "answers": answers, "usage": {"input_tokens": ntok, "output_tokens": 0}}

    def label_logps(self, request: dict, qid: str) -> tuple[list[str], list[float]]:
        """(shim option names, raw label log-probs) of one question: the same forward as ``__call__``, before the
        shim's T / floor (neither changes the arg-max: the floor only commits a yes/no answer to its own side)."""
        import torch

        state_txt = self.mod.render_state(request["state"], self.c.state_compact)
        text, names, _, labels = self.c.question_prompt(state_txt, request["questions"][qid], qid)
        ids = self.spec.ids(text)
        with torch.inference_mode():
            out = self.bb.model(input_ids=torch.tensor([ids], dtype=torch.long, device=self.W.device), use_cache=False)
            h = out.last_hidden_state[0, -1].float()
            lab = torch.tensor([self.spec.label_ids[x] for x in labels], device=self.W.device)
            lp = (self.W.index_select(0, lab).float() @ h.to(self.W.device)).tolist()
        return [str(n) for n in names], lp


#: the evidence lane's score-item invariance scripts (claude/ev-score-inv @c47c0a463936d8aff586ee69728291521bc31336),
#: vendored byte-identical; the job params pin their sha256 and the screen refuses to run them if they differ
SCORE_INV_SCRIPTS = ("scripts/ev_score_perturb.py", "scripts/ev_score_flip.py")


def score_inv_scripts_check(pins: dict) -> list[str]:
    root = Path(__file__).resolve().parents[2]
    bad = []
    for f in SCORE_INV_SCRIPTS:
        q = root / f
        got = hashlib.sha256(q.read_bytes()).hexdigest() if q.is_file() else None
        if pins.get(f) != got:
            bad.append(f"{f}: {got} != {pins.get(f)}")
    return bad


def _score_inv_request(row: dict) -> dict:
    """An ev_score_perturb row as a decide request (the typed layout reads question.options / question.scale)."""
    q = row["question"]
    return {"state": row["state"], "questions": {"q": {"type": "score", "instructions": q.get("instructions"),
                                                       "scale": list(q["scale"]), "options": dict(q["options"])}}}


def score_inv_native(decider, row: dict) -> dict:
    """The trained arm's untempered head distribution over the row's handles (exp of the head log-probs)."""
    import math

    lg = decider.backend.score(_score_inv_request(row), lambda: None).logits["q"]
    return {"type": "score", "probabilities": {h: math.exp(lg[h]) for h in row["labels"]}}


def score_inv_stock(eng: "StockShimEngine", row: dict) -> dict:
    """The control: stock Qwen3.5 under the shim prompt reading the same contents as a CHOICE over the row's handles
    (in display order). It reads handles and positions, so it is expected to fail the gate (the check has teeth)."""
    q = row["question"]
    req = {"state": row["state"], "questions": {"q": {"type": "choice", "instructions": q.get("instructions"),
                                                      "criteria": dict(q["options"])}}}
    names, lp = eng.label_logps(req, "q")
    pr = eng.mod.probabilities(lp, 1.0)
    d = dict(zip(names, pr))
    return {"type": "score", "probabilities": {h: float(d[h]) for h in row["labels"]}}


def score_inv_run(spec: dict, decider, eng: "StockShimEngine", d: Path) -> dict:
    """KEEP gate 2: ``scripts/ev_score_perturb.py --public --out d --shuffle-seeds 0,1,2`` (base.jsonl, perturbed.jsonl,
    manifest.json), then BOTH arms answer BOTH files (the base readout comes from base.jsonl), then
    ``scripts/ev_score_flip.py --public --dir d --out d/<arm> --base-preds .. --pert-preds .. --max-dp 0`` per arm
    (exit 0 PASS / 3 FAIL / 2 missing). Preds and reports ship in the artifact (the evidence rescore re-runs the check)."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    d.mkdir(parents=True, exist_ok=True)
    seeds = ",".join(str(x) for x in spec["shuffle_seeds"])
    g = subprocess.run([sys.executable, "-I", "-X", "utf8", "scripts/ev_score_perturb.py", "--public", "--out", str(d),
                        "--shuffle-seeds", seeds], cwd=root, capture_output=True, text=True)
    rec: dict = {"perturb_rc": g.returncode, "perturb_stdout": g.stdout.strip()[-2000:],
                 "scripts_sha256": spec["scripts_sha256"]}
    if g.returncode != 0:
        rec["error"] = g.stderr.strip()[-2000:]
        return rec
    rows = {n: [json.loads(x) for x in (d / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            for n in ("base", "perturbed")}
    arms = {"native": lambda r: score_inv_native(decider, r)}
    for nm, fn in arms.items():
        _score_inv_arm(nm, fn, rows, d)
    with eng.lora_off():
        _score_inv_arm("stock_shim", lambda r: score_inv_stock(eng, r), rows, d)
    for nm in ("native", "stock_shim"):
        f = subprocess.run([sys.executable, "-I", "-X", "utf8", "scripts/ev_score_flip.py", "--public", "--dir", str(d),
                            "--out", str(d / nm), "--base-preds", str(d / nm / "base.preds.jsonl"),
                            "--pert-preds", str(d / nm / "perturbed.preds.jsonl"), "--max-dp", str(spec["max_dp"])],
                           cwd=root, capture_output=True, text=True)
        rec[nm] = {"rc": f.returncode, "pass": f.returncode == 0,
                   "verdict": {0: "PASS", 3: "FAIL", 2: "MISSING"}.get(f.returncode, f"rc {f.returncode}")}
        if f.returncode not in (0, 3):
            rec[nm]["stderr"] = f.stderr.strip()[-2000:]
    rec["n_base"], rec["n_perturbed"] = len(rows["base"]), len(rows["perturbed"])
    return rec


def _score_inv_arm(nm: str, fn, rows: dict, d: Path) -> None:
    (d / nm).mkdir(parents=True, exist_ok=True)
    for part, rs in rows.items():
        with open(d / nm / f"{part}.preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                try:
                    out = {"id": r["id"], "answer": fn(r), "status": "ok"}
                except Exception as e:  # noqa: BLE001  an arm that cannot answer is INVALID (the flip check fails closed)
                    out = {"id": r["id"], "answer": None, "status": f"error: {type(e).__name__}: {e}"[:300]}
                f.write(json.dumps(out, separators=(",", ":"), ensure_ascii=False) + "\n")


def canon_stock_answers(decider, eng: "StockShimEngine", items) -> list:
    """The canonical-layout answers of the UNTRAINED host: LoRA off (``eng.lora_off``) and the lm_label head's scalar
    ``scale`` at its init value 1.0 (restored after), through the SAME decider (canonical rows, tie handling)."""
    import torch

    from jevkit.systemone import SystemOneAdapter, run_items

    head = decider.backend.model.head
    keep = head.scale.detach().clone() if hasattr(head, "scale") else None
    ad = SystemOneAdapter(decider, kind="decider", model_name="canon-stock")
    with eng.lora_off():
        try:
            if keep is not None:
                with torch.no_grad():
                    head.scale.fill_(1.0)
            for it in items[:JEVKIT_WARMUP]:
                ad.answer_item(it)
            return run_items(ad, items)
        finally:
            if keep is not None:
                with torch.no_grad():
                    head.scale.copy_(keep)


def _state_key(r: dict) -> str:
    """The state text key (== scripts/jev12_balanced_draw._request_key: sha256 of the canonical JSON of ``state``)."""
    s = json.dumps(r.get("state"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def slice_state_disjoint(slice_records: Path, train_records: Path) -> dict:
    """Fail closed (ValueError) if any selection-slice row shares an identical STATE text with any training row."""
    import gzip

    def keys(p):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return {_state_key(json.loads(x)) for x in f if x.strip()}

    s, t = keys(slice_records), keys(train_records)
    clash = s & t
    if clash:
        raise ValueError(f"{len(clash)} selection-slice state text(s) also appear in the training records")
    return {"slice_states": len(s), "train_states": len(t), "shared": 0}


def slice_first(jk: dict, decider, model_dir: str, out_dir: Path, data_dir: str | None) -> int:
    """JEV3-H2O: the held-out slice work (native_tfit's calib probabilities + T, and the step-0 floor vs the untrained
    host) done BEFORE any screen row, so the selection-slice evidence is fixed before public-231 is touched. Writes
    tfit.json, calib_probs.jsonl, slice_floor.json, slice_floor_rows.jsonl and slice_first.json (with UTC times)."""
    if not (jk.get("tfit") and jk.get("slice_floor") and jk.get("stock_shim")):
        print("identity: slice_first needs jevkit.tfit, slice_floor and stock_shim", file=sys.stderr)
        return 5
    if not data_dir:
        print("identity: slice_first needs screen score --data (the data root)", file=sys.stderr)
        return 10
    out_dir.mkdir(parents=True, exist_ok=True)
    ident = {"slice_first": {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    if jk.get("slice_state_disjoint"):  # JEV3-SYM: no slice row may share a STATE text with any training row
        try:
            ident["slice_state_disjoint"] = slice_state_disjoint(
                Path(data_dir) / jk["tfit"]["records"], Path(data_dir) / jk["slice_state_disjoint"]["train_records"])
        except (OSError, ValueError, KeyError) as e:
            print(f"identity: slice_state_disjoint: {type(e).__name__}: {e}", file=sys.stderr)
            return 10
    try:
        ident["tfit"] = fit_native_temperature(decider.backend, Path(data_dir) / jk["tfit"]["records"],
                                               out_dir / "tfit.json", float(jk["tfit"].get("lo", 0.25)),
                                               float(jk["tfit"].get("hi", 4.0)), jk["tfit"].get("ids_sha256"))
        eng = StockShimEngine(decider.backend, model_dir)
        pt = jk["tfit"].get("per_type")
        ident["slice_floor"] = slice_floor(eng, Path(data_dir) / jk["tfit"]["records"], out_dir / "calib_probs.jsonl",
                                           out_dir / "slice_floor.json", jk["tfit"].get("ids_sha256"),
                                           stock_band=pt is not None)
        if pt is not None:  # JEV3-LM: T per type; T_noul = min slice NLL s.t. band <= stock's (shim readout) + margin
            sb = ident["slice_floor"]["stock_noul_band"]["rate"]
            res = fit_per_type_from_calib(out_dir / "calib_probs.jsonl", float(jk["tfit"].get("lo", 0.25)),
                                          float(jk["tfit"].get("hi", 4.0)), int(pt.get("min_cell", 200)),
                                          stock_band=sb, margin=float(pt.get("band_margin", 0.05)))
            tf_path = out_dir / "tfit.json"
            tf = json.loads(tf_path.read_text(encoding="utf-8"))
            tf["per_type"] = res
            tf_path.write_text(json.dumps(tf, indent=1), encoding="utf-8")
            ident["tfit"]["per_type"] = res
    except (OSError, ValueError, KeyError) as e:
        print(f"identity: slice_first: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    ident["slice_first"]["done_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "slice_first.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
    return 0


def slice_floor(eng: "StockShimEngine", records_path: Path, calib_probs: Path, out: Path,
                expect_ids_sha256: str | None = None, stock_band: bool = False) -> dict:
    """A-PREREG-JEV2-H2O step-0 FLOOR on the fixed held-out jev2 slice (the calib slice; ids sha recorded): top-1 accuracy
    of the selected trained checkpoint (``calib_probs.jsonl``, written by ``fit_native_temperature``) vs the untrained
    host = step 0 (``eng`` with every LoRA off, the host's own shim prompt + tied LM head) on the SAME decisions. The
    primary readout (shim T 0.8 + the yes/no floor 0.801) never changes a decision's arg-max, so top-1 is the readout's
    own accuracy. ``pass`` iff the trained arm is strictly better; otherwise the result is "step 0 = no change"
    (negative). Per-family deltas are written beside it. Decisions whose gold label is not a shim option name are
    excluded from both arms and counted."""
    import gzip

    from train.run_cell import check_ids_pin, to_train_record

    with gzip.open(records_path, "rt", encoding="utf-8") as f:
        raws = [json.loads(x) for x in f if x.strip()]
    sha = check_ids_pin([r["id"] for r in raws], expect_ids_sha256, "slice floor")["sha"]  # both conventions logged
    native = {}
    for line in calib_probs.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ps = r["probs"]
            native[r["id"]] = (r.get("family"), r.get("type"), max(range(len(ps)), key=ps.__getitem__) == r["gold"])
    rows, unmatched = [], 0
    with eng.lora_off():
        for raw in raws:
            rec = to_train_record(raw)
            req, labels = rec["request"], rec.get("labels") or {}
            for qid in sorted(req["questions"]):
                key = f"{raw['id']}/{qid}"
                if key not in native or qid not in labels:
                    unmatched += 1
                    continue
                names, lp = eng.label_logps(req, qid)
                gold = str(labels[qid])
                if gold not in names:
                    unmatched += 1
                    continue
                fam, typ, n_ok = native[key]
                rows.append({"id": key, "family": fam, "type": typ, "native": bool(n_ok),
                             "step0": names[max(range(len(lp)), key=lp.__getitem__)] == gold})
                if stock_band and typ == "noul" and len(lp) == 2:  # the shim's own scored P(yes): its T + commit floor
                    rows[-1]["stock_p_shim"] = float(eng.mod.commit_noul(
                        eng.mod.probabilities(lp, eng.c.temperature)[0], eng.c.noul_floor))
    if not rows:
        raise ValueError("slice floor: no decision matched between the trained arm and the host")
    with open(out.parent / "slice_floor_rows.jsonl", "w", encoding="utf-8", newline="\n") as g:
        for r in rows:
            g.write(json.dumps(r, separators=(",", ":")) + "\n")

    def acc(rs, k):
        return sum(r[k] for r in rs) / len(rs)

    fams: dict = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    rep = {"ids_sha256": sha, "n_decisions": len(rows), "unmatched_excluded": unmatched,
           "native_top1": acc(rows, "native"), "step0_top1": acc(rows, "step0"),
           "per_family": {str(f): {"n": len(rs), "native": acc(rs, "native"), "step0": acc(rs, "step0"),
                                   "delta": acc(rs, "native") - acc(rs, "step0")} for f, rs in sorted(fams.items(),
                                                                                                      key=lambda x: str(x[0]))},
           "rule": "A-PREREG-JEV2-H2O floor: the selected checkpoint must beat step 0 (the untrained host) on the held-out "
                   "slice top-1; otherwise step 0 = no change and the run is NEGATIVE"}
    rep["delta"] = rep["native_top1"] - rep["step0_top1"]
    rep["pass"] = bool(rep["delta"] > 0)
    if stock_band:
        rep["stock_noul_band"] = noul_band_rate([r["stock_p_shim"] for r in rows if "stock_p_shim" in r])
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    keep = ("n_decisions", "native_top1", "step0_top1", "delta", "pass", "ids_sha256") + (("stock_noul_band",) if stock_band
                                                                                          else ())
    return {k: rep[k] for k in keep}


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


def fit_temperature(logps: list[list[float]], labels: list[int], lo: float = 0.25, hi: float = 4.0) -> dict:
    """Scalar T minimising the mean NLL of softmax(logp / T) at the gold option (golden-section on log T, bounded
    [lo, hi]); pure Python. ``logps`` = per question the head's per-option log-probabilities (any additive offset)."""
    import math

    if not logps or len(logps) != len(labels):
        raise ValueError("tfit: no calibration questions / labels mismatch")

    def nll(t: float) -> float:
        s = 0.0
        for lp, y in zip(logps, labels):
            z = [x / t for x in lp]
            m = max(z)
            s -= z[y] - (m + math.log(sum(math.exp(v - m) for v in z)))
        return s / len(logps)

    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(math.exp(c)), nll(math.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(math.exp(d))
    t = math.exp((a + b) / 2)
    return {"T": t, "nll_T": nll(t), "nll_1": nll(1.0), "n": len(logps), "bounds": [lo, hi],
            "at_bound": bool(abs(math.log(t) - math.log(lo)) < 1e-3 or abs(math.log(t) - math.log(hi)) < 1e-3)}


NOUL_BAND = (0.2, 0.8)  # jevkit scores a yes/no answer with p strictly inside this band as wrong (r2 rescore a8ca3e7e)


def noul_band_rate(ps: list[float]) -> dict:
    """Share of yes/no P(yes) values strictly inside NOUL_BAND."""
    n = len(ps)
    k = sum(1 for p in ps if NOUL_BAND[0] < float(p) < NOUL_BAND[1])
    return {"rate": (k / n) if n else None, "in_band": k, "n": n, "band": list(NOUL_BAND)}


def _p_true(lp: list[float], i_true: int, t: float) -> float:
    import math

    z = [x / t for x in lp]
    m = max(z)
    return math.exp(z[i_true] - m) / sum(math.exp(v - m) for v in z)


def constrain_t_by_band(fit: dict, noul: list[tuple[list[float], int]], steps: int = 60) -> dict:
    """JEV2-4B: the NLL-optimal T is kept only if it does not raise the yes/no band rate on the calib slice above its
    T = 1 value; otherwise T moves toward 1 (log grid) to the feasible value closest to the NLL optimum."""
    import math

    def band(t):
        return noul_band_rate([_p_true(lp, i, t) for lp, i in noul])["rate"]

    t_nll = float(fit["T"])
    b1, bt = band(1.0), band(t_nll)
    out = {"T_nll": t_nll, "band_T1": b1, "band_T_nll": bt, "constrained": False}
    if b1 is None or bt <= b1:
        return {**out, "T": t_nll, "band_T": bt}
    for j in range(1, steps + 1):  # from the NLL optimum toward 1
        t = math.exp(math.log(t_nll) * (1 - j / steps))
        if band(t) <= b1:
            return {**out, "T": t, "band_T": band(t), "constrained": True}
    return {**out, "T": 1.0, "band_T": b1, "constrained": True}


def retemper_answer(question: dict, answer: dict, t: float) -> dict:
    """The native SystemOne answer with its distribution re-tempered (p_i^(1/T), renormalised); derived fields
    recomputed by jevkit.systemone.normalize_answer. noul: P(yes) as the two-option distribution."""
    from jevkit.systemone import normalize_answer

    def temper(ps: list[float]) -> list[float]:
        w = [max(float(p), 1e-12) ** (1.0 / t) for p in ps]
        s = sum(w)
        return [x / s for x in w]

    if question["type"] == "noul":
        py = float(answer["noul"])
        return {"type": "noul", "noul": temper([py, 1.0 - py])[0]}
    names = list(answer["probabilities"])
    ps = temper([answer["probabilities"][n] for n in names])
    return normalize_answer(question, {"type": question["type"], "probabilities": dict(zip(names, ps))})


def slice_ids_sha256(record_ids) -> str:
    """The calibration slice identity (EV-PREREG-AREA-TEMP addendum): sha256 of the sorted RECORD ids, newline-joined
    (scripts/jev12_balanced_draw.py writes the same value as ``calib_ids_sha256``)."""
    return hashlib.sha256("\n".join(sorted(record_ids)).encode()).hexdigest()


def fit_native_temperature(backend, records_path: Path, out: Path, lo: float, hi: float,
                           expect_ids_sha256: str | None = None) -> dict:
    """GPU side of the tfit: the CALIB slice's rows (scripts/jev12_balanced_draw.py ``-calib``: >= 200 per family,
    hashed-id order, disjoint from the training draw by id AND content; never JevBench items) through the native
    backend (same template / layout / encoder as the jevkit native arm). Exports EVERY slice decision's per-option
    probabilities to calib_probs.jsonl (id, family, type, option ids, probs, gold index) so T_J / per-family T_f can be
    refitted offline without a rerun, then fits the scalar T (``fit_temperature``) into tfit.json, frozen before any
    eval answer is re-tempered. ``expect_ids_sha256`` (the cell config's pin, unless a placeholder) must match."""
    import gzip
    import math

    import torch

    from train.data import items_from_records
    from train.run_cell import to_train_record

    raw = records_path.read_bytes()
    with gzip.open(records_path, "rt", encoding="utf-8") as f:
        raws = [json.loads(x) for x in f if x.strip()]
    from train.run_cell import check_ids_pin

    sha = check_ids_pin([r["id"] for r in raws], expect_ids_sha256, "tfit slice")["sha"]  # both conventions logged
    fam_of = {r["id"]: next((v.get("family8") or v.get("family") for v in (r.get("aug") or {}).values()
                             if isinstance(v, dict) and (v.get("family8") or v.get("family"))), r.get("family"))
              for r in raws}
    items = items_from_records([to_train_record(r) for r in raws], backend.encode, layout=backend.layout,
                               template=backend.template)
    logps, labels, noul, cells = [], [], [], []
    m = backend.model
    was = m.training
    m.eval()
    try:
        with torch.no_grad(), open(out.parent / "calib_probs.jsonl", "w", encoding="utf-8", newline="\n") as g:
            for it in items:
                lp = m([it.row]).log_probs[0].float().cpu()[: it.row.n_options]
                if not torch.isfinite(lp).all():
                    raise ValueError(f"tfit: non-finite log-probs on {it.id}")
                v = lp.tolist()
                logps.append(v)
                labels.append(int(it.label))
                cells.append((it.row.qtype, it.row.n_options))
                if it.row.qtype == "noul" and "true" in list(it.row.option_ids):
                    noul.append((v, list(it.row.option_ids).index("true")))
                mx = max(v)
                z = math.log(sum(math.exp(x - mx) for x in v)) + mx
                rid = it.id.rsplit("/", 1)[0]
                g.write(json.dumps({"id": it.id, "family": fam_of.get(rid), "type": it.row.qtype,
                                    "option_ids": list(it.row.option_ids), "probs": [math.exp(x - z) for x in v],
                                    "gold": int(it.label)}, separators=(",", ":")) + "\n")
    finally:
        m.train(was)
    fit = fit_temperature(logps, labels, lo, hi)
    band = constrain_t_by_band(fit, noul)  # the yes/no band rate on the slice may not rise above its T = 1 value
    rep = fit | band | {
        "nll_T_nll": fit["nll_T"], "n_noul": len(noul),
        "records": str(records_path), "records_gz_sha256": hashlib.sha256(raw).hexdigest(), "ids_sha256": sha,
        "calib_probs": "calib_probs.jsonl (per-option probabilities of every slice decision, untempered)",
        "rule": "scalar T = the NLL minimiser over all calib questions (bounded), moved toward 1 if it would raise the "
                "slice's yes/no 0.2-0.8 band rate above T = 1's; fitted before any eval answer is tempered"}
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return {k: rep[k] for k in ("T", "T_nll", "constrained", "band_T1", "band_T", "n", "nll_1", "at_bound",
                                "ids_sha256")}


def fit_per_type(logps, labels, cells, noul, lo: float, hi: float, min_cell: int = 200) -> dict:
    """T per question type (choice / noul / score) and per (type, K) cell with >= ``min_cell`` slice decisions, each
    the bounded NLL minimiser on its own decisions; the noul T is moved toward 1 if it would raise the slice's yes/no
    0.2-0.8 band rate (``constrain_t_by_band``). A type with no slice decisions keeps T = 1."""
    out: dict = {"min_cell": min_cell, "types": {}, "cells": {}}
    for t in ("choice", "noul", "score"):
        idx = [i for i, (q, _) in enumerate(cells) if q == t]
        if not idx:
            out["types"][t] = {"T": 1.0, "n": 0}
            continue
        f = fit_temperature([logps[i] for i in idx], [labels[i] for i in idx], lo, hi)
        if t == "noul":
            f = f | constrain_t_by_band(f, noul)
        out["types"][t] = {"T": float(f["T"]), "n": len(idx), "nll_T": f.get("nll_T"), "nll_1": f.get("nll_1"),
                           "at_bound": f.get("at_bound")}
        for k in sorted({kk for q, kk in cells if q == t}):
            jdx = [i for i in idx if cells[i][1] == k]
            if len(jdx) >= min_cell and t != "noul":
                g = fit_temperature([logps[i] for i in jdx], [labels[i] for i in jdx], lo, hi)
                out["cells"][f"{t}:{k}"] = {"T": float(g["T"]), "n": len(jdx), "at_bound": g.get("at_bound")}
    return out


def fit_noul_t_banded(noul: list, labels: list, lo: float, hi: float, thr: float, steps: int = 400) -> dict:
    """T_noul = argmin slice NLL over a log grid of T in [lo, hi] SUBJECT TO the slice yes/no 0.2-0.8 band rate at T
    <= ``thr`` (stock's shim-readout band + margin). ``noul`` = [(log-probs, index of 'true')], ``labels`` = gold index.
    Infeasible everywhere -> the T with the lowest band rate (flagged)."""
    import math

    def nll(t):
        tot = 0.0
        for (lp, _), y in zip(noul, labels):
            z = [x / t for x in lp]
            m = max(z)
            tot += -(z[y] - (m + math.log(sum(math.exp(v - m) for v in z))))
        return tot / len(noul)

    grid = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (steps - 1)) for i in range(steps)]
    cand = [(t, nll(t), noul_band_rate([_p_true(lp, i, t) for lp, i in noul])["rate"]) for t in grid]
    feas = [c for c in cand if c[2] <= thr]
    if feas:
        t, v, b = min(feas, key=lambda c: (c[1], c[0]))
        return {"T": t, "nll_T": v, "band_T": b, "feasible": True, "threshold": thr, "n": len(noul),
                "nll_1": nll(1.0), "band_T1": noul_band_rate([_p_true(lp, i, 1.0) for lp, i in noul])["rate"]}
    t, v, b = min(cand, key=lambda c: (c[2], c[1]))
    return {"T": t, "nll_T": v, "band_T": b, "feasible": False, "threshold": thr, "n": len(noul), "nll_1": nll(1.0),
            "band_T1": noul_band_rate([_p_true(lp, i, 1.0) for lp, i in noul])["rate"],
            "flag": "no T in [lo, hi] meets the band threshold: T with the lowest band rate"}


def fit_per_type_from_calib(calib_probs: Path, lo: float, hi: float, min_cell: int, stock_band: float,
                            margin: float = 0.05) -> dict:
    """A-PREREG-JEV3-LM per-type T from the slice's exported probabilities (calib_probs.jsonl, untempered): choice and
    score T (+ (type, K) cells with >= min_cell decisions) = the bounded slice-NLL minimiser; T_noul = min slice NLL
    subject to the slice yes/no band rate <= stock's (shim readout, on the same slice) + ``margin``."""
    import math

    rows = [json.loads(x) for x in calib_probs.read_text(encoding="utf-8").splitlines() if x.strip()]
    logps = [[math.log(max(p, 1e-300)) for p in r["probs"]] for r in rows]
    labels = [int(r["gold"]) for r in rows]
    cells = [(r["type"], len(r["probs"])) for r in rows]
    out = fit_per_type(logps, labels, cells, [], lo, hi, min_cell)
    nidx = [i for i, r in enumerate(rows) if r["type"] == "noul" and "true" in r["option_ids"]]
    if nidx:
        noul = [(logps[i], rows[i]["option_ids"].index("true")) for i in nidx]
        thr = float(stock_band) + margin
        out["types"]["noul"] = fit_noul_t_banded(noul, [labels[i] for i in nidx], lo, hi, thr)
    out["rule"] = ("T_noul is chosen to minimise slice NLL subject to slice yes/no band rate <= stock's + "
                   f"{margin:g} (stock = the in-job stock + shim readout on the same slice); choice / score T = the "
                   "slice-NLL minimiser (+ (type, K) cells with >= min_cell decisions)")
    out["stock_noul_band"] = stock_band
    return out


def t_for(question: dict, t) -> float:
    """The frozen T of a question: a scalar, or a per_type map (its (type, K) cell if fitted, else its type)."""
    if isinstance(t, (int, float)):
        return float(t)
    typ = question["type"]
    k = 2 if typ == "noul" else len(question["criteria"])
    c = (t.get("cells") or {}).get(f"{typ}:{k}")
    return float(c["T"]) if c else float(((t.get("types") or {}).get(typ) or {}).get("T", 1.0))


def tfit_preds(src: Path, dst: Path, items, t) -> None:
    """native.preds.jsonl -> native_tfit.preds.jsonl with every ok answer re-tempered by the frozen T (score-only);
    ``t`` a scalar or the per_type map (``t_for``)."""
    by_id = {it.id: it for it in items}
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            a, it = r.get("answer"), by_id.get(r.get("id"))
            if r.get("status") == "ok" and isinstance(a, dict) and it is not None:
                r["answer"] = retemper_answer(it.question, a, t_for(it.question, t))
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
    arms, tables, axes, answers_by = {}, {}, {}, {}
    names = (["native"] + (["lean"] if jk.get("lean_template") else []) + sorted(jk.get("extra_templates") or {})
             + (["stock_shim"] if jk.get("stock_shim") else []) + (["canon_stock"] if jk.get("canon_stock") else []))
    if jk.get("shim_calibration"):  # score-only: the shim's own post-processing on the native answers (no GPU)
        shim_calibrate(d / "native.preds.jsonl", d / "native_shimcal.preds.jsonl", items)
        names.append("native_shimcal")
    tfit = None
    if jk.get("tfit"):  # JEV2-4B: the frozen calib-draw T applied to the native answers (score-only)
        tfit = json.loads((d / "tfit.json").read_text(encoding="utf-8"))
        tfit_preds(d / "native.preds.jsonl", d / "native_tfit.preds.jsonl", items,
                   tfit.get("per_type") or float(tfit["T"]))
        names.append("native_tfit")
    for name in names:
        pf = d / f"{name}.preds.jsonl"
        if not pf.is_file():
            raise ScreenError(f"jevkit {name} predictions missing ({pf})", 9)
        rep_path = d / f"{name}.jevkit.json"
        if JC.main(["score", "--public", "--pred", str(pf), "--price", jk["price"], "--out", str(rep_path)]) != 0:
            raise ScreenError(f"jevkit score {name} failed", 9)
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        preds = [json.loads(x) for x in pf.read_text(encoding="utf-8").splitlines() if x.strip()]
        if jk.get("assert_no_truncation"):  # JEV2-4B: no JevBench item may be refused / cut for length (fail closed)
            cut = [p["id"] for p in preds if "input_tokens_exceeded" in str(p.get("error") or "")]
            if cut:
                raise ScreenError(f"jevkit {name}: {len(cut)} item(s) refused for input length, e.g. {cut[:3]}", 9)
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
        answers_by[name] = preds_to_answers(preds)
        tables[name] = JS.build_table(items, answers_by[name])
        axes[name] = (rep["axes"]["speed"], rep["axes"]["cost"])
    out = {"price": jk["price_parsed"], "lean_template": jk.get("lean_template"),
           "extra_templates": jk.get("extra_templates") or {}, "arms": arms,
           "note": "open split only (the 231 pinned v1.2 public items, not the v1.6.1 P300 draw): Intelligence basis "
                   "open-only, no gap penalty; latency in-process on this H100 (self-hosted x2 + 0.15 s lane)"}
    if "stock_shim" in tables:  # A-PREREG-JEV1-4B: every trained arm minus the stock + shim baseline, paired
        out["vs_stock_shim"] = {nm: paired_vs(tables, axes, nm, "stock_shim") for nm in tables if nm != "stock_shim"}
        prim = jk.get("paired_primary") or "native"
        dp = out["vs_stock_shim"].get(prim)  # (was ``d``: shadowed the report dir used by the g1 / floor reads below)
        if dp is not None:
            lo = dp["composite_A"]["ci95"][0] if dp["composite_A"]["ci95"] else None
            out["primary"] = {"arm": prim, "baseline": "stock_shim", "delta_composite_A": dp["composite_A"],
                              "rule": "A-PREREG-JEV1-4B: SUCCESS iff delta Composite A > 0 with its 95% CI excluding 0",
                              "success": bool(lo is not None and lo > 0)}
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
    if tfit is not None:
        out["tfit"] = {k: tfit.get(k) for k in ("T", "T_nll", "constrained", "band_T1", "band_T", "band_T_nll", "n",
                                                "n_noul", "nll_T_nll", "nll_1", "at_bound", "ids_sha256",
                                                "records_gz_sha256")}
    # JEV2-4B reported gate: the yes/no 0.2-0.8 band rate per arm (jevkit counts those answers wrong)
    noul_ids = {it.id for it in items if it.question.get("type") == "noul"}
    out["noul_band"] = {nm: noul_band_rate([a["noul"] for i, a in answers_by[nm].items()
                                            if i in noul_ids and isinstance(a, dict) and "noul" in a])
                        for nm in answers_by}
    if "stock_shim" in out["noul_band"] and "native" in out["noul_band"]:
        nb, sb = out["noul_band"]["native"]["rate"], out["noul_band"]["stock_shim"]["rate"]
        out["noul_band_gate"] = {"rule": "native yes/no band rate <= stock_shim's (reported gate, A-PREREG-JEV2-4B Am. 2)",
                                 "native": nb, "stock_shim": sb, "pass": bool(nb is not None and sb is not None and nb <= sb)}
    if jk.get("perturb_ops"):  # JEV2-4B G1 (guarantee check beside the primary)
        g1p = d / "g1" / "native" / "perturb_report.json"
        g1 = json.loads((g1p if g1p.is_file() else d / "g1" / "perturb_report.json").read_text(encoding="utf-8"))
        out["g1"] = {k: g1.get(k) for k in ("ops", "n_base", "n_perturbed", "g1_pass", "g1_failures",
                                            "rows_all_identical", "duplicate_tie_items", "max_dp_bound", "status")} | {
            "per_operator": g1["flip_report"]["per_operator"],
            "rule": f"G1: 0 flips, 0 invalid, every op n > 0, rows bit-identical, max |dp| <= {g1.get('max_dp_bound')} "
                    "(jobs.sym_eval.perturb_eval.g1_verdict, fail closed; file order asserted)"}
        if jk.get("g1_keep_gate"):  # A-PREREG-JEV3-SYM: a checkpoint that loses invariance is not a candidate
            keep = bool(g1.get("g1_pass"))
            out["g1"]["keep_gate_pass"] = keep
            if "primary" in out:
                out["primary"]["g1_keep_gate_pass"] = keep
                if not keep:
                    out["primary"]["success"] = False
                    out["primary"]["candidate"] = False
                    out["primary"]["g1_verdict"] = "NOT A CANDIDATE: the trained checkpoint fails the bitwise G1 gate"
    if jk.get("flip_ops_all") and (d / "flip8" / "perturb_report.json").is_file():  # all 8 jevkit ops (report)
        f8 = json.loads((d / "flip8" / "perturb_report.json").read_text(encoding="utf-8"))
        fr = f8["flip_report"]
        out["flip_all_ops"] = {"per_operator": fr["per_operator"], "average_flip_rate": fr.get("average_flip_rate"),
                               "pooled": fr.get("pooled"), "reference": jk.get("flip_reference")}
    tp = d / "target_ref.preds.jsonl"
    if jk.get("target_ref"):  # A-PREREG-JEV3-SYM secondary: gap to the H2O target, paired on its EXISTING preds
        if tp.is_file():
            rp = d / "target_ref.jevkit.json"
            if JC.main(["score", "--public", "--pred", str(tp), "--price", jk["price"], "--out", str(rp)]) != 0:
                raise ScreenError("jevkit score target_ref failed", 9)
            rr = json.loads(rp.read_text(encoding="utf-8"))
            tpreds = [json.loads(x) for x in tp.read_text(encoding="utf-8").splitlines() if x.strip()]
            tables["target_ref"] = JS.build_table(items, preds_to_answers(tpreds))
            axes["target_ref"] = (rr["axes"]["speed"], rr["axes"]["cost"])
            prim = jk.get("paired_primary") or "native"
            out["gap_to_target"] = {"arm": prim, "target": jk["target_ref"].get("name"),
                                    "paired": paired_vs(tables, axes, prim, "target_ref"),
                                    "target_summary": {"composite": rr["scores"], "S": rr["axes"]["speed"],
                                                       "K": rr["axes"]["cost"]},
                                    "caveat": jk["target_ref"].get("caveat")}
        else:
            out["gap_to_target"] = {"status": "UNAVAILABLE: target_ref preds not shipped / sha mismatch"}
    cov = jk.get("covered_families")
    if cov and "stock_shim" in tables:  # JEV2-4B secondary: the paired deltas on covered vs uncovered families
        out["by_coverage"] = {}
        for grp in ("covered", "uncovered"):
            sub = [it for it in items if (it.family in cov) == (grp == "covered")]
            if not sub:
                continue
            st_ = {nm: JS.build_table(sub, answers_by[nm]) for nm in answers_by}
            out["by_coverage"][grp] = {"n_items": len(sub), "families": sorted({it.family for it in sub}),
                                       "vs_stock_shim": {nm: paired_vs(st_, axes, nm, "stock_shim")
                                                         for nm in st_ if nm != "stock_shim"}}
    if jk.get("by_family") and "stock_shim" in tables:  # A-PREREG-JEV2-H2O: per-family deltas on public-231 (report)
        prim = jk.get("paired_primary") or "native"
        out["by_family"] = {}
        for fam in sorted({it.family for it in items}):
            sub = [it for it in items if it.family == fam]
            st_ = {nm: JS.build_table(sub, answers_by[nm]) for nm in (prim, "stock_shim") if nm in answers_by}
            if len(st_) == 2:
                d_ = paired_vs(st_, axes, prim, "stock_shim", B=200)
                out["by_family"][fam] = {"n_items": len(sub), "arm": prim, "delta_I": d_["I"],
                                         "delta_composite_A": d_["composite_A"]}
    if jk.get("slice_floor"):  # A-PREREG-JEV2-H2O: the selected checkpoint must beat step 0 on the held-out slice
        fp = d / "slice_floor.json"
        fl = json.loads(fp.read_text(encoding="utf-8")) if fp.is_file() else None
        out["slice_floor"] = fl if fl is not None else {"pass": None, "status": "UNEVALUATED (see identity.json)"}
        if "primary" in out:
            out["primary"]["slice_floor_pass"] = None if fl is None else fl["pass"]
            if fl is None or not fl["pass"]:
                out["primary"]["success"] = False
                out["primary"]["floor_verdict"] = ("NEGATIVE: no trained checkpoint beats step 0 (H2O untrained) on the "
                                                   "held-out slice; reported as step 0 = no change"
                                                   if fl is not None else "NEGATIVE: the step-0 floor was not evaluated")
    if jk.get("letter_prior"):  # JEV3-LM: the gold-letter shortcut check (canonical letters = canonical positions)
        out["letter_prior"] = letter_prior_report(d, items, answers_by.get(jk.get("paired_primary") or "native"),
                                                  jk["letter_prior"])
    if jk.get("score_inv"):  # JEV3-TYPED KEEP gate 2: the trained arm must PASS ev_score_flip (bitwise, every op n > 0)
        out["score_inv"] = score_inv_report(d / "score_inv")
        keep = bool(out["score_inv"].get("native", {}).get("score_inv_pass"))
        out["score_inv"]["keep_gate_pass"] = keep
        if "primary" in out:
            out["primary"]["score_inv_keep_gate_pass"] = keep
            if not keep:
                out["primary"]["success"] = False
                out["primary"]["candidate"] = False
                out["primary"]["score_inv_verdict"] = ("NOT A CANDIDATE: the trained checkpoint fails the score-item "
                                                       "invariance gate (ev_score_flip, bitwise)")
    return out


def _canon_pos(question: dict, oid: str) -> int | None:
    from decide.canon_shim import canonical_question

    try:
        return canonical_question(question)["ids"].index(str(oid))
    except (ValueError, KeyError):
        return None


def letter_prior_report(d: Path, items, answers: dict | None, spec: dict) -> dict:
    """Gold-letter shortcut check (A-PREREG-JEV3-LM): under the CANONICAL order, the gold letter's distribution vs the
    model's predicted-letter distribution (a) on the jev3 slice (calib_probs.jsonl: option_ids are the canonical
    order) and (b) on the public CHOICE items (format-level: the canonical position of the gold and of the primary
    arm's choice); FLAG when the model's A share exceeds the gold A share by more than ``flag_pp``; slice choice
    accuracy with and without the ``ambiguous`` family."""
    flag_pp = float(spec.get("flag_pp", 5.0))

    def dist(xs, k=8):
        xs = [x for x in xs if x is not None]
        n = len(xs)
        return {"n": n, **{chr(65 + i): (sum(1 for x in xs if x == i) / n if n else None) for i in range(k)}}

    out: dict = {"rule": f"FLAG if the model's predicted-A share exceeds the gold-A share by > {flag_pp:g} pp (choice)"}
    cp = d / "calib_probs.jsonl"
    if cp.is_file():
        rows = [json.loads(x) for x in cp.read_text(encoding="utf-8").splitlines() if x.strip()]
        ch = [r for r in rows if r.get("type") == "choice"]
        gold = [r["gold"] for r in ch]
        pred = [max(range(len(r["probs"])), key=r["probs"].__getitem__) for r in ch]
        sl = {"gold": dist(gold), "pred": dist(pred)}
        ga, pa = sl["gold"].get("A"), sl["pred"].get("A")
        sl["A_excess_pp"] = None if ga is None or pa is None else 100 * (pa - ga)
        sl["flag"] = bool(sl["A_excess_pp"] is not None and sl["A_excess_pp"] > flag_pp)
        acc = lambda rs: (sum(1 for r in rs if max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"])  # noqa: E731
                          / len(rs)) if rs else None
        sl["choice_acc"] = acc(ch)
        sl["choice_acc_without_ambiguous"] = acc([r for r in ch if r.get("family") != "ambiguous"])
        sl["n_choice_ambiguous"] = sum(1 for r in ch if r.get("family") == "ambiguous")
        out["slice"] = sl
    else:
        out["slice"] = {"status": "UNEVALUATED (no calib_probs.jsonl)"}
    if answers is not None:
        gold, pred = [], []
        for it in items:
            if it.type != "choice" or it.expected is None:
                continue
            a = answers.get(it.id)
            gold.append(_canon_pos(it.question, it.expected))
            pred.append(_canon_pos(it.question, a.get("choice")) if isinstance(a, dict) and a.get("choice") is not None
                        else None)
        pub = {"gold": dist(gold), "pred": dist(pred)}
        ga, pa = pub["gold"].get("A"), pub["pred"].get("A")
        pub["A_excess_pp"] = None if ga is None or pa is None else 100 * (pa - ga)
        pub["flag"] = bool(pub["A_excess_pp"] is not None and pub["A_excess_pp"] > flag_pp)
        out["public_choice"] = pub
    out["flag"] = bool((out.get("slice") or {}).get("flag") or (out.get("public_choice") or {}).get("flag"))
    return out


def score_inv_report(d: Path) -> dict:
    """Per arm: ev_score_flip's score_inv.json (pass, bitwise, per op n / invalid / flips / max|dp| / |dE[level]|).
    A missing report = FAIL (fail closed)."""
    out: dict = {"rule": "KEEP gate 2 (A-PREREG-JEV3-TYPED): the trained (native) arm PASSES scripts/ev_score_flip.py "
                         "--max-dp 0 on every public score item under score_reverse / score_shuffle (seeds 0,1,2) / "
                         "score_relabel / score_permute (claude/ev-score-inv @c47c0a46); the stock-shim control is "
                         "reported for teeth; the evidence rescore re-runs the check from the shipped preds"}
    for nm in ("native", "stock_shim"):
        f = d / nm / "score_inv.json"
        if not f.is_file():
            out[nm] = {"score_inv_pass": False, "status": "MISSING (fail closed)"}
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        out[nm] = {"score_inv_pass": r["score_inv_pass"], "bitwise": r["bitwise"], "failures": r["failures"],
                   "per_operator": {op: {k: v.get(k) for k in ("n", "invalid", "flips", "max_abs_dp", "max_abs_dE",
                                                                "mean_abs_dE")} for op, v in r["per_operator"].items()}}
    return out


def paired_vs(tables: dict, axes: dict, a: str, b: str, B: int = 1000, seed: int = 16) -> dict:
    """a - b on the SAME stratified resamples (jevkit: B 1000, seed 16, strata of jevkit.scoring.strata): open-split
    Intelligence, Calibration, Capability and Composite A (Speed / Cost held at each arm's measured axes, as
    jevkit.scoring.bootstrap does). Point = the full-set difference."""
    import numpy as np

    from jevkit import scoring as JS

    def axes_of(nm, idx=None):
        r = JS.aggregate(tables[nm], idx, detail=False)
        i, c = r["intelligence"]["I"], r["calibration"]["score"]
        sp, k = axes[nm]
        return {"I": i, "C": c, "capability": JS.capability(i, c), "composite_A": JS.composite(i, c, sp, k, "A")}

    pa, pb = axes_of(a), axes_of(b)
    rng = np.random.default_rng(seed)
    st = JS.strata(tables[b])
    draws: dict = {k: [] for k in pa}
    for _ in range(B):
        idx = np.concatenate([rng.choice(s_, size=len(s_), replace=True) for s_ in st])
        xa, xb = axes_of(a, idx), axes_of(b, idx)
        for k in draws:
            if xa[k] is not None and xb[k] is not None:
                draws[k].append(xa[k] - xb[k])

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None

    return {k: {"point": (pa[k] - pb[k]) if pa[k] is not None and pb[k] is not None else None, "ci95": ci(draws[k]),
                "a": pa[k], "b": pb[k]} for k in draws} | {"B": B, "seed": seed, "a": a, "b": b}


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
    s.add_argument("--data", default=None, help="the data root (jevkit.tfit / slice_floor records: the calib slice)")
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
