"""Reproduce the published JevBench v1.6.1 board from its own published sub-fields.

For every system row of ``/api/jevbench/v1.6.1`` this recomputes, with the SAME formulas the item scorer uses:

=========  =============================================================  ======================================
layer      recomputed from                                                compared with
=========  =============================================================  ======================================
composite  ``axes`` (I, C, S, K)                                          ``scores.A`` / ``.B`` / ``.C``
capability ``axes`` (I, C)                                                ``capability``
cc         ``intelligence.per_type_split.*.tiers`` (+ clip at 0)          ``per_type_split.*.cc``
I_split    the six published ``cc``                                       ``I_open`` / ``I_sealed``
I          ``I_open``, ``I_sealed``, ``G_med``                            base / gap / excess / penalty / axes.I
C parts    ``calibration.parts`` ECE / nRPS                               parts ``score``
C          the three part scores (missing supported part -> 0)            ``axes.calibration``
S          ``speed.p50_s_adjusted`` / ``p95_s_adjusted`` (+ raw x2+0.15)  ``axes.speed`` (+ adjusted latencies)
K          ``cost.usd_per_1000``                                          ``axes.cost``
views      ``axes`` (I, S, K)                                             ``views`` (sort views; not scored)
=========  =============================================================  ======================================

Bootstrap intervals and anything below the published aggregates (item answers, sealed items) are NOT derivable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import scoring as SC
from . import spec

TOL = 1e-9


def recompute_row(row: dict, g_med: float = spec.G_MED_V161) -> dict[str, tuple[float | None, float | None]]:
    """{check name: (recomputed, published)} for one published system row."""
    out: dict[str, tuple[Any, Any]] = {}
    ax = row.get("axes") or {}
    i, c, s, k = (ax.get(x) for x in ("intelligence", "calibration", "speed", "cost"))
    for o in spec.OPTIONS:
        out[f"composite_{o}"] = (SC.composite(i, c, s, k, o), (row.get("scores") or {}).get(o))
    out["capability"] = (SC.capability(i, c), row.get("capability"))

    it = row.get("intelligence") or {}
    pts = it.get("per_type_split") or {}
    cc_side: dict[str, dict[str, float | None]] = defaultdict(dict)
    for key, v in pts.items():
        side, typ = key.split("|")
        cc = SC.tiered_cc(v.get("tiers") or {})
        out[f"cc[{key}]"] = (None if cc is None else max(0.0, cc), v.get("cc"))
        cc_side[side][typ] = v.get("cc")
    for side in ("open", "sealed"):
        if cc_side.get(side):
            out[f"I_{side}"] = (SC.split_intelligence(cc_side[side]), it.get(f"I_{side}"))
    intel = SC.intelligence(it.get("I_open"), it.get("I_sealed"), g_med)
    for f in ("base", "gap", "excess", "penalty"):
        out[f] = (intel[f], it.get(f))
    out["intelligence"] = (intel["I"], i)

    parts = (row.get("calibration") or {}).get("parts") or {}
    rec_parts: dict[str, float | None] = {}
    if (parts.get("choice") or {}).get("ece_hard") is not None:
        rec_parts["choice"] = SC.calibration_choice(parts["choice"]["ece_hard"])
        out["cal_part[choice]"] = (rec_parts["choice"], parts["choice"].get("score"))
    if (parts.get("noul") or {}).get("ece") is not None:
        rec_parts["noul"] = SC.calibration_noul(parts["noul"]["ece"])
        out["cal_part[noul]"] = (rec_parts["noul"], parts["noul"].get("score"))
    if (parts.get("score") or {}).get("top_ece") is not None:
        rec_parts["score"] = SC.calibration_score(parts["score"]["top_ece"], parts["score"]["nrps"])
        out["cal_part[score]"] = (rec_parts["score"], parts["score"].get("score"))
    for typ in spec.TYPES:  # "provider_confidence_only" rows (fastino-gliner-2-5-decide): 100 (1 - 2 topECE) per type
        pt = parts.get(typ) or {}
        if typ not in rec_parts and pt.get("top_label_ece") is not None:
            rec_parts[typ] = SC.calibration_choice(pt["top_label_ece"])
            out[f"cal_part[{typ}]"] = (rec_parts[typ], pt.get("score"))
    supported = [t for t, v in (row.get("support") or {}).items() if v and v != "not supported"]
    if supported and c is not None:
        out["calibration"] = (SC.calibration(rec_parts, supported), c)

    sp = row.get("speed") or {}
    if sp.get("p50_s_adjusted") is not None and sp.get("p95_s_adjusted") is not None:
        out["speed"] = (SC.speed(sp["p50_s_adjusted"], sp["p95_s_adjusted"], adjusted=True)["score"], s)
        if "none" not in str(sp.get("adjustment", "none")) and sp.get("p50_s_raw") is not None:
            out["p50_adjusted"] = (SC.adjust_latency(sp["p50_s_raw"], True), sp["p50_s_adjusted"])
            out["p95_adjusted"] = (SC.adjust_latency(sp["p95_s_raw"], True), sp["p95_s_adjusted"])
    usd = (row.get("cost") or {}).get("usd_per_1000")
    if usd is not None:
        out["cost"] = (SC.cost_score(usd), k)
    v = SC.views(i, s, k)
    for name, pub in (row.get("views") or {}).items():
        out[f"view[{name}]"] = (v.get(name), pub)
    return out


def verify(api: dict, tol: float = TOL) -> dict:
    """Per check: rows compared, exact matches (|diff| <= tol), max |diff|, and every mismatching row."""
    g_med = api.get("G_med", spec.G_MED_V161)
    checks: dict[str, dict] = {}
    for row in api["systems"]:
        for name, (rec, pub) in recompute_row(row, g_med).items():
            fam = name.split("[")[0] if name.startswith(("cc[", "cal_part[", "view[")) else name
            if name.startswith(("cal_part[", "view[")):
                fam = name  # keep parts / views separate
            d = checks.setdefault(fam, {"rows": 0, "exact": 0, "both_null": 0, "max_abs_diff": 0.0, "mismatches": []})
            if rec is None and pub is None:
                d["both_null"] += 1
                continue
            d["rows"] += 1
            if rec is None or pub is None:
                d["mismatches"].append({"key": row.get("key"), "check": name, "recomputed": rec, "published": pub})
                continue
            diff = abs(rec - pub)
            d["max_abs_diff"] = max(d["max_abs_diff"], diff)
            if diff <= tol:
                d["exact"] += 1
            else:
                d["mismatches"].append({"key": row.get("key"), "check": name, "recomputed": rec, "published": pub,
                                        "abs_diff": diff})
    return {"revision": api.get("revision"), "source_sha256": api.get("source_sha256"), "g_med": g_med,
            "n_systems": len(api["systems"]), "tolerance": tol, "checks": checks}


def verify_file(path: str | Path, tol: float = TOL) -> dict:
    return verify(json.loads(Path(path).read_text(encoding="utf-8")), tol)


def summary_lines(rep: dict) -> list[str]:
    lines = [f"JevBench {rep['revision']} board ({rep['n_systems']} rows), tolerance {rep['tolerance']}:"]
    for name, d in sorted(rep["checks"].items()):
        lines.append(f"  {name:42s} {d['exact']:4d}/{d['rows']:<4d} exact  max|diff| {d['max_abs_diff']:.3g}"
                     + (f"  ({len(d['mismatches'])} mismatches)" if d["mismatches"] else ""))
    return lines
