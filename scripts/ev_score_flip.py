"""Score-item invariance flips from files (evidence lane; reported ALONGSIDE jevkit G1, never replacing it).

  python -I -X utf8 scripts/ev_score_flip.py (--public | --items <items.jsonl>) --dir <dir> --out <outdir> \
      [--base-preds <p>] [--pert-preds <p>] [--ops score_reverse,score_shuffle,score_relabel,score_permute] [--max-dp 0]

<dir> holds the requests written by scripts/ev_score_perturb.py (base.jsonl, perturbed.jsonl) and the head's preds on
them (base.preds.jsonl, perturbed.preds.jsonl; the jevkit preds schema {id, answer, status}). A score answer must carry
``probabilities`` keyed by exactly the request's handles (``labels``); a label-only answer {"choice": <handle>} is a
point mass; anything else is invalid. The base preds MUST come from base.jsonl (level-as-content rendering).

Per op: n valid pairs, invalid, flips (argmax level changes after mapping the perturbed distribution back to levels
through ``label_map``; ties to the smallest level, a content rule), max|dp| over levels, and the expected-level shift
|E[level] - E'[level]| (the quantity the score metric uses), mean and max.

FAIL CLOSED. PASS iff for every required op: n > 0, 0 invalid, 0 flips, max|dp| <= --max-dp (0 = bitwise) and
max|dE| <= --max-dp * (K-1); every ON-DISK perturbed row lists its options in the regenerated in-memory order (a
sort_keys dump fails here or as vacuous) and really differs from base (display order or handles); state,
instructions, scale and the content multiset equal the base request's; base.jsonl equals the regenerated base requests.
Writes <out>/score_inv.json and <out>/SCORE_INV.md. Exit 0 PASS, 3 FAIL, 2 inputs missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevkit.answers import argmax_label, validate_distribution  # noqa: E402
from scripts import ev_score_perturb as SP  # noqa: E402

RULE = ("SCORE-INV PASS iff every required op n>0, 0 invalid, 0 flips, max|dp| <= bound (0 = bitwise), max|dE| <= "
        "bound*(K-1), every on-disk row in the in-memory order, changed vs base, content-identical to base; separate "
        "from jevkit G1 (which never perturbs score items)")


def rj(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _answers(preds: list[dict]) -> dict:
    out = {}
    for r in preds:
        out[str(r["id"])] = r.get("answer") if r.get("status", "ok") == "ok" else None
    return out


def dist(answer, handles: list[str]) -> dict[str, float] | None:
    """Distribution over the request's handles, or None (invalid). Never repairs beyond jevkit's renorm band."""
    if not isinstance(answer, dict) or answer.get("type") not in (None, "score"):
        return None
    pr = answer.get("probabilities")
    if pr is None:
        c = answer.get("choice")
        return {h: float(h == str(c)) for h in handles} if c is not None and str(c) in handles else None
    try:
        d, _, _ = validate_distribution(pr, list(handles))
    except ValueError:
        return None
    return d


def exp_level(d: dict[str, float]) -> float:
    return sum(int(k) * d[k] for k in sorted(d, key=int))  # fixed (canonical) summation order


def _regen(items_by: dict, rid: str) -> dict:
    base, op, seed = rid.rsplit("~", 2)
    p = SP.perturb(items_by[base], op, int(seed))
    if p is None or p["id"] != rid:
        raise ValueError(f"{rid}: ev_score_perturb does not generate this request")
    return p


def score_inv_check(items, base_rows: list[dict], pert_rows: list[dict], base_preds: list[dict],
                    pert_preds: list[dict], max_dp: float = 0.0, ops=SP.OPS) -> dict:
    ops = list(ops)
    by = {it.id: it for it in items if it.type == "score"}
    fails: list[str] = []
    notes: list[str] = []
    # ---- base requests faithful
    bmap = {r["id"]: r for r in base_rows}
    base_ok = 0
    for iid, it in by.items():
        r = bmap.get(iid)
        want = SP.base_request(it)
        if r is not None and r == want and list(r["question"]["options"]) == list(want["question"]["options"]):
            base_ok += 1
    if base_ok != len(by) or not by:
        fails.append(f"base.jsonl: {base_ok}/{len(by)} score items have a base request equal to the regenerated one "
                     "(level-as-content rendering, canonical order)")
    extra = sorted(set(bmap) - set(by))
    if extra:
        notes.append(f"base.jsonl: {len(extra)} rows that are not score items of this item set (ignored)")
    ba, pa = _answers(base_preds), _answers(pert_preds)
    per: dict[str, dict] = {op: {"n_rows": 0, "n": 0, "invalid": 0, "flips": 0, "max_abs_dp": None, "max_abs_dE": None,
                                 "mean_abs_dE": None, "matches_regenerated": 0, "changed_vs_base": 0,
                                 "content_identical": 0, "flip_ids": [], "not_matching_ids": [], "_dE": [], "_dp": []}
                            for op in ops}
    for r in pert_rows:
        m = r.get("perturbation") or {}
        op = m.get("op")
        if op not in per:
            continue
        o = per[op]
        o["n_rows"] += 1
        it = by.get(m.get("base_id"))
        if it is None:
            fails.append(f"{r.get('id')}: base id {m.get('base_id')} is not a score item")
            continue
        try:
            want = _regen(by, r["id"])
        except Exception as e:  # noqa: BLE001
            fails.append(str(e))
            continue
        q = r.get("question") or {}
        disk_order = list(q.get("options") or {})
        same = (disk_order == list(want["question"]["options"]) and list(r.get("labels") or []) == disk_order
                and r == want)
        o["matches_regenerated"] += int(same)
        if not same and len(o["not_matching_ids"]) < 10:
            o["not_matching_ids"].append(r["id"])
        lm = {str(k): str(v) for k, v in (m.get("label_map") or {}).items()}
        inv = {v: k for k, v in lm.items()}
        if op in SP.REORDER_OPS:
            changed = [inv.get(h, h) for h in disk_order] != list(it.labels)
        else:
            changed = any(k != v for k, v in lm.items())
        o["changed_vs_base"] += int(changed)
        b = SP.base_request(it)
        content_same = (r.get("state") == b["state"] and q.get("instructions") == b["question"]["instructions"]
                        and q.get("scale") == b["question"]["scale"]
                        and sorted((q.get("options") or {}).values()) == sorted(b["question"]["options"].values())
                        and all((q.get("options") or {}).get(lm.get(lab)) == b["question"]["options"][lab] for lab in it.labels))
        o["content_identical"] += int(content_same)
        # ---- preds
        db = dist(ba.get(it.id), list(it.labels))
        dq = dist(pa.get(r["id"]), list(r.get("labels") or disk_order))
        if db is None or dq is None or sorted(lm) != sorted(it.labels) or set(lm.values()) != set(dq):
            o["invalid"] += 1
            continue
        back = {lab: dq[lm[lab]] for lab in it.labels}
        o["n"] += 1
        if argmax_label(db) != argmax_label(back):
            o["flips"] += 1
            if len(o["flip_ids"]) < 20:
                o["flip_ids"].append(r["id"])
        o["_dp"].append(max(abs(db[lab] - back[lab]) for lab in it.labels))
        o["_dE"].append((abs(exp_level(db) - exp_level(back)), it.k))
    kmax = max((it.k for it in by.values()), default=2)
    for op in ops:
        o = per[op]
        dps, dEs = o.pop("_dp"), o.pop("_dE")
        o["max_abs_dp"] = max(dps) if dps else None
        o["max_abs_dE"] = max(x for x, _ in dEs) if dEs else None
        o["mean_abs_dE"] = (sum(x for x, _ in dEs) / len(dEs)) if dEs else None
        o["flip_rate"] = (o["flips"] / o["n"]) if o["n"] else None
        if not o["n"]:
            fails.append(f"{op}: no valid pairs (n=0; vacuous)")
            continue
        if o["invalid"]:
            fails.append(f"{op}: {o['invalid']} invalid")
        if o["flips"]:
            fails.append(f"{op}: {o['flips']} flips")
        if o["max_abs_dp"] > max_dp:
            fails.append(f"{op}: max|dp| {o['max_abs_dp']} > {max_dp}")
        if any(x > max_dp * (k - 1) for x, k in dEs):
            fails.append(f"{op}: max|dE[level]| {o['max_abs_dE']} > {max_dp}*(K-1)")
        if o["matches_regenerated"] != o["n_rows"]:
            fails.append(f"{op}: {o['n_rows'] - o['matches_regenerated']} on-disk rows NOT in the in-memory order / "
                         "not the regenerated request (sort_keys dump?)")
        if o["changed_vs_base"] == 0:
            fails.append(f"{op}: no on-disk row differs from base order / handles (vacuous)")
        elif o["changed_vs_base"] != o["n_rows"]:
            fails.append(f"{op}: {o['n_rows'] - o['changed_vs_base']} on-disk rows keep base order / handles (vacuous pairs)")
        if o["content_identical"] != o["n_rows"]:
            fails.append(f"{op}: {o['n_rows'] - o['content_identical']} rows differ from base beyond order / handles")
    seen = [per[op]["max_abs_dp"] for op in ops if per[op]["max_abs_dp"] is not None]
    mx = max(seen) if seen else None
    ok = not fails
    return {"schema": "ev-score-inv/1", "rendering": SP.RENDERING, "ops": ops, "n_score_items": len(by),
            "n_base_rows_faithful": base_ok, "n_perturbed_rows": sum(per[o]["n_rows"] for o in ops),
            "per_operator": per, "max_abs_dp_observed": mx, "max_dp_bound": max_dp, "k_max": kmax,
            "score_inv_pass": ok, "bitwise": bool(ok and mx == 0.0),
            "non_bitwise_within_tolerance": bool(ok and mx is not None and mx > 0),
            "failures": fails, "notes": notes, "rule": RULE,
            "separate_from_g1": "jevkit G1 (option_reverse / option_shuffle / key_relabel) never perturbs score items; "
                                "this check does not change or replace the G1 verdict"}


def render_md(r: dict, title: str = "Score-item invariance (separate from G1)") -> str:
    L = [f"# {title}: {'PASS' if r['score_inv_pass'] else 'FAIL'}", "",
         f"{r['n_score_items']} score items, {r['n_perturbed_rows']} perturbed rows, base requests faithful "
         f"{r['n_base_rows_faithful']}/{r['n_score_items']}; bitwise {r['bitwise']}, max|dp| {r['max_abs_dp_observed']} "
         f"(bound {r['max_dp_bound']}). Rendering: {r['rendering']}.", "",
         "| op | n | invalid | flips | max abs dp | max abs dE[level] | mean abs dE | on-disk in-memory order | changed vs base | content = base |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for op, v in r["per_operator"].items():
        L.append(f"| {op} | {v['n']} | {v['invalid']} | {v['flips']} | {v['max_abs_dp']} | {v['max_abs_dE']} | "
                 f"{v['mean_abs_dE']} | {v['matches_regenerated']}/{v['n_rows']} | {v['changed_vs_base']}/{v['n_rows']} | "
                 f"{v['content_identical']}/{v['n_rows']} |")
    L += ["", *(f"- FAIL: {x}" for x in r["failures"]), *(f"- note: {x}" for x in r["notes"]), "", f"Rule: {r['rule']}.",
          f"{r['separate_from_g1']}.", ""]
    return "\n".join(L)


def run_dir(items, d: Path, base_preds=None, pert_preds=None, max_dp: float = 0.0, ops=SP.OPS) -> dict | None:
    d = Path(d)
    need = [d / "base.jsonl", d / "perturbed.jsonl", Path(base_preds or d / "base.preds.jsonl"),
            Path(pert_preds or d / "perturbed.preds.jsonl")]
    miss = [str(p) for p in need if not p.is_file()]
    if miss:
        return {"missing": miss}
    return score_inv_check(items, rj(need[0]), rj(need[1]), rj(need[2]), rj(need[3]), max_dp, ops)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -X utf8 scripts/ev_score_flip.py", description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--public", action="store_true")
    g.add_argument("--items", default=None)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-preds", default=None)
    ap.add_argument("--pert-preds", default=None)
    ap.add_argument("--ops", default=",".join(SP.OPS))
    ap.add_argument("--max-dp", type=float, default=0.0)
    a = ap.parse_args(argv)
    from jevkit.items import load_items, load_public
    items = load_public() if a.public else load_items(a.items)
    res = run_dir(items, Path(a.dir), a.base_preds, a.pert_preds, a.max_dp, [o for o in a.ops.split(",") if o])
    if "missing" in res:
        print(json.dumps(res), file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score_inv.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (out / "SCORE_INV.md").write_text(render_md(res), encoding="utf-8")
    print(json.dumps({"score_inv_pass": res["score_inv_pass"], "bitwise": res["bitwise"], "failures": res["failures"],
                      "per_op": {k: {x: v[x] for x in ("n", "invalid", "flips", "max_abs_dp", "max_abs_dE")}
                                 for k, v in res["per_operator"].items()}}, indent=1))
    return 0 if res["score_inv_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
