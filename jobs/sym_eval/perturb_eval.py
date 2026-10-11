"""jevkit perturb operators 4-6 (option_reverse, option_shuffle, key_relabel) on any decision-head checkpoint, for
the guarantee G1 of A-PREREG-JEV1-4B-SYM (docs/PROOF_INVARIANCE.md). Eval-only: the pinned public JevBench items.

In process: ``run(decider, items, out_dir, row_fn=...)`` answers the base items and their perturbed copies serially
through ``jevkit.systemone.SystemOneAdapter`` (the same adapter ``jevkit score`` predictions come from), writes the
files ``python -m jevkit perturb score`` reads, and the flip report (``jevkit.perturb.flip_report``, frozen). With
``row_fn(request) -> LayoutRow`` it also checks the rows themselves: under a sym-fork layout every reorder / relabel
perturbation must give the SAME prompt ids, continuations and their order (the bit-identity argument of the proof).
Exact duplicates (``tie_set`` of more than one id) are reported apart: a single id among them cannot be invariant.

GPU / any checkpoint (the screen's own loader, ``jobs.m2_first_cell.fulldev_eval.load_child``, untempered arm)::

    python -m jobs.sym_eval.perturb_eval --cell CELL.json --ckpt-dir DIR [--ckpt FILE] --model-dir SNAP \
        --guard-readiness G --records R --train-dir T --pool-dir P --device cuda --out-dir OUT [--layout sym-fork]

``--layout`` scores a checkpoint under another layout (e.g. the JEV1-4B g1v2 checkpoint, for its flip-rate baseline).
Gate verdict (``g1_verdict``, fail closed): ``g1_pass`` = every requested operator present with n > 0, zero flips,
zero invalid, max |dp| <= ``--max-dp`` (default 0: bit-identity, as the proof claims; ``max_abs_dp_observed`` is
reported apart) on every operator, AND ``rows_all_identical`` (the rows themselves; no row check = no pass).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

REORDER_RELABEL_OPS = ("option_reverse", "option_shuffle", "key_relabel")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False, default=str) + "\n")


def row_identity(items, perturbed, row_fn: Callable[[dict], Any]) -> dict:
    """Per operator: perturbations whose row equals the base row (ids, continuations, order) with option ids mapped
    through the operator's label map (choice); and the ones that differ (must be 0 under sym-fork)."""
    from jevkit.systemone import item_request

    base = {it.id: it for it in items}
    out: dict[str, dict] = {}
    for p in perturbed:
        b = base[p.base_id]
        rb, rp = row_fn(item_request(b)), row_fn(item_request(p.item))
        same = rb.ids == rp.ids and rb.continuations == rp.continuations
        if same and b.type == "choice":
            same = [p.label_map.get(o, o) for o in rb.option_ids] == rp.option_ids
        r = out.setdefault(p.op, {"n": 0, "identical": 0, "differ_ids": []})
        r["n"] += 1
        r["identical"] += int(same)
        if not same:
            r["differ_ids"].append(p.item.id)
    return out


def run(decider: Any, items: list, out_dir: str | Path, ops: Iterable[str] = REORDER_RELABEL_OPS,
        seeds: Iterable[int] = (0,), row_fn: Callable[[dict], Any] | None = None, max_dp: float = 0.0,
        kind: str = "decider") -> dict:
    from jevkit import perturb
    from jevkit.systemone import SystemOneAdapter, preds_to_answers, run_items

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ops = list(ops)
    ad = SystemOneAdapter(decider, kind=kind, model_name="sym-eval")
    base = run_items(ad, items)
    ps = perturb.make_perturbations(items, ops, list(seeds))
    pert = run_items(ad, [p.item for p in ps])
    _write_jsonl(out / "base.preds.jsonl", base)
    _write_jsonl(out / "perturbed.preds.jsonl", pert)
    # NOT sort_keys: a sorted dump puts every option list back in BASE order (jevkit/cli.py:133 did exactly that, so
    # reorder flips read from that file were vacuously 0); the in-process run above never round-trips through JSON
    with open(out / "perturbed.jsonl", "w", encoding="utf-8", newline="\n") as fh:  # same rows as `jevkit perturb make`
        fh.writelines(json.dumps(p.to_dict(), ensure_ascii=False) + "\n" for p in ps)
    rep = perturb.flip_report(items, ps, preds_to_answers(base), preds_to_answers(pert))
    ties = [r["id"] for r in base if isinstance(r.get("answer"), dict) and len(r["answer"].get("tie_set") or []) > 1]
    report = {"ops": ops, "seeds": list(seeds), "n_base": len(items), "n_perturbed": len(ps), "flip_report": rep,
              "duplicate_tie_items": ties, "max_dp_bound": max_dp,
              "status": {"base": _count(r["status"] for r in base), "perturbed": _count(r["status"] for r in pert)}}
    if row_fn is not None:
        report["row_identity"] = row_identity(items, ps, row_fn)
    report["presented_order"] = presented_order(items, ps, [p.item for p in ps], row_fn)
    report.update(g1_verdict(ops, rep["per_operator"], report.get("row_identity"), max_dp, report["presented_order"]))
    (out / "perturb_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


REORDER_OPS = ("option_reverse", "option_shuffle")
RELABEL_OPS = ("key_relabel",)


def _base_order(question: dict, label_map: dict) -> list[str]:
    """The option ids of a (perturbed) question in the order they are LISTED, mapped back to base ids."""
    inv = {str(v): str(k) for k, v in (label_map or {}).items()}
    crit = question.get("criteria") or {}
    return [inv.get(str(o), str(o)) for o in (crit if isinstance(crit, dict) else range(len(crit)))]


def presented_order(items, perturbed, sent_items, row_fn: Callable[[dict], Any] | None = None) -> dict:
    """Per operator: did the perturbation actually reach the decider? ``request_changed`` counts perturbations whose
    request (the exact item handed to ``run_items``) lists the options in a different base-id order than the base
    item (reorder ops) or under different ids (relabel). ``model_order_changed`` (with ``row_fn``) counts those whose
    model row lists them in a different order: 0 under sym-fork BY DESIGN (decide.schema.canonical_options sorts by
    id, schema.py:98, and the sym axis is content-sorted), so the reorder evidence at the model level is the row
    identity + the structural axis-permutation tests (tests/test_sym_invariance.py), not this count."""
    from jevkit.systemone import item_request

    base = {it.id: it for it in items}
    out: dict[str, dict] = {}
    for p, sent in zip(perturbed, sent_items):
        b = base[p.base_id]
        r = out.setdefault(p.op, {"n": 0, "request_changed": 0, "model_order_changed": 0 if row_fn else None})
        r["n"] += 1
        lm = getattr(p, "label_map", None) or {}
        if p.op in RELABEL_OPS:
            changed = any(str(k) != str(v) for k, v in lm.items())
        else:
            changed = _base_order(sent.question, lm) != _base_order(b.question, {})
        r["request_changed"] += int(changed)
        if row_fn is not None:
            rb, rp = row_fn(item_request(b)), row_fn(item_request(sent))
            inv = {str(v): str(k) for k, v in lm.items()}
            r["model_order_changed"] += int([inv.get(o, o) for o in rp.option_ids] != list(rb.option_ids))
    return out


def g1_verdict(ops: Iterable[str], per: dict, rows: dict | None, max_dp: float = 0.0,
               order: dict | None = None) -> dict:
    """The G1 gate, FAIL CLOSED: every requested operator must be present with n > 0 (an operator that produced no
    perturbation would make its check vacuous), with zero flips, zero invalid answers and max |dp| <= ``max_dp``
    (default 0: bit-identity, as the proof claims; the observed max |dp| is reported apart), AND the rows themselves
    must be bit-identical for every operator (``rows`` from ``row_identity``; no row check = no pass), AND every
    reorder / relabel operator must have actually changed the request the decider received (``order`` from
    ``presented_order``: request_changed > 0; a perturbation file that put the options back in base order -- the
    sort_keys bug -- makes the reorder check vacuous, so it FAILS)."""
    ops = list(ops)
    fails: list[str] = []
    if not ops:
        fails.append("no operators requested")
    for op in ops:
        if op in REORDER_OPS + RELABEL_OPS:
            o = (order or {}).get(op)
            if not o or not o.get("request_changed"):
                fails.append(f"{op}: no perturbation changed the presented request (vacuous)")
    for op in ops:
        v = per.get(op)
        if not v or not v.get("n"):
            fails.append(f"{op}: no perturbations (vacuous)")
            continue
        dp = v.get("max_abs_dp")
        if v.get("flips") != 0:
            fails.append(f"{op}: {v.get('flips')} flips")
        if v.get("invalid") != 0:
            fails.append(f"{op}: {v.get('invalid')} invalid")
        if dp is None or dp > max_dp:
            fails.append(f"{op}: max|dp| {dp} > {max_dp}")
    if rows is None:
        fails.append("rows not checked (row_fn missing)")
        rows_ok = None
    else:
        rows_ok = all(op in rows and rows[op]["n"] > 0 and rows[op]["identical"] == rows[op]["n"] for op in ops)
        if not rows_ok:
            fails.append("rows not bit-identical (or an operator has no row check)")
    seen = [v.get("max_abs_dp") for v in per.values() if v.get("max_abs_dp") is not None]
    return {"g1_pass": not fails, "g1_failures": fails, "rows_all_identical": rows_ok,
            "max_abs_dp_observed": max(seen) if seen else None}


def _count(xs) -> dict:
    d: dict[str, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def main(argv=None) -> int:  # pragma: no cover - GPU path (load_child needs a trained checkpoint)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for k in ("cell", "ckpt_dir", "model_dir", "guard_readiness", "records", "train_dir", "pool_dir", "out_dir"):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--teacher-dir", dest="teacher_dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layout", default=None, help="score under this layout instead of the cell's (same model)")
    ap.add_argument("--ops", default=",".join(REORDER_RELABEL_OPS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--max-dp", dest="max_dp", type=float, default=0.0)
    a = ap.parse_args(argv)
    from heads.layout import build_row
    from jevkit.items import load_public
    from jobs.m2_first_cell.fulldev_eval import load_child

    cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))["config"]
    lay = a.layout or cfg["layout"]
    decider, _, ident = load_child(a, temperature=False, readout=(cfg["template"], lay))
    be = decider.backend
    rep = run(decider, load_public(), a.out_dir, a.ops.split(","), [int(s) for s in a.seeds.split(",")],
              row_fn=lambda req: build_row(req, "q", be.encode, template=be.template, layout=be.layout),
              max_dp=a.max_dp)
    (Path(a.out_dir) / "identity.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"g1_pass": rep["g1_pass"], "g1_failures": rep["g1_failures"],
                      "presented_order": rep["presented_order"],
                      "max_abs_dp_observed": rep["max_abs_dp_observed"],
                      "rows_all_identical": rep.get("rows_all_identical"),
                      "per_operator": {k: {x: v[x] for x in ("n", "flips", "max_abs_dp")}
                                       for k, v in rep["flip_report"]["per_operator"].items()}}, indent=1))
    return 0 if rep["g1_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
