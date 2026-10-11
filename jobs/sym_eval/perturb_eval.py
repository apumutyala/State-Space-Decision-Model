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
Gate verdict: ``g1_pass`` = zero flips and max |dp| <= ``--max-dp`` (default 1e-3, bf16 jitter) on every operator.
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
        seeds: Iterable[int] = (0,), row_fn: Callable[[dict], Any] | None = None, max_dp: float = 1e-3,
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
    # NOT sort_keys: a sorted dump puts every option list back in BASE order (#245: 0/213 reordered rows in a receipt)
    with open(out / "perturbed.jsonl", "w", encoding="utf-8", newline="\n") as fh:  # same rows as `jevkit perturb make`
        fh.writelines(json.dumps(p.to_dict(), ensure_ascii=False) + "\n" for p in ps)
    with open(out / "perturbed.jsonl", encoding="utf-8") as fh:  # the file must keep the in-memory option order
        got = [list(((r.get("item") or r).get("question") or {}).get("criteria") or [])
               for r in (json.loads(x) for x in fh if x.strip())]
    if got != [list(p.item.question.get("criteria") or []) for p in ps]:
        raise AssertionError("perturbed.jsonl does not preserve the in-memory option order")
    rep = perturb.flip_report(items, ps, preds_to_answers(base), preds_to_answers(pert))
    ties = [r["id"] for r in base if isinstance(r.get("answer"), dict) and len(r["answer"].get("tie_set") or []) > 1]
    per = rep["per_operator"]
    g1 = all((v["flips"] == 0 and (v["max_abs_dp"] or 0.0) <= max_dp and v["invalid"] == 0) for v in per.values())
    report = {"ops": ops, "seeds": list(seeds), "n_base": len(items), "n_perturbed": len(ps), "flip_report": rep,
              "duplicate_tie_items": ties, "max_dp_bound": max_dp, "g1_pass": bool(g1 and per),
              "status": {"base": _count(r["status"] for r in base), "perturbed": _count(r["status"] for r in pert)}}
    if row_fn is not None:
        report["row_identity"] = row_identity(items, ps, row_fn)
        report["rows_all_identical"] = all(v["identical"] == v["n"] for v in report["row_identity"].values())
    (out / "perturb_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


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
    ap.add_argument("--max-dp", dest="max_dp", type=float, default=1e-3)
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
    print(json.dumps({"g1_pass": rep["g1_pass"], "rows_all_identical": rep.get("rows_all_identical"),
                      "per_operator": {k: {x: v[x] for x in ("n", "flips", "max_abs_dp")}
                                       for k, v in rep["flip_report"]["per_operator"].items()}}, indent=1))
    return 0 if rep["g1_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
