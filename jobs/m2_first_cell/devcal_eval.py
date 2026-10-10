"""Mode devcal (B-PREREG-D19-TEMP, PREREG addendum 1): per-question probabilities of a SAVED checkpoint on the
train-side held-out cohort-dev items the cell's own dev cards sample from, so ONE temperature T can be fitted on CPU.

Why: art-p3gpi (D19) stores its dev cards as aggregates only, and D19 drew the whole bschema-draw-v3b pool, so no v3b
row is outside its training draw. The fit set is therefore the cell's cohort-dev pool (cell.json dev_card_sample
n_pool; the 200 dev-card ids are a subset), excluded from training by the cohort guard (id, content, source item).

GPU (``score``): the checkpoint loads exactly as ``fulldev_eval.load_child`` loads it (cohort, guard, training-item
lineage, ``Trainer.load``); the eval items are rebuilt exactly as ``train.run_cell`` builds them (``cell_items`` on the
cohort dev ids with the eval max_k, then the T_max filter). Identity refusals (exit 10): pool size != cell.json
n_pool, the dev-card sample (ids + sha) != cell.json dev_card_sample, any eval id in the training ids. Then
``train.evalcard.run_card(keep_units=True)`` on (1) all pool items and (2) the dev-card sample alone; units carry
``decision_probs`` (raw, untempered) + labels per item.

CPU (``fit``, stdlib + numpy): scalar T over [T_MIN, T_MAX] minimising mean NLL over every question of the primary
set (all pool items); also the dev-card 200 and the remaining rows, before/after. Gate: the dev-card subset at T = 1
must reproduce the training run's card at that step within DEVCARD_TOL. Writes T.json.

Never DEV / proxy600 / JevBench rows, never the cohort calibration / retention roles."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "devcal/v1"
T_MIN, T_MAX = 0.25, 4.0
DEVCARD_TOL = {"nll": 0.01, "accuracy": 0.01}
ROLE = "dev"  # the cohort role run_cell's eval items come from (load_cell_data); never calibration / retention


def ids_sha256(ids) -> str:
    """sha256 of the sorted ids joined by newlines (the convention of train.evalcard's eval hash)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ GPU: score
def load(a):
    """(cell, model, trainer, eval_items, train_ids, checkpoint record) for a saved checkpoint."""
    import torch

    from heads.model import DecisionModel
    from jobs.m2_pilot.guard_ready import load_ready_guard
    from train.cohort import Cohort
    from train.run_cell import (attach_order_views, attach_teacher, cell_items, load_cell_data, load_parent_backbone,
                                parent_encoder, train_config)
    from train.trainer import Trainer, with_execution

    cell = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    cfg = cell["config"]
    cohort = Cohort(expected_sha256=cfg["cohort_sha256"])
    guard = load_ready_guard(a.guard_readiness)
    parent = cfg.get("parent", "rwkv7")
    enc = parent_encoder(parent, a.model_dir)
    train_recs, eval_recs, tr_ids, ev_ids, _ = load_cell_data(cfg, a.records, a.train_dir, cohort)
    cohort.assert_training_ids(tr_ids)
    t_max = cell["ladder"]["T_max"]
    t_cap = max(cfg["t_ladder"])
    eval_items = [x for x in cell_items(eval_recs, ev_ids, cfg, enc, guard, t_cap, max_k=cfg["max_k"])
                  if len(x.row.ids) <= t_max]  # run_cell: eval_items filtered to the measured T_max before any card
    train_items = [x for x in cell_items(train_recs, tr_ids, cfg, enc, guard, t_cap) if len(x.row.ids) <= t_max]
    attach_order_views(cfg, train_items, train_recs, enc, guard, t_max)
    attach_teacher(cfg, train_items, a.teacher_dir, tr_ids)
    bb = load_parent_backbone(parent, a.model_dir, cfg, None, device=a.device)
    cuda = torch.cuda.is_available() and str(a.device).startswith("cuda")
    model = DecisionModel(bb, cfg["head"], cfg.get("head_kwargs"), pad_id=0, seed=cfg["seed"])
    model.to(bb.device)
    tr = Trainer(model, with_execution(train_config(cfg, cfg["max_steps"], a.ckpt_dir, cuda), cfg), train_items,
                 guard=guard)
    ckpt = Path(a.ckpt)
    tr.load(ckpt)  # config hash + lineage (backbone digest/provenance, tokenizer, layout, training items)
    ck = {"path": ckpt.name, "step": tr.step, "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}
    return cell, model, tr, eval_items, set(tr_ids), ck


def check_pool(cell: dict, eval_items: list, train_ids: set, joint: bool) -> dict:
    """Identity of the rebuilt pool against the training run's cell.json; raises ValueError on any difference."""
    from train.trainer import dev_card_record, item_record

    want = cell["dev_card_sample"]
    if len(eval_items) != want["n_pool"]:
        raise ValueError(f"pool: {len(eval_items)} eval items, cell.json dev_card_sample n_pool {want['n_pool']}")
    got = dev_card_record(eval_items, want["rows"], want["seed"], joint)
    for k in ("n", "ids", "ids_sha256", "per_source"):
        if k in want and got.get(k) != want[k]:
            raise ValueError(f"dev-card sample {k} != cell.json dev_card_sample (rebuilt pool differs from training)")
    ids = [it.id for it in eval_items]
    leak = sorted({i for i in ids if i in train_ids} | {item_record(it) for it in eval_items if item_record(it) in train_ids})
    if leak:
        raise ValueError(f"{len(leak)} pool ids / records are training ids (first {leak[0]})")
    return {"n_pool": len(ids), "pool_ids_sha256": ids_sha256(ids), "dev_card_ids_sha256": got.get("ids_sha256"),
            "dev_card_n": got["n"], "cohort_role": ROLE, "train_overlap": 0}


def score(a) -> int:
    from train.evalcard import run_card
    from train.trainer import dev_card_sample

    t0 = time.time()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        cell, model, tr, items, tr_ids, ck = load(a)
        ident = check_pool(cell, items, tr_ids, bool(getattr(tr, "joint", False)))
    except ValueError as e:
        print(f"identity: {e}", file=sys.stderr, flush=True)
        return 10
    if a.expect_step is not None and ck["step"] != a.expect_step:
        print(f"identity: checkpoint step {ck['step']} != expected {a.expect_step}", file=sys.stderr, flush=True)
        return 10
    t_load = time.time() - t0
    want = cell["dev_card_sample"]
    dev = dev_card_sample(items, want["rows"], want["seed"], bool(getattr(tr, "joint", False)))
    pool_card = run_card(model, items, ck["step"], out / "pool", keep_units=True)
    dev_card = run_card(model, dev, ck["step"], out / "devcard", keep_units=True)
    for c in (pool_card, dev_card):
        c.pop("units", None)
    rec = {"schema": SCHEMA, "checkpoint": ck, "identity": ident, "load_s": round(t_load, 1),
           "pool": {"n": pool_card["n_items"], "raw": pool_card["raw"], "seconds": pool_card["seconds"],
                    "units": f"pool/units-step{ck['step']:07d}.json"},
           "devcard": {"n": dev_card["n_items"], "raw": dev_card["raw"], "seconds": dev_card["seconds"],
                       "units": f"devcard/units-step{ck['step']:07d}.json"},
           "training_card": next((c for c in cell.get("dev_cards", []) if c.get("step") == ck["step"]), None)}
    (out / "devcal-score.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("checkpoint", "identity")}, default=str), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ CPU: fit
def _softmax_t(p: list[float], t: float) -> list[float]:
    z = [math.log(max(x, 1e-300)) / t for x in p]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def mean_nll(probs, labels, t: float) -> float:
    return sum(-math.log(max(_softmax_t(p, t)[y], 1e-12)) for p, y in zip(probs, labels)) / len(labels)


def fit_t(probs, labels, lo: float = T_MIN, hi: float = T_MAX, tol: float = 1e-6) -> float:
    """Golden-section on beta = 1/T in [1/hi, 1/lo]: mean NLL of softmax(beta * log p) is convex in beta."""
    f = lambda b: mean_nll(probs, labels, 1.0 / b)  # noqa: E731
    a, b = 1.0 / hi, 1.0 / lo
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 1.0 / ((a + b) / 2)


def _metrics(probs, labels, t: float) -> dict:
    from train.metrics import decision_metrics

    m = decision_metrics([_softmax_t(p, t) for p in probs], labels)
    return {"nll": m["nll"], "accuracy": m["accuracy"], "brier": m["brier"], "ece_binned15": m["ece_binned15"],
            "ece_debiased_l2": m["ece_debiased"]["debiased_l2"], "mean_conf": m["mean_conf"]}


def _units(p: Path) -> tuple[list[str], list[list[float]], list[int]]:
    u = json.loads(p.read_text(encoding="utf-8"))
    ids, probs, labels = u["item_ids"], u["decision_probs"], [int(y) for y in u["labels"]]
    if not (len(ids) == len(probs) == len(labels)) or len(set(ids)) != len(ids):
        raise ValueError(f"{p}: ids / probs / labels lengths differ or ids repeat")
    return ids, probs, labels


def fit(a) -> dict:
    d = Path(a.score_dir)
    sc = json.loads((d / "devcal-score.json").read_text(encoding="utf-8"))
    ids, probs, labels = _units(d / sc["pool"]["units"])
    dids, dprobs, dlabels = _units(d / sc["devcard"]["units"])
    if ids_sha256(ids) != sc["identity"]["pool_ids_sha256"]:
        raise ValueError("pool units ids != the identity the score phase checked")
    by = dict(zip(ids, range(len(ids))))
    if not set(dids) <= set(by):
        raise ValueError("dev-card ids are not a subset of the pool")
    gate = {}
    card = sc.get("training_card") or {}
    for k, tol in DEVCARD_TOL.items():  # the dev-card subset at T = 1 vs the training run's card at this step
        got = _metrics(dprobs, dlabels, 1.0)[k]
        ref = card.get(k)
        gate[k] = {"rescored": got, "training_card": ref, "tol": tol,
                   "pass": ref is not None and abs(got - ref) <= tol}
    gate["pass"] = all(v["pass"] for v in gate.values() if isinstance(v, dict))
    # metric consistency: our T = 1 metrics equal the in-job card's (same decision_metrics, same probs)
    if abs(_metrics(probs, labels, 1.0)["nll"] - sc["pool"]["raw"]["nll"]) > 1e-6:
        raise ValueError("pool NLL at T = 1 != the in-job card's raw NLL")
    dset = set(dids)
    rest = [i for i in ids if i not in dset]
    sets = {"pool": ids, "devcard": [i for i in ids if i in dset], "rest": rest}
    t_star = fit_t(probs, labels)
    report: dict[str, Any] = {"schema": SCHEMA, "primary": "pool", "T": round(t_star, 6), "bounds": [T_MIN, T_MAX],
                              "objective": "mean NLL over every question, scalar T on raw decision_probs: "
                                           "softmax(log p / T)",
                              "checkpoint": sc["checkpoint"], "identity": sc["identity"], "devcard_gate": gate,
                              "sets": {}}
    for name, sid in sets.items():
        pr = [probs[by[i]] for i in sid]
        lb = [labels[by[i]] for i in sid]
        t_own = fit_t(pr, lb) if sid else None
        report["sets"][name] = {"n": len(sid), "ids_sha256": ids_sha256(sid),
                                "before": _metrics(pr, lb, 1.0), "after": _metrics(pr, lb, t_star),
                                "T_fit_on_this_set": None if t_own is None else round(t_own, 6)}
    report["n"] = report["sets"]["pool"]["n"]
    report["ids_sha256"] = report["sets"]["pool"]["ids_sha256"]
    report["usable"] = bool(gate["pass"]) and T_MIN < t_star < T_MAX
    return report


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: the score phase wrote both cards' units and passed identity."""
    try:
        return (d.get("schema") == SCHEMA and d["identity"]["train_overlap"] == 0
                and d["pool"]["n"] == d["identity"]["n_pool"] and d["devcard"]["n"] == d["identity"]["dev_card_n"]
                and math.isfinite(d["pool"]["raw"]["nll"]) and bool(d["checkpoint"]["sha256"]))
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="devcal_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--cell", required=True)
    s.add_argument("--ckpt-dir", required=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--records", required=True)
    s.add_argument("--train-dir", required=True)
    s.add_argument("--teacher-dir", default=None)
    s.add_argument("--guard-readiness", required=True)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--expect-step", type=int, default=None)
    s.add_argument("--out-dir", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--score-dir", required=True, help="the dir holding devcal-score.json + pool/ + devcard/")
    f.add_argument("--out", required=True, help="T.json")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        return score(a)
    rep = fit(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("T", "n", "ids_sha256", "usable")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
