"""DRY-RUN stand-in for the two GPU scoring phases of jobs/q9af_cell (never on the paid path; the runner calls it only
under Q9AFC_DRYRUN=1, and every file it writes says ``dry_run: true``).

The paid path scores the frozen cohort (v3) + steer variants + the full private DEV slice through
``jobs.m2_parent_arm.score_parent`` (parent) and ``train.m2_gates`` (child). None of those inputs exists on a laptop, so
the dry run keeps the SAME file contracts and code paths downstream of them:

    parent  -> <out>/parent_arm/{arm.json, units-g13.json} (train.m2_gates.write_arm_cache) + <out>/dev-rows.jsonl.gz
               parent = the converted tiny backbone with an UNTRAINED fork head (stand-in for the stock readout)
    child   -> <out>/report.json (train.m2_gates schema, valid_report) + <out>/units-child-g13.json
               child  = the trained checkpoint restored through Trainer.load (config hash + lineage checked)

Rows are synthetic kit-format DEV rows (scripts.m2_e2e_dryrun.synthetic_dev_rows); Index results come from
``train.m2_gates.index_predict``; G13's capability / ECE terms from ``train.promotion.retention_comparison``; the steer
and retention criteria have no units here and fail closed exactly as they would on an empty gate (that is the point:
the runner must carry a REJECT through to the summary and the verdict).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def _decider(model, enc, cfg, ckpt=None):
    from decide import Decider
    from heads.decide_backend import DecisionModelBackend

    return Decider(DecisionModelBackend(model, enc, template=cfg["template"], layout=cfg["layout"],
                                        checkpoint=ckpt or {"path": "untrained", "step": 0}))


def _backbone(cfg, a):
    from train.run_cell import load_parent_backbone

    return load_parent_backbone("q9af", a.model_dir, cfg, a.gdn_ckpt, device="cpu")


def parent(a, cfg) -> int:
    from heads.model import DecisionModel
    from scripts.m2_e2e_dryrun import synthetic_dev_rows
    from train import m2_gates as G
    from train.run_cell import parent_encoder

    enc = parent_encoder("q9af", a.model_dir)
    bb = _backbone(cfg, a)
    model = DecisionModel(bb, cfg["head"], cfg.get("head_kwargs"), pad_id=0, seed=cfg["seed"]).eval()
    rows = synthetic_dev_rows(a.n_groups)
    res = G.index_predict(_decider(model, enc, cfg).decide, rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "dev-rows.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ident = {"stage": "g13", "dry_run": True, "n_rows": len(rows)}
    G.write_arm_cache(out / "parent_arm", "g13", {"index_results": res, "dry_run": True}, ident,
                      {"parent": G.expected_parent_identity("q9af", bb.provenance), "role": "parent",
                       "dry_run": True, "note": "untrained fork head on the converted tiny backbone"})
    print(json.dumps({"rows": len(rows), "ok": sum(r.get("status") == "ok" for r in res.values())}))
    return 0


def child(a, cfg) -> int:
    from heads.model import DecisionModel
    from train import m2_gates as G
    from train import promotion as P
    from train.guard import SplitGuard
    from train.run_cell import cell_items, load_cell_data, parent_encoder, train_config
    from train.trainer import Trainer, with_execution

    cell = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    enc = parent_encoder("q9af", a.model_dir)
    guard = SplitGuard.synthetic_only("q9af_cell dry run: toy records only")
    train_recs, _, tr_ids, _, _ = load_cell_data(cfg, a.records)
    t_max = cell["ladder"]["T_max"]
    items = [x for x in cell_items(train_recs, tr_ids, cfg, enc, guard, max(cfg["t_ladder"])) if len(x.row.ids) <= t_max]
    bb = _backbone(cfg, a)
    model = DecisionModel(bb, cfg["head"], cfg.get("head_kwargs"), pad_id=0, seed=cfg["seed"])
    tr = Trainer(model, with_execution(train_config(cfg, cfg["max_steps"], a.ckpt_dir, False), cfg), items, guard=guard)
    ckpt = Trainer.latest(a.ckpt_dir)
    tr.load(ckpt)  # config hash + lineage, as m2_gates
    model.eval()
    with gzip.open(Path(a.parent_dir) / "dev-rows.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    res = G.index_predict(_decider(model, enc, cfg, {"path": ckpt.name, "step": tr.step}).decide, rows)
    par = json.loads((Path(a.parent_dir) / "parent_arm" / "units-g13.json").read_text(encoding="utf-8"))["index_results"]
    ret = P.retention_comparison(rows, res, par, B=a.B, seed=0)
    v13 = P.verdict_g13(retention=ret, steer_child_rows=[], steer_parent_rows=[], expected_variants=[],
                        require_decision_retention=True)
    v12 = P.verdict_g12(decision={"identical_items": False}, steer_child_rows=[], steer_parent_rows=[],
                        expected_variants=[])
    verdicts = {"g12": v12, "g13": v13}
    rep = {"schema": G.REPORT_SCHEMA, "dry_run": True, "identities": {"dry_run": True, "checkpoint": ckpt.name},
           "verdicts": verdicts, "stages": {}, "promotion": G.promotion_verdict(verdicts)}
    code = G.exit_code(verdicts, P.margins_pending(P.MARGINS))
    rep["exit_code"] = code
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "units-child-g13.json").write_text(json.dumps({"index_results": res, "dry_run": True}), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    assert G.valid_report(rep)
    print(json.dumps({"verdict": rep["promotion"]["verdict"], "exit": code}))
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.q9af_cell.dry_gates")
    ap.add_argument("role", choices=["parent", "child"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gdn-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--cell", default=None)
    ap.add_argument("--records", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--parent-dir", default=None)
    ap.add_argument("--B", type=int, default=200)
    a = ap.parse_args(argv)
    from scripts.m2_e2e_dryrun import ensure_kit

    ensure_kit()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    return parent(a, cfg) if a.role == "parent" else child(a, cfg)


if __name__ == "__main__":
    sys.exit(main())
