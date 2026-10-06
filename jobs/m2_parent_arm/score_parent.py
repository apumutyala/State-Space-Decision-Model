"""Reusable parent arm: score the frozen m1 parent ONCE per gate on the frozen development cohort and cache the units.

    python -X utf8 -m jobs.m2_parent_arm.score_parent --parent rwkv7|q9af --model-dir <snapshot> [--gdn-ckpt <pt>] \
        --cell-config jobs/m2_pilot/m2_pilot_cell.json --records <records.jsonl.gz> \
        --guard-readiness <readiness.json> --index-rows <dev-subset-rows.jsonl.gz> --out-dir <cache> [--stage all]
    python -X utf8 -m jobs.m2_parent_arm.score_parent estimate --tok-s <measured prefill tok/s>

Every m2 cell on the same parent compares against these units (``train.m2_gates.read_arm_cache``), so the parent is
scored once instead of once per cell (halves each cell's eval volume). The parent is served exactly as m0/m1 serve it:
the stock decision readout (``decide.backends.hf_readout``) through ``decide.Decider`` with the identity calibration.

The cache is valid for a child only when ``train.m2_gates.units_identity`` agrees: same cohort sha256, same unit spec
(K cap, pre-registered token cap, template, layout), same decision item ids, same steer variants (``unit_sha`` of
every unit), same Index DEV run ids. The parent identity (family, repo@revision, converted-checkpoint sha256 for Q9-AF,
served revision id) is stored beside the units; ``m2_gates`` refuses a cache from another parent.

Exit codes: 0 cached | 10 identity (cohort / guard / pins) | 15 projected past the cap from a MEASURED rate | 2 usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

#: Token volumes per stage, measured 2026-10-06 on the committed cohort with the RWKV7-G1k tokenizer (every question
#: row of every role record; steer units clean + D1 variants; Index DEV = the subset's kit proxy tokens x READ_FACTOR 2).
VOLUME_TOKENS = {
    "g12": {"dev_records": 3_487_676, "steer_dev_units": 5_378_738},
    "g13": {"retention_records": 6_688_286, "steer_gate_units": 5_430_015, "index_dev_subset": 2 * 2_370_062},
}
USD_PER_MIN = 0.0495  # SF H100 batch rate (jobs/DECISION_TREE.md)
FIXED_MIN = 35  # build 15 + setup/download/verify 15 + finalize 5 (expected, rwkv7 stage-A measured shape)


def estimate(tok_s: float, stages: Sequence[str] = ("g12", "g13"), fixed_min: float = FIXED_MIN) -> dict[str, Any]:
    """Projected minutes / $ at a MEASURED prefill rate (no number here is a measurement of the parent)."""
    if not tok_s > 0:
        raise ValueError("tok_s must be a positive measured rate")
    per = {s: sum(VOLUME_TOKENS[s].values()) for s in stages}
    minutes = {s: n / tok_s / 60 for s, n in per.items()}
    total = sum(minutes.values()) + fixed_min
    return {"tok_s": tok_s, "tokens": per, "score_minutes": minutes, "total_minutes": total,
            "usd": round(total * USD_PER_MIN, 2), "fixed_minutes": fixed_min, "usd_per_min": USD_PER_MIN}


def score_parent_arm(*, stages: Sequence[str], cohort, spec: dict, decide, steer, items_by_role: dict,
                     records_by_id: dict, out_dir: str | Path, parent_identity: dict,
                     index_rows: list[dict] | None = None) -> dict[str, Any]:
    """Score + cache each stage (device-agnostic; tests drive it with tiny CPU models)."""
    from train import m2_gates as G

    entries = {}
    for stage in stages:
        variants = G.stage_variants(cohort, stage)
        ident = G.units_identity(stage, cohort, spec, {r: items_by_role[r] for r in G.STAGE_ROLES[stage]}, variants,
                                 index_rows if stage == "g13" else None)
        units = G.score_stage(stage, decide, steer, items_by_role, records_by_id, variants,
                              index_rows if stage == "g13" else None)
        entries[stage] = G.write_arm_cache(out_dir, stage, units, ident, parent_identity)
    return entries


def parent_decider(parent: str, model_dir: str, gdn_ckpt: str | None, cfg: dict, device: str = "cuda"):
    """The parent's stock readout through decide (identity calibration), plus its identity block."""
    from decide import Decider
    from decide.backends.hf_readout import HFReadoutBackend

    if parent == "rwkv7":
        import third_party.rwkv7_hf as V
        from decide.backends.hf_readout import load_rwkv7_hf

        plug = load_rwkv7_hf(model_dir, device=device)
        ident = {"parent": "rwkv7", "repo": V.REPO, "revision": V.REVISION}
    elif parent == "q9af":
        from convert.readout import load_converted_plugin
        from heads.backbone import q9af_pin_check

        prov = q9af_pin_check(cfg.get("q9af") or {}, gdn_ckpt, model_dir)
        import torch

        layers = sorted(int(k) for k in torch.load(gdn_ckpt, map_location="cpu", weights_only=True))
        plug = load_converted_plugin(model_dir, gdn_ckpt, layers, device=device)  # every attention layer -> GDN
        ident = {"parent": "q9af", "repo": prov["base_repo"], "revision": prov["base_revision"],
                 "converted_sha256": prov["converted_sha256"], "converted_layers": layers}
    else:
        raise ValueError(parent)
    dec = Decider(HFReadoutBackend(plug, cfg["template"]), backend_max_input_tokens=plug.max_context)
    return dec, {"parent": ident, "revision": dec.revision()["revision_id"], "calibration": "uncalibrated_identity"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["estimate"]:
        ap = argparse.ArgumentParser(prog="score_parent estimate")
        ap.add_argument("--tok-s", type=float, required=True)
        ap.add_argument("--cap-min", type=float, default=None)
        ap.add_argument("--fixed-min", type=float, default=FIXED_MIN,
                        help="non-scoring minutes to add (0 for a scoring phase inside a running job)")
        a = ap.parse_args(argv[1:])
        est = estimate(a.tok_s, fixed_min=a.fixed_min)
        print(json.dumps(est, indent=1))
        return 15 if a.cap_min is not None and est["total_minutes"] > 0.95 * a.cap_min else 0
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_parent_arm.score_parent")
    ap.add_argument("--parent", choices=["rwkv7", "q9af"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gdn-ckpt", default=None)
    ap.add_argument("--cell-config", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--guard-readiness", required=True)
    ap.add_argument("--index-rows", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage", choices=["g12", "g13", "all"], default="all")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    stages = ["g12", "g13"] if a.stage == "all" else [a.stage]
    if "g13" in stages and not a.index_rows:
        print("g13 needs --index-rows", file=sys.stderr)
        return 2

    from jobs.m2_pilot.guard_ready import GuardNotReady, load_ready_guard
    from safety_eval.decide_bridge import DecideBridge
    from train import m2_gates as G
    from train.cohort import Cohort
    from train.run_cell import parent_encoder

    cfg = json.loads(Path(a.cell_config).read_text(encoding="utf-8"))
    if cfg.get("parent", "rwkv7") != a.parent:
        print(f"cell config is for parent {cfg.get('parent', 'rwkv7')}, not {a.parent}", file=sys.stderr)
        return 10
    try:
        cohort = Cohort(expected_sha256=cfg["cohort_sha256"])
        guard = load_ready_guard(a.guard_readiness)
    except (ValueError, KeyError, GuardNotReady) as e:
        print(f"identity: {e}", file=sys.stderr)
        return 10
    enc = parent_encoder(a.parent, a.model_dir)
    spec = G.unit_spec(cfg)
    recs = G._read_records(a.records)
    role_recs = {}
    for role in ("dev", "retention"):
        rr, _ = cohort.load_role_records(a.records, role)
        role_recs[role] = [recs[r["id"]] for r in rr]
    items_by_role = {r: G.decision_items(v, enc, spec, guard) for r, v in role_recs.items()}
    index_rows = None
    if a.index_rows:
        from harness.suite import read_rows

        index_rows = read_rows(a.index_rows)
    try:
        dec, ident = parent_decider(a.parent, a.model_dir, a.gdn_ckpt, cfg, a.device)
    except Exception as e:  # noqa: BLE001 - LineageError / pins: never score an unpinned parent
        print(f"parent identity: {e!r}", file=sys.stderr)
        return 10
    ident["guard"] = guard.identity()
    entries = score_parent_arm(stages=stages, cohort=cohort, spec=spec, decide=dec.decide, steer=DecideBridge(dec),
                               items_by_role=items_by_role,
                               records_by_id={r["id"]: r for rs in role_recs.values() for r in rs},
                               out_dir=a.out_dir, parent_identity=ident, index_rows=index_rows)
    print(json.dumps({s: {"sha256": e["sha256"], "units": e["units_identity"]["id"]} for s, e in entries.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
