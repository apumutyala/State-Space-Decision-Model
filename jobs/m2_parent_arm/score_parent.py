"""Reusable parent arm: score the frozen m1 parent ONCE per gate on the frozen development cohort and cache the units.

    python -X utf8 -m jobs.m2_parent_arm.score_parent --parent rwkv7|q9af --model-dir <snapshot> [--gdn-ckpt <pt>] \
        --cell-config jobs/m2_pilot/m2_pilot_cell.json --records <records.jsonl.gz> \
        --guard-readiness <readiness.json> --index-rows <dev-rows.jsonl.gz>         (--index-frozen-out <frozen.json> | --index-frozen <frozen.json> --index-frozen-sha256 <sha>)         --out-dir <cache> [--stage all]
    python -X utf8 -m jobs.m2_parent_arm.score_parent estimate --tok-s <measured prefill tok/s>

Every m2 cell on the same parent compares against these units (``train.m2_gates.read_arm_cache``), so the parent is
scored once instead of once per cell (halves each cell's eval volume). The parent is served exactly as m0/m1 serve it:
the stock decision readout (``decide.backends.hf_readout``) through ``decide.Decider`` with the identity calibration.

The cache is valid for a child only when ``train.m2_gates.units_identity`` agrees: same cohort sha256, same unit spec
(K cap, pre-registered token cap, template, layout), same decision item ids, same steer variants (``unit_sha`` of
every unit), same Index DEV run ids and frozen-set hashes. G13's Index term is the FULL DEV slice minus the rows the
parent's decider refuses before any forward, frozen by ``jobs.common.dev_full.freeze`` HERE, before anything is scored
(``--index-frozen-out``); pin the printed sha256 as ``index_frozen_sha256`` (issue #36). The parent identity (family, repo@revision, converted-checkpoint sha256 for Q9-AF,
served revision id) is stored beside the units; ``m2_gates`` refuses a cache from another parent.

Exit codes: 0 cached | 10 identity (cohort / guard / pins) | 15 projected past the cap from a MEASURED rate | 2 usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

#: Token volumes per stage, measured 2026-10-06 on cohort v2 (``VOLUME_COHORT_SHA256``) with the RWKV7-G1k tokenizer.
#: Upper bounds: every question row of every role record (records whose rows the K/T filter drops still count), steer
#: units = clean + D1 variants. Index DEV = the FULL DEV slice (issue #36): a PROJECTION, the EXACT 2,008-row-subset
#: fork positions (``jobs/common/readout_volume_rwkv7.json`` totals.fork.positions = 6,393,208; PERF_AUDIT) scaled by
#: the full/subset proxy-token ratio of the committed DEV manifest (14,379,163 / 2,370,062 = 6.067) = 38.79 M. Re-measure
#: on the first parent-arm run (exact positions need the private DEV rows). The subset number replaced "proxy tokens x
#: 2" (4.74 M), a 50x under-count of the recompute readout (236.7 M) and a 35% under-count of the fork path.
#: The per-record volumes assume one prompt read per question row: true for the prefix-fork readout (parent) and the
#: shared-prefix decision-head eval (child); the old per-option recompute readout read each prompt K times.
VOLUME_COHORT_SHA256 = "be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2"
VOLUME_TOKENS = {
    "g12": {"dev_records": 3_887_640, "steer_dev_units": 5_378_738},
    "g13": {"retention_records": 5_724_231, "steer_gate_units": 5_430_015, "index_dev_full": 38_787_584},
}
#: What the CHILD evals phase scores on top of its arm (PR #29 round-2 review B4): the temperature fit forwards every
#: calibration item and the eval card forwards every dev item (prefix + every option continuation, all labelled
#: items, no K/T filter: upper bounds), same cohort / tokenizer.
CHILD_EXTRA_TOKENS = {"calibration_fit": 3_393_271, "dev_card": 3_993_020}
#: measurement stage ``index`` (jobs/m2_first_cell): the Index DEV subset alone (same exact fork positions as in g13)
#: plus the readout order diagnostic (jobs/m2_first_cell/readout_diag.py): its 160 rows hold 193,815 of the subset's
#: 2,370,062 proxy tokens (8.18%), read twice (forward + reversed); x 1.25 planning margin (the relabelled ids are
#: slightly longer). Calls: 2 x 160/2008 x 4,536 x 1.25.
#: the 2,008-row subset's exact prefix-fork positions (jobs/common/readout_volume_rwkv7.json; G13 now reads the FULL DEV)
INDEX_SUBSET_TOKENS = 6_393_208
DIAG_TOKENS = round(1.25 * 2 * INDEX_SUBSET_TOKENS * 193_815 / 2_370_062)
DIAG_CALLS = round(1.25 * 2 * 160 / 2008 * 4_536)
VOLUME_TOKENS["index"] = {"index_dev_subset": INDEX_SUBSET_TOKENS, "order_diag": DIAG_TOKENS}
#: Forward CALLS per stage (upper-bound planning counts, used only with a MEASURED per-call latency ``call_s``; the
#: fitted cost model is wall = positions / tok_s + calls x call_s, jobs/PERF_AUDIT.md). Index DEV: the subset's exact
#: prefix-fork calls (readout_volume_rwkv7.json, 4,536). Decision records: 2 calls each (shared prefix + packed tails;
#: dev 835, retention 1,670, calibration 1,000 records). Steer: 1 call per unit (~600-token requests, one pack), 9,000
#: units per steer role. Dev card: 835 records x 2.
VOLUME_CALLS = {"g12": {"dev_records": 2 * 835, "steer_dev_units": 9_000},
                # G13 reads the full DEV (12,104 rows): calls scaled from the subset's 4,536 by rows (planning count)
                "g13": {"retention_records": 2 * 1_670, "steer_gate_units": 9_000,
                        "index_dev_full": round(4_536 * 12_104 / 2_008)},
                "index": {"index_dev_subset": 4_536, "order_diag": DIAG_CALLS}}
CHILD_EXTRA_CALLS = {"calibration_fit": 2 * 1_000, "dev_card": 2 * 835}
USD_PER_MIN = 0.0495  # SF H100 batch rate (jobs/DECISION_TREE.md)
FIXED_MIN = 35  # build 15 + setup/download/verify 15 + finalize 5 (expected, rwkv7 stage-A measured shape)


def estimate(tok_s: float, stages: Sequence[str] = ("g12", "g13"), fixed_min: float = FIXED_MIN,
             mode: str = "parent", call_s: float = 0.0) -> dict[str, Any]:
    """Projected minutes / $ at a MEASURED prefill rate (no number here is a measurement of the model).
    ``mode="parent"``: the cached parent arm only. ``mode="child"``: the m2_pilot evals phase = the child's arm PLUS
    the temperature fit over the calibration items and the dev eval card (``CHILD_EXTRA_TOKENS``).
    ``call_s`` (a MEASURED per-call latency; 0 keeps the legacy token-only projection) adds calls x call_s."""
    if not tok_s > 0:
        raise ValueError("tok_s must be a positive measured rate")
    if mode not in ("parent", "child"):
        raise ValueError(mode)
    if call_s < 0:
        raise ValueError("call_s must be >= 0")
    per = {s: sum(VOLUME_TOKENS[s].values()) for s in stages}
    calls = {s: sum(VOLUME_CALLS[s].values()) for s in stages}
    if mode == "child":
        per.update(CHILD_EXTRA_TOKENS)
        calls.update(CHILD_EXTRA_CALLS)
    minutes = {s: n / tok_s / 60 + calls[s] * call_s / 60 for s, n in per.items()}
    total = sum(minutes.values()) + fixed_min
    return {"tok_s": tok_s, "call_s": call_s, "mode": mode, "tokens": per, "total_tokens": sum(per.values()),
            "calls": calls, "total_calls": sum(calls.values()),
            "score_minutes": minutes, "total_minutes": total, "usd": round(total * USD_PER_MIN, 2),
            "fixed_minutes": fixed_min, "usd_per_min": USD_PER_MIN, "volume_cohort_sha256": VOLUME_COHORT_SHA256}


def min_tok_s(cap_min: float, mode: str = "child", stages: Sequence[str] = ("g12", "g13"),
              fixed_min: float = 0.0, headroom: float = 0.95) -> float:
    """The smallest measured rate the preflight accepts for ``cap_min`` (same 95% headroom rule)."""
    tokens = estimate(1.0, stages, 0.0, mode)["total_tokens"]
    return tokens / ((headroom * cap_min - fixed_min) * 60)


def score_parent_arm(*, stages: Sequence[str], cohort, spec: dict, decide, steer, items_by_role: dict,
                     records_by_id: dict, out_dir: str | Path, parent_identity: dict,
                     index_rows: list[dict] | None = None, index_frozen: dict | None = None,
                     progress: str | Path | None = None) -> dict[str, Any]:
    """Score + cache each stage (device-agnostic; tests drive it with tiny CPU models)."""
    from train import m2_gates as G

    entries = {}
    for stage in stages:
        variants = G.stage_variants(cohort, stage)
        rows = index_rows if G._uses_index(stage) else None
        ident = G.units_identity(stage, cohort, spec, {r: items_by_role[r] for r in G.STAGE_ROLES[stage]}, variants,
                                 rows, index_frozen if stage == "g13" else None)
        units = G.score_stage(stage, decide, steer, items_by_role, records_by_id, variants, rows,
                              progress=Path(str(progress).replace("{stage}", stage)) if progress else None)
        entries[stage] = G.write_arm_cache(out_dir, stage, units, ident, parent_identity)
    return entries


def parent_decider(parent: str, model_dir: str, gdn_ckpt: str | None, cfg: dict, device: str = "cuda"):
    """The parent's stock readout through decide (identity calibration), plus its identity block."""
    from decide import Decider
    from decide.backends.hf_readout import HFReadoutBackend

    if parent == "rwkv7":
        import third_party.rwkv7_hf as V
        from decide.backends.hf_readout import load_rwkv7_hf

        plug = load_rwkv7_hf(model_dir, device=device, wkv=cfg.get("rwkv7_wkv"))  # same WKV as the child
        # the WKV implementation the stock readout actually ran (the model config after select_wkv): m2_gates refuses a
        # cache scored with another one (a "chunked" cache against a chunked_lean child; PR #74 review N1)
        ident = {"parent": "rwkv7", "repo": V.REPO, "revision": V.REVISION,
                 "wkv": str(plug.kernel_stack.get("wkv_implementation"))}
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
        ap.add_argument("--mode", choices=["parent", "child"], default="parent",
                        help="child = the m2_pilot evals phase (arm + temperature fit + dev card)")
        ap.add_argument("--stages", default="g12,g13", help="comma list of g12, g13, index")
        ap.add_argument("--call-s", type=float, default=0.0, help="MEASURED per-call latency (probe small_call_s)")
        a = ap.parse_args(argv[1:])
        stages = [s for s in a.stages.split(",") if s]
        if not stages or any(s not in VOLUME_TOKENS for s in stages):
            ap.error(f"--stages must name some of {sorted(VOLUME_TOKENS)}")
        est = estimate(a.tok_s, stages=stages, fixed_min=a.fixed_min, mode=a.mode, call_s=a.call_s)
        print(json.dumps(est, indent=1))
        return 15 if a.cap_min is not None and est["total_minutes"] > 0.95 * a.cap_min else 0
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_parent_arm.score_parent")
    ap.add_argument("--parent", choices=["rwkv7", "q9af"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gdn-ckpt", default=None)
    ap.add_argument("--cell-config", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--guard-readiness", required=True)
    ap.add_argument("--index-rows", default=None, help="the FULL Index DEV rows")
    ap.add_argument("--index-frozen-out", default=None, help="freeze the unsupported-row set here BEFORE scoring")
    ap.add_argument("--index-frozen", default=None, help="an already frozen set (with --index-frozen-sha256)")
    ap.add_argument("--index-frozen-sha256", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage", choices=["g12", "g13", "index", "all"], default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-dir", action="append", default=None, metavar="NAME=DIR",
                    help="cohort pool outside the PR #16 file (cohort v3 retention: m2-150k=<dir>)")
    ap.add_argument("--progress", default=None, help="live progress file for jobs.common.rate_guard run-weighted "
                    "({stage} is replaced by the stage name)")
    a = ap.parse_args(argv)
    stages = ["g12", "g13"] if a.stage == "all" else [a.stage]
    if "index" in stages and not a.index_rows:
        print(f"{a.stage} needs --index-rows (the Index DEV subset rows)", file=sys.stderr)
        return 2
    if "g13" in stages and not (a.index_rows and (a.index_frozen_out or (a.index_frozen and a.index_frozen_sha256))):
        print("g13 needs --index-rows and --index-frozen-out (or --index-frozen + --index-frozen-sha256)",
              file=sys.stderr)
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
    try:
        pools = G.pool_dirs_arg(a.pool_dir)
        # only the roles these stages score; cohort-v3 retention records from the m2-150k pool (--pool-dir)
        role_recs = {role: G.role_records(cohort, a.records, role, recs, pools)
                     for role in sorted({r for s in stages for r in G.STAGE_ROLES[s]})}
    except ValueError as e:
        print(f"cohort: {e}", file=sys.stderr)
        return 10
    items_by_role = {r: G.decision_items(v, enc, spec, guard) for r, v in role_recs.items()}
    try:
        dec, ident = parent_decider(a.parent, a.model_dir, a.gdn_ckpt, cfg, a.device)
    except Exception as e:  # noqa: BLE001 - LineageError / pins: never score an unpinned parent
        print(f"parent identity: {e!r}", file=sys.stderr)
        return 10
    ident["guard"] = guard.identity()
    index_rows, index_frozen = None, None
    if a.index_rows and "index" in stages:  # measurement stage: the 2,008-row subset (no frozen full-DEV set)
        from harness.suite import read_rows

        index_rows = read_rows(a.index_rows)
    if a.index_rows and "g13" in stages:
        from harness.suite import read_rows
        from jobs.common import dev_full as F

        rows = read_rows(a.index_rows)
        try:
            if a.index_frozen_out:  # pre-declared: frozen with the parent's own decider before ANY arm is scored
                index_frozen = F.freeze(rows, dec)
                sha = F.write_frozen(a.index_frozen_out, index_frozen)
                print(json.dumps({"index_frozen": a.index_frozen_out, "sha256": sha, **F.identity(index_frozen)}))
            else:
                index_frozen = F.load_frozen(a.index_frozen, a.index_frozen_sha256)
            index_rows = F.select_rows(rows, index_frozen)
        except F.FrozenMismatch as e:
            print(f"identity: {e}", file=sys.stderr)
            return 10
    entries = score_parent_arm(stages=stages, cohort=cohort, spec=spec, decide=dec.decide, steer=DecideBridge(dec),
                               items_by_role=items_by_role,
                               records_by_id={r["id"]: r for rs in role_recs.values() for r in rs},
                               out_dir=a.out_dir, parent_identity=ident, index_rows=index_rows,
                               index_frozen=index_frozen, progress=a.progress)
    print(json.dumps({s: {"sha256": e["sha256"], "units": e["units_identity"]["id"]} for s, e in entries.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
