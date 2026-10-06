"""m2 evals driver: G12 (every checkpoint) and G13 (gates) on the frozen development cohort, then the promotion rule.

    python -X utf8 -m train.m2_gates --cell <out>/cell.json --ckpt-dir <ck>/cell --model-dir <snapshot> \
        [--gdn-ckpt <converted.pt>] --records <records.jsonl.gz> --guard-readiness <readiness.json> \
        --parent-arm <dir> --index-rows <dev-subset-rows.jsonl.gz> --out <report.json> [--stage g12|g13|all]

What it does (child = the trained checkpoint, parent = the frozen parent arm cached by ``jobs/m2_parent_arm``):
1. identities: the cell config's cohort sha256 must equal the committed cohort; the guard comes only from a verified
   readiness file (``jobs.m2_pilot.guard_ready.load_ready_guard``); the checkpoint is restored through
   ``Trainer.load`` with its config hash AND lineage (frozen-backbone digest + provenance, tokenizer, layout, training
   items rebuilt from the same cohort split) checked, so a checkpoint from another run or parent is refused.
2. temperature: ``FeatureTemperature`` fit on the cohort ``calibration`` role, with disjointness proven against the
   ``dev`` and ``retention`` roles and the training ids; served through ``decide.Decider`` (``+T:<id>`` revision).
3. eval card on ``dev`` (raw and tempered decision metrics, cohort stamped).
4. arms: every unit is scored through ``decide`` (decision items: served probabilities of the full record request;
   steer: ``safety_eval.metrics.run`` over ``DecideBridge``; Index DEV rows: the kit's results format). The SAME
   ``score_stage`` scores the parent in ``jobs/m2_parent_arm``, and the parent cache is accepted only if its unit
   identity (cohort, unit spec, item ids, variant unit hashes, Index row ids) equals the child's.
   Like-for-like (PR #29 review): BOTH arms are scored untempered (the parent arm is the stock readout with the
   identity calibration), so ``g13_ece`` / ``g13_retention_ece`` compare like with like; the served (tempered)
   behaviour is reported on the eval card, and the temperature's disjointness proof is re-verified at serve time
   on the real dev / retention content (``FeatureTemperature.verify_disjoint``).
5. verdicts (``train.promotion``, fail-closed: any missing / errored / non-identical unit fails its criterion):
   G12 = ``verdict_g12`` on ``dev`` + ``steer_dev`` (clean + D1-A);
   G13 = ``verdict_g13`` on ``steer_gate`` (clean + held-out D1-B, ``final=True``) + the 2,008-row Index DEV subset
   (``retention_comparison``) + the cohort ``retention`` role (``decision_retention``; coordinator decision
   2026-10-06). Index TEST is never read.
6. promotion: ``accept`` iff every stage that ran accepted; margins come from ``train.promotion.MARGINS`` and the
   report carries their status (PENDING USER APPROVAL until the user signs them off).

Exit codes: 0 every stage accepted | 40 G12 rejected | 41 G13 rejected | 42 both rejected |
            10 identity / cache / cohort mismatch | 2 usage. The report is written in every case that got that far.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

STAGES = ("g12", "g13")
ARM_SCHEMA = "ssdm.m2.arm.v1"
REPORT_SCHEMA = "ssdm.m2.gates.v1"
EXIT = {"accept": 0, "g12": 40, "g13": 41, "both": 42, "identity": 10, "usage": 2}
STAGE_ROLES = {"g12": ("dev",), "g13": ("retention",)}


class IdentityMismatch(ValueError):
    """Cohort, unit set or parent cache does not match what this evaluation expects."""


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ units
def unit_spec(cfg: dict) -> dict[str, Any]:
    """Which decision items a cohort role contributes: the cell's K cap and a PRE-REGISTERED token cap (not the
    trained T_max, so the parent cache does not depend on a training run)."""
    return {"max_k": int(cfg["max_k"]), "max_t": int(cfg.get("eval_max_t") or max(cfg["t_ladder"])),
            "template": cfg["template"], "layout": cfg["layout"]}


def decision_items(records: Sequence[dict], encode, spec: dict, guard) -> list:
    """Guarded rows of the role's records kept by ``spec`` (``train.run_cell.keep_item``)."""
    from train.data import items_from_records
    from train.run_cell import keep_item

    its = items_from_records(records, encode, layout=spec["layout"], template=spec["template"], guard=guard)
    return [x for x in its if keep_item(x.row, spec["max_k"], spec["max_t"])]


def stage_variants(cohort, stage: str) -> list[dict]:
    from safety_eval import attacks

    if stage == "g12":
        return attacks.build_variants(cohort.steer_requests("steer_dev"), hijack=[], d1=None, final=False)
    if stage == "g13":  # held-out D1-B: built only here, at a gate (final=True)
        return attacks.build_variants(cohort.steer_requests("steer_gate"), hijack=[], d1=list(attacks.D1_TEMPLATES_B),
                                      final=True)
    raise ValueError(stage)


def units_identity(stage: str, cohort, spec: dict, items_by_role: dict[str, list], variants: list[dict],
                   index_rows: list[dict] | None) -> dict[str, Any]:
    from safety_eval.metrics import unit_fingerprint

    ident = {"stage": stage, "cohort_sha256": cohort.sha256, "unit_spec": spec,
             "items": {r: _sha(sorted(i.id for i in its)) for r, its in sorted(items_by_role.items())},
             "n_items": {r: len(its) for r, its in sorted(items_by_role.items())},
             "variants_sha256": _sha(sorted((v["request_id"], unit_fingerprint(v)) for v in variants)),
             "n_variants": len(variants)}
    if stage == "g13":
        ids = sorted(r["_evaluation"]["run_id"] for r in (index_rows or []))
        ident["index_rows"] = {"n": len(ids), "run_ids_sha256": _sha(ids)}
    ident["id"] = _sha(ident)[:16]
    return ident


# ------------------------------------------------------------------------------------------------ arm scoring
def arm_decision_fields(decide: Callable[[dict], dict], records_by_id: dict[str, dict], items: Sequence
                        ) -> tuple[dict[str, dict], list[dict]]:
    """Served-probability decision fields per item (``promotion.decision_fields`` format), one ``decide`` call per
    record (the full request: rows depend on every schema field). A non-ok response leaves its items OUT, which
    makes the paired comparison non-identical, so the criterion fails closed; the errors are returned."""
    import math

    from train.guard import record_request

    by_rec: dict[str, list] = {}
    for it in items:
        by_rec.setdefault(it.id.rsplit("/", 1)[0], []).append(it)
    fields, errors = {}, []
    for rid in sorted(by_rec):
        req = dict(record_request(records_by_id[rid]))
        resp = decide(req)
        ss = resp.get("ssdm") or {}
        if ss.get("status") != "ok":
            errors.append({"record": rid, "status": ss.get("status"), "code": (ss.get("error") or {}).get("code")})
            continue
        for it in by_rec[rid]:
            served = ss["served_probabilities"].get(it.row.qid) or {}
            p = [float(served.get(o, float("nan"))) for o in it.row.option_ids]
            if not all(math.isfinite(x) for x in p):
                errors.append({"record": rid, "item": it.id, "status": "error", "code": "missing_option_probability"})
                continue
            k = max(range(len(p)), key=p.__getitem__)
            fields[it.id] = {"conf": min(1.0, max(0.0, p[k])), "correct": float(k == it.label), "cluster": rid}
    return fields, errors


def index_predict(decide: Callable[[dict], dict], rows: Sequence[dict], model: str = "ssdm-m2") -> dict[str, dict]:
    """Index rows through ``decide`` in the kit's results format (``harness.score`` input)."""
    from decision_index.engines.base import validate

    res = {}
    for r in rows:
        e = r["_evaluation"]
        t = time.perf_counter()
        resp = decide({"model": model, "state": r["state"], "questions": r["questions"]})
        status = (resp.get("ssdm") or {}).get("status", "error") if "answers" in resp else "error"
        out = {**e, "status": status, "total_wall_ms": (time.perf_counter() - t) * 1000}
        if status == "ok":
            response = {"answers": resp["answers"]}
            validate(r["questions"], response)
            out["response"] = response
        else:
            out["error"] = json.dumps((resp.get("ssdm") or {}).get("error"))
        res[e["run_id"]] = out
    return res


def score_stage(stage: str, decide: Callable[[dict], dict], steer: Callable[[dict], dict],
                items_by_role: dict[str, list], records_by_id: dict[str, dict], variants: list[dict],
                index_rows: list[dict] | None = None) -> dict[str, Any]:
    """One arm's units for one stage. ``decide``: ``decide.Decider.decide``-compatible; ``steer``: a safety_eval
    decide callable (``DecideBridge``). Used for the child here and for the parent in ``jobs/m2_parent_arm``."""
    from safety_eval import metrics

    t0 = time.perf_counter()
    out: dict[str, Any] = {"stage": stage, "decision": {}, "decision_errors": {}}
    for role in STAGE_ROLES[stage]:
        f, err = arm_decision_fields(decide, records_by_id, items_by_role[role])
        out["decision"][role], out["decision_errors"][role] = f, err
    out["steer_rows"] = metrics.run(steer, variants)
    if stage == "g13":
        out["index_results"] = index_predict(decide, index_rows or [])
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return out


# ------------------------------------------------------------------------------------------------ parent cache
def write_arm_cache(cache_dir: str | Path, stage: str, units: dict, ident: dict, arm_identity: dict) -> dict:
    """``<dir>/units-<stage>.json`` + ``<dir>/arm.json`` (one entry per stage: unit identity, arm identity, file
    sha256). Written once per parent and gate; every m2 cell on that parent reuses it."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"units-{stage}.json"
    text = json.dumps(units, sort_keys=True, default=float)
    f.write_text(text, encoding="utf-8")
    man_p = d / "arm.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {"schema": ARM_SCHEMA, "stages": {}}
    man["stages"][stage] = {"units_identity": ident, "arm": arm_identity, "file": f.name,
                            "sha256": hashlib.sha256(text.encode()).hexdigest()}
    man_p.write_text(json.dumps(man, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return man["stages"][stage]


def read_arm_cache(cache_dir: str | Path, stage: str, expect_ident: dict, expect_parent: dict | None = None,
                   expect_sha256: str | None = None) -> tuple:
    """(units, manifest entry). Refuses (``IdentityMismatch``) a cache whose units identity differs from the child's
    (other cohort / unit spec / items / variants / Index rows), whose file hash drifted, or whose parent identity
    differs from ``expect_parent`` (keys present in it must match)."""
    d = Path(cache_dir)
    man_p = d / "arm.json"
    if not man_p.exists():
        raise IdentityMismatch(f"no parent arm cache at {d}")
    if expect_sha256 is not None and hashlib.sha256(man_p.read_bytes()).hexdigest() != expect_sha256:
        raise IdentityMismatch("parent arm.json sha256 != the pinned one (job_params parent_arm_sha256)")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if man.get("schema") != ARM_SCHEMA or stage not in man.get("stages", {}):
        raise IdentityMismatch(f"parent arm cache has no {stage} stage (schema {man.get('schema')})")
    ent = man["stages"][stage]
    if ent["units_identity"] != expect_ident:
        diff = sorted(k for k in set(ent["units_identity"]) | set(expect_ident)
                      if ent["units_identity"].get(k) != expect_ident.get(k))
        raise IdentityMismatch(f"parent arm units differ from the child's on {diff}")
    for k, v in (expect_parent or {}).items():
        if (ent.get("arm") or {}).get("parent", {}).get(k) != v:
            raise IdentityMismatch(f"parent arm identity {k}={ent.get('arm', {}).get('parent', {}).get(k)!r} != {v!r}")
    text = (d / ent["file"]).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != ent["sha256"]:
        raise IdentityMismatch(f"parent arm {ent['file']} sha256 drifted from its manifest")
    return json.loads(text), ent


# ------------------------------------------------------------------------------------------------ verdicts
def stage_verdict(stage: str, child: dict, parent: dict, variants: list[dict], index_rows: list[dict] | None,
                  margins: dict | None = None, n_boot: int = 2000) -> dict[str, Any]:
    from train import promotion as P

    margins = margins or P.MARGINS
    if stage == "g12":
        dec = P.paired_decision(child["decision"]["dev"], parent["decision"]["dev"], n_boot=n_boot)
        v = P.verdict_g12(decision=dec, steer_child_rows=child["steer_rows"], steer_parent_rows=parent["steer_rows"],
                          expected_variants=variants, margins=margins, n_boot=n_boot)
        v["paired"] = {"dev": dec}
        return v
    ret = P.retention_comparison(index_rows or [], child["index_results"], parent["index_results"], B=n_boot, seed=0)
    dret = P.paired_decision(child["decision"]["retention"], parent["decision"]["retention"], n_boot=n_boot)
    v = P.verdict_g13(retention=ret, steer_child_rows=child["steer_rows"], steer_parent_rows=parent["steer_rows"],
                      expected_variants=variants, margins=margins, n_boot=n_boot, decision_retention=dret,
                      require_decision_retention=True)
    v["paired"] = {"index_dev": ret, "retention": dret}
    return v


def promotion_verdict(verdicts: dict[str, dict]) -> dict[str, Any]:
    from train.promotion import MARGINS

    failed = {s: v["failed"] for s, v in verdicts.items() if not v["accept"]}
    pending = "PENDING USER APPROVAL" in str(MARGINS.get("status"))
    acc = not failed and bool(verdicts)
    return {"verdict": "accept" if acc else "reject", "stages": sorted(verdicts), "failed": failed,
            "margins": {k: v for k, v in MARGINS.items()}, "margins_pending_user_approval": pending,
            "promotable": acc and not pending and set(verdicts) == set(STAGES),
            "rule": "accept iff every stage run accepted (fail-closed); promotable only with both G12 and G13 and "
                    "user-approved margins"}


def exit_code(verdicts: dict[str, dict]) -> int:
    if not verdicts:  # nothing was evaluated: never "accept"
        return EXIT["usage"]
    rej = {s for s, v in verdicts.items() if not v["accept"]}
    if not rej:
        return EXIT["accept"]
    return EXIT["both"] if rej == {"g12", "g13"} else EXIT[rej.pop()]


def valid_report(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: a complete, self-consistent report."""
    try:
        pv = d["promotion"]
        return (d.get("schema") == REPORT_SCHEMA and pv["verdict"] in ("accept", "reject")
                and set(pv["stages"]) <= set(STAGES) and pv["stages"]
                and all(s in d["verdicts"] and isinstance(d["verdicts"][s]["accept"], bool) for s in pv["stages"])
                and (pv["verdict"] == "accept") == all(d["verdicts"][s]["accept"] for s in pv["stages"])
                and isinstance(d.get("identities"), dict))
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------ orchestration
def fit_child_temperature(model, calib_items: list, disjoint_from: dict[str, Iterable]):
    """Fit on the calibration items; ``disjoint_from`` = name -> ITEMS (or content hashes). The proof is content-based
    (normalized record content), so a content twin under another id is a leak (PR #29 review B1/B3)."""
    import torch

    from train.calibration import FeatureTemperature, features, item_content
    from train.evalcard import predict

    def content(v):
        v = list(v)
        if v and hasattr(v[0], "row"):
            return item_content(v)
        if not all(isinstance(h, str) and len(h) == 64 for h in v):  # ids are not content; refuse the mix-up
            raise ValueError("disjoint_from values must be items or 64-hex content hashes, not ids")
        return v

    lp, labels, ev, ks = predict(model, calib_items)
    return FeatureTemperature.fit(lp, torch.tensor(labels), features(ev, ks), [i.id for i in calib_items],
                                  disjoint_from={k: content(v) for k, v in disjoint_from.items()},
                                  calib_content=item_content(calib_items))


def run_gates(*, stages: Sequence[str], cohort, spec: dict, child_decide, child_steer, child_identity: dict,
              items_by_role: dict[str, list], records_by_id: dict[str, dict], parent_arm: str | Path,
              expect_parent: dict | None = None, index_rows: list[dict] | None = None, n_boot: int = 2000,
              parent_arm_sha256: str | None = None,
              out: str | Path | None = None, extra: dict | None = None) -> tuple[int, dict]:
    """Score the child per stage, load the matching parent units, apply the verdicts and the promotion rule."""
    from train.promotion import MARGINS

    report: dict[str, Any] = {"schema": REPORT_SCHEMA, "identities": {"cohort": cohort.identity(), "unit_spec": spec,
                                                                      "child": child_identity, **(extra or {})},
                              "verdicts": {}, "stages": {}}
    for stage in stages:
        variants = stage_variants(cohort, stage)
        ident = units_identity(stage, cohort, spec, {r: items_by_role[r] for r in STAGE_ROLES[stage]}, variants,
                               index_rows if stage == "g13" else None)
        parent, ent = read_arm_cache(parent_arm, stage, ident, expect_parent, parent_arm_sha256)
        child = score_stage(stage, child_decide, child_steer, items_by_role, records_by_id, variants,
                            index_rows if stage == "g13" else None)
        v = stage_verdict(stage, child, parent, variants, index_rows, MARGINS, n_boot)
        report["verdicts"][stage] = v
        report["stages"][stage] = {"units_identity": ident, "parent_arm": {k: ent[k] for k in ("sha256", "arm")},
                                   "child_seconds": child["seconds"],
                                   "child_decision_errors": child["decision_errors"]}
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).with_name(f"units-child-{stage}.json").write_text(
                json.dumps(child, sort_keys=True, default=float), encoding="utf-8")
    report["promotion"] = promotion_verdict(report["verdicts"])
    code = exit_code(report["verdicts"])
    report["exit_code"] = code
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return code, report


def expected_parent_identity(parent: str, provenance: dict | None = None) -> dict:
    """What the parent arm cache must say about its parent: family AND repo@revision (and, for Q9-AF, the converted
    checkpoint sha256), so a cache scored on another snapshot of the same family is refused."""
    if parent == "rwkv7":
        import third_party.rwkv7_hf as V

        return {"parent": "rwkv7", "repo": V.REPO, "revision": V.REVISION}
    if parent == "q9af":
        from heads.backbone import Q9AF_BASE_REPO, Q9AF_BASE_REVISION

        prov = provenance or {}
        if not prov.get("converted_sha256"):
            raise IdentityMismatch("q9af child has no converted_sha256 provenance")
        return {"parent": "q9af", "repo": Q9AF_BASE_REPO, "revision": Q9AF_BASE_REVISION,
                "converted_sha256": prov["converted_sha256"]}
    raise ValueError(parent)


# ------------------------------------------------------------------------------------------------ CLI (GPU job)
def _read_records(path: str | Path) -> dict[str, dict]:
    from train.run_cell import to_train_record

    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {r["id"]: to_train_record(r) for r in map(json.loads, f)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.m2_gates")
    ap.add_argument("--cell", required=True, help="cell.json written by train.run_cell (config + ladder)")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt", default=None, help="a specific checkpoint (default: latest in --ckpt-dir)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gdn-ckpt", default=None)
    ap.add_argument("--records", required=True)
    ap.add_argument("--guard-readiness", required=True)
    ap.add_argument("--parent-arm", required=True)
    ap.add_argument("--parent-arm-sha256", required=True, help="pinned sha256 of <parent-arm>/arm.json")
    ap.add_argument("--index-rows", default=None, help="the 2,008-row Index DEV subset rows (needed for g13)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["g12", "g13", "all"], default="all")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    stages = list(STAGES) if a.stage == "all" else [a.stage]
    if "g13" in stages and not a.index_rows:
        print("g13 needs --index-rows (Index DEV subset)", file=sys.stderr)
        return EXIT["usage"]

    import torch

    from decide import Decider
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel
    from jobs.m2_pilot.guard_ready import GuardNotReady, load_ready_guard
    from train.cohort import Cohort, CohortLeak
    from train.evalcard import run_card
    from train.run_cell import cell_items, cohort_split, load_parent_backbone, parent_encoder, train_config
    from train.trainer import Trainer

    cell = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    cfg = cell["config"]
    try:
        cohort = Cohort(expected_sha256=cfg["cohort_sha256"])
        guard = load_ready_guard(a.guard_readiness)
    except (ValueError, KeyError, GuardNotReady) as e:
        print(f"identity: {e}", file=sys.stderr)
        return EXIT["identity"]
    parent = cfg.get("parent", "rwkv7")
    enc = parent_encoder(parent, a.model_dir)
    spec = unit_spec(cfg)
    recs = _read_records(a.records)
    role_recs = {}
    for role in ("dev", "calibration", "retention"):
        rr, _ = cohort.load_role_records(a.records, role)
        role_recs[role] = [recs[r["id"]] for r in rr]
    try:
        tr_ids, _, _ = cohort_split(sorted(recs), cfg, recs)
        cohort.assert_training_ids(tr_ids)
    except (ValueError, CohortLeak) as e:
        print(f"cohort: {e}", file=sys.stderr)
        return EXIT["identity"]
    t_max = cell["ladder"]["T_max"]
    train_items = [x for x in cell_items(recs, tr_ids, cfg, enc, guard, max(cfg["t_ladder"])) if len(x.row.ids) <= t_max]
    bb = load_parent_backbone(parent, a.model_dir, cfg, a.gdn_ckpt, device=a.device)
    cuda = torch.cuda.is_available()
    model = DecisionModel(bb, cfg["head"], cfg.get("head_kwargs"), pad_id=0, seed=cfg["seed"])
    model.to(bb.device)
    tr = Trainer(model, train_config(cfg, cfg["max_steps"], a.ckpt_dir, cuda), train_items, guard=guard)
    ckpt = Path(a.ckpt) if a.ckpt else Trainer.latest(a.ckpt_dir)
    if ckpt is None:
        print("no checkpoint", file=sys.stderr)
        return EXIT["identity"]
    try:
        tr.load(ckpt)  # config hash + lineage (backbone digest/provenance, tokenizer, layout, training items)
    except (ValueError, KeyError) as e:
        print(f"checkpoint lineage: {e}", file=sys.stderr)
        return EXIT["identity"]
    items_by_role = {r: decision_items(role_recs[r], enc, spec, guard) for r in role_recs}
    from train.calibration import item_content

    ft = fit_child_temperature(model, items_by_role["calibration"],
                               {"eval": items_by_role["dev"], "retention": items_by_role["retention"],
                                "train": train_items})
    backend = DecisionModelBackend(model, enc, template=cfg["template"], layout=cfg["layout"],
                                   checkpoint={"path": ckpt.name, "step": tr.step,
                                               "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()})
    served = ft.to_serving()
    decider = Decider(backend, calibration=served)  # what is SERVED (deployment identity, card)
    for role, name in (("dev", "eval"), ("retention", "retention")):  # serve-time proof re-check on the real sets
        served.verify_disjoint(name, item_content(items_by_role[role]))
    # Arms are compared like-for-like (review: g13_ece): the parent arm is the stock readout with the identity
    # calibration, so the child arm is scored UNTEMPERED as well; temperature effects are reported on the card.
    arm_decider = Decider(backend)
    out = Path(a.out)
    card = run_card(model, items_by_role["dev"], tr.step, out.parent / "cards", temperature=ft,
                    cohort=cohort.identity())
    index_rows = None
    if a.index_rows:
        from harness.suite import read_rows

        index_rows = read_rows(a.index_rows)
    records_by_id = {r["id"]: r for rs in role_recs.values() for r in rs}
    prov = getattr(bb, "provenance", None) or {}
    expect_parent = expected_parent_identity(parent, prov)
    try:
        code, rep = _run(stages, cohort, spec, decider, arm_decider, items_by_role, records_by_id, a, expect_parent,
                         index_rows, out, guard, backend, tr, bb, ft, card)
    except IdentityMismatch as e:
        print(f"identity: {e}", file=sys.stderr)
        return EXIT["identity"]
    print(json.dumps({"verdict": rep["promotion"]["verdict"], "exit": code,
                      "failed": rep["promotion"]["failed"]}))
    return code


def _run(stages, cohort, spec, decider, arm_decider, items_by_role, records_by_id, a, expect_parent, index_rows, out,
         guard, backend, tr, bb, ft, card):
    from safety_eval.decide_bridge import DecideBridge
    from train.lineage import backbone_identity

    return run_gates(stages=stages, cohort=cohort, spec=spec, child_decide=arm_decider.decide,
                          child_steer=DecideBridge(arm_decider),
                          child_identity={"served": decider.revision(), "arm": arm_decider.revision(),
                                          "arm_calibration": "untempered, like the parent arm (identity calibration)"},
                          items_by_role=items_by_role, records_by_id=records_by_id, parent_arm=a.parent_arm,
                          expect_parent=expect_parent, index_rows=index_rows, n_boot=a.n_boot, out=out,
                          parent_arm_sha256=a.parent_arm_sha256,
                          extra={"guard": guard.identity(), "checkpoint": backend.checkpoint,
                                 "lineage": tr.lineage, "backbone": backbone_identity(bb),
                                 "temperature": {**ft.identity(), "disjointness": ft.disjointness,
                                                 "proof_sha256": ft.proof_sha256, "fit_report": ft.fit_report},
                                 "card": {k: card.get(k) for k in ("step", "n_items", "raw", "tempered")},
                                 "command": sys.argv})


if __name__ == "__main__":
    sys.exit(main())
