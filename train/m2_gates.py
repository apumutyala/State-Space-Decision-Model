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

STAGES = ("g12", "g13")  # the promotion stages (G12 / G13 of SAFETY_SPEC)
#: measurement-only stage (first trained cell, jobs/m2_first_cell): the 2,008-row Index DEV subset alone, both arms
#: through the same ``score_stage``. It reports capability / ECE with the G13 capability criteria for information but
#: is NOT a promotion stage: it never enters ``promotion_verdict`` or ``exit_code``.
MEASURE_STAGES = ("index",)
ALL_STAGES = STAGES + MEASURE_STAGES
ARM_SCHEMA = "ssdm.m2.arm.v1"
REPORT_SCHEMA = "ssdm.m2.gates.v1"
EXIT = {"accept": 0, "g12": 40, "g13": 41, "both": 42, "identity": 10, "usage": 2}
STAGE_ROLES = {"g12": ("dev",), "g13": ("retention",), "index": ()}
#: progress weight of one scored unit = request JSON chars + this constant (the fitted per-call cost, 0.098 s at
#: ~4k tok/s and ~3.5 chars/token ~= 1,400 chars; jobs/PERF_AUDIT.md). Used only by the live rate guard.
CALL_WEIGHT_CHARS = 1500


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
    if stage == "index":
        return []
    raise ValueError(stage)


def _uses_index(stage: str) -> bool:
    return stage in ("g13", "index")


def units_identity(stage: str, cohort, spec: dict, items_by_role: dict[str, list], variants: list[dict],
                   index_rows: list[dict] | None) -> dict[str, Any]:
    from safety_eval.metrics import unit_fingerprint

    ident = {"stage": stage, "cohort_sha256": cohort.sha256, "unit_spec": spec,
             "items": {r: _sha(sorted(i.id for i in its)) for r, its in sorted(items_by_role.items())},
             "n_items": {r: len(its) for r, its in sorted(items_by_role.items())},
             "variants_sha256": _sha(sorted((v["request_id"], unit_fingerprint(v)) for v in variants)),
             "n_variants": len(variants)}
    if _uses_index(stage):
        ids = sorted(r["_evaluation"]["run_id"] for r in (index_rows or []))
        ident["index_rows"] = {"n": len(ids), "run_ids_sha256": _sha(ids)}
    if stage == "index":  # the readout order diagnostic's rows and settings (jobs/m2_first_cell/readout_diag.py)
        from jobs.m2_first_cell import readout_diag as RD

        d = sorted(r["_evaluation"]["run_id"] for r in RD.diag_rows(index_rows or []))
        ident["order_diag"] = {"n": len(d), "run_ids_sha256": _sha(d), "n_target": RD.DIAG_N, "seed": RD.DIAG_SEED,
                               "conditions": list(RD.CONDITIONS)}
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


def unit_weight(req: dict) -> int:
    return len(json.dumps({"state": req.get("state"), "questions": req.get("questions")}, default=str)) + CALL_WEIGHT_CHARS


class Progress:
    """Live progress for ``jobs.common.rate_guard run-weighted``: ``<path>.total.json`` = total weight, written before
    the first unit; then one ``{"w": weight}`` line per scored unit (decision record, steer variant, Index row)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None

    def plan(self, weights: list[int]) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
            Path(str(self.path) + ".total.json").write_text(
                json.dumps({"total_weight": int(sum(weights)), "n_units": len(weights)}), encoding="utf-8")

    def wrap(self, fn):
        if not self.path:
            return fn
        path = self.path

        def log(req):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"w": unit_weight(req)}) + "\n")

        def call(req):
            try:
                return fn(req)
            finally:
                log(req)

        if hasattr(fn, "decide_many"):  # keep the batched path (same semantics), log after the batch
            def many(reqs):
                try:
                    return fn.decide_many(reqs)
                finally:
                    for r in reqs:
                        log(r)
            call.decide_many = many
        return call


def stage_unit_weights(stage: str, items_by_role: dict, records_by_id: dict, variants: list[dict],
                       index_rows: list[dict] | None) -> list[int]:
    from safety_eval.data import strip_meta
    from train.guard import record_request

    w = []
    for role in STAGE_ROLES[stage]:
        for rid in sorted({it.id.rsplit("/", 1)[0] for it in items_by_role[role]}):
            w.append(unit_weight(record_request(records_by_id[rid])))
    w += [unit_weight(strip_meta(v)) for v in variants]
    if _uses_index(stage):
        w += [unit_weight(r) for r in (index_rows or [])]
    if stage == "index":
        from jobs.m2_first_cell import readout_diag as RD

        w += [unit_weight(r) for r in RD.diag_requests(RD.diag_rows(index_rows or []))]
    return w


def score_stage(stage: str, decide: Callable[[dict], dict], steer: Callable[[dict], dict],
                items_by_role: dict[str, list], records_by_id: dict[str, dict], variants: list[dict],
                index_rows: list[dict] | None = None, progress: str | Path | None = None) -> dict[str, Any]:
    """One arm's units for one stage. ``decide``: ``decide.Decider.decide``-compatible; ``steer``: a safety_eval
    decide callable (``DecideBridge``). Used for the child here and for the parent in ``jobs/m2_parent_arm``.
    ``progress``: optional live-progress file for the rate guard (scores are unchanged by it)."""
    from safety_eval import metrics

    if isinstance(progress, Progress):  # planned by the caller (several stages share one progress file)
        pr = progress
    else:
        pr = Progress(progress)
        pr.plan(stage_unit_weights(stage, items_by_role, records_by_id, variants, index_rows) if progress else [])
    decide, steer = pr.wrap(decide), pr.wrap(steer)
    t0 = time.perf_counter()
    out: dict[str, Any] = {"stage": stage, "decision": {}, "decision_errors": {}}
    for role in STAGE_ROLES[stage]:
        f, err = arm_decision_fields(decide, records_by_id, items_by_role[role])
        out["decision"][role], out["decision_errors"][role] = f, err
    out["steer_rows"] = metrics.run(steer, variants) if variants else []
    if _uses_index(stage):
        out["index_results"] = index_predict(decide, index_rows or [])
    if stage == "index":  # readout diagnostic: same options, forward vs reversed presentation (not a criterion)
        from jobs.m2_first_cell import readout_diag as RD

        out["order_diag"] = RD.order_diagnostic(decide, RD.diag_rows(index_rows or []))
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
    if stage == "index":
        out = index_measurement(index_rows or [], child["index_results"], parent["index_results"], margins, n_boot)
        from jobs.m2_first_cell import readout_diag as RD

        out["readout_diagnostics"] = {
            "position_bias": {"child": RD.position_bias(index_rows or [], child["index_results"]),
                              "parent": RD.position_bias(index_rows or [], parent["index_results"])},
            "order": {"child": child["order_diag"]["summary"], "parent": parent["order_diag"]["summary"],
                      "paired": RD.paired_order(child["order_diag"], parent["order_diag"], n_boot=n_boot)}}
        return out
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


def index_measurement(rows: list[dict], child_results: dict, parent_results: dict, margins: dict,
                      n_boot: int = 2000) -> dict[str, Any]:
    """Index DEV subset, child vs parent (measurement stage ``index``): absolute Index score + 95% CI per arm and per
    area, the paired difference (``harness.bootstrap.compare``), debiased ECE per arm, and the G13 capability / ECE
    criteria evaluated for INFORMATION (``retention_comparison`` under the same draws). ``accept`` mirrors those two
    criteria so the report shape matches the gate stages, but this stage is never part of the promotion verdict."""
    from harness import bootstrap as HB
    from harness import calibration as HC
    from harness import score as S
    from train import promotion as P

    # Rows that BOTH arms refuse for the same deterministic reason (e.g. input_tokens_exceeded: 8 of the 2,008 subset
    # rows exceed the 16,384-token input cap, job-tkfaj) are scored as the kit scores them (not ok = wrong) in the
    # Index scores below, and left out of the strict paired criteria, which require every compared row to be ok.
    def _code(r):
        try:
            return (json.loads((r or {}).get("error") or "null") or {}).get("code")
        except (ValueError, AttributeError):
            return (r or {}).get("error")

    def _same_refusal(rid):
        c, p = child_results.get(rid), parent_results.get(rid)
        return (c is not None and p is not None and c.get("status") != "ok" and p.get("status") != "ok"
                and _code(c) is not None and _code(c) == _code(p))

    both = sorted(r["_evaluation"]["run_id"] for r in rows if _same_refusal(r["_evaluation"]["run_id"]))
    crit_rows = [r for r in rows if r["_evaluation"]["run_id"] not in set(both)]
    ret = P.retention_comparison(crit_rows, child_results, parent_results, B=n_boot, seed=0)
    v = P.verdict_g13(retention=ret, steer_child_rows=[], steer_parent_rows=[], expected_variants=[],
                      margins=margins, n_boot=n_boot)
    crits = [c for c in v["criteria"] if c["criterion"] in ("g13_capability", "g13_ece")]
    out: dict[str, Any] = {"accept": all(c["pass"] for c in crits), "failed": [c["criterion"] for c in crits
                                                                               if not c["pass"]],
                           "criteria": crits, "margins": dict(margins), "informational": True,
                           "cohort": "Index DEV 2,008-row subset (development slice; TEST sealed); not a promotion stage"}
    out["rows_not_ok_in_both_arms"] = {"n": len(both), "run_ids": both[:20],
                                       "treatment": "scored as the kit scores them (not ok = no credit) in 'scores'; "
                                                    "excluded from the paired criteria"}
    missing = [r["_evaluation"]["run_id"] for r in rows for res in (child_results, parent_results)
               if r["_evaluation"]["run_id"] not in res]
    if not missing:  # official kit scoring over EVERY subset row (same rows, same treatment as job-tkfaj / the board)
        cmp = HB.compare(S.decompose(rows, child_results), S.decompose(rows, parent_results), B=n_boot, seed=0)
        out["scores"] = {"child": cmp["candidate"], "parent": cmp["baseline"], "diff": cmp["diff"],
                         "B": n_boot, "seed": 0}
        out["ece"] = {"child": HC.report(HC.fields(rows, child_results)),
                      "parent": HC.report(HC.fields(rows, parent_results)), "diff": ret.get("ece_diff")}
    else:
        out["scores_error"] = f"{len(missing)} subset rows have no result in an arm, e.g. {missing[:3]}"
    out["paired"] = {"index_dev": ret}
    return out


def promotion_verdict(verdicts: dict[str, dict]) -> dict[str, Any]:
    from train.promotion import MARGINS

    verdicts = {s: v for s, v in verdicts.items() if s in STAGES}
    failed = {s: v["failed"] for s, v in verdicts.items() if not v["accept"]}
    pending = "PENDING USER APPROVAL" in str(MARGINS.get("status"))
    acc = not failed and bool(verdicts)
    return {"verdict": "accept" if acc else "reject", "stages": sorted(verdicts), "failed": failed,
            "margins": {k: v for k, v in MARGINS.items()}, "margins_pending_user_approval": pending,
            "promotable": acc and not pending and set(verdicts) == set(STAGES),
            "rule": "accept iff every stage run accepted (fail-closed); promotable only with both G12 and G13 and "
                    "user-approved margins"}


def exit_code(verdicts: dict[str, dict]) -> int:
    verdicts = {s: v for s, v in verdicts.items() if s in STAGES}  # measurement stages never set the exit code
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
              out: str | Path | None = None, extra: dict | None = None,
              progress_dir: str | Path | None = None) -> tuple[int, dict]:
    """Score the child per stage, load the matching parent units, apply the verdicts and the promotion rule."""
    from train.promotion import MARGINS

    report: dict[str, Any] = {"schema": REPORT_SCHEMA, "identities": {"cohort": cohort.identity(), "unit_spec": spec,
                                                                      "child": child_identity, **(extra or {})},
                              "verdicts": {}, "stages": {}}
    variants_by = {s: stage_variants(cohort, s) for s in stages}
    pr = Progress(Path(progress_dir) / "progress.jsonl" if progress_dir else None)
    if progress_dir:  # one progress file over every stage of this run (the rate guard projects the whole phase)
        pr.plan([w for s in stages for w in stage_unit_weights(s, items_by_role, records_by_id, variants_by[s],
                                                                index_rows if _uses_index(s) else None)])
    for stage in stages:
        variants = variants_by[stage]
        ident = units_identity(stage, cohort, spec, {r: items_by_role[r] for r in STAGE_ROLES[stage]}, variants,
                               index_rows if _uses_index(stage) else None)
        parent, ent = read_arm_cache(parent_arm, stage, ident, expect_parent, parent_arm_sha256)
        child = score_stage(stage, child_decide, child_steer, items_by_role, records_by_id, variants,
                            index_rows if _uses_index(stage) else None, progress=pr)
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
    ap.add_argument("--stage", choices=["g12", "g13", "index", "g12+index", "all"], default="all",
                    help="all = g12 + g13 (promotion); g12+index = the first-cell measurement (jobs/m2_first_cell)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-dir", default=None, help="cell config train_set: the pinned training-set dir "
                    "(the training items are rebuilt from it for the checkpoint lineage check)")
    ap.add_argument("--progress-dir", default=None, help="live progress for jobs.common.rate_guard run-weighted")
    a = ap.parse_args(argv)
    stages = {"all": list(STAGES), "g12+index": ["g12", "index"]}.get(a.stage, [a.stage])
    if any(_uses_index(s) for s in stages) and not a.index_rows:
        print(f"{a.stage} needs --index-rows (Index DEV subset)", file=sys.stderr)
        return EXIT["usage"]

    import torch

    from decide import Decider
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel
    from jobs.m2_pilot.guard_ready import GuardNotReady, load_ready_guard
    from train.cohort import Cohort, CohortLeak
    from train.evalcard import run_card
    from train.run_cell import cell_items, load_cell_data, load_parent_backbone, parent_encoder, train_config
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
    try:
        train_recs, recs, tr_ids, _, _ = load_cell_data(cfg, a.records, a.train_dir, cohort)
        cohort.assert_training_ids(tr_ids)
    except (ValueError, CohortLeak) as e:
        print(f"cohort: {e}", file=sys.stderr)
        return EXIT["identity"]
    role_recs = {}
    for role in ("dev", "calibration", "retention"):
        rr, _ = cohort.load_role_records(a.records, role)
        role_recs[role] = [recs[r["id"]] for r in rr]
    t_max = cell["ladder"]["T_max"]
    train_items = [x for x in cell_items(train_recs, tr_ids, cfg, enc, guard, max(cfg["t_ladder"]))
                   if len(x.row.ids) <= t_max]
    bb = load_parent_backbone(parent, a.model_dir, cfg, a.gdn_ckpt, device=a.device)
    cuda = torch.cuda.is_available() and str(a.device).startswith("cuda")  # follows --device
    model = DecisionModel(bb, cfg["head"], cfg.get("head_kwargs"), pad_id=0, seed=cfg["seed"])
    model.to(bb.device)
    from train.trainer import with_execution  # same execution flags as training (not in the config hash)

    tr = Trainer(model, with_execution(train_config(cfg, cfg["max_steps"], a.ckpt_dir, cuda), cfg), train_items,
                 guard=guard)
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
                          parent_arm_sha256=a.parent_arm_sha256, progress_dir=a.progress_dir,
                          extra={"guard": guard.identity(), "checkpoint": backend.checkpoint,
                                 "lineage": tr.lineage, "backbone": backbone_identity(bb),
                                 "temperature": {**ft.identity(), "disjointness": ft.disjointness,
                                                 "proof_sha256": ft.proof_sha256, "fit_report": ft.fit_report},
                                 "card": {k: card.get(k) for k in ("step", "n_items", "raw", "tempered")},
                                 "command": sys.argv})


if __name__ == "__main__":
    sys.exit(main())
