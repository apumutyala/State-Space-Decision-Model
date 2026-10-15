"""Learn-then-Test of the COMPLETE local gate, with clusters as the risk units.

No model fitting occurs here. Fix the judge, detectors, population, loss, grid and
weights independently of calibration. Fit/calibration/evaluation cluster overlap
is rejected. Evaluation labels are not inspected until threshold selection ends.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

from jevkit.guarantees import (Scored, binom_cdf, cluster_pick, crr_betting_pvalue, ids_sha256, ltt_select)
from keel.protocol import DecisionRequest, sha256_json

from .gate import ActDeferGate, GateAssessment, GateThresholds, JudgeOutputs

CERTIFICATION_VERSION = "keel-gate-certification/1"


@dataclass(frozen=True)
class LabeledDecision:
    """One request-level decision; any wrong answer makes the whole action wrong.

    Gold uses protocol decision labels (choice option ids, score level strings,
    and yes/no for noul). It is passed only to certification, never to a signal.
    ``under_attack`` is an independently supplied annotation, not a detector score.
    """

    request: DecisionRequest
    outputs: JudgeOutputs
    gold: Mapping[str, str]
    cluster_id: str
    under_attack: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cluster_id, str) or not self.cluster_id:
            raise ValueError("cluster_id must be a nonempty string")
        if not isinstance(self.request.request_id, str) or not self.request.request_id:
            raise ValueError("request_id must be a nonempty string")
        if self.under_attack is not None and not isinstance(self.under_attack, bool):
            raise ValueError("under_attack must be bool or None")
        if set(self.gold) != {question.qid for question in self.request.questions}:
            raise ValueError("gold must cover exactly the request's questions")
        for question in self.request.questions:
            labels = {"yes", "no"} if question.type == "noul" else set(self.outputs.trust[question.qid])
            if self.gold[question.qid] not in labels:
                raise ValueError(f"{question.qid}: gold is not a protocol decision label")
        object.__setattr__(self, "gold", MappingProxyType(dict(self.gold)))

    @property
    def id(self) -> str:
        return self.request.request_id


@dataclass(frozen=True)
class ClusterSplits:
    fit: tuple[LabeledDecision, ...]
    calibration: tuple[LabeledDecision, ...]
    evaluation: tuple[LabeledDecision, ...]
    rule: str = "externally assigned disjoint clusters"

    def __post_init__(self) -> None:
        for name in ("fit", "calibration", "evaluation"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def split_by_cluster(rows: Sequence[LabeledDecision], *, seed: int = 0, fit_fraction: float = 0.2,
                     calibration_fraction: float = 0.4) -> ClusterSplits:
    """Answer-independent hash assignment; every cluster stays in one split.

    Set fractions and seed before inspecting outcomes. Hashing can leave a small
    split empty; certification requires nonempty calibration. Empty fit is valid
    only for an already-fixed pipeline; empty evaluation makes no evaluation claim.
    """
    if not (math.isfinite(fit_fraction) and math.isfinite(calibration_fraction)
            and 0 < fit_fraction < 1 and 0 < calibration_fraction < 1
            and fit_fraction + calibration_fraction < 1):
        raise ValueError("split fractions must be positive with sum below one")
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("decision ids must be unique")
    parts: list[list[LabeledDecision]] = [[], [], []]
    for row in rows:
        value = int(hashlib.sha256(f"{seed}|keel-gate-split/1|{row.cluster_id}".encode()).hexdigest(), 16) / 2**256
        index = 0 if value < fit_fraction else (1 if value < fit_fraction + calibration_fraction else 2)
        parts[index].append(row)
    return ClusterSplits(*(tuple(part) for part in parts),
                         rule=f"sha256({seed}|keel-gate-split/1|cluster): fit<{fit_fraction}; "
                              f"calibration<{fit_fraction + calibration_fraction}; else evaluation")


@dataclass(frozen=True)
class CertificationPlan:
    """Immutable preregistration specification, not proof of when it was registered.

    The caller must commit this plan before accessing calibration/evaluation data.
    ``pipeline_id`` must identify the pinned judge, rendering and frozen detectors.
    ``wrong_or_attack`` controls the union event, hence bounds each component too.
    A fixed-sequence path is explicit, ordered, and may be a subset of the grid.
    """

    population: str
    pipeline_id: str
    uncertainty_thresholds: tuple[float, ...]
    manipulation_thresholds: tuple[float, ...]
    epsilon: float = 0.1
    delta: float = 0.05
    loss: str = "wrong"
    method: str = "bonferroni"
    cluster_test: str = "crr"
    crr_cap: float = 1.0
    test_seed: int = 0
    fixed_sequence: tuple[GateThresholds, ...] = ()

    def __post_init__(self) -> None:
        if not self.population.strip() or not self.pipeline_id.strip():
            raise ValueError("population and pipeline_id must be declared")
        for name in ("uncertainty_thresholds", "manipulation_thresholds"):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError("threshold axes must be nonempty and unique")
            object.__setattr__(self, name, values)
        for point in self.grid:
            GateThresholds(point.uncertainty, point.manipulation)
        if not (0 < self.epsilon < 1 and 0 < self.delta < 1):
            raise ValueError("epsilon and delta must be strictly between zero and one")
        if self.loss not in ("wrong", "wrong_or_attack"):
            raise ValueError("loss must be wrong or wrong_or_attack")
        if self.method not in ("bonferroni", "fixed_sequence"):
            raise ValueError("method must be bonferroni or fixed_sequence")
        if self.cluster_test not in ("crr", "pick"):
            raise ValueError("cluster_test must be crr or pick")
        if not math.isfinite(self.crr_cap) or self.crr_cap < 1:
            raise ValueError("crr_cap must be finite and at least one")
        object.__setattr__(self, "fixed_sequence", tuple(self.fixed_sequence))
        if self.method == "fixed_sequence":
            if (not self.fixed_sequence or len(set(self.fixed_sequence)) != len(self.fixed_sequence)
                    or not set(self.fixed_sequence) <= set(self.grid)):
                raise ValueError("fixed_sequence must be a nonempty, duplicate-free path inside the grid")
        elif self.fixed_sequence:
            raise ValueError("fixed_sequence is only used with method='fixed_sequence'")

    @property
    def grid(self) -> tuple[GateThresholds, ...]:
        return tuple(GateThresholds(uncertainty, manipulation)
                     for uncertainty in self.uncertainty_thresholds for manipulation in self.manipulation_thresholds)


@dataclass(frozen=True)
class _Observed:
    row: LabeledDecision
    assessment: GateAssessment
    wrong: bool
    loss: bool


def _measure(rows: Sequence[LabeledDecision], gate: ActDeferGate, loss: str) -> list[_Observed]:
    observed = []
    for row in rows:
        if loss == "wrong_or_attack" and row.under_attack is None:
            raise ValueError("wrong_or_attack requires attack labels for every calibration/evaluation unit")
        assessment = gate.assess(row.request, row.outputs)
        wrong = any(assessment.predictions[qid] != gold for qid, gold in row.gold.items())
        observed.append(_Observed(row, assessment, wrong, wrong or (loss == "wrong_or_attack" and row.under_attack)))
    return observed


def _manifest(splits: ClusterSplits) -> dict:
    seen_ids: set[str] = set()
    seen_clusters: set[str] = set()
    manifest = {"rule": splits.rule}
    for name in ("fit", "calibration", "evaluation"):
        rows = getattr(splits, name)
        ids, clusters = {row.id for row in rows}, {row.cluster_id for row in rows}
        if len(ids) != len(rows) or ids & seen_ids:
            raise ValueError("decision ids must be unique across all splits")
        if clusters & seen_clusters:
            raise ValueError("fit/calibration/evaluation splits overlap by cluster")
        seen_ids.update(ids)
        seen_clusters.update(clusters)
        manifest[name] = {"n_decisions": len(rows), "n_clusters": len(clusters),
                          "ids_sha256": ids_sha256(ids), "clusters_sha256": ids_sha256(clusters)}
    if not splits.calibration:
        raise ValueError("calibration must contain at least one cluster")
    return manifest


def _statistics(observed: Sequence[_Observed], point: GateThresholds, cap: float) -> dict:
    by_cluster: dict[str, dict] = {}
    attack_labels_complete = bool(observed) and all(unit.row.under_attack is not None for unit in observed)
    for unit in observed:
        cluster = by_cluster.setdefault(unit.row.cluster_id,
                                        {"cluster_id": unit.row.cluster_id, "size": 0, "n_act": 0,
                                         "wrong": 0, "attack_acts": 0, "losses": 0})
        cluster["size"] += 1
        if point.accepts(unit.assessment.uncertainty, unit.assessment.manipulation):
            cluster["n_act"] += 1
            cluster["wrong"] += int(unit.wrong)
            cluster["attack_acts"] += int(unit.row.under_attack is True)
            cluster["losses"] += int(unit.loss)
    clusters = [by_cluster[key] for key in sorted(by_cluster)]
    for cluster in clusters:
        cluster["weight"] = min(1.0, cap / cluster["size"])
        cluster["defer_rate"] = 1.0 - cluster["n_act"] / cluster["size"]
        if not attack_labels_complete:
            cluster["attack_acts"] = None
    weighted_size = math.fsum(cluster["weight"] * cluster["size"] for cluster in clusters)
    weighted_acts = math.fsum(cluster["weight"] * cluster["n_act"] for cluster in clusters)
    weighted_losses = math.fsum(cluster["weight"] * cluster["losses"] for cluster in clusters)
    weighted_wrong = math.fsum(cluster["weight"] * cluster["wrong"] for cluster in clusters)
    n_act = sum(cluster["n_act"] for cluster in clusters)
    losses = sum(cluster["losses"] for cluster in clusters)
    coverage = weighted_acts / weighted_size if weighted_size else None
    return {"n_decisions": len(observed), "n_clusters": len(clusters), "n_act": n_act,
            "losses": losses, "wrong": sum(cluster["wrong"] for cluster in clusters),
            "attack_acts": (sum(cluster["attack_acts"] for cluster in clusters) if attack_labels_complete else None),
            "weighted_acts": weighted_acts, "weighted_losses": weighted_losses,
            "coverage": coverage, "defer_rate": 1.0 - coverage if coverage is not None else None,
            "risk": weighted_losses / weighted_acts if weighted_acts else None,
            "wrong_risk": weighted_wrong / weighted_acts if weighted_acts else None,
            "unweighted_coverage": n_act / len(observed) if observed else None,
            "unweighted_defer_rate": 1.0 - n_act / len(observed) if observed else None,
            "unweighted_risk": losses / n_act if n_act else None, "per_cluster": clusters}


def _encoded_gate(observed: Sequence[_Observed], point: GateThresholds) -> list[Scored]:
    """Loss-preserving adapter for ltt_select, NOT new judge probabilities.

    A singleton at qhat=0 exists exactly when the complete gate acts. Its label
    agrees exactly when the requested loss is zero. Deferred rows remain in the
    input so representative selection cannot depend on the gate's decisions.
    """
    return [Scored(unit.row.id, "gate",
                   {"act": 1.0, "defer": 0.0} if point.accepts(unit.assessment.uncertainty,
                                                             unit.assessment.manipulation)
                   else {"act": 0.5, "defer": 0.5},
                   "defer" if unit.loss else "act") for unit in observed]


def _representative_ids(rows: Sequence[LabeledDecision], seed: int) -> set[str]:
    # cluster_pick only inspects ids/cluster keys; keep its exact shared selection rule.
    return {row.id for row in cluster_pick(rows, lambda row: row.cluster_id, seed)}


def certify(plan: CertificationPlan, gate: ActDeferGate, splits: ClusterSplits) -> dict:
    """Certify a threshold pair or explicitly return ``status='no certificate'``.

    All candidates use the same frozen complete gate; ``gate.thresholds`` is
    ignored in favour of the preregistered grid. Select maximum calibration
    coverage among certified candidates (ties use grid order). No evaluation
    value participates in selection. Returned curves are empirical, not bounds.
    """
    manifest = _manifest(splits)
    preregistration = {"plan": asdict(plan), "gate": gate.specification(), "risk_unit": "cluster"}
    level = plan.delta / len(plan.grid) if plan.method == "bonferroni" else plan.delta
    observed = _measure(splits.calibration, gate, plan.loss)
    selected_ids = _representative_ids(splits.calibration, plan.test_seed) if plan.cluster_test == "pick" else None
    tested_units = [unit for unit in observed if unit.row.id in selected_ids] if selected_ids is not None else observed
    clusters = {row.id: row.cluster_id for row in splits.calibration}
    fit = {"mondrian": False, "qhat": {"all": 0.0}, "_ids": frozenset(row.id for row in splits.fit),
           "ids_sha256": manifest["fit"]["ids_sha256"]}
    curve = []
    for point in plan.grid:
        stats = _statistics(tested_units, point, plan.crr_cap if plan.cluster_test == "crr" else 1.0)
        if plan.cluster_test == "crr":
            ordered = sorted(stats["per_cluster"],
                             key=lambda cluster: (hashlib.sha256(
                                 f"{plan.test_seed}|crr-order|{cluster['cluster_id']}".encode()).hexdigest(),
                                                  cluster["cluster_id"]))
            residuals = [cluster["weight"] * (cluster["losses"] - plan.epsilon * cluster["n_act"]) / plan.crr_cap
                         for cluster in ordered]
            p_value, log_capital = crr_betting_pvalue(residuals, plan.epsilon, level)
            rejected = stats["n_act"] > 0 and p_value <= level
        else:
            # A single-candidate LTT at the externally multiplicity-adjusted level.
            # Never use jevkit's internal fixed-sequence zero-act skipping policy.
            selection = ltt_select(_encoded_gate(observed, point), fit, plan.epsilon, level,
                                   lambdas=(1.0,), method="bonferroni", cluster_of=lambda row: clusters[row.id],
                                   pick_seed=plan.test_seed)
            p_value = binom_cdf(stats["losses"], stats["n_act"], plan.epsilon) if stats["n_act"] else 1.0
            log_capital = None
            rejected = selection["lambda"] is not None
        entry = {"thresholds": asdict(point), "calibration": stats, "p_value": p_value,
                 "log_capital": log_capital, "test_level": level, "tested": False, "certified": False,
                 "rejects_at_level": rejected}
        if selected_ids is not None:
            entry["all_decisions_calibration"] = _statistics(observed, point, 1.0)
        curve.append(entry)
    path = plan.grid if plan.method == "bonferroni" else plan.fixed_sequence
    for point in path:
        entry = curve[plan.grid.index(point)]
        entry["tested"] = True
        entry["certified"] = entry["rejects_at_level"]
        if plan.method == "fixed_sequence" and not entry["certified"]:
            break  # Including zero acts: no data-dependent skipping of hypotheses.
    accepted = [entry for entry in curve if entry["certified"]]
    best = max(accepted, key=lambda entry: entry["calibration"]["coverage"]) if accepted else None
    thresholds = best["thresholds"] if best is not None else None

    # Selection is now final. Only here are held-out outcomes/scores evaluated.
    evaluation = _measure(splits.evaluation, gate, plan.loss)
    evaluation_ids = _representative_ids(splits.evaluation, plan.test_seed) if plan.cluster_test == "pick" else None
    test_evaluation = ([unit for unit in evaluation if unit.row.id in evaluation_ids]
                       if evaluation_ids is not None else evaluation)
    for point, entry in zip(plan.grid, curve):
        entry["evaluation"] = _statistics(test_evaluation, point, plan.crr_cap if plan.cluster_test == "crr" else 1.0)
        if evaluation_ids is not None:
            entry["all_decisions_evaluation"] = _statistics(evaluation, point, 1.0)
    estimand = ("E[w_c L_c] / E[w_c A_c], w_c=min(1, crr_cap / full_cluster_size); "
                "coverage=E[w_c A_c]/E[w_c full_cluster_size]"
                if plan.cluster_test == "crr" else
                "P(loss | act) for the fixed ID-hash representative of a random cluster; "
                "coverage=P(representative acts)")
    return {"version": CERTIFICATION_VERSION, "status": "certified" if best is not None else "no certificate",
            "thresholds": thresholds, "deployment_action": "use certified thresholds" if best else "defer_all",
            "preregistration": preregistration, "preregistration_sha256": sha256_json(preregistration),
            "splits": manifest, "estimand": estimand, "epsilon": plan.epsilon, "delta": plan.delta,
            "n_certified": len(accepted), "risk_coverage_curve": curve,
            "review_load_curve": [{"thresholds": entry["thresholds"], "certified": entry["certified"],
                                   "calibration_defer_rate": entry["calibration"]["defer_rate"],
                                   "evaluation_defer_rate": entry["evaluation"]["defer_rate"],
                                   "calibration_unweighted_defer_rate": entry.get(
                                       "all_decisions_calibration", entry["calibration"])["unweighted_defer_rate"],
                                   "evaluation_unweighted_defer_rate": entry.get(
                                       "all_decisions_evaluation", entry["evaluation"])["unweighted_defer_rate"]}
                                  for entry in curve],
            "assumptions": ["independent exchangeable clusters from the declared deployment population",
                            "pipeline, signals, loss, grid/path, delta and weight rule fixed before calibration",
                            "cluster sizes include all decisions, not only acted or correctly labelled units",
                            "fit, calibration and evaluation clusters are disjoint; no evaluation-based selection",
                            "no guarantee against adaptive attackers or distribution shift"]}
