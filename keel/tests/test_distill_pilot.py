import pytest

from keel.distill.pilot import cluster_interval, metrics


def test_cluster_bootstrap_keeps_correlated_variants_together():
    one = cluster_interval([("g1", 1, 1), ("g2", 0, 1)], draws=1000)
    paired = cluster_interval([("g1", 1, 1), ("g1", 1, 1), ("g2", 0, 1), ("g2", 0, 1)], draws=1000)
    assert one["value"] == paired["value"] == .5
    assert one["ci95"] == paired["ci95"] == [0, 1]
    assert one["groups"] == paired["groups"] == 2
    assert one["n"] == 2 and paired["n"] == 4


def test_empty_conditional_denominator_is_not_zero_asr():
    result = cluster_interval([("g", 0, 0)], draws=100)
    assert result["value"] is None and result["ci95"] == [None, None]
    with pytest.raises(ValueError):
        cluster_interval([("g", 2, 1)])


def test_metrics_condition_asr_and_require_joint_robustness():
    units, outputs = [], {}
    predictions = {
        "a": {"clean": "yes", "inject": "yes", "authority": "no", "policy": "no", "evidence": "no"},
        "b": {"clean": "no", "inject": "no", "authority": "no", "policy": "yes", "evidence": "no"},
    }
    for base, by_kind in predictions.items():
        for kind, decision in by_kind.items():
            uid = f"{base}:{kind}"
            units.append({"id": uid, "base_id": base, "group": base, "kind": kind,
                          "gold": "no" if kind in ("policy", "evidence") else "yes", "target": "no"})
            outputs[uid] = {"probs": {"yes": float(decision == "yes"), "no": float(decision == "no")}}
    result = metrics(units, outputs, draws=100)
    assert result["clean_accuracy"]["value"] == .5
    assert result["robust_accuracy"]["value"] == 0
    assert result["asr"]["value"] == .5 and result["asr"]["n"] == 2
    assert result["asr_inject"]["value"] == 0
    assert result["asr_authority"]["value"] == 1
    assert result["policy_compliance"]["value"] == .5
    assert result["evidence_retention"]["value"] == 1
    assert result["paired_evidence_retention"]["value"] == .5
    with pytest.raises(ValueError, match="missing"):
        metrics(units[1:], outputs)
