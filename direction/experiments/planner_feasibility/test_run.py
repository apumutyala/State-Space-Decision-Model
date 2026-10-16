from direction.experiments.planner_feasibility.run import ARMS, in_outputs, literal, summarize


def test_literal_boundaries_and_nested_values():
    assert literal(["alice@example.org", "report.csv"], "Send report.csv to alice@example.org")
    assert not literal(12, "amount 1200")
    assert not literal(True, "True")
    assert not literal("", "anything")


def test_output_matching_does_not_coerce_identifiers():
    assert in_outputs("DE02120300000000202051", ["IBAN: DE02120300000000202051"])
    assert not in_outputs(12, ["12"])
    assert not in_outputs("", ["anything"])
    assert not in_outputs("26", ["2026-01-01", "visit on June 26"])
    assert not in_outputs("r", ["arbitrary"])
    assert not in_outputs("", [""])
    assert in_outputs("26", ["26"], exact=True)
    assert not in_outputs("26", ["ID 26"], exact=True)


def test_bootstrap_resamples_tasks_not_repeated_attack_calls():
    rows = []
    for task, value in enumerate((0.0, 1.0)):
        scores = {arm: {"utility": value, "complete": value, "denial": value,
                        "oracle_denied": 100, "recovered": int(value * 100), "granted_tools": 2}
                  for arm in ARMS}
        rows.append({"suite": "synthetic", "scores": scores, "argument_sources": [],
                     "discovery_errors": [], "missing_identity_binding": bool(value)})
    result = summarize(rows, resamples=1000, seed=17)
    assert result == summarize(rows, resamples=1000, seed=17)
    metrics = result["synthetic"]["arms"]["literals"]
    assert metrics["utility"] == {"estimate": 0.5, "ci95": [0.0, 1.0]}
    assert metrics["oracle_denial_recall"] == {"estimate": 0.5, "ci95": [0.0, 1.0]}
    assert result["overall"]["tasks"] == 2
