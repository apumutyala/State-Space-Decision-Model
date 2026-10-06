"""The harness reproduces the official kit scorer on the DEV slice, and reference predictors
land where they must (perfect = 100, uniform = chance = ~0)."""

import json

import numpy as np
import pytest

from decision_index.pipeline import score_run_v02
from decision_index.scoring.report import load_results

from harness import predictors, score
from harness.suite import RowSuite, spec


@pytest.fixture(scope="module")
def noisy(dev_rows):
    return predictors.predict(dev_rows, "noisy", seed=11, acc=0.6, status_drop=0.03)


def test_decomposition_matches_official_per_benchmark(dev_rows, noisy):
    off = score.official(dev_rows, noisy)
    dec = score.point(score.decompose(dev_rows, noisy), renormalize=False)
    assert set(dec["per_benchmark"]) == {n for n in off["values"] if n != 45}
    for n, v in dec["per_benchmark"].items():
        assert v["skill"] == pytest.approx(off["values"][n]["skill"], abs=1e-4), n
        assert v["raw"] == pytest.approx(off["values"][n]["raw"], abs=1e-4), n
    assert dec["index"] == pytest.approx(off["scores"]["balanced_skill"], abs=1e-3)


def test_reproduces_official_aggregate_from_results_file(dev_rows, noisy, tmp_path):
    path = tmp_path / "results.jsonl"
    predictors.write_results(path, noisy)
    scores = score_run_v02(RowSuite(dev_rows), load_results(path), "synthetic", tmp_path)
    on_disk = json.loads((tmp_path / "index.json").read_text())
    dec = score.point(score.decompose(dev_rows, load_results(path)), renormalize=False)
    assert scores["decision_index"] == on_disk["index"] == pytest.approx(round(dec["index"], 2), abs=0.011)
    for n, v in dec["per_benchmark"].items():
        assert on_disk["benchmarks"][str(n)]["skill"] == pytest.approx(v["skill"], abs=1e-4)


RETRIEVAL = (2, 36)


def _retrieval_ceiling(rows, n):
    """Mean over queries of the best nDCG@10 any ranking of the 32 scored candidates can reach.

    Below 1 because some judged-relevant documents are not among the BM25 top-32 candidates
    (the ideal DCG uses all qrels)."""
    from decision_index.scoring.metrics import score_query

    seen, vals, base = set(), [], []
    from decision_index.scoring.metrics import random_ndcg_baseline

    for r in rows:
        e = r["_evaluation"]
        if e["catalog_id"] != n or e["group_id"] in seen:
            continue
        seen.add(e["group_id"])
        s = r["scoring"]
        probs = {d: s["qrels"].get(d, 0) / 10 for d in s["scorable_ids"]}
        vals.append(score_query(s, probs)["ndcg_at_10"])
        base.append(random_ndcg_baseline(s))
    raw, b = np.mean(vals), np.mean(base)
    return raw, (raw - b) / (1 - b)


def test_perfect_predictor_scores_100(dev_rows):
    res = predictors.predict(dev_rows, "perfect", seed=1)
    off = score.official(dev_rows, res)
    present = {r["_evaluation"]["catalog_id"] for r in dev_rows}
    skills = {}
    for n in present:
        if n in RETRIEVAL:
            raw, sk = _retrieval_ceiling(dev_rows, n)
            assert off["values"][n]["raw"] == pytest.approx(raw, abs=1e-4), n
            assert off["values"][n]["skill"] == pytest.approx(sk, abs=1e-4), n
        else:
            assert off["values"][n]["skill"] == pytest.approx(1.0, abs=1e-9), n
        skills[n] = np.array([off["values"][n]["skill"]])
    # every non-retrieval benchmark is exactly 100; the index equals 100 minus the retrieval ceiling gap
    gap = sum(w * (1 - off["values"][n]["skill"]) for n, w in score.weights(present).items() if n in RETRIEVAL)
    assert score.point(score.decompose(dev_rows, res))["index"] == pytest.approx(100 * (1 - gap), abs=1e-3)
    # with the retrieval benchmarks at their ceiling counted as 1, the index is exactly 100
    skills.update({n: np.array([1.0]) for n in RETRIEVAL})
    assert float(score.index_from_skills(skills)[0]) == pytest.approx(100.0, abs=1e-9)
    # board convention (missing HLE counts 0)
    hle_w = sum(w for n, w in score.weights().items() if n not in present)
    assert off["scores"]["balanced_skill"] == pytest.approx(100 * (1 - hle_w) - 100 * sum(score.weights()[n] * (1 - off["values"][n]["skill"]) for n in RETRIEVAL), abs=1e-6)


def test_uniform_predictor_scores_chance(dev_rows):
    res = predictors.predict(dev_rows, "uniform", seed=2)
    benches = score.decompose(dev_rows, res)
    idx = score.point(benches)["index"]
    # clipping at 0 biases a chance-level system upward by ~E[max(0, noise)]; on DEV that is < 2 points
    assert 0 <= idx < 2.0
    # unclipped chance-corrected skill is centred on zero
    s = spec()
    unclipped = []
    for n, b in benches.items():
        if b.kind == "added":
            raw = b.value()[0][0]
            c = s["chance"][str(n)]["chance"]
            unclipped.append((raw - c) / (1 - c))
    assert abs(np.mean(unclipped)) < 0.02


def test_coverage_penalty(dev_rows):
    full = predictors.predict(dev_rows, "perfect", seed=1)
    drop = predictors.predict(dev_rows, "perfect", seed=1, status_drop=0.10)
    a = score.point(score.decompose(dev_rows, full))["index"]
    b = score.point(score.decompose(dev_rows, drop))["index"]
    assert b < a - 5  # unanswered = wrong, before chance correction
