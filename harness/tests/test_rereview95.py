"""PR #95 re-review: missing results are refused (B2), unaligned pairs are refused, plain replicates need an
explicit edition and score.point checks its edition (S1), and the TEST GSM8K addendum is gated at the source (S2)."""

import numpy as np
import pytest

from harness import bootstrap, config, gsm8k03, predictors, score
from harness.tests.conftest import needs_suite

needs_gsm = pytest.mark.skipif(not config.GSM8K_03_ROWS.exists(), reason="rebuilt 0.3 GSM8K rows not built locally")


@pytest.fixture(scope="module")
def rows03():
    from harness.cli import load_slice

    return load_slice("dev", "0.3")


@pytest.fixture(scope="module")
def full_and_old(rows03):
    full = predictors.predict(rows03, "noisy", seed=31, acc=0.6, status_drop=0.03)
    old = {k: v for k, v in full.items() if not k.startswith(config.GSM8K_03_RUN_ID_PREFIX)}  # file without twins
    return full, old


@needs_suite
@needs_gsm
def test_missing_results_are_refused_unless_allowed(rows03, full_and_old):
    full, old = full_and_old
    assert score.missing_results(rows03, full) == {}
    assert score.missing_results(rows03, old) == {30: 262}
    with pytest.raises(SystemExit, match="omit-gsm8k03"):
        score.require_results(rows03, old)
    assert score.require_results(rows03, old, allow_missing=True) == {30: 262}


@needs_suite
@needs_gsm
def test_unaligned_pair_is_refused(rows03, full_and_old):
    full, old = full_and_old
    a = score.decompose(rows03, full, edition="0.3")
    b = score.decompose(rows03, old, edition="0.3")
    with pytest.raises(ValueError, match="different rows"):
        bootstrap.compare(a, b, B=5)
    out = bootstrap.compare(a, b, B=5, allow_unaligned=True)  # explicit
    assert out["diff"]["benchmarks"]["30"]["point"] > 0  # the silent-zero hazard the check exists for
    assert bootstrap.compare(a, a, B=5)["diff"]["index"]["point"] == 0


@needs_suite
@needs_gsm
def test_cli_score_refuses_old_file_under_03(tmp_path, full_and_old):
    from harness.cli import main

    _, old = full_and_old
    path = tmp_path / "results.jsonl"
    predictors.write_results(path, old)
    with pytest.raises(SystemExit):
        main(["score", "--results", str(path), "--edition", "0.3", "--B", "5"])
    main(["score", "--results", str(path), "--edition", "0.3", "--B", "5", "--omit-gsm8k03"])
    main(["score", "--results", str(path), "--edition", "0.3", "--B", "5", "--allow-missing"])


@needs_suite
@needs_gsm
def test_dev_subset_03_path_checks_counts_and_missing(tmp_path):
    from harness.cli import load_slice
    from jobs.common import dev_subset

    sub = dev_subset.load_subset()
    rows21 = [r for r in load_slice("dev") if r["_evaluation"]["run_id"] in set(sub["run_ids"])]
    path = tmp_path / "r.jsonl"
    predictors.write_results(path, predictors.predict(rows21, "noisy", seed=5, acc=0.6))  # an old-style file
    with pytest.raises(SystemExit):
        dev_subset.main(["score", "--results", str(path), "--edition", "0.3", "--B", "5"])
    assert dev_subset.main(["score", "--results", str(path), "--edition", "0.3", "--B", "5", "--omit-gsm8k03"]) == 0
    assert len(dev_subset._rows(sub, "0.3")) == len(dev_subset._rows(sub, "0.3", False)) + 44


def test_plain_replicates_need_an_explicit_edition():
    plain = {1: (np.array([0.5, 0.5]), np.array([0.5, 0.5]))}
    with pytest.raises(ValueError, match="untagged"):
        bootstrap.summarize_one(plain)
    assert bootstrap.summarize_one(plain, edition="0.3")["edition"] == "0.3"
    tagged = bootstrap.Replicates(plain, "0.3")
    with pytest.raises(ValueError):
        bootstrap.summarize_one(tagged, edition="0.2.1")


@needs_suite
def test_point_checks_explicit_edition(dev_rows):
    res = predictors.predict(dev_rows, "noisy", seed=3, acc=0.6)
    b03 = score.decompose(dev_rows, res, edition="0.3")
    assert score.point(b03, edition="0.3")["edition"] == "0.3"
    with pytest.raises(ValueError):
        score.point(b03, edition="0.2.1")


def test_test_addendum_is_gated_at_the_source(monkeypatch):
    monkeypatch.delenv("SSDM_UNSEAL_TEST", raising=False)
    with pytest.raises(SystemExit, match="sealed"):
        gsm8k03.addendum_rows("test")
