import math

import pytest

from jevkit.answers import answer_from_probs, parse_answer, validate_distribution

from ._fixtures import ch_ans, choice, nl_ans, noul, sc_ans, score


def test_exact_label_set_required():
    it = choice("c", k=3)
    assert not parse_answer(it, ch_ans({"o0": 0.5, "o1": 0.5})).valid  # missing
    assert not parse_answer(it, ch_ans({"o0": 0.5, "o1": 0.25, "o2": 0.25, "x": 0.0})).valid  # extra
    assert parse_answer(it, ch_ans({"o0": 0.5, "o1": 0.25, "o2": 0.25})).valid


def test_sum_tolerance_bands():
    it = choice("c", k=2)
    p = parse_answer(it, ch_ans({"o0": 0.6, "o1": 0.4005}))
    assert p.valid and p.strict_valid and not p.renormalized
    p = parse_answer(it, ch_ans({"o0": 0.6, "o1": 0.415}))  # sum 1.015: inside 2 %, renormalised
    assert p.valid and p.renormalized and not p.strict_valid
    assert sum(p.probs.values()) == pytest.approx(1.0)
    assert not parse_answer(it, ch_ans({"o0": 0.6, "o1": 0.43})).valid  # 1.03: invalid, never repaired


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.2, True, "0.5", None])
def test_non_numbers_invalid(bad):
    assert not parse_answer(choice("c", k=2), ch_ans({"o0": bad, "o1": 0.5})).valid


def test_argmax_not_choice_field_and_tie_break():
    it = choice("c", k=3)
    p = parse_answer(it, {"type": "choice", "choice": "o2", "probabilities": {"o0": 0.4, "o1": 0.4, "o2": 0.2}})
    assert p.predicted == "o0" and p.choice_field_mismatch


def test_label_only_choice():
    p = parse_answer(choice("c", k=3), {"type": "choice", "choice": "o1"})
    assert p.valid and p.label_only and p.probs is None and p.predicted == "o1"
    assert not parse_answer(choice("c", k=3), {"type": "choice", "choice": "zz"}).valid


def test_noul_thresholds_and_abstention():
    it = noul("n")
    assert parse_answer(it, nl_ans(0.8)).predicted == "yes"
    assert parse_answer(it, nl_ans(0.2)).predicted == "no"
    p = parse_answer(it, nl_ans(0.5))
    assert p.valid and p.predicted is None  # abstention: valid for calibration, wrong for Intelligence
    assert not parse_answer(it, nl_ans(1.01)).valid
    assert not parse_answer(it, {"type": "choice", "noul": 0.9}).valid  # type mismatch


def test_score_expected_level():
    it = score("s", k=4)
    p = parse_answer(it, sc_ans([0.1, 0.2, 0.3, 0.4]))
    assert p.expected_level == pytest.approx(2.0) and p.predicted == "3"
    p = parse_answer(it, {"type": "score", "score": 1.4})
    assert p.valid and p.label_only and p.expected_level == 1.4
    assert not parse_answer(it, {"type": "score", "score": 3.5}).valid


def test_answer_from_probs_roundtrip():
    for it, probs in ((choice("c", k=3), {"o0": 0.2, "o1": 0.5, "o2": 0.3}),
                      (score("s", k=3), {"0": 0.2, "1": 0.5, "2": 0.3}),
                      (noul("n"), {"no": 0.25, "yes": 0.75})):
        p = parse_answer(it, answer_from_probs(it, probs))
        assert p.valid and all(math.isclose(p.probs[k], v) for k, v in probs.items())


def test_validate_distribution_reports_reason():
    with pytest.raises(ValueError, match="label set"):
        validate_distribution({"a": 1.0}, ["a", "b"])
