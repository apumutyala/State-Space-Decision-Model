from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from keel.distill.authoring import complete_pairs, materialize, plan, rebuild_options, summarize, validate_pair
from keel.distill.assembly_v1 import CANDIDATES, assemble, mix, select_candidate, validate_source
from keel.distill.sources import ContaminationError, assert_public_disjoint, digest
from keel.distill.triplets import request_of, unit
from keel.distill.verification import accept_verdict, arithmetic, blind_payload, capability_labels, check_calculation
from train import jevbench_guard as JG


def pair(group="pair-1"):
    job = {"id": group, "family": "documents_policies", "type": "noul"}
    obj = {"state": ("The archive admits a courier only if the courier has both a current permit and a matching seal. "
                     "A fee receipt alone does not replace either condition. The permit is current and was registered before noon. "
                     "The seal inspection found {{FACT}}. A separate courier arrived yesterday with a red bag but did not enter. "
                     "The archive log distinguishes deliveries from visits; this request is a visit. All visitors need a seal, "
                     "even when no parcels are being deposited. The courier has paid the relevant visiting fee."),
           "instructions": "Does this visitor qualify for admission under the archive's rules?",
           "options": ["yes", "no"], "answer": "yes", "criterion": "The visitor meets every admission condition.",
           "fact_a": "a matching seal", "fact_b": "a nonmatching seal", "answer_b": "no",
           "solution": "The permit is current; the seal is decisive because fee payment cannot substitute for it."}
    return materialize(job, obj, {"model": "Qwen/Qwen3.5-9B", "revision": "fixture"})


def labeled(row, capability=True):
    u = unit(row, "capability" if capability else "clean", request_of(row), row["labels"]["q"])
    u["teacher_probs"] = {"true": float(u["gold"] == "true"), "false": float(u["gold"] == "false")}
    if capability:
        u.update(pair=True, verification={"qwen_samples": 2, "cross_family": "openai/gpt-oss-20b",
                                        "verdict_sha256": "fixture", "eval_only": False})
    return u


def test_minimal_pair_has_single_fact_edit_and_flipped_label():
    rows = pair()
    validate_pair(rows)
    assert rows[0]["questions"] == rows[1]["questions"]
    assert {r["labels"]["q"] for r in rows} == {"true", "false"}
    for key, change in (("state", "An unrelated new event."), ("labels", {"q": "true"})):
        broken = copy.deepcopy(rows)
        broken[1][key] = change
        with pytest.raises(ValueError):
            validate_pair(broken)


def test_complete_pairs_preserves_exact_noul_balance_after_rejection():
    rows = pair("a") + pair("b") + pair("c")[:1]
    kept = complete_pairs(rows)
    assert len(kept) == 4
    assert summarize(kept)["noul_labels"] == {"true": 2, "false": 2}
    assert complete_pairs(pair()[:1]) == []


def test_planned_type_mix_and_all_seven_families():
    jobs = plan()
    counts = {t: sum((2 if t == "noul" else 1) for j in jobs if j["type"] == t) for t in ("noul", "choice", "score")}
    assert counts == {"noul": 3600, "choice": 2520, "score": 1080}
    assert len({j["family"] for j in jobs}) == 7
    assert plan(20) == plan(20)
    with pytest.raises(ValueError):
        plan(21)


def test_option_keys_rebuilt_from_exact_text_after_shuffle():
    options = ["Keep the existing owner", "Transfer to the successor", "Retire the asset", "Defer for missing records"]
    for seed in ("a", "b", "c", "d"):
        criteria, gold = rebuild_options(options, "Transfer to the successor", "choice", seed)
        assert criteria[gold] == "Transfer to the successor"
    assert rebuild_options(["small", "medium", "large"], "medium", "score", "x") == (["small", "medium", "large"], "1")
    with pytest.raises(ValueError):
        rebuild_options(["A", " a ", "B"], "A", "choice", "x")
    with pytest.raises(ValueError):
        rebuild_options(options, "o1", "choice", "x")


def test_public_overlap_fixture_fails_closed_all_matchers():
    from data.dedup import Fingerprint
    planted = {"state": "At sunrise the amber signal tower transmitted seventeen coded acknowledgments to the eastern depot.",
               "questions": pair()[0]["questions"], "_evaluation": {"catalog_id": JG.CATALOG_ID}}
    fp = Fingerprint.build([planted], {"source": "synthetic-planted-eval-fixture"}, log=lambda *_: None)
    fp.exact_hashes = JG.exact_segment_hashes([planted], set())
    with pytest.raises(ContaminationError):
        assert_public_disjoint({"id": "leak", **planted}, fp)
    JG.set_for_tests(False)
    try:
        with pytest.raises(ContaminationError, match="disabled"):
            assert_public_disjoint(pair()[0])
    finally:
        JG.set_for_tests(None)


def test_blind_verifier_cannot_see_authored_answer_or_pair_metadata():
    payload = blind_payload(pair()[0])
    assert set(payload) == {"state", "question"}
    assert "labels" not in payload and "distill" not in payload
    assert "solution" not in payload


def test_deterministic_numeric_and_business_date_audits():
    assert arithmetic("(17 - 8 + 3) / 4") == Fraction(3)
    state = "There were 17 binders; 8 left, and 3 returned. The start date is 2026-10-09."
    check = {"kind": "arithmetic", "source_quotes": ["17 binders; 8 left, and 3 returned"],
             "expression": "17-8+3", "expected": "12"}
    assert check_calculation(check, state)
    assert not check_calculation({**check, "expected": "13"}, state)
    assert not check_calculation({**check, "source_quotes": ["invented evidence"]}, state)
    dates = {"kind": "business_days", "source_quotes": ["The start date is 2026-10-09"],
             "start": "2026-10-09", "days": 2, "holidays": ["2026-10-12"], "expected": "2026-10-14"}
    assert check_calculation(dates, state)
    assert not check_calculation({**dates, "expected": "2026-10-13"}, state)
    for bad in ("__import__('os').system('false')", "2**100000", "[1][0]", "x+1", "True+1"):
        with pytest.raises(ValueError):
            arithmetic(bad)


def test_verifier_ambiguity_disagreement_and_incorrect_math_drop():
    row = pair()[0]
    verdict = {"answer_text": "yes", "ambiguous": False, "checks": []}
    assert accept_verdict(row, verdict) == (True, "accepted")
    assert not accept_verdict(row, {**verdict, "ambiguous": True})[0]
    assert not accept_verdict(row, {**verdict, "answer_text": "no"})[0]
    row["distill"]["family"] = "temporal_numeric"
    assert accept_verdict(row, verdict)[1] == "numeric_unchecked"


def test_teacher_requires_both_samples_and_whole_pairs():
    rows = pair()
    verdicts = {r["id"]: {"answer_text": "yes" if r["labels"]["q"] == "true" else "no", "ambiguous": False, "checks": []} for r in rows}
    teacher = {r["id"]: {"samples": [labeled(r)["teacher_probs"]] * 2} for r in rows}
    kept, report = capability_labels(rows, verdicts, teacher)
    assert len(kept) == 2
    assert report["verifier_agreement"] == {"comparable": 2, "same_decision": 2}
    teacher[rows[1]["id"]]["samples"] = [{"true": 0.9, "false": 0.1}, {"true": 0.0, "false": 1.0}]
    assert capability_labels(rows, verdicts, teacher)[0] == []


def test_training_union_contains_all_v0_and_no_dev_or_public_items():
    v0 = [labeled(r, False) for r in pair("old-train")]
    old_dev = [labeled(r, False) for r in pair("old-dev")]
    capability = [labeled(r) for g in ("new-a", "new-b", "new-c") for r in pair(g)]
    # Distinct contents are necessary in addition to distinct identifiers.
    for r in v0 + old_dev + capability:
        r["source"]["content_sha256"] = digest(r["id"])
    train, dev, report = assemble(v0, capability, old_dev, dev_groups=1)
    assert {r["id"] for r in v0} <= {r["id"] for r in train}
    assert not {r["group"] for r in train} & {r["group"] for r in dev}
    assert report["public_rows"] == 0
    mixed = mix(train, 2 / 3)
    assert {r["id"] for r in train} <= {r["id"] for r in mixed}
    planted = copy.deepcopy(v0)
    planted[0]["source"]["source_split"] = "public"
    with pytest.raises(ContaminationError):
        assemble(planted, capability, old_dev, dev_groups=1)
    planted[0]["source"]["source_split"] = "synthetic"
    planted[0]["source"]["source"] = "jevbench-public"
    with pytest.raises(ContaminationError):
        validate_source(planted[0])


def test_selection_uses_only_declared_synthetic_dev_metric():
    results = [{"name": c["name"], "dev": {"balanced_loss": loss, "balanced_accuracy": .8}} for c, loss in zip(CANDIDATES, (.2, .15, .3))]
    assert select_candidate(results)["name"] == CANDIDATES[1]["name"]
    with pytest.raises(ValueError):
        select_candidate(results[:2])
