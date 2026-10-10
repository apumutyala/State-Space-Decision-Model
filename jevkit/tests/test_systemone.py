"""/v1/systemone adapter over decide.Decider (oracle backend), a kit-style engine and a request callable."""

import pytest

from decide import Decider
from decide.backends import OracleBackend
from jevkit import systemone as S1
from jevkit.answers import parse_answer
from jevkit.items import Item

from ._fixtures import choice, noul, score


class FakeUnsupported(ValueError):
    pass


class FakeKitEngine:
    """Decision Index kit Engine protocol: engine(state, questions) -> (response, raw)."""

    name = "fake-kit"

    def __init__(self, fail=False):
        self.fail = fail
        self.seen = []

    def warmup(self):
        pass

    def __call__(self, state, questions):
        self.seen.append(questions)
        if self.fail:
            raise FakeUnsupported("too long")
        ans = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                ans[k] = {"type": "choice", "probabilities": {x: 1 / len(keys) for x in keys}}
            elif q["type"] == "noul":
                ans[k] = {"type": "noul", "noul": 0.7}
            else:
                n = len(q["criteria"])
                ans[k] = {"type": "score", "probabilities": {str(i): 1 / n for i in range(n)}}
        return {"model": "fake", "answers": ans, "usage": {"input_tokens": 11}}, {"raw": True}


@pytest.mark.parametrize("item", [choice("c", k=3, gold=2), noul("n", "yes"), score("s", k=4, gold=2)])
def test_decider_path_gives_valid_jevbench_answers(item):
    gold = {"choice": "o2", "noul": "true", "score": "2"}[item.type]
    ad = S1.SystemOneAdapter(Decider(OracleBackend(labels={"q": gold}, margin=8.0)))
    assert ad.kind == "decider"
    rec = ad.answer_item(item)
    assert rec["status"] == "ok" and rec["usage"]["input_tokens"] > 0 and rec["latency_s"] >= 0
    p = parse_answer(item, rec["answer"])
    assert p.valid and p.has_probs
    assert p.predicted == ("yes" if item.type == "noul" else str(item.expected))
    if item.type == "score":
        assert set(rec["answer"]) >= {"score", "legend", "confidence", "probabilities"}


def test_neutral_question_id_never_leaks_item_id():
    eng = FakeKitEngine()
    ad = S1.SystemOneAdapter(eng)
    assert ad.kind == "engine"
    ad.answer_item(choice("hard-opus-a-long_policy-01"))
    assert list(eng.seen[0]) == [S1.NEUTRAL_QID]


def test_non_ok_status_is_a_record_not_an_answer():
    ad = S1.SystemOneAdapter(Decider(OracleBackend(fail_with=("too_long", "nope"))))
    rec = ad.answer_item(choice("c"))
    assert rec["status"] == "unsupported" and rec["answer"] is None
    ad = S1.SystemOneAdapter(FakeKitEngine(fail=True), unsupported_exc=(FakeUnsupported,))
    assert ad.answer_item(noul("n"))["status"] == "unsupported"


def test_request_callable_and_normalisation():
    def endpoint(req):
        q = req["questions"]["q"]
        return {"model": "hosted", "answers": {"q": {"probabilities": {k: 0.5 for k in q["criteria"]}}},
                "usage": {"input_tokens": 5, "output_tokens": 2}}

    ad = S1.SystemOneAdapter(endpoint, kind="request")
    rec = ad.answer_item(choice("c", k=2))
    assert rec["answer"]["type"] == "choice" and rec["answer"]["choice"] == "o0"
    assert rec["answer"]["confidence"] == pytest.approx(0.0) and rec["usage"]["output_tokens"] == 2


def test_crash_is_recorded_as_error():
    def boom(req):
        raise RuntimeError("x")

    rec = S1.SystemOneAdapter(boom, kind="request").answer_item(noul("n"))
    assert rec["status"] == "error" and rec["answer"] is None


def test_decision_index_round_trip():
    di = {"id": "r1", "family": "ragtruth", "split": "test", "state": {"prompt": "p", "response": "r"},
          "questions": {"halluc": {"type": "noul", "instructions": "Hallucinated?"},
                        "pick": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}},
                        "skip": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}},
          "expected": {"halluc": True, "pick": "b", "skip": None}, "_evaluation": {"run_id": "run-9"}}
    its = S1.di_row_to_items(di, tier="judge", split="private")
    assert [i.id for i in its] == ["run-9::halluc", "run-9::pick"]
    assert its[0].expected == "yes" and its[0].labels == ["no", "yes"] and its[1].tier == "judge"
    back = S1.item_to_di_row(its[0])
    assert back["expected"] == {"q": True} and back["_evaluation"]["run_id"] == "run-9::halluc"
    res = {"run_id": "run-9::halluc", "status": "ok", "total_wall_ms": 120.0,
           "response": {"model": "m", "answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 3}}}
    pred = S1.di_result_to_pred(res)
    assert pred["latency_s"] == pytest.approx(0.12) and parse_answer(its[0], pred["answer"]).predicted == "yes"


def test_preds_to_answers_both_layouts():
    a = S1.preds_to_answers([{"id": "x", "answer": {"noul": 1.0}, "status": "ok"},
                             {"id": "y", "answer": {"noul": 1.0}, "status": "error"},
                             {"answers": {"z": {"noul": 0.0}}}])
    assert a == {"x": {"noul": 1.0}, "y": None, "z": {"noul": 0.0}}
    with pytest.raises(ValueError):
        S1.preds_to_answers([{"foo": 1}])


def test_run_items_serial():
    ad = S1.SystemOneAdapter(FakeKitEngine())
    recs = S1.run_items(ad, [choice("a"), noul("b"), score("c")])
    assert [r["id"] for r in recs] == ["a", "b", "c"] and all(r["status"] == "ok" for r in recs)
    assert isinstance(recs[0], dict) and isinstance(Item, type)
