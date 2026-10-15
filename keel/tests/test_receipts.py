"""Real JSONL tampering, protected-tip boundaries and prompt replay tests."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json

import pytest

from keel.protocol import Answer, DecisionRequest, DecisionResult, OperatorPolicy, Question, Segment, render
from keel.protocol.receipts import (GENESIS, LogCheckpoint, PromptUse, Receipt, ReceiptIntegrityError,
                                    ReceiptLog, replay_check)
from keel.protocol.render import RENDER_VERSION, TRUST_POLICY_VERSION
from keel.protocol.types import sha256_json


def example():
    request = DecisionRequest("r", OperatorPolicy("p", "Only route.", authorized_actions=("route",)),
                              (Segment("u", {"text": "a\u2028b\u0085c"}, "untrusted", "tool_output", "search"),
                               Segment("t", "host note", "trusted", "operator_note", "operator")),
                              (Question("q", "choice", "Route?", {"a": "A", "b": "B"}),))
    result = DecisionResult("r", {"q": Answer("q", {"a": 0.7, "b": 0.3}, "a")}, True)
    return request, result


def capture(request=None, result=None, **kwargs):
    default_request, default_result = example()
    return Receipt.capture(request or default_request, result or default_result, model_id="judge",
                           model_revision="pinned-revision", gate_version="gate/1", calibration_id="cal-12",
                           outcome="act", reasons=("authorized_dispatch",), **kwargs)


def test_receipt_records_required_fields_and_snapshots_mutable_values(tmp_path):
    request, result = example()
    receipt = capture(request, result, attack_tag="role_forgery\u2028unicode\u0085separators")
    result.answers["q"].probs["a"] = 0.0
    assert receipt.answers["q"]["probs"] == {"a": 0.7, "b": 0.3}
    assert receipt.request_fingerprint == request.fingerprint()
    assert receipt.trust_map["u"] == {"trust": "untrusted", "kind": "tool_output", "source": "search",
                                        "content_sha256": sha256_json(request.segments[0].content), "parent_ids": []}
    assert receipt.render_version == RENDER_VERSION and receipt.trust_policy_version == TRUST_POLICY_VERSION
    assert receipt.model_id == "judge" and receipt.model_revision == "pinned-revision"
    assert receipt.gate_version == "gate/1" and receipt.calibration_id == "cal-12"
    assert receipt.attack_tag == "role_forgery\u2028unicode\u0085separators" and receipt.timestamp.endswith("+00:00")
    log = ReceiptLog(tmp_path / "audit.jsonl")
    sealed = log.append(receipt)
    assert sealed.prev_hash == GENESIS and sealed.receipt_id == sealed.digest()
    assert log.verify() == LogCheckpoint(1, sealed.receipt_id)
    assert replay_check(sealed, request)
    assert ReceiptLog(tmp_path / "audit.jsonl").receipts() == (sealed,)


@pytest.mark.parametrize("tamper", ["edit", "first", "middle", "tail", "all", "reorder", "space", "newline", "rewrite"])
def test_all_log_edits_deletions_and_reordering_detected(tmp_path, tamper):
    path = tmp_path / "receipts.jsonl"
    log = ReceiptLog(path)
    for _ in range(3):
        log.append(capture())
    lines = path.read_text().split("\n")[:-1]
    if tamper == "edit":
        lines[1] = lines[1].replace('"model_id":"judge"', '"model_id":"attacker"')
    elif tamper in ("first", "middle", "tail"):
        lines.pop({"first": 0, "middle": 1, "tail": 2}[tamper])
    elif tamper == "all":
        lines = []
    elif tamper == "reorder":
        lines.reverse()
    elif tamper == "space":
        lines[0] += " "
    elif tamper == "rewrite":
        previous = GENESIS
        rewritten = []
        for line in lines:
            receipt = replace(Receipt.from_json(json.loads(line)), prev_hash=previous, model_id="rewritten", receipt_id="")
            receipt = replace(receipt, receipt_id=receipt.digest())
            previous = receipt.receipt_id
            rewritten.append(json.dumps(receipt.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        lines = rewritten
    path.write_text("\n".join(lines) + ("\n" if lines and tamper != "newline" else ""))
    with pytest.raises(ReceiptIntegrityError):
        log.verify()
    with pytest.raises(ReceiptIntegrityError):
        log.append(capture())
    with pytest.raises(ReceiptIntegrityError):
        ReceiptLog(path)


def test_checkpoint_is_required_and_external_anchor_detects_replaced_tip(tmp_path):
    path = tmp_path / "receipts.jsonl"
    log = ReceiptLog(path)
    log.append(capture())
    anchor = log.verify()
    log.append(capture())
    checkpoint = path.with_name(path.name + ".checkpoint")
    path.write_text("")
    checkpoint.write_text(json.dumps({"count": 0, "head_hash": GENESIS}))
    # Rewriting both files defeats an unanchored chain: explicitly demonstrate scope.
    assert log.verify() == LogCheckpoint()
    with pytest.raises(ReceiptIntegrityError):
        log.verify(expected=anchor)
    checkpoint.unlink()
    with pytest.raises(ReceiptIntegrityError):
        ReceiptLog(path)


def test_missing_log_is_not_silently_recreated(tmp_path):
    path = tmp_path / "receipts.jsonl"
    log = ReceiptLog(path)
    log.append(capture())
    path.unlink()
    with pytest.raises(ReceiptIntegrityError):
        ReceiptLog(path)


def test_concurrent_independent_writers_preserve_chain(tmp_path):
    path = tmp_path / "receipts.jsonl"
    logs = [ReceiptLog(path) for _ in range(4)]
    receipt = capture()
    with ThreadPoolExecutor(max_workers=4) as pool:
        written = list(pool.map(lambda index: logs[index % 4].append(receipt), range(16)))
    assert logs[0].verify().count == 16
    assert len({r.receipt_id for r in written}) == 16


@pytest.mark.parametrize("mode,canonical", [("trust", True), ("trust", False), ("base", True), ("base", False)])
def test_replay_every_question_and_actual_thinking_prompt(mode, canonical):
    request, result = example()
    request = replace(request, questions=request.questions + (Question("yesno", "noul", "True?"),))
    result.answers["yesno"] = Answer("yesno", {"true": 0.8, "false": 0.2}, "yes")
    prompts = []
    for q in request.questions:
        rendered = render(request, q.qid, mode=mode, think=True, canonical=canonical)
        prompts.append(PromptUse.from_rendered(q.qid, rendered))
        for truncated in (False, True):
            prompts.append(PromptUse.from_rendered(q.qid, rendered, reasoning="Reasoned evidence.  ", truncated=truncated))
    receipt = ReceiptLog().append(capture(request, result, prompts=prompts))
    assert len(receipt.prompts) == 6 and replay_check(receipt, request)
    assert receipt.prompts[0].sha256 == hashlib.sha256(prompts[0].replay(request).encode()).hexdigest()
    bad = replace(receipt, prompts=(replace(receipt.prompts[0], sha256="0" * 64), *receipt.prompts[1:]))
    assert not replay_check(bad, request)
    changed = replace(request, segments=(replace(request.segments[0], content="different evidence"), request.segments[1]))
    assert not replay_check(receipt, changed)
    assert not replay_check(replace(receipt, render_version="future"), request)
    assert not replay_check(replace(receipt, trust_policy_version="future"), request)
    assert not replay_check(replace(receipt, trust_map={}), request)


def test_receipt_capture_rejects_incomplete_or_forged_prompt_manifest():
    request, result = example()
    with pytest.raises(ValueError, match="manifest"):
        capture(request, result, prompts=[])
    wrong = PromptUse.from_rendered("q", replace(render(request), prompt="not the renderer's output"))
    with pytest.raises(ValueError, match="manifest"):
        capture(request, result, prompts=[wrong])
    with pytest.raises(ValueError, match="unsupported"):
        PromptUse.from_rendered("q", replace(render(request), version="future"))
    with pytest.raises(ValueError):
        ReceiptLog().append(replace(capture(), timestamp="2026-01-01T01:00:00+01:00"))
    closed = PromptUse.from_rendered("q", render(request, think=True), reasoning="Evidence")
    with pytest.raises(ValueError, match="manifest"):
        capture(request, result, prompts=[closed])


def test_nonfinite_values_never_enter_jsonl():
    request, result = example()
    result.answers["q"].probs["a"] = float("nan")
    with pytest.raises(ValueError):
        capture(request, result)
