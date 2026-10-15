import json

import pytest
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime

from keel.protocol.receipts import ReceiptIntegrityError
from keel.tests.test_workflow_gate import FakeJudge, assistant, make_gate
from keel.workflows.audit import verify_run


def test_run_audit_verifies_chain_episode_links_and_saved_requests(tmp_path):
    gate = make_gate(tmp_path, FakeJudge({"allow": 0, "hold": 1}))
    calls = [FunctionCall(function="send_message", args={"body": str(i)}, id=str(i)) for i in range(2)]
    gate.query("Read.", FunctionsRuntime(), messages=[assistant(*calls)])
    checkpoint = gate.kernel.receipt_log.verify()
    row = {"receipt_ids": [outcome.receipt_id for outcome in gate.outcomes],
           "receipt_chain_verified_count": checkpoint.count}
    (tmp_path / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    report = verify_run(tmp_path, sample_size=20, seed=7, expected=checkpoint)
    assert report["receipts_verified"] == report["receipts_replayed"] == 2
    assert report["episode_checkpoints_verified"] == 1
    assert report["sample_indices"] == [0, 1]
    request = gate.requests[0]
    path = gate.request_dir / f"{request.fingerprint()}.json"
    tampered = json.loads(path.read_text())
    tampered["segments"][0]["content"] = "Forged trusted task"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ReceiptIntegrityError, match="replay"):
        verify_run(tmp_path, sample_size=20)


def test_run_audit_rejects_unlinked_receipt(tmp_path):
    gate = make_gate(tmp_path, FakeJudge({"allow": 0, "hold": 1}))
    gate.query("Read.", FunctionsRuntime(), messages=[assistant(FunctionCall(function="send_message", args={}, id="x"))])
    (tmp_path / "episodes.jsonl").write_text(json.dumps({"receipt_ids": []}) + "\n")
    with pytest.raises(ReceiptIntegrityError, match="episode dispatches"):
        verify_run(tmp_path)


def test_run_audit_does_not_create_missing_log(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_run(tmp_path)
    assert not (tmp_path / "receipts.jsonl").exists()
