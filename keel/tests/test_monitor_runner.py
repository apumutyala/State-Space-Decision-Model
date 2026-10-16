"""Host-side refusal and budget behavior without any paid endpoint."""
import json
import time
from types import SimpleNamespace

import pytest

from keel.monitor.__main__ import main
from keel.monitor.run import Scorer, SpendLimit, adaptive_tiers, evaluate


def test_evaluation_refuses_under_target_data(tmp_path):
    (tmp_path / "data_report.json").write_text(json.dumps({"status": "data_target_not_met"}))
    with pytest.raises(ValueError, match="data target not met"):
        evaluate(tmp_path, tmp_path / "aggregates", ["stock"])


def test_cli_rejects_in_worktree_private_outputs(monkeypatch):
    from pathlib import Path
    import keel.monitor.__main__ as cli
    repo = Path(cli.__file__).resolve().parents[2]
    monkeypatch.setattr("sys.argv", ["monitor", "data", "--private-dir", str(repo / "forbidden-private")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_spend_deadline_prevents_new_call(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("paid call must not start")
    scorer = Scorer("stock", tmp_path, SimpleNamespace(score=SimpleNamespace(spawn=forbidden)), time.monotonic() - 1)
    with pytest.raises(SpendLimit):
        scorer.bounded_remote([])


def test_modal_timeout_cancels_container_not_just_client_wait(tmp_path):
    from modal.exception import TimeoutError as ModalTimeoutError
    events = []
    async def get(timeout):
        events.append(("get", timeout))
        raise ModalTimeoutError("client wait expired")
    async def cancel(terminate_containers=False):
        events.append(("cancel", terminate_containers))
    async def spawn(*args, **kwargs):
        return SimpleNamespace(get=SimpleNamespace(aio=get), cancel=SimpleNamespace(aio=cancel))
    scorer = Scorer("stock", tmp_path,
                    SimpleNamespace(score=SimpleNamespace(spawn=SimpleNamespace(aio=spawn))),
                    time.monotonic() + 100)
    with pytest.raises(ModalTimeoutError):
        scorer.bounded_remote([])
    assert events[-1] == ("cancel", True)
    assert len([event for event in events if event[0] == "get"]) == 3
    assert len([event for event in events if event[0] == "cancel"]) == 3
    assert 0 < events[0][1] <= 100


def test_priority_and_batch_size_are_enforced(tmp_path):
    assert adaptive_tiers("student")[0] == ("t2", None)
    assert adaptive_tiers("student") == adaptive_tiers("stock")
    assert adaptive_tiers("think") == adaptive_tiers("teacher") == []
    scorer = Scorer("stock", tmp_path, None, time.monotonic() + 100)
    with pytest.raises(ValueError, match="at most 256"):
        scorer.bounded_remote([{}] * 257)
