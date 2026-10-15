"""Host-side refusal and budget behavior without any paid endpoint."""
import json
import time
from types import SimpleNamespace

import pytest

from keel.monitor.__main__ import main
from keel.monitor.run import Scorer, SpendLimit, evaluate


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
    scorer = Scorer("stock", tmp_path, SimpleNamespace(score=SimpleNamespace(spawn=forbidden)), time.time() - 1)
    with pytest.raises(SpendLimit):
        scorer.bounded_remote([])


def test_modal_timeout_cancels_container_not_just_client_wait(tmp_path):
    from modal.exception import TimeoutError as ModalTimeoutError
    events = []
    class Call:
        def get(self, timeout):
            events.append(("get", timeout))
            raise ModalTimeoutError("client wait expired")
        def cancel(self, terminate_containers=False):
            events.append(("cancel", terminate_containers))
    scorer = Scorer("stock", tmp_path, SimpleNamespace(score=SimpleNamespace(spawn=lambda *a, **k: Call())),
                    time.time() + 100)
    with pytest.raises(SpendLimit):
        scorer.bounded_remote([])
    assert events[-1] == ("cancel", True)
    assert 0 < events[0][1] <= 100
