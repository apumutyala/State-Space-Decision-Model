"""The spend guard refuses unapproved or over-cap runs, accounts every run, and stops a run at its cap."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import time

import pytest

from keel.runtime import budget as B


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SSDM_BUDGET", str(tmp_path / "BUDGET.json"))
    monkeypatch.setenv("SSDM_PRIVATE", str(tmp_path / "private"))
    return tmp_path


def approve(env, usd, expires=None, approval_id="a1"):
    expires = expires or (dt.date.today() + dt.timedelta(days=7)).isoformat()
    (env / "BUDGET.json").write_text(json.dumps({"approval_id": approval_id, "approved_usd": usd,
                                                  "approved_by": "owner", "expires": expires}))


def ledger(env):
    path = env / "private" / "spend.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class FakeApp:
    def __init__(self):
        self.started = self.stopped = 0

    @contextlib.contextmanager
    def run(self):
        self.started += 1
        try:
            yield
        finally:
            self.stopped += 1


def test_no_approval_refuses_and_records_nothing(env):
    app = FakeApp()
    with pytest.raises(B.BudgetNotApproved, match="no owner-approved budget"):
        with B.approved_run(app, label="x", cap_usd=1.0):
            pass
    assert app.started == 0 and ledger(env) == []


def test_expired_approval_refuses(env):
    approve(env, 10, expires=(dt.date.today() - dt.timedelta(days=1)).isoformat())
    with pytest.raises(B.BudgetNotApproved, match="expired"):
        with B.approved_window(label="x", cap_usd=1.0):
            pass


def test_cap_must_fit_remaining_counting_unfinished_runs_at_full_cap(env):
    approve(env, 5.0)
    entries = [{"run_id": "r1", "approval_id": "a1", "event": "start", "cap_usd": 2.0},
               {"run_id": "r1", "approval_id": "a1", "event": "end", "usd_estimate": 1.25},
               {"run_id": "r2", "approval_id": "a1", "event": "start", "cap_usd": 3.0},
               {"run_id": "r9", "approval_id": "other", "event": "start", "cap_usd": 99.0}]
    (env / "private").mkdir()
    (env / "private" / "spend.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    assert B.spent("a1") == pytest.approx(4.25)  # finished at estimate + unfinished at cap; other approvals ignored
    with pytest.raises(B.BudgetNotApproved, match=r"\$0\.75 left"):
        with B.approved_window(label="too big", cap_usd=1.0):
            pass
    with B.approved_window(label="fits", cap_usd=0.75):
        pass


def test_approved_run_starts_and_stops_app_and_records_estimate(env):
    approve(env, 10.0)
    app = FakeApp()
    with B.approved_run(app, label="ok", cap_usd=2.0, gpus=2):
        time.sleep(0.05)
    assert (app.started, app.stopped) == (1, 1)
    start, end = ledger(env)
    assert start["event"] == "start" and start["cap_usd"] == 2.0 and end["status"] == "ok"
    assert end["usd_estimate"] == pytest.approx(end["wall_s"] / 3600 * 3.95 * 2, rel=1e-3)
    assert start["deadline_s"] == pytest.approx(2.0 / (3.95 * 2) * 3600)


def test_run_is_stopped_at_its_cap(env):
    approve(env, 10.0)
    app = FakeApp()
    cap = 0.2 / 3600 * 3.95  # a 0.2 s deadline on one H100
    with pytest.raises(B.BudgetExceeded):
        with B.approved_run(app, label="runaway", cap_usd=cap):
            time.sleep(5)
    assert app.stopped == 1
    end = ledger(env)[-1]
    assert end["status"] == "killed_at_cap" and end["wall_s"] < 2


def test_guard_blocks_modal_app_run_outside_an_approved_window(env, monkeypatch):
    import modal

    calls = []
    monkeypatch.setattr(B, "_ORIGINAL_RUN", None)
    monkeypatch.setattr(modal.App, "run", lambda self, *a, **k: calls.append(self) or contextlib.nullcontext())
    B.install_guard()
    app = modal.App("keel-budget-test")
    with pytest.raises(B.BudgetNotApproved, match="outside"):
        app.run()
    assert calls == []
    approve(env, 1.0)
    with B.approved_window(label="inside", cap_usd=0.5):
        app.run()
    assert calls == [app]
