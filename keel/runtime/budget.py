"""Owner-approved spend guard for every paid Modal run.

A Keel process may start paid work on Modal only inside ``approved_window`` (or ``approved_run``, which also starts the
app). A run starts only when all three hold:

1. the account owner has approved a budget in ``$SSDM_BUDGET`` (default ``~/ssdm/BUDGET.json``);
2. that approval has not expired;
3. the run's declared cap fits in what is left of the approval. Earlier finished runs count at their estimate;
   unfinished runs count at their full cap.

When the cap's GPU wall time is used up, ``BudgetExceeded`` is raised in the main thread. The app context exits and
Modal stops the app's containers. Every run is recorded in ``$SSDM_PRIVATE/spend.jsonl``: a start line reserving the
cap, then an end line with the estimate. The estimate is client wall time × GPUs × hourly rate. It excludes image
builds and is not an invoice.

Modal's workspace budget (Usage & Billing → Workspace budget) is the hard limit and only the owner can set it. This
guard is the process control in front of it.

**Only the account owner writes BUDGET.json.** Agents and scripts never create or edit it. Example:

    {"approval_id": "2026-10-16-a", "approved_usd": 20, "approved_by": "<owner>", "expires": "2026-10-23",
     "note": "resume student v1"}

``install_guard()`` runs when ``keel.runtime.modal_app`` is imported. After that, ``modal.App.run`` outside an approved
window raises ``BudgetNotApproved``.

    python -X utf8 -m keel.runtime.budget      # show the approval, spend and what is left
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import json
import os
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from keel.paths import private_root

GPU_HOURLY_USD = {"H100": 3.95}
_REQUIRED = ("approval_id", "approved_usd", "approved_by", "expires")
_ACTIVE: contextvars.ContextVar[str | None] = contextvars.ContextVar("keel_budget_run", default=None)
_ORIGINAL_RUN: Any = None


class BudgetNotApproved(RuntimeError):
    """No valid owner approval covers the requested run."""


class BudgetExceeded(RuntimeError):
    """A run used up its approved cap and is being stopped."""


def budget_path() -> Path:
    value = os.environ.get("SSDM_BUDGET")
    return Path(value).expanduser() if value else Path.home() / "ssdm" / "BUDGET.json"


def ledger_path() -> Path:
    return private_root() / "spend.jsonl"


def load_approval(today: dt.date | None = None) -> dict:
    path = budget_path()
    if not path.exists():
        raise BudgetNotApproved(f"no owner-approved budget at {path}; paid Modal runs are disabled. "
                                "Only the account owner may create it (format: keel/runtime/budget.py).")
    approval = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED if k not in approval]
    if missing:
        raise BudgetNotApproved(f"{path} is missing {missing}")
    if not float(approval["approved_usd"]) > 0:
        raise BudgetNotApproved(f"{path}: approved_usd must be positive")
    if dt.date.fromisoformat(str(approval["expires"])) < (today or dt.date.today()):
        raise BudgetNotApproved(f"approval {approval['approval_id']} expired on {approval['expires']}")
    return approval


def _entries() -> list[dict]:
    path = ledger_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def spent(approval_id: str) -> float:
    """USD charged to ``approval_id``: finished runs at their estimate, unfinished runs at their full cap."""
    runs: dict[str, dict] = {}
    for entry in _entries():
        if entry.get("approval_id") == approval_id:
            runs.setdefault(entry["run_id"], {})[entry["event"]] = entry
    return sum(float(r["end"]["usd_estimate"]) if "end" in r else float(r["start"]["cap_usd"])
               for r in runs.values() if "start" in r)


def _append(entry: dict) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def approved_window(*, label: str, cap_usd: float, gpus: int = 1, gpu: str = "H100") -> Iterator[str]:
    """Admit a paid run under the owner's approval, enforce its cap as a deadline, and record it. Yields the run id."""
    if threading.current_thread() is not threading.main_thread():
        raise BudgetNotApproved("paid runs must start in the main thread (the cap deadline uses SIGALRM)")
    if not (cap_usd > 0 and gpus >= 1 and gpu in GPU_HOURLY_USD):
        raise ValueError("cap_usd must be positive, gpus >= 1, gpu a known type")
    approval = load_approval()
    approved, used = float(approval["approved_usd"]), spent(approval["approval_id"])
    if used + cap_usd > approved + 1e-9:
        raise BudgetNotApproved(f"run {label!r} needs ${cap_usd:.2f}; approval {approval['approval_id']} has "
                                f"${max(0.0, approved - used):.2f} left of ${approved:.2f}")
    rate = GPU_HOURLY_USD[gpu] * gpus
    deadline_s = cap_usd / rate * 3600.0
    run_id = uuid.uuid4().hex[:12]
    base = {"run_id": run_id, "approval_id": approval["approval_id"], "label": label, "gpu": gpu, "gpus": gpus}
    _append({**base, "event": "start", "cap_usd": cap_usd, "deadline_s": deadline_s, "at": _now()})

    def expire(signum: int, frame: Any) -> None:
        raise BudgetExceeded(f"run {label!r} reached its ${cap_usd:.2f} cap after {deadline_s:.0f} s; stopping")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, deadline_s)
    token = _ACTIVE.set(run_id)
    start, status = time.monotonic(), "ok"
    try:
        yield run_id
    except BudgetExceeded:
        status = "killed_at_cap"
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        _ACTIVE.reset(token)
        wall = time.monotonic() - start
        _append({**base, "event": "end", "status": status, "wall_s": wall,
                 "usd_estimate": wall / 3600.0 * rate, "at": _now()})


@contextlib.contextmanager
def approved_run(app: Any, *, label: str, cap_usd: float, gpus: int = 1, gpu: str = "H100") -> Iterator[str]:
    """``approved_window`` around ``app.run()``: leaving the block (normally, on error or at the cap) stops the app."""
    with approved_window(label=label, cap_usd=cap_usd, gpus=gpus, gpu=gpu) as run_id, app.run():
        yield run_id


def install_guard() -> None:
    """Make ``modal.App.run`` refuse to start outside an approved window (idempotent)."""
    global _ORIGINAL_RUN
    import modal

    if _ORIGINAL_RUN is not None:
        return
    _ORIGINAL_RUN = modal.App.run

    def guarded_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _ACTIVE.get() is None:
            raise BudgetNotApproved("modal.App.run outside keel.runtime.budget.approved_window: paid runs need an "
                                    "owner-approved budget")
        return _ORIGINAL_RUN(self, *args, **kwargs)

    modal.App.run = guarded_run


def main() -> None:
    try:
        approval = load_approval()
    except BudgetNotApproved as exc:
        print(f"paid runs disabled: {exc}")
        return
    used = spent(approval["approval_id"])
    print(f"approval {approval['approval_id']} by {approval['approved_by']}, expires {approval['expires']}: "
          f"${used:.2f} used of ${float(approval['approved_usd']):.2f}, "
          f"${float(approval['approved_usd']) - used:.2f} left (ledger {ledger_path()})")


if __name__ == "__main__":
    main()
