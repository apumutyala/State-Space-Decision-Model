"""Narrow public text projection for the L0 semantic planner.

The caller supplies trusted host authority and evidence. This callable does not
provide network authentication or a timing-side-channel guarantee.
"""

from __future__ import annotations

from typing import Any

from .kernel import check_program
from .planner import plan_checked_answer
from .realize import render_checked


def public_answer(task_input: str, host: dict[str, Any], catalog: dict[str, Any],
                  provider: Any, *, model: str, parent_revision: str,
                  expected_served_config: dict[str, Any]) -> dict[str, str]:
    """Return freshly released text or one fixed, nonrevealing failure shape."""
    try:
        if not isinstance(host, dict) or host.get("audience") != "public":
            return {"status": "unavailable"}
        planned = plan_checked_answer(
            task_input, host, catalog, provider, model=model,
            parent_revision=parent_revision, expected_served_config=expected_served_config,
        )
        if planned.status != "accepted" or planned.program is None:
            return {"status": "unavailable"}
        checked = check_program(planned.program, host, catalog)
        if checked.status != "accepted":
            return {"status": "unavailable"}
        text = render_checked(checked, host, catalog)
        if not isinstance(text, str) or not text:
            return {"status": "unavailable"}
        return {"status": "ok", "text": text}
    except Exception:
        return {"status": "unavailable"}
