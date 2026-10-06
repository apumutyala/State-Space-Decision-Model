"""Finalize validator for jobs/m2_parent_arm: the cache manifest has both stages, each with a units identity, a parent
identity and a file hash (``jobs/common/finalize.py`` writes the PASS marker only when this returns True)."""
from __future__ import annotations

from typing import Any


def valid_cache(d: Any) -> bool:
    from train.m2_gates import ARM_SCHEMA, STAGES

    try:
        st = d["stages"]
        return d.get("schema") == ARM_SCHEMA and set(st) == set(STAGES) and all(
            len(e["sha256"]) == 64 and e["units_identity"]["stage"] == s and e["arm"]["parent"]["parent"]
            and e["units_identity"]["cohort_sha256"] for s, e in st.items())
    except Exception:  # noqa: BLE001
        return False
