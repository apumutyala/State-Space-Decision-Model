"""Rebind a byte-verified HF snapshot location to its qualified metadata.

Only the cache root may differ. Serving revision pins remain byte-for-byte
identical after restoring the saved nonfunctional name_or_path metadata.
"""
from __future__ import annotations

import json
import re

_HF_SNAPSHOT = re.compile(
    r"(?:^|[\\/])models--([^\\/]+?)--([^\\/]+)[\\/]snapshots[\\/]([0-9a-f]{40})[\\/]?\Z")
_WEIGHT_FIELDS = {"weights_digest", "weights_digest_mode"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hub_snapshot(value):
    if (not isinstance(value, str) or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or "://" in value or any(part in (".", "..") for part in re.split(r"[\\/]", value))
            or not (value.startswith("/") or value.startswith("\\\\") or re.match(r"[A-Za-z]:[\\/]", value))):
        return None
    match = _HF_SNAPSHOT.search(value)
    return (match[1], match[2], match[3]) if match else None


def bind_qualified_snapshot_location(config, actual: dict, expected: dict, *, live_config_identity) -> bool:
    """Permit only identical pinned HF repo/revision content in another cache root.

    The caller verifies the complete local snapshot inventory before loading
    and supplies the live full backbone identity, including weight digest.
    Every field except name_or_path must match. Arbitrary local directories,
    changed revisions and different config/weight/runtime identities refuse.
    Rebinding happens before the backend computes/caches its served revision.
    """
    if (not isinstance(actual, dict) or not isinstance(expected, dict)
            or set(actual) != set(expected) or "name_or_path" not in actual):
        raise ValueError("reconstructed backbone identity key set mismatch")
    if _canonical(actual) == _canonical(expected):
        return False
    without_location = lambda value: {key: item for key, item in value.items() if key != "name_or_path"}
    if _canonical(without_location(actual)) != _canonical(without_location(expected)):
        raise ValueError("reconstructed backbone identity differs beyond snapshot cache location")
    actual_snapshot, expected_snapshot = _hub_snapshot(actual["name_or_path"]), _hub_snapshot(expected["name_or_path"])
    if actual_snapshot is None or expected_snapshot is None or actual_snapshot != expected_snapshot:
        raise ValueError("snapshot location must name the same pinned HF repository and revision")
    if expected.get("commit_hash") not in (None, expected_snapshot[2]):
        raise ValueError("snapshot path revision disagrees with qualified commit hash")
    if config is None or not hasattr(config, "_name_or_path"):
        raise ValueError("backbone config lacks standard HF snapshot location metadata")
    live_location = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if live_location != actual["name_or_path"]:
        raise ValueError("live snapshot location changed before qualification binding")
    original = config._name_or_path
    config._name_or_path = expected["name_or_path"]
    try:
        live = live_config_identity()
        expected_config = {key: value for key, value in expected.items() if key not in _WEIGHT_FIELDS}
        if _canonical(live) != _canonical(expected_config):
            raise ValueError("snapshot metadata binding changed the qualified backbone configuration")
    except Exception:
        config._name_or_path = original
        raise
    return True
