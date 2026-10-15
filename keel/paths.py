"""Workstation locations kept outside git.

``SSDM_PRIVATE`` (default ``~/ssdm/private``) holds everything the data rules keep out of the repository:
eval-only per-item outputs, caches, search traces and private training data. Public JevBench items are located by
``SSDM_DATA`` (``jevkit.items``).
"""
from __future__ import annotations

import os
from pathlib import Path


def private_root() -> Path:
    """``$SSDM_PRIVATE`` when set, else ``~/ssdm/private``."""
    value = os.environ.get("SSDM_PRIVATE")
    return Path(value).expanduser() if value else Path.home() / "ssdm" / "private"
