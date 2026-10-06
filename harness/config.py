"""Pinned identities and local paths. Large data lives outside git."""

import os
from pathlib import Path

# Official kit (MIT), installed into the worktree venv with `pip install --no-deps`.
KIT_REPO = "https://github.com/apolinario/decision-index"
KIT_COMMIT = "87d4650b42b377c0291a89c1f1a879f9b31082bf"  # "Version 0.2.1", 2026-09-27
SPACE_REPO = "multimodalart/jev-decision-index"
SPACE_COMMIT = "cdbd1cab3b6eb1811ebd82d2686ad563a4d0fc0d"  # methodology-v0.2.1.json, index-v0.2.1.json
METHODOLOGY_SHA256 = "3724d15e6698d9a5c39fd2fc9ab70c5cc94a03aeea8fdefe2b8b30ec72420ce8"
BOARD_INDEX_SHA256 = "37526dd3f3d8f06e37722ef749344a7c877ed31260bad37de2e348cca1220d50"

EDITION = "0.2.1"
# Official uncompressed hashes of the two row files (editions.py at KIT_COMMIT).
ROWS_SHA256 = "b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5"
ADDED_SHA256 = "7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89"

# Untrusted data root (suite rows, source downloads). Override with SSDM_DATA.
DATA_ROOT = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))
SUITE_DIR = Path(os.environ.get("SSDM_DI_SUITE", str(DATA_ROOT / "suite-0.2")))
KIT_CLONE = Path(os.environ.get("SSDM_DI_KIT", str(DATA_ROOT / "decision-index")))
SPACE_DIR = DATA_ROOT / "di-space"

# Slices: manifests are committed (IDs + hashes only); materialised rows stay outside git.
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "harness" / "manifests"
SLICE_ROWS_DIR = DATA_ROOT / "slices"

SLICE_SEED = 20261006

# Request-length bins on the kit's own proxy (`_evaluation.proxy_tokens` = len(json payload) // 4).
LENGTH_BIN_EDGES = (0, 128, 512, 2048, 8192)
LENGTH_BIN_LABELS = ("<128", "128-511", "512-2047", "2048-8191", ">=8192")


def length_bin(proxy_tokens):
    label = LENGTH_BIN_LABELS[0]
    for edge, name in zip(LENGTH_BIN_EDGES, LENGTH_BIN_LABELS):
        if proxy_tokens >= edge:
            label = name
    return label
