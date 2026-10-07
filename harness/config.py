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

# Default edition stays 0.2.1 so every existing receipt reproduces byte-for-byte. Pass
# `edition="0.3"` (or `--edition 0.3` on the CLI) for the 0.3 *public* index. See INDEX_0.3.md.
EDITION = "0.2.1"
EDITIONS = ("0.2.1", "0.3")

# Edition 0.3 (live on the board 2026-10-06). The kit's 0.3 code is still an OPEN PR (#74,
# head 62d2f51, not merged at 2026-10-06T17:40Z), so the 0.3 index spec is vendored from that
# head and pinned by hash; the installed kit (87d4650b) is only used for row-level scoring.
KIT_03_PR = 74
KIT_03_COMMIT = "62d2f51de34a2de64906345b6bc3e98e27ff55c7"
INDEX_03_SPEC_SHA256 = "158c38439c004a1bfb371dab02ec614046aabd3de08a02629ae89e7137d67167"
INDEX_03_SPEC = Path(__file__).resolve().parent / "editions_data" / "index-0.3.json"
GSM8K_03_RUN_ID_PREFIX = "v0.3-gsm8k:"  # editions.py@62d2f51: replaced={30: "v0.3-gsm8k:"}
# Rebuilt 0.3 GSM8K rows (kit PR #74 gsm8k_v3.build over the pinned GSM8K test parquet + our 0.2 rows);
# uncompressed sha256 pinned in editions.py@62d2f51 and reproduced locally byte-identically.
GSM8K_03_SHA256 = "759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a"
RETIRED = {"0.2.1": (), "0.3": (48,)}  # ForecastBench retired in 0.3
REPLACED = {"0.2.1": {}, "0.3": {30: GSM8K_03_RUN_ID_PREFIX}}  # GSM8K rows rebuilt in 0.3
# Board data for 0.3 (Space multimodalart/jev-decision-index @ 6ceab83, fetched 2026-10-06).
SPACE_03_COMMIT = "6ceab8392133b34f880d6f87f66ba068f068d2af"
BOARD_V03_SHA256 = "9f8a0a0327b800c33219042fbed6db597557819608c7254dc37fb183e13b01af"  # data/v03.json
BOARD_INDEX_03_SHA256 = "62b016ec9eb4f5bbd5bc67a2d1b0fad62372b5ed674264fae7dca72601e0d4dc"  # data/index.json (0.3 public)
METHODOLOGY_03_SHA256 = "d68c323f4d8d87d2efc4a1218751b5005629116e7ee2a858d53bac68f9418294"  # data/methodology.json


def edition_id(edition=None):
    e = str(edition or EDITION)
    e = {"v0.2.1": "0.2.1", "release-v2.1": "0.2.1", "v0.3": "0.3", "release-v3": "0.3"}.get(e, e)
    if e not in EDITIONS:
        raise ValueError(f"unsupported edition {edition!r}; choose from {EDITIONS}")
    return e
# Official uncompressed hashes of the two row files (editions.py at KIT_COMMIT).
ROWS_SHA256 = "b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5"
ADDED_SHA256 = "7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89"

# Untrusted data root (suite rows, source downloads). Override with SSDM_DATA.
DATA_ROOT = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))
SUITE_DIR = Path(os.environ.get("SSDM_DI_SUITE", str(DATA_ROOT / "suite-0.2")))
KIT_CLONE = Path(os.environ.get("SSDM_DI_KIT", str(DATA_ROOT / "decision-index")))
SPACE_DIR = DATA_ROOT / "di-space"
GSM8K_03_ROWS = Path(os.environ.get("SSDM_DI_GSM8K03", str(DATA_ROOT / "suite-0.3-gsm8k" / "gsm8k-rows.jsonl.gz")))

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
