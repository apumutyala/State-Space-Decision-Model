"""Paths and pins. Large data and every download live outside git under the untrusted data root."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))
# Snapshot of the official 0.2.1 row files (copied from suite-0.2 after hash check; see dedup.py).
SUITE_DIR = Path(os.environ.get("SSDM_M2_SUITE", str(DATA_ROOT / "m2" / "suite-snapshot")))
M2_ROOT = DATA_ROOT / "m2"
SOURCES_DIR = M2_ROOT / "sources"          # one sub-directory per source, raw downloads only
FINGERPRINT = M2_ROOT / "index-fp-v2.npz"  # built from the full suite; sha256 recorded in the receipt
BUILD_DIR = M2_ROOT / "build"              # materialised training records (never committed)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

SEED = 20261006
