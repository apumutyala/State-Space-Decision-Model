"""L-14: byte-exact evidence must survive Git's text conversion rules."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "receipts/2026-10-07-L0-operation-only-cache-run1"


def test_operation_cache_receipt_artifacts_match_the_committed_hash_manifest():
    manifest = json.loads((RECEIPT / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert manifest
    for name, expected in manifest.items():
        path = RECEIPT / name
        assert path.is_file(), name
        packed = path.read_bytes()
        if name.endswith(".json.gz"):
            assert hashlib.sha256(packed).hexdigest() == expected["compressed_sha256"], name
            payload = gzip.decompress(packed)
            assert hashlib.sha256(payload).hexdigest() == expected["uncompressed_sha256"], name
            json.loads(payload)
        else:
            assert hashlib.sha256(packed).hexdigest() == expected["sha256"], name
