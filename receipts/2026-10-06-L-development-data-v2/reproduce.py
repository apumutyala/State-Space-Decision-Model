"""Recreate only the two iterative synthetic development samples in this receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from language_growth.development_data import development_manifest, generate_development_tasks

OUT = Path(__file__).resolve().parent
for split in ("development_train", "development_validation"):
    tasks = generate_development_tasks(17, 30, choice_format="mixed", split=split)
    rows = "".join(json.dumps(task, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False) + "\n" for task in tasks).encode("ascii")
    manifest = development_manifest(tasks)
    assert hashlib.sha256(rows).hexdigest() == manifest["sample_sha256"]
    (OUT / f"{split}.jsonl").write_bytes(rows)
    (OUT / f"{split}-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
