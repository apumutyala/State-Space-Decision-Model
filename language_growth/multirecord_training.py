"""Export the pinned L1 development-train fixture; data plumbing only.

No model training or L1 capability is demonstrated. Strong comparator,
decision retention, adversarial soundness and causal intervention remain gates.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from .multirecord_data import MULTIWORLD_TRAIN_SPLIT
from .multirecord_eval import MULTIWORLD_FIXTURE_DIR, _canonical
from .training_records import L1_GENERATOR_SOURCE, L1_TRAINING_ALLOWLIST, export_training_records


def load_multirecord_training(*, split=MULTIWORLD_TRAIN_SPLIT, fixture_dir=MULTIWORLD_FIXTURE_DIR):
    """Relocation is allowed; replacing either pinned source file is not."""
    if split not in L1_TRAINING_ALLOWLIST:
        raise ValueError("only the pinned L1 development-train cohort is supported")
    data = (Path(fixture_dir) / f"{split}.jsonl").read_bytes()
    manifest_bytes = (Path(fixture_dir) / f"{split}-manifest.json").read_bytes()
    pins = L1_TRAINING_ALLOWLIST[split]
    if (hashlib.sha256(data).hexdigest() != pins["sample_sha256"]
            or hashlib.sha256(manifest_bytes).hexdigest() != pins["manifest_sha256"]):
        raise ValueError("L1 training cohort or manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes)
    records = export_training_records([json.loads(line) for line in data.splitlines()],
                                      split=split, l1_manifest=manifest)
    return records, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args()
    records, source_manifest = load_multirecord_training()
    rows = b"".join((_canonical(row) + "\n").encode("ascii") for row in records)
    manifest = {"schema": "ssdm.l1_training_export.v1", "record_count": len(records),
                "training_records_sha256": hashlib.sha256(rows).hexdigest(),
                "source_manifest": source_manifest, **L1_TRAINING_ALLOWLIST[MULTIWORLD_TRAIN_SPLIT],
                **L1_GENERATOR_SOURCE,
                "scope": __doc__.strip()}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Publish the pair only after both writes succeed. A failed write leaves
    # no final output and TemporaryDirectory removes the incomplete staging.
    with tempfile.TemporaryDirectory(prefix=f".{args.out.name}-", dir=args.out.parent) as directory:
        staged = Path(directory)
        (staged / "training_records.jsonl").write_bytes(rows)
        (staged / "export-manifest.json").write_bytes((_canonical(manifest) + "\n").encode("ascii"))
        staged.rename(args.out)


if __name__ == "__main__":
    main()
