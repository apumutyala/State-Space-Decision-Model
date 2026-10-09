"""Small allocation, admission, and byte-preservation tests; no tokenizer/model."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from data.c_multirecord_synth import canonical, make_record
from data.c_multirecord_subset import allocations, export_subset, stratum
from train.cohort import record_content_hash


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CMultirecordSubsetTests(unittest.TestCase):
    def source(self, root):
        directory = root / "source"
        directory.mkdir()
        records = [make_record(20261008, i * 228, revision=2) for i in range(6)]
        # Deliberately noncanonical record whitespace must survive the export.
        lines = [json.dumps(r, separators=(",", ": "), ensure_ascii=False).encode() + b"\n" for r in records]
        blobs = {"records": b"".join(lines),
                 "manifest": b"".join(canonical({"id": r["id"], "sha256": sha(raw[:-1]), "source": r["source"], "family": r["family"]}) + b"\n" for r, raw in zip(records, lines)),
                 "content_map": b"".join(canonical({"id": r["id"], "content_sha256": record_content_hash(r)}) + b"\n" for r in records)}
        names = {"records": "records.jsonl.gz", "manifest": "source.manifest.jsonl.gz", "content_map": "source.content.jsonl.gz"}
        for name, blob in blobs.items():
            (directory / names[name]).write_bytes(gzip.compress(blob, mtime=0))
        pointer = {"n_records": 6, "manifest": names["manifest"], "content_map": names["content_map"],
                   "excludes_cohort": "synthetic fixture; consumer guard remains required",
                   **{name + "_uncompressed_sha256": sha(blob) for name, blob in blobs.items()}}
        pointer_path = root / "source.pointer.json"
        pointer_path.write_bytes(canonical(pointer) + b"\n")
        summary = {"pointer": pointer, "pointer_sha256": sha(pointer_path.read_bytes()), "records": 6,
                   "label_method_verified_records": 6, "generator": {"version": 2},
                   "token_limits": {"max_tokens": 2048, "scope": "toy source receipt"}}
        summary_path = root / "source.summary.json"
        summary_path.write_bytes(canonical(summary) + b"\n")
        args = {"source": directory, "source_pointer": pointer_path,
                "source_pointer_sha256": sha(pointer_path.read_bytes()), "source_summary": summary_path,
                "source_summary_sha256": sha(summary_path.read_bytes()), "data_root": root, "count": 5, "seed": 73}
        return args, records, lines

    def test_allocation_has_two_or_three_and_style_balanced_seeded_extras(self):
        counts = {(2, style, pos): 4 for style in ("letters", "words") for pos in range(2)}
        result = allocations(counts, 10, 19)
        self.assertEqual(result, allocations(dict(reversed(list(counts.items()))), 10, 19))
        self.assertEqual(Counter(result.values()), {2: 2, 3: 2})
        self.assertEqual({style: sum(n for (_, s, _), n in result.items() if s == style)
                          for style in ("letters", "words")}, {"letters": 5, "words": 5})
        with self.assertRaises(ValueError):
            allocations({(2, "letters", 0): 1}, 2, 19)
        with self.assertRaises(ValueError):
            allocations(counts, 13, 19)

    def test_stratum_uses_canonical_choices_and_rejects_forged_position(self):
        record = make_record(20261008, 228, revision=2)
        self.assertEqual(stratum(record), (2, "letters", 1))
        forged = deepcopy(record)
        meta = next(m for m in forged["aug"]["question_roles"].values() if m["kind"] == "top_profile")
        meta["gold_position"] = 0
        with self.assertRaises(ValueError):
            stratum(forged)

    def test_export_is_deterministic_preserves_exact_source_bytes_and_sidecars(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, records, lines = self.source(root)
            expected = {stratum(r) for r in records}
            source_bytes = (args["source"] / "records.jsonl.gz").read_bytes()
            with patch("data.c_multirecord_subset.EXPECTED_STRATA", expected):
                first = export_subset(**args, out=root / "subset-a")
                second = export_subset(**args, out=root / "subset-b")
            a = gzip.decompress((root / "subset-a/train_set/records.jsonl.gz").read_bytes())
            b = gzip.decompress((root / "subset-b/train_set/records.jsonl.gz").read_bytes())
            self.assertEqual(a, b)
            selected = a.splitlines(keepends=True)
            self.assertEqual(len(selected), 5)
            self.assertTrue(all(raw in lines for raw in selected))
            self.assertEqual((args["source"] / "records.jsonl.gz").read_bytes(), source_bytes)
            self.assertEqual(first["stratum_count_histogram"], {2: 1, 3: 1})
            self.assertEqual(first["per_style_records"], {"letters": 5})
            self.assertEqual(sum(first["joint_score_labels"].values()), 5)
            self.assertEqual(sum(first["case_exact_labels"].values()), 5)
            pointer = first["pointer"]
            for name in ("records", "manifest", "content_map"):
                filename = "records.jsonl.gz" if name == "records" else pointer[name]
                blob = gzip.decompress((root / "subset-a/train_set" / filename).read_bytes())
                self.assertEqual(sha(blob), pointer[name + "_uncompressed_sha256"])
            manifest = [json.loads(line) for line in gzip.decompress((root / "subset-a/train_set" / pointer["manifest"]).read_bytes()).splitlines()]
            self.assertEqual([entry["sha256"] for entry in manifest], [sha(line[:-1]) for line in selected])
            with self.assertRaises(FileExistsError):
                export_subset(**args, out=root / "subset-a")

    def test_bad_pins_missing_strata_and_payload_overflow_refuse_before_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, records, _ = self.source(root)
            out = root / "subset"
            bad = dict(args, source_pointer_sha256="0" * 64)
            with self.assertRaises(ValueError):
                export_subset(**bad, out=out)
            with self.assertRaises(ValueError):
                export_subset(**args, out=out)  # full production domain absent
            with patch("data.c_multirecord_subset.EXPECTED_STRATA", {stratum(r) for r in records}), \
                 patch("data.c_multirecord_subset.PAYLOAD_LIMIT", 1), self.assertRaises(MemoryError):
                export_subset(**args, out=out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
