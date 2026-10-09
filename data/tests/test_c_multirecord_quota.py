"""Small C-only quota proofs; no model, tokenizer, or external data."""
from collections import Counter
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from data.c_multirecord_compact import compact_record, load_pools
from data.c_multirecord_quota import export_quota, select_quota
from data.c_multirecord_subset import _sha, stratum
from data.c_multirecord_synth import make_record
from data.tests.test_c_multirecord_union import component, write_json


class CMultirecordQuotaTests(unittest.TestCase):
    def fixture(self, root):
        records = [make_record(19, i, revision=2) for i in (0, 228, 456, 76, 304, 1)]
        records.append(compact_record(make_record(19, 71, revision=2), load_pools()))
        directory, pointer, pointer_path, _, blobs = component(root, "source", records)
        # Noncanonical whitespace in BOTH sidecars must survive the selected export.
        for name in ("manifest", "content_map"):
            blobs[name] = b"".join(json.dumps(json.loads(raw), separators=(", ", ":  ")).encode() + b"\n"
                                   for raw in blobs[name].splitlines())
            (directory / pointer[name]).write_bytes(gzip.compress(blobs[name], mtime=0))
            pointer[name + "_uncompressed_sha256"] = _sha(blobs[name])
        pointer_path, pointer_pin = write_json(pointer_path, pointer)
        bounds = {"layout": "g1v2", "max_effective_tokens": {"v2f": 1646, "v3w": 2048}}
        proof = {"schema": "ssdm.c_multirecord_union_independent_verification.v1", "status": "verified",
                 "pointer_sha256": pointer_pin, "b_inputs_read": False, "protected_rows_read": False,
                 "refused": 0, "check_record_passed": len(records),
                 "counts": {name: len(records) for name in blobs},
                 "uncompressed_sha256": {name: _sha(blob) for name, blob in blobs.items()},
                 "all_k": sorted({stratum(record)[0] for record in records}), "token_bounds_verified": bounds}
        proof_path, proof_pin = write_json(root / "verification.json", proof)
        args = {"source": directory, "source_pointer": pointer_path, "source_pointer_sha256": pointer_pin,
                "verification": proof_path, "verification_sha256": proof_pin,
                "data_root": root, "count": 5, "seed": 73}
        return args, records, blobs, proof

    def test_selection_covers_groups_obeys_capacity_and_ignores_input_order(self):
        rows = [(f"a{i}", (2, "letters", i % 2)) for i in range(6)]
        rows += [(f"b{i}", (3, "words", i)) for i in range(2)] + [("c0", (77, "opaque", 76))]
        buckets = dict(rows)
        sizes = Counter(bucket[:2] for _, bucket in rows)
        for count in (3, 6, 9):
            with self.subTest(count=count):
                ids, quota = select_quota(rows, count, 19)
                self.assertEqual((ids, quota), select_quota(list(reversed(rows)), count, 19))
                self.assertEqual(len(ids), count)
                self.assertEqual(len(set(ids)), count)
                self.assertEqual(Counter(buckets[rid][:2] for rid in ids), quota)
                self.assertEqual(set(quota), set(sizes))
                self.assertTrue(all(1 <= q <= sizes[b] for b, q in quota.items()))
        self.assertEqual(set(select_quota(rows, 9, 19)[0]), set(buckets))

    def test_gold_positions_are_used_before_repeating_a_position(self):
        rows = [(f"p{p}-{i}", (5, "letters", p)) for p, n in ((0, 3), (2, 2), (4, 1)) for i in range(n)]
        selected, _ = select_quota(rows, 3, 11)
        self.assertEqual({dict(rows)[rid][2] for rid in selected}, {0, 2, 4})

    def test_invalid_selection_inputs_refuse(self):
        rows = [("a", (2, "letters", 0)), ("b", (77, "words", 1))]
        for count, seed in ((0, 1), (1, 1), (3, 1), (True, 1), (2, -1), (2, True)):
            with self.subTest(count=count, seed=seed), self.assertRaises(ValueError):
                select_quota(rows, count, seed)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_quota(rows + [("a", (2, "letters", 1))], 2, 1)
        with self.assertRaises(ValueError):
            select_quota([], 0, 1)

    def test_export_preserves_every_selected_record_and_sidecar_byte(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, records, blobs, proof = self.fixture(root)
            source_before = {path: path.read_bytes() for path in args["source"].iterdir()}
            first = export_quota(**args, out=root / "first")
            second = export_quota(**args, out=root / "second")
            self.assertEqual(first["selected_ids"], second["selected_ids"])
            self.assertEqual((first["records"], first["questions"]), (5, 35))
            self.assertEqual(first["token_bounds"], proof["token_bounds_verified"])
            self.assertTrue(first["record_and_sidecar_bytes_preserved"])
            selected = first["selected_ids"]
            for name, blob in blobs.items():
                source_lines = {json.loads(raw)["id"]: raw for raw in blob.splitlines(keepends=True)}
                expected = b"".join(source_lines[rid] for rid in selected)
                outputs = []
                for label, result in (("first", first), ("second", second)):
                    filename = "records.jsonl.gz" if name == "records" else result["pointer"][name]
                    compressed = (root / label / "train_set" / filename).read_bytes()
                    raw = gzip.decompress(compressed)
                    self.assertEqual(raw, expected)
                    self.assertEqual(_sha(raw), result["pointer"][name + "_uncompressed_sha256"])
                    self.assertEqual(_sha(compressed), result["compressed_sha256"][name])
                    outputs.append(compressed)
                self.assertEqual(*outputs)
            buckets = {record["id"]: stratum(record) for record in records}
            self.assertEqual({buckets[rid][:2] for rid in selected}, {bucket[:2] for bucket in buckets.values()})
            self.assertEqual(set(first["per_k_records"]), {2, 3, 73})
            self.assertEqual(set(first["prefix_baseline"]["k"]), {2})
            self.assertEqual({path: path.read_bytes() for path in args["source"].iterdir()}, source_before)
            pointer_raw = (root / "first/train_set/pointer.json").read_bytes()
            self.assertEqual(_sha(pointer_raw), first["pointer_sha256"])
            with self.assertRaises(FileExistsError):
                export_quota(**args, out=root / "first")

    def test_bad_pins_insufficient_quota_and_failed_proof_refuse_before_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _, _, proof = self.fixture(root)
            out = root / "out"
            for changed in ({"source_pointer_sha256": "0" * 64}, {"verification_sha256": "0" * 64}, {"count": 3}, {"count": 8}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    export_quota(**dict(args, **changed), out=out)
                self.assertFalse(out.exists())
            for field, value in (("status", "incomplete"), ("protected_rows_read", True), ("refused", 1),
                                 ("all_k", [2, 3])):
                _, pin = write_json(args["verification"], {**proof, field: value})
                with self.subTest(field=field), self.assertRaises(ValueError):
                    export_quota(**dict(args, verification_sha256=pin), out=out)
                self.assertFalse(out.exists())
            bad_bounds = {**proof["token_bounds_verified"], "max_effective_tokens": {"v2f": 1646, "v3w": 2049}}
            _, pin = write_json(args["verification"], {**proof, "token_bounds_verified": bad_bounds})
            with self.assertRaises(ValueError):
                export_quota(**dict(args, verification_sha256=pin), out=out)
            self.assertFalse(out.exists())

    def test_changed_source_record_bytes_refuse_before_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _, _, _ = self.fixture(root)
            path = args["source"] / "records.jsonl.gz"
            path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()).replace(b": ", b":  ", 1)))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                export_quota(**args, out=root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
