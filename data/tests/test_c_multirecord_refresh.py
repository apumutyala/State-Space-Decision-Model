"""Tiny refresh admission, exact retained bytes, quota, and duplicate regressions."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from data.c_multirecord_refresh import FAMILY, NAMES, V1, V2, c_only, refresh
from data.c_multirecord_subset import _rank, _sha
from data.c_multirecord_synth import canonical, make_record
from train.cohort import record_content_hash


def write_json(path, value):
    path.write_bytes(canonical(value) + b"\n")
    return str(path), _sha(path.read_bytes())


def package(root, name, records):
    source = root / name
    source.mkdir()
    lines = [json.dumps(r, separators=(",", ": ")).encode() + b"\n" for r in records]
    blobs = {"records": b"".join(lines),
             "manifest": b"".join(canonical({"id": r["id"], "sha256": _sha(line[:-1]), "source": r["source"], "family": r["family"]}) + b"\n" for r, line in zip(records, lines)),
             "content_map": b"".join(canonical({"id": r["id"], "content_sha256": record_content_hash(r)}) + b"\n" for r in records)}
    names = {"records": "records.jsonl.gz", "manifest": name + ".manifest.jsonl.gz", "content_map": name + ".content.jsonl.gz"}
    for field, raw in blobs.items():
        (source / names[field]).write_bytes(gzip.compress(raw, mtime=0))
    pointer = {"n_records": len(records), "manifest": names["manifest"], "content_map": names["content_map"],
               "excludes_cohort": "toy synthetic receipt", **{field + "_uncompressed_sha256": _sha(raw) for field, raw in blobs.items()}}
    path, pin = write_json(root / (name + ".pointer.json"), pointer)
    return source, pointer, path, pin, blobs


class CMultirecordRefreshTests(unittest.TestCase):
    def fixtures(self, root, b_change=None, c_change=None, census_change=None):
        retained = {"id": "toy-retained", "source": "toy_train", "family": "toy_mcq",
                    "state": "Unique train-only fixture", "questions": {"q": {"type": "choice", "criteria": {"a": "one", "b": "two"}}}, "labels": {"q": "a"}}
        b = [retained, make_record(51, 0), make_record(51, 1)]
        # Same worlds as removed v1: versioned IDs and corrected visible rules differ.
        c = [make_record(51, i, revision=2) for i in range(5)]
        if b_change:
            b_change(b)
        if c_change:
            c_change(c)
        bdir, bp, bpath, bpin, bblobs = package(root, "b", b)
        cdir, cp, cpath, cpin, cblobs = package(root, "c", c)
        by_source = {}
        for row in b:
            by_source[row["source"]] = by_source.get(row["source"], 0) + 1
        report = {"select": {"n_records": len(b), "quotas": {"P3:extra": 2}, "by_source": by_source,
                              "tokens": {"t_v3w": {"max": 2048}}, **{n + "_uncompressed_sha256": bp[n + "_uncompressed_sha256"] for n in NAMES}}}
        b_report, b_report_pin = write_json(root / "build-report.json", report)
        b_exclusion, b_exclusion_pin = write_json(root / "exclusion-check.json", {"pass": True, "n_rows": len(b), "failures": {}, "pointer": bpath})
        c_summary, c_summary_pin = write_json(root / "c-summary.json", {"schema": "ssdm.c_multirecord_subset.v1", "pointer": cp,
            "pointer_sha256": cpin, "records": len(c), "record_bytes_preserved": True, "generator_spec_check_passed": True,
            "reused_token_bounds": {"source_token_limits": {"layout": "symmetric", "template": "jrt-rwkv-g1-v1", "max_tokens": 2048}}})
        census = [{"id": record["id"], "v2f": 500, "v3w": 600} for record in c]
        if census_change:
            census_change(census)
        census_path = root / "census.jsonl.gz"
        census_path.write_bytes(gzip.compress(b"".join(canonical(row) + b"\n" for row in census), mtime=0))
        census_summary, census_pin = write_json(root / "census-summary.json", {
            "schema": "ssdm.c_multirecord_trainer_census.v1", "records": len(c), "source_pointer_sha256": cpin,
            "source_records_sha256": cp["records_uncompressed_sha256"], "layout": "g1v2",
            "length_scope": "prompt plus longest continuation", "lengths": {name: {"n": len(c)} for name in ("v2f", "v3w")},
            "census_compressed_sha256": _sha(census_path.read_bytes()), "trainer_commit": "a" * 40})
        args = dict(b_source=bdir, b_pointer=bpath, b_pointer_sha256=bpin, b_report=b_report, b_report_sha256=b_report_pin,
                    b_exclusion=b_exclusion, b_exclusion_sha256=b_exclusion_pin, c_source=cdir, c_pointer=cpath, c_pointer_sha256=cpin,
                    c_summary=c_summary, c_summary_sha256=c_summary_pin, c_census_summary=census_summary,
                    c_census_summary_sha256=census_pin, quota=2, expected_records=len(b), seed=91, data_root=root)
        return args, bblobs, cblobs

    def only_args(self, args):
        return {key: value for key, value in args.items() if not key.startswith("b_") and key != "expected_records"}

    def test_refresh_preserves_all_non_c_record_and_sidecar_bytes_and_exact_quota(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args, b, c = self.fixtures(root)
            one = refresh(**args, out=root / "one")
            two = refresh(**args, out=root / "two")
            for name in NAMES:
                filename = "records.jsonl.gz" if name == "records" else one["pointer"][name]
                output = gzip.decompress((root / "one/train_set" / filename).read_bytes())
                filename2 = "records.jsonl.gz" if name == "records" else two["pointer"][name]
                self.assertEqual(output, gzip.decompress((root / "two/train_set" / filename2).read_bytes()))
                lines = output.splitlines(keepends=True)
                self.assertEqual(lines[0], b[name].splitlines(keepends=True)[0])
                self.assertTrue(all(line in c[name].splitlines(keepends=True) for line in lines[1:]))
                self.assertEqual(_sha(output), one["uncompressed_sha256"][name])
            self.assertEqual(one["by_source"], {"toy_train": 1, V2: 2})
            self.assertEqual((one["records"], one["removed_c_v1"], one["added_c_v2"]), (3, 2, 2))
            self.assertEqual(one["retained_byte_proof"]["source"], one["retained_byte_proof"]["output"])
            self.assertEqual(sum(one["fresh_c_label_counts"]["case_exact"].values()), 2)
            with self.assertRaises(FileExistsError):
                refresh(**args, out=root / "one")

    def test_bad_receipts_unknown_c_and_duplicate_pool_are_rejected(self):
        cases = (lambda rows: rows.__setitem__(1, make_record(51, 0, revision=2)),
                 lambda rows: rows[1].__setitem__("family", "synthetic_multirecord_unknown"))
        for change in cases:
            with self.subTest(change=change), TemporaryDirectory() as temp:
                root = Path(temp)
                args, _, _ = self.fixtures(root, b_change=change)
                with self.assertRaises(ValueError):
                    refresh(**args, out=root / "out")
                self.assertFalse((root / "out").exists())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _ = self.fixtures(root, c_change=lambda rows: rows.__setitem__(1, deepcopy(rows[0])))
            with self.assertRaises(ValueError):
                refresh(**args, out=root / "out")
            bad = dict(args, b_exclusion_sha256="0" * 64)
            with self.assertRaises(ValueError):
                refresh(**bad, out=root / "out")

    def test_source_hash_mismatch_and_payload_overflow_refuse_before_output(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _ = self.fixtures(root)
            with patch("data.c_multirecord_refresh.PAYLOAD_LIMIT", 1), self.assertRaises(MemoryError):
                refresh(**args, out=root / "out")
            source = Path(args["b_source"]) / "records.jsonl.gz"
            source.write_bytes(gzip.compress(gzip.decompress(source.read_bytes()) + b"\n"))
            with self.assertRaises(ValueError):
                refresh(**args, out=root / "out")
            self.assertFalse((root / "out").exists())

    def test_second_b_pass_change_never_produces_completed_pointer(self):
        import data.c_multirecord_refresh as module
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _ = self.fixtures(root)
            original, calls = module._triples, [0]
            def changed(directory, pointer):
                if Path(directory) == Path(args["b_source"]):
                    calls[0] += 1
                    if calls[0] == 2:
                        path = Path(directory) / "records.jsonl.gz"
                        raw = gzip.decompress(path.read_bytes()).replace(b": ", b":  ", 1)
                        path.write_bytes(gzip.compress(raw, mtime=0))
                yield from original(directory, pointer)
            with patch.object(module, "_triples", changed), self.assertRaises(ValueError):
                refresh(**args, out=root / "out")
            self.assertEqual(calls[0], 2)
            self.assertFalse((root / "out/train_set/pointer.json").exists())
            self.assertFalse((root / "out/summary.json").exists())

    def test_nli4ct_quarantine_refuses_before_any_record_read(self):
        import data.c_multirecord_refresh as module
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _ = self.fixtures(root)
            path = Path(args["b_report"])
            report = json.loads(path.read_bytes())
            report["select"]["by_source"]["nli4ct"] = 1
            _, args["b_report_sha256"] = write_json(path, report)
            with patch.object(module, "_triples", side_effect=AssertionError("quarantine read record bytes")), \
                 self.assertRaisesRegex(ValueError, "quarantined B mix"):
                refresh(**args, out=root / "out")
            self.assertFalse((root / "out").exists())

    def test_known_quarantined_pointer_refuses_before_report_or_record_read(self):
        import data.c_multirecord_refresh as module
        self.assertIn("10206b05e976bd0a043653233abd95f77575c4ddaed6b1f309a1dc97be935895", module.QUARANTINED_B_POINTERS)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _ = self.fixtures(root)
            # Once pointer bytes validate, deny even a missing/misleading report.
            args["b_report"] = root / "must-not-open.json"
            with patch.object(module, "QUARANTINED_B_POINTERS", {args["b_pointer_sha256"]}), \
                 patch.object(module, "_triples", side_effect=AssertionError("quarantine read record bytes")), \
                 self.assertRaisesRegex(ValueError, "quarantined B pointer"):
                refresh(**args, out=root / "out")
            self.assertFalse((root / "out").exists())

    def test_c_only_selects_eligible_whole_records_and_never_reads_b(self):
        import data.c_multirecord_refresh as module
        rejected = set()
        def overlength(rows):
            lowest = sorted(rows, key=lambda row: _rank(91, "refresh-c", row["id"]))[:2]
            lowest[0]["v3w"] = 2049
            lowest[1]["v2f"] = 2049
            rejected.update(row["id"] for row in lowest)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, c = self.fixtures(root, census_change=overlength)
            original = module._triples
            def only_c(directory, pointer):
                self.assertEqual(Path(directory), Path(args["c_source"]))
                yield from original(directory, pointer)
            with patch.object(module, "_triples", only_c):
                result = c_only(**self.only_args(args), out=root / "out")
            output = gzip.decompress((root / "out/train_set/records.jsonl.gz").read_bytes())
            records = [json.loads(line) for line in output.splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(row["id"] not in rejected and len(row["questions"]) == 7 for row in records))
            self.assertTrue(all(line in c["records"].splitlines(keepends=True) for line in output.splitlines(keepends=True)))
            self.assertEqual((result["eligible_records"], result["excluded_records"]), (3, 2))
            self.assertEqual(result["token_bounds"]["selected_max_effective_tokens"], {"v2f": 500, "v3w": 600})
            self.assertFalse(result["b_inputs_read"])

    def test_missing_duplicate_invalid_or_mismatched_census_fails_without_output(self):
        changes = (lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, dict(rows[0])),
                   lambda rows: rows[0].__setitem__("id", "unknown-source-id"),
                   lambda rows: rows[0].__setitem__("v3w", True),
                   lambda rows: rows[0].__setitem__("error", "failed item"))
        for change in changes:
            with self.subTest(change=change), TemporaryDirectory() as temp:
                root = Path(temp)
                args, _, _ = self.fixtures(root, census_change=change)
                with self.assertRaises(ValueError):
                    c_only(**self.only_args(args), out=root / "out")
                self.assertFalse((root / "out").exists())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _ = self.fixtures(root)
            (root / "census.jsonl.gz").write_bytes(gzip.compress(b"{}\n"))
            with self.assertRaisesRegex(ValueError, "compressed byte pin"):
                c_only(**self.only_args(args), out=root / "out")

    def test_c_only_capacity_failure_does_not_fill_with_overlength_records(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _ = self.fixtures(root, census_change=lambda rows: [row.__setitem__("v3w", 2049) for row in rows[:4]])
            with self.assertRaisesRegex(ValueError, "1 eligible, 2 required"):
                c_only(**self.only_args(args), out=root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
