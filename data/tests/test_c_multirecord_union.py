"""Small exact two-component union tests, no tokenizer/model or external data."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from data.c_multirecord_compact import compact_record, load_pools
from data.c_multirecord_subset import _sha
from data.c_multirecord_synth import canonical, make_record
from data.c_multirecord_union import NAMES, TEMPLATES, TRAINER, union
from train.cohort import record_content_hash


def write_json(path, value):
    path.write_bytes(canonical(value) + b"\n")
    return path, _sha(path.read_bytes())


def component(root, name, records):
    directory = root / name
    directory.mkdir()
    lines = [json.dumps(row, separators=(",", ": ")).encode() + b"\n" for row in records]
    blobs = {"records": b"".join(lines),
             "manifest": b"".join(canonical({"id": row["id"], "source": row["source"], "family": row["family"], "sha256": _sha(raw[:-1])}) + b"\n" for row, raw in zip(records, lines)),
             "content_map": b"".join(canonical({"id": row["id"], "content_sha256": record_content_hash(row)}) + b"\n" for row in records)}
    names = {"records": "records.jsonl.gz", "manifest": name + ".manifest.jsonl.gz", "content_map": name + ".content.jsonl.gz"}
    for key, blob in blobs.items():
        (directory / names[key]).write_bytes(gzip.compress(blob, mtime=0))
    pointer = {"n_records": len(records), "manifest": names["manifest"], "content_map": names["content_map"],
               **{key + "_uncompressed_sha256": _sha(blob) for key, blob in blobs.items()}}
    path, pin = write_json(root / (name + ".pointer.json"), pointer)
    return directory, pointer, path, pin, blobs


class CMultirecordUnionTests(unittest.TestCase):
    def fixtures(self, root, duplicate_world=False, overlimit=False, tokenizer_mismatch=False):
        base = [make_record(19, i, revision=2) for i in (0, 1)]
        high = [compact_record(make_record(19, 71, revision=2), load_pools())]
        if duplicate_world:
            high[0]["source_id"] = base[0]["source_id"]
        bd, bp, bpath, bpin, bb = component(root, "base", base)
        hd, hp, hpath, hpin, hb = component(root, "high", high)
        trainer_pins, tokenizer_pins = {"train/run_cell.py": "a" * 64}, {"tokenizer.json": "b" * 64}
        census = {"source_pointer_sha256": "c" * 64, "source_records_sha256": "d" * 64, "records": 5,
                  "trainer_commit": TRAINER, "trainer_source_sha256": trainer_pins, "tokenizer_sha256": tokenizer_pins}
        cpath, cpin = write_json(root / "census-summary.json", census)
        bs = {"schema": "ssdm.c_multirecord_admitted_subset.v1", "records": 2, "pointer": bp, "pointer_sha256": bpin,
              "b_inputs_read": False, "source_records": 5, "source_pins": {"c_census_summary": cpin, "c_pointer": "c" * 64},
              "source_files_uncompressed_sha256": {"records": "d" * 64},
              "token_bounds": {"selected_whole_record_2k": True, "layout": "g1v2", "templates": TEMPLATES, "trainer_commit": TRAINER,
                               "trainer_source_sha256": trainer_pins, "selected_max_effective_tokens": {"v2f": 1000, "v3w": 2048 if not overlimit else 2049}}}
        hs = {"schema": "ssdm.c_multirecord_compact.v1", "records": 1, "pointer": hp, "pointer_sha256": hpin,
              "b_inputs_read": False, "torch_imported": False, "layout": "g1v2", "templates": TEMPLATES, "trainer_commit": TRAINER,
              "trainer_source_sha256": trainer_pins, "tokenizer_sha256": {"tokenizer.json": "e" * 64} if tokenizer_mismatch else tokenizer_pins,
              "lengths": {"v2f": {"max": 1100}, "v3w": {"max": 1500}}, "source_world_ids": [row["source_id"] for row in high],
              **{key: True for key in ("all_worlds_retained", "inverse_alpha_semantics_passed", "verify_record_check_record_passed", "direct_trainer_row_equality_passed")}}
        bsummary, bsummary_pin = write_json(root / "base-summary.json", bs)
        hsummary, hsummary_pin = write_json(root / "high-summary.json", hs)
        args = dict(base_source=bd, base_pointer=bpath, base_pointer_sha256=bpin, base_summary=bsummary, base_summary_sha256=bsummary_pin,
                    base_census_summary=cpath, high_source=hd, high_pointer=hpath, high_pointer_sha256=hpin,
                    high_summary=hsummary, high_summary_sha256=hsummary_pin, data_root=root)
        return args, bb, hb

    def run_toy(self, args, out):
        with patch("data.c_multirecord_union.BASE_RECORDS", 2), patch("data.c_multirecord_union.HIGH_RECORDS", 1), \
             patch("data.c_multirecord_union.ALL_K", {2, 3, 73}):
            return union(**args, out=out)

    def test_all_records_and_sidecar_bytes_are_retained_in_documented_order(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, base, high = self.fixtures(root)
            first = self.run_toy(args, root / "one")
            second = self.run_toy(args, root / "two")
            for name in NAMES:
                filename = "records.jsonl.gz" if name == "records" else first["pointer"][name]
                raw = gzip.decompress((root / "one/train_set" / filename).read_bytes())
                filename2 = "records.jsonl.gz" if name == "records" else second["pointer"][name]
                self.assertEqual(raw, gzip.decompress((root / "two/train_set" / filename2).read_bytes()))
                self.assertEqual(raw, base[name] + high[name])
                self.assertEqual(_sha(raw), first["pointer"][name + "_uncompressed_sha256"])
            self.assertEqual((first["records"], first["questions"]), (3, 21))
            self.assertEqual(first["per_k_records"], {2: 1, 3: 1, 73: 1})
            self.assertEqual(first["token_bounds"]["max_effective_tokens"], {"v2f": 1100, "v3w": 2048})
            self.assertEqual(sum(first["label_counts"]["case_exact"].values()), 3)
            self.assertEqual(len(first["world_source_ids"]), 3)
            with self.assertRaises(FileExistsError):
                self.run_toy(args, root / "one")

    def test_duplicate_world_refuses_before_any_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _, _ = self.fixtures(root, duplicate_world=True)
            with self.assertRaisesRegex(ValueError, "not disjoint"):
                self.run_toy(args, root / "out")
            self.assertFalse((root / "out").exists())

    def test_bad_hash_and_changed_component_data_refuse_before_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _, _ = self.fixtures(root)
            bad = dict(args, base_pointer_sha256="0" * 64)
            with self.assertRaises(ValueError):
                self.run_toy(bad, root / "out")
            path = Path(args["high_source"]) / "records.jsonl.gz"
            path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()).replace(b": ", b":  ", 1)))
            with self.assertRaises(ValueError):
                self.run_toy(args, root / "out")
            self.assertFalse((root / "out").exists())

    def test_failed_bounds_or_tokenizer_identity_refuse_qualification(self):
        for options in ({"overlimit": True}, {"tokenizer_mismatch": True}):
            with self.subTest(options=options), TemporaryDirectory() as temporary:
                root = Path(temporary)
                args, _, _ = self.fixtures(root, **options)
                with self.assertRaises(ValueError):
                    self.run_toy(args, root / "out")
                self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
