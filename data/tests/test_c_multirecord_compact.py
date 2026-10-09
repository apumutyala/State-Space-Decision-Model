"""Small alpha/pin/cohort/emission tests; full trainer qualification is a separate run."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from data.c_multirecord_compact import POOLS_PATH, POOLS_SHA256, compact_record, export_compact, load_pools
from data.c_multirecord_refresh import _triples
from data.c_multirecord_subset import _sha, stratum
from data.c_multirecord_synth import canonical, make_record, verify_record
from train.cohort import record_content_hash


class FakeRuntime:
    def assert_pins(self):
        pass
    def measure(self, row):
        return {"v2f": 1300, "v3w": 1400, "serialized": 1200}


class CMultirecordCompactTests(unittest.TestCase):
    def test_alpha_bijection_preserves_world_values_gold_order_and_all_references(self):
        manifest = load_pools()
        for k in range(2, 78):
            for style in range(3):
                old = make_record(20261008, k - 2 + 76 * style + 228 * 71, revision=2)
                snapshot = deepcopy(old)
                row = compact_record(old, manifest)
                self.assertEqual(old, snapshot)
                self.assertEqual(row["state"], old["state"])
                self.assertEqual(row["source_id"], old["source_id"])
                self.assertEqual(sorted(row["questions"]), list("abcdefg"))
                self.assertEqual(stratum(row), stratum(old))
                self.assertEqual(len(row["questions"]), 7)
                self.assertEqual(sum(m["kind"] == "multi_select_member" for m in row["aug"]["question_roles"].values()), 4)
                for name in ("case_exact", "multi_select"):
                    self.assertTrue(set(row["aug"][name]["question_ids"]) <= set(row["questions"]))
                choice = next(q for q, m in row["aug"]["question_roles"].items() if m["kind"] == "top_profile")
                self.assertEqual(len(row["questions"][choice]["criteria"]), k)
                self.assertEqual(set(row["aug"]["compact_view"]), {"version", "source_record_id", "pool_sha256"})
                self.assertEqual(row, compact_record(old, manifest))
                verify_record(row)

    def test_alpha_view_preserves_original_question_and_label_iteration_order(self):
        old = make_record(20261008, 0, revision=2)
        order = list(reversed(old["questions"]))
        for name in ("questions", "labels"):
            old[name] = {qid: old[name][qid] for qid in order}
        old["aug"]["question_roles"] = {qid: old["aug"]["question_roles"][qid] for qid in order}
        qmap = dict(zip(sorted(order), "abcdefg", strict=True))
        row = compact_record(old, load_pools())
        for mapping in (row["questions"], row["labels"], row["aug"]["question_roles"]):
            self.assertEqual(list(mapping), [qmap[qid] for qid in order])

    def test_pool_pins_duplicates_reserved_ids_and_bad_style_are_refused(self):
        original = load_pools()
        with self.assertRaises(ValueError):
            load_pools(POOLS_PATH, "0" * 64)
        mutations = (lambda data: data["pools"]["words"][0].__setitem__("id", "true"),
                     lambda data: data["pools"]["opaque"][0].__setitem__("id", "<>"),
                     lambda data: data["pools"]["letters"][0].__setitem__("answer_token", data["pools"]["letters"][1]["answer_token"]))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "pools.json"
            for change in mutations:
                data = deepcopy(original)
                change(data)
                path.write_bytes(canonical(data))
                with self.assertRaises(ValueError):
                    load_pools(path, _sha(path.read_bytes()))

    def source(self, root, missing=False, all_worlds=False):
        source = root / "source"
        source.mkdir()
        records = [make_record(20261008, k - 2 + 76 * style, revision=2)
                   for k in (range(2, 78) if all_worlds else range(73, 78)) for style in range(3)]
        if missing:
            records[-1] = make_record(20261008, 228, revision=2)
        raw = [canonical(record) + b"\n" for record in records]
        blobs = {"records": b"".join(raw),
                 "manifest": b"".join(canonical({"id": row["id"], "source": row["source"], "family": row["family"], "sha256": _sha(line[:-1])}) + b"\n" for row, line in zip(records, raw)),
                 "content_map": b"".join(canonical({"id": row["id"], "content_sha256": record_content_hash(row)}) + b"\n" for row in records)}
        names = {"records": "records.jsonl.gz", "manifest": "source.manifest.jsonl.gz", "content_map": "source.content.jsonl.gz"}
        for name, blob in blobs.items():
            (source / names[name]).write_bytes(gzip.compress(blob, mtime=0))
        pointer = {"n_records": len(records), "manifest": names["manifest"], "content_map": names["content_map"],
                   **{name + "_uncompressed_sha256": _sha(blob) for name, blob in blobs.items()}}
        pointer_path = root / "pointer.json"
        pointer_path.write_bytes(canonical(pointer))
        source_pin = _sha(pointer_path.read_bytes())
        summary = {"pointer": pointer, "pointer_sha256": source_pin, "records": len(records), "label_method_verified_records": len(records),
                   "generator": {"version": 2}, "source_sha256": {"data/c_multirecord_synth.py": load_pools()["generator_source_sha256"]}}
        summary_path = root / "summary.json"
        summary_path.write_bytes(canonical(summary))
        return dict(source=source, source_pointer=pointer_path, source_pointer_sha256=source_pin,
                    source_summary=summary_path, source_summary_sha256=_sha(summary_path.read_bytes()),
                    trainer_root=root / "trainer", tokenizer_dir=root / "tokenizer", data_root=root)

    def test_all_worlds_two_pass_stream_retains_all_k_styles_and_exact_worlds(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.source(root, all_worlds=True)
            passes, measured = [], []
            out = root / "out"
            def streaming_source(*arguments):
                passes.append(0)
                number = len(passes)
                for record, content, raw in _triples(*arguments):
                    if number == 1:
                        self.assertFalse(out.exists(), "no output before all source hashes pass")
                    else:
                        # Previous record was measured before the next source row is read.
                        self.assertEqual(len(measured), passes[-1])
                    passes[-1] += 1
                    yield record, content, raw
            class StreamingRuntime(FakeRuntime):
                def measure(runtime, row):
                    measured.append(row["source_id"])
                    return super().measure(row)
            with patch("data.c_multirecord_compact.SOURCE_RECORDS", 228), \
                 patch("data.c_multirecord_compact._triples", side_effect=streaming_source), \
                 patch("data.c_multirecord_compact.TrainerRuntime", return_value=StreamingRuntime()):
                report = export_compact(**args, out=out, all_worlds=True)
            self.assertEqual(passes, [228, 228])
            self.assertEqual(report["records"], 228)
            self.assertEqual(report["questions"], 1596)
            self.assertEqual(report["per_k_records"], {k: 3 for k in range(2, 78)})
            self.assertEqual(report["per_style_records"], {style: 76 for style in ("letters", "words", "opaque")})
            self.assertEqual(report["source_world_ids"], measured)
            self.assertEqual(report["retained_raw_source_payload_bytes"], 0)
            with gzip.open(args["source"] / "records.jsonl.gz", "rt", encoding="utf-8") as stream:
                originals = [json.loads(line) for line in stream]
            with gzip.open(out / "train_set/records.jsonl.gz", "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(rows, [compact_record(row, load_pools()) for row in originals])

    def test_each_source_file_hash_is_validated_before_output_creation(self):
        for name in ("records", "manifest", "content_map"):
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = self.source(root, all_worlds=True)
                # Keep pointer/summary internally bound, but make one full-stream pin false.
                pointer = json.loads(args["source_pointer"].read_bytes())
                pointer[name + "_uncompressed_sha256"] = "0" * 64
                args["source_pointer"].write_bytes(canonical(pointer))
                args["source_pointer_sha256"] = _sha(args["source_pointer"].read_bytes())
                summary = json.loads(args["source_summary"].read_bytes())
                summary.update(pointer=pointer, pointer_sha256=args["source_pointer_sha256"])
                args["source_summary"].write_bytes(canonical(summary))
                args["source_summary_sha256"] = _sha(args["source_summary"].read_bytes())
                with patch("data.c_multirecord_compact.SOURCE_RECORDS", 228), \
                     patch("data.c_multirecord_compact.TrainerRuntime", return_value=FakeRuntime()), self.assertRaises(ValueError):
                    export_compact(**args, out=root / "out", all_worlds=True)
                self.assertFalse((root / "out").exists())

    def test_all_worlds_missing_k_and_overlimit_never_publish(self):
        class TooLong(FakeRuntime):
            def measure(self, row):
                result = super().measure(row)
                result["serialized"] = 2049
                return result
        for missing, runtime in ((True, FakeRuntime()), (False, TooLong())):
            with self.subTest(missing=missing), TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = self.source(root, missing=missing, all_worlds=True)
                with patch("data.c_multirecord_compact.SOURCE_RECORDS", 228), \
                     patch("data.c_multirecord_compact.TrainerRuntime", return_value=runtime), self.assertRaises(ValueError):
                    export_compact(**args, out=root / "out", all_worlds=True)
                self.assertFalse((root / "out/train_set/pointer.json").exists())
                self.assertFalse((root / "out/summary.json").exists())

    def test_complete_cohort_export_is_deterministic_and_retains_every_world(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.source(root)
            original = (args["source"] / "records.jsonl.gz").read_bytes()
            with patch("data.c_multirecord_compact.SOURCE_RECORDS", 15), patch("data.c_multirecord_compact.WORLDS_PER_K", 3), \
                 patch("data.c_multirecord_compact.TrainerRuntime", return_value=FakeRuntime()):
                one = export_compact(**args, out=root / "one")
                two = export_compact(**args, out=root / "two")
            a = gzip.decompress((root / "one/train_set/records.jsonl.gz").read_bytes())
            b = gzip.decompress((root / "two/train_set/records.jsonl.gz").read_bytes())
            self.assertEqual(a, b)
            self.assertEqual(one["records"], 15)
            self.assertEqual(one["questions"], 105)
            self.assertEqual(one["per_k_records"], {k: 3 for k in range(73, 78)})
            self.assertEqual(len(set(one["source_world_ids"])), 15)
            self.assertEqual(one["lengths"]["v3w"]["max"], 1400)
            self.assertEqual((args["source"] / "records.jsonl.gz").read_bytes(), original)
            self.assertTrue(all(len(json.loads(line)["questions"]) == 7 for line in a.splitlines()))
            for name in ("records", "manifest", "content_map"):
                filename = "records.jsonl.gz" if name == "records" else one["pointer"][name]
                raw = gzip.decompress((root / "one/train_set" / filename).read_bytes())
                self.assertEqual(_sha(raw), one["pointer"][name + "_uncompressed_sha256"])
            with self.assertRaises(FileExistsError):
                export_compact(**args, out=root / "one")

    def test_missing_world_overlimit_and_bad_source_pin_never_complete(self):
        class TooLong(FakeRuntime):
            def measure(self, row):
                return {"v2f": 1300, "v3w": 2049, "serialized": 1200}
        for missing, runtime in ((True, FakeRuntime()), (False, TooLong())):
            with self.subTest(missing=missing), TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = self.source(root, missing)
                with patch("data.c_multirecord_compact.SOURCE_RECORDS", 15), patch("data.c_multirecord_compact.WORLDS_PER_K", 3), \
                     patch("data.c_multirecord_compact.TrainerRuntime", return_value=runtime), self.assertRaises(ValueError):
                    export_compact(**args, out=root / "out")
                self.assertFalse((root / "out/train_set/pointer.json").exists())
                self.assertFalse((root / "out/summary.json").exists())
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.source(root)
            args["source_pointer_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                export_compact(**args, out=root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
