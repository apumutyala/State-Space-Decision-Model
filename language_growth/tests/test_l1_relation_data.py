"""Focused CPU-only tests for the conditional relation data boundary."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from language_growth.l1_relation_data import (
    CONDITIONAL_SCOPE, RELATION_QID, PinnedRWKVEncoder, assert_content_disjoint,
    check_token_lengths, encoded_inputs, load_l1_relation_data,
)
from language_growth.l1_world_training import L1WorldTrainingData
from language_growth.multirecord_data import MULTIWORLD_VALIDATION_SPLIT
from language_growth.multirecord_eval import MULTIWORLD_FIXTURE_DIR, _canonical
from train.guard import SplitGuard


class TinyEncoder:
    def __call__(self, text, trusted):
        return [ord(c) for c in text[:1]]

    def identity(self):
        return {"kind": "test-only"}


class L1RelationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_l1_relation_data()

    def test_exact_complete_cohorts_and_pinned_lineage(self):
        data = self.data
        for name, rows, count, worlds in (
            ("fit", data.fit_records, 576, 24),
            ("calibration", data.calibration_records, 192, 8),
            ("teacher_forced_validation", data.teacher_forced_validation_records, 768, 32),
        ):
            summary = data.manifest["partitions"][name]
            self.assertEqual((len(rows), summary["row_count"], summary["world_count"]), (count, count, worlds))
            self.assertEqual(set(summary["world_rows"].values()), {24})
            raw = b"".join((_canonical(r) + "\n").encode("ascii") for r in rows)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), summary["rows_sha256"])
            for row in rows:
                self.assertEqual(set(row["request"]["questions"]), {RELATION_QID})
                self.assertNotIn(row["metadata"]["world_id"], repr(row["request"]))
                for forbidden in ("labels", "metadata", "evaluator_only", "targets", "program", "world_id"):
                    self.assertNotIn(f'"{forbidden}"', json.dumps(row["request"]))
        self.assertEqual(len(data.teacher_forced_validation_targets), 768)
        self.assertFalse(data.manifest["tokens_checked"])

    def test_validation_prefix_exactly_earlier_operation_left_right_and_label_separate(self):
        tasks = {t["task_id"]: t for t in map(json.loads, (MULTIWORLD_FIXTURE_DIR /
                 f"{MULTIWORLD_VALIDATION_SPLIT}.jsonl").read_bytes().splitlines())}
        for row in self.data.teacher_forced_validation_records:
            task = tasks[row["id"]]
            expected = deepcopy(task["model_inputs"][3])
            prior = []
            for request, target in zip(task["model_inputs"][:3], task["evaluator_only"]["targets"][:3]):
                question = next(iter(request["questions"].values()))
                prior.append({**target, "description": question["criteria"][target["option_id"]]})
            state = json.loads(expected["state"])
            state["proposed_slot_choices"] = prior
            expected["state"] = _canonical(state)
            self.assertEqual(row["request"], expected)
            self.assertEqual([p["slot"] for p in prior], ["operation", "left", "right"])
            self.assertNotIn("relation", [p["slot"] for p in prior])
            self.assertNotIn("labels", row)
            self.assertEqual(self.data.teacher_forced_validation_targets[f"{row['id']}/{RELATION_QID}"],
                             task["evaluator_only"]["targets"][3]["option_id"])
            self.assertEqual(row["metadata"]["scope"], CONDITIONAL_SCOPE)
            self.assertTrue(row["metadata"]["teacher_forced"])
            self.assertTrue(row["metadata"]["evaluation_only"])

    def test_full_fit_conversion_precedes_relation_selection_and_preserves_sidecar(self):
        seen = []

        def spy(worlds, *args, **kwargs):
            seen.append(len(worlds.fit_records))
            # The actual consumer eagerly imports Torch. R0 proves dispatch
            # ordering with protocol objects; L1r1 must execute real conversion.
            items, sidecar = [], {}
            for record in worlds.fit_records:
                qid = next(iter(record["request"]["questions"]))
                item_id = f"{record['id']}/{qid}"
                items.append(SimpleNamespace(id=item_id, row=SimpleNamespace(qid=qid,
                    ids=[1], continuations=[[2]], option_seqs=[], meta={})))
                sidecar[item_id] = record["metadata"]["world_id"]
            return items, sidecar

        with patch.object(L1WorldTrainingData, "fit_items", spy):
            items, sidecar = self.data.fit_items(TinyEncoder(), guard=SplitGuard.synthetic_only("L1r0 test"))
        self.assertEqual(seen, [3072])
        self.assertEqual((len(items), len(sidecar)), (576, 576))
        self.assertEqual(set(sidecar.values()), self.data.world_data.fit_world_ids)
        for item in items:
            self.assertEqual(item.row.qid, RELATION_QID)
            self.assertNotIn("world_id", item.row.meta)
            self.assertNotIn("labels", item.row.meta)

    def test_label_mutation_never_changes_request_or_cache_inputs(self):
        rows = deepcopy(list(self.data.fit_records[:2]))
        before = [encoded_inputs(row, TinyEncoder()) for row in rows]
        requests = [deepcopy(row["request"]) for row in rows]
        for record in rows:
            options = record["request"]["questions"][RELATION_QID]["criteria"]
            record["labels"][RELATION_QID] = next(o for o in options if o != record["labels"][RELATION_QID])
            record["metadata"]["world_id"] = "evaluator-sidecar-mutation"
        after = [encoded_inputs(row, TinyEncoder()) for row in rows]
        self.assertEqual(before, after)
        self.assertEqual(requests, [row["request"] for row in rows])
        self.assertEqual(hashlib.sha256(_canonical(before).encode()).hexdigest(),
                         hashlib.sha256(_canonical(after).encode()).hexdigest())
        changed_targets = deepcopy(self.data.teacher_forced_validation_targets)
        first = next(iter(changed_targets))
        changed_targets[first] = "mutated-label-only"
        changed = replace(self.data, teacher_forced_validation_targets=changed_targets)
        self.assertEqual(changed.teacher_forced_validation_records, self.data.teacher_forced_validation_records)
        with self.assertRaisesRegex(ValueError, "target sidecar changed"):
            changed._verify()

    def test_complete_world_verification_rejects_empty_short_or_crossing_before_encoding(self):
        data = self.data.world_data
        cases = (
            replace(data, fit_records=()),
            replace(data, fit_records=data.fit_records[:-1]),
            replace(data, fit_records=self.data.fit_records),  # pre-filtered input is forbidden
            replace(data, fit_records=data.fit_records + data.calibration_records[:1]),
            replace(data, fit_records=data.fit_records + self.data.teacher_forced_validation_records[:1]),
            replace(data, calibration_records=()),
            replace(data, validation_tasks=data.validation_tasks[:-1]),
            replace(data, fit_world_ids=data.fit_world_ids | data.calibration_world_ids),
        )
        for worlds in cases:
            encoder = unittest.mock.Mock(side_effect=AssertionError("must reject before encoding"))
            with self.subTest(count=len(worlds.fit_records)), self.assertRaises(ValueError):
                replace(self.data, world_data=worlds).fit_items(encoder,
                    guard=SplitGuard.synthetic_only("partition substitution test"))
            encoder.assert_not_called()

    def test_normalized_content_overlap_rejected_even_with_new_identity_and_labels(self):
        left = deepcopy(self.data.fit_records[0])
        right = deepcopy(left)
        right["id"] = "new-task"
        right["metadata"]["world_id"] = "new-world"
        right["labels"][RELATION_QID] = "new-label"
        state = json.loads(right["request"]["state"])
        right["request"]["state"] = _canonical(state).upper()  # casefold-equivalent content
        with self.assertRaisesRegex(ValueError, "normalized-content overlap"):
            assert_content_disjoint({"fit": [left], "validation": [right]})
        for summary in self.data.manifest["normalized_content"].values():
            self.assertGreater(summary["unique_contents"], 0)

    def test_validation_source_byte_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # No train source access is needed to prove the validation byte pin.
            for suffix in (".jsonl", "-manifest.json"):
                src = MULTIWORLD_FIXTURE_DIR / f"{MULTIWORLD_VALIDATION_SPLIT}{suffix}"
                (root / src.name).write_bytes(src.read_bytes())
            path = root / f"{MULTIWORLD_VALIDATION_SPLIT}.jsonl"
            path.write_bytes(path.read_bytes() + b" ")
            from language_growth.l1_relation_data import _validation_view
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _validation_view(root, self.data.world_data)

    def test_exact_2048_token_boundary_checks_continuations_without_truncation(self):
        lengths = {"example": {"prefix_tokens": 2046, "max_call_tokens": 2048}}
        self.assertEqual(check_token_lengths(lengths)["example"]["max_call_tokens"], 2048)
        lengths["example"]["max_call_tokens"] = 2049
        with self.assertRaisesRegex(ValueError, "2049 tokens exceeds 2048.*no truncation"):
            check_token_lengths(lengths)
        self.assertEqual(lengths["example"], {"prefix_tokens": 2046, "max_call_tokens": 2049})
        with self.assertRaisesRegex(ValueError, "nonempty"):
            check_token_lengths({})

    def test_r0_census_cannot_import_torch_or_item_consumer(self):
        import builtins
        real_import = builtins.__import__

        def no_models(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch.") or name in ("train.data", "heads", "heads.layout"):
                raise AssertionError(f"R0 imported model dependency {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", no_models):
            report = self.data.token_length_report(TinyEncoder())
        self.assertEqual({name: s["row_count"] for name, s in report["summary"].items()},
                         {"full_fit": 3072, "full_reserved_calibration": 1024, "teacher_forced_validation": 768})
        self.assertTrue(report["tokenizer_files_only"])
        self.assertEqual(report["template"], "jrt-rwkv-g1-v1")

    def test_tokenizer_pin_rejects_substitution_without_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.txt"
            path.write_text("not the pinned tokenizer", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "vocabulary SHA-256 mismatch"):
                PinnedRWKVEncoder(path)


if __name__ == "__main__":
    unittest.main()
