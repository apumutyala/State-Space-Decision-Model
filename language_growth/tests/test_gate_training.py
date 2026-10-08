"""Small CPU-only tests of actual train GatePlanner slot supervision."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language_growth.gate_suites import LEVELS, build_suite, digest
from language_growth.gate_training import (
    TRAIN_SPLIT, _replay_task, export_training, generate_training_records,
)
from language_growth.parent_bridge import decision_request, legacy_request_fingerprint
from language_growth.planner import _candidate_sets, _fact_description
from train.run_cell import _cell_records, _development_split
from train.guard import SplitGuard


class GateTrainingTests(unittest.TestCase):
    def test_all_actual_slots_and_exact_target_mapping(self):
        rows = generate_training_records(worlds=1)
        self.assertEqual(len(rows), 39)
        self.assertEqual(len({row["id"] for row in rows}), 12)
        self.assertEqual({row["metadata"]["level"] for row in rows}, set(LEVELS))
        selected = {}
        for row in rows:
            qid, target = next(iter(row["labels"].items()))
            description = row["request"]["questions"][qid]["criteria"][target]
            selected.setdefault(row["metadata"]["slot"], set()).add(description)
        self.assertEqual(selected["semantic-operation"],
                         {"ReadFact", "Compare", "ExactDifference", "Conjoin", "Hypothesize", "Plan"})
        self.assertEqual(selected["licensed-production"],
                         {"l3_fact_pair_coordination_v1", "l3_fact_comparison_contrast_v1"})
        self.assertEqual(selected["energy-operation"], {"ReadTechnology"})
        self.assertEqual(len(selected["energy-record"]), 2)
        self.assertEqual(len(selected["plan-step"]), 2)
        for level in LEVELS:
            tasks, refs = build_suite(level, TRAIN_SPLIT, worlds=1)
            for task, ref in zip(tasks, refs, strict=True):
                exported = [row for row in rows if row["id"] == task["task_id"]]
                self.assertTrue(exported)
                self.assertTrue(all(row["metadata"]["reference_program_sha256"] == digest(ref["program"])
                                    for row in exported))

    def test_requests_exactly_rebuild_and_prefix_contains_only_earlier_choices(self):
        rows = generate_training_records(worlds=1)
        tasks = {task["task_id"]: task for level in LEVELS
                 for task in build_suite(level, TRAIN_SPLIT, worlds=1)[0]}
        prefixes = {}
        for row in rows:
            request, meta = row["request"], row["metadata"]
            task = tasks[row["id"]]
            qid, question = next(iter(request["questions"].items()))
            prior = prefixes.setdefault(row["id"], [])
            expected, _ = decision_request(
                task_input=task["task_input"], host=task["host"], catalog=task["catalog"],
                model=request["model"], parent_revision=request["ssdm"]["parent_revision"],
                request_id=request["ssdm"]["request_id"], question_id=qid,
                question=question["instructions"], options=question["criteria"],
                proposed_slot_choices=prior)
            self.assertEqual(request, expected)
            self.assertEqual(meta["request_fingerprint"], legacy_request_fingerprint(request))
            state = json.loads(request["state"])
            self.assertEqual(state["proposed_slot_choices"], prior)
            self.assertEqual(state["accepted_partial_program"], [])
            self.assertEqual(meta["earlier_target_choices"], len(prior))
            self.assertTrue(meta["diagnostic_teacher_forced"])
            self.assertFalse(meta["free_running"] or meta["promotion_eligible"])
            serialized = json.dumps(request)
            for forbidden in ("training_only", "reference_program", "label_provenance", '"labels"',
                              "input_sha256", "target_option_id", "meaning", '"nodes"', "promotion_eligible"):
                self.assertNotIn(forbidden, serialized)
            target = row["labels"][qid]
            prior.append({"slot": meta["slot"], "option_id": target,
                          "description": question["criteria"][target]})

    def test_train_only_generation_and_invalid_configs_before_source_access(self):
        real_build = build_suite
        with patch("language_growth.gate_training.build_suite", wraps=real_build) as build:
            generate_training_records(worlds=1)
            self.assertEqual([call.args[1] for call in build.call_args_list], [TRAIN_SPLIT] * 5)
        with patch("language_growth.gate_training.build_suite") as build:
            for config in ({"split": "development_validation"}, {"split": "language_heldout"},
                           {"split": "TEST"}, {"levels": ["L1", "L1"]}, {"levels": []},
                           {"worlds": True}, {"worlds": 65}, {"seed": False}):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    generate_training_records(**config)
            build.assert_not_called()
        task, ref = (items[0] for items in real_build("L1", TRAIN_SPLIT, worlds=1))
        for split in ("development_validation", "language_heldout"):
            changed = deepcopy(task)
            changed["split"] = split
            with self.assertRaisesRegex(ValueError, "development_train"):
                _replay_task(changed, ref, model="test", parent_revision="test")
        changed = deepcopy(task)
        changed["world_id"] = "language-development_validation-world"
        with self.assertRaisesRegex(ValueError, "worlds"):
            _replay_task(changed, ref, model="test", parent_revision="test")

    def test_no_checker_filtering_full_candidates_and_one_final_check(self):
        tasks, refs = build_suite("L2", TRAIN_SPLIT, worlds=1)
        from language_growth.kernel import check_program
        with patch("language_growth.gate_planner.check_program", wraps=check_program) as check:
            rows = _replay_task(tasks[1], refs[1], model="test", parent_revision="test")
            self.assertEqual(check.call_count, 1)
            self.assertEqual(check.call_args.args[0], refs[1]["program"])
        for row in rows:
            criteria = next(iter(row["request"]["questions"].values()))["criteria"]
            if row["metadata"]["slot"] == "comparison-relation":
                self.assertEqual(set(criteria.values()), {"less_than", "greater_than", "equal"})
            if row["metadata"]["slot"] == "right-operand":
                facts, _, _ = _candidate_sets(tasks[1]["host"], tasks[1]["catalog"], tasks[1]["task_input"], "entity_name")
                self.assertEqual(set(criteria.values()), {_fact_description(fact) for fact in facts})
                self.assertGreater(len(criteria), 1)
        rows = _replay_task(tasks[2], refs[2], model="test", parent_revision="test")
        steps = [row for row in rows if row["metadata"]["slot"] == "plan-step"]
        self.assertEqual(len(next(iter(steps[0]["request"]["questions"].values()))["criteria"]), 2)
        self.assertEqual(len(next(iter(steps[1]["request"]["questions"].values()))["criteria"]), 1)

    def test_scorer_metadata_has_no_influence_and_invalid_reference_fails(self):
        tasks, refs = build_suite("L1", TRAIN_SPLIT, worlds=1)
        task, ref = tasks[1], refs[1]
        original = _replay_task(task, ref, model="test", parent_revision="test")
        changed = deepcopy(ref)
        changed.update(meaning={"leak_marker": "do-not-send"}, text="do-not-send", kind="fake-kind",
                       label_provenance="do-not-send", score=999)
        self.assertEqual(_replay_task(task, changed, model="test", parent_revision="test"), original)
        changed["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            _replay_task(task, changed, model="test", parent_revision="test")
        changed = deepcopy(ref)
        compared = changed["program"]["nodes"][2]
        compared["relation"] = "less_than" if compared["relation"] != "less_than" else "greater_than"
        with self.assertRaisesRegex(ValueError, "reference replay failed"):
            _replay_task(task, changed, model="test", parent_revision="test")
        task = deepcopy(task)
        task["training_only"] = {"target": "do-not-send"}
        changed = deepcopy(ref)
        changed["input_sha256"] = digest(task)
        with self.assertRaisesRegex(ValueError, "reference replay failed"):
            _replay_task(task, changed, model="test", parent_revision="test")

    def test_reproducible_export_hashes_and_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("a", "b")]
            manifests = [export_training(path, worlds=1, model="native-pin", parent_revision="revision-pin")
                         for path in paths]
            self.assertEqual(manifests[0], manifests[1])
            data = (paths[0] / "training_records.jsonl").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), manifests[0]["training_records_sha256"])
            self.assertEqual(data, (paths[1] / "training_records.jsonl").read_bytes())
            self.assertEqual(len(data.splitlines()), manifests[0]["records"])
            self.assertFalse(manifests[0]["validation_or_heldout_generated"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_training(paths[0], worlds=1)
            self.assertEqual(data, (paths[0] / "training_records.jsonl").read_bytes())

    def test_existing_d_development_loader_and_whole_world_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train"
            manifest = export_training(path, worlds=2)
            records_path = str(path / "training_records.jsonl.gz")
            records = _cell_records(records_path, development=True)
            self.assertEqual(len(records), manifest["records"])
            with self.assertRaisesRegex(ValueError, "development loader only"):
                _cell_records(records_path, development=False)
            tr, ev, split = _development_split(records, {"n_train_records": 2, "n_eval_records": 1, "seed": 3})
            self.assertEqual(split["split_unit"], "world/task")
            self.assertTrue({records[key]["group_id"] for key in tr}.isdisjoint(records[key]["group_id"] for key in ev))
            self.assertEqual(len(tr) + len(ev), len(records))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "D Item package requires optional torch")
    def test_existing_d_item_consumer(self):
        from train.data import items_from_records
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train"
            manifest = export_training(path, worlds=1)
            records = _cell_records(str(path / "training_records.jsonl.gz"), development=True)
            # Byte IDs exercise the actual D layout/Item consumer without any
            # tokenizer, learned model, tensor operations, or protected data access.
            guard = SplitGuard.synthetic_only("autogenerated gate development_train only")
            items = items_from_records(records.values(), lambda text, trusted: list(text.encode("utf-8")), guard=guard)
            self.assertEqual(len(items), manifest["records"])
            self.assertEqual(len({item.id for item in items}), len(items))
            by_id = {item.id: item for item in items}
            for storage_key, row in records.items():
                qid, target = next(iter(row["labels"].items()))
                item = by_id[storage_key]
                self.assertEqual(item.row.option_ids[item.label], target)
                self.assertEqual(row["group_id"], row["metadata"]["world_id"])
                self.assertTrue(item.id.endswith("/" + qid))


if __name__ == "__main__":
    unittest.main()
