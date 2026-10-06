import copy
import json
import unittest

from language_growth.development_data import generate_development_tasks, comparison_swap_counterfactual
from language_growth.training_records import export_training_records
from language_growth.transparent_cpu import feature_rows, fit, LinearOptionBackend, rollout


class TransparentCPUTests(unittest.TestCase):
    def setUp(self):
        self.tasks = generate_development_tasks(17, 30)
        self.records = export_training_records(self.tasks)

    def test_id_and_option_permutation_equivariance(self):
        request = copy.deepcopy(self.records[0]["request"])
        qid, _, original = feature_rows(request)
        q = request["questions"][qid]
        q["criteria"] = {"new-" + oid: desc for oid, desc in reversed(list(q["criteria"].items()))}
        request["ssdm"]["request_id"] = "arbitrary"
        self.assertEqual(dict(original), {oid[4:]: f for oid, f in feature_rows(request)[2]})

    def test_targets_not_features(self):
        changed = copy.deepcopy(self.records[0])
        changed["labels"] = {"bad": "changed"}
        changed["id"] = "unrelated"
        self.assertEqual(feature_rows(changed["request"]), feature_rows(self.records[0]["request"]))

    def test_counterfactual_changes_only_relation_sign_features(self):
        task = next(t for t in self.tasks if t["training_only"]["kind"] == "compare"
                    and t["training_only"]["program"]["nodes"][2]["relation"] != "equal")
        other = comparison_swap_counterfactual(task)
        a, b = feature_rows(task["model_inputs"][-1])[2], feature_rows(other["model_inputs"][-1])[2]
        self.assertNotEqual(a, b)
        self.assertEqual(feature_rows(task["model_inputs"][0])[2], feature_rows(other["model_inputs"][0])[2])

    def test_optimizer_learns_and_reloaded_weights_reach_planner(self):
        model, initial, history = fit(self.records, 17, 100, 2.0)
        self.assertLess(history[-1]["cross_entropy"], history[0]["cross_entropy"])
        self.assertNotEqual(model["weights"], initial["weights"])
        restored = json.loads(json.dumps(model))
        self.assertEqual(LinearOptionBackend(model).revision(), LinearOptionBackend(restored).revision())
        report, calls = rollout(generate_development_tasks(19, 9), LinearOptionBackend(restored))
        self.assertTrue(calls)
        self.assertFalse(any(r["status"] in ("runner_error", "provider_error", "provider_rejected")
                             for r in report["observations"]))

    def test_unsupported_slot_is_explicit(self):
        model, _, _ = fit(self.records, 17, 1, 2.0)
        request = copy.deepcopy(self.records[0]["request"])
        request["questions"] = {"slot-001-unknown": next(iter(request["questions"].values()))}
        from decide.backends.base import BackendUnsupported
        with self.assertRaises(BackendUnsupported):
            LinearOptionBackend(model).plan(request)

    def test_development_validation_is_separate_and_attribution_sums(self):
        validation = generate_development_tasks(19, 9, split="development_validation")
        records = export_training_records(validation, split="development_validation")
        with self.assertRaises(ValueError):
            export_training_records(validation)
        train_names = {r["entity_ref"] for t in self.tasks for r in t["training_only"]["catalog"]["records"]}
        validation_names = {r["entity_ref"] for t in validation for r in t["training_only"]["catalog"]["records"]}
        self.assertFalse(train_names & validation_names)
        model, _, _ = fit(self.records, 17, 1, 2.0)
        backend = LinearOptionBackend(model)
        for record in records:
            scored = backend.score(record["request"], lambda: None)
            logits = next(iter(scored.logits.values()))
            for oid, contributions in scored.extras["feature_contributions"].items():
                self.assertEqual(sum(contributions.values()), logits[oid])


if __name__ == "__main__":
    unittest.main()
