"""Semantic safety and recurrence checks for the bounded CPU diagnostic."""
import copy
import json
import unittest

import torch

from language_growth.development_data import generate_development_tasks, comparison_swap_counterfactual
from language_growth.typed_emission_ssm import (
    TypedEmissionSSM, adversarial, byte_batch, compile_target, gold_target,
    predict, project, row,
)


class TypedEmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tasks = generate_development_tasks(12, 6)

    def test_vectorized_encoder_matches_recurrence(self):
        model = TypedEmissionSSM()
        ids = byte_batch(["abc", "z", ""])
        a = model.decay_logit.sigmoid()
        h = torch.zeros(3, model.embedding.embedding_dim)
        accumulated = torch.zeros_like(h)
        for position in range(ids.shape[1]):
            active = ids[:, position].ne(0)[:, None]
            update = a * h + (1 - a) * model.input(model.embedding(ids[:, position])).tanh()
            h = torch.where(active, update, h)
            accumulated += h * active
        expected = torch.cat((h, accumulated / ids.ne(0).sum(1)[:, None].clamp_min(1)), 1)
        torch.testing.assert_close(model.encode(ids), expected, atol=1e-6, rtol=1e-5)

    def test_inference_only_visible_state_and_order_invariance(self):
        model = TypedEmissionSSM()
        states = [json.loads(t["model_inputs"][0]["state"]) for t in self.tasks]
        for state in states:
            candidates = project(state)
            self.assertEqual(len(candidates), 12)
            self.assertEqual(sum(c["target"]["kind"] == "compare" for c in candidates), 4)
        reversed_states = copy.deepcopy(states)
        for state in reversed_states:
            state["records"].reverse()
            state["context_ref"] = "ignored-context-identity"
        self.assertEqual(predict(model, states), predict(model, reversed_states))
        comparison = self.tasks[2]
        before = json.loads(comparison["model_inputs"][0]["state"])
        after = json.loads(comparison_swap_counterfactual(comparison)["model_inputs"][0]["state"])
        self.assertEqual([c["target"] for c in project(before)], [c["target"] for c in project(after)])
        self.assertEqual(before["task_input"], after["task_input"])

    def test_compilation_and_adversarial_boundaries(self):
        rows = [row(t, t["task_id"], "original") for t in self.tasks]
        for r in rows:
            expected = r["evaluation_only"]["program"]
            self.assertEqual(compile_target(r["state"], gold_target(expected)), expected)
        report = adversarial(rows)
        self.assertEqual(report["wrong_relation"]["reason"], "unsupported_assertion")
        self.assertEqual(report["unit_mismatch"]["reason"], "type_unit_mismatch")
        self.assertEqual(report["wrong_record"]["kernel_status"], "accepted")
        self.assertFalse(report["wrong_record"]["whole_answer_correct"])


if __name__ == "__main__":
    unittest.main()
