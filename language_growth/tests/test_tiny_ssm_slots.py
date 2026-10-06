import copy
import unittest

import torch

from language_growth.development_data import generate_development_tasks
from language_growth.tiny_ssm_slots import DiagonalSSM, TinySlotSSM, byte_batch, model_text


class TinySSMTests(unittest.TestCase):
    def test_closed_form_matches_recurrence_and_mean(self):
        torch.manual_seed(3)
        model = DiagonalSSM(8)
        ids = byte_batch(["a", "causal order"])
        actual = model(ids)
        expected = []
        for row in ids:
            h = torch.zeros(8)
            states = []
            for token in row[row.ne(0)]:
                a = model.decay_logit.sigmoid()
                u = torch.tanh(model.input(model.embedding(token)))
                h = a * h + (1 - a) * u
                states.append(h)
            expected.append(torch.cat((h, torch.stack(states).mean(0))))
        torch.testing.assert_close(actual, torch.stack(expected))
        self.assertFalse(torch.allclose(model(byte_batch(["ab"])), model(byte_batch(["ba"]))))

    def test_option_ids_and_metadata_do_not_change_features(self):
        original = generate_development_tasks(17, 1)[0]["model_inputs"][0]
        changed = copy.deepcopy(original)
        qid, q = next(iter(changed["questions"].items()))
        q["criteria"] = {f"unrelated-{i}": desc for i, desc in enumerate(reversed(list(q["criteria"].values())))}
        changed["ssdm"] = {"leaked_label": "x"}
        changed["labels"] = {"x": "bad"}
        left, right = model_text(original), model_text(changed)
        self.assertEqual((left[0], left[2]), (right[0], right[2]))

    def test_candidate_scores_are_invariant_to_batch_order(self):
        torch.manual_seed(9)
        model = TinySlotSSM(8).eval()
        context = byte_batch(["task and allowed records"])
        candidates = ["Assert fact", "Compare values", "Explain fact"]
        owners = torch.zeros(3, dtype=torch.long)
        first = model(context, byte_batch(candidates), owners)
        second = model(context, byte_batch(list(reversed(candidates))), owners)
        torch.testing.assert_close(first, second.flip(0))


if __name__ == "__main__":
    unittest.main()
