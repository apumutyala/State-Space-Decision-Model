"""One CPU optimizer step through the real D trainer and back into the L planner.

This is an integration test on a tiny random recurrent backbone, never a
language capability result. No downloads or GPU are used.
"""

import hashlib
import importlib.util
import json
import math
import unittest
from copy import deepcopy


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Track D's torch test environment")
class ParentTrainingIntegrationTests(unittest.TestCase):
    def test_export_train_and_plan_through_actual_head(self):
        import torch
        from decide.adapter import Decider
        from decide.calibration import OptionCountTemperature
        from heads import ByteEncoder, DecisionModel
        from heads.backbone import tiny_backbone
        from heads.decide_backend import DecisionModelBackend
        from train.data import items_from_records
        from train.guard import SplitGuard
        from train.trainer import Trainer, TrainConfig
        from language_growth.development_data import generate_development_tasks, development_manifest
        from language_growth.free_running_dev import run_development_tasks
        from language_growth.training_records import export_training_records

        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            examples = generate_development_tasks(17, 1, choice_format="production")
            records = export_training_records(examples)
            guard = SplitGuard.synthetic_only("fresh language integration-test worlds")
            encoder = ByteEncoder()
            items = items_from_records(records, encoder, guard=guard)
            model = DecisionModel(tiny_backbone(seed=17, hidden=8, layers=1), "snapshot", seed=17)
            self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
            cfg = TrainConfig(head="snapshot", lora_rank=2, batch_size=1, max_steps=1, seed=17)
            trainer = Trainer(model, cfg, items, guard=guard)
            before = {k: v.clone() for k, v in trainer.trainable_state().items()}
            history = trainer.train()
            self.assertEqual(len(history), 1)
            self.assertTrue(math.isfinite(history[0]["loss"]))
            after = trainer.trainable_state()
            self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before))
            digest = hashlib.sha256()
            for name, tensor in sorted(model.state_dict().items()):
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            backend = DecisionModelBackend(model, encoder, checkpoint={"state_sha256": digest.hexdigest()})
            calibration = OptionCountTemperature()
            provider = Decider(backend, calibration=calibration)
            calls = []

            class RecordingProvider:
                def decide(self, request):
                    response = provider.decide(request)
                    calls.append({"request": deepcopy(request), "response": deepcopy(response)})
                    return response

            validation = generate_development_tasks(19, 1, choice_format="production", split="development_validation")
            report = run_development_tasks(validation, RecordingProvider(), model="tiny-integration-test",
                                           parent_revision=backend.revision()["revision_id"],
                                           expected_served_config={"calibration": calibration.identity(),
                                               "policy_masking": "none (served covers every supplied option)"})
            row = report["observations"][0]
            self.assertGreater(row["provider_calls"], 0)
            self.assertNotIn(row["status"], ("runner_error", "provider_error", "provider_rejected", "precondition_failed"))
            self.assertEqual(row["provider_calls"], row["forward_calls"])
            print(json.dumps({"scope": "tiny random CPU integration, not capability",
                              "config": cfg.to_json(), "history": history, "state_sha256": digest.hexdigest(),
                              "train_manifest": development_manifest(examples),
                              "validation_manifest": development_manifest(validation),
                              "calls": calls, "raw_report": report}, sort_keys=True))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
