"""Public synthetic protocol tests, never a real heldout/model evaluation."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language_growth.gate_evidence import load_evidence_manifest, load_prepared_suite, run_registered_interventions, evaluate_prepared_level
from language_growth.gate_suites import build_suite, canonical, digest, SUITE_SCHEMA
from language_growth.gates import sha_file
from language_growth.planner import PlannerResult
from language_growth.tests.test_language_gates import FakeParent


def fake_intervention_adapter(tasks, parent, prereg):
    """All arms fail identically: protocol shape alone cannot qualify effects."""
    return {"instrumentation_valid":False,
            "arms": {name: [PlannerResult("error", None, None, "fake_no_model") for _ in tasks]
                     for name in ("semantic", "sham", "random", "output_only")},
            "provenance": [{"task_id": task["task_id"], "input_sha256": digest(task),
                            "parent_revision": prereg["parent_revision"],
                            "registration_sha256": prereg["registration_sha256"],
                            "capture_identity_sha256": prereg["capture_identity_sha256"],
                            "arms": ["semantic", "sham", "random", "output_only"],
                            "state_unchanged_sha256_before": "a"*64, "state_unchanged_sha256_after": "a"*64}
                           for task in tasks]}


_SCOPED_TEST_MODE = "valid"


def fake_scoped_adapter(tasks, parent, prereg):
    """Only applicability protocol; the test supplies no neural measurements."""
    arms = ("semantic", "sham", "random", "output_only")
    provenance = []
    for task in tasks:
        captures = {}
        for arm in arms:
            def call(qid, targeted):
                return {"question_id": qid, "targeted": targeted, "head_calls": 1, "status": "ok",
                        "capture": {"hidden_delta_l2": 0, "matched_delta_l2": 0, "output_delta_l2": 0,
                                    "hidden_before_sha256": "a"*64, "hidden_after_sha256": "a"*64,
                                    "logits_before_sha256": "a"*64, "logits_after_sha256": "a"*64}}
            first = call("slot-001-semantic-operation", False)
            second = call("slot-004-comparison-relation", True)
            if _SCOPED_TEST_MODE == "no_target":
                second = call("slot-002-fact", False)
            if _SCOPED_TEST_MODE == "wrong_marking":
                second["targeted"] = False
            if _SCOPED_TEST_MODE == "edited_untargeted":
                first["capture"]["hidden_after_sha256"] = "b"*64
            captures[arm] = [first, second]
        provenance.append({"task_id": task["task_id"], "input_sha256": digest(task),
                           "parent_revision": prereg["parent_revision"],
                           "registration_sha256": prereg["registration_sha256"],
                           "capture_identity_sha256": prereg["capture_identity_sha256"],
                           "arms": list(arms), "state_unchanged_sha256_before": "a"*64,
                           "state_unchanged_sha256_after": "a"*64, "instrumentation_valid": True,
                           "target_slots": prereg["target_slots"], "captures": captures})
    counts = {arm: len(tasks) for arm in arms}
    if _SCOPED_TEST_MODE == "wrong_count":
        counts["random"] = 0
    return {"instrumentation_valid": True, "targeted_cohort_valid": True,
            "targeted_capture_counts": counts, "provenance": provenance,
            "arms": {arm: [PlannerResult("kernel_rejected", None, None, "unsupported_assertion")
                            for _ in tasks] for arm in arms}}


class EvidenceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "native applicability helpers require torch runtime")
    def test_scoped_interventions_cannot_pass_without_valid_targeted_calls_in_every_arm(self):
        tasks, refs = build_suite("L0", "development_validation", worlds=1)
        parent = FakeParent([])
        prereg = {"schema": "ssdm.language_intervention_preregistration.v2",
                  "parent_revision": parent.planner_kwargs()["parent_revision"],
                  "discovery_world_ids": ["separate-public-discovery-world"],
                  "capture_identity_sha256": "b"*64, "capture_boundary": "native_fork_prehead.v1",
                  "scope": "final_hidden_semantic_axis_only", "output_control": "fitted_candidate_count_logit_direction.v1",
                  "target_slots": ["comparison-relation"]}
        prereg["registration_sha256"] = digest(prereg)
        with tempfile.TemporaryDirectory() as tmp:
            registration = Path(tmp)/"registration.json"
            registration.write_bytes(canonical(prereg))
            kwargs = {"adapter": __name__+":fake_scoped_adapter", "adapter_source_sha256": sha_file(Path(__file__)),
                      "registration": registration, "registration_sha256": sha_file(registration), "draws":100}
            # Force a positive score so failed applicability cannot hide behind
            # the ordinary all-arm tie in this protocol-only fixture.
            with patch("language_growth.gates.intervention_comparison", side_effect=lambda *_a, **_k: {"criterion_observed": True}):
                for mode in ("no_target", "wrong_marking", "edited_untargeted", "wrong_count", "valid"):
                    with self.subTest(mode=mode), patch(__name__+"._SCOPED_TEST_MODE", mode):
                        report = run_registered_interventions(tasks, refs, parent, **kwargs)
                        self.assertEqual(report["instrumentation_valid"], mode == "valid")
                        self.assertEqual(report["passed"], mode == "valid")

    def test_machine_evidence_requires_exact_artifact_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.json"
            path.write_bytes(canonical({"schema": "ssdm.language_gate_evidence.v1"}))
            with self.assertRaises(ValueError):
                load_evidence_manifest(path, sha_file(path), parent_sha256="a"*64, levels=["L0"])

    def test_prepared_suite_hashes_before_labels_and_joint_strata_required(self):
        tasks, refs = build_suite("L0", "development_validation", worlds=1)
        # Temporary DEV-derived protocol fixtures are deliberately not a real
        # heldout generator draw or model evaluation.
        for task, ref in zip(tasks, refs):
            task["split"] = "language_heldout"
            ref["input_sha256"] = digest(task)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_bytes = b"".join(canonical(t)+b"\n" for t in tasks)
            ref_bytes = b"".join(canonical(r)+b"\n" for r in refs)
            (root/"inputs.jsonl").write_bytes(input_bytes)
            (root/"evaluator.jsonl").write_bytes(ref_bytes)
            manifest = {"schema": SUITE_SCHEMA, "level": "L0", "split": "language_heldout", "worlds": 1,
                        "tasks": 2, "training_allowed": False, "scope": "synthetic_protocol_only",
                        "inputs_sha256": hashlib.sha256(input_bytes).hexdigest(), "evaluator_sha256": hashlib.sha256(ref_bytes).hexdigest()}
            manifest["development_inventories"] = {split:{"world_ids":["separate-public-world"], "input_content_sha256":["f"*64]}
                                                    for split in ("development_train","development_validation")}
            (root/"manifest.json").write_bytes(canonical(manifest))
            with self.assertRaisesRegex(ValueError, "strata"):
                load_prepared_suite(root, "L0", manifest_sha256=sha_file(root/"manifest.json"))
            refs[0]["stratum"] = "usefulness"
            refs[1]["stratum"] = "joint_adversarial"
            ref_bytes = b"".join(canonical(r)+b"\n" for r in refs)
            (root/"evaluator.jsonl").write_bytes(ref_bytes)
            manifest["evaluator_sha256"] = hashlib.sha256(ref_bytes).hexdigest()
            (root/"manifest.json").write_bytes(canonical(manifest))
            loaded = load_prepared_suite(root, "L0", manifest_sha256=sha_file(root/"manifest.json"))
            self.assertEqual(len(loaded[1]), 2)
            (root/"evaluator.jsonl").write_text("untrusted changed labels")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_prepared_suite(root, "L0", manifest_sha256=sha_file(root/"manifest.json"))

    def test_source_pinned_raw_adapter_ties_never_pass_and_overlap_rejects(self):
        tasks, refs = build_suite("L0", "development_validation", worlds=2)
        parent = FakeParent([])
        prereg = {"schema": "ssdm.language_intervention_preregistration.v1",
                  "parent_revision": parent.planner_kwargs()["parent_revision"],
                  "discovery_world_ids": ["separate-public-discovery-world"],
                  "capture_identity_sha256": "b"*64, "capture_boundary": "native_fork_prehead.v1",
                  "scope": "final_hidden_semantic_axis_only", "output_control": "fitted_candidate_count_logit_direction.v1"}
        prereg["registration_sha256"] = digest(prereg)
        with tempfile.TemporaryDirectory() as tmp:
            registration = Path(tmp)/"registration.json"
            registration.write_bytes(canonical(prereg))
            kwargs = {"adapter": __name__+":fake_intervention_adapter", "adapter_source_sha256": sha_file(Path(__file__)),
                      "registration": registration, "registration_sha256": sha_file(registration), "draws":100}
            report = run_registered_interventions(tasks, refs, parent, **kwargs)
            self.assertFalse(report["passed"])
            self.assertFalse(report["criterion_observed"])
            prereg["discovery_world_ids"] = [tasks[0]["world_id"]]
            prereg["registration_sha256"] = digest({k:v for k,v in prereg.items() if k != "registration_sha256"})
            registration.write_bytes(canonical(prereg))
            kwargs["registration_sha256"] = sha_file(registration)
            with self.assertRaisesRegex(ValueError, "overlap"):
                run_registered_interventions(tasks, refs, parent, **kwargs)

    def test_missing_comparator_cannot_qualify_and_late_adapter_failure_preserves_outputs(self):
        tasks, refs = build_suite("L0", "development_validation", worlds=1)
        refs[0]["stratum"], refs[1]["stratum"] = "usefulness", "joint_adversarial"
        identity = {"model_id":"selected-open-model", "revision":"pinned", "open_weight_source":"selected-open-model",
                    "unrestricted_generation":True,"constrained_decoding":False}
        comparator = {"schema":"ssdm.language_comparator.v1", "identity":identity, "observations":[]}
        entry = {"manifest_sha256":"a"*64,"comparator_artifact_sha256":"b"*64,"comparator_identity_sha256":digest(identity),
                 "comparator_model_id":identity["model_id"],"comparator_revision":identity["revision"],
                 "comparator_source":identity["open_weight_source"],"decision_authorization_sha256":"c"*64,
                 "intervention_adapter_source_sha256":sha_file(Path(__file__)),"intervention_registration_sha256":"e"*64}
        manifest = {"scope":"public-synthetic-protocol","label_provenance":"authored_test_only"}
        parent = FakeParent([])
        parent.identity = lambda: {"artifact_sha256":"f"*64,"parent_revision":"scripted"}
        kwargs = dict(review_entry=entry, comparator_artifact=Path("unused"), decision_authorization=Path("unused"),
                      decision_parent=Path("unused"),decision_candidate=Path("unused"),decision_scoring_rows=Path("unused"),
                      intervention_adapter=__name__+":fake_intervention_adapter",intervention_registration=Path("unused"),candidate_sha256="f"*64,draws=100)
        rows = [{"task_id":t["task_id"],"world_id":t["world_id"],"correct":True,"unsupported_release":False,"trace":[]} for t in tasks]
        with patch("language_growth.gate_evidence.load_prepared_suite",return_value=(manifest,tasks,refs)), \
             patch("language_growth.gate_evidence.read_pinned",return_value=comparator), \
             patch("language_growth.gates.run_tasks",return_value=rows) as runner, \
             patch("language_growth.gate_decision_retention.decision_index_retention",return_value={"passed":False}):
            with self.assertRaisesRegex(ValueError,"complete operational"):
                evaluate_prepared_level(Path("unused"),"L0",parent,**kwargs)
            runner.assert_not_called()
            comparator["observations"] = [{"task_id":tasks[0]["task_id"],"input_sha256":digest(tasks[0]),
                                           "status":"ok","raw_text":canonical(refs[0]["meaning"]).decode()}]
            with tempfile.TemporaryDirectory() as tmp, \
                 patch("language_growth.gate_evidence.run_registered_interventions",side_effect=ValueError("failed capture")):
                out = Path(tmp)/"L0"
                report = evaluate_prepared_level(Path("unused"),"L0",parent,evidence_out=out,**kwargs)
                self.assertTrue((out/"candidate.json").exists())
                self.assertFalse(report["passed"])
                self.assertEqual(report["gates"]["semantic_intervention"]["status"],"unmeasured")


if __name__ == "__main__":
    unittest.main()
