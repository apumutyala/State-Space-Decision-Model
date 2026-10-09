"""No-model tests of strict qualified-parent versus fitted-child CLI routing."""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from language_growth.direction_fit_admission import preflight_direction_pairs, resolve_direction_fit_parent
from language_growth.gate_operation_pairs import development_operation_pairs
from language_growth.parent_bridge import decision_request, legacy_request_fingerprint


class DirectionFitAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "candidate_checkpoint": "candidate.pt",
            "parent_report": "report.json", "parent_report_sha256": "1" * 64,
            "parent_qualification": "qualification.json", "parent_qualification_sha256": "2" * 64,
            "parent_snapshot": "snapshot-dir", "device": "cpu", "dtype": "bfloat16",
        }
        self.qualified = Mock(return_value="qualified-parent")
        self.child = Mock(return_value="language-child")

    def test_direction_pair_preflight_accepts_paired_development_cohorts(self):
        discovery = development_operation_pairs(split="development_train", worlds=4, seed=1201)
        validation = development_operation_pairs(split="development_validation", worlds=4, seed=1202)
        result = preflight_direction_pairs(discovery, validation)
        self.assertEqual(result, {
            "discovery_pairs": 4, "validation_pairs": 4,
            "concept": "Compare_to_ExactDifference",
            "discovery_option_counts": [4], "validation_option_counts": [4],
            "target_slots": None,
        })
        known = {"semantic-operation", "comparison-relation"}
        scoped = preflight_direction_pairs(discovery, validation, target_slots=["semantic-operation"],
                                           known_target_slots=known)
        self.assertEqual(scoped["target_slots"], ["semantic-operation"])
        with self.assertRaisesRegex(ValueError, "declared target slot"):
            preflight_direction_pairs(discovery, validation, target_slots=["comparison-relation"],
                                      known_target_slots=known)
        zero_ordinal = deepcopy(discovery)
        for pair in zero_ordinal:
            pair["source"]["question_id"] = "slot-000-semantic-operation"
            pair["effect"]["question_id"] = "slot-000-semantic-operation"
        with self.assertRaisesRegex(ValueError, "declared target slot"):
            preflight_direction_pairs(zero_ordinal, validation, target_slots=["semantic-operation"],
                                      known_target_slots=known)
        with self.assertRaisesRegex(ValueError, "registered GatePlanner"):
            preflight_direction_pairs(discovery, validation, target_slots=["unregistered"],
                                      known_target_slots=known)

    def test_direction_pair_preflight_rejects_overlap_and_malformed_rows(self):
        discovery = development_operation_pairs(split="development_train", worlds=2, seed=1203)
        validation = development_operation_pairs(split="development_validation", worlds=2, seed=1204)
        duplicated = [*discovery, deepcopy(discovery[0])]
        with self.assertRaisesRegex(ValueError, "duplicate pair IDs/content"):
            preflight_direction_pairs(duplicated, validation)
        overlapped = deepcopy(validation)
        overlapped[0]["world_id"] = discovery[0]["world_id"]
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            preflight_direction_pairs(discovery, overlapped)
        malformed = deepcopy(discovery)
        malformed[0]["source"].pop("catalog")
        with self.assertRaisesRegex(ValueError, "request specification"):
            preflight_direction_pairs(malformed, validation)

    def test_keeps_qualified_parent_routing_when_child_pins_are_absent(self):
        result = resolve_direction_fit_parent(
            **self.kwargs, load_qualified_parent=self.qualified, load_language_child=self.child)
        self.assertEqual(result, "qualified-parent")
        self.qualified.assert_called_once_with(
            "candidate.pt", report="report.json", report_sha256="1" * 64,
            qualification="qualification.json", qualification_sha256="2" * 64,
            snapshot="snapshot-dir", device="cpu", dtype="bfloat16")
        self.child.assert_not_called()

    def test_direction_fit_cli_routes_complete_child_pins_without_model_import(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            discovery, validation = root / "train.json", root / "validation.json"
            discovery_raw = json.dumps(development_operation_pairs(
                split="development_train", worlds=1, seed=1211))
            discovery.write_text(discovery_raw, encoding="utf-8")
            validation.write_text(json.dumps(development_operation_pairs(
                split="development_validation", worlds=1, seed=1212)), encoding="utf-8")
            child_parent = object()
            load_child = Mock(return_value=child_parent)
            load_qualified = Mock(return_value=object())
            gates_module = ModuleType("language_growth.gates")
            gates_module._strict_json = json.loads
            child_module = ModuleType("language_growth.language_child")
            child_module.load_language_child = load_child
            qualified_module = ModuleType("language_growth.qualified_parent")
            qualified_module.load_qualified_parent = load_qualified
            qualified_module.file_sha256 = lambda _path: "0" * 64
            bridge_module = ModuleType("language_growth.parent_bridge")
            bridge_module.DECISION_SCHEMA = "schema"
            bridge_module.FINGERPRINT_ALGORITHM = "fingerprint"
            bridge_module.decision_request = decision_request
            bridge_module.legacy_request_fingerprint = legacy_request_fingerprint
            intervention_module = ModuleType("language_growth.gate_interventions")
            for name, value in {
                "ARMS": ("semantic", "sham", "random", "output_only"), "BOUNDARY": "boundary",
                "DIRECTION_SCHEMA": "direction", "DIRECTION_SCHEMA_V2": "direction-v2",
                "OUTPUT_CONTROL": "control", "SCHEMA": "schema", "SCHEMA_V2": "schema-v2",
                "SCOPE": "scope", "TARGET_SLOTS": {"semantic-operation"}, "_ArmProvider": object,
                "_OWNER": object(), "_native": object(), "_parent_state_sha": object(), "_sha": object(),
                "_tensor_sha": object(), "native_capture_identity": object(), "registration_sha256": object(),
                "question_slot": object(), "validate_target_slots": object(),
            }.items():
                setattr(intervention_module, name, value)
            modules = {
                "language_growth.gates": gates_module,
                "language_growth.language_child": child_module,
                "language_growth.qualified_parent": qualified_module,
                "language_growth.parent_bridge": bridge_module,
                "language_growth.gate_interventions": intervention_module,
            }
            module_name = "language_growth._test_direction_fit_cli"
            with patch.dict(sys.modules, modules):
                spec = importlib.util.spec_from_file_location(
                    module_name, Path(__file__).parents[1] / "gate_direction_fit.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                module.fit_native_directions = Mock(return_value={
                    "evidence": {"validation_effect_target_selected": 0.5}})
                module.write_direction_fit = Mock(return_value={"registration_sha256": "a" * 64})
                args = ["--parent", "child.pt", "--parent-report", "report.json",
                        "--parent-report-sha256", "1" * 64, "--parent-qualification", "qualification.json",
                        "--parent-qualification-sha256", "2" * 64, "--parent-snapshot", "snapshot",
                        "--stage1-parent-checkpoint", "stage1.pt", "--fit-receipt", "fit.json",
                        "--fit-receipt-sha256", "3" * 64, "--discovery-pairs", str(discovery),
                        "--validation-pairs", str(validation), "--out", str(root / "out"),
                        "--scale", "0.02", "--random-seed", "7", "--target-slots", "semantic-operation"]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(module.main(args), 0)
                load_child.assert_called_once_with(
                    "child.pt", stage1_checkpoint="stage1.pt", receipt="fit.json",
                    receipt_sha256="3" * 64, report="report.json", report_sha256="1" * 64,
                    qualification="qualification.json", qualification_sha256="2" * 64,
                    snapshot="snapshot", device="cpu", dtype="bfloat16")
                load_qualified.assert_not_called()
                self.assertIs(module.fit_native_directions.call_args.args[0], child_parent)
                module.fit_native_directions.assert_called_once()

                malformed = json.loads(discovery.read_text(encoding="utf-8"))
                malformed[0]["source"].pop("catalog")
                discovery.write_text(json.dumps(malformed), encoding="utf-8")
                bad_args = args.copy()
                bad_args[bad_args.index("--out") + 1] = str(root / "malformed-out")
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "request specification"):
                    module.main(bad_args)
                load_child.assert_called_once()
                load_qualified.assert_not_called()
                module.fit_native_directions.assert_called_once()
                discovery.write_text(discovery_raw, encoding="utf-8")

                partial = args.copy()
                partial.remove("--stage1-parent-checkpoint")
                partial.remove("stage1.pt")
                partial[partial.index("--out") + 1] = str(root / "partial-out")
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "pinned fit receipt"):
                    module.main(partial)
                load_child.assert_called_once()
                load_qualified.assert_not_called()
                sys.modules.pop(module_name, None)

    def test_routes_child_only_with_complete_receipt_and_stage1_pins(self):
        result = resolve_direction_fit_parent(
            **self.kwargs, stage1_parent_checkpoint="stage1.pt", fit_receipt="fit.json",
            fit_receipt_sha256="3" * 64,
            load_qualified_parent=self.qualified, load_language_child=self.child)
        self.assertEqual(result, "language-child")
        self.child.assert_called_once_with(
            "candidate.pt", stage1_checkpoint="stage1.pt", receipt="fit.json",
            receipt_sha256="3" * 64, report="report.json", report_sha256="1" * 64,
            qualification="qualification.json", qualification_sha256="2" * 64,
            snapshot="snapshot-dir", device="cpu", dtype="bfloat16")
        self.qualified.assert_not_called()

    def test_partial_child_pins_fail_before_either_loader(self):
        partials = (
            {"fit_receipt": "fit.json"},
            {"stage1_parent_checkpoint": "stage1.pt", "fit_receipt": "fit.json"},
            {"stage1_parent_checkpoint": "stage1.pt", "fit_receipt_sha256": "3" * 64},
        )
        for extra in partials:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "pinned fit receipt"):
                resolve_direction_fit_parent(
                    **self.kwargs, **extra,
                    load_qualified_parent=self.qualified, load_language_child=self.child)
        self.qualified.assert_not_called()
        self.child.assert_not_called()

    def test_child_requires_strict_local_snapshot(self):
        kwargs = {**self.kwargs, "parent_snapshot": None}
        with self.assertRaisesRegex(ValueError, "local snapshot"):
            resolve_direction_fit_parent(
                **kwargs, stage1_parent_checkpoint="stage1.pt", fit_receipt="fit.json",
                fit_receipt_sha256="3" * 64,
                load_qualified_parent=self.qualified, load_language_child=self.child)
        self.qualified.assert_not_called()
        self.child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
