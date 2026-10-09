"""No-model tests of strict qualified-parent versus fitted-child CLI routing."""
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

from language_growth.direction_fit_admission import resolve_direction_fit_parent


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
            discovery.write_text(json.dumps([{"split": "development_train"}]), encoding="utf-8")
            validation.write_text(json.dumps([{"split": "development_validation"}]), encoding="utf-8")
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
            bridge_module.decision_request = Mock()
            bridge_module.legacy_request_fingerprint = Mock()
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
