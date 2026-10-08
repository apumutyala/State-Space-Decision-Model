"""The shared serving-pin validator also gates the trainable Stage 1 path."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires the repository Torch runtime")
class NativeParentPinValidatorTests(unittest.TestCase):
    def setUp(self):
        from tempfile import TemporaryDirectory

        from language_growth.native_parent import NativeParentPin
        from language_growth.tests.test_native_parent_m2_bundle import fixture

        self.directory = TemporaryDirectory(prefix="native-parent-pin-validator-")
        self.addCleanup(self.directory.cleanup)
        self.artifact, self.loaded, report = fixture(Path(self.directory.name))
        self.pin = NativeParentPin.from_m2_report(report)

    def test_valid_serving_identity_passes_without_freezing(self):
        from language_growth.native_parent import validate_loaded_native_parent

        decider = validate_loaded_native_parent(self.loaded, self.pin)
        self.assertIsNotNone(decider)
        self.assertTrue(self.loaded.backend.model.training)
        self.assertTrue(self.loaded.backend.model.grad_enabled)

    def test_every_serving_identity_component_is_bound(self):
        from language_growth.native_parent import validate_loaded_native_parent

        mutations = {
            "checkpoint": lambda loaded: loaded.backend.checkpoint.update(path="other-artifact.pt"),
            "calibration": lambda loaded: replace(
                loaded, calibration=replace(loaded.calibration, w=(.5 + 1e-12, .03, .1))),
            "runtime": lambda loaded: replace(loaded, backend_max_input_tokens=2048),
            "provenance": lambda loaded: replace(loaded, enable_experimental_provenance=True),
            "delta_flag": lambda loaded: replace(loaded, enable_v0_1_deltas=True),
            "backbone_config": lambda loaded: setattr(
                loaded.backend.model.backbone.model.config, "implementation", "drift"),
        }
        for name, mutate in mutations.items():
            with self.subTest(component=name):
                # Rebuild the fixture so a prior mutation cannot affect this case.
                from language_growth.native_parent import NativeParentPin
                from language_growth.tests.test_native_parent_m2_bundle import fixture

                artifact, loaded, report = fixture(Path(self.directory.name))
                pin = NativeParentPin.from_m2_report(report)
                mutated = mutate(loaded) or loaded
                with self.assertRaises(ValueError):
                    validate_loaded_native_parent(mutated, pin)
                self.assertTrue(mutated.backend.model.training)
                self.assertTrue(mutated.backend.model.grad_enabled)


if __name__ == "__main__":
    unittest.main()
