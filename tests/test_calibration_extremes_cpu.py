"""Pure-Python numerical boundary checks; no Torch or data imports."""

import math
import unittest

from decide import Decider
from decide.backends.base import Plan, ScoreResult
from decide.calibration import FeatureTemperature, OptionCountTemperature, CalibrationNotServable, softmax


class ExtremeBackend:
    def revision(self):
        return {"revision_id": "synthetic-extremes-v1"}

    def plan(self, request):
        return Plan(1, 1, 1)

    def score(self, request, check_deadline):
        check_deadline()
        return ScoreResult({"q": {"a": 1e308, "b": 0.0}}, 1, 1)


class CalibrationExtremesTests(unittest.TestCase):
    def test_small_temperature_and_huge_finite_logits_are_not_nan(self):
        self.assertEqual(softmax([1e308, 0.0], .05), [1.0, 0.0])
        self.assertEqual(softmax([-1e308, -1e308], .05), [.5, .5])

    def test_large_temperature_preserves_opposite_sign_extreme_gap(self):
        expected = softmax([1.0, -1.0], 1.0)
        self.assertEqual(softmax([1e308, -1e308], 1e308), expected)
        self.assertTrue(all(math.isfinite(x) for x in expected))

    def test_served_adapter_has_finite_distributions(self):
        parent = Decider(ExtremeBackend(), calibration=OptionCountTemperature(t0=.05, t_min=.05))
        response = parent.decide({"model": "synthetic", "state": "x", "questions": {
            "q": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}}})
        self.assertEqual(response["ssdm"]["status"], "ok")
        self.assertEqual(response["ssdm"]["served_probabilities"]["q"], {"a": 1.0, "b": 0.0})
        self.assertEqual(response["answers"]["q"]["confidence"], 1.0)

    def test_manual_perturbation_cannot_use_identity_proof_exemption(self):
        base = FeatureTemperature(t_min=.07)
        shifted = FeatureTemperature(t_min=.07, w=(base.w[0] + 5e-13, 0.0, 0.0))
        self.assertTrue(base.is_identity)
        self.assertEqual(base.temperature_for(2, {"evidence_tokens": 10}), 1.0)
        self.assertFalse(shifted.is_identity)
        self.assertEqual(shifted.identity()["status"], "unfitted_manual")
        with self.assertRaises(CalibrationNotServable):
            shifted.assert_servable()


if __name__ == "__main__":
    unittest.main()
