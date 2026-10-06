"""CPU L planner replay through D's real adapter, using synthetic oracle logits."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from language_growth.integration_smoke import run


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = Path(os.environ.get("SSDM_D_ADAPTER_ROOT", ROOT.parent / "ssdm-codex-l0-adapter-review"))
FIXTURES = ROOT / "fixtures" / "language_growth"


@unittest.skipUnless((ADAPTER / "decide/backends/replay.py").is_file()
                     and (FIXTURES / "catalog.json").is_file(),
                     "requires the separate D checkout and synthetic development fixture pack")
class ReplayIntegrationTests(unittest.TestCase):
    def test_oracle_recordings_replay_same_checked_meaning_and_miss_on_change(self):
        report = run(ADAPTER, FIXTURES)
        self.assertTrue(report["passed"])
        self.assertEqual([(row["status"], row["replay_status"]) for row in report["rows"]],
                         [("accepted", "accepted"), ("accepted", "accepted"),
                          ("kernel_rejected", "kernel_rejected")])
        for row in report["rows"]:
            self.assertTrue(row["replay_matches"])
            self.assertTrue(row["semantic_matches"])
            self.assertTrue(row["response_matches"])
            self.assertTrue(row["request_ids_differ"])
            self.assertEqual(row["recorded_fingerprints"], row["replay_fingerprints"])
            self.assertEqual(row["checked_meaning"], row["replay_meaning"])
            self.assertEqual(row["text"], row["replay_text"])
        for miss in report["rows"][0]["misses"].values():
            self.assertEqual(miss, {"planner_status": "provider_rejected",
                                    "adapter_status": "unsupported",
                                    "adapter_error_code": "replay_miss"})

    def test_scaled_replay_logits_preserve_program_but_fail_probability_equivalence(self):
        baseline = run(ADAPTER, FIXTURES)
        self.assertTrue(baseline["passed"])
        from decide.backends import ReplayBackend

        original_score = ReplayBackend.score

        def scaled_score(backend, request, check_deadline):
            scored = original_score(backend, request, check_deadline)
            scored.logits = {qid: {option: logit * 0.5 for option, logit in row.items()}
                             for qid, row in scored.logits.items()}
            return scored

        with patch.object(ReplayBackend, "score", scaled_score):
            changed = run(ADAPTER, FIXTURES)
        self.assertFalse(changed["passed"])
        self.assertTrue(all(row["semantic_matches"] for row in changed["rows"]))
        self.assertTrue(all(not row["response_matches"] for row in changed["rows"]))
        self.assertTrue(all(not row["replay_matches"] for row in changed["rows"]))


if __name__ == "__main__":
    unittest.main()
