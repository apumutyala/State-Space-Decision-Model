"""Guard the known PR #91 quality-ceiling overclaims (LESSONS L-25); CPU/stdlib only."""

import re
import unittest
from pathlib import Path


def unsupported_claims(text):
    patterns = (
        r"can only be equal or weaker",
        r"no head reaches.{0,30}no Q9-AF variant will",
        r"rank holds between stock and 1-layer features, it should hold on Q9-AF",
    )
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


class PremortemClaimScopeTest(unittest.TestCase):
    def test_original_absolute_claims_are_rejected(self):
        for claim in (
            "Conversion can only be equal or weaker.",
            "If no head reaches ~50, no Q9-AF variant will.",
            "If a recipe's rank holds between stock and 1-layer features, it should hold on Q9-AF.",
        ):
            with self.subTest(claim=claim):
                self.assertTrue(unsupported_claims(claim))

    def test_premortem_retains_scope_and_conditional_spend_gate(self):
        doc = (Path(__file__).resolve().parents[1] / "PREMORTEM.md").read_text(encoding="utf-8")
        self.assertEqual(unsupported_claims(doc), [])
        for qualification in (
            "does not establish a monotonic quality bound on downstream training",
            "If no tested head reaches ~50, pause S1 spend",
            "This finite recipe sweep is not an upper bound",
            "Rank stability between stock and 1-layer features is a transfer hypothesis",
            "Re-check the top-2 recipes on Q9-AF features",
        ):
            with self.subTest(qualification=qualification):
                self.assertIn(qualification, doc)


if __name__ == "__main__":
    unittest.main()
