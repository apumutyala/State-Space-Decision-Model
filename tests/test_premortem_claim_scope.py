"""Guard PR #91 claim scope and retained-cache/transfer assumptions (LESSONS L-25/L-39)."""

import re
import unittest
from pathlib import Path


def unsupported_claims(text):
    patterns = (
        r"can only be equal or weaker",
        r"no head reaches.{0,30}no Q9-AF variant will",
        r"rank holds between stock and 1-layer features, it should hold on Q9-AF",
        r"20k rows\s*≈\s*0\.6 GB",
        r"one\*\* pair of real cells.{0,150}before trusting the ranking",
    )
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


class PremortemClaimScopeTest(unittest.TestCase):
    def test_original_absolute_claims_are_rejected(self):
        for claim in (
            "Conversion can only be equal or weaker.",
            "If no head reaches ~50, no Q9-AF variant will.",
            "If a recipe's rank holds between stock and 1-layer features, it should hold on Q9-AF.",
            "fork/final-state heads need only fork + q_end + e, so 20k rows ≈ 0.6 GB.",
            "calibrate with **one** pair of real cells (best and worst) before trusting the ranking.",
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

    def test_cache_formula_and_transfer_screening_are_explicit(self):
        doc = (Path(__file__).resolve().parents[1] / "PREMORTEM.md").read_text(encoding="utf-8")
        for requirement in (
            "retains the complete `HeadInput` even for fork/final-state heads",
            "4 * D * (T + 2 + 2 * K) + 2 * T + K",
            "tensor.numel() * tensor.element_size()",
            "bf16 autocast does not make these retained features bf16",
            "this is not a RAM-fit claim",
            "best/worst two-recipe pair is **screening only**",
            "requires ≥ 4 matched recipes with both proxy and real",
            "Ranking transfer remains provisional until those measurements pass",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, doc)


if __name__ == "__main__":
    unittest.main()
