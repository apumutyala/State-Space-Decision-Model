"""L-41: synthetic counterexamples and scope guards; no models or private inputs."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


RECEIPT_DIR = Path(__file__).resolve().parents[1] / "receipts/2026-10-07-D-order-debias-handoff"
SPEC = importlib.util.spec_from_file_location("order_debias_reconciliation", RECEIPT_DIR / "reconcile.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_invariant_original_endpoint_choice_has_positive_endpoint_excess():
    presented_forward = ["a", "b", "c", "d"]
    presented_reverse = list(reversed(presented_forward))
    choose = lambda presented: "a"  # Original label is chosen regardless of presentation.
    forward_choice, reverse_choice = choose(presented_forward), choose(presented_reverse)
    assert forward_choice == reverse_choice
    excess = AUDIT.endpoint_excess(4, forward_choice == presented_forward[0],
                                  reverse_choice == presented_reverse[0])
    assert excess == 0.25  # Positive despite zero order sensitivity.


@pytest.mark.parametrize("mutation", [
    {"label_scheme": "numeric-prefix/v1"},
    {"probabilities_saved": False},
    {"rows": 160},
    {"run_ids_sha256": "different-row-set"},
    {"condition": "forward"},
])
def test_matched_pair_metadata_guard_rejects_mixed_labels_or_missing_pass_data(mutation):
    forward = {"condition": "forward", "label_scheme": "original", "rows": 2008,
               "run_ids_sha256": "same-row-set", "probabilities_saved": True}
    reverse = {**forward, "condition": "reversed"}
    assert AUDIT.matched_probability_pair_available(forward, reverse)
    reverse.update(mutation)
    assert not AUDIT.matched_probability_pair_available(forward, reverse)


def test_committed_receipt_scope_and_timing_pass_the_guard():
    receipt = json.loads((RECEIPT_DIR / "reconciliation.json").read_text(encoding="utf-8"))
    AUDIT.validate_interpretation(receipt)


@pytest.mark.parametrize("field,value", [
    ("first_choice_statistic", "isolated_position_effect"),
    ("matched_probability_pair_available", True),
    ("probability_averaging_status", "measured"),
    ("new_full_passes_for_symmetric_relabelled_pair", 1),
    ("index_stage_timer_scope", "forward_only"),
    ("forward_latency_source", "stage_seconds_divided_by_rows"),
])
def test_receipt_guard_rejects_reintroduced_interpretation_claims(field, value):
    receipt = json.loads((RECEIPT_DIR / "reconciliation.json").read_text(encoding="utf-8"))
    receipt["interpretation"][field] = value
    with pytest.raises(ValueError):
        AUDIT.validate_interpretation(receipt)


def test_receipt_guard_rejects_stage_time_used_as_mean_request_latency():
    receipt = json.loads((RECEIPT_DIR / "reconciliation.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(receipt)
    bad["forward_request_latency_all"]["mean_ms"] = (
        bad["whole_index_stage_seconds_including_order_diag"] * 1000 / bad["rows"])
    with pytest.raises(ValueError, match="mean request latency"):
        AUDIT.validate_interpretation(bad)
