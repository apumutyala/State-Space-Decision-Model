"""Debias controls (CPU): each control detects the bias it is designed for."""

import hashlib

import pytest
from conftest import base_request

from decide import Decider
from decide.backends import OracleBackend
from decide.debias import description_swap, neutral_cyclic, neutral_ids, run_debias_controls
from decide.schema import canonical_options, render


def _h(*parts):
    return int.from_bytes(hashlib.sha256("\x1f".join(parts).encode()).digest()[:4], "big") / 2**32


class PositionBiased(OracleBackend):
    """Ignores meaning; strongly prefers whichever option is presented first."""

    def score(self, request, check_deadline):
        r = super().score(request, check_deadline)
        for qid, q in request["questions"].items():
            for pos, (o, _) in enumerate(canonical_options(q)):
                r.logits[qid][o] = 5.0 if pos == 0 else 0.0
        return r


class DescriptionReader(OracleBackend):
    """Scores by description content only (no id, no position effect)."""

    def score(self, request, check_deadline):
        r = super().score(request, check_deadline)
        for qid, q in request["questions"].items():
            for o, desc in canonical_options(q):
                r.logits[qid][o] = 4.0 * _h(render(desc))
        return r


def test_neutral_ids_sort_in_presentation_order():
    ids = neutral_ids(12)
    assert ids == sorted(ids) and len(set(map(len, ids))) == 1


def test_variant_construction():
    req = base_request()
    swapped, owner = description_swap(req, "status")
    ids = [o for o, _ in canonical_options(req["questions"]["status"])]
    assert list(sorted(swapped["questions"]["status"]["criteria"])) == ids
    for o in ids:
        assert swapped["questions"]["status"]["criteria"][owner[o]] == req["questions"]["status"]["criteria"][o]
    variants = neutral_cyclic(req, "status")
    assert len(variants) == 3
    for r, v, id_of in variants:
        assert sorted(v["questions"]["status"]["criteria"]) == neutral_ids(3)
    with pytest.raises(ValueError):
        description_swap(req, "large")


def test_detects_position_bias():
    rep = run_debias_controls(Decider(PositionBiased()), base_request(), "status")
    assert rep["neutral_cyclic"]["position_bias_tv"] > 0.5
    assert rep["description_swap"]["id_sticky"] and not rep["description_swap"]["follows_description"]
    assert rep["content_free"]["prior_on_sorted_first"] > 0.9


def test_description_reader_has_no_position_bias():
    rep = run_debias_controls(Decider(DescriptionReader()), base_request(), "status")
    assert rep["neutral_cyclic"]["position_bias_tv"] == pytest.approx(0.0, abs=1e-12)
    assert rep["description_swap"]["follows_description"]
    # rotation averaging recovers the per-description preference
    assert rep["neutral_cyclic"]["rotation_averaged_argmax"] == rep["base"]["argmax"]


def test_controls_run_through_hf_readout_on_tiny_cpu_lm():
    from test_equivariance import tiny_plugin

    from decide.backends.hf_readout import HFReadoutBackend

    rep = run_debias_controls(Decider(HFReadoutBackend(tiny_plugin(), "jrt-plain-v1")), base_request(), "status")
    assert 0.0 <= rep["neutral_cyclic"]["position_bias_tv"] <= 1.0
    assert sum(rep["neutral_cyclic"]["rotation_averaged_probabilities"].values()) == pytest.approx(1.0)
