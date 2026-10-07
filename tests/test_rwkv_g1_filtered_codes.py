"""No-model component tests. Fake tokenization asserts mechanics, not real tokenizer parity."""
import copy
from dataclasses import FrozenInstanceError
import json
import math

import pytest

from decide import Decider
from decide.backends.base import BackendUnsupported
from decide.backends.rwkv_g1_filtered_codes import (CODEBOOK_ID, G1FilteredTokenReadout, MAX_K, PROOF_CONTEXT,
    TABLE_PATH, TABLE_SHA256, TOKENIZER_SHA256, TOKENIZER_SOURCE_SHA256, VOCAB_SIZE, load_frozen_table)
from decide.prompt import TEMPLATES
from decide.schema import RequestRejected


def fake_codec():
    table = load_frozen_table()
    encoded = {" " + c.code: c.token_id for c in table}
    decoded = {token: text for text, token in encoded.items()}
    def encode(text):
        if text == PROOF_CONTEXT:
            return [10, 11]
        if text.startswith(PROOF_CONTEXT):
            return [10, 11, encoded[text[len(PROOF_CONTEXT):]]]
        return [encoded[text]]
    return encode, lambda ids: "".join(decoded[i] for i in ids)


def make_component(encode=None, decode=None, **kwargs):
    default_encode, default_decode = fake_codec()
    return G1FilteredTokenReadout(encode or default_encode, decode or default_decode,
        tokenizer_sha256=TOKENIZER_SHA256, tokenizer_source_sha256=TOKENIZER_SOURCE_SHA256, **kwargs)


@pytest.fixture
def component():
    return make_component()


def question(k):
    return {"type": "choice", "criteria": {f"semantic-{i:03}": f"description {i}" for i in reversed(range(k))}}


def test_all_255_token_ids_map_and_parse_to_the_original_semantic_ids(component):
    q = question(MAX_K)
    before = copy.deepcopy(q)
    table = load_frozen_table()
    mapping = component.option_token_ids(q)
    assert len(mapping) == len(set(mapping.values())) == 255
    for i, entry in enumerate(table):
        assert mapping[f"semantic-{i:03}"] == entry.token_id
        assert component.parse_exact_token_ids(q, [entry.token_id]) == f"semantic-{i:03}"
    assert q == before


def test_prefix_collision_A_and_AA_are_distinct_exact_token_answers(component):
    q = question(27)
    assert component.parse_exact_token_ids(q, [300]) == "semantic-000"
    assert component.parse_exact_token_ids(q, [3601]) == "semantic-026"
    for bad in ([300, 300], " AA", ["3601"], [True], [], [0], [3967]):
        with pytest.raises(ValueError):
            component.parse_exact_token_ids(q, bad)


def test_noul_and_score_semantic_mapping_preserves_declared_level_order(component):
    assert component.option_token_ids({"type": "noul"}) == {"true": 300, "false": 301}
    score = {"type": "score", "criteria": ["low", "middle", "high"]}
    assert component.option_token_ids(score) == {"0": 300, "1": 301, "2": 302}
    assert component.parse_exact_token_ids(score, [302]) == "2"


def test_vocabulary_logits_cover_every_option_and_ignore_unoffered_tokens(component):
    q = question(255)
    logits = [0.0] * VOCAB_SIZE
    logits[60000] = 1000.0  # A vocabulary token not assigned to any option.
    logits[3967] = 2.0  # LB is the last frozen code and must remain offered.
    mapped = component.logits_for_options(q, logits)
    probabilities = component.raw_probabilities(q, logits)
    assert len(mapped) == len(probabilities) == 255
    assert mapped["semantic-254"] == 2.0
    assert math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-12)
    assert all(p > 0 for p in probabilities.values())
    assert component.wire_answer_from_logits(q, logits)["choice"] == "semantic-254"


def test_ties_follow_external_semantic_id_rule_and_request_order_is_irrelevant(component):
    q = {"type": "choice", "criteria": {"zeta": None, "alpha": None}}
    reverse = {"type": "choice", "criteria": dict(reversed(list(q["criteria"].items())))}
    logits = [0.0] * VOCAB_SIZE
    assert component.wire_answer_from_logits(q, logits)["choice"] == "alpha"
    assert component.raw_probabilities(q, logits) == component.raw_probabilities(reverse, logits)
    assert component.wire_answer_from_logits({"type": "score", "criteria": ["lo", "hi"]}, logits)["score"] == 0.5
    assert component.wire_answer_from_logits({"type": "noul"}, logits)["noul"] == 0.5


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), True, "0", None])
def test_nonfinite_or_nonreal_logits_fail_closed_even_at_an_unoffered_token(component, bad_value):
    logits = [0.0] * VOCAB_SIZE
    logits[60000] = bad_value
    with pytest.raises(ValueError):
        component.logits_for_options(question(2), logits)


@pytest.mark.parametrize("bad_vector", [[], [0.0] * 3968, "0", {0: 0.0}])
def test_incomplete_or_wrong_logit_vector_shape_is_rejected(component, bad_vector):
    with pytest.raises(ValueError):
        component.logits_for_options(question(2), bad_vector)


@pytest.mark.parametrize("fault", ["multi_token", "context_merge", "decode", "control"])
def test_constructor_requires_all_frozen_codes_and_exact_context_boundaries(fault):
    encode, decode = fake_codec()
    def broken_encode(text):
        if fault == "multi_token" and text == " LB":
            return [300, 301]
        if fault == "context_merge" and text == PROOF_CONTEXT + " LB":
            return [10, 3967]
        return encode(text)
    broken_decode = (lambda ids: "wrong") if fault == "decode" else decode
    with pytest.raises(ValueError, match="proof failed"):
        make_component(broken_encode, broken_decode, special_ids={0, 3967} if fault == "control" else {0})


def test_source_identity_and_frozen_table_drift_fail_closed(tmp_path):
    encode, decode = fake_codec()
    with pytest.raises(ValueError, match="pinned G1k"):
        G1FilteredTokenReadout(encode, decode, tokenizer_sha256="wrong", tokenizer_source_sha256=TOKENIZER_SOURCE_SHA256)
    data = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    data["codes"][-1]["token_id"] = data["codes"][0]["token_id"]
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        make_component(table_path=altered)


def test_no_pruning_over_k_or_duplicate_canonical_semantic_ids(component):
    with pytest.raises(RequestRejected) as rejected:
        component.option_token_ids(question(256))
    assert rejected.value.code == "too_many_options"
    with pytest.raises(ValueError, match="duplicate canonical"):
        component.option_token_ids({"type": "choice", "criteria": {1: "one", "1": "duplicate"}})


def test_component_is_frozen_and_revision_binds_table_without_claiming_model_identity(component):
    with pytest.raises(FrozenInstanceError):
        component._codes = ()
    revision = component.revision()
    assert revision["codebook_id"] == CODEBOOK_ID and revision["codebook_table_sha256"] == TABLE_SHA256
    assert revision["model_id"] is None and revision["renderer_integration"] == "blocked"
    revision["codebook_table_sha256"] = "wrong"
    assert component.revision()["codebook_table_sha256"] == TABLE_SHA256


def test_complete_decide_backend_stays_blocked_and_defaults_are_unchanged(component):
    assert set(TEMPLATES) == {"jrt-chatml-v1", "jrt-rwkv-g1-v1", "jrt-plain-v1"}
    request = {"model": "test", "state": "keep\n\nthese\r\nbytes", "questions": {"q": question(2)}}
    before = copy.deepcopy(request)
    with pytest.raises(BackendUnsupported, match="renderer"):
        component.plan(request)
    with pytest.raises(BackendUnsupported, match="renderer"):
        component.score(request, lambda: pytest.fail("must not start inference"))
    response = Decider(component).decide(request)
    assert response["ssdm"]["status"] == "unsupported"
    assert response["ssdm"]["error"]["code"] == "renderer_integration_blocked"
    assert response["answers"] == {} and request == before
