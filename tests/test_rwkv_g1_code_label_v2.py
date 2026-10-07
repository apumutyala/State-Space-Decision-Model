"""CPU scripted-state parity/accounting; no model/checkpoint construction or real forwards."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from decide.backends.base import BackendUnsupported
from decide.backends.hf_readout import ModelPlugin
from decide.backends.rwkv_g1_code_label import G1CodeLabelBackend, G1_REVISION, render_code_labels
from decide.backends.rwkv_g1_code_label_v2 import (FORK_TOKEN_BUDGET, MAX_FORK_ROWS, G1CodeLabelV2Backend,
    TEMPLATE_ID, render_all_field_code_labels)
from decide.backends.rwkv_g1_filtered_codes import PROOF_CONTEXT, TOKENIZER_SHA256, VOCAB_SIZE, load_frozen_table
from decide.backends.state_fork import common_prefix_len, fork_plan
from decide.prompt import TEMPLATES, build_prompt_labelled
from decide.provenance import validate_provenance


class ScriptedStateOps:
    def __init__(self):
        self.calls = []
        self.prefill_states = []

    @staticmethod
    def logits(sums):
        out = torch.zeros((*sums.shape, VOCAB_SIZE), device="cpu")
        out[..., 300] = sums.float() / 97
        out[..., 301] = (sums % 7).float() / 7
        out[..., 3967] = (sums % 11).float() / 11
        return out

    def prefill(self, ids):
        sums = ids.cumsum(1) % 97
        state = sums[:, -1].clone()
        self.calls.append(("prefix", tuple(ids.shape)))
        self.prefill_states.append((state, state.clone()))
        return self.logits(sums[:, -1:]), state

    def extend(self, state, ids, n_last):
        sums = (ids.cumsum(1) + state.expand(ids.shape[0])[:, None]) % 97
        self.calls.append(("pack", tuple(ids.shape)))
        return self.logits(sums[:, -n_last:]), sums[:, -1].clone()


def plugin():
    encoded = {" " + c.code: c.token_id for c in load_frozen_table()}
    decoded = {token: text for text, token in encoded.items()}
    def encode(text, trusted=True):
        if text == PROOF_CONTEXT:
            return [55000]
        if text.startswith(PROOF_CONTEXT) and text[len(PROOF_CONTEXT):] in encoded:
            return [55000, encoded[text[len(PROOF_CONTEXT):]]]
        if text in encoded:
            return [encoded[text]]
        if "<|endoftext|>" in text:
            return [0]
        return [10000 + int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 40000]
    ops = ScriptedStateOps()
    full_calls = []
    def full(ids, n_last):
        full_calls.append(tuple(ids.shape))
        sums = ids.cumsum(1) % 97
        return ops.logits(sums[:, -n_last:])
    plug = ModelPlugin("rwkv7", "scripted-state-only", None, encode, full,
        files={"tokenizer": TOKENIZER_SHA256, "weights": "not-loaded"}, notes={"vendored_code": G1_REVISION},
        special_ids=frozenset({0}), dtype="float32", device="cpu", max_context=25600, state_ops=ops)
    return plug, lambda ids: "".join(decoded[i] for i in ids), ops, full_calls


def request(k=2, n_questions=8):
    return {"model": "synthetic", "state": "exact\n\nraw\r\nstate " + "long shared context " * 200,
            "questions": {f"q{i:02}": {"type": "choice", "instructions": f"Choose {i}",
                "criteria": {f"original-{j:03}": f"Description {j}" for j in range(k)}} for i in range(n_questions)}}


def schema_state_segments(labelled):
    stop = next(i for i, (text, label) in enumerate(labelled)
                if label == "template" and text.startswith("\nEND STATE\n\nQUESTION"))
    return labelled[:stop]


def test_all_fields_make_schema_state_prefix_identical_and_non_template_exact():
    plug, decode, _, _ = plugin()
    backend = G1CodeLabelV2Backend(plug, tokenizer_decode=decode)
    r = request()
    before = copy.deepcopy(r)
    rendered = [render_all_field_code_labels(r, qid) for qid in r["questions"]]
    assert all(schema_state_segments(x) == schema_state_segments(rendered[0]) for x in rendered)
    for qid, labelled in zip(r["questions"], rendered):
        base = build_prompt_labelled(TEMPLATES["jrt-rwkv-g1-v1"], r, qid)
        assert [s for s in labelled if s[1] != "template"] == [s for s in base if s[1] != "template"]
        assert len([t for t, label in labelled if label == "template" and t.startswith("[ANSWER CODE ")]) == 16
    rows = backend._rows(r)
    prompts = {row["qid"]: row["prompt"] for row in rows}
    prefix = backend._encode([(t, label == "template") for t, label in schema_state_segments(rendered[0])])
    assert common_prefix_len(list(prompts.values())) >= len(prefix)
    assert all(prompt[:len(prefix)] == prefix for prompt in prompts.values())
    assert r == before


def test_fork_recompute_scores_match_on_identical_v2_prompts_with_lower_volume():
    plug, decode, ops, full_calls = plugin()
    fork = G1CodeLabelV2Backend(plug, tokenizer_decode=decode)
    full = G1CodeLabelV2Backend(plug, tokenizer_decode=decode, fork=False)
    r = request()
    plan = fork.plan(r)
    a = fork.score(r, lambda: None)
    assert not full_calls
    b = full.score(r, lambda: None)
    assert a.logits.keys() == b.logits.keys()
    for q in a.logits:
        for option in a.logits[q]:
            assert a.logits[q][option] == pytest.approx(b.logits[q][option], abs=1e-6)
    assert a.total_input_tokens == plan.total_input_tokens < b.total_input_tokens / 2
    assert a.inference_calls == plan.inference_calls == len(ops.calls)
    assert all(shape[0] <= MAX_FORK_ROWS for kind, shape in ops.calls if kind == "pack")
    assert all(torch.equal(state, before) for state, before in ops.prefill_states)
    rows = fork._rows(r)
    _, calls = fork_plan(rows, MAX_FORK_ROWS, FORK_TOKEN_BUDGET)
    padded = sum(c.tokens if c.kind == "prefix" else c.width * len(c.tails) for c in calls)
    assert padded == sum(b * t for _, (b, t) in ops.calls)
    assert fork.revision()["revision_id"] != full.revision()["revision_id"]


def test_provenance_splits_remain_exact_and_shared():
    r = request()
    text = r["state"]
    r["ssdm"] = {"provenance": {"schema": "ssdm.provenance.v0", "serializer": "synthetic",
        "segments": {"/state": [{"start": 0, "end": 2, "trust": "instruction", "origin": "host_template"},
                                  {"start": 2, "end": len(text), "trust": "data", "origin": "record"}]}}}
    provenance = validate_provenance(r, r["ssdm"]["provenance"])
    prefixes = []
    for qid in r["questions"]:
        a = render_all_field_code_labels(r, qid, provenance)
        b = build_prompt_labelled(TEMPLATES["jrt-rwkv-g1-v1"], r, qid, provenance)
        assert [s for s in a if s[1] != "template"] == [s for s in b if s[1] != "template"]
        prefixes.append(schema_state_segments(a))
    assert all(prefix == prefixes[0] for prefix in prefixes)


def test_every_k_1_through_255_and_53_supports_all_original_ids():
    plug, decode, ops, _ = plugin()
    backend = G1CodeLabelV2Backend(plug, tokenizer_decode=decode)
    table = load_frozen_table()
    for k in range(1, 256):
        r = request(k, 2)
        labelled = render_all_field_code_labels(r, "q00")
        assert len([t for t, label in labelled if label == "template" and t.startswith("[ANSWER CODE ")]) == 2 * k
        assert len(backend.codebook.option_token_ids(r["questions"]["q00"])) == k
        assert backend.parse_exact_token_ids(r["questions"]["q00"], [table[k - 1].token_id]) == f"original-{k - 1:03}"
    r = request(53, 2)
    plan = backend.plan(r)
    result = backend.score(r, lambda: None)
    assert result.inference_calls == plan.inference_calls
    assert all(len(result.logits[q]) == 53 for q in r["questions"])
    assert all(shape[0] <= 4 for kind, shape in ops.calls if kind == "pack")


def test_v1_and_default_remain_distinct_and_unchanged():
    plug, decode, _, _ = plugin()
    v1 = G1CodeLabelBackend(plug, tokenizer_decode=decode)
    v2 = G1CodeLabelV2Backend(plug, tokenizer_decode=decode)
    r = request()
    assert len([t for t, label in render_code_labels(r, "q00") if label == "template" and t.startswith("[ANSWER CODE ")]) == 2
    assert v1.revision()["prompt_template"]["id"] == "jrt-rwkv-g1-code-label-v1" and not v1.fork
    assert v2.revision()["prompt_template"]["id"] == TEMPLATE_ID and v2.fork
    assert set(TEMPLATES) == {"jrt-chatml-v1", "jrt-rwkv-g1-v1", "jrt-plain-v1"}


@pytest.mark.parametrize("kwargs", [{"fork_max_rows": 8}, {"fork_max_rows": True}, {"fork_token_budget": 8192}])
def test_pack_config_cannot_exceed_registered_cap(kwargs):
    plug, decode, _, _ = plugin()
    with pytest.raises(ValueError, match="conservative cap"):
        G1CodeLabelV2Backend(plug, tokenizer_decode=decode, **kwargs)


def test_missing_state_operations_fail_without_fallback():
    plug, decode, _, full_calls = plugin()
    plug.state_ops = None
    with pytest.raises(ValueError, match="no recompute fallback"):
        G1CodeLabelV2Backend(plug, tokenizer_decode=decode)
    assert not full_calls


def test_nonfinite_state_output_is_refused():
    plug, decode, ops, full_calls = plugin()
    ops.prefill = lambda ids: (torch.full((1, 1, VOCAB_SIZE), float("nan")), None)
    backend = G1CodeLabelV2Backend(plug, tokenizer_decode=decode)
    with pytest.raises(BackendUnsupported, match="nonfinite"):
        backend.score(request(), lambda: None)
    assert not full_calls


def test_committed_exact_volume_sums_preserve_rows_options_and_admission():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "receipts/2026-10-07-D-code-label-v2-fork/dev-volume.json").read_text(encoding="utf-8"))
    assert result["input_rows_sha256"] == "406d40e6891690fad48fb82252c44b89a5337c603ff79e03ceb11fffc4eb28f2"
    assert result["rows"] == len(result["per_row"]) == 2008
    assert result["volume_script_sha256"] == hashlib.sha256((root / "scripts/volume_rwkv_g1_code_label_v2.py").read_bytes()).hexdigest()
    per = list(result["per_row"].values())
    assert all(row["admitted"]["v1"] == row["admitted"]["v2"] for row in per)
    for arm in ("v1", "v2"):
        for key in ("tokens", "positions", "calls", "n_options"):
            assert sum(row[arm][key] for row in per) == result["totals"]["all_planned"][arm][key]
            assert sum(row[arm][key] for row in per if row["admitted"][arm]) == result["totals"]["admitted"][arm][key]
    totals = result["totals"]["all_planned"]
    assert totals["v2"]["positions"] == 6675792 < totals["v1"]["positions"] / 8
    assert totals["v2"]["calls"] == 14437 > totals["v1"]["calls"] == 8467
    assert totals["v1"]["n_options"] == totals["v2"]["n_options"] == 46807
    assert totals["v2"]["max_pack_rows"] == 4 and totals["v2"]["max_pack_positions"] <= 4096
    assert totals["v2"]["oversized_singleton_packs"] == 0
