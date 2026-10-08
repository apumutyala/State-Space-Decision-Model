"""CPU fake-cache contract tests; no pretrained checkpoint or model load."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from decide.adapter import Decider
from decide.backends.base import BackendUnsupported
from decide.backends.state_fork import RWKV7StateOps
from decide.calibration import OptionCountTemperature
from decide.prompt import Template
from heads.layout import ByteEncoder, build_row
from heads.layout import LAYOUT_VERSION, TEMPLATES
from language_growth.causal_layout_backend import RWKV7LayoutCausalBackend
from language_growth.parent_bridge import decision_request
from language_growth.planner import plan_checked_answer
from language_growth.semantic_state_intervention import (
    InterventionError, RegisteredStateDirection, RWKV7SemanticStateIntervention,
)


TEMPLATE = "jrt-plain-v1"
PARENT = {"model_id": "fake-rwkv7", "snapshot": "fake-test-only"}


class FakeLayer:
    def __init__(self):
        self.recurrent_states = {i: torch.zeros(1, 1) for i in range(3)}
        self.is_recurrent_states_initialized = dict.fromkeys(range(3), True)
        self.has_previous_state = dict.fromkeys(range(3), True)


class FakeCache:
    def __init__(self):
        self.config = {"shape": "causal-layout-test"}
        self.layers = [FakeLayer()]


class FakeRWKVModel:
    def __init__(self, *, uniform: bool = False):
        self.marker = torch.tensor([1.0])
        self.config = {"model_type": "fake-rwkv7-layout", "vocab_size": 260, "wkv_implementation": "fake"}
        self.uniform = uniform
        self.forward_calls = []

    def state_dict(self):
        return {"marker": self.marker}

    def __call__(self, *, input_ids, state=None, use_cache, logits_to_keep):
        self.forward_calls.append((tuple(input_ids.shape), input_ids.device.type, logits_to_keep))
        if state is None:
            state = FakeCache()
        all_rows = []
        vocab = torch.arange(260, dtype=torch.float32)
        for batch in range(input_ids.shape[0]):
            token_logits = []
            for token in input_ids[batch].tolist():
                for slot, tensor in state.layers[0].recurrent_states.items():
                    state.layers[0].recurrent_states[slot] = tensor + float(token + 1) * (slot + 1)
                if self.uniform:
                    value = torch.zeros_like(vocab)
                else:
                    state_value = sum(x.sum() for x in state.layers[0].recurrent_states.values())
                    value = vocab * (state_value * 1e-7) + torch.sin(vocab + state_value) * 1e-3
                token_logits.append(value)
            all_rows.append(torch.stack(token_logits))
        logits = torch.stack(all_rows)
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return type("Output", (), {"logits": logits, "state": state})()


def fixtures():
    """Tiny synthetic fixture so ByteEncoder inputs remain below the 2048-token protocol cap."""
    context, policy, registry = "tiny-context", "tiny-public", "tiny-registry"
    host = {"context_ref": context, "policy_ref": policy, "registry_ref": registry, "audience": "public",
            "query_constants": {}, "unmet_obligations": [], "limits": {
                "max_program_nodes": 8, "max_questions": 1, "max_options_per_question": 8,
                "max_proposal_calls": 4, "max_repair_calls": 0, "max_repair_attempts": 0,
            }, "provider": {"input_context_ref": context, "state_handle": None,
                           "state_history_context_refs": []}}
    catalog = {
        "schema": "ssdm.causal-layout-test-catalog.v1",
        "sources": {"tiny-source": {"premise_admission": "accepted"}},
        "records": [{"record_ref": "record-1", "version": 1, "entity_ref": "Aster",
                      "source_ref": "tiny-source", "audience": "public",
                      "epistemic_class": "authoritative_structured_premise", "freshness": "current",
                      "conflicts": [], "fields": {"count": {
                          "value": 7, "value_type": "integer", "unit": "item",
                          "temporal_scope": "2026-10", "quantity_kind": "count"}}}],
        "snapshots": {context: {"audience": "public", "record_refs": ["record-1"]}},
        "policies": {policy: {"status": "current", "audience": "public", "forbidden_audiences": ["private"]}},
        "registries": {registry: {"allowed_ops": ["ReadFact", "Assert", "ExplainDerivation"],
                                   "registered_fields": ["count"], "field_predicates": {"count": "Count"},
                                   "comparators": [], "ask_categories": [], "style_ids": ["plain"]}},
    }
    return {"question": "What is Aster's count in October 2026?", "host": host}, catalog


def backend_for(host, *, model=None, encoder=None, uniform=False, edit_sign=None,
                direction_value=1.0, max_context_tokens=2048):
    encoder = encoder or ByteEncoder()
    model = model or FakeRWKVModel(uniform=uniform)
    wrapper = RWKV7SemanticStateIntervention(
        state_ops=RWKV7StateOps(model), parent_identity=PARENT,
        tokenizer_identity=encoder.identity(), implementation_sha256="a" * 64,
        backend_identity="fake-rwkv7-state-ops:v1", template_id=TEMPLATE,
        context_refs=[host["context_ref"]], registrations=[],
        capture_identity={"layout_version": LAYOUT_VERSION, "template_identity": TEMPLATES[TEMPLATE].identity()},
    )
    registration = RegisteredStateDirection.freeze(
        edit_id="test-direction", runtime_identity_sha256=wrapper.runtime_identity_sha256,
        discovery_cohort_sha256="b" * 64, validation_cohort_sha256="c" * 64,
        scale=0.1, tensors={(0, 0): torch.full((1, 1), direction_value)},
    )
    wrapper.register(registration)
    return RWKV7LayoutCausalBackend(
        intervention=wrapper, encoder=encoder, encoder_identity=encoder.identity(), template_id=TEMPLATE,
        parent_identity=PARENT, vocab_size=260,
        edit_id=None if edit_sign is None else registration.edit_id, edit_sign=edit_sign,
        max_context_tokens=max_context_tokens,
    ), model, wrapper


def make_request(case, catalog, *, task=None, options=None, qid="slot-test-choice"):
    host = copy.deepcopy(case["host"])
    return decision_request(
        task_input=task or case["question"], host=host, catalog=catalog,
        model="fake-rwkv7", parent_revision="unbound-test-revision", request_id="backend-test-request",
        question_id=qid, question="Select one option.", options=options or {"a": "A", "b": "B"},
    )[0], host


class CausalLayoutBackendTests(unittest.TestCase):
    def test_split_option_scores_match_uninterrupted_forward_for_multi_token_options(self):
        case, catalog = fixtures()
        request, host = make_request(case, catalog, options={"a": "alpha", "b": "beta", "c": "gamma"})
        backend, model, _wrapper = backend_for(host)
        accounted_calls = []
        result = backend.score(request, lambda: accounted_calls.append(True))
        row = build_row(request, "slot-test-choice", backend.encoder, template=TEMPLATE,
                        layout="ordered", empty_state="error")
        prefix_end = row.state_span[1]
        suffix = row.ids[prefix_end:]
        full_scores = {}
        for option_id, cont in zip(row.option_ids, row.continuations):
            full_input = row.ids + cont[:-1]
            direct = FakeRWKVModel()(input_ids=torch.tensor([full_input]), state=None,
                                     use_cache=True, logits_to_keep=len(full_input)).logits
            positions = torch.arange(len(cont)) + len(row.ids) - 1
            expected = torch.log_softmax(direct[0].index_select(0, positions), dim=-1)
            full_scores[option_id] = float(expected.gather(1, torch.tensor(cont)[:, None]).sum().item())
        self.assertEqual(result.inference_calls, 1 + len(row.option_ids))
        self.assertEqual(len(accounted_calls), result.inference_calls)
        self.assertEqual(set(result.logits["slot-test-choice"]), set(row.option_ids))
        self.assertEqual(list(result.logits["slot-test-choice"]), row.option_ids)
        self.assertEqual(result.total_input_tokens, prefix_end + sum(len(suffix) + len(cont) - 1
                                                                      for cont in row.continuations))
        for option_id in row.option_ids:
            self.assertAlmostEqual(result.logits["slot-test-choice"][option_id], full_scores[option_id], places=5)
        self.assertEqual(model.forward_calls[0][0][0], 1)
        self.assertEqual(len(model.forward_calls), result.inference_calls)
        self.assertTrue(all(device == "cpu" and shape[0] == 1 for shape, device, _ in model.forward_calls))
        self.assertGreater(len(suffix), 0)

    def test_context_empty_and_oversize_refuse_before_prefill(self):
        case, catalog = fixtures()
        request, host = make_request(case, catalog)
        backend, model, _ = backend_for(host)
        changed = copy.deepcopy(request)
        changed["ssdm"]["context_ref"] = "unregistered"
        with self.assertRaises(BackendUnsupported) as context:
            backend.plan(changed)
        self.assertEqual(context.exception.code, "unsupported_context")
        empty = copy.deepcopy(request)
        empty["state"] = ""
        with self.assertRaises(BackendUnsupported) as context:
            backend.plan(empty)
        self.assertEqual(context.exception.code, "unsupported_layout")
        large, _ = make_request(case, catalog, task="x" * 3000)
        with self.assertRaises(BackendUnsupported) as context:
            backend.plan(large)
        self.assertEqual(context.exception.code, "context_exceeded")
        self.assertEqual(model.forward_calls, [])

    def test_selected_direction_bytes_are_part_of_backend_revision(self):
        case, catalog = fixtures()
        _request, host = make_request(case, catalog)
        first, _, _ = backend_for(host, edit_sign=1, direction_value=1.0)
        second, _, _ = backend_for(host, edit_sign=1, direction_value=2.0)
        self.assertNotEqual(first.revision()["revision_id"], second.revision()["revision_id"])

    def test_mutated_registered_direction_refuses_before_prefill(self):
        case, catalog = fixtures()
        request, host = make_request(case, catalog)
        backend, model, wrapper = backend_for(host, edit_sign=1)
        wrapper._registrations["test-direction"].tensors[(0, 0)].add_(1)
        with self.assertRaises(BackendUnsupported) as context:
            backend.plan(request)
        self.assertEqual(context.exception.code, "identity_mismatch")
        self.assertEqual(model.forward_calls, [])

    def test_encoder_template_and_prompt_id_drift_refuse_before_prefill(self):
        class MutableEncoder:
            def __init__(self, *, invalid=False):
                self.base = ByteEncoder()
                self.ident = self.base.identity()
                self.invalid = invalid

            def identity(self):
                return self.ident

            def __call__(self, text, trusted=True):
                if self.invalid and trusted and "SCHEMA FIELDS:" in text:
                    return [260]
                return self.base(text, trusted)

        case, catalog = fixtures()
        request, host = make_request(case, catalog)
        encoder = MutableEncoder()
        backend, model, _ = backend_for(host, encoder=encoder)
        encoder.ident = {"kind": "mutated"}
        with self.assertRaises(BackendUnsupported) as context:
            backend.plan(request)
        self.assertEqual(context.exception.code, "identity_mismatch")
        self.assertEqual(model.forward_calls, [])

        encoder = MutableEncoder()
        backend, model, _ = backend_for(host, encoder=encoder)
        changed_templates = {TEMPLATE: Template(TEMPLATE, "changed {system}", "\nANSWER:", "\n")}
        with patch("language_growth.causal_layout_backend.TEMPLATES", changed_templates):
            with self.assertRaises(BackendUnsupported) as context:
                backend.plan(request)
        self.assertEqual(context.exception.code, "identity_mismatch")
        self.assertEqual(model.forward_calls, [])

        encoder = MutableEncoder(invalid=True)
        backend, model, _ = backend_for(host, encoder=encoder)
        with self.assertRaises(BackendUnsupported) as context:
            backend.plan(request)
        self.assertEqual(context.exception.code, "unsupported_layout")
        self.assertEqual(model.forward_calls, [])

    def test_decider_call_budget_rejects_before_prefill(self):
        case, catalog = fixtures()
        host = copy.deepcopy(case["host"])
        host["limits"]["max_inference_calls"] = 1
        backend, model, _ = backend_for(host)
        decider = Decider(backend)
        request = decision_request(
            task_input=case["question"], host=host, catalog=catalog,
            model="fake-rwkv7", parent_revision=decider.revision()["revision_id"],
            request_id="budget-test", question_id="slot-budget-choice", question="Select.",
            options={"a": "A", "b": "B"},
        )[0]
        response = decider.decide(request)
        self.assertEqual(response["ssdm"]["status"], "exhausted")
        self.assertEqual(response["ssdm"]["error"]["code"], "inference_calls_exceeded")
        self.assertEqual(model.forward_calls, [])

    def test_decider_planner_kernel_renderer_is_the_only_release_path(self):
        case, catalog = fixtures()
        host = copy.deepcopy(case["host"])
        host["limits"].update(max_proposal_calls=4, max_repair_calls=0, max_repair_attempts=0)
        backend, model, wrapper = backend_for(host, uniform=True, edit_sign=1)
        decider = Decider(backend)
        revision = decider.revision()["revision_id"]
        expected = {"calibration": decider.calibration.identity(),
                    "policy_masking": "none (served covers every supplied option)"}
        result = plan_checked_answer(case["question"], host, catalog, decider,
                                     model="fake-rwkv7", parent_revision=revision,
                                     expected_served_config=expected)
        self.assertEqual(result.status, "accepted")
        self.assertIsInstance(result.text, str)
        self.assertIsNotNone(result.program)
        self.assertEqual(result.program["schema"], "ssdm.semantic_program.v0.1")
        self.assertNotIn("raw_logits", result.text)
        self.assertEqual(wrapper._state_ops.model, model)
        self.assertGreater(result.forward_calls, 0)


if __name__ == "__main__":
    unittest.main()
