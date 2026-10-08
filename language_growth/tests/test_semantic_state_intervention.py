"""CPU contract tests with a tiny fake RWKV cache; no pretrained weights are loaded."""

from dataclasses import replace
import unittest

import torch

from decide.backends.state_fork import RWKV7StateOps
from language_growth.semantic_state_intervention import (
    InterventionError,
    RegisteredStateDirection,
    RWKV7SemanticStateIntervention,
)


class FakeLayer:
    def __init__(self):
        self.recurrent_states = {i: torch.zeros(1, 1) for i in range(3)}
        self.is_recurrent_states_initialized = dict.fromkeys(range(3), True)
        self.has_previous_state = dict.fromkeys(range(3), True)


class FakeCache:
    def __init__(self):
        self.config = {"shape": "tiny-test-only"}
        self.layers = [FakeLayer()]


class FakeRWKVModel:
    def __init__(self):
        self.marker = torch.tensor(7.0)
        self.config = {"model_type": "fake-rwkv7", "wkv_implementation": "fake-test"}

    def state_dict(self):
        return {"marker": self.marker}

    def __call__(self, *, input_ids, state=None, use_cache, logits_to_keep):
        if state is None:
            state = FakeCache()
        rows = []
        for batch in range(input_ids.shape[0]):
            token_logits = []
            for token in input_ids[batch].tolist():
                for slot, tensor in state.layers[0].recurrent_states.items():
                    state.layers[0].recurrent_states[slot] = tensor + float(token + 1) * (slot + 1)
                value = sum(x.sum() for x in state.layers[0].recurrent_states.values())
                token_logits.append(torch.stack((value, -value, self.marker.to(value.device))))
            rows.append(torch.stack(token_logits))
        return type("Output", (), {"logits": torch.stack(rows), "state": state})()


def protocol(*, parent="parent-v1", registrations=()):
    model = FakeRWKVModel()
    wrapper = RWKV7SemanticStateIntervention(
        state_ops=RWKV7StateOps(model),
        parent_identity={"model_id": parent, "snapshot": "pinned-test-snapshot"},
        tokenizer_identity={"tokenizer": "tiny-test", "revision": "t1"},
        implementation_sha256="1" * 64,
        backend_identity="rwkv7-test-backend:v1",
        template_id="schema-state-question-v1",
        context_ref="ctx-test-v1",
        registrations=list(registrations),
    )
    return wrapper, model


def direction(wrapper, *, path=(0, 0), values=None, edit_id="assert-explain-v1"):
    values = torch.tensor([[1.0]]) if values is None else values
    return RegisteredStateDirection.freeze(
        edit_id=edit_id,
        runtime_identity_sha256=wrapper.runtime_identity_sha256,
        discovery_cohort_sha256="a" * 64,
        validation_cohort_sha256="b" * 64,
        scale=0.25,
        tensors={path: values},
    )


class SemanticStateInterventionTests(unittest.TestCase):
    def test_preregistered_context_set_reuses_one_direction_and_rejects_unknown_context(self):
        model = FakeRWKVModel()
        wrapper = RWKV7SemanticStateIntervention(
            state_ops=RWKV7StateOps(model),
            parent_identity={"model_id": "parent-v1", "snapshot": "pinned-test-snapshot"},
            tokenizer_identity={"tokenizer": "tiny-test", "revision": "t1"},
            implementation_sha256="1" * 64,
            backend_identity="rwkv7-test-backend:v1",
            template_id="schema-state-question-v1",
            context_refs=["ctx-b", "ctx-a"],
            registrations=[],
        )
        reg = direction(wrapper)
        wrapper.register(reg)
        for context in ("ctx-a", "ctx-b"):
            capture = wrapper.capture([1, 2], context_ref=context)
            edited = wrapper.apply_registered_edit(capture, edit_id=reg.edit_id, sign=1)
            self.assertEqual(wrapper.replay(edited, [3]).logits.shape, (1, 1, 3))
            self.assertEqual(capture.context_ref, context)
        retained = wrapper.capture([1, 2], context_ref="ctx-a")
        wrapper._context_refs = frozenset({"ctx-a", "ctx-b", "ctx-injected"})
        with self.assertRaisesRegex(InterventionError, "runtime identity"):
            wrapper.capture([1], context_ref="ctx-injected")
        with self.assertRaisesRegex(InterventionError, "runtime identity"):
            wrapper.replay(retained, [2])
        wrapper._context_refs = frozenset({"ctx-a", "ctx-b"})
        with self.assertRaisesRegex(InterventionError, "context"):
            wrapper.capture([1], context_ref="ctx-unregistered")

    def test_sham_replay_is_deterministic_and_does_not_mutate_capture(self):
        wrapper, _model = protocol()
        capture = wrapper.capture([2, 3, 4], context_ref="ctx-test-v1")
        before = capture.state_sha256
        a = wrapper.replay(capture, [5, 6])
        b = wrapper.replay(capture, [5, 6])
        torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        direct = FakeRWKVModel()(input_ids=torch.tensor([[2, 3, 4, 5, 6]]), state=None,
                                 use_cache=True, logits_to_keep=5)
        torch.testing.assert_close(a.logits, direct.logits[:, 3:, :], rtol=0, atol=0)
        self.assertEqual(capture.state_sha256, before)

    def test_registered_recurrent_edit_persists_through_replay_and_clones(self):
        wrapper, model = protocol()
        registration = direction(wrapper)
        wrapper.register(registration)
        capture = wrapper.capture([2, 3, 4], context_ref="ctx-test-v1")
        original_state_sha = capture.state_sha256
        original_model_marker = model.marker.clone()
        sham = wrapper.replay(capture, [5, 6])
        edited_capture = wrapper.apply_registered_edit(capture, edit_id=registration.edit_id, sign=1)
        edited = wrapper.replay(edited_capture, [5, 6])
        self.assertNotEqual(edited_capture.state_sha256, capture.state_sha256)
        self.assertNotEqual(edited_capture.state_manifest["layer/0/recurrent/0"]["sha256"],
                            capture.state_manifest["layer/0/recurrent/0"]["sha256"])
        for path in ("layer/0/recurrent/1", "layer/0/recurrent/2"):
            self.assertEqual(edited_capture.state_manifest[path]["sha256"], capture.state_manifest[path]["sha256"])
        self.assertFalse(torch.equal(sham.logits, edited.logits))
        self.assertEqual(capture.state_sha256, original_state_sha)
        torch.testing.assert_close(model.marker, original_model_marker, rtol=0, atol=0)

    def test_unknown_edit_wrong_sign_context_and_parent_identity_reject(self):
        wrapper, _model = protocol()
        reg = direction(wrapper)
        wrapper.register(reg)
        capture = wrapper.capture([1], context_ref="ctx-test-v1")
        with self.assertRaisesRegex(InterventionError, "unknown or mutated"):
            wrapper.apply_registered_edit(capture, edit_id="unregistered", sign=1)
        with self.assertRaisesRegex(InterventionError, "sign"):
            wrapper.apply_registered_edit(capture, edit_id=reg.edit_id, sign=0)
        with self.assertRaisesRegex(InterventionError, "context"):
            wrapper.capture([1], context_ref="other-context")
        other, _ = protocol(parent="different-parent")
        with self.assertRaisesRegex(InterventionError, "identity"):
            other.register(reg)

    def test_shape_nonfinite_and_mutated_directions_fail_closed(self):
        wrapper, _model = protocol()
        mismatch = direction(wrapper, values=torch.ones(1, 2))
        wrapper.register(mismatch)
        capture = wrapper.capture([1, 2], context_ref="ctx-test-v1")
        with self.assertRaisesRegex(InterventionError, "shape/dtype"):
            wrapper.apply_registered_edit(capture, edit_id=mismatch.edit_id, sign=1)

        with self.assertRaisesRegex(InterventionError, "finite, nonzero"):
            direction(wrapper, values=torch.tensor([[float("nan")]]), edit_id="nan")

        wrapper2, _ = protocol()
        mutable = direction(wrapper2, edit_id="mutable")
        wrapper2.register(mutable)
        mutable.tensors[(0, 0)].add_(1)
        capture2 = wrapper2.capture([1], context_ref="ctx-test-v1")
        with self.assertRaisesRegex(InterventionError, "unknown or mutated"):
            wrapper2.apply_registered_edit(capture2, edit_id="mutable", sign=1)

    def test_external_capture_mutation_and_non_cpu_state_are_rejected(self):
        wrapper, _model = protocol()
        capture = wrapper.capture([1, 2], context_ref="ctx-test-v1")
        capture._state.layers[0].recurrent_states[0].add_(1)
        with self.assertRaisesRegex(InterventionError, "state changed"):
            wrapper.replay(capture, [3])

    def test_parent_weight_mutation_is_detected_before_replay(self):
        wrapper, model = protocol()
        capture = wrapper.capture([1, 2], context_ref="ctx-test-v1")
        model.marker.add_(1)
        with self.assertRaisesRegex(InterventionError, "parent weights"):
            wrapper.replay(capture, [3])

    def test_registration_metadata_and_capture_prefix_are_pinned(self):
        wrapper, _model = protocol()
        registration = direction(wrapper)
        altered = replace(registration, scale=registration.scale * 2)
        with self.assertRaisesRegex(InterventionError, "identity, bytes or ID mismatch"):
            wrapper.register(altered)
        wrapper.register(registration)
        capture = wrapper.capture([1, 2], context_ref="ctx-test-v1")
        altered_capture = replace(capture, prefix_sha256="c" * 64)
        with self.assertRaisesRegex(InterventionError, "lineage mismatch"):
            wrapper.replay(altered_capture, [3])

    def test_registration_set_seals_at_first_capture(self):
        wrapper, _model = protocol()
        wrapper.capture([1], context_ref="ctx-test-v1")
        late = direction(wrapper, edit_id="late-registration")
        with self.assertRaisesRegex(InterventionError, "sealed"):
            wrapper.register(late)

    def test_model_and_cache_configuration_drift_are_detected(self):
        wrapper, model = protocol()
        capture = wrapper.capture([1, 2], context_ref="ctx-test-v1")
        model.config["wkv_implementation"] = "changed-after-capture"
        with self.assertRaisesRegex(InterventionError, "parent weights"):
            wrapper.replay(capture, [3])

        wrapper2, _model2 = protocol()
        capture2 = wrapper2.capture([1, 2], context_ref="ctx-test-v1")
        capture2._state.layers[0].has_previous_state[0] = False
        with self.assertRaisesRegex(InterventionError, "state changed"):
            wrapper2.replay(capture2, [3])


if __name__ == "__main__":
    unittest.main()
