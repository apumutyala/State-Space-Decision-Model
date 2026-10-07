"""Scripted CPU vocabulary tensors, never pretrained weights or an HF model."""
import hashlib


def fixture():
    import torch

    from decide import Decider
    from decide.backends.hf_readout import ModelPlugin
    from decide.backends.rwkv_g1_code_label import G1CodeLabelBackend, G1_REVISION
    from decide.backends.rwkv_g1_code_label_v2 import G1CodeLabelV2Backend
    from decide.backends.rwkv_g1_filtered_codes import (PROOF_CONTEXT, TOKENIZER_SHA256, VOCAB_SIZE, load_frozen_table)

    torch.set_num_threads(1)
    table = load_frozen_table()
    encoded = {" " + code.code: code.token_id for code in table}
    decoded = {token: text for text, token in encoded.items()}

    def encode(text, trusted=True):
        if text == PROOF_CONTEXT:
            return [55000]
        if text.startswith(PROOF_CONTEXT) and text[len(PROOF_CONTEXT):] in encoded:
            return [55000, encoded[text[len(PROOF_CONTEXT):]]]
        if text in encoded:
            return [encoded[text]]
        if text == "synthetic-oversized-state":
            return [20000] * 17000
        return [10000 + int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 40000]

    def logits(batch, length):
        value = torch.zeros((batch, length, VOCAB_SIZE), device="cpu")
        value[..., table[0].token_id] = 0.3
        value[..., table[1].token_id] = 0.1
        return value

    class StateOps:
        def prefill(self, ids):
            assert ids.device.type == "cpu"
            return logits(1, 1), torch.zeros(1, device="cpu")

        def extend(self, state, ids, n_last):
            assert state.device.type == ids.device.type == "cpu"
            return logits(ids.shape[0], n_last), torch.zeros(ids.shape[0], device="cpu")

    def full(ids, n_last):
        assert ids.device.type == "cpu"
        return logits(ids.shape[0], n_last)

    def plugin():
        return ModelPlugin("rwkv7", "synthetic-scripted-no-model", None, encode, full,
            files={"tokenizer": TOKENIZER_SHA256, "weights": "not-loaded"}, notes={"vendored_code": G1_REVISION},
            special_ids=frozenset({0}), dtype="float32", device="cpu", max_context=25600, state_ops=StateOps())

    decode = lambda ids: "".join(decoded[i] for i in ids)
    backends = {"v1": G1CodeLabelBackend(plugin(), tokenizer_decode=decode),
                "v2": G1CodeLabelV2Backend(plugin(), tokenizer_decode=decode)}
    rows = []
    for i in range(12):
        state = f"Synthetic state {i}" if i < 11 else "synthetic-oversized-state"
        rows.append({"state": state,
            "questions": {f"q{j}": {"type": "choice", "instructions": "Choose a supplied option.",
                                     "criteria": {"a": "Alpha", "b": "Beta"}} for j in range(2)},
            "expected": {"q0": "a", "q1": "a"},
            "_evaluation": {"run_id": f"synthetic:{i}", "catalog_id": 3, "track": "synthetic",
                "group_id": f"synthetic-group-{i}", "proxy_tokens": 17000 if i == 11 else 64,
                "payload_sha256": hashlib.sha256(state.encode()).hexdigest()}})
    return rows, {arm: Decider(backend, enable_v0_1_deltas=True, backend_max_input_tokens=25600)
                  for arm, backend in backends.items()}
