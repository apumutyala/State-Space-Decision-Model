"""CPU mocks for the Clef baseline: a tiny tokenizer, a tiny text backbone, and Cloudflare's real
(vendored) ``JointSchemaHead`` / ``ClefModel`` at toy width. No weights are downloaded."""

from __future__ import annotations

import torch

SPECIAL = {"<|im_start|>": 300, "<|im_end|>": 301, "<|endoftext|>": 302}  # special: split by the flag
ADDED = {"<think>": 303, "</think>": 304}  # added, non-special: survive split_special_tokens
VOCAB = 320
HIDDEN = 32


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class _Tok:
    def __init__(self, content):
        self.content = content


class FakeTok:
    """Char-level tokenizer with Qwen-like control tokens. ``split_special_tokens=True`` splits the
    special ones only (as HF does); added non-special tokens still match."""

    all_special_ids = sorted(SPECIAL.values())
    all_special_tokens = sorted(SPECIAL)
    added_tokens_decoder = {i: _Tok(s) for s, i in {**SPECIAL, **ADDED}.items()}
    pad_token_id = 302

    def __call__(self, text, add_special_tokens=False, split_special_tokens=False):
        table = dict(ADDED) if split_special_tokens else {**SPECIAL, **ADDED}
        ids, i = [], 0
        while i < len(text):
            for s, sid in sorted(table.items(), key=lambda kv: -len(kv[0])):
                if text.startswith(s, i):
                    ids.append(sid)
                    i += len(s)
                    break
            else:
                ids.append(min(ord(text[i]), 255))
                i += 1
        return _Enc(ids)

    def __len__(self):
        return VOCAB

    def decode(self, ids):
        inv = {v: k for k, v in {**SPECIAL, **ADDED}.items()}
        return "".join(inv.get(i, chr(i)) for i in ids)


class MockProcessor:
    def __init__(self, tok):
        self.tokenizer = tok


class _Out:
    def __init__(self, h):
        self.last_hidden_state = h


#: perturbation of the mocked "fast kernel" relative to the reference (tests flip it to simulate a bad kernel)
FAST_KERNEL_ERROR = {"scale": 1e-4}


def torch_chunk_gated_delta_rule(x):
    """Reference path (named like transformers' Qwen3.5 torch fallback; looked up by module name)."""
    return torch.tanh(x)


def fast_chunk_gated_delta_rule(x):
    """Stand-in for fla's kernel: the reference plus a small, deterministic numeric difference."""
    return torch.tanh(x) + FAST_KERNEL_ERROR["scale"] * torch.cos(3 * x)


class MockGDN(torch.nn.Module):
    """Has the two attributes ``reference_gdn_path`` switches, like Qwen3_5GatedDeltaNet."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN)
        self.chunk_gated_delta_rule = fast_chunk_gated_delta_rule
        self.causal_conv1d_fn = object()  # "kernel present"; the switch sets it to None

    def forward(self, x):
        return x + self.chunk_gated_delta_rule(self.proj(x))


class MockText(torch.nn.Module):
    """Embedding + one causal-ish mixing layer; accepts input_ids or inputs_embeds like HF."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        self.mix = torch.nn.Linear(HIDDEN, HIDDEN)
        self.gdn = MockGDN()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, use_cache=False, return_dict=True, **kw):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        csum = torch.cumsum(x, dim=1) / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        return _Out(self.gdn(torch.tanh(self.mix(csum)) + x))


class MockLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockText()
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.model.embed_tokens


def build_mock_model(jsm, seed: int = 0):
    torch.manual_seed(seed)
    head = jsm.JointSchemaHead(hidden_size=HIDDEN, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    with torch.no_grad():  # zero-init scales would make the joint term vanish; give every path weight
        head.prior_logit_scale.fill_(1.0)
        head.joint_logit_scale.fill_(1.0)
        head.residual_gate.fill_(0.5)
    return jsm.ClefModel(MockLM(), head).eval()


_STANDIN = {}


def load_standin(model_dir, release="clef-flash", *, encoding="ssdm", max_length=16384, **kw):
    """``CLEF_STANDIN=clef_mocks:load_standin`` for the run_job.sh CPU dry run: the same (backend, model, processor)
    triple ``baselines.clef_backend.load_clef`` returns, on the tiny mocked Clef (seed 4, fp32, CPU). Deterministic
    across processes, so the kit's decide and native runs and the probes all see the same model."""
    from baselines import clef_backend as CB

    if "model" not in _STANDIN:
        _STANDIN["jsm"] = CB.load_jsm()
        _STANDIN["model"] = build_mock_model(_STANDIN["jsm"], seed=4)
    be = CB.ClefBackend(_STANDIN["model"], FakeTok(), release=release, encoding=encoding, jsm=_STANDIN["jsm"],
                        device="cpu", dtype="float32", max_length=max_length,
                        files_sha256={"note": "STANDIN mock (dry run), no weights"})
    return be, _STANDIN["model"], MockProcessor(FakeTok())
