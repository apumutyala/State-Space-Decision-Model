"""Evidence-harness plug for the canonicalised shim (layout g1v2-canon, A-PREREG-JEV3-LM):
``SSDM_INV_HEAD=jobs.sym_eval.canon_plug:score_fn`` makes tests/guarantees/test_reorder_invariance.py check the SERVED
system (canonicaliser + shim rows + an lm_label tiny tied Qwen3.5 whose vocabulary is the real Qwen3.5-4B tokenizer's;
``SSDM_BPE_TOKENIZER`` = the tokenizer dir). ``score_fn(state, options)`` with options a list of (id, content) returns
the served per-id probabilities (canonical rows, tie equalisation)."""
from __future__ import annotations

import functools
import os
from typing import Any, Sequence

TEMPLATE = "jrt-h2o-shim:qwen3.5-4b"


@functools.lru_cache(maxsize=1)
def _backend():
    import torch
    import transformers as tf

    from decide.shim_format import bind_template
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel

    spec = bind_template(TEMPLATE, os.environ["SSDM_BPE_TOKENIZER"])
    torch.manual_seed(0)
    cfg = tf.Qwen3_5TextConfig(vocab_size=len(spec.tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=8, linear_num_key_heads=2,
                               linear_num_value_heads=2, linear_key_head_dim=4, linear_value_head_dim=4,
                               layer_types=["linear_attention", "full_attention"], tie_word_embeddings=True)
    from heads.backbone import HiddenStateBackbone, qwen35_spec

    lm = tf.Qwen3_5ForCausalLM(cfg).float().eval()
    bb = HiddenStateBackbone(lm.model, spec=qwen35_spec(cfg, "tiny-qwen35-canon", "tiny"), fork_mode="cache")
    model = DecisionModel(bb, "lm_label", seed=0).float().eval()
    return DecisionModelBackend(model, None, template=TEMPLATE, layout="g1v2-canon")


def score_fn(state: Any, options: Sequence[tuple[str, str]]) -> dict[str, float]:
    import math

    req = {"state": state, "questions": {"q": {"type": "choice", "instructions": "Pick the best option.",
                                              "criteria": {str(i): str(c) for i, c in options}}}}
    lg = _backend().score(req, lambda: None).logits["q"]
    return {k: math.exp(v) for k, v in lg.items()}
