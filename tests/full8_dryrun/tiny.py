"""Tiny random Qwen3.5 stand-in with the REAL 9B layer layout: 32 layers, full attention at 3, 7, ..., 31 (8 attention
layers, so the pre-registered rule applies unchanged), hidden 32, fp32, pure-PyTorch GDN kernel. Deterministic from
the seed, so a restarted attempt rebuilds the identical base model (as the real job reloads the pinned snapshot)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

VOCAB = 101
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
LETTERS = ["A", "B", "C", "D"]


def text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=32, vocab_size=VOCAB,
        layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)],
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    cfg._attn_implementation = "eager"
    return cfg


def tiny_convertible(seed: int = 0):
    import torch
    import torch.nn as nn
    from transformers.models.qwen3_5 import modeling_qwen3_5 as Mq

    from convert.model import ConvertibleQwen, replace_native_gdn

    torch.manual_seed(seed)
    cfg = text_config()
    lm = Mq.Qwen3_5TextModel(cfg).float()
    for p in lm.parameters():
        torch.nn.init.normal_(p, std=0.05)
    head = nn.Linear(32, VOCAB, bias=False)
    full = SimpleNamespace(model=SimpleNamespace(language_model=lm), lm_head=head)
    m = ConvertibleQwen(full, kernel="chunk")
    replace_native_gdn(m, kernel="chunk")
    for p in list(lm.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    m.eval()
    return m


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"w{i}": i for i in range(VOCAB - len(SPECIALS) - 1 - len(LETTERS))}
    for c in LETTERS:  # MMLU answer letters get their own ids (probes.core scores " A".." D")
        vocab[c] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    for s in SPECIALS:
        vocab[s] = len(vocab)
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", eos_token="<|endoftext|>",
                                  additional_special_tokens=SPECIALS[1:])
    return tok


def write_snapshot(root) -> Path:
    """A fake pinned snapshot dir (config.json in the multimodal Qwen3_5 layout, text_config = the tiny model's), as
    sf_fetch leaves it: <root>/models--tiny--q9b/snapshots/<rev>/."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    d = Path(root) / "models--tiny--q9b" / "snapshots" / "dryrun0"
    d.mkdir(parents=True, exist_ok=True)
    tc = text_config().to_dict()
    cfg = Qwen3_5Config(text_config=tc)
    (d / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1, default=str), encoding="utf-8")
    return d


def stub_probes(model, tok, eval_toks, cfg):
    """probes.core.run_all's shape, computed from the tiny model in its current modes (real numbers, tiny volume)."""
    import math

    import torch

    x = torch.as_tensor(eval_toks[:4 * 24].astype("int64")).view(4, 24)
    with torch.no_grad():
        lp = model.lm_head(model.hidden(x)).float().log_softmax(-1)
    nll = float(-lp[:, :-1].gather(-1, x[:, 1:, None]).mean())
    acc = float((lp[:, :-1].argmax(-1) == x[:, 1:]).float().mean())
    return {"ppl": {"ctx1024": {"ppl": math.exp(nll)}, "ctx4096": {"ppl": math.exp(nll)}},
            "niah": {**{f"L{n}": {"acc": 1.0} for n in cfg["probes"]["niah_lengths"]}, "mean_acc": 1.0},
            "mqar": {"mean_acc": acc}, "mmlu": {"acc": acc, "n": 285}}


def mmlu_rows(n=14042):
    """A synthetic stand-in for the full MMLU test set with the real item count (the rule requires n == 14,042)."""
    return [{"subject": f"s{i % 57}", "question": " ".join(f"w{(i * 7 + j) % 90}" for j in range(6)),
             "choices": [f"w{(i + j) % 90}" for j in range(4)], "answer": i % 4} for i in range(n)]
