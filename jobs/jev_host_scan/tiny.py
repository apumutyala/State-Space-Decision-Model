"""CPU test fixtures: tiny random-weight models with each candidate family's REAL config layout, tokenizer and (for the
vendored loaders) code, so the scan's loaders, readouts and scoring run end to end on CPU without downloading weights.

    python -m jobs.jev_host_scan.tiny <kind> <out_dir> [--tokenizers <root>]

kinds: qwen35 (Qwen3_5ForCausalLM text config + the Qwen3.5-4B tokenizer; head_dtype float32 as H2O v1.1), rwkv7
(vendored G1k code + G1k tokenizer; rwkv7_g1i: the same without code files), d1 (vendored D1Model + d1 tokenizer),
nemotron_h, lfm2 (stock transformers classes + their tokenizers). gemma-4 is not built here (its heterogeneous per-layer
config does not shrink generically); the job's strict load check guards it. Weights are random; only plumbing is tested.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

TOK = {
    "qwen35": "Qwen--Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "h2o": "h2oai--h2o-lightning-4b@193ad740925b176a3b70a5a13a7cff2f2fadd01e",
    "rwkv7": "RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba",
    "rwkv7_g1i": "RWKV--RWKV7-2.9B-20260805@aae64511841dd32f5235f6bcc7a931c6ef806e6c",
    "d1": "LiquidAI--d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432",
    "nemotron_h": "nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16@dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
    "lfm2": "LiquidAI--LFM2.5-2.6B@654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
}
TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "vocab.json",
             "merges.txt", "generation_config.json")


def default_root() -> Path:
    return Path(os.environ.get("SSDM_TOKENIZERS", Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))
                               / "tokenizers"))


def _copy_tok(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in TOK_FILES:
        if (src / f).exists():
            shutil.copyfile(src / f, out / f)


def _shrink_text(t: dict, layer_types: list[str] | None = None) -> dict:
    t = dict(t)
    t.update(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    if layer_types is not None:
        t["layer_types"] = layer_types
        t["num_hidden_layers"] = len(layer_types)
    return t


def build(kind: str, out: Path, root: Path | None = None) -> Path:
    import torch

    torch.manual_seed(0)
    root = root or default_root()
    src = root / TOK[kind]
    out = Path(out)
    _copy_tok(src, out)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8")) if (src / "config.json").exists() else {}
    if kind in ("qwen35", "h2o"):
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

        t = cfg.get("text_config", cfg)
        t = _shrink_text(t, ["linear_attention", "full_attention"])
        t.update(linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                 mtp_num_hidden_layers=0)
        t.pop("model_type", None)
        c = Qwen3_5TextConfig(**{k: v for k, v in t.items() if k not in ("architectures", "transformers_version")})
        c.head_dtype = "float32"
        Qwen3_5ForCausalLM(c).to(torch.bfloat16).save_pretrained(out)
    elif kind in ("rwkv7", "rwkv7_g1i"):
        import third_party.rwkv7_hf as V
        from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
        from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

        c = Rwkv7Config(**{**{k: v for k, v in cfg.items() if k not in ("architectures", "auto_map", "model_type",
                                                                        "transformers_version")},
                           "hidden_size": 128, "intermediate_size": 256, "num_heads": 2, "head_dim": 64,
                           "num_hidden_layers": 2, "a_low_rank_dim": 8, "decay_low_rank_dim": 8, "gate_low_rank_dim": 8,
                           "v_low_rank_dim": 8})
        Rwkv7ForCausalLM(c).to(torch.bfloat16).save_pretrained(out)
        if kind == "rwkv7":
            for f in V.CODE_SHA256:
                shutil.copyfile(V.HERE / f, out / f)
        else:
            for f in V.CODE_SHA256:
                (out / f).unlink(missing_ok=True)
    elif kind == "d1":
        import third_party.d1_3b as D
        from transformers import Lfm2VlConfig

        from third_party.d1_3b.modeling_d1 import D1Model

        t = _shrink_text(cfg["text_config"], ["conv", "full_attention", "conv"])
        t.update(block_dim=64, conv_dim=64, block_auto_adjust_ff_dim=False)
        v = dict(cfg["vision_config"], hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2)
        top = {k: x for k, x in cfg.items() if k not in ("text_config", "vision_config", "architectures", "auto_map",
                                                         "transformers_version")}
        top.update(projector_hidden_size=64)
        c = Lfm2VlConfig(**{**top, "text_config": t, "vision_config": v})
        c.auto_map = {"AutoModel": "modeling_d1.D1Model"}
        D1Model(c).to(torch.bfloat16).save_pretrained(out)
        for f in D.CODE_SHA256:
            shutil.copyfile(D.HERE / f, out / f)
    elif kind == "nemotron_h":
        from transformers import AutoModelForCausalLM, NemotronHConfig

        c = dict(cfg)
        c.update(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                 hybrid_override_pattern="M*M-", num_hidden_layers=4, mamba_num_heads=8, mamba_head_dim=16,
                 n_groups=2, ssm_state_size=16, chunk_size=16, time_step_rank=8, use_mamba_kernels=False)
        c = NemotronHConfig(**{k: x for k, x in c.items() if k not in ("architectures", "auto_map", "model_type",
                                                                       "transformers_version")})
        AutoModelForCausalLM.from_config(c).to(torch.bfloat16).save_pretrained(out)
    elif kind == "lfm2":
        from transformers import AutoModelForCausalLM, Lfm2Config

        t = _shrink_text(cfg, ["conv", "full_attention", "conv"])
        t.update(block_dim=64, conv_dim=64, block_auto_adjust_ff_dim=False)
        c = Lfm2Config(**{k: x for k, x in t.items() if k not in ("architectures", "model_type", "transformers_version")})
        AutoModelForCausalLM.from_config(c).to(torch.bfloat16).save_pretrained(out)
    else:
        raise ValueError(kind)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=sorted(TOK))
    ap.add_argument("out")
    ap.add_argument("--tokenizers", default=None)
    a = ap.parse_args(argv)
    print(build(a.kind, Path(a.out), Path(a.tokenizers) if a.tokenizers else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
