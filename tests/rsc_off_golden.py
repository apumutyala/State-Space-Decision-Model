"""OFF-path golden for stack/rwkv7/sym-contractive: uses ONLY APIs that exist on the parent (stack/rwkv7/d20-v4-chain
@9323085c), so the cpu_tests job runs it in the parent tree (``python tests/rsc_off_golden.py`` with the parent on
PYTHONPATH) and tests/test_sym_contractive.py recomputes it in this tree: equal JSON = the flags-off path is
bit-identical (float.hex) to the parent for a fixed seed. Covers the D19 row builders (g1v2 / v3w and ordered), the
cache fork, joint_record (lexical prior + record mixer), LoRA install and the LoRA gradients / state-dict keys."""
from __future__ import annotations

import json


def golden() -> dict:
    import torch

    from decide.prompt_g1v3 import G1V3W_ID
    from heads.backbone import tiny_hf_backbone
    from heads.layout import ByteEncoder, build_row
    from heads.model import DecisionModel
    from train.data import TOY_TEMPLATE
    from train.lora import apply_lora, lora_parameters, lora_state_dict

    enc = ByteEncoder()
    req = {"model": "toy", "state": {"msg": "card declined abroad", "region": "EU"},
           "questions": {"q1": {"type": "choice", "instructions": "Which intent?",
                                "criteria": {"opt_a": "card declined", "opt_b": "travel exchange rate",
                                             "opt_c": "lost or stolen"}},
                         "q2": {"type": "noul", "instructions": "Is it urgent?"}}}
    torch.manual_seed(0)
    bb = tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache")
    model = DecisionModel(bb, "joint_record", seed=0).float()
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        model.head.w.copy_(torch.randn(model.head.w.shape, generator=g))
        model.head.lex_scale.fill_(1.5)
    rows = [build_row(req, q, enc, template=G1V3W_ID, layout="g1v2") for q in req["questions"]]
    ordered = [build_row(req, q, enc, template=TOY_TEMPLATE, layout="ordered") for q in req["questions"]]
    out: dict = {}
    with torch.no_grad():
        out["g1v2_lp"] = [float.hex(x) for x in model(rows).log_probs.flatten().tolist() if x > -1e30]
        out["ordered_lp"] = [float.hex(x) for x in model(ordered).log_probs.flatten().tolist() if x > -1e30]
    out["g1v2_ids"] = [r.ids for r in rows]
    found = apply_lora(bb.model, bb.spec.lora_groups, rank=4, check_counts=False,
                       generator=torch.Generator().manual_seed(2))
    out["lora_groups"] = {k: len(v) for k, v in sorted(found.items())}
    with torch.no_grad():
        for p in lora_parameters(bb.model):
            if p.ndim == 2 and not p.abs().sum():  # lora_B starts at zero: give it values so the forward moves
                p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel())) * 0.05)
    lp = model(rows).log_probs
    loss = -lp[:, 0].sum()
    loss.backward()
    out["loss"] = float.hex(float(loss))
    out["lora_grad_sums"] = [None if p.grad is None else float.hex(float(p.grad.double().sum()))
                             for p in lora_parameters(bb.model)]
    out["lora_keys"] = sorted(lora_state_dict(bb.model))
    out["trainable"] = sorted(n for n, p in bb.model.named_parameters() if p.requires_grad)
    return out


if __name__ == "__main__":
    print(json.dumps(golden(), sort_keys=True))
