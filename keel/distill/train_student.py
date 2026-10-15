"""One-epoch bf16 LoRA training on one-pass trust-mode answer-slot probabilities."""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def restricted_kl_loss(label_logits: torch.Tensor, teacher: torch.Tensor, gold: torch.Tensor,
                       mask: torch.Tensor | None = None, ce_weight: float = 0.2) -> torch.Tensor:
    """KL(teacher || student) + .2 CE(gold), each distribution label-restricted.

    Padding labels are excluded from both normalizers. Teacher zeros contribute
    exactly zero, never 0*log(0). Float32 normalization remains stable under bf16.
    """
    if label_logits.ndim != 2 or teacher.shape != label_logits.shape or gold.shape != label_logits.shape[:1]:
        raise ValueError("loss shapes must be [batch, labels], [batch, labels], [batch]")
    valid = torch.ones_like(teacher, dtype=torch.bool) if mask is None else mask.bool()
    if valid.shape != teacher.shape or not valid.any(dim=-1).all():
        raise ValueError("every row needs a nonempty label set")
    if not torch.isfinite(teacher).all() or (teacher < 0).any() or (teacher[~valid] != 0).any():
        raise ValueError("teacher probabilities must be finite, nonnegative and zero on padding")
    if not torch.allclose(teacher.float().sum(-1), torch.ones(teacher.shape[0], device=teacher.device), atol=1e-5):
        raise ValueError("teacher distributions must sum to one")
    if (gold < 0).any() or (gold >= label_logits.shape[1]).any() or not valid.gather(1, gold[:, None]).all():
        raise ValueError("gold must name a valid label")
    logp = F.log_softmax(label_logits.float().masked_fill(~valid, -torch.inf), dim=-1)
    t = teacher.float()
    # Replace masked -inf before multiplication: masked gradients and losses stay finite.
    safe_logp = logp.masked_fill(~valid, 0)
    entropy_term = torch.xlogy(t, t)
    kl = (entropy_term - t * safe_logp).sum(-1)
    ce = -logp.gather(1, gold[:, None]).squeeze(1)
    return (kl + ce_weight * ce).mean()


def label_token_loss(vocab_logits: torch.Tensor, label_ids: torch.Tensor, teacher: torch.Tensor,
                     gold: torch.Tensor, mask: torch.Tensor | None = None, ce_weight: float = 0.2) -> torch.Tensor:
    """Reference adapter used by CPU tests: no nonlabel vocabulary logit enters."""
    if vocab_logits.ndim != 2 or label_ids.ndim != 2 or label_ids.shape[0] != vocab_logits.shape[0]:
        raise ValueError("vocabulary and label ids must have matching batch axes")
    return restricted_kl_loss(vocab_logits.gather(1, label_ids), teacher, gold, mask, ce_weight)


def prepare_rows(rows: list[dict], tokenizer, max_length: int = 2048) -> list[dict]:
    from keel.protocol import DecisionRequest, render

    prepared = []
    for row in rows:
        req = DecisionRequest.from_json(row["request"])
        r = render(req, mode="trust", think=False)
        ids = tokenizer.encode(r.prompt, add_special_tokens=False)
        if len(ids) > max_length:
            raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds max_length; truncation would move the answer slot")
        label_ids = []
        for label in r.labels:
            full = tokenizer.encode(r.prompt + " " + label, add_special_tokens=False)
            if len(full) != len(ids) + 1 or full[:-1] != ids:
                raise ValueError(f"{label!r} is not a single answer-slot token")
            label_ids.append(full[-1])
        if len(set(label_ids)) != len(label_ids):
            raise ValueError("rendered labels must have distinct token ids")
        probs = row["teacher_probs"]
        if set(probs) != set(r.option_ids):
            raise ValueError("teacher labels differ from rendered options")
        prepared.append({"id": row["id"], "ids": ids, "label_ids": label_ids,
                         "teacher": [probs[k] for k in r.option_ids], "gold": r.option_ids.index(row["gold"])})
    return prepared


def collate(rows: list[dict], pad_id: int, device: str = "cuda") -> dict[str, torch.Tensor]:
    b, n, k = len(rows), max(len(r["ids"]) for r in rows), max(len(r["label_ids"]) for r in rows)
    ids = torch.full((b, n), pad_id, dtype=torch.long)
    attention = torch.zeros((b, n), dtype=torch.long)
    labels = torch.zeros((b, k), dtype=torch.long)
    teacher = torch.zeros((b, k), dtype=torch.float32)
    mask = torch.zeros((b, k), dtype=torch.bool)
    gold = torch.empty(b, dtype=torch.long)
    for i, r in enumerate(rows):
        length, options = len(r["ids"]), len(r["label_ids"])
        ids[i, -length:] = torch.tensor(r["ids"])
        attention[i, -length:] = 1
        labels[i, :options] = torch.tensor(r["label_ids"])
        teacher[i, :options] = torch.tensor(r["teacher"])
        mask[i, :options] = True
        gold[i] = r["gold"]
    return {name: value.to(device) for name, value in {"input_ids": ids, "attention_mask": attention,
            "label_ids": labels, "teacher": teacher, "label_mask": mask, "gold": gold}.items()}


def answer_logits(model, batch: dict) -> torch.Tensor:
    """Skip the vision branch and full-vocabulary LM projection, retaining all LoRA modules."""
    base = model.get_base_model()
    attention = batch["attention_mask"]
    hidden = base.model.language_model(input_ids=batch["input_ids"], attention_mask=attention,
                position_ids=(attention.cumsum(-1) - 1).clamp_min(0), use_cache=False).last_hidden_state[:, -1]
    # One tiny projection per distinct label token, not batch*sequence*vocabulary.
    unique, inverse = torch.unique(batch["label_ids"], return_inverse=True)
    weights = base.lm_head.weight.index_select(0, unique)
    logits = F.linear(hidden, weights)
    return logits.gather(1, inverse)


def train(rows: list[dict], output: str, *, smoke: bool = False, batch_size: int = 4,
          accumulation: int = 4, lr: float = 1e-4, seed: int = 20261015, max_length: int = 2048,
          rank: int = 16, epochs: int = 1, dev_rows: list[dict] | None = None, export: bool = True) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    import transformers
    import peft
    from keel.distill.checkpoint import save_processor_assets
    from keel.runtime.modal_app import MODELS

    if not rows or batch_size < 1 or accumulation < 1 or rank < 1 or epochs < 1:
        raise ValueError("training requires rows and positive batch/accumulation/rank/epochs")
    start = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    model_id, revision = MODELS["qwen3.5-4b"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    prepared = prepare_rows(rows, tok, max_length)
    model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                      attn_implementation="sdpa").to("cuda")
    modules = (r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
               r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
               r"mlp\.(?:gate_proj|up_proj|down_proj))")
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                            target_modules=modules, bias="none"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, fused=True)
    random.shuffle(prepared)
    if smoke:
        prepared = prepared[:min(8, len(prepared))]
    unique_rows = len(prepared)
    if epochs > 1:
        additional = []
        for _ in range(epochs - 1):
            epoch_rows = list(prepared)
            random.shuffle(epoch_rows)
            additional.extend(epoch_rows)
        prepared += additional
    steps = math.ceil(len(prepared) / batch_size)
    losses, grad_norms = [], []
    optimizer.zero_grad(set_to_none=True)
    for step, offset in enumerate(range(0, len(prepared), batch_size)):
        batch = collate(prepared[offset:offset + batch_size], tok.pad_token_id or tok.eos_token_id)
        logits = answer_logits(model, batch)
        loss = restricted_kl_loss(logits, batch["teacher"], batch["gold"], batch["label_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        # The final accumulation window may contain fewer microbatches.
        window_start = (step // accumulation) * accumulation
        window_end = min(window_start + accumulation, steps)
        window_examples = min(window_end * batch_size, len(prepared)) - window_start * batch_size
        (loss * (batch["input_ids"].shape[0] / window_examples)).backward()
        losses.append(float(loss.detach()))
        if (step + 1) % accumulation == 0 or step + 1 == steps:
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm) or float(norm) <= 0:
                raise FloatingPointError("missing or nonfinite adapter gradient")
            grad_norms.append(float(norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print({"step": step, "steps": steps, "loss": losses[-1], "elapsed_s": time.time() - start}, flush=True)
    result = {"model": model_id, "revision": revision, "rows": unique_rows, "rows_seen": len(prepared), "epochs": epochs,
              "microbatches": steps, "updates": len(grad_norms), "loss_mean": sum(losses) / len(losses),
              "loss_first": losses[0], "loss_last": losses[-1], "gradient_norm_min": min(grad_norms),
              "gradient_norm_max": max(grad_norms), "trainable_parameters": sum(p.numel() for p in params),
              "bf16": True, "lr": lr, "seed": seed, "batch_size": batch_size, "accumulation": accumulation,
              "lora_rank": rank, "lora_alpha": 2 * rank, "ce_weight": 0.2, "smoke": smoke,
              "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
              "max_prompt_tokens": max(len(r["ids"]) for r in prepared),
              "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    if dev_rows:
        result["dev"] = evaluate_dev(model, dev_rows, tok, max_length, batch_size)
    if not smoke and export:
        out = Path(output)
        model.save_pretrained(out / "adapter")
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged", safe_serialization=True, max_shard_size="4GB")
        tok.save_pretrained(out / "merged")
        result["processor_assets"] = save_processor_assets(out / "merged")
        result["checkpoint"] = str(out / "merged")
    result["wall_s"] = time.time() - start
    return result


def evaluate_dev(model, rows: list[dict], tokenizer, max_length: int, batch_size: int) -> dict:
    """Selection-only synthetic objective, weighted equally across capability and v0."""
    import collections

    prepared = prepare_rows(rows, tokenizer, max_length)
    sources = ["capability" if r["kind"] == "capability" else "v0" for r in rows]
    totals = collections.defaultdict(lambda: {"n": 0, "loss_sum": 0.0, "correct": 0})
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(prepared), batch_size):
            batch = collate(prepared[offset:offset + batch_size], tokenizer.pad_token_id or tokenizer.eos_token_id)
            logits = answer_logits(model, batch)
            for i, source in enumerate(sources[offset:offset + batch_size]):
                loss = restricted_kl_loss(logits[i:i + 1], batch["teacher"][i:i + 1],
                                         batch["gold"][i:i + 1], batch["label_mask"][i:i + 1])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite synthetic dev loss")
                totals[source]["n"] += 1
                totals[source]["loss_sum"] += float(loss)
                pred = logits[i].masked_fill(~batch["label_mask"][i], -torch.inf).argmax()
                totals[source]["correct"] += int(pred == batch["gold"][i])
    by_source = {s: {"n": v["n"], "loss": v["loss_sum"] / v["n"], "accuracy": v["correct"] / v["n"]}
                 for s, v in totals.items()}
    return {"by_source": by_source, "balanced_loss": sum(v["loss"] for v in by_source.values()) / len(by_source),
            "balanced_accuracy": sum(v["accuracy"] for v in by_source.values()) / len(by_source)}
