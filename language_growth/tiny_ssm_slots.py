"""CPU development probe: diagonal causal SSM and shared option scorer.

No option IDs, request IDs, catalog hashes, labels or gold programs are encoded.
Earlier proposed descriptions are permitted state; training uses teacher forcing,
whereas planner evaluation supplies only this model's previous predictions.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import nn


def model_text(request: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Explicit evidence-only projection; IDs are retained only as output keys."""
    state = json.loads(request["state"])
    if len(request["questions"]) != 1:
        raise ValueError("one typed question required")
    question = next(iter(request["questions"].values()))
    context = {"task": state["task_input"], "question": question["instructions"],
               "records": [{"record_ref": r["record_ref"], "entity": r["entity_ref"],
                            "fields": r["fields"]} for r in state["records"]],
               "proposals": [{"slot": p["slot"], "description": p["description"]}
                             for p in state.get("proposed_slot_choices", [])]}
    # Canonical by description, independently of option ID or insertion order.
    choices = sorted(question["criteria"].items(), key=lambda item: item[1])
    return (json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
            [oid for oid, _ in choices], [description for _, description in choices])


def byte_batch(texts: list[str]) -> torch.Tensor:
    encoded = [list(text.encode("utf-8")) for text in texts]
    out = torch.zeros((len(texts), max(map(len, encoded))), dtype=torch.long)
    for index, row in enumerate(encoded):
        out[index, :len(row)] = torch.tensor(row, dtype=torch.long) + 1
    return out


class DiagonalSSM(nn.Module):
    """h[t] = a*h[t-1] + (1-a)*B(embedding(byte[t])).

    Terminal states are evaluated using the exact closed form for this diagonal
    recurrence. This is causal, attention-free, and sequence-order-sensitive;
    batching is a computational optimization, not a bag-of-words approximation.
    Padding never advances the state. Memory scales are learned from training.
    """
    def __init__(self, width: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.input = nn.Linear(width, width)
        decay = torch.linspace(0.5, 0.999, width)
        self.decay_logit = nn.Parameter(torch.logit(decay))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids.ne(0)
        length = mask.sum(1)
        u = torch.tanh(self.input(self.embedding(ids)))
        lag = (length[:, None] - 1 - torch.arange(ids.shape[1])[None, :]).clamp_min(0)
        a = self.decay_logit.sigmoid()
        weight = a[None, None, :].pow(lag[:, :, None]) * (1 - a)
        terminal = (u * weight * mask[:, :, None]).sum(1)
        # Mean of every recurrent state, still an exact closed-form SSM readout.
        remaining = (length[:, None] - torch.arange(ids.shape[1])[None, :]).clamp_min(0)
        mean_weight = (1 - a[None, None, :].pow(remaining[:, :, None]))
        mean_state = (u * mean_weight * mask[:, :, None]).sum(1) / length[:, None].clamp_min(1)
        return torch.cat((terminal, mean_state), 1)


class TinySlotSSM(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.ssm = DiagonalSSM(width)
        self.scorer = nn.Sequential(nn.Linear(width * 6, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, contexts: torch.Tensor, candidates: torch.Tensor,
                owners: torch.Tensor) -> torch.Tensor:
        context = self.ssm(contexts)[owners]
        candidate = self.ssm(candidates)
        return self.scorer(torch.cat((context, candidate, context * candidate), 1)).squeeze(-1)


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TinySSMBackend:
    name = "development_tiny_diagonal_ssm"

    def __init__(self, model: TinySlotSSM, resource_check=lambda: None):
        self.model = model.eval()
        self.resource_check = resource_check
        self.identity = {"revision_id": f"tiny-diagonal-ssm:{state_hash(model)}",
                         "model_id": self.name, "state_sha256": state_hash(model)}
        self.predictions = []

    def revision(self):
        return self.identity

    def plan(self, request):
        from decide.backends.base import Plan
        context, _, descriptions = model_text(request)
        return Plan(inference_calls=1, max_call_input_tokens=max(len(context.encode()),
                    max(len(x.encode()) for x in descriptions)),
                    total_input_tokens=len(context.encode()) + sum(len(x.encode()) for x in descriptions))

    @torch.no_grad()
    def score(self, request, check_deadline):
        from decide.backends.base import ScoreResult
        self.resource_check()
        check_deadline()
        context, ids, descriptions = model_text(request)
        scores = self.model(byte_batch([context]), byte_batch(descriptions),
                            torch.zeros(len(ids), dtype=torch.long)).tolist()
        qid = next(iter(request["questions"]))
        self.predictions.append({"request": request, "projected_context": context,
                                 "option_ids": ids, "descriptions": descriptions, "logits": scores})
        return ScoreResult(logits={qid: dict(zip(ids, scores))}, inference_calls=1,
                           total_input_tokens=self.plan(request).total_input_tokens)
