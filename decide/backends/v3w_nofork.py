"""v3w readout WITHOUT the prefix fork: ``decide.v3w_generic.make_v3w(backbone_id)`` scored by one full forward per question.

The prompt ids, option order and scored tokens are exactly ``V3W.questions(request)`` (the backbone's own chat template,
its own single-token code table, word ids by their unique first token, whole-string tokenization; for RWKV7-G1k the
legacy v3w byte for byte). What changes against ``V3WGenericBackend`` / ``G1V3Backend(mode="words")`` is only the call
structure: no shared-prefix prefill and no state copies, so the score cannot depend on bf16 state-passing path effects
(job-gen87: fork != recompute by 0.26 nat on a hybrid MoE in bf16). A JevBench request carries one question, so the
fork would save nothing there anyway.

  coded / word-id question:  one forward of the whole prompt, logit(option) = log p(answer token | prompt)
  fallback (k > code table): one forward of prompt + " <option id>\\n" per option, sum of its log-probabilities

Needs only ``plugin.forward(ids [1, T], n_last) -> logits [1, n_last, V]``; no state ops.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable

from .. import versions
from ..schema import canonical_json, sha256_text
from .base import BackendUnsupported, Plan, ScoreResult
from .hf_readout import sha256_file


class V3WNoForkBackend:
    name = "v3w_nofork"

    def __init__(self, plugin, v3w) -> None:
        self.plugin, self.v3w = plugin, v3w
        ident = v3w.identity()
        self._v3w_identity = ident
        here = Path(__file__)
        config = {
            "scoring": "coded / word-id: log p(single answer token | v3w prompt); fallback k > code table: "
                       "sum_logprob(' ' + option_id + '\\n')",
            "scoring_path": "recompute: one full forward per question (fallback: per option), batch 1, no prefix fork",
            "tokenization": "whole prompt string (decide.v3w_generic; specials only where the format puts them)",
            "v3w": ident, "codes": len(v3w.codes), "dtype": plugin.dtype, "device": plugin.device,
            "kernel_stack": plugin.kernel_stack, "untrusted_encoding": "v3w encoder: fail closed on control ids",
            "source_sha256": {p.name: sha256_file(p) for p in (here, here.parents[1] / "v3w_generic.py",
                                                              here.parents[1] / "prompt_g1v3.py")},
        }
        cfg_hash = sha256_text(canonical_json(config))
        self._revision = {
            "revision_id": (f"{plugin.family}:{plugin.model_id}@{(plugin.snapshot or 'local')[:12]}"
                            f";w={plugin.files.get('weights', 'unknown')[:12]}"
                            f";tok={plugin.files.get('tokenizer', 'unknown')[:12]}"
                            f";prompt={ident['id']}.{ident['sha256'][:8]};cfg={cfg_hash[:8]}"
                            f";adapter={versions.ADAPTER_VERSION}"),
            "backend": self.name, "family": plugin.family, "model_id": plugin.model_id, "snapshot": plugin.snapshot,
            "files_sha256": plugin.files, "prompt_template": ident, "inference_config": config,
            "inference_config_sha256": cfg_hash, "adapter_version": versions.ADAPTER_VERSION,
            "max_context": plugin.max_context, "notes": plugin.notes,
        }

    def revision(self) -> dict[str, Any]:
        return copy.deepcopy(self._revision)

    def _questions(self, request: dict[str, Any]):
        try:
            return self.v3w.questions(request)
        except BackendUnsupported:
            raise
        except ValueError as exc:  # a prompt the format cannot carry (control text in the request)
            raise BackendUnsupported("control_tokens_in_input", str(exc)) from exc

    def plan(self, request: dict[str, Any]) -> Plan:
        qs = self._questions(request)
        calls, total, longest = 0, 0, 0
        for q in qs:
            if q["coded"]:
                calls += 1
                total += len(q["prompt"])
                longest = max(longest, len(q["prompt"]))
            else:
                calls += len(q["options"])
                total += sum(len(q["prompt"]) + len(c) for c in q["conts"])
                longest = max(longest, len(q["prompt"]) + max(map(len, q["conts"])))
        return Plan(inference_calls=calls, max_call_input_tokens=longest, total_input_tokens=total,
                    per_question={q["qid"]: {"prompt_tokens": len(q["prompt"]), "options": len(q["options"]),
                                             "coded": q["coded"], "words": q.get("words", False)} for q in qs},
                    prompt_token_labels={"total": sum(len(q["prompt"]) for q in qs)}, provenance_applied=False)

    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        import torch

        dev = self.plugin.device
        qs = self._questions(request)
        logits: dict[str, dict[str, float]] = {}
        call_ms: list[float] = []
        total = 0

        def run(ids: list[int], n_last: int):
            check_deadline()
            t0 = time.perf_counter()
            with torch.inference_mode():
                lg = self.plugin.forward(torch.tensor([ids], dtype=torch.long, device=dev), n_last)
                if not bool(torch.isfinite(lg).all()):
                    raise BackendUnsupported("non_finite_logits", "nonfinite logits")
                lp = torch.log_softmax(lg[0].float(), dim=-1)
            call_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
            return lp

        for q in qs:
            if q["coded"]:
                lp = run(q["prompt"], 1)[-1]
                vals = lp.index_select(0, torch.tensor(q["tokens"], device=lp.device)).tolist()
                logits[q["qid"]] = dict(zip(q["options"], vals))
                total += len(q["prompt"])
            else:
                out = {}
                for o, c in zip(q["options"], q["conts"]):
                    lp = run(q["prompt"] + c, len(c) + 1)  # positions predicting each continuation token
                    out[o] = float(sum(lp[t, c[t]].item() for t in range(len(c))))
                    total += len(q["prompt"]) + len(c)
                logits[q["qid"]] = out
        return ScoreResult(logits=logits, inference_calls=len(call_ms), total_input_tokens=total,
                           extras={"call_latency_ms": call_ms, "scoring_path": "v3w-nofork",
                                   "coded": {q["qid"]: q["coded"] for q in qs},
                                   "words": {q["qid"]: q.get("words", False) for q in qs}})
