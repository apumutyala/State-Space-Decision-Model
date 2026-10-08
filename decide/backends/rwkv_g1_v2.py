"""jrt-rwkv-g1-v2 readout backend: the vendor-compliant RWKV7 G1 format (decide.prompt.build_prompt_g1v2).

Opt-in (readout family ``rwkv7-hf-g1-v2``); the stock ``jrt-rwkv-g1-v1`` HFReadoutBackend stays the default.

Per request:
  * every question's prompt = [0] + encode(whole v2 prompt string)    (F2 token 0, F5 one tokenization)
  * the shared token prefix of all question prompts is prefilled once (RWKV state), each question's tail is
    extended from it in packed calls, and the next-token distribution after "ANSWER:" is read once per question
  * coded question (k <= codebook size): logit(option) = log p(" <CODE>" | prompt), one token per option
  * fallback question (k > codebook size): logit(option) = sum log p(" <option id>\\n" | prompt), multi-token
Request text that encodes to the control id 0 is refused (fail closed), as in v1.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable

from ..prompt import CANARY_REQUEST, G1V2_ID, G1V2_SYSTEM_PROMPT, G1V2_TEMPLATE, render_g1v2
from ..schema import canonical_json, canonical_options, canonical_question_ids, sha256_text
from .base import BackendUnsupported, Plan, ScoreResult
from .hf_readout import HFReadoutBackend, sha256_file
from .rwkv_g1_filtered_codes import TABLE_SHA256, load_frozen_table
from .state_fork import PAD_ID, common_prefix_len, fork_score

BOS_ID = 0
PACK_ROWS = 32
PACK_TOKENS = 4096  # tail positions per packed call (logits <= 4096 x 65,536 bf16 = 0.5 GB)


def renderer_identity() -> dict[str, str]:
    table = [c.code for c in load_frozen_table()]
    canary = {qid: render_g1v2(CANARY_REQUEST, qid, table)
              for qid in canonical_question_ids(CANARY_REQUEST["questions"])}
    value = {"id": G1V2_ID, "head": G1V2_TEMPLATE.head, "tail": G1V2_TEMPLATE.tail, "system": G1V2_SYSTEM_PROMPT,
             "bos": BOS_ID, "codebook_table_sha256": TABLE_SHA256, "canary": canary}
    return {"id": G1V2_ID, "sha256": sha256_text(canonical_json(value))}


class G1V2Backend(HFReadoutBackend):
    name = "rwkv_g1_v2"

    def __init__(self, plugin, pack_rows: int = PACK_ROWS, pack_tokens: int = PACK_TOKENS) -> None:
        if plugin.family != "rwkv7" or BOS_ID not in plugin.special_ids or plugin.state_ops is None:
            raise ValueError("jrt-rwkv-g1-v2 needs the RWKV7-G1 plugin with state ops and control id 0")
        self.codes = load_frozen_table()
        self.code_table = [c.code for c in self.codes]
        self.pack_rows, self.pack_tokens = pack_rows, pack_tokens
        self._full_memo: dict[str, list[int]] = {}
        # F4 proof on the plugin's own tokenizer: every code is ONE token after "ANSWER:" in whole-string tokenization
        ctx = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"
        base = plugin.encode(ctx, True)
        for c in self.codes:
            if plugin.encode(ctx + " " + c.code, True) != base + [c.token_id]:
                raise ValueError(f"code {c.code!r} is not one token after the v2 tail")
        super().__init__(plugin, G1V2_TEMPLATE, max_batch_rows=pack_rows, fork=True, fork_max_rows=pack_rows,
                         fork_token_budget=pack_tokens)

    def _build_revision(self) -> dict[str, Any]:
        revision = super()._build_revision()
        config = dict(revision["inference_config"])
        config.update({
            "scoring": "coded: log p(single code token | [0] + v2 prompt); fallback k>codebook: "
                       "sum_logprob(' ' + option_id + '\\n')",
            "scoring_path": f"prefix-fork, one tail row per question, <= {self.pack_rows} rows / "
                            f"{self.pack_tokens} positions per call",
            "tokenization": "whole prompt string, [0] prepended",
            "codebook_table_sha256": TABLE_SHA256,
            "source_sha256": {p.name: sha256_file(p) for p in (Path(__file__), Path(__file__).parents[1] / "prompt.py",
                                                              Path(__file__).with_name("state_fork.py"))},
        })
        cfg_hash = sha256_text(canonical_json(config))
        tmpl = renderer_identity()
        revision.update({
            "prompt_template": tmpl, "inference_config": config, "inference_config_sha256": cfg_hash,
            "revision_id": (f"{self.plugin.family}:{self.plugin.model_id}@{(self.plugin.snapshot or 'local')[:12]}"
                            f";w={self.plugin.files.get('weights', 'unknown')[:12]}"
                            f";tok={self.plugin.files.get('tokenizer', 'unknown')[:12]}"
                            f";prompt={tmpl['id']}.{tmpl['sha256'][:8]};cfg={cfg_hash[:8]}"
                            f";adapter={revision['adapter_version']}"),
        })
        return revision

    def revision(self) -> dict[str, Any]:
        return copy.deepcopy(self._revision)

    # ---------------------------------------------------------------- prompts
    def _full(self, text: str) -> list[int]:
        ids = self._full_memo.get(text)
        if ids is None:
            ids = self.plugin.encode(text, True)
            if BOS_ID in ids:
                raise BackendUnsupported("control_tokens_in_input",
                                         "request text encoded to the control id 0 (<|endoftext|>); refused")
            if len(self._full_memo) > 512:
                self._full_memo.clear()
            self._full_memo[text] = ids
        return [BOS_ID] + ids

    def _questions(self, request: dict[str, Any], label_counts: dict[str, int] | None = None):
        out = []
        for qid in canonical_question_ids(request["questions"]):
            prompt = self._full(render_g1v2(request, qid, self.code_table))
            if label_counts is not None:  # approximate per-label counts (whole-string tokens, not per segment)
                label_counts["total"] = label_counts.get("total", 0) + len(prompt)
            opts = [o for o, _ in canonical_options(request["questions"][qid])]
            coded = len(opts) <= len(self.codes)
            q = {"qid": qid, "prompt": prompt, "options": opts, "coded": coded}
            if coded:
                q["tokens"] = [self.codes[j].token_id for j in range(len(opts))]
            else:
                q["conts"] = [self.plugin.encode(" " + o + G1V2_TEMPLATE.terminator, True) for o in opts]
            out.append(q)
        return out

    def _calls(self, qs):
        coded = [q for q in qs if q["coded"]]
        L = common_prefix_len([q["prompt"] for q in coded]) if coded else 0
        tails = sorted(((q, q["prompt"][L:]) for q in coded), key=lambda t: (-len(t[1]), t[0]["qid"]))
        packs, cur = [], []
        for t in tails:
            if cur and (len(cur) >= self.pack_rows or len(cur[0][1]) * (len(cur) + 1) > self.pack_tokens):
                packs.append(cur)
                cur = []
            cur.append(t)
        if cur:
            packs.append(cur)
        return L, packs

    def plan(self, request: dict[str, Any]) -> Plan:
        labels: dict[str, int] = {}
        qs = self._questions(request, labels)
        L, packs = self._calls(qs)
        n_calls = (1 if packs and L > 0 else 0) + len(packs)
        total = L + sum(len(t) for p in packs for _, t in p)
        for q in qs:
            if not q["coded"]:
                n_calls += 1 + len(q["options"])
                total += len(q["prompt"]) + sum(len(c) for c in q["conts"])
        lengths = [len(q["prompt"]) + (1 if q["coded"] else max(map(len, q["conts"]))) for q in qs]
        return Plan(inference_calls=n_calls, max_call_input_tokens=max(lengths), total_input_tokens=total,
                    per_question={q["qid"]: {"prompt_tokens": len(q["prompt"]), "options": len(q["options"]),
                                             "coded": q["coded"]} for q in qs},
                    prompt_token_labels=labels, provenance_applied=False)

    # ---------------------------------------------------------------- scoring
    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        import torch

        ops, dev = self.plugin.state_ops, self.plugin.device
        qs = self._questions(request)
        L, packs = self._calls(qs)
        logits: dict[str, dict[str, float]] = {}
        call_ms: list[float] = []
        total = 0
        state = None
        if packs:
            check_deadline()
            t0 = time.perf_counter()
            with torch.inference_mode():
                _, state = ops.prefill(torch.tensor([packs[0][0][0]["prompt"][:L]], dtype=torch.long, device=dev))
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            call_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
            total += L
        for pack in packs:
            check_deadline()
            B, W = len(pack), len(pack[0][1])
            ids = torch.full((B, W), PAD_ID, dtype=torch.long)
            last = torch.zeros(B, dtype=torch.long)
            for b, (_, tail) in enumerate(pack):
                ids[b, : len(tail)] = torch.tensor(tail, dtype=torch.long)
                last[b] = len(tail) - 1
            t0 = time.perf_counter()
            with torch.inference_mode():
                lg, _ = ops.extend(state, ids.to(dev), W)  # [B, W, V]
                sel = lg[torch.arange(B, device=lg.device), last.to(lg.device)]  # [B, V]
                if not bool(torch.isfinite(sel).all()):
                    raise BackendUnsupported("non_finite_logits", "nonfinite next-token logits")
                lp = torch.log_softmax(sel.float(), dim=-1)
                for b, (q, _) in enumerate(pack):
                    vals = lp[b].index_select(0, torch.tensor(q["tokens"], device=lp.device)).tolist()
                    logits[q["qid"]] = dict(zip(q["options"], vals))
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            call_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
            total += sum(len(t) for _, t in pack)
        for q in qs:  # fallback: multi-token option ids through the existing fork scorer
            if q["coded"]:
                continue
            rows = [{"qid": q["qid"], "option": o, "prompt": q["prompt"], "cont": c}
                    for o, c in zip(q["options"], q["conts"])]
            lg, n, tok, ms = fork_score(rows, ops, self.pack_rows, dev, check_deadline, self.pack_tokens, None)
            logits[q["qid"]] = lg[q["qid"]]
            call_ms += ms
            total += tok
        return ScoreResult(logits=logits, inference_calls=len(call_ms), total_input_tokens=total,
                           extras={"call_latency_ms": call_ms, "scoring_path": "g1-v2-prefix-fork",
                                   "coded": {q["qid"]: q["coded"] for q in qs}, "prefix_tokens": L})
