"""LM-logit v3w readout for any HF backbone: ``decide.v3w_generic.V3W`` prompts scored by a prefix-fork plugin.

``V3WGenericBackend(plugin, v3w)`` scores every question exactly as ``G1V2Backend`` / ``G1V3Backend(mode="words")``
do (shared prompt prefix prefilled once, each question's tail extended from it in packed calls, the next-token
distribution after "ANSWER:" read once per question), with the PROMPTS and SCORED TOKENS taken verbatim from
``v3w.questions(request)`` (PR #236: the backbone's own chat template, the generic single-token code table, word ids by
their unique first token, the code key ladder under the budget). So:

  * coded / word question: logit(option) = log p(that option's single token | v3w prompt);
  * fallback (k > 255 codes): logit(option) = sum log p(" <option id>\\n" | prompt) through ``state_fork.fork_score``.

It is attention-free end to end whenever the plugin's model is (an all-GDN Q9-AF backbone): no head, no attention
pooling, only the LM's next-token logits. jobs/q9af_v2 mode clefflash uses it as arm-pair B (stock vs converted
Clef-flash backbone through the SAME readout). The fork packs respect the plugin's ``fork_state_token_budget`` (a kept
attention layer's KV is copied once per packed row; job-5cmat's POP909-CL OOM).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..prompt import G1V2_TEMPLATE
from ..schema import canonical_json, sha256_text
from .hf_readout import HFReadoutBackend, sha256_file
from .rwkv_g1_v2 import PACK_ROWS, PACK_TOKENS, G1V2Backend


class V3WGenericBackend(G1V2Backend):
    name = "v3w_generic"

    def __init__(self, plugin, v3w, pack_rows: int = PACK_ROWS, pack_tokens: int = PACK_TOKENS) -> None:
        if plugin.state_ops is None:
            raise ValueError("the v3w generic readout needs a plugin with state ops (prefix fork)")
        self.v3w = v3w
        self.codes = tuple(v3w.codes)
        self.code_table = list(v3w.table)
        self.pack_rows, self.pack_tokens = pack_rows, pack_tokens
        self._full_memo: dict[str, list[int]] = {}
        # G1V2Backend.__init__ is RWKV-only (control id 0, frozen RWKV table); the generic V3W proved its own codes
        HFReadoutBackend.__init__(self, plugin, G1V2_TEMPLATE, max_batch_rows=pack_rows, fork=True,
                                  fork_max_rows=pack_rows, fork_token_budget=pack_tokens)

    def _build_revision(self) -> dict[str, Any]:
        revision = HFReadoutBackend._build_revision(self)
        config = dict(revision["inference_config"])
        here = Path(__file__)
        tmpl = self.v3w.identity()
        config.update({
            "scoring": "coded/word: log p(single scored token | v3w prompt); fallback k>255: "
                       "sum_logprob(' ' + option_id + '\\n')",
            "scoring_path": (f"prefix-fork, one tail row per question, <= {self.pack_rows} rows / "
                             f"{self.pack_tokens} positions per call, rows x prefix <= "
                             f"{self.plugin.fork_state_token_budget}"),
            "prompts": "decide.v3w_generic.V3W.questions (verbatim)",
            "v3w_identity": tmpl,
            "source_sha256": {p.name: sha256_file(p) for p in (here, here.parents[1] / "v3w_generic.py",
                                                              here.with_name("rwkv_g1_v2.py"),
                                                              here.with_name("state_fork.py"))},
        })
        cfg_hash = sha256_text(canonical_json(config))
        revision.update({
            "backend": self.name, "prompt_template": tmpl, "inference_config": config, "inference_config_sha256": cfg_hash,
            "revision_id": (f"{self.plugin.family}:{self.plugin.model_id}@{(self.plugin.snapshot or 'local')[:12]}"
                            f";w={self.plugin.files.get('weights', 'unknown')[:12]}"
                            f";gdn={self.plugin.files.get('gdn_ckpt', 'none')[:12]}"
                            f";prompt={tmpl['id']}.{tmpl['sha256'][:8]};cfg={cfg_hash[:8]}"
                            f";adapter={revision['adapter_version']}"),
        })
        return revision

    def revision(self) -> dict[str, Any]:
        return copy.deepcopy(self._revision)

    def _questions(self, request: dict[str, Any], label_counts: dict[str, int] | None = None):
        qs = self.v3w.questions(request)
        if label_counts is not None:
            label_counts["total"] = label_counts.get("total", 0) + sum(len(q["prompt"]) for q in qs)
        return qs

    def _calls(self, qs):
        L, packs = super()._calls(qs)
        budget = self.plugin.fork_state_token_budget
        if not budget or L <= 0:
            return L, packs
        cap = max(1, budget // L)
        return L, [p[i:i + cap] for p in packs for i in range(0, len(p), cap)]
