"""v3w-generic readout backend: ``decide.v3w_generic.make_v3w(backbone_id)`` scored on any plug-in with state ops.

The v3w readout for a non-RWKV7 backbone (the backbone's OWN chat template, its own single-token code table, word ids
by their unique first token, whole-string tokenization with special tokens only where the format puts them), scored
on the G1V2 path: the shared token prefix of every question prompt prefilled once, each question's tail extended from
that state in packed calls, one next-token read per question after "ANSWER:" (coded / word-id questions), and the
multi-token fallback (k > code table) through ``state_fork.fork_score``. Prompt ids, option order and scored tokens
are exactly ``V3W.questions(request)``; this module adds no prompt text of its own.

Used by the backbone scout's readout-artifact check (jobs/backbone_screen MODE=qrwkv6_32b_v3w, arm b): the same
weights as the stock chatml screen, only the readout changes.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..prompt import Template
from ..schema import canonical_json, sha256_text
from .hf_readout import HFReadoutBackend, sha256_file
from .rwkv_g1_v2 import G1V2Backend

#: rows per packed tail call: each row carries its own copy of the prefix state (RWKV6-Qwen2 32B: ~168 MB / row)
PACK_ROWS = 16
PACK_TOKENS = 4096


class V3WGenericBackend(G1V2Backend):
    name = "v3w_generic"

    def __init__(self, plugin, v3w, pack_rows: int = PACK_ROWS, pack_tokens: int = PACK_TOKENS) -> None:
        if plugin.state_ops is None:
            raise ValueError("v3w-generic needs a plug-in with state ops (prefix prefill + packed tails)")
        if v3w.is_legacy_rwkv:
            raise ValueError("RWKV7-G1k v3w is G1V3Backend(mode='words'); this backend is for the generic formats")
        self.v3w = v3w
        self.codes = v3w.codes  # len() bounds the coded path (G1V2Backend.plan / score read q["coded"] only)
        self.code_table = list(v3w.table)
        self.pack_rows, self.pack_tokens = pack_rows, pack_tokens
        ident = v3w.identity()
        tmpl = Template(id=ident["id"], head=v3w.fmt.head, tail=v3w.fmt.tail, terminator="\n")
        self._v3w_identity = ident
        HFReadoutBackend.__init__(self, plugin, tmpl, max_batch_rows=pack_rows, fork=True, fork_max_rows=pack_rows,
                                  fork_token_budget=pack_tokens)

    def _build_revision(self) -> dict[str, Any]:
        revision = HFReadoutBackend._build_revision(self)
        config = dict(revision["inference_config"])
        here = Path(__file__)
        config.update({
            "scoring": "coded / word-id: log p(single answer token | v3w prompt); fallback k > code table: "
                       "sum_logprob(' ' + option_id + '\\n')",
            "scoring_path": f"prefix-fork, one tail row per question, <= {self.pack_rows} rows / "
                            f"{self.pack_tokens} positions per call",
            "tokenization": "whole prompt string (decide.v3w_generic HFEncoder; specials only where the format puts them)",
            "v3w_generic": self._v3w_identity,
            "codes": len(self.codes),
            "source_sha256": {p.name: sha256_file(p) for p in (here, here.parents[1] / "v3w_generic.py",
                                                              here.parents[1] / "prompt_g1v3.py",
                                                              here.with_name("rwkv_g1_v2.py"),
                                                              here.with_name("state_fork.py"))},
        })
        cfg_hash = sha256_text(canonical_json(config))
        tmpl = self._v3w_identity
        revision.update({
            "backend": self.name, "prompt_template": tmpl, "inference_config": config, "inference_config_sha256": cfg_hash,
            "revision_id": (f"{self.plugin.family}:{self.plugin.model_id}@{(self.plugin.snapshot or 'local')[:12]}"
                            f";w={self.plugin.files.get('weights', 'unknown')[:12]}"
                            f";tok={self.plugin.files.get('tokenizer', 'unknown')[:12]}"
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
