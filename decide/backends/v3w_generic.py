"""Generic v3w readout backend: the v3w readout (decide.prompt_g1v3 v3w body: v3 code key, word ids by their unique first
token) of ANY backbone ``decide.v3w_generic.make_v3w`` knows, scored on that backbone's own chat format and tokenizer.

Scoring is the G1 v2/v3 path (``G1V3Backend(mode="words")`` with the spec's questions):
  * prompt ids = ``spec.questions(request)``: the format's ``bos`` + the whole prompt string tokenized once, special
    tokens only where the chat format puts them (request text that encodes to a control token is refused);
  * the shared token prefix of every question prompt is prefilled ONCE through ``plugin.state_ops`` (the shared-prefix
    fork), each question's tail is extended from that state in packed calls (right-padded; the read position is each
    row's own last prompt token, so padding is never read), and the next-token distribution after "ANSWER:" is read
    once per question;
  * coded question: logit(option) = log p(" <CODE>" | prompt), the spec's single-token code; word-id question:
    log p(first token of " <id>" | prompt); k > code table: sum log p(" <id>\\n"), multi-token, through the fork scorer.
The scored token ids ARE the training continuations of ``spec.train_row`` (scripts/check_q9af_v3w_equivalence.py).

``V3WGenericBackend(plugin, spec)``: any decide plugin with ``state_ops`` whose tokenizer is the spec's (checked on the
answer prefix at construction). For Qwen3.5-9B: ``v3w_qwen_plugin`` = ``convert.readout.load_converted_plugin`` (stock
with ``gdn_ckpt=None``, Q9-AF with the converted GDN checkpoint: every attention layer run as GDN).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..adapter import DEFAULT_LIMITS
from ..prompt import Template
from ..schema import canonical_json, sha256_text
from .hf_readout import HFReadoutBackend, sha256_file
from .rwkv_g1_filtered_codes import TokenCode
from .rwkv_g1_v2 import PACK_ROWS, PACK_TOKENS
from .rwkv_g1_v3 import G1V3Backend


def spec_template(spec) -> Template:
    """The ``decide.prompt.Template`` stand-in of a generic spec (heads.layout.resolve_template returns the same)."""
    from ..v3w_generic import TERMINATOR

    return Template(id=spec.template_id, head=spec.fmt.head, tail=spec.fmt.tail, terminator=TERMINATOR)


class V3WGenericBackend(G1V3Backend):
    """G1V3Backend(mode="words") with the spec's prompts and tokens (no RWKV-only checks)."""

    def __init__(self, plugin, spec, pack_rows: int = PACK_ROWS, pack_tokens: int = PACK_TOKENS) -> None:
        if plugin.state_ops is None:
            raise ValueError("generic v3w readout needs a plugin with state ops (shared-prefix fork)")
        self.spec = spec
        self.mode, self.calibrate, self.reread, self.permute, self.words = "words", False, False, False, True
        self._first_tok: dict[str, int | None] = {}
        self.name = f"v3w_generic:{spec.backbone.family}"
        ctx = plugin.max_context or DEFAULT_LIMITS["max_input_tokens"]
        if spec.budget > ctx - 1:  # the spec's catalog ladder would accept prompts the model cannot read
            raise ValueError(f"spec budget {spec.budget} > the plugin's context {ctx} - 1")
        self.budget = spec.budget
        self._cf_memo: dict = {}
        self.cf_stats = {"memo_hits": 0, "memo_misses": 0}
        self.codes = tuple(TokenCode(code=e["code"], token_id=e["token_id"], source_ordinal=i)
                           for i, e in enumerate(spec.codes))
        self.code_table = list(spec.table)
        self.code_token = dict(spec.code_token)
        self.pack_rows, self.pack_tokens = pack_rows, pack_tokens
        self._full_memo: dict[str, list[int]] = {}
        # the plugin must tokenize exactly as the spec (same tokenizer files): checked on the full answer prefix
        # (chat head with its special tokens + v3w instruction + tail + "ANSWER:") and on every code continuation
        want = spec.prompt_ids(spec.fmt.answer_ctx) if not spec.is_legacy_rwkv else None
        got = list(spec.fmt.bos) + list(plugin.encode(spec.fmt.answer_ctx, True))
        if want is not None and got != want:
            raise ValueError("plugin tokenizer != the v3w spec's on the answer prefix")
        base = plugin.encode(spec.fmt.answer_ctx, True)
        for c in self.codes:
            if plugin.encode(spec.fmt.answer_ctx + " " + c.code, True) != base + [c.token_id]:
                raise ValueError(f"v3w code {c.code!r} is not one token after the answer prefix (plugin tokenizer)")
        HFReadoutBackend.__init__(self, plugin, spec_template(spec), max_batch_rows=pack_rows, fork=True,
                                  fork_max_rows=pack_rows, fork_token_budget=pack_tokens)

    def _build_revision(self) -> dict[str, Any]:
        revision = HFReadoutBackend._build_revision(self)
        config = dict(revision["inference_config"])
        here = Path(__file__)
        config.update({
            "scoring": "coded / word ids: log p(single answer token | spec prompt ids); k > code table: "
                       "sum_logprob(' ' + option_id + '\\n')",
            "scoring_path": f"prefix-fork, one tail row per question, <= {self.pack_rows} rows / "
                            f"{self.pack_tokens} positions per call",
            "tokenization": "decide.v3w_generic spec: format bos + whole prompt string, specials only where the "
                            "chat format puts them",
            "template_variant": self.spec.template_id,
            "word_ids": "first token of ' <id>' when ids are non-opaque and first tokens unique",
            "catalog": f"code key for coded k >= catalog min; label ladder under a {self.budget}-token budget",
            "source_sha256": {p.name: sha256_file(p) for p in (
                here, here.parents[1] / "v3w_generic.py", here.parents[1] / "prompt_g1v3.py",
                here.with_name("rwkv_g1_v3.py"), here.with_name("rwkv_g1_v2.py"), here.with_name("state_fork.py"))},
        })
        cfg_hash = sha256_text(canonical_json(config))
        tmpl = self.spec.identity()
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

    def first_token(self, oid: str) -> int | None:
        return self.spec.first_token(oid)

    def word_qids(self, request: dict[str, Any]) -> frozenset[str]:
        return self.spec.word_qids(request)

    def _full(self, text: str) -> list[int]:
        return self.spec.prompt_ids(text)

    def _questions(self, request: dict[str, Any], label_counts: dict[str, int] | None = None):
        out = self.spec.questions(request)
        if label_counts is not None:
            label_counts["total"] = label_counts.get("total", 0) + sum(len(q["prompt"]) for q in out)
        return out


def v3w_qwen_plugin(model_dir: str, gdn_ckpt: str | None = None, device: str = "cuda", dtype: str = "bfloat16"):
    """Qwen3.5-9B through convert.readout (the same code path for both arms): ``gdn_ckpt`` None = the stock model; else
    the converted GDN checkpoint's layers (every attention layer for a full Q9-AF export) run as GDN."""
    from convert.readout import load_converted_plugin

    layers: list[int] = []
    if gdn_ckpt:
        import torch

        layers = sorted(int(k) for k in torch.load(gdn_ckpt, map_location="cpu", weights_only=True))
    return load_converted_plugin(model_dir, gdn_ckpt, layers, device=device, dtype=dtype)


def v3w_decider(plugin, template: str, tokenizer_dir: str | None = None):
    """(Decider, backend) of the generic v3w readout ``template`` on ``plugin`` (identity calibration)."""
    from .. import Decider
    from ..v3w_generic import bind_template, spec_for_template

    spec = bind_template(template, tokenizer_dir) if tokenizer_dir else spec_for_template(template)
    backend = V3WGenericBackend(plugin, spec)
    return Decider(backend, backend_max_input_tokens=plugin.max_context), backend
