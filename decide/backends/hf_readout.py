"""HFReadoutBackend: a stock causal LM scored by option log-likelihood.

For each question q (one row per question, as Kev does for hybrid bases):

    prompt_q  = JRT prompt (schema + options -> state -> re-read of q)
    logit(o)  = sum_t log p(cont_o[t] | prompt_q, cont_o[:t]),
    cont_o    = " " + option_id + terminator

The terminator makes the score a proper sequence probability, so an option id
that is a prefix of another ("bill" / "billing") is not favoured. Option-count
temperature is applied by the adapter (``decide.calibration``), never here.

Rows are grouped by total length so no padding or attention masks are needed
(recurrent / hybrid layers then see exactly the unpadded sequence), and each
group is split into chunks of ``max_batch_rows``. One chunk = one inference
call. Groups and chunks are built from the canonical option order, so batch
composition (and hence float rounding) is invariant to request key order.

Model plug-ins expose ``forward(input_ids [B, T], n_last) -> logits [B, n_last, V]``.
Loaders below cover Qwen3.5 (transformers, hybrid Gated DeltaNet + attention),
RWKV7 (``fla`` package) and Mamba-3 (``mamba_ssm``) when those import.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import versions
from ..prompt import TEMPLATES, Segment, Template, build_prompt_labelled, continuation_segments
from ..provenance import validate_provenance
from ..schema import canonical_json, canonical_options, sha256_text
from ..untrusted import UNTRUSTED_ENCODER_ID, ControlTokenError, UntrustedEncoder
from .base import BackendUnsupported, Plan, ScoreResult
from .state_fork import KV_STATE_TOKEN_BUDGET, HFCacheStateOps, RWKV7StateOps


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class ModelPlugin:
    family: str
    model_id: str
    snapshot: str | None
    #: encode(text, trusted) -> ids. Untrusted text (anything from the request) must be
    #: encoded so it cannot yield control/special tokens (e.g. split_special_tokens=True).
    encode: Callable[[str, bool], list[int]]
    forward: Callable[[Any, int], Any]
    files: dict[str, str] = field(default_factory=dict)  # role -> sha256
    max_context: int | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    notes: dict[str, Any] = field(default_factory=dict)
    kernel_stack: dict[str, Any] = field(default_factory=dict)  # part of the revision hash
    #: ids that must never come from untrusted text (special + added tokens); checked per segment
    special_ids: frozenset[int] = frozenset()
    #: identity of the untrusted-text encoder (decide.untrusted); part of the revision hash
    untrusted_encoder_id: str = UNTRUSTED_ENCODER_ID
    #: optional prefill/extend/select over the model's recurrent state + KV cache (decide.backends.state_fork):
    #: when present the backend reads the shared prompt prefix once and forks per question and option
    state_ops: Any = None
    #: prefix-fork packs copy the prefix state once per row; for a KV-cache model that is rows x |prefix| x KV bytes.
    #: Cap rows x |prefix| per pack (``state_fork.KV_STATE_TOKEN_BUDGET``); None = uncapped (|prefix|-independent state)
    fork_state_token_budget: int | None = None


def _pkg_versions(*names: str) -> dict[str, Any]:
    from importlib import metadata

    out: dict[str, Any] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:
            out[n] = None
    return out


class HFReadoutBackend:
    name = "hf_readout"

    def __init__(self, plugin: ModelPlugin, template: str | Template, max_batch_rows: int = 16,
                 fork: bool | None = None, fork_max_rows: int = 256, fork_token_budget: int = 4096) -> None:
        self.plugin = plugin
        self.template = TEMPLATES[template] if isinstance(template, str) else template
        self.max_batch_rows = max_batch_rows
        #: segment -> ids memo. Encoding is a pure function of (text, trusted) for a fixed plugin, and every question
        #: of a request re-encodes the same schema + state segments (and plan() then score() repeats it): a
        #: 17-question 4.5k-token row was 34 Python tokenisations of the state. Ids are identical, only CPU time moves.
        self._enc_memo: dict[tuple[str, bool], list[int]] = {}
        #: prefix-fork scoring (decide.backends.state_fork) when the plugin exposes state ops; default on if available
        self.fork = (plugin.state_ops is not None) if fork is None else bool(fork)
        self.fork_max_rows, self.fork_token_budget = fork_max_rows, fork_token_budget
        if self.fork and plugin.state_ops is None:
            raise ValueError("fork=True needs a plugin with state_ops")
        self._revision = self._build_revision()

    # ----------------------------------------------------------------- identity
    def _build_revision(self) -> dict[str, Any]:
        import torch

        try:
            import transformers

            tf_version = transformers.__version__
        except Exception:  # pragma: no cover
            tf_version = None
        inference_config = {
            "scoring": "sum_logprob(' ' + option_id + terminator)",
            "untrusted_encoding": self.plugin.untrusted_encoder_id,
            "rows": "one per question; grouped by length; unpadded",
            # different call structure -> different float rounding -> different revision id
            "scoring_path": (f"prefix-fork(shared prefix once; packed tails, <= {self.fork_max_rows} rows / "
                             f"{self.fork_token_budget} positions per call)") if self.fork
            else "recompute(prompt + continuation per option)",
            "max_batch_rows": self.max_batch_rows,
            **({"fork_state_token_budget": self.plugin.fork_state_token_budget}
               if self.fork and self.plugin.fork_state_token_budget is not None else {}),
            "dtype": self.plugin.dtype,
            "device": self.plugin.device,
            "torch": torch.__version__,
            "transformers": tf_version,
            "kernel_stack": self.plugin.kernel_stack,
        }
        cfg_hash = sha256_text(canonical_json(inference_config))
        tmpl = self.template.identity()
        weights = self.plugin.files.get("weights", "unknown")
        tok = self.plugin.files.get("tokenizer", "unknown")
        revision_id = (
            f"{self.plugin.family}:{self.plugin.model_id}@{(self.plugin.snapshot or 'local')[:12]}"
            f";w={weights[:12]};tok={tok[:12]};prompt={tmpl['id']}.{tmpl['sha256'][:8]}"
            f";cfg={cfg_hash[:8]};adapter={versions.ADAPTER_VERSION}"
        )
        return {
            "revision_id": revision_id,
            "backend": self.name,
            "family": self.plugin.family,
            "model_id": self.plugin.model_id,
            "snapshot": self.plugin.snapshot,
            "files_sha256": self.plugin.files,
            "prompt_template": tmpl,
            "inference_config": inference_config,
            "inference_config_sha256": cfg_hash,
            "adapter_version": versions.ADAPTER_VERSION,
            "max_context": self.plugin.max_context,
            "notes": self.plugin.notes,
        }

    def revision(self) -> dict[str, Any]:
        return self._revision

    # --------------------------------------------------------------- planning
    def _encode(self, segments: list[Segment], counts: dict[str, int] | None = None,
                labels: list[str] | None = None) -> list[int]:
        """Encode (text, trusted) segments one by one (no BPE merge across segments).
        Untrusted segments go through decide.untrusted and fail closed on control ids."""
        ids: list[int] = []
        for k, (text, trusted) in enumerate(segments):
            key = (text, bool(trusted))
            seg_ids = self._enc_memo.get(key)
            if seg_ids is None:
                try:
                    seg_ids = self.plugin.encode(text, trusted)
                except ControlTokenError as exc:
                    raise BackendUnsupported(exc.code, str(exc)) from exc
                if len(self._enc_memo) > 4096:  # bounded: plan + score of a few requests
                    self._enc_memo.clear()
                self._enc_memo[key] = seg_ids
            if not trusted and self.plugin.special_ids and any(i in self.plugin.special_ids for i in seg_ids):
                raise BackendUnsupported(
                    "control_tokens_in_input", "request text encoded to control/special token ids; refused"
                )
            ids += seg_ids
            if counts is not None and labels is not None:
                counts[labels[k]] = counts.get(labels[k], 0) + len(seg_ids)
        return ids

    @staticmethod
    def _provenance(request: dict[str, Any]) -> dict[str, Any] | None:
        env = request.get("ssdm") if isinstance(request.get("ssdm"), dict) else {}
        raw = env.get("provenance")
        return None if raw is None else validate_provenance(request, raw)

    def _rows(self, request: dict[str, Any], label_counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
        rows = []
        prov = self._provenance(request)
        for qid in request["questions"]:
            labelled = build_prompt_labelled(self.template, request, qid, prov)
            labels = [lab for _, lab in labelled]
            prompt_ids = self._encode([(t, lab == "template") for t, lab in labelled], label_counts, labels)
            options = [o for o, _ in canonical_options(request["questions"][qid])]
            for o in options:
                cont = self._encode(continuation_segments(self.template, o))
                if not cont:
                    raise BackendUnsupported("tokenization_failed", f"{qid}/{o}: empty continuation")
                rows.append({"qid": qid, "option": o, "prompt": prompt_ids, "cont": cont})
        return rows

    def _chunks(self, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        chunks = []
        by_q: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_q.setdefault(r["qid"], []).append(r)
        for qid, q_rows in by_q.items():  # one question's rows share a prompt
            groups: dict[int, list[dict[str, Any]]] = {}
            for r in q_rows:
                groups.setdefault(len(r["cont"]), []).append(r)
            for length in sorted(groups):
                g = groups[length]
                for i in range(0, len(g), self.max_batch_rows):
                    chunks.append(g[i : i + self.max_batch_rows])
        return chunks

    def _use_fork(self, rows: list[dict[str, Any]]) -> bool:
        from .state_fork import common_prefix_len

        if not self.fork:
            return False
        prompts = {r["qid"]: r["prompt"] for r in rows}
        return common_prefix_len(list(prompts.values())) >= 1

    def plan(self, request: dict[str, Any]) -> Plan:
        label_counts: dict[str, int] = {}
        rows = self._rows(request, label_counts)
        lengths = [len(r["prompt"]) + len(r["cont"]) for r in rows]
        if self._use_fork(rows):
            from .state_fork import fork_plan

            _, calls = fork_plan(rows, self.fork_max_rows, self.fork_token_budget, self.plugin.fork_state_token_budget)
            n_calls, total = len(calls), sum(c.tokens for c in calls)
        else:
            chunks = self._chunks(rows)
            n_calls, total = len(chunks), sum(lengths)
        per_q: dict[str, dict[str, Any]] = {}
        for r in rows:
            d = per_q.setdefault(r["qid"], {"prompt_tokens": len(r["prompt"]), "options": 0})
            d["options"] += 1
        return Plan(
            inference_calls=n_calls,
            # the context the model conditions on is still prompt + continuation, whichever path computes it
            max_call_input_tokens=max(lengths),
            total_input_tokens=total,
            per_question=per_q,
            prompt_token_labels=label_counts,
        )

    # ---------------------------------------------------------------- scoring
    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        import torch

        rows = self._rows(request)
        if self._use_fork(rows):
            from .state_fork import fork_score

            logits, n_calls, total_tokens, call_ms = fork_score(rows, self.plugin.state_ops, self.fork_max_rows,
                                                                self.plugin.device, check_deadline,
                                                                self.fork_token_budget,
                                                                self.plugin.fork_state_token_budget)
            return self._result(rows, logits, n_calls, total_tokens, call_ms)
        chunks = self._chunks(rows)
        logits: dict[str, dict[str, float]] = {qid: {} for qid in request["questions"]}
        call_ms = []
        total_tokens = 0
        for chunk in chunks:
            check_deadline()
            n_cont = len(chunk[0]["cont"])
            seqs = [r["prompt"] + r["cont"] for r in chunk]
            ids = torch.tensor(seqs, dtype=torch.long, device=self.plugin.device)
            total_tokens += ids.numel()
            t0 = time.perf_counter()
            with torch.inference_mode():
                # positions predicting the continuation: last n_cont+1 .. last-1
                out = self.plugin.forward(ids, n_cont + 1)
                logp = torch.log_softmax(out[:, :-1, :].float(), dim=-1)
                target = ids[:, -n_cont:]
                tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, n_cont]
                seq_lp = tok_lp.sum(-1).tolist()
            if self.plugin.device.startswith("cuda"):
                torch.cuda.synchronize()
            call_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
            for r, lp in zip(chunk, seq_lp):
                logits[r["qid"]][r["option"]] = lp
        return self._result(rows, logits, len(chunks), total_tokens, call_ms)

    def _result(self, rows, logits, n_calls, total_tokens, call_ms) -> ScoreResult:
        return ScoreResult(
            logits=logits,
            inference_calls=n_calls,
            total_input_tokens=total_tokens,
            extras={
                "call_latency_ms": call_ms,
                "rows": len(rows),
                "scoring_path": "prefix-fork" if self._use_fork(rows) else "recompute",
                # diagnostic only: sum-logprob carries a token-length prior over option ids
                "continuation_tokens": {
                    qid: {r["option"]: len(r["cont"]) for r in rows if r["qid"] == qid} for qid in logits
                },
                "length_normalized_logits": {
                    qid: {r["option"]: logits[qid][r["option"]] / len(r["cont"]) for r in rows if r["qid"] == qid}
                    for qid in logits
                },
            },
        )


# ------------------------------------------------------------------- loaders
def _snapshot_of(model_dir: Path) -> str | None:
    # HF cache layout: .../models--org--name/snapshots/<commit>/
    if model_dir.parent.name == "snapshots":
        return model_dir.name
    return None


def _hash_files(model_dir: Path, patterns: list[str]) -> str:
    files = sorted(p for pat in patterns for p in model_dir.glob(pat))
    if not files:
        return "missing"
    if len(files) == 1:
        return sha256_file(files[0].resolve())
    return sha256_text("".join(f"{f.name}:{sha256_file(f.resolve())}\n" for f in files))


#: Trained context of the small recurrent checkpoints (RWKV7 config max_position_embeddings;
#: Mamba-3 SISO model card "Context length 2,048"). Recurrent models do not truncate, they
#: degrade silently past it, so it is enforced as the backend input cap.
TRAINED_CTX_DEFAULT = 2048


def load_qwen35(model_dir: str | Path, device: str = "cuda", dtype: str = "bfloat16", model_id: str | None = None) -> ModelPlugin:
    """Qwen3.5-* via transformers (text-only causal LM view of the checkpoint)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=getattr(torch, dtype), device_map={"": device}
    ).eval()
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config", cfg)
    layer_types = text_cfg.get("layer_types") or []

    untrusted = UntrustedEncoder(tok)

    def encode(text: str, trusted: bool = True) -> list[int]:
        if trusted:
            return tok(text, add_special_tokens=False).input_ids
        return untrusted.encode(text)

    def forward(ids, n_last: int):
        return model(input_ids=ids, logits_to_keep=n_last, use_cache=False).logits

    return ModelPlugin(
        family="qwen3.5",
        model_id=model_id or cfg.get("_name_or_path") or model_dir.name,
        snapshot=_snapshot_of(model_dir),
        encode=encode,
        forward=forward,
        special_ids=untrusted.control_ids,
        files={
            "weights": _hash_files(model_dir, ["*.safetensors"]),
            "tokenizer": _hash_files(model_dir, ["tokenizer.json"]),
            "config": _hash_files(model_dir, ["config.json"]),
        },
        max_context=text_cfg.get("max_position_embeddings"),
        dtype=dtype,
        device=device,
        kernel_stack=_pkg_versions("flash-linear-attention", "fla-core", "causal-conv1d", "triton", "triton-windows"),
        state_ops=HFCacheStateOps(model),
        fork_state_token_budget=KV_STATE_TOKEN_BUDGET if "full_attention" in layer_types or not layer_types else None,
        notes={
            "model_class": type(model).__name__,
            "layer_types": {t: layer_types.count(t) for t in sorted(set(layer_types))},
            "attention_free": False,
        },
    )


def load_rwkv7(model_dir: str | Path, device: str = "cuda", dtype: str = "bfloat16", model_id: str | None = None) -> ModelPlugin:
    """RWKV7 via the ``fla`` package's registered RWKV7ForCausalLM (no remote code)."""
    import torch

    try:
        import fla  # noqa: F401  (registers rwkv7 with transformers Auto classes)
    except Exception as exc:
        raise ImportError(f"RWKV7 needs the flash-linear-attention (fla) package: {exc!r}") from exc
    from transformers import AutoModelForCausalLM

    from ..rwkv_world_tokenizer import RWKVWorldTokenizer

    model_dir = Path(model_dir)
    tok = RWKVWorldTokenizer(model_dir / "rwkv_vocab_v20230424.txt")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=getattr(torch, dtype), device_map={"": device}, trust_remote_code=False
    ).eval()

    def forward(ids, n_last: int):
        return model(input_ids=ids, use_cache=False).logits[:, -n_last:, :]

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return ModelPlugin(
        family="rwkv7",
        model_id=model_id or model_dir.name,
        snapshot=_snapshot_of(model_dir),
        encode=lambda text, trusted=True: tok.encode(text),  # World tokenizer has no control tokens
        untrusted_encoder_id="rwkv-world-greedy (no control tokens)",
        forward=forward,
        files={
            "weights": _hash_files(model_dir, ["*.safetensors"]),
            "tokenizer": _hash_files(model_dir, ["rwkv_vocab_v20230424.txt"]),
            "config": _hash_files(model_dir, ["config.json"]),
        },
        max_context=cfg.get("max_position_embeddings") or TRAINED_CTX_DEFAULT,  # trained ctx, enforced
        dtype=dtype,
        device=device,
        kernel_stack=_pkg_versions("flash-linear-attention", "fla-core", "triton", "triton-windows"),
        notes={"model_class": type(model).__name__, "trained_ctx": cfg.get("max_position_embeddings"), "attention_free": True},
    )


#: README of RWKV/RWKV7-G1k-13.3B-20260930: "Training context: 25600 tokens" (config.json carries no context field)
RWKV7_G1K_TRAINED_CTX = 25600


def load_rwkv7_hf(model_dir: str | Path, device: str = "cuda", dtype: str = "bfloat16", model_id: str | None = None,
                  wkv: str | None = None, repo: str | None = None, trained_ctx: int = RWKV7_G1K_TRAINED_CTX) -> ModelPlugin:
    """RWKV7-G1k (``RWKV/RWKV7-G1k-*-20260930``): the repo's bundled HF code, vendored and sha-pinned in
    ``third_party/rwkv7_hf`` (no ``trust_remote_code``). ``load_rwkv7`` above cannot read these repos: they ship
    ``tokenizer.json`` (no ``rwkv_vocab_v20230424.txt``) and are not the ``fla`` class.

    Untrusted text: the World tokenizer's only special token is ``<|endoftext|>`` (id 0) and ordinary bytes never
    map to it, so untrusted text that yields any special id is refused (fail closed, ``ControlTokenError``)."""
    import torch

    import third_party.rwkv7_hf as V
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

    model_dir = Path(model_dir)
    repo = repo or V.REPO  # another G1k size of the same release: same code, its own pinned config.json
    code = V.verify_snapshot_code(model_dir, repo)
    tok = Rwkv7Tokenizer.from_pretrained(str(model_dir))
    special = frozenset(int(i) for i in tok.all_special_ids) | {0}
    model = Rwkv7ForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype), device_map={"": device}).eval()
    if wkv is not None:  # heads/rwkv7_wkv.py ("fla" refuses until GPU-verified); None keeps the checkpoint's "chunked"
        from heads.rwkv7_wkv import select_wkv

        select_wkv(model, wkv)

    def encode(text: str, trusted: bool = True) -> list[int]:
        ids = tok(text, add_special_tokens=False).input_ids
        if not trusted and special.intersection(ids):
            raise ControlTokenError("untrusted text encodes to an RWKV special token (<|endoftext|>)")
        return ids

    def forward(ids, n_last: int):
        return model(input_ids=ids, logits_to_keep=n_last, use_cache=False).logits

    return ModelPlugin(
        family="rwkv7",
        model_id=model_id or repo,
        snapshot=_snapshot_of(model_dir),
        encode=encode,
        untrusted_encoder_id="rwkv7-hf-trie (fail closed on <|endoftext|>)",
        forward=forward,
        special_ids=special,
        files={
            "weights": _hash_files(model_dir, ["*.safetensors"]),
            "tokenizer": _hash_files(model_dir, ["tokenizer.json"]),
            "config": _hash_files(model_dir, ["config.json"]),
            "code": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
        },
        max_context=trained_ctx,
        dtype=dtype,
        device=device,
        kernel_stack={**_pkg_versions("transformers", "torch"), "wkv_implementation": model.config.wkv_implementation},
        state_ops=RWKV7StateOps(model),
        notes={"model_class": type(model).__name__, "trained_ctx": trained_ctx, "attention_free": True,
               "vendored_code": V.REVISION, "repo": repo},
    )


def load_hf_causal(model_dir: str | Path, family: str, device: str = "cuda", dtype: str = "bfloat16",
                   model_id: str | None = None, fp32_head: bool = False, strict: bool = True) -> ModelPlugin:
    """Any stock transformers causal LM (``AutoModelForCausalLM``, no remote code) as a plug-in WITHOUT state ops: every
    readout on it recomputes the whole prompt (no prefix fork; job-gen87: fork != recompute in bf16 on hybrid models).

    ``strict``: refuse a load that reports missing / unexpected / mismatched weights (a native transformers class that
    does not match the checkpoint would otherwise score random heads). ``fp32_head``: the lm head is applied in fp32
    to the final hidden state (vLLM's ``head_dtype: float32``); the plug-in then also exposes ``plugin.notes['fp32_head']``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model, info = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=getattr(torch, dtype), device_map={"": device}, trust_remote_code=False,
        output_loading_info=True)
    model = model.eval()
    # missing / mismatched = a module left at random init: refused. Unexpected = checkpoint weights the text model does
    # not use (a multimodal checkpoint's vision / audio towers read as a causal LM): recorded only.
    bad = {k: sorted(map(str, v))[:8] for k, v in (info or {}).items() if v and k in ("missing_keys", "mismatched_keys")}
    unexpected = sorted(map(str, (info or {}).get("unexpected_keys") or []))
    if strict and bad:
        raise RuntimeError(f"{model_dir.name}: weights do not match {type(model).__name__}: {bad}")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config", cfg)
    untrusted = UntrustedEncoder(tok)
    head = model.get_output_embeddings()
    captured: dict[str, Any] = {}
    if fp32_head and text_cfg.get("final_logit_softcapping"):
        raise ValueError("fp32_head replaces the logits; a model with final_logit_softcapping would lose its softcap")
    if fp32_head:
        head.register_forward_hook(lambda mod, args, out: captured.__setitem__("h", args[0]))

    def encode(text: str, trusted: bool = True) -> list[int]:
        if trusted:
            return tok(text, add_special_tokens=False).input_ids
        return untrusted.encode(text)

    keep = {"ok": True}

    def forward(ids, n_last: int):
        if keep["ok"]:
            try:
                out = model(input_ids=ids, logits_to_keep=n_last, use_cache=False).logits
            except TypeError:  # an architecture without logits_to_keep: full logits, sliced below
                keep["ok"] = False
                out = model(input_ids=ids, use_cache=False).logits
        else:
            out = model(input_ids=ids, use_cache=False).logits
        if fp32_head:
            h = captured.pop("h")[:, -n_last:, :]
            out = torch.nn.functional.linear(h.float(), head.weight.float(),
                                             None if head.bias is None else head.bias.float())
        return out[:, -n_last:, :]

    return ModelPlugin(
        family=family,
        model_id=model_id or cfg.get("_name_or_path") or model_dir.name,
        snapshot=_snapshot_of(model_dir),
        encode=encode,
        forward=forward,
        special_ids=untrusted.control_ids,
        files={
            "weights": _hash_files(model_dir, ["*.safetensors"]),
            "tokenizer": _hash_files(model_dir, ["tokenizer.json"]),
            "config": _hash_files(model_dir, ["config.json"]),
        },
        max_context=text_cfg.get("max_position_embeddings"),
        dtype=dtype,
        device=device,
        kernel_stack=_pkg_versions("transformers", "torch", "flash-linear-attention", "fla-core", "mamba-ssm",
                                   "causal-conv1d", "triton"),
        notes={"model_class": type(model).__name__, "fp32_head": bool(fp32_head),
               "unexpected_keys": {"n": len(unexpected), "first": unexpected[:8]},
               "layer_types": {t: (text_cfg.get("layer_types") or []).count(t)
                               for t in sorted(set(text_cfg.get("layer_types") or []))},
               "final_logit_softcapping": text_cfg.get("final_logit_softcapping")},
    )


def load_mamba3(model_dir: str | Path, tokenizer_dir: str | Path, device: str = "cuda", dtype: str = "bfloat16", model_id: str | None = None) -> ModelPlugin:
    """Mamba-3 SISO via ``mamba_ssm`` (CUDA kernels) and the Llama-3.1 tokenizer."""
    import torch

    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    except Exception as exc:
        raise ImportError(f"Mamba-3 needs the mamba_ssm package with Mamba3 support: {exc!r}") from exc
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = MambaLMHeadModel.from_pretrained(str(model_dir), device=device, dtype=getattr(torch, dtype)).eval()

    untrusted = UntrustedEncoder(tok)

    def encode(text: str, trusted: bool = True) -> list[int]:
        if trusted:
            return tok(text, add_special_tokens=False).input_ids
        return untrusted.encode(text)

    def forward(ids, n_last: int):
        return model(ids, num_last_tokens=n_last).logits

    return ModelPlugin(
        family="mamba3",
        model_id=model_id or model_dir.name,
        snapshot=_snapshot_of(model_dir),
        encode=encode,
        forward=forward,
        special_ids=untrusted.control_ids,
        files={
            "weights": _hash_files(model_dir, ["*.bin", "*.safetensors"]),
            "tokenizer": _hash_files(Path(tokenizer_dir), ["tokenizer.json"]),
            "config": _hash_files(model_dir, ["config.json"]),
        },
        max_context=TRAINED_CTX_DEFAULT,
        dtype=dtype,
        device=device,
        kernel_stack=_pkg_versions("mamba_ssm", "triton", "triton-windows"),
        notes={"attention_free": True, "tokenizer": "Llama-3.1 tokenizer (ungated mirror)"},
    )
