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
from ..prompt import TEMPLATES, Template, build_prompt, continuation
from ..schema import canonical_json, canonical_options, sha256_text
from .base import BackendUnsupported, Plan, ScoreResult


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
    encode: Callable[[str], list[int]]
    forward: Callable[[Any, int], Any]
    files: dict[str, str] = field(default_factory=dict)  # role -> sha256
    max_context: int | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    notes: dict[str, Any] = field(default_factory=dict)
    kernel_stack: dict[str, Any] = field(default_factory=dict)  # part of the revision hash


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

    def __init__(self, plugin: ModelPlugin, template: str | Template, max_batch_rows: int = 16) -> None:
        self.plugin = plugin
        self.template = TEMPLATES[template] if isinstance(template, str) else template
        self.max_batch_rows = max_batch_rows
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
            "rows": "one per question; grouped by length; unpadded",
            "max_batch_rows": self.max_batch_rows,
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
    def _rows(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for qid in request["questions"]:
            prompt_ids = self.plugin.encode(build_prompt(self.template, request, qid))
            options = [o for o, _ in canonical_options(request["questions"][qid])]
            for o in options:
                cont = self.plugin.encode(continuation(self.template, o))
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

    def plan(self, request: dict[str, Any]) -> Plan:
        rows = self._rows(request)
        chunks = self._chunks(rows)
        lengths = [len(r["prompt"]) + len(r["cont"]) for r in rows]
        per_q: dict[str, dict[str, Any]] = {}
        for r in rows:
            d = per_q.setdefault(r["qid"], {"prompt_tokens": len(r["prompt"]), "options": 0})
            d["options"] += 1
        return Plan(
            inference_calls=len(chunks),
            max_call_input_tokens=max(lengths),
            total_input_tokens=sum(lengths),
            per_question=per_q,
        )

    # ---------------------------------------------------------------- scoring
    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        import torch

        rows = self._rows(request)
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
        return ScoreResult(
            logits=logits,
            inference_calls=len(chunks),
            total_input_tokens=total_tokens,
            extras={"call_latency_ms": call_ms, "rows": len(rows)},
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

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False).input_ids

    def forward(ids, n_last: int):
        return model(input_ids=ids, logits_to_keep=n_last, use_cache=False).logits

    return ModelPlugin(
        family="qwen3.5",
        model_id=model_id or cfg.get("_name_or_path") or model_dir.name,
        snapshot=_snapshot_of(model_dir),
        encode=encode,
        forward=forward,
        files={
            "weights": _hash_files(model_dir, ["*.safetensors"]),
            "tokenizer": _hash_files(model_dir, ["tokenizer.json"]),
            "config": _hash_files(model_dir, ["config.json"]),
        },
        max_context=text_cfg.get("max_position_embeddings"),
        dtype=dtype,
        device=device,
        kernel_stack=_pkg_versions("flash-linear-attention", "fla-core", "causal-conv1d", "triton", "triton-windows"),
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
        encode=tok.encode,
        forward=forward,
        files={
            "weights": _hash_files(model_dir, ["*.safetensors"]),
            "tokenizer": _hash_files(model_dir, ["rwkv_vocab_v20230424.txt"]),
            "config": _hash_files(model_dir, ["config.json"]),
        },
        max_context=None,  # recurrent; trained ctx recorded in notes
        dtype=dtype,
        device=device,
        kernel_stack=_pkg_versions("flash-linear-attention", "fla-core", "triton", "triton-windows"),
        notes={"model_class": type(model).__name__, "trained_ctx": cfg.get("max_position_embeddings"), "attention_free": True},
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

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False).input_ids

    def forward(ids, n_last: int):
        return model(ids, num_last_tokens=n_last).logits

    return ModelPlugin(
        family="mamba3",
        model_id=model_id or model_dir.name,
        snapshot=_snapshot_of(model_dir),
        encode=encode,
        forward=forward,
        files={
            "weights": _hash_files(model_dir, ["*.bin", "*.safetensors"]),
            "tokenizer": _hash_files(Path(tokenizer_dir), ["tokenizer.json"]),
            "config": _hash_files(model_dir, ["config.json"]),
        },
        max_context=None,
        dtype=dtype,
        device=device,
        kernel_stack=_pkg_versions("mamba_ssm", "triton", "triton-windows"),
        notes={"attention_free": True, "tokenizer": "Llama-3.1 tokenizer (ungated mirror)"},
    )
