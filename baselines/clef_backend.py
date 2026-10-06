"""Cloudflare Clef / Clef-flash as a ``decide()`` backend (baseline for SAFETY_SPEC section 7).

The model, the joint schema head and the record layout are Cloudflare's own code, imported from the
vendored, sha256-pinned ``third_party/cloudflare_clef/joint_schema_model.py`` (Apache-2.0, unmodified).
This module only (a) rebuilds Clef's token sequence so that untrusted text can be routed through
``decide/untrusted.py``, (b) maps the head's per-option logits to the ``decide`` envelope, and
(c) offers Clef's own ``systemone`` path unchanged, for parity and for the silent-truncation probe.

Two encodings (``encoding=``):

* ``"ssdm"`` (default, the like-for-like arm): exactly Clef's layout and constant template text,
  but every request-derived string (state, question ids, instructions, option ids/descriptions)
  goes through ``decide.untrusted.UntrustedEncoder``. On text without special/added-token strings
  the ids are identical to Clef's own ``encode_record`` (tested), so logits are identical too.
* ``"clef_native"``: request text tokenized exactly as Clef does (no ``split_special_tokens``).
  This is Cloudflare's serving behaviour and is the "before" arm of the G2(b) measurement.

Mapping to the envelope (the adapter does the rest, same code path as every other backend):

* ``ScoreResult.logits[qid][option_id]`` = the joint head's logit for that option. The adapter's
  ``raw_probabilities`` = softmax(logits) = Clef's own per-question softmax. ``served`` equals
  ``raw`` under the default identity calibration (we never calibrate a competitor).
* ``status``: ``ok``; ``exhausted / input_tokens_exceeded`` when the full sequence exceeds
  ``max_length`` (Clef would silently truncate the state instead); ``unsupported /
  control_tokens_in_input`` when the untrusted encoder fails closed; ``unsupported /
  media_unsupported`` for images/videos (adapter validation).
* One joint forward per request: ``inference_calls = 1``; ``usage.input_tokens`` = the length of
  the single joint sequence (the same number Clef reports when it does not truncate).

Where Clef's own path differs (each is measured, not assumed):

1. Silent truncation of the state at ``max_length`` (default 16,384) -> we return ``exhausted``.
   ``native_systemone`` reports how many state tokens Clef dropped (``truncated_state_tokens``).
2. Control tokens from untrusted text (see ``encoding``).
3. Tie-break: Clef picks the first option in request order; the adapter picks the smallest id.
   Probabilities are unaffected; only exact ties can change ``choice``.
4. Question order: Clef numbers fields in request order, so reordering questions changes the
   input. We keep request order (to measure Clef as shipped) and record the order in ``extras``.
5. Validation: the adapter is stricter (1-level ``score``, bad ids) and answers ``error`` with
   ``answers == {}`` where Clef either accepts or raises ``ValueError``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from decide import versions
from decide.backends.base import BackendUnsupported, Plan, ScoreResult
from decide.schema import RequestRejected, canonical_json, sha256_text
from decide.untrusted import UNTRUSTED_ENCODER_ID, ControlTokenError, UntrustedEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
JSM_PATH = REPO_ROOT / "third_party" / "cloudflare_clef" / "joint_schema_model.py"
JSM_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
CLEF_MAX_LENGTH = 16384  # encode_record / systemone default

#: Pinned releases (HF API, 2026-10-06). LFS sha256 per file; bf16 parameter counts from the Hub.
RELEASES: dict[str, dict[str, Any]] = {
    "clef-flash": {
        "repo": "Cloudflare/clef-flash",
        "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
        "base_model": "Qwen/Qwen3.5-9B",
        "params_bf16": 9_409_813_744,
        "repo_bytes": 19_083_377_402,
        "layers": {"linear_attention": 24, "full_attention": 8},
        "lfs_sha256": {
            "joint_head.safetensors": "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "model-00001-of-00004.safetensors": "8b45a8e968141cdcc58fb71c9adfc258e2c77b5f062bc636c1fd5bc5d916b565",
            "model-00002-of-00004.safetensors": "7590856c713eed844a2dcf48e6c43c4de165b788bc3f80e328311183cdbc7db8",
            "model-00003-of-00004.safetensors": "e6eac2467952c33361ed7dcb3c7959d1086bbe57201cd3749c3d769fdc17fe63",
            "model-00004-of-00004.safetensors": "9fcecc6556b39171238373a465f409794b7f821fb4cd1e6459e3a9c0fe317af7",
        },
    },
    "clef": {
        "repo": "Cloudflare/clef",
        "revision": "2f3de3dd85f379784083b0814d997ab627200f0c",
        "base_model": "Qwen/Qwen3.8-27B",
        "params_bf16": 27_356_728_560,
        "repo_bytes": 54_989_894_057,
        "layers": {"linear_attention": 48, "full_attention": 16},
        "lfs_sha256": {
            "joint_head.safetensors": "a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "model-00001-of-00012.safetensors": "54d83c1d36631de231876217a8e0c2483eccee8746369a482b79442bdfc5d958",
            "model-00002-of-00012.safetensors": "464086af08be8e2ec14960a4dcff083ebc39974ade00d79d35497385f960ab3a",
            "model-00003-of-00012.safetensors": "092212d3a02fafacd6424723eda59d60e5282d2068d68f0e37cb891f63bbb658",
            "model-00004-of-00012.safetensors": "d06ff197668c782145fafa74bba61bbc296fb27e39afb15fd522918ce3514dc5",
            "model-00005-of-00012.safetensors": "cc693b8829614a72e0c2be303fb290cf05dbb4d7ded872a817b97bb222a78427",
            "model-00006-of-00012.safetensors": "e7cce15da2443cb8b84aaed66a9a71f0c87dc9d043f83b5f58f4a89f64ba60ad",
            "model-00007-of-00012.safetensors": "75fc7e76b57d5d17a5d85fff3e879d07dd33edc885a8ee04ad437a899bcd5307",
            "model-00008-of-00012.safetensors": "189b15cb6b1af48d5f118951446e15639bfeaf76081d5f20aed1f1b4253afe1d",
            "model-00009-of-00012.safetensors": "8101e2664bb14684fc7051f2e1f84903dbdf5489b17cae3212ac08a0af744a60",
            "model-00010-of-00012.safetensors": "a8e69016a1a8dab1c9ce8dd151c0d3224412ab864cf06d3a5e2b8a5775cfdfb8",
            "model-00011-of-00012.safetensors": "b328d21c36ab384696e40a30ac86a95aaf6dd82da89438ba009cb97d87c1d6b9",
            "model-00012-of-00012.safetensors": "7505eed910a84ea18e66953572476f8a8a6a6ef54192a6643d6e9cd21a3bb978",
        },
    },
}

ENCODINGS = ("ssdm", "clef_native")


class JSMIntegrityError(RuntimeError):
    pass


def kernel_stack() -> dict[str, Any]:
    """Installed versions + importability of the fast GDN path (part of the revision identity).

    If ``fla``/``causal_conv1d`` do not import, transformers runs Qwen3.5's torch fallback for the
    linear-attention layers: outputs agree up to kernel numerics, latency does not.
    """
    from importlib import metadata

    out: dict[str, Any] = {}
    for n in ("torch", "transformers", "flash-linear-attention", "fla-core", "causal-conv1d", "triton"):
        try:
            out[n] = metadata.version(n)
        except Exception:  # noqa: BLE001
            out[n] = None
    for mod in ("fla", "causal_conv1d"):
        try:
            importlib.import_module(mod)
            out[f"import:{mod}"] = True
        except Exception:  # noqa: BLE001
            out[f"import:{mod}"] = False
    # fla #640: which backward kernel fla dispatches depends on FLA_TILELANG and on tilelang being installed
    import os

    out["env:FLA_TILELANG"] = os.environ.get("FLA_TILELANG")
    out["tilelang_installed"] = importlib.util.find_spec("tilelang") is not None
    return out


class reference_gdn_path:
    """Context manager: run every Qwen3.5 Gated-DeltaNet layer on transformers' own torch reference
    (``torch_chunk_gated_delta_rule`` + the conv1d fallback) instead of the fla / causal-conv1d kernels.

    Used by ``clef_probes`` to show, inside the job image, that the fast kernel path scores the same as
    the reference path (the check that matters after moving off the model card's torch 2.11; see
    JOB_SPEC "Image"). ``FusedRMSNormGated`` (an fla Triton norm) is left in place: its parameters are
    not interchangeable with the torch norm without re-wiring, and it has no #640 exposure (forward only
    in this probe).

    ``__enter__`` returns the number of GDN layers whose kernel was ACTUALLY replaced, i.e. whose
    ``chunk_gated_delta_rule`` was not already the torch reference. transformers sets
    ``self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule`` and falls back
    silently when fla is unavailable (no CUDA, import failure, fla < 0.2.2); such layers are counted in
    ``n_found`` and ``n_already_reference`` but not as switched, so the probe cannot pass vacuously (PR #43 B1).
    """

    def __init__(self, model: Any) -> None:
        self.model, self.saved = model, []
        self.n_found = self.n_already_reference = self.n_switched = 0

    def __enter__(self) -> int:
        for m in self.model.modules():
            if hasattr(m, "chunk_gated_delta_rule") and hasattr(m, "causal_conv1d_fn"):
                ref = getattr(sys.modules[type(m).__module__], "torch_chunk_gated_delta_rule", None)
                if ref is None:
                    raise RuntimeError(f"{type(m).__name__}: no torch_chunk_gated_delta_rule reference in its module")
                self.n_found += 1
                if m.chunk_gated_delta_rule is ref or m.chunk_gated_delta_rule is None:
                    self.n_already_reference += 1
                else:
                    self.n_switched += 1
                self.saved.append((m, m.chunk_gated_delta_rule, m.causal_conv1d_fn))
                m.chunk_gated_delta_rule, m.causal_conv1d_fn = ref, None
        return self.n_switched

    def __exit__(self, *exc) -> None:
        for m, chunk, conv in self.saved:
            m.chunk_gated_delta_rule, m.causal_conv1d_fn = chunk, conv
        self.saved = []


def load_jsm(path: Path = JSM_PATH, expected_sha256: str = JSM_SHA256):
    """Import the vendored Clef module after checking its bytes against the pin."""
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != expected_sha256:
        raise JSMIntegrityError(f"{path}: sha256 {got} != pinned {expected_sha256}; refusing to import")
    name = "ssdm_vendored_clef_jsm"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------------- encoding
@dataclass
class ClefParts:
    """Clef's record split into its four segments; spans are relative to ``schema_ids``."""

    prefix_ids: list[int]
    state_ids: list[int]
    schema_ids: list[int]
    suffix_ids: list[int]
    questions: list[Any]  # jsm.EncodedQuestion with unshifted spans
    record_id: str

    def length(self, state_len: int | None = None) -> int:
        s = len(self.state_ids) if state_len is None else state_len
        return len(self.prefix_ids) + s + len(self.schema_ids) + len(self.suffix_ids)


class ClefEncoder:
    """Clef's ``encode_record`` layout (text-only), with untrusted text optionally hardened.

    Constant template strings (Clef's system frame, ``SCHEMA FIELDS``, ``OPTION i:`` ...) are
    tokenized exactly as Clef does. Request-derived strings use ``UntrustedEncoder`` in ``ssdm``
    mode. Where Clef tokenizes one f-string that mixes template text with a question id, the whole
    string is treated as untrusted (the template part contains no control strings, so its ids are
    unchanged), which keeps BPE merges identical to Clef's.
    """

    def __init__(self, tokenizer: Any, jsm: Any, encoding: str = "ssdm") -> None:
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        self.tok, self.jsm, self.encoding = tokenizer, jsm, encoding
        self.untrusted = UntrustedEncoder(tokenizer) if encoding == "ssdm" else None

    def identity(self) -> dict[str, Any]:
        if self.untrusted is not None:
            return {"encoding": "ssdm", **self.untrusted.identity()}
        return {"encoding": "clef_native", "id": "clef-encode_record-plain (no split_special_tokens)"}

    def trusted(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False).input_ids)

    def untrusted_ids(self, text: str) -> list[int]:
        if self.untrusted is None:
            return self.trusted(text)
        return self.untrusted.encode(text)  # raises ControlTokenError (fail closed)

    def state_ids(self, state: Any) -> list[int]:
        return self.untrusted_ids(self.jsm.render(state))

    def parts(self, record: dict[str, Any]) -> ClefParts:
        jsm, T, U = self.jsm, self.trusted, self.untrusted_ids
        schema_ids = T("\n\nSCHEMA FIELDS:\n")
        questions = []
        for qi, (qid, q) in enumerate(record["questions"].items()):
            schema_ids += U(f"\nFIELD {qi + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
            q_start = len(schema_ids)
            instructions = q.get("instructions")
            if instructions is None or instructions == "":
                instructions = str(qid)
            schema_ids += U(jsm.render(instructions))
            q_end = len(schema_ids)
            schema_ids += T("\nALLOWED OPTIONS:\n")
            spans, oids = [], []
            for oi, (oid, desc) in enumerate(jsm.question_options(q)):
                schema_ids += T(f"OPTION {oi + 1}: ")
                o_start = len(schema_ids)
                sem = {"option_id": oid}
                if desc is not None:
                    sem["description"] = desc
                schema_ids += U(jsm.render(sem))
                spans.append((o_start, len(schema_ids)))
                oids.append(oid)
                schema_ids += T("\n")
            schema_ids += T("END FIELD\n")
            questions.append(jsm.EncodedQuestion(
                question_id=str(qid), question_type=jsm.QUESTION_TYPES[str(q["type"])],
                question_span=(q_start, q_end), option_spans=tuple(spans), option_ids=tuple(oids)))
        prefix = T(f"<|im_start|>system\n{jsm.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
        suffix = T("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
        return ClefParts(prefix, self.state_ids(record["state"]), schema_ids, suffix, questions,
                         str(record.get("id", "unknown")))

    def assemble(self, parts: ClefParts, state_ids: list[int] | None = None):
        """``EncodedRecord`` with the given state ids (never truncated)."""
        jsm = self.jsm
        state = parts.state_ids if state_ids is None else list(state_ids)
        off = len(parts.prefix_ids) + len(state)
        qs = tuple(
            jsm.EncodedQuestion(
                question_id=q.question_id, question_type=q.question_type,
                question_span=(q.question_span[0] + off, q.question_span[1] + off),
                option_spans=tuple((a + off, b + off) for a, b in q.option_spans),
                option_ids=q.option_ids)
            for q in parts.questions)
        ids = tuple(parts.prefix_ids + state + parts.schema_ids + parts.suffix_ids)
        return jsm.EncodedRecord(input_ids=ids, questions=qs, record_id=parts.record_id, media=None)


# --------------------------------------------------------------------------------- backend
class ClefBackend:
    """``decide`` backend over a loaded ``ClefModel`` (vendored code) and its tokenizer."""

    name = "clef_joint_head"

    def __init__(self, model: Any, tokenizer: Any, *, release: str = "clef-flash", encoding: str = "ssdm",
                 jsm: Any = None, device: str = "cuda", dtype: str = "bfloat16",
                 max_length: int = CLEF_MAX_LENGTH, files_sha256: dict[str, str] | None = None,
                 release_info: dict[str, Any] | None = None) -> None:
        self.jsm = jsm or load_jsm()
        self.model, self.tok = model, tokenizer
        self.release = release
        self.info = release_info if release_info is not None else RELEASES.get(release, {})
        self.encoder = ClefEncoder(tokenizer, self.jsm, encoding)
        self.encoding = encoding
        self.device, self.dtype, self.max_length = device, dtype, max_length
        self.files_sha256 = files_sha256 or {}
        self.kernel_stack = kernel_stack()
        self._revision = self._build_revision()

    def _build_revision(self) -> dict[str, Any]:
        cfg = {
            "scoring": "Clef joint schema head logits (vendored joint_schema_model.py, unmodified)",
            "encoding": self.encoder.identity(),
            "max_length": self.max_length,
            "truncation": "never (over-length -> exhausted)",
            "question_order": "request order (as Clef)",
            "dtype": self.dtype,
            "device": self.device,
            "kernel_stack": self.kernel_stack,
        }
        cfg_hash = sha256_text(canonical_json(cfg))
        rev = (self.info.get("revision") or "local")[:12]
        enc_id = UNTRUSTED_ENCODER_ID if self.encoding == "ssdm" else "clef-native"
        rid = (f"clef:{self.info.get('repo', self.release)}@{rev};jsm={JSM_SHA256[:12]};enc={enc_id}"
               f";cfg={cfg_hash[:8]};adapter={versions.ADAPTER_VERSION}")
        return {
            "revision_id": rid, "backend": self.name, "release": self.release,
            "model_id": self.info.get("repo"), "snapshot": self.info.get("revision"),
            "base_model": self.info.get("base_model"), "files_sha256": self.files_sha256,
            "vendored_jsm_sha256": JSM_SHA256, "inference_config": cfg, "inference_config_sha256": cfg_hash,
            "adapter_version": versions.ADAPTER_VERSION, "max_context": self.max_length,
            "notes": {"attention_free": False, "layers": self.info.get("layers")},
        }

    def revision(self) -> dict[str, Any]:
        return self._revision

    def _parts(self, request: dict[str, Any]) -> ClefParts:
        if request.get("images") or request.get("videos"):
            raise BackendUnsupported("media_unsupported", "text-only Clef baseline")
        try:
            return self.encoder.parts(request)
        except ControlTokenError as exc:
            raise BackendUnsupported(exc.code, str(exc)) from exc

    def plan(self, request: dict[str, Any]) -> Plan:
        p = self._parts(request)
        n = p.length()
        return Plan(inference_calls=1, max_call_input_tokens=n, total_input_tokens=n,
                    per_question={q.question_id: {"options": len(q.option_ids)} for q in p.questions})

    def logits_for_records(self, records: list[Any]) -> list[list[Any]]:
        """Raw head output (per record, per question tensors) for already-built records."""
        import torch

        pad = getattr(self.tok, "pad_token_id", None)
        pad = 0 if pad is None else pad
        batch = self.jsm.collate_records(records, pad, torch.device(self.device))
        with torch.inference_mode():
            return self.model(batch)

    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        import torch

        parts = self._parts(request)
        if parts.length() > self.max_length:  # adapter's plan check normally fires first; same status here
            raise RequestRejected("exhausted", "input_tokens_exceeded",
                                  f"longest model input {parts.length()} tokens > {self.max_length} (no truncation)")
        record = self.encoder.assemble(parts)
        check_deadline()
        t0 = time.perf_counter()
        out = self.logits_for_records([record])[0]
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        ms = round((time.perf_counter() - t0) * 1000.0, 3)
        logits: dict[str, dict[str, float]] = {}
        for q, ql in zip(record.questions, out):
            vals = ql.float().tolist()
            logits[q.question_id] = {oid: float(v) for oid, v in zip(q.option_ids, vals)}
        return ScoreResult(
            logits=logits, inference_calls=1, total_input_tokens=len(record.input_ids),
            extras={"call_latency_ms": [ms], "encoding": self.encoding,
                    "question_order": [q.question_id for q in record.questions],
                    "sequence_tokens": len(record.input_ids)})


# --------------------------------------------------------------------------------- Clef's own path
def native_systemone(jsm: Any, model: Any, processor: Any, request: dict[str, Any],
                     max_length: int = CLEF_MAX_LENGTH) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cloudflare's ``systemone`` unchanged, plus what it does not tell the caller.

    Returns ``(clef_response, meta)``; ``meta.truncated_state_tokens`` is how many state tokens
    Clef silently dropped (0 if none) and ``meta.control_ids_from_request`` counts special/added
    token ids that request text produced inside Clef's sequence.
    """
    tok = processor.tokenizer
    enc = ClefEncoder(tok, jsm, "clef_native")
    parts = enc.parts(request)
    full = parts.length()
    resp = jsm.systemone(model, processor, request, max_length=max_length)
    from decide.untrusted import control_ids_and_contents

    ctrl, _ = control_ids_and_contents(tok)
    template = set(enc.trusted(f"<|im_start|>system\n{jsm.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")) | set(
        parts.suffix_ids)
    from_request = [i for i in parts.state_ids + parts.schema_ids if i in ctrl]
    meta = {
        "path": "clef_native_systemone",
        "full_length_tokens": full,
        "served_length_tokens": resp["usage"]["input_tokens"],
        "truncated_state_tokens": max(0, full - resp["usage"]["input_tokens"]),
        "control_ids_from_request": len(from_request),
        "template_control_ids": sorted(i for i in template if i in ctrl),
    }
    return resp, meta


# --------------------------------------------------------------------------------- loading
def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify_snapshot(model_dir: str | Path, release: str, hash_weights: bool = True) -> dict[str, Any]:
    """Compare a local snapshot with the pinned LFS sha256s and the vendored jsm."""
    d = Path(model_dir)
    pins = RELEASES[release]["lfs_sha256"]
    out: dict[str, Any] = {"release": release, "files": {}, "ok": True}
    for name, want in pins.items():
        if not hash_weights and name.startswith("model-"):
            continue
        p = d / name
        got = sha256_file(p.resolve()) if p.exists() else "missing"
        out["files"][name] = {"sha256": got, "ok": got == want}
        out["ok"] &= got == want
    jsm_remote = d / "joint_schema_model.py"
    got = sha256_file(jsm_remote) if jsm_remote.exists() else "missing"
    out["files"]["joint_schema_model.py"] = {"sha256": got, "ok": got == JSM_SHA256}
    out["ok"] &= got == JSM_SHA256
    return out


def load_clef(model_dir: str | Path, release: str = "clef-flash", *, encoding: str = "ssdm",
              device: str = "cuda", dtype: str = "bfloat16", max_length: int = CLEF_MAX_LENGTH,
              verify: bool = True) -> tuple[ClefBackend, Any, Any]:
    """Load a local, pinned snapshot with Cloudflare's own loader. Returns (backend, model, processor).

    ``model_dir`` must be a directory (we never let the vendored loader call ``snapshot_download``).
    """
    import torch

    d = Path(model_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is not a local snapshot directory")
    check = verify_snapshot(d, release) if verify else {"ok": None, "note": "not verified"}
    if verify and not check["ok"]:
        raise JSMIntegrityError(f"snapshot does not match pins: {json.dumps(check)}")
    jsm = load_jsm()
    model, processor = jsm.load_release_model(d, device=device, dtype=getattr(torch, dtype))
    files = {k: v["sha256"] for k, v in check.get("files", {}).items()}
    backend = ClefBackend(model, processor.tokenizer, release=release, encoding=encoding, jsm=jsm,
                          device=device, dtype=dtype, max_length=max_length, files_sha256=files)
    return backend, model, processor


def build_decider(backend: ClefBackend, **kwargs: Any):
    """A ``decide.Decider`` over the Clef backend: identity calibration, v0.1 deltas on (safety_eval
    needs ``policy_restriction``), input cap = Clef's ``max_length`` so over-length is ``exhausted``."""
    from decide import Decider

    kwargs.setdefault("enable_v0_1_deltas", True)
    kwargs.setdefault("backend_max_input_tokens", backend.max_length)
    return Decider(backend, **kwargs)
