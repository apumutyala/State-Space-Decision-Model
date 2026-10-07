"""Frozen-Q9 feature extraction for the proxy head loop (jobs/q9_feature_cache/JOB_SPEC.md; train/proxy_rule.py).

One request at a time, no grad, no LoRA, the m2 ``ordered`` layout (byte-identical to the m0 readout prompt) and the
same template the stock readout uses (``jrt-chatml-v1``):

  1. the shared prefix (template + schema of all fields + state) runs ONCE: ``L = min(common token prefix of the
     request's rows, end of the state span)`` (``decide.backends.state_fork.common_prefix_len`` caps it so every row
     keeps >= 1 own token);
  2. per question, the prefix state is copied ``c`` times (options in chunks of ``opt_chunk``) and each copy reads
     ``suffix (question re-read) + continuation_k`` (right-padded; only positions <= the row's last are read); when
     copying the suffix K times costs more than ``TWO_STAGE_MIN_SAVED`` positions (many options, long re-read), the
     suffix runs once and only the continuations are forked from its end state (same values, fewer positions);
  3. stored per question (bf16 unless noted):
       fork [K, D]   final feature after prefix + continuation_k  (== heads.features ``fork``, cache-fork mode)
       e    [K, D]   mean feature over option k's span in the schema (== ``HeadInput.e``, ordered layout)
       opt_tok / opt_off   option k's span token ids (int32, concatenated; offsets [K+1]). The lexical feature (mean
                     OUTPUT-embedding row over those tokens, Clef's lexical-prior input) is recomputed on CPU from the
                     arm's ``lex_vocab.pt`` (lm_head rows of every token any stored option uses), not stored per option
                     (review B1: the cache must fit one <= 4.5 GiB job artifact)
       q_pre [D], q_end [D]   target field mean / answer-position feature (== ``HeadInput.q_pre / q_end``)
       lm_lp [K] fp32, lm_ntok [K]   summed log-likelihood of continuation_k (``" " + option_id + <|im_end|>``) under
                     the frozen LM: the stock readout's own option score (E4 input; the stock readout's
                     probabilities are softmax(lm_lp) with the identity option-count temperature)
       key_class     ``heads.option_keys.key_class`` of the option ids (E5 / key-class input)
     and per request, for "rich" requests, ``banks``: up to ``bank_chunks`` equal-count chunk means [S, D] of each
     distinct evidence span (the state; or a question's re-read instruction when the state is empty).
Feature tensors are the backbone's final (post-norm) hidden states, ``last_hidden_state`` (== ``hidden_states[-1]``,
what ``heads.backbone.HiddenStateBackbone.features`` returns; tested).

Arms: ``stock`` (Qwen/Qwen3.5-9B @ c2022362), ``conv1`` (layer 23's attention replaced by the converted GDN from
art-2bybh ``gdn_converted.pt``, sha256 pinned; HF-native ``Qwen3_5GatedDeltaNet`` as ``heads.backbone.q9af_from_stock``
builds it, so the m2 path, not the convert/ path), ``e5`` (stock weights, opaque option keys rendered as short codes,
``heads.option_keys.e5_recode``; option ids are stored as the ORIGINAL ids).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from decide.backends.state_fork import common_prefix_len, expand
from heads.layout import LayoutRow, build_row
from heads.option_keys import e5_recode, key_class

TEMPLATE = "jrt-chatml-v1"
LAYOUT = "ordered"
FEATURE_VERSION = "q9-cache-v1"


# ----------------------------------------------------------------------------------------------- model loading
def convert_layers(text_model, gdn_state: dict, layers: Sequence[int]) -> None:
    """Replace the attention of ``layers`` with native HF GDNs loaded strictly from ``gdn_state`` (a PARTIAL
    ``heads.backbone.q9af_from_stock``: the other attention layers stay)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    cfg = text_model.config
    attn = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    for i in layers:
        if i not in attn:
            raise ValueError(f"layer {i} is not an attention layer ({attn})")
        sd = gdn_state[i] if i in gdn_state else gdn_state[str(i)]
        types = list(cfg.layer_types)
        types[i] = "linear_attention"
        cfg.layer_types = types
        layer = text_model.layers[i]
        ref = layer.mlp.down_proj.weight
        gdn = Qwen3_5GatedDeltaNet(cfg, i).to(device=ref.device, dtype=ref.dtype)
        gdn.load_state_dict(sd, strict=True)
        gdn.requires_grad_(False)
        layer.linear_attn = gdn
        del layer.self_attn
        layer.block_type = "linear_attention"
    if hasattr(text_model, "layer_types"):
        text_model.layer_types = list(cfg.layer_types)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def load(model_dir: str, device: str = "cuda", dtype: str = "bfloat16", gdn_ckpt: str | None = None,
         gdn_sha256: str | None = None, layers: Sequence[int] = ()) -> tuple[torch.nn.Module, torch.Tensor, dict]:
    """-> (text model, lm_head weight [V, D], provenance). The 9B is ``Qwen3_5ForConditionalGeneration`` (vision tower
    dropped); a text-only config (the dry-run stand-in) loads as ``Qwen3_5ForCausalLM``."""
    import transformers as tf

    prov: dict[str, Any] = {"model_dir": str(model_dir), "dtype": dtype, "converted_layers": list(layers)}
    if layers:
        if not gdn_ckpt or not gdn_sha256:
            raise ValueError("a converted arm needs gdn_ckpt and its pinned sha256")
        got = sha256_file(gdn_ckpt)
        if got != gdn_sha256:
            raise ValueError(f"gdn checkpoint sha256 {got} != pinned {gdn_sha256}")
        prov["gdn_sha256"] = got
    cfg = tf.AutoConfig.from_pretrained(model_dir)
    dt = getattr(torch, dtype)
    if hasattr(cfg, "text_config"):
        full = tf.Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=dt, device_map={"": device})
        text, w = full.model.language_model, full.lm_head.weight.detach()
        if hasattr(full.model, "visual"):
            del full.model.visual
    else:
        full = tf.Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=dt).to(device)
        text, w = full.model, full.lm_head.weight.detach()
    text.eval().requires_grad_(False)
    if layers:
        state = torch.load(gdn_ckpt, map_location="cpu", weights_only=True)
        convert_layers(text, state, layers)
    prov["layer_types"] = list(text.config.layer_types)
    return text, w, prov


# ----------------------------------------------------------------------------------------------- one request
def _cache(text):
    from heads.backbone import _linear_safe_cache

    c = _linear_safe_cache(text.config)
    c.seen = 0
    return c


def _run(text, ids: torch.Tensor, cache):
    out = text(input_ids=ids, past_key_values=cache, use_cache=True)
    c = out.past_key_values
    c.seen = getattr(cache, "seen", 0) + ids.shape[1]
    return out.last_hidden_state, c


def _clone(cache):
    from heads.prefix_features import _clone as clone

    return clone(cache)


def _span_mean(h: torch.Tensor, span) -> torch.Tensor:
    a, b = span
    return h[a:b].float().mean(0)


def _bank(h: torch.Tensor, span, n_chunks: int) -> torch.Tensor:
    a, b = span
    n = b - a
    s = min(n_chunks, n)
    edges = [a + (i * n) // s for i in range(s + 1)]
    return torch.stack([h[edges[i]:edges[i + 1]].float().mean(0) for i in range(s)])


#: run the question suffix ONCE (then fork only the continuations) when copying it K times would cost more positions
#: than this; one-stage is one call per option chunk, two-stage adds one call (the suffix)
TWO_STAGE_MIN_SAVED = 512


def two_stage(ns: int, k: int) -> bool:
    return (k - 1) * ns > TWO_STAGE_MIN_SAVED


#: cap on (state copies x cached tokens) per call: the 8 attention layers keep a KV cache of 32 KiB per token (job-5cmat
#: throughput.stock.state.kv_bytes_per_token), so 2**19 copied tokens ~ 16 GiB; a 28k-token DEV row is forked in
#: chunks of <= 18 options instead of 32 (the GDN states are fixed-size and small)
KV_COPY_TOKENS = 1 << 19


def chunk_size(cached_len: int, opt_chunk: int) -> int:
    return max(1, min(opt_chunk, KV_COPY_TOKENS // max(cached_len, 1)))


@torch.no_grad()
def extract_request(text, w_out: torch.Tensor, rows: Sequence[LayoutRow], opt_chunk: int = 32, bank_chunks: int = 0,
                    pad_id: int = 0, fork_dtype: torch.dtype = torch.bfloat16) -> tuple[list[dict], dict]:
    """Features of every row of ONE request (rows share the schema + state prefix). -> (per-question dicts, banks)."""
    dev = w_out.device
    seqs = [r.ids for r in rows]
    L = min(common_prefix_len(seqs) if len(seqs) > 1 else len(seqs[0]) - 1, min(r.state_span[1] for r in rows))
    L = max(L, 1)
    hp, sp = _run(text, torch.tensor([seqs[0][:L]], dtype=torch.long, device=dev), _cache(text))
    hp = hp[0]
    out, banks = [], {}
    for r in rows:
        if r.ids[:L] != seqs[0][:L]:
            raise ValueError("rows of one request must share the prefix")
        suf = r.ids[L:]
        ns = len(suf)
        conts = r.continuations
        k_all = len(conts)
        fork = torch.empty((k_all, hp.shape[-1]), dtype=fork_dtype, device=dev)  # bf16 stored; fp32 for the self-check
        lm_lp = torch.empty(k_all, dtype=torch.float32, device=dev)
        hs = None
        two = two_stage(ns, k_all)
        base, off = sp, ns  # one stage: every copy reads suffix + continuation from the prefix state
        if two:  # two stages: the suffix once, then only the continuations from the suffix state
            hs1, base = _run(text, torch.tensor([suf], dtype=torch.long, device=dev), _clone(sp))
            hs, off = hs1[0], 0
        step = chunk_size(L + (ns if two else 0), opt_chunk)
        for c0 in range(0, k_all, step):
            part = conts[c0:c0 + step]
            c = len(part)
            wmax = max(len(x) for x in part)
            ids = torch.full((c, off + wmax), pad_id, dtype=torch.long, device=dev)
            for j, x in enumerate(part):
                if off:
                    ids[j, :off] = torch.tensor(suf, dtype=torch.long, device=dev)
                ids[j, off:off + len(x)] = torch.tensor(x, dtype=torch.long, device=dev)
            st = _clone(base)
            if c > 1:
                expand(st, c)
            h, _ = _run(text, ids, st)  # [c, off + wmax, D]
            if hs is None:
                hs = h[0, :ns]
            last = torch.tensor([off + len(x) - 1 for x in part], device=dev)
            fork[c0:c0 + c] = h[torch.arange(c, device=dev), last].to(fork_dtype)
            # LM log-likelihood of each continuation: token t is predicted by the feature before it (position
            # off+t-1; for t = 0 in two-stage mode that is the suffix's last feature)
            kk = torch.tensor([j for j, x in enumerate(part) for _ in x], device=dev)
            tt = [t for x in part for t in range(len(x))]
            pp = torch.tensor([max(off + t - 1, 0) for t in tt], device=dev)
            hsel = h[kk, pp]
            if two:
                first = torch.tensor([t == 0 for t in tt], device=dev).unsqueeze(-1)
                hsel = torch.where(first, hs[-1].expand_as(hsel), hsel)
            tg = torch.tensor([tok for x in part for tok in x], device=dev)
            logits = (hsel @ w_out.T).float()
            tok_lp = torch.log_softmax(logits, -1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            lm_lp[c0:c0 + c] = torch.zeros(c, device=dev).index_add_(0, kk, tok_lp)
            del h, st
        hrow = torch.cat([hp, hs], 0)  # [T, D]
        e = torch.stack([_span_mean(hrow, sp_) for sp_ in r.option_spans])
        toks = [r.ids[a:b] for a, b in r.option_spans]
        off = [0]
        for t in toks:
            off.append(off[-1] + len(t))
        q = {"qid": r.qid, "qtype": r.qtype, "option_ids": list(r.option_ids), "T": len(r.ids), "K": r.n_options,
             "evidence_len": r.evidence_len, "evidence": r.meta.get("evidence"), "key_class": key_class(r.option_ids),
             "fork": fork.cpu(), "e": e.to(torch.bfloat16).cpu(),
             "opt_tok": torch.tensor([x for t in toks for x in t], dtype=torch.int32),
             "opt_off": torch.tensor(off, dtype=torch.int32),
             "q_pre": _span_mean(hrow, r.field_span).to(torch.bfloat16).cpu(),
             "q_end": hrow[r.answer_pos].to(torch.bfloat16).cpu(), "lm_lp": lm_lp.cpu(),
             "lm_ntok": torch.tensor([len(x) for x in conts], dtype=torch.int16)}
        if bank_chunks:
            a, b = r.state_span
            key = "state" if b <= L else f"q:{r.qid}"
            if key not in banks:
                banks[key] = _bank(hrow, (a, b), bank_chunks).to(torch.bfloat16).cpu()
            q["bank_key"] = key
        out.append(q)
    return out, banks


@torch.no_grad()
def plain_forward(text, w_out: torch.Tensor, row: LayoutRow, k: int) -> tuple[torch.Tensor, float, torch.Tensor]:
    """A plain full-sequence forward over ``row.ids + continuation_k`` (no cache, no fork): (fork feature fp32 on CPU,
    summed LM log-likelihood of the continuation, the row's own hidden states h[:T] fp32 on CPU)."""
    dev = w_out.device
    seq = row.ids + row.continuations[k]
    h = text(input_ids=torch.tensor([seq], device=dev), use_cache=False).last_hidden_state[0]
    lp = torch.log_softmax((h[len(row.ids) - 1: len(seq) - 1] @ w_out.T).float(), -1)
    score = float(lp.gather(-1, torch.tensor(row.continuations[k], device=dev).unsqueeze(-1)).sum())
    return h[-1].float().cpu(), score, h[: len(row.ids)].float().cpu()


class fp32_weights:
    """Context: every parameter and buffer of ``text`` in fp32, restored EXACTLY to its own dtype on exit (bf16 -> fp32
    -> bf16 is lossless; parameters a bf16 load keeps in fp32 stay fp32), so production numerics are untouched."""

    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self.dtypes = {n: t.dtype for n, t in list(self.text.named_parameters()) + list(self.text.named_buffers())}
        try:
            for _, t in list(self.text.named_parameters()) + list(self.text.named_buffers()):
                if t.is_floating_point():
                    t.data = t.data.float()
        except BaseException:
            self.__exit__()
            raise
        return self.text

    def __exit__(self, *exc):
        for n, t in list(self.text.named_parameters()) + list(self.text.named_buffers()):
            t.data = t.data.to(self.dtypes[n])
        return False


class KernelSwapError(RuntimeError):
    """The pure-torch reference kernels could not be put in place: the fp32 'truth' would not be IEEE fp32."""


def _impl_of(f) -> str:
    """Module of the implementation a transformers hub/fla kernel wrapper actually calls (its ``implementation``
    closure variable; the function itself when it is not a wrapper)."""
    import inspect

    try:
        impl = inspect.getclosurevars(f).nonlocals.get("implementation", f)
    except TypeError:
        impl = f
    return f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__name__', '?')}"


class reference_kernels:
    """Context: the Qwen3.5 GDN layers run their PURE-TORCH reference functions (the ``__wrapped__`` originals of
    transformers' hub/fla kernel wrappers: chunk / recurrent gated delta rule, causal conv1d) and TF32 is off for
    matmuls and cuDNN, so an fp32 forward is an IEEE fp32 truth on GPU as on CPU. fla's fp32 chunk kernel uses TF32
    (``SOLVE_TRIL_DOT_PRECISION = 'tf32'``, fla 0.5.2), which would make the "truth" itself ~1e-3 noisy.

    FAIL-CLOSED (review B1): every one of ``NAMES`` must exist, be a wrapper whose ``__wrapped__`` is defined in the
    Qwen3.5 modeling module itself, and be swapped; otherwise ``KernelSwapError``. ``record`` names the implementation
    each name resolved to before the swap (e.g. fla on the GPU image) and during it."""

    NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
             "causal_conv1d_update")

    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as M

        self.M, self.saved, self.record = M, {}, {}
        for n in self.NAMES:
            f = getattr(M, n, None)
            ref = getattr(f, "__wrapped__", None)
            if f is None or ref is None or getattr(ref, "__module__", None) != M.__name__:
                for m, g in self.saved.items():
                    setattr(M, m, g)
                raise KernelSwapError(f"{n}: no pure-torch reference to swap in (got {f!r}, __wrapped__={ref!r})")
            self.saved[n] = f
            self.record[n] = {"production": _impl_of(f), "truth": f"{ref.__module__}.{ref.__name__}"}
            setattr(M, n, ref)
        if set(self.saved) != set(self.NAMES):
            raise KernelSwapError(f"swapped {sorted(self.saved)} != {list(self.NAMES)}")
        self.tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return self

    def __exit__(self, *exc):
        for n, f in self.saved.items():
            setattr(self.M, n, f)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.tf32
        return False


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6))


def _free():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_keys(row: LayoutRow) -> list[int]:
    """Options checked per row: {0, K//2, K-1} (review S1)."""
    return sorted({0, row.n_options // 2, row.n_options - 1})


@torch.no_grad()
def selfcheck(text, w_out: torch.Tensor, requests, opt_chunk: int, sc: dict) -> tuple[bool, list[dict], dict]:
    """Numerics gate of the cached-fork extractor against an fp32 TRUTH (LESSONS L-06 / L-28). -> (ok, records, info).

    Every row of each request (its first two rows, which share the prefix) and options {0, K//2, K-1} of each row:
      production   the cached-fork features in the run's dtype (``extract_request``) and a plain full-sequence forward
                   in the same dtype (``plain_forward``): two equally valid computations whose difference is rounding
      truth        a plain forward with every weight in fp32 and the pure-torch reference kernels, TF32 off
                   (``fp32_weights`` + ``reference_kernels``, fail-closed; weights restored exactly afterwards), plus
                   the cached-fork path under the same conditions with an fp32 fork feature (the LOGIC check)
      fla fp32     the cached fork and the plain forward with fp32 weights but the PRODUCTION kernels (fla on GPU,
                   TF32 inside): the only pass that runs fla's continuation kernel at fp32 precision (review S2)
    Pass iff, for every checked row / option:
      logic   rel(fork_cache_fp32, truth) <= logic_fork_max;  |lm_cache_fp32 - truth| <= logic_lm_per_token x n;
              per row: rel(q_end / q_pre / e of the fp32 extraction, the same quantities from the plain fp32 hidden
              states) <= feature_rel_max (they are stored in bf16: ~2e-3 floor)
      cache   rel(fork_cache, truth) <= ratio x rel(fork_plain, truth) + fork_abs;
              |lm_cache - truth| <= ratio x |lm_plain - truth| + lm_abs_per_token x n   (the LM half is in practice
              floor-driven: on the exact path the LM ratio reached 2.3 on Qwen3.5-0.8B and passes on the floor)
      fla     rel(fork_fla32, truth) <= fla_ratio x rel(fork_plain_fla32, truth) + fla_fork_abs. GATING only when
              ``sc["fla_gate"]`` is true; on the first launch it is RECORD-ONLY (its noise on the 9B is unmeasured).
    Every record carries its ratios and the bounds applied (review S3)."""
    prod = []
    for rows in requests:
        rows = list(rows)[:2]
        feats, _ = extract_request(text, w_out, rows, opt_chunk)
        plains = {(i, k): plain_forward(text, w_out, r, k)[:2] for i, r in enumerate(rows) for k in check_keys(r)}
        prod.append((rows, feats, plains))
    _free()
    out, info = [], {"bounds": {k: sc[k] for k in ("ratio", "fork_abs", "lm_abs_per_token", "logic_fork_max",
                                                     "logic_lm_per_token", "feature_rel_max", "fla_ratio", "fla_fork_abs",
                                                     "fla_gate")}}
    with fp32_weights(text):
        w32 = w_out.float()
        fla = []
        for rows, _, _ in prod:  # fla fp32 pass: production kernels, fp32 weights (record-only unless fla_gate)
            f_fla, _ = extract_request(text, w32, rows, opt_chunk, fork_dtype=torch.float32)
            fla.append((f_fla, {(i, k): plain_forward(text, w32, r, k)[:2] for i, r in enumerate(rows)
                                for k in check_keys(r)}))
        _free()
        with reference_kernels() as rk:
            info["kernels"] = rk.record
            for (rows, f16s, plains), (f_fla, p_fla) in zip(prod, fla):
                f32s, _ = extract_request(text, w32, rows, opt_chunk, fork_dtype=torch.float32)
                for i, r in enumerate(rows):
                    hrow = None
                    for k in check_keys(r):
                        tf, tl, h = plain_forward(text, w32, r, k)
                        hrow = h if hrow is None else hrow
                        pf, pl = plains[(i, k)]
                        ff, fl = p_fla[(i, k)]
                        n = len(r.continuations[k])
                        f16, f32, fla32 = f16s[i], f32s[i], f_fla[i]
                        rec = {"row": i, "k": k, "K": r.n_options, "n_cont": n, "T": len(r.ids),
                               "fork_rel_cache_vs_plain": _rel(f16["fork"][k], pf),  # the v1 metric, for comparison
                               "fork_rel_cache_vs_truth": _rel(f16["fork"][k], tf), "fork_rel_plain_vs_truth": _rel(pf, tf),
                               "fork_rel_cache32_vs_truth": _rel(f32["fork"][k], tf),
                               "fork_rel_fla32_vs_truth": _rel(fla32["fork"][k], tf),
                               "fork_rel_plainfla32_vs_truth": _rel(ff, tf),
                               "lm_cache_vs_truth": abs(float(f16["lm_lp"][k]) - tl), "lm_plain_vs_truth": abs(pl - tl),
                               "lm_cache32_vs_truth": abs(float(f32["lm_lp"][k]) - tl),
                               "lm_fla32_vs_truth": abs(float(fla32["lm_lp"][k]) - tl),
                               "lm_plainfla32_vs_truth": abs(fl - tl)}
                        rec["fork_ratio"] = rec["fork_rel_cache_vs_truth"] / max(rec["fork_rel_plain_vs_truth"], 1e-12)
                        rec["lm_ratio"] = rec["lm_cache_vs_truth"] / max(rec["lm_plain_vs_truth"], 1e-12)
                        rec["fla_fork_ratio"] = rec["fork_rel_fla32_vs_truth"] / max(rec["fork_rel_plainfla32_vs_truth"],
                                                                                     1e-12)
                        rec["bound_fork_cache"] = sc["ratio"] * rec["fork_rel_plain_vs_truth"] + sc["fork_abs"]
                        rec["bound_lm_cache"] = sc["ratio"] * rec["lm_plain_vs_truth"] + sc["lm_abs_per_token"] * n
                        rec["bound_fork_fla"] = sc["fla_ratio"] * rec["fork_rel_plainfla32_vs_truth"] + sc["fla_fork_abs"]
                        rec["logic_ok"] = (rec["fork_rel_cache32_vs_truth"] <= sc["logic_fork_max"]
                                           and rec["lm_cache32_vs_truth"] <= sc["logic_lm_per_token"] * n)
                        rec["cache_ok"] = (rec["fork_rel_cache_vs_truth"] <= rec["bound_fork_cache"]
                                           and rec["lm_cache_vs_truth"] <= rec["bound_lm_cache"])
                        rec["fla_ok"] = rec["fork_rel_fla32_vs_truth"] <= rec["bound_fork_fla"]
                        out.append(rec)
                    # row features: q_end / q_pre / e of the fp32 extraction vs the plain fp32 hidden states
                    f32 = f32s[i]
                    e_ref = torch.stack([hrow[a:b].mean(0) for a, b in r.option_spans])
                    feat = {"row": i, "k": None, "q_end_rel": _rel(f32["q_end"], hrow[len(r.ids) - 1]),
                            "q_pre_rel": _rel(f32["q_pre"], hrow[r.field_span[0]:r.field_span[1]].mean(0)),
                            "e_rel": _rel(f32["e"], e_ref), "bound_feature": sc["feature_rel_max"]}
                    feat["logic_ok"] = max(feat["q_end_rel"], feat["q_pre_rel"], feat["e_rel"]) <= sc["feature_rel_max"]
                    feat["cache_ok"] = feat["fla_ok"] = True
                    out.append(feat)
    del w32
    _free()
    gate_fla = bool(sc.get("fla_gate"))
    info["fla_gating"] = gate_fla
    info["fla_ok_all"] = all(r["fla_ok"] for r in out)
    ok = bool(out) and all(r["logic_ok"] and r["cache_ok"] and (r["fla_ok"] or not gate_fla) for r in out)
    return ok, out, info


@torch.no_grad()
def recompute_check(text, w_out: torch.Tensor, row: LayoutRow, feats: dict, k: int = 0) -> dict:
    """Numerics gate: the cached fork feature and LM score of option ``k`` vs a plain full-sequence forward over
    ``row.ids + continuation_k`` (no cache, no fork)."""
    dev = w_out.device
    seq = row.ids + row.continuations[k]
    h = text(input_ids=torch.tensor([seq], device=dev), use_cache=False).last_hidden_state[0]
    n = len(row.continuations[k])
    lp = torch.log_softmax((h[len(row.ids) - 1: len(seq) - 1] @ w_out.T).float(), -1)
    score = float(lp.gather(-1, torch.tensor(row.continuations[k], device=dev).unsqueeze(-1)).sum())
    f_ref = h[-1].float().cpu()
    f = feats["fork"][k].float()
    return {"fork_rel_diff": float((f - f_ref).norm() / f_ref.norm().clamp_min(1e-6)),
            "lm_abs_diff": abs(score - float(feats["lm_lp"][k])), "n_cont": n}


# ----------------------------------------------------------------------------------------------- sources
def dev_request(row: dict, arm: str) -> tuple[dict, dict]:
    req = {"model": "q9-cache", "state": row["state"], "questions": row["questions"]}
    back: dict = {}
    if arm == "e5":
        req, back = e5_recode(req)
    return req, back


def build_rows(req: dict, encode, qids=None) -> tuple[list[LayoutRow], dict]:
    rows, bad = [], {}
    for qid in sorted(req["questions"]) if qids is None else qids:
        try:
            rows.append(build_row(req, qid, encode, template=TEMPLATE, layout=LAYOUT))
        except ValueError as exc:  # ControlTokenError subclasses ValueError: the question is unanswerable
            bad[qid] = f"{type(exc).__name__}: {exc}"[:300]
    return rows, bad


# ----------------------------------------------------------------------------------------------- shards
class ShardWriter:
    """Requests -> ``<out>/shard_<n>.pt`` (torch.save of a list of request dicts, weights_only-loadable) every
    ``per_shard`` requests; each shard's sha256 + counts appended to ``<out>/MANIFEST.jsonl``; one line per finished
    request in ``progress`` (the live rate guard counts these lines)."""

    def __init__(self, out: Path, per_shard: int, progress: Path) -> None:
        self.out, self.per, self.progress = Path(out), per_shard, Path(progress)
        self.out.mkdir(parents=True, exist_ok=True)
        self.buf: list[dict] = []
        self.n_shards = 0
        self.n_requests = 0
        self.n_questions = 0

    def add(self, rec: dict) -> None:
        self.buf.append(rec)
        self.n_requests += 1
        self.n_questions += len(rec["questions"])
        with open(self.progress, "a", encoding="utf-8") as f:
            f.write(json.dumps({"rid": rec["rid"], "q": len(rec["questions"]), "t": round(time.time(), 2)}) + "\n")
        if len(self.buf) >= self.per:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        name = f"shard_{self.n_shards:05d}.pt"
        p = self.out / name
        torch.save(self.buf, p)
        line = {"file": name, "sha256": sha256_file(p), "bytes": p.stat().st_size, "requests": len(self.buf),
                "questions": sum(len(r["questions"]) for r in self.buf)}
        with open(self.out / "MANIFEST.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        self.n_shards += 1
        self.buf = []


def load_shards(d: str | Path, verify: bool = True) -> list[dict]:
    """All request records of one split directory, in order; every shard re-hashed against MANIFEST.jsonl."""
    d = Path(d)
    out = []
    for line in (d / "MANIFEST.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        p = d / m["file"]
        if verify and sha256_file(p) != m["sha256"]:
            raise ValueError(f"{p}: sha256 mismatch with MANIFEST.jsonl")
        out += torch.load(p, map_location="cpu", weights_only=True)
    return out


def n_chunks_calls(rows: Sequence[LayoutRow], opt_chunk: int) -> tuple[int, int]:
    """(positions, calls) this extractor runs for one request (the volume model of the preflight)."""
    seqs = [r.ids for r in rows]
    L = min(common_prefix_len(seqs) if len(seqs) > 1 else len(seqs[0]) - 1, min(r.state_span[1] for r in rows))
    L = max(L, 1)
    pos, calls = L, 1
    for r in rows:
        ns = len(r.ids) - L
        off = ns
        if two_stage(ns, r.n_options):
            pos, calls, off = pos + ns, calls + 1, 0
        step = chunk_size(L + (ns if off == 0 else 0), opt_chunk)
        for c0 in range(0, r.n_options, step):
            part = r.continuations[c0:c0 + step]
            pos += len(part) * (off + max(len(x) for x in part))
            calls += 1
    return pos, calls


def lex_vocab(w_out: torch.Tensor, token_ids) -> dict:
    """lm_head rows of the given token ids (bf16, CPU): the lexical table the proxy loader averages per option."""
    ids = torch.tensor(sorted(set(int(t) for t in token_ids)), dtype=torch.long)
    return {"ids": ids, "rows": w_out[ids.to(w_out.device)].to(torch.bfloat16).cpu()}


def lexical(q: dict, table: dict) -> torch.Tensor:
    """[K, D] bf16: mean lm_head row over each option's span tokens (== the extractor's former ``lex`` field)."""
    pos = {int(t): i for i, t in enumerate(table["ids"].tolist())}
    tok, off = q["opt_tok"].tolist(), q["opt_off"].tolist()
    rows = table["rows"]
    return torch.stack([rows[[pos[t] for t in tok[off[k]:off[k + 1]]]].float().mean(0)
                        for k in range(len(off) - 1)]).to(torch.bfloat16)


def option_tokens(rows: Sequence[LayoutRow]) -> set:
    return {t for r in rows for a, b in r.option_spans for t in r.ids[a:b]}


def lm_tokens(rows: Sequence[LayoutRow]) -> int:
    return sum(len(x) for r in rows for x in r.continuations)
