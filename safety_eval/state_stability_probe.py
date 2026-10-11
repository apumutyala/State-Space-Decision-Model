"""State-stability probe (docs/STATE_STABILITY.md section 6): per-layer decay and transition-norm statistics of a
trained checkpoint over Decision Index DEV prompts. GPU job script; DRAFT, NOT SUBMITTED; nothing here trains.

    python -X utf8 -m safety_eval.state_stability_probe --family rwkv7 --model-dir SNAP \
        [--trainable trainable-stepN.safetensors] --texts prompts.jsonl --out OUT [--max-items 256]
    python -X utf8 -m safety_eval.state_stability_probe --family gdn --model-dir QWEN35_4B ... (hybrid main line)

What it measures, per recurrent layer (no per-step state is stored: memory is O(T*H) per layer, not O(T*H*N*N)):

* ``decay``: RWKV-7 ``max_i w_{t,i}`` / GDN ``alpha_t`` per (t, head): max and quantiles (slowest forgetting).
* ``step_norm``: a per-step upper bound ``n_t >= ||A_t||_2`` per (t, head).
  GDN: EXACT closed form ``alpha (max(1, |1 - beta ||k||^2|))`` on the kernel's own (l2-normalised, rounded) k.
  RWKV-7: the certificate ``U_t = max(w_max, abar q - w_min) + sqrt(q) ||(a - abar) * kk||``, ``q = ||kk||^2``,
  ``abar = sum a kk^2 / q`` (docs T2(b)); plus EXACT ``||A_t||_2`` (SVD) on every ``--exact-stride``-th position,
  which also checks ``U_t >= ||A_t||_2`` on real data (``cert_violations`` must be 0).
* ``forgetting``: per window length L, ``max over (start, head) of sum log n_t`` = log of the worst CERTIFIED
  per-input factor by which a state perturbation present at the window start can survive L clean steps (docs T4),
  and the EXACT ``log ||A_{t+L} ... A_{t+1}||_2`` on ``--n-windows`` random windows (tightness of the product bound;
  > 0 means real growth on this input).
* ``iss``: the final state's ``||S_T||_F`` per head against the per-input ISS bound
  ``prod n_t ||S_0|| + sum_s prod_{u>s} n_u ||B_s||`` (docs T3); ``iss_violations`` must be 0 up to rounding.
* ``universal`` (weights only): GDN ``-log rho_h`` in log space (``rho_h`` itself rounds to 1.0 in float64 once
  ``1 - rho_h < 1e-16``) and the e-fold horizon ``1/(-log rho_h)`` tokens; RWKV-7 the diagonal decay ceiling.

Status: the per-input certificates are CERTIFIED per input (theorem applied to computed numbers, float rounding
aside); every sup over the item set is EMPIRICAL about unseen inputs. SAFETY_SPEC 5.2: not a safety claim.

Inputs: ``--texts`` JSONL, one object per prompt with ``text`` (or pre-tokenised ``ids``) and optional ``id``.
Use DEV prompts only (rendered with the cell's readout template); never TEST or held-out rows.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

WINDOWS = (16, 64, 256)


# ------------------------------------------------------------------------------------------------ statistics
def _q(x: torch.Tensor, qs=(0.5, 0.99)) -> dict:
    x = x.flatten().double()
    if x.numel() > 1_000_000:  # torch.quantile input cap; a deterministic stride subsample is enough
        x = x[:: x.numel() // 1_000_000 + 1]
    return {f"q{int(q * 100)}": float(x.quantile(q)) for q in qs}


def window_logprod_max(logn: torch.Tensor, L: int) -> float | None:
    """max over (start, head) of sum_{s in window} log n_s.  logn: [T, H] float64."""
    T = logn.shape[0]
    if T < L:
        return None
    c = torch.cat([torch.zeros_like(logn[:1]), logn.cumsum(0)], 0)
    return float((c[L:] - c[:-L]).max())


def iss_bound(logn: torch.Tensor, b: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
    """Per head: prod_t n_t ||S_0|| + sum_s prod_{u>s} n_u b_s, in log space. logn, b: [T, H]; s0: [H]."""
    cum = logn.cumsum(0)
    total = cum[-1]
    terms = (total[None] - cum) + torch.log(b.clamp_min(1e-300))  # [T, H]
    lead = total + torch.log(s0.clamp_min(1e-300))
    return torch.exp(torch.logsumexp(torch.cat([terms, lead[None]], 0), 0))


def product_norm(mats_fn, T: int, L: int, n: int, gen: torch.Generator) -> list[float]:
    """EXACT log ||A_{t+L} ... A_{t+1}||_2 (max over heads) on n random windows; mats_fn(t0, t1) -> [L, H, N, N]."""
    out = []
    if T < L + 1:
        return out
    for _ in range(n):
        t0 = int(torch.randint(0, T - L + 1, (1,), generator=gen))
        A = mats_fn(t0, t0 + L).double()  # step norms <= ~1.16 and >= ~0, so float64 cannot overflow at L <= 256
        P = A[0]
        for i in range(1, L):
            P = A[i] @ P
        out.append(float(torch.log(torch.linalg.matrix_norm(P, ord=2).clamp_min(1e-300)).max()))
    return out


class Acc:
    """Per-layer accumulator of per-item summaries (aggregated at the end)."""

    def __init__(self):
        self.items: list[dict] = []

    def add(self, d: dict):
        self.items.append(d)

    def summary(self) -> dict:
        if not self.items:
            return {}
        out = {"n_items": len(self.items)}
        keys = sorted({k for d in self.items for k, v in d.items() if isinstance(v, (int, float)) and v is not None})
        for k in keys:
            vals = torch.tensor([d[k] for d in self.items if isinstance(d.get(k), (int, float))], dtype=torch.float64)
            if vals.numel():
                agg = "sum" if k.endswith(("_violations", "_steps", "_gt1")) else None
                out[k] = float(vals.sum()) if agg else {"max": float(vals.max()), "median": float(vals.median()),
                                                         "min": float(vals.min())}
        exact = [x for d in self.items for x in d.get("exact_window_logprod", {}).get("all", [])]
        if exact:
            out["exact_window_logprod_all"] = {"max": max(exact), "n": len(exact)}
        return out


# ------------------------------------------------------------------------------------------------ RWKV-7 hook
def rwkv7_stats(w_log, kk, a, k, v, state_in, state_out, cfg) -> dict:
    """All inputs [1, T, H, N] (state [1, H, V, K]). Returns the per-item summary for one layer."""
    w = torch.exp(w_log[0].float())  # [T, H, N]
    kk_, a_ = kk[0].float(), a[0].float()
    q = (kk_ * kk_).sum(-1)  # [T, H]
    abar = (a_ * kk_ * kk_).sum(-1) / q.clamp_min(1e-30)
    spread = torch.sqrt(((a_ - abar[..., None]) ** 2 * kk_ * kk_).sum(-1)) * torch.sqrt(q)
    wmax, wmin = w.amax(-1), w.amin(-1)
    U = torch.maximum(wmax, abar * q - wmin) + spread  # [T, H] certified >= ||A_t||_2
    T, H = U.shape
    d = {"T": T, "decay_max": float(wmax.max()), **{f"decay_{k_}": v_ for k_, v_ in _q(wmax).items()},
         "one_minus_decay_min": float((1 - wmax).min()),
         "step_bound_max": float(U.max()), **{f"step_bound_{k_}": v_ for k_, v_ in _q(U).items()},
         "step_bound_gt1": int((U > 1).sum()), "n_steps": T * H,
         "a_spread_max": float(spread.max()), **{f"a_spread_{k_}": v_ for k_, v_ in _q(spread).items()}}

    def mats_at(sel):
        return torch.diag_embed(w[sel]) - (kk_[sel] * a_[sel]).unsqueeze(-1) * kk_[sel].unsqueeze(-2)

    def mats(t0, t1):
        return mats_at(slice(t0, t1))

    idx = torch.arange(0, T, cfg.exact_stride, device=U.device)
    ex = torch.cat([torch.linalg.matrix_norm(mats_at(idx[j: j + 64]), ord=2)  # [|idx|, H], 64 positions per SVD batch
                    for j in range(0, idx.numel(), 64)])
    d.update({"exact_norm_max": float(ex.max()), "exact_norm_gt1": int((ex > 1).sum()),
              "exact_norm_steps": int(ex.numel()),
              "cert_violations": int((ex > U[idx] * (1 + 1e-5) + 1e-6).sum()),
              "cert_slack_median": float((U[idx] - ex).median())})
    logn = torch.log(U.double().clamp_min(1e-300))
    for L in WINDOWS:
        d[f"cert_window_logprod_L{L}"] = window_logprod_max(logn, L)
    g = torch.Generator().manual_seed(cfg.seed)
    allx = []
    for L in WINDOWS:
        px = product_norm(mats, T, L, cfg.n_windows, g)
        if px:
            d[f"exact_window_logprod_L{L}"] = max(px)
            allx += [x / L for x in px]
    d["exact_window_logprod"] = {"all": allx}
    b = (torch.linalg.vector_norm(k[0].float(), dim=-1) * torch.linalg.vector_norm(v[0].float(), dim=-1)).double()
    s0 = torch.linalg.matrix_norm(state_in[0].double())  # [H] Frobenius
    sT = torch.linalg.matrix_norm(state_out[0].double())
    bound = iss_bound(logn, b, s0)
    d.update({"state_fro_max": float(sT.max()), "iss_bound_max": float(bound.max()),
              "iss_ratio_max": float((sT / bound.clamp_min(1e-300)).max()),
              "iss_violations": int((sT > bound * (1 + 1e-3) + 1e-6).sum())})
    return d


def install_rwkv7(model, cfg, accs):
    from heads.rwkv7_wkv import WkvName
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    from controlsafe import adapters

    mcfg = adapters.config(model)
    inner = M.RWKV7_WKV_FUNCTIONS[str(mcfg.wkv_implementation)]
    current = {"layer": None}

    def kernel(r, w_log, k, v, kk, a, state, cu_seq_lens=None, **kw):
        out, new_state = inner(r, w_log, k, v, kk, a, state, cu_seq_lens=cu_seq_lens, **kw)
        with torch.no_grad():
            accs.setdefault(current["layer"], Acc()).add(rwkv7_stats(w_log, kk, a, k, v, state, new_state, cfg))
        return out, new_state

    M.RWKV7_WKV_FUNCTIONS["stability_probe"] = kernel
    old = mcfg.wkv_implementation
    mcfg.wkv_implementation = WkvName("stability_probe")  # batched Linears, like "chunked" (heads/rwkv7_wkv.py)
    for i, blk in enumerate(adapters.blocks(model)):
        blk.att.register_forward_pre_hook(lambda mod, args, _i=i: current.__setitem__("layer", _i))
    return {"wrapped_wkv": str(old)}


# ------------------------------------------------------------------------------------------------ GDN hook
def gdn_stats(key, value, g, beta, l2, state_in, state_out, cfg) -> dict:
    """key/value [1, T, H, K/V] as handed to the kernel (after GQA repeat), g/beta [1, T, H]."""
    k = key[0].float()
    if l2:  # the kernel's normalisation (fla l2norm: x * rsqrt(sum x^2 + 1e-6), fp32, stored in the input dtype)
        k = (k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)).to(key.dtype).float()
    alpha = torch.exp(g[0].float())  # [T, H]
    bt = beta[0].float()
    q = (k * k).sum(-1)
    n = alpha * torch.clamp((1 - bt * q).abs(), min=1.0)  # EXACT ||A_t||_2
    T, H = n.shape
    d = {"T": T, "decay_max": float(alpha.max()), **{f"decay_{k_}": v_ for k_, v_ in _q(alpha).items()},
         "one_minus_decay_min": float((1 - alpha).min()),
         "step_bound_max": float(n.max()), **{f"step_bound_{k_}": v_ for k_, v_ in _q(n).items()},
         "step_bound_gt1": int((n > 1).sum()), "n_steps": T * H,
         "beta_q_max": float((bt * q).max()), "k_sq_norm_max": float(q.max()),
         "alpha_eq_1_steps": int((alpha >= 1.0).sum())}
    logn = torch.log(n.double().clamp_min(1e-300))
    for L in WINDOWS:
        d[f"cert_window_logprod_L{L}"] = window_logprod_max(logn, L)
    eye = torch.eye(k.shape[-1], device=k.device)

    def mats(t0, t1):
        kk = k[t0:t1]
        return alpha[t0:t1, :, None, None] * (eye - bt[t0:t1, :, None, None] * kk.unsqueeze(-1) * kk.unsqueeze(-2))

    gen = torch.Generator().manual_seed(cfg.seed)
    allx = []
    for L in WINDOWS:
        px = product_norm(mats, T, L, cfg.n_windows, gen)
        if px:
            d[f"exact_window_logprod_L{L}"] = max(px)
            allx += [x / L for x in px]
    d["exact_window_logprod"] = {"all": allx}
    b = (bt * torch.sqrt(q) * torch.linalg.vector_norm(value[0].float(), dim=-1)).double()
    s0 = (torch.linalg.matrix_norm(state_in[0].double()) if state_in is not None
          else torch.zeros(H, dtype=torch.float64, device=n.device))
    sT = torch.linalg.matrix_norm(state_out[0].double())
    bound = iss_bound(logn, b, s0)
    d.update({"state_fro_max": float(sT.max()), "iss_bound_max": float(bound.max()),
              "iss_ratio_max": float((sT / bound.clamp_min(1e-300)).max()),
              "iss_violations": int((sT > bound * (1 + 1e-3) + 1e-6).sum())})
    return d


def install_gdn(model, cfg, accs):
    from controlsafe import adapters

    wrapped = []
    for i, blk in enumerate(adapters.blocks(model)):
        if not hasattr(blk, "linear_attn"):
            continue
        la = blk.linear_attn
        for attr in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
            inner = getattr(la, attr)

            def fn(query, key, value, g=None, beta=None, initial_state=None, output_final_state=False,
                   use_qk_l2norm_in_kernel=False, _inner=inner, _i=i, **kw):
                o, st = _inner(query, key, value, g=g, beta=beta, initial_state=initial_state, output_final_state=True,
                               use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
                with torch.no_grad():
                    accs.setdefault(_i, Acc()).add(gdn_stats(key, value, g, beta, use_qk_l2norm_in_kernel,
                                                             initial_state, st, cfg))
                return o, (st if output_final_state else None)

            setattr(la, attr, fn)
        wrapped.append(i)
    return {"gdn_layers": wrapped}


# ------------------------------------------------------------------------------------------------ universal
def universal(model, family: str) -> dict:
    from controlsafe import adapters

    out = {}
    for i, blk in enumerate(adapters.blocks(model)):
        if family == "gdn":
            if not hasattr(blk, "linear_attn"):
                out[i] = {"family": "attention", "note": "no recurrent state: no decay bound (docs F4)"}
                continue
            la = blk.linear_attn
            gamma = blk.input_layernorm.weight.detach().double()
            d = gamma.numel()
            wa = adapters._eff_weight(la.in_proj_a).to(gamma.device)
            lo = -(wa * (1.0 + gamma)).norm(dim=-1) * (d ** 0.5) + la.dt_bias.detach().double()
            nlr = la.A_log.detach().double().exp() * torch.nn.functional.softplus(lo)  # -log rho_h, exact in log space
            out[i] = {"family": "gdn", "neg_log_rho_min": float(nlr.min()), "efold_horizon_tokens_max": float(1 / nlr.min()),
                      "rho_max_float64": float(torch.exp(-nlr).max()), "certified_lt1_exact": bool((nlr > 0).all()),
                      "rho_rounds_to_1_in_float64": bool((torch.exp(-nlr) >= 1.0).any())}
        else:
            att = blk.att
            lo = att.w0.detach().double().reshape(-1) - att.w2.detach().double().abs().sum(0)
            wbar = torch.exp(-0.6065306597126334 * torch.sigmoid(lo))
            out[i] = {"family": "rwkv7", "decay_ceiling": float(wbar.max()),
                      "scalar_a_universal_rho": float(wbar.max()),  # = w_bar (docs T2(c); exact arithmetic)
                      "note": "full transition: no universal bound (docs T2(d,e)); scalar_a_universal_rho holds only "
                              "if a were shared across a head's channels (docs T2(c))"}
    return out


# ------------------------------------------------------------------------------------------------ loading
def install_trainable(model: nn.Module, path: str, alpha: float | None) -> dict:
    """Wrap every Linear named in a train.ckpt_export safetensors (``*.lora_A`` / ``*.lora_B``; non-LoRA tensors =
    the head, ignored: the head reads the state, it does not change it). Names are matched by longest suffix."""
    from safetensors.torch import load_file

    from train.lora import LoRALinear

    sd = load_file(path)
    lin = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    done = []
    for key in sorted(k[: -len(".lora_A")] for k in sd if k.endswith(".lora_A")):
        cand = [n for n in lin if key == n or key.endswith("." + n)]
        if not cand:
            continue  # e.g. a head-internal LoRA; reported below
        n = max(cand, key=len)
        A, B = sd[key + ".lora_A"], sd[key + ".lora_B"]
        parent_name, _, attr = n.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrap = LoRALinear(lin[n], A.shape[0], alpha, 0.0)
        with torch.no_grad():
            wrap.lora_A.copy_(A.float())
            wrap.lora_B.copy_(B.float())
        setattr(parent, attr, wrap)
        done.append(n)
    n_lora = sum(1 for k in sd if k.endswith(".lora_A"))
    if n_lora and not done:
        raise SystemExit(f"--trainable: none of {n_lora} LoRA modules matched this backbone's Linear names")
    return {"path": path, "lora_modules_in_file": n_lora, "installed": len(done)}


def load(family: str, model_dir: str, device: str):
    if family == "rwkv7":
        from third_party.rwkv7_hf import verify_snapshot_code
        from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM
        from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

        verify_snapshot_code(model_dir)
        lm = Rwkv7ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": device})
        return lm.rwkv7, Rwkv7Tokenizer.from_pretrained(model_dir)
    from transformers import AutoConfig, AutoTokenizer

    if hasattr(AutoConfig.from_pretrained(model_dir), "text_config"):  # same split as heads/backbone.py:838-843
        from transformers import Qwen3_5ForConditionalGeneration

        full = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": device})
        text = full.model.language_model
    else:
        from transformers import Qwen3_5ForCausalLM

        full = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": device})
        text = full.model
    del full
    return text, AutoTokenizer.from_pretrained(model_dir)


def read_items(path: str, tok, max_items: int, max_tokens: int):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n >= max_items:
                return
            row = json.loads(line)
            ids = row.get("ids") or tok(row["text"], add_special_tokens=False).input_ids
            ids = [int(i) for i in ids][-max_tokens:]  # keep the END: the decision position is last
            yield str(row.get("id", n)), ids


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="state_stability_probe")
    ap.add_argument("--family", choices=("rwkv7", "gdn"), required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--trainable", default="", help="train.ckpt_export trainable-*.safetensors (LoRA installed)")
    ap.add_argument("--lora-alpha", type=float, default=None, help="default: alpha = rank (scale 1, train/lora.py)")
    ap.add_argument("--texts", required=True, help="JSONL(.gz) of {id, text | ids}: DEV prompts only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--exact-stride", type=int, default=16)
    ap.add_argument("--n-windows", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    cfg = ap.parse_args(argv)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = load(cfg.family, cfg.model_dir, cfg.device)
    meta = {"commit": git_sha(), "argv": sys.argv[1:] if argv is None else argv, "torch": torch.__version__,
            "python": platform.python_version(), "family": cfg.family, "model_dir": cfg.model_dir}
    if cfg.trainable:
        meta["trainable"] = install_trainable(model, cfg.trainable, cfg.lora_alpha)
    model.eval()
    uni = universal(model, cfg.family)
    (out / "universal.json").write_text(json.dumps({"meta": meta, "universal": uni}, indent=1), encoding="utf-8")
    accs: dict[int, Acc] = {}
    meta["hook"] = install_rwkv7(model, cfg, accs) if cfg.family == "rwkv7" else install_gdn(model, cfg, accs)
    n_items, n_tok = 0, 0
    with torch.no_grad(), open(out / "items.jsonl", "w", encoding="utf-8") as fi:
        for item_id, ids in read_items(cfg.texts, tok, cfg.max_items, cfg.max_tokens):
            before = {l: len(a.items) for l, a in accs.items()}
            model(input_ids=torch.tensor([ids], device=cfg.device), use_cache=False)
            row = {"id": item_id, "T": len(ids), "layers": {}}
            for l, a in accs.items():
                if len(a.items) > before.get(l, 0):
                    d = dict(a.items[-1]); d.pop("exact_window_logprod", None)
                    row["layers"][l] = d
            fi.write(json.dumps(row) + "\n")
            n_items += 1; n_tok += len(ids)
    layers = {l: accs[l].summary() for l in sorted(accs)}
    flags = {"cert_violations": sum(v.get("cert_violations", 0) for v in layers.values()),
             "iss_violations": sum(v.get("iss_violations", 0) for v in layers.values()),
             "layers_with_step_bound_gt1": [l for l, v in layers.items() if v.get("step_bound_gt1", 0) > 0],
             "layers_with_exact_growth": [l for l, v in layers.items()
                                          if (v.get("exact_window_logprod_all") or {}).get("max", -1) > 0]}
    summary = {"meta": meta, "n_items": n_items, "n_tokens": n_tok, "minutes": round((time.time() - t0) / 60, 2),
               "flags": flags, "layers": layers,
               "status": "per-input certificates CERTIFIED (float rounding aside); sups over items EMPIRICAL"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"n_items": n_items, "flags": flags}), flush=True)
    return 0 if flags["cert_violations"] == 0 and flags["iss_violations"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
