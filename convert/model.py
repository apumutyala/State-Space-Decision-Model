"""Load Qwen3.5 (text path only) and swap attention mixers for GDN mixers in place.

Each full-attention layer's ``self_attn`` is replaced by a :class:`MixerSwap` that holds both the original
attention (frozen teacher) and an optional new GDN (student). Its ``mode`` selects what runs:

* ``teacher`` - original softmax attention (stock model)
* ``student`` - the GDN replacement (attention-free at this layer)
* ``swa``     - original attention restricted to a sliding window + attention sinks (zero-training control)
* ``zero``    - mixer output = 0 (zero-training lower bound: attention layer deleted)

Teacher and student therefore share every other weight and one copy of the model lives on the GPU.
"""
from __future__ import annotations

import contextlib
import glob
import os

import torch
import torch.nn as nn

from .gdn import GDN

from .data import DATA_ROOT  # noqa: E402

DEFAULT_MODEL_DIR = os.path.join(DATA_ROOT, "hf", "models--Qwen--Qwen3.5-0.8B", "snapshots")


def resolve_model_dir(path: str | None) -> str:
    path = path or DEFAULT_MODEL_DIR
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    snaps = sorted(glob.glob(os.path.join(path, "*", "config.json")))
    if not snaps:
        raise FileNotFoundError(path)
    if len(snaps) > 1:
        # Fail closed: never pick a snapshot by sort order. Pinned runs must name the exact revision dir.
        raise ValueError(f"ambiguous model dir {path!r}: {len(snaps)} snapshots; pass the exact pinned snapshot path")
    return os.path.dirname(snaps[0])


def tokenizer_dir(cfg: dict) -> str:
    """The tokenizer a conversion config tokenizes with: ``tokenizer_dir`` when the config pins one (jobs/q9af_v2 mode
    clefflash: Clef-flash's weights, the Qwen3.5-9B tokenizer every other q9af_v2 run used), else ``model_dir``."""
    return resolve_model_dir(os.path.expandvars(cfg.get("tokenizer_dir") or cfg["model_dir"]))


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir(cfg))


class MixerSwap(nn.Module):
    def __init__(self, attn: nn.Module, layer_idx: int):
        super().__init__()
        self.attn = attn
        self.gdn: GDN | None = None
        self.layer_idx = layer_idx
        self.mode = "teacher"
        self.swa_window = 512
        self.swa_sinks = 4
        self._mask_cache = {}

    def _swa_mask(self, T, device):
        key = (T, self.swa_window, self.swa_sinks, str(device))
        if key not in self._mask_cache:
            i = torch.arange(T, device=device)[:, None]
            j = torch.arange(T, device=device)[None, :]
            m = (j <= i) & ((i - j < self.swa_window) | (j < self.swa_sinks))
            self._mask_cache = {key: m[None, None]}
        return self._mask_cache[key]

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
        if self.mode == "teacher":
            return self.attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                             past_key_values=past_key_values, **kw)
        if self.mode == "student":
            store = getattr(past_key_values, "_ssdm_state", None) if past_key_values is not None else None
            if store is None:
                return self.gdn(hidden_states), None
            # cached readout (decide.backends.state_fork via ConvertedStateOps): continue the GDN state
            key = ("swap", self.layer_idx)
            out, store[key] = self.gdn.step(hidden_states, store.get(key))
            # position bookkeeping only: this layer's cache slot is attention-typed, and the cache derives positions
            # and mask sizes from attention KV lengths, so append a [B, 1, T, 1] placeholder (no real KV is kept)
            B, T, _ = hidden_states.shape
            ph = hidden_states.new_zeros(B, 1, T, 1)
            past_key_values.update(ph, ph, self.layer_idx)
            return out, None
        if self.mode == "swa":
            m = self._swa_mask(hidden_states.shape[1], hidden_states.device)
            return self.attn(hidden_states, position_embeddings=position_embeddings, attention_mask=m,
                             past_key_values=None, **kw)
        if self.mode == "zero":
            return torch.zeros_like(hidden_states), None
        raise ValueError(self.mode)


class ConvertibleQwen(nn.Module):
    """Thin wrapper: text model + lm_head + the swap points."""

    def __init__(self, full_model, kernel="fla"):
        super().__init__()
        self.lm = full_model.model.language_model
        self.lm_head = full_model.lm_head
        self.cfg = self.lm.config
        self.kernel = kernel
        self.attn_layers = [i for i, t in enumerate(self.cfg.layer_types) if t == "full_attention"]
        self.swaps: dict[int, MixerSwap] = {}
        for i in self.attn_layers:
            layer = self.lm.layers[i]
            sw = MixerSwap(layer.self_attn, i)
            layer.self_attn = sw
            self.swaps[i] = sw

    # ---------------- modes ----------------
    def set_modes(self, modes: dict[int, str]):
        for i, sw in self.swaps.items():
            sw.mode = modes.get(i, "teacher")

    def get_modes(self):
        return {i: sw.mode for i, sw in self.swaps.items()}

    @contextlib.contextmanager
    def modes(self, modes: dict[int, str]):
        old = self.get_modes()
        self.set_modes(modes)
        try:
            yield
        finally:
            self.set_modes(old)

    # ---------------- forward ----------------
    def hidden(self, ids):
        return self.lm(input_ids=ids, use_cache=False).last_hidden_state

    # ---------------- split forward (shared prefix / partial teacher passes) ----------------
    def embed(self, ids):
        """Embeddings + the per-call context every decoder layer needs (same construction as the HF text model)."""
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

        h = self.lm.embed_tokens(ids)
        pos = torch.arange(h.shape[1], device=h.device).view(1, 1, -1).expand(4, h.shape[0], -1)
        text_pos, rope_pos = pos[0], pos[1:]
        kw = {"config": self.cfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": None,
              "position_ids": text_pos}
        masks = {"full_attention": create_causal_mask(**kw), "linear_attention": create_recurrent_attention_mask(**kw)}
        ctx = {"pe": self.lm.rotary_emb(h, rope_pos), "masks": masks, "text_pos": text_pos}
        return h, ctx

    def run_layers(self, h, ctx, start, end, final_norm=False):
        """Decoder layers [start, end) on hidden state h; equals the corresponding slice of ``hidden``."""
        for i in range(start, end):
            h = self.lm.layers[i](h, position_embeddings=ctx["pe"], attention_mask=ctx["masks"][self.cfg.layer_types[i]],
                                  position_ids=ctx["text_pos"], past_key_values=None, use_cache=False)
        return self.lm.norm(h) if final_norm else h

    def logits(self, ids):
        return self.lm_head(self.hidden(ids))

    def gdn_parameters(self, layers=None):
        for i, sw in self.swaps.items():
            if sw.gdn is not None and (layers is None or i in layers):
                yield from sw.gdn.parameters()


def replace_native_gdn(model: ConvertibleQwen, kernel="fla"):
    """Swap the model's own (HF) GDN layers for our parameter-identical GDN so they use the same kernel."""
    n = 0
    for i, t in enumerate(model.cfg.layer_types):
        if t != "linear_attention":
            continue
        layer = model.lm.layers[i]
        old = layer.linear_attn
        new = GDN.from_config(model.cfg, kernel=kernel).to(device=old.out_proj.weight.device,
                                                          dtype=old.out_proj.weight.dtype)
        new.load_state_dict(old.state_dict(), strict=True)
        new.requires_grad_(False)  # native layers stay frozen (fresh modules default to requires_grad=True)
        layer.linear_attn = _NativeGDNShim(new, i)
        n += 1
    return n


class _NativeGDNShim(nn.Module):
    """Accepts the HF decoder layer's linear_attn call signature."""

    def __init__(self, gdn: GDN, layer_idx: int = -1):
        super().__init__()
        self.gdn = gdn
        self.layer_idx = layer_idx

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
        if cache_params is None:
            return self.gdn(hidden_states)
        store = getattr(cache_params, "_ssdm_state", None)
        if store is None:
            raise NotImplementedError("only a ConvertedStateOps cache (with _ssdm_state) is supported here")
        key = ("native", self.layer_idx)
        out, store[key] = self.gdn.step(hidden_states, store.get(key))
        return out


def _force_memory_efficient_sdpa():
    """HF passes enable_gqa=True to SDPA, which rules out the memory-efficient kernel on this torch build and falls
    back to the O(T^2)-memory math kernel (+5.5 GB at 8k). Repeat KV heads in HF instead -> mem-efficient kernel."""
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False


def load(model_dir=None, device="cuda", dtype=torch.bfloat16, kernel="fla", native_kernel=True):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    d = resolve_model_dir(model_dir)
    _force_memory_efficient_sdpa()
    full = Qwen3_5ForConditionalGeneration.from_pretrained(d, dtype=dtype, device_map=device,
                                                           attn_implementation="sdpa")
    full.model.visual = None  # text path only
    full.eval()
    for p in full.parameters():
        p.requires_grad_(False)
    m = ConvertibleQwen(full, kernel=kernel)
    if native_kernel:
        replace_native_gdn(m, kernel=kernel)
    tok = AutoTokenizer.from_pretrained(d)
    return m, tok, d


# ---------------------------------------------------------------- initialisation of the new GDN layer
def init_gdn(model: ConvertibleQwen, i: int, how: str = "reuse_attn", seed: int = 0) -> GDN:
    """Create the student GDN for attention layer ``i``.

    how = ``random``         HF-style fresh init (std 0.02 linears, conv/A/dt as HF)
          ``clone_neighbor`` copy the preceding native GDN layer's weights
          ``reuse_attn``     Mamba-in-the-Llama/LoLCATs-style: q/k/v/gate/o from the attention projections
                             (see ``reuse_attn_weights``: head split on 0.8B, per-head SVD key projection on
                             9B), decay/beta/norm from the preceding native GDN layer, conv1d = identity tap.
    """
    cfg = model.cfg
    attn = model.swaps[i].attn
    dev, dt = attn.o_proj.weight.device, attn.o_proj.weight.dtype
    g = torch.Generator().manual_seed(seed + 1000 * i)
    gdn = GDN.from_config(cfg, kernel=model.kernel)
    with torch.no_grad():
        for mod in gdn.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.copy_(torch.randn(mod.weight.shape, generator=g) * cfg.initializer_range)
        A = torch.empty(gdn.nv).uniform_(1, 16, generator=g)
        gdn.A_log.copy_(torch.log(A))
        # dt_bias as in Mamba: inverse-softplus of dt ~ logU(1e-3, 1e-1)
        dt_ = torch.exp(torch.rand(gdn.nv, generator=g) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(1e-3)))
                        + torch.log(torch.tensor(1e-3)))
        gdn.dt_bias.copy_(dt_ + torch.log(-torch.expm1(-dt_)))
    gdn = gdn.to(device=dev, dtype=dt)
    if how == "random":
        return gdn
    # neighbouring native GDN layer
    j = i - 1
    while cfg.layer_types[j] != "linear_attention":
        j -= 1
    nb = model.lm.layers[j].linear_attn
    nb = nb.gdn if hasattr(nb, "gdn") else nb
    if how == "clone_neighbor":
        gdn.load_state_dict(nb.state_dict())
        return gdn
    if how != "reuse_attn":
        raise ValueError(how)
    with torch.no_grad():
        W = reuse_attn_weights(attn, cfg, gdn.nk, gdn.dk, gdn.nv, gdn.dv)
        gdn.in_proj_qkv.weight.copy_(torch.cat([W["q"], W["k"], W["v"]], 0).to(dt))
        gdn.in_proj_z.weight.copy_(W["z"].to(dt))
        gdn.out_proj.weight.copy_(W["o"].to(dt))
        cw = torch.zeros_like(gdn.conv1d.weight)
        cw[..., -1] = 1.0
        gdn.conv1d.weight.copy_(cw)
        gdn.dt_bias.copy_(nb.dt_bias)
        gdn.A_log.copy_(nb.A_log)
        gdn.in_proj_a.load_state_dict(nb.in_proj_a.state_dict())
        gdn.in_proj_b.load_state_dict(nb.in_proj_b.state_dict())
        gdn.norm.load_state_dict(nb.norm.state_dict())
    gdn.reuse_info = W["info"]
    return gdn


@torch.no_grad()
def reuse_attn_weights(attn, cfg, nk, dk, nv, dv):
    """Map a gated GQA softmax-attention layer onto GDN projections (Mamba-in-the-Llama / LoLCATs-style reuse).

    Value side (needs nv*dv == H*hd): value head j <- half (j % s) of attention head j // s, s = hd // dv;
    v from the GQA-repeated v_proj rows, z (output gate) from the q_proj gate rows, out_proj = o_proj.
    Key side, two supported layouts:
      * ``split``  nk*dk == H*hd (Qwen3.5-0.8B: 8x256 -> 16x128): key head i <- half (i % s) of attention head i // s.
      * ``proj``   nk == H, dk < hd, nv/nk == hd/dv (Qwen3.5-9B: 16x256 -> 16x128 keys, 32x128 values; GDN repeats key
                   head i for value heads 2i, 2i+1 = the two halves of attention head i): key head i <- attention head
                   i compressed hd -> dk by a data-free per-head SVD basis P_i (top-dk left singular vectors of
                   [Wq_i | Wk_i]); q'.k' = q^T P^T P k keeps the dominant shared subspace of q and k.
    q_norm / k_norm scales (1 + w) are folded into the rows (direction-preserving only per normalised block); the
    RMS normalisation itself is replaced by GDN's L2 norm after conv + SiLU, so this is an initialisation, not an
    equivalence.
    """
    H, hd, KV, hid = cfg.num_attention_heads, cfg.head_dim, cfg.num_key_value_heads, cfg.hidden_size
    if nv * dv != H * hd or hd % dv:
        raise ValueError(f"reuse_attn: value dims {nv}x{dv} incompatible with attention {H}x{hd}; use clone_neighbor")
    f = lambda t: t.float()  # noqa: E731
    Wq_all = f(attn.q_proj.weight).view(H, 2 * hd, hid)
    Wq, Wgate = Wq_all[:, :hd], Wq_all[:, hd:]
    Wq = Wq * (1.0 + f(attn.q_norm.weight))[None, :, None]
    Wk = f(attn.k_proj.weight).view(KV, hd, hid) * (1.0 + f(attn.k_norm.weight))[None, :, None]
    Wv = f(attn.v_proj.weight).view(KV, hd, hid)
    rep = H // KV
    Wk, Wv = Wk.repeat_interleave(rep, 0), Wv.repeat_interleave(rep, 0)  # [H, hd, hid]
    out = {"v": Wv.reshape(H * hd, hid), "z": Wgate.reshape(H * hd, hid), "o": f(attn.o_proj.weight)}
    if nk * dk == H * hd and hd % dk == 0:
        out["q"], out["k"] = Wq.reshape(H * hd, hid), Wk.reshape(H * hd, hid)
        out["info"] = {"key_map": "split", "retained_energy": 1.0}
    elif nk == H and dk < hd and nv // nk == hd // dv:
        qs, ks, en = [], [], []
        for h in range(H):
            U, S, _ = torch.linalg.svd(torch.cat([Wq[h], Wk[h]], 1), full_matrices=False)
            P = U[:, :dk].T  # [dk, hd], orthonormal rows
            qs.append(P @ Wq[h])
            ks.append(P @ Wk[h])
            en.append(float((S[:dk] ** 2).sum() / (S ** 2).sum()))
        out["q"], out["k"] = torch.cat(qs, 0), torch.cat(ks, 0)
        out["info"] = {"key_map": "proj", "retained_energy": sum(en) / len(en), "retained_energy_min": min(en)}
    else:
        raise ValueError(f"reuse_attn: key dims {nk}x{dk} vs attention {H}x{hd} unsupported; use clone_neighbor")
    return out


def attach_gdn(model: ConvertibleQwen, i: int, gdn: GDN, train_dtype=torch.float32):
    model.swaps[i].gdn = gdn.to(dtype=train_dtype)
    for p in gdn.parameters():
        p.requires_grad_(True)
    return gdn


class ConvertedStateOps:
    """Prefix-fork readout (decide.backends.state_fork) over a ConvertibleQwen in any mix of modes: teacher attention
    keeps its KV in an HF DynamicCache; native and student GDNs keep (conv tail, S) in ``cache._ssdm_state``."""

    def __init__(self, model: ConvertibleQwen, autocast_dtype=None):
        self.model = model
        self.autocast_dtype = autocast_dtype

    def _run(self, ids, cache, n_last):
        dev = ids.device.type
        ctx = (torch.autocast(device_type=dev, dtype=self.autocast_dtype) if self.autocast_dtype is not None
               else contextlib.nullcontext())
        with ctx:
            h = self.model.lm(input_ids=ids, past_key_values=cache, use_cache=True).last_hidden_state
        return self.model.lm_head(h[:, -n_last:]).float(), cache

    def _clone(self, c):
        import copy

        return copy.deepcopy(c, {id(c.config): c.config} if getattr(c, "config", None) is not None else {})

    def prefill(self, ids):
        from transformers import DynamicCache

        cache = DynamicCache(config=self.model.cfg)
        cache._ssdm_state = {}
        return self._run(ids, cache, 1)

    def extend(self, cache, ids, n_last):
        from decide.backends.state_fork import _batch, expand

        c = self._clone(cache)
        if ids.shape[0] > 1 and _batch(c) == 1:
            expand(c, ids.shape[0])
        return self._run(ids, c, n_last)

    def select(self, cache, i):
        from decide.backends.state_fork import select_row

        c = self._clone(cache)
        select_row(c, i)
        return c
