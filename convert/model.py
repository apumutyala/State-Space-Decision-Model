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

DEFAULT_MODEL_DIR = r"C:\Users\apumu\research\CHLabs\ssdm-data\hf\models--Qwen--Qwen3.5-0.8B\snapshots"


def resolve_model_dir(path: str | None) -> str:
    path = path or DEFAULT_MODEL_DIR
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    snaps = sorted(glob.glob(os.path.join(path, "*", "config.json")))
    if not snaps:
        raise FileNotFoundError(path)
    return os.path.dirname(snaps[-1])


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
            return self.gdn(hidden_states), None
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
        layer.linear_attn = _NativeGDNShim(new)
        n += 1
    return n


class _NativeGDNShim(nn.Module):
    """Accepts the HF decoder layer's linear_attn call signature."""

    def __init__(self, gdn: GDN):
        super().__init__()
        self.gdn = gdn

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
        if cache_params is not None:
            raise NotImplementedError("pilot uses no KV/recurrent cache (teacher-forced probes only)")
        return self.gdn(hidden_states)


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
                             (8 heads x 256 split into 16 x 128; GQA kv heads repeated), decay/beta/norm
                             from the preceding native GDN layer, conv1d = identity (delta at last tap).
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
    H, hd = cfg.num_attention_heads, cfg.head_dim
    KV = cfg.num_key_value_heads
    nk, dk, nv, dv = gdn.nk, gdn.dk, gdn.nv, gdn.dv
    assert H * hd == nk * dk == nv * dv, "head-splitting reuse needs H*head_dim == GDN key/value dims"
    split = hd // dk  # 256 -> 2 x 128
    hid = cfg.hidden_size
    with torch.no_grad():
        Wq_all = attn.q_proj.weight.view(H, 2 * hd, hid)
        Wq, Wgate = Wq_all[:, :hd], Wq_all[:, hd:]
        qn = (1.0 + attn.q_norm.weight.float()).to(dt)  # fold zero-centred RMSNorm scale (direction-preserving)
        kn = (1.0 + attn.k_norm.weight.float()).to(dt)
        Wq = Wq * qn[None, :, None]
        Wk = attn.k_proj.weight.view(KV, hd, hid) * kn[None, :, None]
        Wv = attn.v_proj.weight.view(KV, hd, hid)
        rep = H // KV
        Wk = Wk.repeat_interleave(rep, 0)  # [H, hd, hid]
        Wv = Wv.repeat_interleave(rep, 0)
        # GDN head j = split*h + s takes rows s*dk:(s+1)*dk of attention head h -> contiguous reshape
        q_new = Wq.reshape(H * hd, hid)
        k_new = Wk.reshape(H * hd, hid)
        v_new = Wv.reshape(H * hd, hid)
        gdn.in_proj_qkv.weight.copy_(torch.cat([q_new, k_new, v_new], 0))
        gdn.in_proj_z.weight.copy_(Wgate.reshape(H * hd, hid))
        gdn.out_proj.weight.copy_(attn.o_proj.weight)
        cw = torch.zeros_like(gdn.conv1d.weight)
        cw[..., -1] = 1.0
        gdn.conv1d.weight.copy_(cw)
        for name in ("dt_bias", "A_log", "in_proj_a", "in_proj_b", "norm"):
            getattr(gdn, name).load_state_dict(getattr(nb, name).state_dict()) if isinstance(
                getattr(gdn, name), nn.Module) else getattr(gdn, name).copy_(getattr(nb, name))
    _ = split
    return gdn


def attach_gdn(model: ConvertibleQwen, i: int, gdn: GDN, train_dtype=torch.float32):
    model.swaps[i].gdn = gdn.to(dtype=train_dtype)
    for p in gdn.parameters():
        p.requires_grad_(True)
    return gdn
