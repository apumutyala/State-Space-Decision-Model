"""FLOP and memory model for attention->GDN conversion stages (pure arithmetic; CPU-testable).

Parameter counts come from the config dims. FLOPs: 2 per parameter per token for a forward matmul, 2 more for an
activation backward, 2 more for a weight-gradient backward. Memory terms (bytes):

  weights       2 * N_total                                  (bf16 resident model, vision tower dropped)
  gdn_state     4 * P_gdn * n_attached + 12 * P_gdn * n_train (fp32 params; fp32 grad + Adam m, v for trained ones)
  stage 1       ALPHA * u * tokens_mb * graphs_alive          student GDN graph(s) for back-prop
                + (3*I + 6*H) * 2 * tokens_mb                 teacher no-grad transient of one decoder layer
  stage 2       2 * H * tokens_mb * n_suffix_layers           activation-checkpoint inputs of the student suffix
                + ((4*I + 10*H) * 2 + ALPHA * u) * tokens_mb  one layer re-forwarded with grad during backward
                + 2 * 2 * H * tokens_mb                       teacher/student final hidden states
                + 3 * chunk * V * 4                           checkpointed KL chunk (fp32 logits, live copies)
  u = conv_dim + 3 * value_dim of the GDN (q,k,v conv stream + v/z/norm-out streams)

ALPHA (bytes per u-unit per token per live GDN graph) is calibrated on the paid run job-c4zkm: 9B stage 1 with all 8
graphs alive at 8x4096 tokens peaked at 68.92 GB with 17.91 GB of weights -> ALPHA = 7.14. The same ALPHA predicts
the 0.8B stage-1 peak (11.7 vs 11.8 GB observed) and over-predicts the 0.8B stage-2 peak (5.7 GB observed); tests check both.
A FRAG factor (1.10) covers allocator fragmentation.
"""
from __future__ import annotations

from dataclasses import dataclass

ALPHA = 7.14
FRAG = 1.10
H100_PEAK = 989e12


@dataclass
class Dims:
    hidden: int
    inter: int
    layers: int
    layer_types: list
    heads: int
    kv_heads: int
    head_dim: int
    nk: int
    dk: int
    nv: int
    dv: int
    conv_k: int
    vocab: int
    tied: bool

    @classmethod
    def from_config(cls, c):
        return cls(c.hidden_size, c.intermediate_size, c.num_hidden_layers, list(c.layer_types), c.num_attention_heads,
                   c.num_key_value_heads, c.head_dim, c.linear_num_key_heads, c.linear_key_head_dim,
                   c.linear_num_value_heads, c.linear_value_head_dim, c.linear_conv_kernel_dim, c.vocab_size,
                   bool(getattr(c, "tie_word_embeddings", False)))

    # ---- parameter counts
    @property
    def p_mlp(self):
        return 3 * self.hidden * self.inter

    @property
    def p_attn(self):
        H, hd = self.hidden, self.head_dim
        return H * self.heads * hd * 2 + 2 * H * self.kv_heads * hd + self.heads * hd * H + 2 * hd

    @property
    def conv_dim(self):
        return 2 * self.nk * self.dk + self.nv * self.dv

    @property
    def p_gdn(self):
        H, vd = self.hidden, self.nv * self.dv
        return H * self.conv_dim + H * vd + vd * H + 2 * H * self.nv + self.conv_dim * self.conv_k + 2 * self.nv + self.dv

    def p_layer(self, i, converted=()):
        mixer = self.p_gdn if (self.layer_types[i] == "linear_attention" or i in converted) else self.p_attn
        return mixer + self.p_mlp + 2 * self.hidden

    @property
    def n_body(self):
        return sum(self.p_layer(i) for i in range(self.layers))

    @property
    def p_head(self):
        return self.vocab * self.hidden

    @property
    def n_total(self):
        return self.n_body + self.p_head * (1 if self.tied else 2)

    @property
    def attn_layers(self):
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def u(self):
        return self.conv_dim + 3 * self.nv * self.dv


# ---------------------------------------------------------------- FLOPs per token
def flops_stage1(d: Dims, aligned):
    """Teacher forward up to the last aligned layer + fwd/act-bwd/weight-bwd of each aligned GDN."""
    upto = max(aligned) + 1
    return 2 * sum(d.p_layer(i) for i in range(upto)) + 6 * d.p_gdn * len(aligned)


def flops_stage2(d: Dims, student_layers, n_train=1):
    """Shared prefix fwd + teacher suffix fwd + student suffix fwd, checkpoint re-forward and activation backward
    + KL head (teacher/student logits fwd, re-forward of both in the checkpointed chunk, grad wrt hs) + weight grads
    of the trained GDN(s)."""
    first = min(student_layers)
    pre = sum(d.p_layer(i) for i in range(first))
    suf_t = sum(d.p_layer(i) for i in range(first, d.layers))
    suf_s = sum(d.p_layer(i, converted=student_layers) for i in range(first, d.layers))
    head = 2 * d.p_head * (2 + 2 + 1)
    return 2 * pre + 2 * suf_t + 3 * 2 * suf_s + head + 2 * d.p_gdn * n_train


def min_tok_s(flop_per_token, min_mfu, peak=H100_PEAK):
    return min_mfu * peak / flop_per_token


# ---------------------------------------------------------------- memory (bytes)
def weights_bytes(d: Dims):
    return 2 * d.n_total


def gdn_state_bytes(d: Dims, n_attached, n_train):
    return 4 * d.p_gdn * n_attached + 12 * d.p_gdn * n_train


def stage1_peak(d: Dims, tokens_mb, graphs_alive, n_attached=None, frag=FRAG):
    n_att = len(d.attn_layers) if n_attached is None else n_attached
    act = ALPHA * d.u * tokens_mb * graphs_alive + (3 * d.inter + 6 * d.hidden) * 2 * tokens_mb
    return weights_bytes(d) + gdn_state_bytes(d, n_att, n_att) + frag * act


def stage2_peak(d: Dims, tokens_mb, first_student_layer, n_attached=None, n_train=1, chunk=512, frag=FRAG):
    n_att = len(d.attn_layers) if n_attached is None else n_attached
    n_suf = d.layers - first_student_layer
    act = (2 * d.hidden * tokens_mb * n_suf
           + ((4 * d.inter + 10 * d.hidden) * 2 + ALPHA * d.u) * tokens_mb
           + 2 * 2 * d.hidden * tokens_mb
           + 3 * chunk * d.vocab * 4)
    return weights_bytes(d) + gdn_state_bytes(d, n_att, n_train) + frag * act


# ---------------------------------------------------------------- the two Qwen3.5 shapes (config.json values)
def _types(n):
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]


QWEN35_9B = Dims(hidden=4096, inter=12288, layers=32, layer_types=_types(32), heads=16, kv_heads=4, head_dim=256,
                 nk=16, dk=128, nv=32, dv=128, conv_k=4, vocab=248320, tied=False)
QWEN35_0P8B = Dims(hidden=1024, inter=3584, layers=24, layer_types=_types(24), heads=8, kv_heads=2, head_dim=256,
                   nk=16, dk=128, nv=16, dv=128, conv_k=4, vocab=248320, tied=True)


def plan(d: Dims, batch, seq, micro_batch, stage2_first_layer, min_mfu=0.125, target_mfu=0.25):
    """Expected peak GB and tok/s band for a stage-1/stage-2 configuration (micro-batch with accumulation)."""
    t_mb = micro_batch * seq
    A = d.attn_layers
    f1, f2 = flops_stage1(d, A), flops_stage2(d, [stage2_first_layer])
    return {
        "tokens_per_step": batch * seq, "tokens_per_micro_batch": t_mb, "accum_steps": batch // micro_batch,
        "stage1": {"peak_gb": stage1_peak(d, t_mb, graphs_alive=1) / 1e9, "flop_per_token": f1,
                   "tok_s_at_target_mfu": min_tok_s(f1, target_mfu), "min_tok_s": min_tok_s(f1, min_mfu)},
        "stage2": {"peak_gb": stage2_peak(d, t_mb, stage2_first_layer) / 1e9, "flop_per_token": f2,
                   "tok_s_at_target_mfu": min_tok_s(f2, target_mfu), "min_tok_s": min_tok_s(f2, min_mfu)},
    }
