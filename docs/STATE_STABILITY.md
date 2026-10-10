# State stability bounds for our two backbones

Control-of-learning program, leg 4 of `CONTROL_RESEARCH.md` ("stability of the state"). Date: 2026-10-10. It covers
two backbones: RWKV-7 (the attention-free line, D19 on `stack/rwkv7/d19-v3b-joint`) and the Gated DeltaNet layers of
Qwen3.5-4B (the hybrid main line). This is CPU-only math plus numpy receipts
(`scripts/state_stability_numeric_check.py`, prints `ALL PASS`). No model was loaded. The empirical half is a GPU
probe, written but not submitted (section 6). SAFETY_SPEC §5.2 still applies: none of this is a safety claim on its
own. Decay bounds how long an injection stays in the state. They do not bound whether it gets in.

## Takeaway

1. **GDN (Qwen3.5) is contractive for every input.** The exact norm is `‖A_t‖₂ = α_t · max(1, |1 − β_t‖k_t‖²|)`. Our
   code has β = sigmoid ∈ (0,1) and an in-kernel l2norm, so ‖k‖ < 1 and `‖A_t‖₂ = α_t < 1`. The delta rule adds no
   contraction in the worst direction, so the whole forgetting rate is α. The existing universal constant `ρ_h`
   (`controlsafe/adapters.py:293-294`) is sound. In practice it may sit so close to 1 that float64 rounds it to
   exactly 1.0. The probe therefore reports `−log ρ_h` instead.
2. **RWKV-7 is not contractive in general, and this is now proved at the parameter-box level.** Each step has a
   real spectrum inside (−0.455, 1), so ρ(A_t) < 1 always. But ‖A_t‖₂ reaches ≥ 1.13 (supremum 2/√3). More
   importantly, two admissible steps can alternate with ρ(A₂A₁)^{1/2} = 1.105, which grows 4.4e8× in 200 steps. So
   the joint spectral radius is above 1, and no input-independent norm of any kind makes every admissible RWKV-7 step
   non-expansive. This closes the "weighted norm or JSR remains open" item in `controlsafe/CONTROLSAFE.md:94`, in
   the negative.
3. **New for RWKV-7: a cheap per-step certificate and a reparameterisation that fixes it.** The certificate is
   `‖A_t‖₂ ≤ max(w_max, ā − w_min) + σ_a(κ)`, where σ_a is the κ²-weighted spread of the in-context rate `a` across
   a head's channels. If `a` is shared per head (σ_a = 0), RWKV-7 is contractive for every input, with
   `ρ = max(decay ceiling, 0.455)`. This is the reparameterisation `CONTROL_RESEARCH.md` §4 asks for.
4. **ISS and forgetting.** A contractive layer gives `‖S_t‖ ≤ ρ^t‖S_0‖ + b/(1−ρ)`. An injected span's influence on
   a layer whose later inputs are clean is bounded exactly (no linearisation) by `∏‖A_s‖ · ‖ΔS‖` after the span.
   Across a stack of L recurrent layers it decays like `poly(τ)^{L−1} · ρ^τ`, with per-input constants.
5. **Where it fails.**
   - The attention layers of the hybrid. Qwen3.5-4B has 8 of 32, starting at layer 3. They give no decay with
     distance, so after layer 3 there is no forgetting bound at all. For the hybrid, injection robustness has to
     come from G15 trust gating.
   - GDN with β → 2 (the `allow_neg_eigval` mode) combined with bf16-inflated ‖k‖².
   - An unnormalised k.
   - bf16 state storage or α rounding to 1.
   - The output norms (GroupNorm, gated RMSNorm), whose local Lipschitz constant ∝ 1/rms.
   - "Bounded state ⇒ bounded logits" is true but empty, because the final RMSNorm bounds the logits for ANY
     state. The useful statement is the influence (Lipschitz) bound.

## 0. Setup, conventions and the code

Every recurrent layer is written per head and key-first, as in `controlsafe/adapters.py:4-15`:

    X_t = A_t X_{t-1} + B_t,   X_t ∈ R^{N×V},   y_t = X_tᵀ r_t.

The map on the whole state is `I_V ⊗ A_t`. So `‖A_t X‖_F ≤ ‖A_t‖₂ ‖X‖_F`, and the induced Frobenius-to-Frobenius
norm is exactly `‖A_t‖₂`. All norms on states below are Frobenius, and all norms on transitions are spectral.

| | RWKV-7 (vendored HF + fla `chunk_rwkv7`) | GDN (HF `Qwen3_5GatedDeltaNet` + fla `chunk_gated_delta_rule`) |
|---|---|---|
| transition `A_t` | `diag(w_t) − diag(a_t) κ_t κ_tᵀ` (adapters.py:138; the reference `S ← S·diag(w) + S(−κ)(κ⊙a)ᵀ + v kᵀ`, modeling_rwkv7.py:124-128, 143-151) | `α_t (I − β_t k_t k_tᵀ)` (adapters.py:141; HF reference loop modeling_qwen3_5.py:354-364) |
| decay | `w = exp(w_log)`, `w_log = −e^{−1/2} σ(w0 + tanh(x W1) W2)` ⇒ `w ∈ (e^{−e^{−1/2}}, 1) = (0.5452, 1)` (modeling_rwkv7.py:87-91, 561-563) | `α = exp(g)`, `g = −e^{A_log} softplus(W_a x + dt_bias)` ⇒ α ∈ (0,1) (modeling_qwen3_5.py:522) |
| rate | `a = σ(a0 + x A1 A2) ∈ (0,1)` **per channel** (modeling_rwkv7.py:564) | `β = σ(W_b x) ∈ (0,1)` **per head** (modeling_qwen3_5.py:520) |
| key in A | `κ = normalize(k ⊙ k_k)`, F.normalize eps 1e-12 ⇒ ‖κ‖ = 1 (or κ = 0) (modeling_rwkv7.py:490-491) | `k = l2norm(k)` in-kernel, `x·rsqrt(Σx² + 1e-6)` ⇒ ‖k‖ < 1 (modeling_qwen3_5.py:242-245, 536, 547; fla-core 0.5.2 `fla/modules/l2norm.py` eps=1e-6, fp32 compute, stored in input dtype) |
| write `B_t` | `k̃ vᵀ`, `k̃ = k ⊙ (1 + (a−1) k_a)` **unnormalised** (modeling_rwkv7.py:501) | `β k vᵀ` (the erase is inside A_t; adapters.py:218-219) |
| local window w | 1 (token shift, modeling_rwkv7.py:452) | 3 (causal conv, kernel 4; adapters.py:88-94) |
| state dtype | `wkv_state_dtype`, default float32, bf16 allowed (configuration_rwkv7.py:122, 140) | fla returns an fp32 final state; the HF torch reference runs in fp32 (modeling_qwen3_5.py:336-352) |

The prompt that started this work wrote RWKV-7's decay as `exp(−exp(·))`. Our checkpoint and fla 0.5.2
(`fla/layers/rwkv7.py`, `w = −0.6065306597126334 * w_lora(xw).sigmoid()`) both use `exp(−e^{−1/2}σ(·))` instead.
The floor is therefore 0.5452, not 0. The fla mapping (`a = −κ`, `b = κ⊙a`, `scale = 1`, `h = Sᵀ`) is in
`heads/rwkv7_wkv.py:11-18, 155-162`, checked against fla-core 0.5.2 `generalized_delta_rule/dplr/naive.py` (wheel
sha256 `5e830c85…` in `jobs/fla_rwkv7/requirements-fla.txt`). The decay and `a` LoRAs (`w1, w2, a1, a2`) are raw
`nn.Parameter`s (modeling_rwkv7.py:399-404). `train.lora.apply_lora` wraps only `nn.Linear` (train/lora.py:67), so
a trained D19 cell keeps the parent's decay ceilings.

The Qwen3.5 citations are from transformers 5.14.1 (the laptop copy). The image pins 5.17.0
(`jobs/qual/Dockerfile:10`), so check the line numbers there before quoting them elsewhere.

Qwen3.5-4B (HF `config.json`) has 32 layers and `full_attention_interval = 4`. Full attention sits at layers 3, 7, 11,
…, 31, and GDN at the other 24. It has 16 key heads and 32 value heads, d_k = d_v = 128, conv kernel 4, hidden size
2560, and runs in bf16.

## 1. Transition norms

**Theorem T1 (GDN, exact).** For α ≥ 0, β ∈ R and k ∈ R^N:

    ‖α(I − β k kᵀ)‖₂ = α · max(1, |1 − β‖k‖²|).

Hence `‖A_t‖₂ ≤ α_t` **iff** `0 ≤ β_t‖k_t‖² ≤ 2`, and `‖A_t‖₂ = α_t` exactly whenever N ≥ 2.

*Proof.* The matrix `I − βkkᵀ` is symmetric. Its eigenvalue is 1 on k^⊥ (dimension N−1 ≥ 1) and `1 − β‖k‖²` on k. ∎

*In our code.* β ∈ (0,1) and ‖k‖² = s/(s + 1e-6) < 1, so `β‖k‖² ∈ [0,1)` and `‖A_t‖₂ = α_t < 1` for every input.
Two consequences:

- The forgetting rate of a GDN layer is **α alone**. The delta-rule erase shrinks only the k direction, so it never
  helps a worst-case bound.
- **Universal constant (CERTIFIED for all inputs).** The layer input is the RMSNorm output `x = (1+γ) ⊙ x̂` with
  `‖x̂‖₂ < √d` (modeling_qwen3_5.py:746-753). So `W_a[h]·x ≥ −√d ‖W_a[h] ⊙ (1+γ)‖₂`, and

      α_t,h ≤ ρ_h = exp(−e^{A_log_h} softplus(dt_bias_h − √d‖W_a[h]⊙(1+γ)‖₂)) < 1   (adapters.py:286-296).

  Since `‖A_T⋯A_1‖₂ ≤ ρ_h^T` for every input sequence, this is a common induced norm. The LoRA'd `in_proj_a` is
  covered through `_eff_weight` (adapters.py:258-264).

**Theorem T2 (RWKV-7).** Let `A = D − Λκκᵀ` with `D = diag(w)`, `w_i ∈ [w_lo, 1)`, `w_lo = e^{−e^{−1/2}} = 0.5452`,
`Λ = diag(a)`, `a_i ∈ (0,1)` and `‖κ‖ = 1`.

- **(a) The spectrum is real and strictly inside the unit disc.** `A` is similar, via `Λ^{−1/2}`, to the symmetric
  matrix `D − zzᵀ` with `z = Λ^{1/2}κ` and `‖z‖² = Σa_iκ_i² ∈ (0,1)`. So the eigenvalues are real and lie in
  `[w_min − ‖z‖², w_max] ⊂ (w_lo − 1, 1) = (−0.455, 1)`. Therefore ρ(A_t) < 1 for every input. (numpy R3: 0 of 5000
  random draws violate it.)

- **(b) Per-step certificate (CERTIFIED per input; cheap).** For any c ≥ 0:

      ‖A‖₂ ≤ max(w_max, c − w_min) + ‖(Λ − cI)κ‖₂,

  and the best general choice is `c = ā = Σa_iκ_i²`:

      U_t = max(w_max, ā − w_min) + σ_a,   σ_a = (Σ κ_i² (a_i − ā)²)^{1/2}.

  *Proof.* Split `A = (D − cκκᵀ) + (cI − Λ)κκᵀ`. By Weyl, the symmetric first term has eigenvalues in
  `[w_min − c, w_max]`. The second term has rank one and norm `‖(cI − Λ)κ‖ ‖κ‖`. ∎

  For a non-unit κ (bf16 rounding), replace `ā − w_min` with `ā‖κ‖² − w_min` and σ_a with `‖κ‖·‖(Λ − āI)κ‖`, where
  `ā = Σa_iκ_i²/‖κ‖²`. This general form is the one the probe computes. (numpy R2: U ≥ ‖A‖₂ on 20,000 random draws;
  the minimum slack is +7e-5.)

- **(c) Scalar-a heads are contractive for every input.** If `a_t` is constant across a head's channels, then σ_a = 0,
  A is symmetric, and `‖A_t‖₂ ≤ max(w_max, ā − w_min) ≤ max(w̄, 1 − w_lo) < 1`. Here w̄ is the universal per-layer
  decay ceiling `exp(−e^{−1/2} σ(w0_c − ‖W2[:,c]‖₁))` (adapters.py:298-302) and `1 − w_lo = 0.4548`. **Recommended
  reparameterisation** to give RWKV-7 the GDN guarantee: share `a` per head, or penalise or clamp σ_a so that
  `max(w̄, 0.455) + σ_a,max < 1`.

- **(d) There is no common 2-norm bound ≤ 1** (this was already known; `tests/controlsafe` has a counterexample). An
  explicit one: `w = (.99,.99)`, `a = (.99,.01)`, `κ = (1,1)/√2` gives `‖A‖₂ = 1.1295` while ρ(A) = 0.99 (R1). The
  supremum over the box is at least 2/√3 ≈ 1.1547 (the closure `w = 1`, `a = (1,0)`). It embeds into any head size
  by putting κ on two channels.

- **(e) The joint spectral radius of the admissible box is > 1 (new).** Take `w = 0.99` on both channels.
  `A₁` has `a = (.99,.01)` and `κ₁ = (0.4067, 0.9135)`. `A₂` has `a = (.01,.99)` and `κ₂ = (0.9135, 0.4067)`. Then
  `ρ(A₂A₁)^{1/2} = 1.1046`, and `‖(A₂A₁)^{100}‖₂ = 4.4e8`, even though every factor has ρ ≤ 0.99 (R4).
  Consequences:
  1. **No norm, weighted or not, that is independent of the input** gives `‖A‖ ≤ 1` on the whole box, because a JSR
     above 1 rules out every extremal norm. The time-varying weighted norm of (a), `‖Λ_t^{−1/2}·‖`, contracts each
     step, but changing norms between steps costs `max_i (a_{t,i}/a_{t+1,i})^{1/2}`. That cost is unbounded as
     a → 0 or 1, which is exactly the LPV switching failure Cao et al. (ICML 2026) describe for Mamba.
  2. In exact arithmetic, an input that drives `a` and `κ` through such a cycle makes the state of a **zero-input**
     RWKV-7 layer grow geometrically. Whether trained weights can realise such a cycle is an empirical question, and
     the probe measures it (section 6: `exact_window_logprod > 0`).

## 2. Input-to-state stability (ISS / BIBO)

**Theorem T3.** If `‖A_t‖₂ ≤ n_t` and `‖B_t‖_F ≤ b_t`, then

    ‖X_t‖ ≤ (∏_{s≤t} n_s) ‖X_0‖ + Σ_{s≤t} (∏_{u=s+1}^{t} n_u) b_s.

- With `n_t ≤ 1`: `‖X_t‖ ≤ ‖X_0‖ + Σ_s b_s`, which grows at most linearly.
- With `n_t ≤ ρ < 1` and `b_t ≤ b`: `‖X_t‖ ≤ ρ^t ‖X_0‖ + b(1 − ρ^t)/(1 − ρ) ≤ ‖X_0‖ + b/(1−ρ)`, which is bounded
  uniformly in t.

The same bound absorbs additive kernel error `e_t` (chunked forms, rounding) as `Σ ∏ n_u e_s`. So a realised kernel
stays within `e/(1−ρ)` of the exact recurrence when ρ < 1, and is NOT controlled when ρ ≥ 1.

*Proof.* Take the norm of `X_t = A_t X_{t−1} + B_t` and unroll. ∎

- **GDN.** `‖B_t‖ = β‖k‖‖v‖ < ‖v_t‖`. `v = silu(conv₄(W_v x))`, `|silu(z)| ≤ |z|`, so
  `‖v_t‖ ≤ Σ_{j<4} ‖diag(c_j) W_v[h] diag(1+γ)‖₂ √d =: b_h`. That gives a weights-only BIBO constant
  `S_max,h = b_h/(1 − ρ_h)`. It is honest, but vacuous whenever `1 − ρ_h` is tiny (see F3). The useful version is
  per input: plug in the observed `n_t = α_t` and `b_t`. The probe checks the final `‖S_T‖` against it.
- **RWKV-7.** `‖B_t‖ = ‖k̃_t‖‖v_t‖` is bounded through the LayerNorm output and the weights. ISS holds **only
  conditionally**: per input, with `n_t = U_t` (or the exact norm). By T2(e), no universal ISS exists over the
  parameter box. With scalar-a heads (T2(c)), RWKV-7 gets the uniform GDN form.

## 3. Forgetting: the influence of an injected span

Let two token streams agree at every position after `t₁`; the span ends at `t₁`.

**Theorem T4 (one layer, exact, CERTIFIED per input).** Take a recurrent layer whose input stream is identical in
both runs after `t₁`. For `s > t₁ + w`, `A_s` and `B_s` coincide, so `ΔX_s = A_s ΔX_{s−1}` exactly. No linearisation
is involved, because the recurrence is linear in X given its inputs. Hence

    ‖ΔX_{t₁+L}‖ ≤ ∏_{s=t₁+w+1}^{t₁+L} ‖A_s‖₂ · ‖ΔX_{t₁+w}‖ ≤ ρ^{L−w} · 2S_max      (per head; ‖Δy‖ ≤ ‖r‖ ‖ΔX‖).

This is exactly `persistence.layer_local_curve` (persistence.py:5-15, 116-131), which checks it with the observed
`∏‖A_s‖`. The hypothesis "input identical after t₁" holds for the **first recurrent layer** (its input is token
embeddings). It also holds for any layer whose input is provably independent of the span: the G15 protected stream
is the structural case.

**Theorem T5 (cascade across a stack of recurrent layers, per-input constants).** Deeper layers see a perturbed input
after `t₁`, because the upstream states differ. Along the two trajectories, assume local Lipschitz bounds:

    ‖Δx^{l+1}_t‖ ≤ (1 + M_l)‖Δx^l_t‖ + d_l‖ΔX^l_t‖              (residual, readout, output norm, FFN)
    ‖ΔX^l_t‖    ≤ ρ ‖ΔX^l_{t−1}‖ + c_l max_{j≤w} ‖Δx^l_{t−j}‖,    c_l = Lip_x(A)·S_max + Lip_x(B).

Then `‖ΔX^l_{t₁+τ}‖ ≤ K_l (τ+1)^{l−1} ρ^{τ}`, with `K_l` explicit in `(c, d, M, ρ^{−w}, ‖ΔX^1_{t₁+w}‖)`.

*Proof.* Induction on l, using `Σ_{s≤τ} ρ^{τ−s}(s+1)^k ρ^s ≤ (τ+1)^{k+1} ρ^τ`. ∎

**What bounded state buys** is precisely the finiteness of `c_l`: the term `ΔA_t · X_{t−1}` is controlled by
`S_max`. This is the formal version of the "polynomial × exponential" cascade that `end_to_end_curve` measures
empirically (persistence.py:17-19).

**Logits.** Both models end in `logits = W_U · RMSNorm(h)`, so `|logit_i| ≤ √d ‖W_U[i] ⊙ (1+γ_f)‖₂` for **any**
state. "Bounded state ⇒ bounded logits" is therefore true but carries no information. The meaningful corollary is

    ‖Δlogit‖_∞ ≤ max_i ‖W_U[i]⊙(1+γ_f)‖ · Lip_RMS(h) · ‖Δh^L‖ ≤ K' (τ+1)^{L−1} ρ^τ,

where τ is the distance from the span. This is the injection-robustness statement we can state. It holds in this
form for a fully recurrent stack with ρ < 1: GDN, or scalar-a RWKV-7. `M_l` and `d_l` include RMSNorm or GroupNorm
local constants `≈ max|γ|/rms(·)`, so the constants are per input, not universal.

## 4. Where it fails

| # | Failure | Effect | In our stack? |
|---|---|---|---|
| F1 | GDN β → 2 (`allow_neg_eigval`, fla-core 0.5.2 `chunk_gated_delta_rule(..., allow_neg_eigval=False)` scales β by 2) | With exact unit k the k-eigenvalue is `1 − β ∈ (−1, 1)`: still ≤ α, but undamped beyond α as β → 2. bf16 k has ‖k‖² up to 1.0019 (numpy P1), so `β‖k‖² > 2` becomes reachable, and then `‖I − βkkᵀ‖ = β‖k‖² − 1 > 1` (β = 2: 1.0032, G1) | No: HF passes `β = σ(b)` without `allow_neg_eigval` (modeling_qwen3_5.py:520, 539-550). Re-check before any switch to the negative-eigenvalue variant or a custom kernel. |
| F2 | Unnormalised k | `‖I − βkkᵀ‖ = max(1, β‖k‖² − 1)`: unbounded | No: `use_qk_l2norm_in_kernel=True`. RWKV-7's κ is normalised. Its write key k̃ is not, but that only enters b (T3). |
| F3 | Precision | (i) bf16 state storage (RWKV `wkv_state_dtype="bfloat16"`) rounds each step at relative 2^{−9}, so contraction is guaranteed only if `ρ(1 + c·u) < 1`. (ii) α rounds to 1.0 in bf16 once `1 − α < 2^{−9}` (α = 0.9985 → 1.0, P1), and in fp32 once `|g| < 6e-8`. HF computes g in fp32 (modeling_qwen3_5.py:522). (iii) The weights-only `ρ_h` can round to exactly 1.0 in float64, so `adapters.universal_decay_bound` would print `certified=False` for a true bound. Use `−log ρ_h` (the probe does). (iv) Chunked kernels (fla WY/UT, `rwkv7_chunked` 1/c rescaling, modeling_rwkv7.py:188-208) match the recurrence only to ~1e-7 fp32. T3 absorbs that when ρ < 1. | RWKV state defaults to fp32 (configuration_rwkv7.py:122). (iii) is plausible for real weights: the √d worst case makes `softplus(lo)` tiny. |
| F4 | **Attention layers in the hybrid** | Softmax attention reads the span's K/V at every later position with weight up to 1, so there is no decay with distance. T4 holds for GDN layers 0–2 up to the input of layer 3. From layer 3 on, every later GDN layer's input carries the span, so T4's hypothesis fails and T5 does not apply. | **Yes:** Qwen3.5-4B layers 3, 7, …, 31. For the hybrid, forgetting gives nothing past layer 3. Injection robustness must come from G15 (structural isolation), not decay. |
| F5 | RWKV-7 per-channel `a` | T2(d, e): no universal contraction, possible geometric growth under switching. The certificate U_t can exceed 1. | Yes, by architecture. The probe measures how often (`step_bound_gt1`, `exact_norm_gt1`) and whether real windows grow. |
| F6 | Output normalisation | GroupNorm `ln_x` (modeling_rwkv7.py:437, 540) and GDN gated RMSNorm (modeling_qwen3_5.py:189-204) normalise the readout. A tiny ΔX at small ‖y‖ can be amplified by `max|γ|/rms(y)`. The state bound transfers to the block output only through this local constant (T5's d_l). | Yes |
| F7 | Local windows and padding | Bounds start w steps after the span (token shift 1, conv 3). GDN zeroes padded hidden states when B > 1 (modeling_qwen3_5.py:207-216). A pad then has `α = exp(−e^{A_log}softplus(dt_bias)) < 1`, so it contracts rather than acting as the identity, and a padded batch is not state-identical to B = 1. RWKV pads use `w_log = 0`, the exact identity (modeling_rwkv7.py:467-483). | The probe runs B = 1 |
| F8 | Hypotheses of T4/T5 | T4 needs a clean input stream to the layer. That is true for layer 0 and for G15's protected stream, false elsewhere. T5's constants are per input. | — |

## 5. Implications for our guarantees

- **Can claim (CERTIFIED, every input).** Each GDN layer contracts with `ρ_h < 1`. Each RWKV-7 head has a real
  spectrum in (−0.455, 1). Each RWKV-7 step satisfies `‖A_t‖ ≤ U_t` (a checkable certificate).
- **Can claim (CERTIFIED per input).** The T3 ISS bound, the T4 single-layer forgetting bound, and the T5 cascade
  bound with constants measured along the trajectory.
- **Cannot claim.**
  - Universal contraction or ISS for RWKV-7 as parameterised. The JSR is above 1.
  - Any distance-decay bound past the first attention layer of the hybrid.
  - "Bounded logits" as a guarantee, since it is vacuous.
- **Would buy the claim for RWKV-7.** Scalar-a heads (T2(c)), or a σ_a penalty during training. This is a constraint
  that suits `CONTROL_RESEARCH.md` §4's PI-Lagrangian controller. It turns RWKV-7 into a CERTIFIED-contractive
  backbone with `ρ = max(w̄, 0.455)`. Its capability cost is unmeasured.

## 6. Empirical check (script written; NOT submitted)

- `safety_eval/state_stability_probe.py` (GPU, inference only). It loads the parent or a trained cell (installs the
  LoRA from a `train.ckpt_export` safetensors) and hooks each recurrent layer's kernel. For RWKV-7 that is a
  `RWKV7_WKV_FUNCTIONS` entry wrapping the active implementation through `WkvName`, so Linears stay batched. For GDN
  it wraps the instance's `chunk_`/`recurrent_gated_delta_rule`. No per-step state is stored (the `adapters.trace`
  recorder would need ~2 GB per layer at T = 2048 on the 13.3B). Per layer it logs:
  - the max decay and 1 − max decay;
  - per-step `‖A_t‖` bounds: exact closed form for GDN, U_t plus exact SVD every 64th step for RWKV-7, with
    `cert_violations` checking U ≥ ‖A‖ on real data;
  - the worst certified window products `∏‖A_s‖` for L ∈ {16, 64, 256} (T4's bound);
  - exact window product norms (real growth if `log > 0`);
  - the final `‖S_T‖` against the per-input ISS bound (`iss_violations`);
  - the weights-only universal bound in log space.

  It exits 3 if any theorem check fails on real data.
- `jobs/state_stability/job_params.json` is a stub with three arms: the D19 cell, the RWKV-7 parent, and the Qwen3.5-4B
  GDN layers. It reuses the `jobs/rwkv7` image and runs on one H100 in about 30 minutes. TODO before submitting:
  - add a render step that writes DEV prompts with each arm's readout template;
  - fill in the D19 export path;
  - add a job.sh.
- `scripts/state_stability_numeric_check.py` holds the CPU receipts for R1–R4, G1 and P1 above.

Untested on GPU. The script compiles, and its algebra mirrors the numpy checks, but nothing has run it against a
model. Expect the usual crash → fix → relaunch on the first run.
