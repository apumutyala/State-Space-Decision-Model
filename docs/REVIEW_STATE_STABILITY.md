# Review: docs/STATE_STABILITY.md @62df6219 (adversarial, EVIDENCE lane)

Date: 2026-10-10. Scope: the lead's state-stability claims T1–T5, the code paths they cite, and the numpy receipts.
CPU and numpy only. No model was loaded.

- Receipts: `python -I scripts/state_stability_numeric_check.py` reproduces every printed number (R1 ‖A‖ = 1.1295;
  R2 minimum slack 7.212e-05; R3 0/5000; R4 1.1046; ‖(A₂A₁)¹⁰⁰‖ = 4.388e8; G1; P1; `ALL PASS`).
- New adversarial checks: `python -I scripts/review_state_stability_check.py` (V1–V8, `ALL PASS`; output below).
- Code audited:
  - transformers 5.14.1 (laptop copy at `Python312/Lib/site-packages`) `models/qwen3_5/modeling_qwen3_5.py`;
  - fla-core 0.5.2 wheel source `fla/ops/gated_delta_rule/{chunk,fused_recurrent}.py`;
  - `transformers/integrations/hub_kernels.py` and `cache_utils.py`;
  - vendored `third_party/rwkv7_hf/modeling_rwkv7.py`;
  - `controlsafe/adapters.py`.

## Verdicts at a glance

| Claim | Verdict | One line |
|---|---|---|
| T1 GDN `‖A_t‖₂ = α_t < 1`, every input | **CONFIRMED** (wording fix) | Every executed path has sigmoid β (not 2·sigmoid) and l2-normalised k. "‖k‖ < 1" is exact-arithmetic only (bf16-stored k has ‖k‖² up to 1.0039), but β‖k‖² < 2 keeps ‖A_t‖ = α_t. |
| T1 universal ρ_h "CERTIFIED for all inputs" | **OVERCLAIMED** (minor) | It holds in exact arithmetic. The bf16 RMSNorm output and the bf16 `in_proj_a` matmul can push the pre-activation below the √d bound by about 2⁻⁸ relative. When `1 − ρ_h` is tiny, that margin matters. |
| T2(a) RWKV-7 spectrum in (−0.455, 1) | **CONFIRMED** in exact arithmetic; **OVERCLAIMED** in floats | The similarity proof is correct. But `rwkv7_recurrent` computes `exp(w_log)` in the activation dtype, so in a bf16 model w rounds to exactly 1.0. Then ρ(A_t) can equal 1. |
| T2(b) per-step certificate | **CONFIRMED** (can be simplified) | Adversarial hill-climbing could not break it (min slack −4e-16, which is rounding). On the RWKV-7 box `max(w_max, ā − w_min) ≡ w_max`, so `U_t = w_max + σ_a`. |
| T2(c) scalar-a contractive "with ρ = max(w̄, 0.455)" | **CONFIRMED**; constant is loose | The constant is ρ = w̄. The 0.455 branch is never active, because w̄ ≥ 0.5452. |
| T2(d) sup ‖A‖ = 2/√3 (takeaway) / ≥ 2/√3 (body) | **CONFIRMED**, and the takeaway's equality is in fact true | Analytic proof below. Hill-climbing over n ≤ 4 also finds exactly 1.154701. The body should say "=", not "at least". |
| T2(e) JSR > 1; 4.4e8× in 200 steps; closes CONTROLSAFE.md:94 negatively | **CONFIRMED** at the parameter-box level; one caveat | Every factor is inside the real ranges: w = 0.99 ∈ (0.5452, 1), a ∈ {0.01, 0.99} ⊂ (0, 1), ‖κ‖ = 1. But growth needs a common decay of about 0.9 or more on the two channels (V4). Realisability on trained weights is open. |
| T3 ISS | **CONFIRMED** | It is a standard unrolling. Per-input use with observed `n_t` is the only non-vacuous form for GDN. |
| T4 single-layer forgetting `∏‖A_s‖` | **CONFIRMED** as stated (clean-input layer only) | Additive terms cancel exactly once the inputs agree. |
| F4 "T4 holds for GDN layers 0–2" | **WRONG** (scope slip) | T4's hypothesis (clean input after t₁) holds only for layer 0. Layers 1–2 see a perturbed input and need T5. |
| T5 cascade / logit corollary | **OVERCLAIMED** (minor, two points) | (i) Its "local Lipschitz along the two trajectories" has to hold on the segment between them. It is not certifiable from endpoint observations alone, and the l2norm Lipschitz constant ∝ 1/‖k_raw‖ (up to 1e3). (ii) As stated (uniform ρ < 1, fully recurrent), it applies to **no model we currently run**: RWKV-7 has per-channel a, and Qwen3.5 is a hybrid. The doc should say so in the Takeaway. |
| "Residual stream carries the span via other paths" (coordinator's worry) | **Not an overclaim** | Residual, norms, gating and the MLP are position-wise, so they move the span in depth, not in time. In a pure recurrent stack, cross-position transport is only through states and the token-shift or conv windows. That is exactly T5's structure. Attention is the only cross-position path, and F4 already concedes it. |
| F1–F3, F6–F8 failure table | **CONFIRMED**, one gap | Add RWKV-7 decay rounding: w → 1.0 in bf16 (`modeling_rwkv7.py:144`). The table lists only GDN α rounding. |

## 1. GDN: the exact transition and every code path (T1)

The transition is `A_t = α_t(I − β_t k_t k_tᵀ)`. Both recurrences show it:
- HF reference loop, `modeling_qwen3_5.py:361-364`: `S ← α S`, `δ = β(v − Sᵀk)`, `S ← S + k δᵀ`;
- fla `fused_recurrent.py:136,157-158`, which has the same order.

So `S_t = α(I − βkkᵀ)S_{t−1} + βk vᵀ`. The `adapters.py:141` transition and the `_gdn_trace_kernel` (`adapters.py:200-203`) match.

| Question | Finding | Evidence |
|---|---|---|
| β = sigmoid or 2·sigmoid? | **sigmoid, β ∈ (0,1)**. HF applies the sigmoid itself and passes neither `use_beta_sigmoid_in_kernel` nor `allow_neg_eigval`. Both default to False, and the ×2 lives only behind them. | `modeling_qwen3_5.py:520, 528-550`; fla `chunk.py:275-276, 285-287`; `fused_recurrent.py:124-127, 197` |
| k l2-normalised on the chunked prefill path? | Yes. `use_qk_l2norm_in_kernel=True` leads to `l2norm_fwd(k)`, computed in fp32 and stored in the input dtype. | `modeling_qwen3_5.py:547`; fla `chunk.py:281-283` |
| k l2-normalised on the decode path (recurrent, seq_len = 1)? | Yes. It is done inside the kernel in fp32: `b_k / sqrt(Σb_k² + 1e-6)`. | `modeling_qwen3_5.py:527-537`; fla `fused_recurrent.py:114-118` |
| Torch fallbacks (no fla)? | Yes, both of them: `l2norm` on q and k before the fp32 cast. | `modeling_qwen3_5.py:242-245, 261-263, 333-335` |
| Any other path? | `@use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")` (`modeling_qwen3_5.py:373`) can replace the whole forward with `Atlas-Inference/gdn` (`hub_kernels.py:126-136`). That happens only on CUDA capability 12.1 (GB10) with `use_kernels=True`. On H100 (SM90) it is not taken. It is unaudited, so re-check before any GB10 or `use_kernels` run. | as cited |
| α < 1 strictly? | `g = −e^{A_log}·softplus(a + dt_bias)` is computed in fp32 (`:522`), so α = exp(g) ≤ 1 always. α = 1.0 exactly happens in fp32 when \|g\| < 2⁻²⁵ ≈ 3e-8 (V7). In the chunked kernel, α enters as the exp of differences of an fp32 **cumsum** of g. Fp32 addition of a non-positive number is monotone, so the differences stay ≤ 0 and α_eff ≤ 1. A tiny g after a large one can be absorbed, giving α_eff = 1 (contraction lost, never growth). This is a sharper version of the doc's F3(ii). | `modeling_qwen3_5.py:290-291`; V7 |
| State precision | fp32 on every path, including the decode cache: the cache layer is `zeros_like` of the fp32 kernel state, then `copy_`. The doc is correct. | `modeling_qwen3_5.py:336-352`; fla `fused_recurrent.py:102-110`; `cache_utils.py:966, 1025` |

**The ‖k‖ wording.** The doc says `‖k‖² = s/(s + 1e-6) < 1`. That holds only in exact arithmetic. The chunked path stores l2-normalised k in bf16, and V6 measures ‖k‖² up to 1.0039 (the doc's own P1 shows 1.0019). Theorem T1's condition `0 ≤ β‖k‖² ≤ 2` still holds with a wide margin (β < 1), so `‖A_t‖₂ = α_t` holds on real data (V6: max ‖I − βkkᵀ‖ = 1.000000000).

*Corrected statement:* "β = σ(·) ∈ (0,1) and ‖k‖² ≤ 1 + O(2⁻⁸) after the in-kernel l2norm (bf16 storage), so β‖k‖² < 2 and ‖A_t‖₂ = α_t ≤ 1. Strict inequality fails only when α rounds to 1.0."

**Universal ρ_h.** The math is right (`adapters.py:286-296`, `in_proj_a` is bias-free per `modeling_qwen3_5.py:438`, RMSNorm `(1+γ)` per `:753`). But "CERTIFIED for all inputs" describes the exact-arithmetic map. The executed map rounds the RMSNorm output to bf16 (`:754`, so ‖x̂‖ can exceed √d by about 2⁻⁹ relative) and computes `in_proj_a` as a bf16 GEMM. The pre-activation can therefore fall below `dt_bias − √d‖W_a⊙(1+γ)‖` by O(2⁻⁸·|a|), and softplus is monotone, so α can exceed ρ_h. Because the doc itself expects `1 − ρ_h` to be tiny (F3(iii)), it should either add an explicit rounding margin (inflate the √d term by `(1 + 2⁻⁷)`) or label ρ_h "exact-arithmetic certified".

## 2. RWKV-7 (T2)

The parameter ranges match the code:
- `w_log = −e^{−1/2} σ(·)` ⇒ w ∈ (0.5452, 1) (`modeling_rwkv7.py:561-563`);
- `a = σ(·)` per channel, C-wide (`:564`);
- `κ = normalize(k ⊙ k_k)` (`:490-491`);
- reference update `S ← S·diag(w) + S(−κ)(κ⊙a)ᵀ + v kᵀ` (`:144-151`), which is the key-first `A = diag(w) − diag(a)κκᵀ` (`adapters.py:138`).

- **(a) Spectrum. CONFIRMED** (the similarity via Λ^{−1/2} and Weyl are both right; R3 reproduces 0/5000). **Float caveat (new):**
  - `rwkv7_recurrent` computes `decay = torch.exp(w_log[:, t])` **in the activation dtype** (`:144`) before casting to the compute dtype. In a bf16 model, any channel with `w_log > −2⁻⁹` (σ(·) < 0.0032) gets w = 1.0 exactly (V7). The top of the spectrum is then 1, not < 1.
  - The chunked path casts first (`:213`), so it is unaffected.
  - The outer products `vkᵀ` and `κ(κ⊙a)ᵀ` are also formed in the activation dtype (`:146-147`). That is a small perturbation of A, which T3's additive-error clause absorbs only when ρ < 1.
  - Add this to F3.
- **(b) Certificate. CONFIRMED.** V1 ran 300 adversarial hill-climbs (n = 2..8, edges of w and a included). The minimum `U − ‖A‖` was −4.4e-16, which is float noise. That is expected, because the bound is Weyl plus a rank-one triangle inequality and is a theorem.
  - **Simplification:** `ā ≤ 1` and `w_min + w_max ≥ 2·0.5452 = 1.09`, so `ā − w_min < w_max` always (V2: max of `ā − w_min − w_max` = −0.24). The certificate is therefore **`U_t = w_max + σ_a`**, and c = ā is genuinely optimal (V2: a grid over c never improves it).
- **(c) Scalar a. CONFIRMED** (V5: worst 40-step product rate 0.754 under random switching). By the simplification, **ρ = w̄** (the decay ceiling); the 0.455 branch is never active. *Corrected:* "scalar-a heads are contractive for every input with ρ = w̄ (exact arithmetic; w̄ must stay < 1 after bf16 rounding of `exp(w_log)`)."
- **(d) Sup norm.** **2/√3 exactly** over the closure. With w = 1, `A = I − xκᵀ` with x = Λκ, and the non-unit singular values satisfy `σ₁²σ₂² = (1−p)²` and `σ₁² + σ₂² = 2 − 2p + q`, where p = Σaκ² and q = Σa²κ² ≤ p. σ_max grows with q, so q = p, giving `σ_max² = (2 − p + √(p(4−3p)))/2`. That is maximised at p = 1/3, with value 4/3. Hill-climbing over w ∈ [0.5452, 1] and n ≤ 4 finds nothing larger (V3: 1.154701). The doc's takeaway equality is right; the body's "at least" can be tightened.
- **(e) JSR > 1. CONFIRMED.** The pair uses w = 0.99, a = (.99,.01)/(.01,.99) and unit κ, so every factor is inside the real ranges. R4 gives ρ(A₂A₁)^{1/2} = 1.1046 and ‖(A₂A₁)¹⁰⁰‖ = 4.388e8, while every ρ(A_t) = 0.99. The consequence "no input-independent norm makes every admissible step non-expansive" is correct, because the products would otherwise be bounded. **Caveat to add:** this construction needs a common decay of about 0.9 or more on both channels (V4: the rate is 1.016 at w = 0.90 and 0.967 at w = 0.85; with a restricted to [.05, .95] it is 1.002 at 0.90). Whether trained D19 or parent weights can realise it depends on the per-channel ceilings w̄ and on a swinging across channels token to token. That is exactly what the probe's `exact_window_logprod` measures. CONTROLSAFE.md:94 can be closed negatively *for the parameter box*, not for the trained model.

## 3. Influence and forgetting (T3–T5, F4)

- **T4 is correct.** After `t₁ + w`, the clean-input layer has identical `A_s, B_s`, so `ΔX_s = A_s ΔX_{s−1}` holds exactly. The additive input terms cancel. Normalisation, gating and the MLP at that layer act position-wise on `Δy_t = ΔX_tᵀ r_t`. Gated RMSNorm can amplify by `max|γ|/rms` (the doc's F6), but cannot bring back a decayed state.
- **F4 slip (WRONG).** "T4 holds for GDN layers 0–2 up to the input of layer 3." Layer 1's input is layer 0's output, which carries the span through layer 0's state at every later position. So T4's hypothesis holds only for **layer 0**. Layers 1–2 are covered by T5 (poly·ρ^τ), not T4. *Corrected:* "T4 holds for layer 0; T5 covers GDN layers 1–2; nothing decays past layer 3."
- **T5.**
  - The induction and the `(τ+1)^{l−1}ρ^τ` shape are right.
  - Two qualifications. First, `Lip_x(A)` and `Lip_x(B)` must bound the difference over the segment between the two trajectories (a mean-value argument), not at the two points. The l2norm in k has a Lipschitz constant of about `1/‖k_raw‖`, capped only by eps 1e-6 at about 1e3. So the "per-input constants" are not observable from the probe's endpoint quantities, and T5 is a structural result, not a certificate.
  - Second, with a uniform ρ < 1, T5 applies today to none of our deployed models: RWKV-7 has per-channel a and no ρ < 1 (T2(e)), and Qwen3.5 is a hybrid (F4). It applies only to Qwen3.5 GDN layers 0–2 below layer 3 and to a hypothetical scalar-a RWKV-7. A per-input version with observed `n_s` in place of ρ is the usable form. Say so in Takeaway 4.
- **The coordinator's worry** (per-layer contraction ≠ whole-network decay) **does not land on the doc**. The residual stream is position-wise, so in a pure recurrent stack it carries the span only in depth. The doc already makes the one real cross-position exception, attention (F4), its headline caveat. The V8 toy illustrates it: the span effect after a contractive layer is 1e-13 at τ = 300, but after one uniform-attention layer it is 0.32, decaying only like 1/t.
- **Logits.** The vacuity remark is correct (final RMSNorm). The Lipschitz corollary inherits T5's qualifications.

## 4. Precision summary (coordinator's item 4)

| | Status |
|---|---|
| GDN state | fp32 on every path, including the decode cache. No bf16 accumulation of the state. |
| GDN α | α ≤ 1 always in floats. α = 1.0 only for \|g\| < 3e-8 or by cumsum absorption in the chunked kernel. That loses contraction but never causes growth. |
| GDN intra-chunk products | bf16 inputs with fp32 accumulation. This is additive error, bounded by `e/(1−ρ)` only with per-input ρ < 1. With the universal ρ_h ≈ 1 the bound is vacuous. |
| RWKV-7 state | fp32 by default (configurable bf16, the doc's F3(i)). |
| RWKV-7 decay | **New:** `exp(w_log)` is in the activation dtype on the recurrent path (`:144`), so w = 1.0 in bf16 for σ(·) < 0.0032. |

## 5. Bookkeeping

- The doc's Qwen3.5 line numbers (520, 522, 536, 547, 242-245, 354-364, 746-753) match transformers 5.14.1 exactly.
- As the doc says, re-check them against 5.17.0 (the image pin) before quoting elsewhere.

## Receipt (scripts/review_state_stability_check.py)

```
PASS V1: adversarial min(U - ||A||_2) over 300 hill-climbs x 250 steps = -4.441e-16
PASS V2: max(abar - w_min - w_max) over 20k draws = -0.2447 < 0, so max(w_max, abar - w_min) == w_max always
PASS V3: hill-climbed sup ||A||_2 over closure of the box (n<=4) = 1.154701; 2/sqrt3 = 1.154701
INFO V4: rho(A2A1)^(1/2) by common decay w  [a .01/.99] [a .05/.95]
      w=0.99 1.1046 1.0904 | 0.95 1.0653 1.0511 | 0.90 1.0162 1.0021 | 0.85 0.9673 0.9532 | 0.80 0.9185 | 0.5452 0.6745
PASS V5: scalar-a: worst ||A_m...A_1||^(1/m) over 400 random 40-step products = 0.7544 < 1
PASS V6: bf16-stored l2norm k: max ||k||^2 = 1.003907, max ||I - beta k k^T|| = 1.000000000
INFO V7: fp32 exp(-3e-8) == 1.0; bf16 exp(-0.0019) = 1.0, bf16 exp(-0.002) = 0.99609375
INFO V8: toy stack, span 10 tokens: layer-1 |dy| at tau=300: 1.10e-13; after uniform attention: 3.22e-01
ALL PASS
```
