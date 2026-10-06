# controlsafe: what the dynamical-system view buys, stated exactly

Both parent families are the same recurrence per layer and head: `X_t = A_t X_{t-1} + k_t v_t^T`, `y_t = X_t^T r_t` (`adapters.py`).
- RWKV-7: `A_t = diag(w_t) - (kk_t*a_t) kk_t^T`.
- GDN / Q9-AF: `A_t = alpha_t (I - beta_t k_t k_t^T)`.

All of this is CPU-tested on tiny random RWKV-7 and all-GDN configs. **Nothing here is a model result.** SAFETY_SPEC §5.2 still holds: no decay, contraction or Gramian number is a safety claim. Two STRUCTURAL rows are proposed:
- G15, trust-gated reachability.
- G16, protected judge with a bounded evidence channel. Its veto and channel-isolation parts are STRUCTURAL; its radius and independence parts are CERTIFIED.

Review: PR #42 (comment 6020811487). B1 and B2 and the non-blocking items are addressed below.

## Trust labels (precondition of G15 and G16)

`trust_gate.row_trust(row, policy)`, `continuation_trust` and `option_seq_trust` label every token: the prefix, each option continuation and each stand-alone option sequence.

| policy | untrusted tokens | when it is sound |
|---|---|---|
| `provenance` (default) | Every caller segment whose request field the host did not declare instruction-trusted. A declaration lives in `request["ssdm"]["provenance"]` (`decide.provenance` shape), must tile the whole field, and every segment must be `instruction`. Question and option ids are always data, and so is `state`. | Always (fail closed). P holds template text plus the explicitly declared host instructions. |
| `state` | The rendered state only | Only if the host puts **all** untrusted text in `state`. Fails closed (`TrustUnavailable`) on rows whose evidence moved to the instructions (`meta["evidence"] == "reread_instructions"`, m2 `placement: instructions`, ~21 % of the m2 sample). |
| `caller` | Every non-template token | Always. P then holds template text only. |

`train.run_cell.to_train_record` declares a question's instructions trusted only when the builder placed the item in `state` and the instruction is exactly one of the builder's own templates (`data.augment.INSTR`). Everything else stays data, including the passage under `placement: instructions`. Real m2 rows built by `data.augment.render`, `to_train_record` and `build_row` are tested both ways (`tests/controlsafe/test_trust_policy.py`).

Further fail-closed behaviour:
- `TrustGatedBackbone.features(trust=None)` raises.
- `fork_features` requires labels for the prefix and for every continuation.
- Gradient checkpointing is refused with a clear error.

## G15: trust-gated reachability (STRUCTURAL, `trust_gate.py`)

**Statement.** Take two inputs with the same trusted token subsequence. Their untrusted parts may differ in content, length, count and placement, and so may a batch-mate's untrusted content and length. Then every protected-block feature at the i-th trusted token, at every layer, is identical between the two inputs, and so is the protected recurrent state.
- The two are bitwise identical for an identical host-controlled batch shape: B, and the longest trusted subsequence in the batch.
- They are independent of untrusted content always.
- Any readout of protected features only is therefore invariant to untrusted text. This holds for any weights.

**Construction.** P is the pretrained recurrence over trusted tokens only: identity transition and zero write at untrusted positions, with the shift and conv skipping them. It is implemented by compaction plus carry-forward. D is the full model plus zero-initialised P→D bridges. Block lower-triangular coupling means P never reads D. At init the output equals the pretrained model bitwise.

**Scope.**
- G15 is an **inference-time** statement. LoRA is shared by both streams and trained on losses that read untrusted training text, so P's weights do depend on training data; training-data poisoning belongs to G6/G12.
- The default decision readout reads D, so G15 alone does not certify the decision (see G16).
- Cost: `1 + |trusted|/T`.

## G16: protected judge, bounded evidence channel, trusted veto (`g16.py`)

`logit_k = b_k + a_k · E_k` for allowed k, and `-inf` otherwise, where:
- `a_k = A h_P[k] + u_k` and `b_k = w·h_P[k] + β_k` come from P features only, plus option positions.
- `E_k = c·tanh(W_e h_D[k]) ∈ (-c,c)^r`, with `W_e` zero-initialised.
- The mask is the host allowed set (`policy.allowed`), ANDed with an optional P-only gate.

The guarantees below assume the trust labels above and treat the schema (option set, canonical order, allowed set) as host structure.

| | Statement | Tier |
|---|---|---|
| (a) veto | No untrusted text can make a vetoed option the decision | STRUCTURAL |
| (b) isolation | `a`, `b` and the mask are bitwise invariant to untrusted text (G15). The decision is a fixed function of the bounded E. | STRUCTURAL |
| (c) radius | With `μ_ij = logit_i − logit_j`, no `‖E'−E‖₂ < ρ₂ = min_j μ_ij/√(‖a_i‖²+‖a_j‖²)` (joint) flips the decision. Neither does any per-option ball `max_k‖E'_k−E_k‖₂ < ρ_opt = min_j μ_ij/(‖a_i‖+‖a_j‖)`. Both radii are tight. | CERTIFIED per decision |
| (d) independence | If `b_i − b_j > c(‖a_i‖₁+‖a_j‖₁)` for every allowed `j ≠ i`, then the decision is identical for every E in the box, hence for every untrusted text. G4 then holds at zero recompute. | CERTIFIED per decision |
| (e) G4 over E | Otherwise erase-and-check recomputes only E for each origin. P is reused exactly: origins share the trusted subsequence, which is checked token by token. Each origin's decision equals a full G16 recompute. | CERTIFIED (exact G4) |

The review proposed a radius of `μ/L` with `L = max‖a_k‖`. It is **not sound**: it overstates ρ₂ by up to √2 and ρ_opt by up to 2. The counterexample is in `test_g16.py`. `μ/(√2 L) ≤ ρ₂` is a sound simplification.

**Scope.** Untrusted text can still steer the decision within E's box. That steering is bounded, logged (r·K numbers per decision), and either certified or escalated.

**Training arm.** `head: "g16"` in `TrainConfig` / `run_cell` (needs `controlsafe.trust_gate`, readout `decision`). `DecisionModel` routes rows to `G16DecisionHead.forward_rows`. The judge, evidence channel, LoRA and P→D bridges train with the usual losses; the decision is `softmax(logit)` for every question type. `head_kwargs`: `r`, `c` (evidence bound, default 1.0), `stream_mode` (`cache` by default). `stream_mode: cache` uses `TrustGatedBackbone.fork_streams`: one D prefix pass plus K state copies, with P run once per distinct trusted subsequence. It equals recompute up to float rounding, and P is bitwise identical. Tests: `tests/controlsafe/test_g16_train.py`.

**Training-time scope.** (a)–(e) hold for the activations of any fixed weights, before and after training (tested). During training, the losses read untrusted training text through E. The gradients that reach the judge (A, w, u, β), the shared LoRA and the P→D bridges therefore depend on untrusted training tokens. The judge's and P's functions are shaped by untrusted training data; that is G6/G12 territory, as for G15.

- **c ablation:** `python -m train.g16_card sweep`.
- **Eval card:** `card["g16"]` holds served certificate stats. `run_cell` runs `certify` on the safety_eval DEV requests once, after training, into `cell["train"]["g16_safety_eval"]` with its wall-clock. It stays outside `max_minutes`, so the pair keeps step parity.
- **Decide backend:**
  - The veto source is only the validated envelope field `ssdm.policy_restriction`. A top-level `request["policy"]` is ignored; review B1 showed it was read before.
  - The backend returns the unvetoed logits `b + a·E`. `raw` and `served` are therefore the model's distribution over every option, and `policy_masking: none` is true.
  - The adapter emits the restriction as `policy_conditioned_probabilities` (stage `present`).
  - For G16, that stage is the vetoed decision by construction. `softmax` of `b + a·E` restricted to the allowed set equals the served distribution conditioned on it: exactly at T = 1, and in argmax under a served temperature. G2(d) holds: served is never replaced.
  - The certificate is computed with the veto. Its fields go to `diagnostics.backend["g16"]`: `vetoed`, `veto_source`, `decision_stage`, `unvetoed_logits`, radii and independence. The full G4-over-E record is added with `g16_certify_k`.
- **Capability cost:** `train.g16_card compare` on the paired cells `jobs/m2_pilot/m2_pilot_cell_g16{,_base}.json`. Compare at matched steps only.
- **Memory:** for rwkv7-13.3b gated arms, the estimate at rank 64 and T = 2048 is about 70–72 GB against 76 GB usable, which is tight. The ladder may give the two arms different `T_max`. That fails closed (`identical_items: False`); rerun both at the smaller T.

**Not done yet:**
- Snapshot reuse of the D stream inside G4-over-E.
- A measured capability cost (the pair is configured but not run). `m2_job.sh` does not select the pair cells yet.

**Arm result.** On 10 dev requests, with a random judge and every origin verified: k=1 evaluates 36 origins and k=2 evaluates 64. The P stream is computed once per request with 0 cache misses. The independence rate is 0 with random weights, so it says nothing about trained heads.

## Per-layer decay (`persistence.py`)

**GDN — CERTIFIED for every input.**
- `‖A_t‖₂ = α_t ≤ ρ_h = exp(−e^{A_log_h} softplus(dt_bias_h − √d ‖W_a[h]⊙(1+γ)‖₂)) < 1`.
- This is a common induced norm, so `‖A_T…A_1‖₂ ≤ ρ^T`. The constant is the tight one the review suggested.

**RWKV-7 — diagonal decay bound only.**
- There is no common 2-norm bound below 1; the counterexample is in the tests.
- This refutes the common 2-norm, not contraction. A weighted norm or joint spectral radius remains open.
- Full-transition sups are observed-only (EMPIRICAL).

**Both families — CERTIFIED per input.**
- `‖dX_{t+τ}‖ ≤ ∏‖A_s‖ ‖dX_{t+w}‖`.

**End-to-end curves: EMPIRICAL.**

## Untrusted-channel Gramians (`gramian.py`, EMPIRICAL)

- `W_l = Σ_{t∈U} J_t J_t^T`, plus effective rank and readout overlap. Checked against finite differences.
- **Scalability:** this is one autograd call per state coordinate and a dense S×S matrix. That is infeasible at Q9-AF scale (S ≈ 5e5 per layer); use `J^T J` or randomised trace estimates before any 9B run.

## G4 snapshot reuse (`snapshot.py`)

- Exact prefix-trie reuse over the real covered origins and tokenisation.
- Decisions are identical to the C(n,k) recompute (max |Δlogit| ≤ 1.4e-6).
- On 10 dev requests:

| k | full recompute (forward-equivalents) | trie | tokens saved | backbone calls |
|---|---|---|---|---|
| 1 | 4.49 | 2.04 | 55 % | 74 vs 46 (1.6×) |
| 2 | 7.10 | 2.84 | 60 % | 122 vs 74 (1.65×) |

- GPU wall-time savings will be smaller than token savings, because there are more, shorter calls.
- With option continuations, the baseline now counts continuation tokens too, so savings and the measured baseline agree.
- Scope: heads that read the final feature, plus option forks.

## G14: hidden-state poisoning monitor (`monitor.py`): EMPIRICAL, synthetic, detection WEAK and UNVALIDATED

The previously quoted "AUROC 0.96 / 0.91" was seed 0, the best of the seeds tried, and is withdrawn. Ten joint seeds (model and trigger, `safety_eval.controlsafe_arms --arms g14`):

| family | AUROC per seed | mean [95 % bootstrap CI over seeds] | median | min | seeds ≤ chance | AUROC (min-head score) mean | collapse ratio median |
|---|---|---|---|---|---|---|---|
| RWKV-7 | .96 .37 .87 .38 .26 .39 .52 .91 .55 .49 | 0.57 [0.43, 0.72] | 0.49 | 0.26 | 5/10 | 0.50 | 0.76 |
| GDN | .91 .79 .56 .81 .81 .95 1.00 .99 .98 .85 | 0.86 [0.78, 0.93] | 0.85 | 0.56 | 0/10 | 0.64 | 0.27 |

**In SAFETY_SPEC terms.**
- G14's alarm target (AUROC ≥ 0.9) is **not met on either family**.
- On RWKV-7 the retained-norm detector is **no better than chance**.
- G14 detection remains an unvalidated EMPIRICAL instrument.
- The trigger objective (functional collapse) and the detector both measure decay, so even good AUROCs would be a pipeline check, not detection power.
- Not evaluated: an adaptive attacker who also minimises the score, and trained parents.

**What is solid.** G15's P block is bitwise unchanged under the trigger (tested). Protected context cannot collapse. The pinned-seed AUROC thresholds were removed from the tests; a seed-sweep smoke test replaces them.

## Mapping to SAFETY_SPEC (proposed rows)

| Row | What controlsafe adds | Tier |
|---|---|---|
| G2 | G15 and G16 carry host-declared provenance (G2(c)) into the model as non-learned gates. The G16 veto enforces G2(a)'s host allowed set (on the decide path: the validated `ssdm.policy_restriction` only) inside the decision function. That decision is the adapter's policy-conditioned stage, and served stays unvetoed (G2(d)). | STRUCTURAL |
| G4 | Snapshot reuse (cost), G16 independence (zero recompute) and G4-over-E | CERTIFIED (unchanged), cost EMPIRICAL |
| G9 | Influence curves and Gramians: EMPIRICAL (b). Exact zero-dependence of P and of `a`, `b`: STRUCTURAL. | mixed |
| G14 | Telemetry plus synthetic test (weak, unvalidated). P immune by construction. | EMPIRICAL / STRUCTURAL (P only) |
| **G15** | Trust-gated reachability | STRUCTURAL (inference-time) |
| **G16** | Veto plus isolation; radius plus independence plus G4-over-E | STRUCTURAL / CERTIFIED |

## Flags and entry points

- **Training:** `TrainConfig.controlsafe = {"trust_gate": true, "bridge_rank": 8, "readout": ..., "trust_policy": "provenance"}`. An empty dict leaves the config hashes unchanged.
- **Evaluation:**
  - `python -m safety_eval.controlsafe_arms --out DIR [--arms g15,persistence,gramian,g4snap,g14,g16]`
  - `run_guarantees_smoke --controlsafe all`
- **Receipt:** `receipts/2026-10-06-D-controlsafe-cpu/controlsafe.json`.
