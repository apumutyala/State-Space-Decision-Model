# controlsafe: what the dynamical-system view buys, stated exactly

Both parent families are, per layer and head, `X_t = A_t X_{t-1} + k_t v_t^T`, `y_t = X_t^T r_t`
(`adapters.py`; RWKV-7: `A_t = diag(w_t) - (kk_t*a_t) kk_t^T`; GDN/Q9-AF: `A_t = alpha_t (I - beta_t k_t k_t^T)`).
Everything here is CPU-tested on tiny random RWKV-7 and all-GDN configs only. Nothing is a model result.
SAFETY_SPEC section 5.2 still holds: no decay, contraction or Gramian number is a safety claim. The only
STRUCTURAL row added is G15.

## G15 (new): trust-gated reachability (STRUCTURAL, `trust_gate.py`)

**Statement.** Take two inputs with the same trusted token subsequence. Their untrusted spans may differ in content, length, count and placement. Every protected-block feature at the i-th trusted token is then bitwise identical in both, at every layer, and so is the protected recurrent state. Any readout that is a function of protected features alone is therefore invariant to untrusted text. This holds for any weights, including LoRA and trained bridges.

**Construction.** The protected block P is the pretrained recurrence run over trusted tokens only. At an untrusted position P uses the identity transition and a zero write, and the token shift and causal conv skip that position. This is implemented exactly by compaction plus carry-forward. The data block D is the pretrained model over all tokens, plus zero-initialised low-rank P→D bridges. The joint transition is block lower-triangular, `[[A_PP,0],[A_DP,A_DD]]`, and the untrusted write is `[0;B_D]`. Across layers, P reads only P. The code-path argument is that `protected_stream` receives only the ids at trusted positions.

**Tests.** `tests/controlsafe/test_trust_gate.py` checks four things:
- Bitwise invariance under random untrusted content and lengths, leading spans and interleaved spans, with random bridges and random LoRA.
- The P-stream Jacobian with respect to the untrusted embeddings is exactly 0, while the D-stream Gramian is greater than 0.
- At init the result equals the pretrained model bitwise.
- The guarantee survives a training run of the gated arm with DecisionModel, build_head_input and Trainer.

**Scope and caveats.**
- The guarantee covers P only. The default `readout="decision"` reads D, which untrusted text can reach. It has to: the evidence is untrusted (reviews/02 §6). G15 does not certify the decision.
- What G15 does guarantee is that trusted context (schema, question, policy framing, trusted records) cannot be overwritten, steered or collapsed by untrusted text.
- Trust labels come from the host: `meta["untrusted_spans"]` (state) or `caller_spans` (all non-template text) on layout rows. Rows that cannot state their spans fail closed.
- Cost: `compute_multiplier = 1 + |trusted|/T` (1.5 in the arm). The cache fork is replaced by recompute.
- Bitwise identity assumes the same kernels and batch composition.
- Masking mixing in place (a head partition) is not implemented, because it would destroy the init equivalence.

## Per-layer decay (`persistence.py`)

- **CERTIFIED for every input, GDN only.**
  - Statement: `||A_t||_2 = alpha_t <= rho_h = exp(-exp(A_log_h) softplus(dt_bias_h - ||W_a[h]|| sqrt(d) max|1+gamma|)) < 1`.
  - This is a common induced norm, so `||A_T...A_1||_2 <= rho^T`.
  - Tested against random inputs and embeddings with norms up to 1000×.
  - The bound is real but loose: it is close to 1 whenever `||W_a||` is large.
- **RWKV-7.**
  - Only the diagonal decay part has a universal bound.
  - The full transition has no universal 2-norm bound below 1; an explicit counterexample is in the tests.
  - So the full-transition sup is reported over observed inputs only (EMPIRICAL).
- **CERTIFIED per input, both families.**
  - Perturb layer l's mixer input at t. For `tau > w` (w = 1 for the RWKV-7 shift, `conv_kernel-1` for GDN), `||dX_{t+tau}|| <= prod ||A_s|| ||dX_{t+w}|| <= rho_obs^(tau-w) ||dX_{t+w}||`.
  - This holds exactly for that sequence, apart from float rounding, and is checked in the tests.
- **EMPIRICAL.**
  - End-to-end curves: perturb or swap an embedding and measure every layer's hidden state, state and final feature.
  - These include the cross-layer cascade, for which no bound is claimed.
  - Observed-sup norms in the 2, 1 and ∞ norms.

## Untrusted-channel controllability (`gramian.py`, EMPIRICAL)

- `W_l = Σ_{t∈U} J_t J_t^T`, taken with respect to untrusted embeddings (end-to-end) or with respect to layer l's own mixer input (local).
- Reported: effective rank (entropy and participation ratio), numerical rank, and the overlap `tr(W P_R)/tr(W)` with the readable directions `span(r_T)⊗R^Nv`. The isotropic reference is `1/Nk`.
- The Jacobian is checked against finite differences.
- This is a first-order statement at one input, not a reachable-set certificate.

## G4: snapshot erase-and-check (`snapshot.py`)

- Builds the token trie of the observed request and all of its covered origins (`certify.origin_requests`, the same dedup, the real `build_row` tokenisation) and runs each shared prefix once.
- States are cloned only at branch points, so k=1 holds O(n) states.
- It needs no assumption that the tokenizer composes across segments.
- Decisions were verified identical to the C(n,k) full recompute on every origin (max |Δlogit| ≤ 1.4e-6).
- The certificate is unchanged. Only its cost changes.
- Scope: heads that read the final feature, plus option forks. Integrator heads would also need cached prefix features.

Measured on 10 dev-split generator requests (`receipts/2026-10-06-D-controlsafe-cpu/`), using backbone tokens and the same savings for both families:

| k | full-recompute forward-equivalents / request | snapshot forward-equivalents / request | tokens saved |
|---|---|---|---|
| 1 | 4.49 | 2.04 | 55% |
| 2 | 7.10 | 2.84 | 60% |

Backbone call count rises, because there are more and shorter runs.

## G14: hidden-state poisoning monitor (`monitor.py`, EMPIRICAL, synthetic)

**Statistics.** For each layer, over the untrusted span:
- retained fraction `||Φ X_{a-1}||/||X_{a-1}||`
- `σ_max(Φ)` and log-volume
- norm growth

**Detector.** `-mean_l log retained`.

**Synthetic test.**
- Uses a long-memory tiny model with input-dependent decay.
- The trigger is found by greedy coordinates against a functional objective (making two trusted prefixes indistinguishable), not against the detector.
- Result: AUROC 0.96 (RWKV-7) and 0.91 (GDN), with collapse ratio 0.59 and 0.26.

**Not evaluated.** An adaptive attacker who also minimises the detector score. Trained parents.

**Structural link.** The G15 P block is bitwise unchanged under the trigger (tested). Memory collapse cannot touch protected context.

## Mapping to SAFETY_SPEC

| Row | What controlsafe adds | Tier |
|---|---|---|
| G2 | G15 carries host-declared provenance (G2(c)) into the model as an out-of-band, non-learned gate. ISE/ASIDE remain the learned, empirical channel | STRUCTURAL (P block only) |
| G4 | Exact prefix-state reuse. Same certificate, measured cost | CERTIFIED (unchanged); cost EMPIRICAL |
| G9 | Perturb-and-recompute influence curves and Gramians locate untrusted influence. G15 gives an exact zero-dependence statement for P features | (b) EMPIRICAL; zero-dependence STRUCTURAL |
| G14 | Spectral and retained-norm telemetry plus a synthetic poisoning test. P block immune by construction | EMPIRICAL; immunity STRUCTURAL (P only) |
| **G15** | Trust-gated reachability as stated above | STRUCTURAL |

## Arms and flags

- **Training:** `TrainConfig.controlsafe = {"trust_gate": true, "bridge_rank": 8, "readout": "decision"|"concat"|"protected", "trust_policy": "state"|"caller"}`. In `run_cell`, the cell config uses the key `controlsafe`.
  - The bridges train with the LoRA group.
  - Lineage records the arm.
  - An empty dict leaves the config hashes unchanged.
- **Safety evaluation:**
  - `python -m safety_eval.controlsafe_arms --out DIR [--arms g15,persistence,gramian,g4snap,g14]`
  - `python -m safety_eval.run_guarantees_smoke --out DIR --controlsafe all`
- **Capability cost:** `trust_gate.capability_cost(base_fn, gated_fn, items, metric)` (paired deltas, 0 at init) and `compute_multiplier`.
