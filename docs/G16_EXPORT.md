# G16-sym export: per-option trusted-only intervals for injection-robust sets

Code: `controlsafe/g16_sym.py` (head, export, matched calibration, losses), `controlsafe/g16_sym_plug.py` (tiny CPU
head for the invariance harness). Tests: `tests/controlsafe/test_g16_sym.py`. Theory: `docs/guarantees/REORDER_INVARIANCE.md`,
`docs/guarantees/INJECTION_ROBUST_CONFORMAL.md` (evidence lane, f7170cb).

## Takeaway

- `G16SymHead` is G16 without the per-slot parameters `u_k`, `beta_k` and without first-index `argmax`. One shared
  judge scores every option from its own sym-fork continuation: `a_k = A h_P[k]`, `b_k = w . h_P[k]`,
  `score_k = b_k + a_k . E_k`, `E_k = c tanh(W_e h_D[k])`.
- Options are separated by **trusted option content** instead of slots: a description is in the protected stream iff
  the host declared `/questions/<qid>/criteria/<id>` instruction-trusted (or the caller passes
  `option_trust="host_schema"`). Undeclared descriptions are data; then every option gets the same interval and the
  robust set is everything (fail closed, tested).
- Per decision it exports, per option, `lo`, `hi` (trusted-only, bitwise invariant to untrusted text) and the point
  `score`. Ties are reported as a set; a single representative is picked by content hash, never by position or id.
- Composition: reorder + relabel invariance (bitwise, harness tol = 0) and G15 on `[lo, hi]` hold at the same time,
  so the matched robust set is invariant to reorder, relabel AND any untrusted text.
- **The injection-robust action is `act`**, the singleton of the trusted-only matched robust set (from `[lo, hi]`
  alone; `null` = escalate). `decision_set` / `decision` are the argmax of `score = b + a·E`, and E reads untrusted
  text, so they are NOT G15-invariant (only reorder/relabel-invariant). Only `lo`, `hi`, the veto and `act` are
  G15-invariant.

## Record schema (`ssdm.g16sym.export.v1`, one JSON object per line)

```json
{"schema": "ssdm.g16sym.export.v1",
 "decision_id": "req123/q",
 "options": [{"id": "refund", "lo": -1.73, "hi": 0.41, "score": -0.22, "content_sha256": "9f2c..."},
             {"id": "escalate", "lo": 0.05, "hi": 1.90, "score": 1.31, "content_sha256": "11ab..."}],
 "vetoed": ["close"],
 "decision_set": ["escalate"],
 "decision": "escalate",
 "gold": "escalate",
 "act": "escalate",
 "act_rule": {"qhat": 0.83, "kappa": null}}
```

| Field | Required | Meaning |
|---|---|---|
| `schema` | yes | `ssdm.g16sym.export.v1` |
| `decision_id` | yes | string; unique per (request, question) |
| `options[]` | yes | the host-ALLOWED options only. Order carries no meaning (consumers must key by `id`). |
| `options[].id` | yes | wire option id (mapped back outside the model; never read by it) |
| `options[].lo`, `.hi` | yes | finite floats, `lo <= score <= hi`. `lo = b - c‖a‖₁`, `hi = b + c‖a‖₁`, computed in float64 and rounded outward by `(r+4)·2⁻²⁴·(|b| + c‖a‖₁)` plus one ulp, so the float32 score always lies inside (assumption A5) |
| `options[].score` | yes | the point logit `b + a.E` (reads untrusted text; never used for a guaranteed action) |
| `options[].content_sha256` | no | SHA-256 of the option's continuation token ids (the tie-break key) |
| `vetoed` | yes | host-vetoed ids (keyed by id, they travel with the option); they have no interval and can never be decided |
| `decision_set` | no | ids whose score equals the max (exact float equality). Reads E, hence untrusted text: NOT G15-invariant, never a guaranteed action |
| `decision` | no | content-hash representative of `decision_set`, or `null` when duplicate contents tie (same caveat) |
| `gold` | calibration rows | gold option id. A vetoed gold is legal: it scores `s_lo = +inf` in calibration (an honest miss) |
| `act` | yes (may be `null`) | THE injection-robust action: `act(rec, qhat, kappa)`, the singleton of the matched robust set built from `[lo, hi]` only, else `null` (escalate). `null` when exported without a calibrated `qhat`. G15-invariant |
| `act_rule` | with `act` | `{"qhat", "kappa"}` used for `act` (`null` without a qhat). `validate_record` recomputes `act` from `[lo, hi]` and refuses a record whose `act` differs |

`validate_record(rec)` checks the schema and returns `{id: [lo, hi]}`, exactly the `intervals` argument of
`jevkit.guarantees.robust_set` / `robust_min_score`. `write_jsonl` / `read_jsonl` validate every row and refuse NaN.

## How jevkit's matched mode consumes it

1. Calibration file (records with `gold`): `q̂ = conformal_qhat([robust_min_score(iv_i, gold_i)], alpha)`
   (`controlsafe.g16_sym.matched_qhat`). Needs only `[lo, hi]` and `gold`, no clean logits.
2. Test file: `C(x) = robust_set(iv, q̂)` (`g16_sym.robust_set`). Coverage ≥ 1 − α for every attacker (Theorem A);
   exact (≤ 1 − α + 1/(n+1)) for continuous scores.
3. Act only on the trusted-only singleton: `act(rec, q̂, kappa=None)` returns the element of a singleton set (optionally
   requiring its worst-case box probability ≥ kappa), else `None` = escalate. Never act on `decision`/`score`.
   `export_record(..., qhat=q̂, kappa=...)` writes exactly that as the record's `act` field.

The JSON keys line up with the recommended `calibration="trusted_lo"` mode in INJECTION_ROBUST_CONFORMAL.md §6
(`--cal-intervals` = a calibration JSONL of these records). jevkit is not edited here.

## Guarantees and their assumptions

| Property | Holds because | Assumptions | Test |
|---|---|---|---|
| Reorder + relabel invariance, bitwise | (C1) prefix has no option list; (C2) one shared judge, no slot index, no id in the continuation; (C3) set softmax with `fsum`; (C4) argmax set + content hash. Row axis is content-canonical, and the batch is ordered by trusted token lists | non-empty descriptions (`row.meta["id_as_content"]` lists exceptions); `sym-fork` layout (`sym-fork-ids` is the non-relabel ablation) | `test_harness_reorder_and_relabel_bitwise` (evidence harness, tol 0), `test_lo_hi_score_travel_with_the_option_bitwise`, structural axis permutation + slot-bias teeth |
| G15 on `[lo, hi]` | `a`, `b` read `h_P` only; `h_P` is a function of the trusted token lists; batch order is too | trust labels correct (A1), G15 kernels (A2); under BPE, sym-fork v3 trust-segment tokenization (no token straddles a trust boundary; `tests/test_sym_g15_bpe.py`) | `test_untrusted_text_cannot_move_lo_hi` (recompute and cache stream modes), undeclared-description edit test |
| Robust-set coverage | Theorem A on `s_lo` | `(P, Y)` exchangeable (A3), clean-task gold (A4), fixed weights (A7) | `test_matched_robust_set_coverage_under_box_adversary` |

## Training in a cell

What is trained: `A`, `w` (judge), `W_e` (evidence), the shared LoRA and the P→D bridges (as for `G16DecisionHead`).
Loss per decision, all closed form from one forward (`g16sym_losses`):

- `ce` = cross-entropy of softmax(score). Trains the point head, including how E is used.
- `robust_ce` = cross-entropy at the worst box corner (gold at `lo`, rivals at `hi`). This loss trains `lo`/`hi`.
  It upper-bounds the CE under any untrusted text (IBP-style, but exact: the box is closed form). Minimising it
  widens the trusted margin `b_y − b_k > c(‖a_y‖₁ + ‖a_k‖₁)` that makes `s_lo` small for the gold and large for
  rivals, i.e. matched robust sets of size 1.
- `width` = mean `c‖a_k‖₁` over allowed options (interval half-width). Penalising it shrinks the box, so sets get
  smaller, but it also caps how far untrusted evidence can move a logit. `c` and `width_coef` set that trade-off.
- Recipe: `total = ce + λ_r·robust_ce + λ_w·width`. Warm-start: λ_r ramps 0 → 0.5 over the first 20% of steps (as in
  IBP training; a cold robust loss collapses E to zero), λ_w ∈ {0, 1e-3, 1e-2}. Sweep `c` ∈ {0.5, 1, 2}. Optional
  conformal-training term (Stutz et al. 2022): a smooth size penalty on the matched `s_lo` set computed per batch.
- Data: the cell's train rows rendered `sym-fork` with the host provenance declaring instructions + criteria
  instruction-trusted (the training renderer must emit those segments; otherwise every interval collapses).
- Report (card): coverage under attack, mean `|C_lo|` vs `|C_clean|`, singleton/act rate, acted-on error, the
  independence rate, and the paired on/off ablation against the stock sym-fork `fork` head (Index + Composite).

Cost:
- Head: `r·d + d + r·d` parameters (d = 2560, r = 8: about 43k). Head FLOPs are negligible. `robust_ce` and `width`
  need no extra forward.
- Backbone: G15 doubles the trusted part of the work. The P stream runs over the trusted tokens of each `prefix + cont_k`
  (deduplicated when trusted lists coincide; with trusted descriptions they differ, so about K × (trusted prefix +
  cont) P tokens in recompute mode). D runs `T + Σ|cont_k|` in cache mode. Rough estimate against a plain sym-fork
  cell at the same K: about 1.3–2× tokens per row, depending on the trusted fraction (prefix instructions + option
  text trusted, state untrusted). Caching the shared trusted prefix in P once and forking it, as D already does, would
  bring P back to `T_trusted + Σ|cont_k|`. That fork is not implemented (`fork_streams` runs P on K full rows); it is
  the first thing to build if the cell is P-bound.
- No change to the eval path: export is a forward pass plus a JSON write.

## Not covered

- Paraphrase of option text, rewording of the question, or a different option set. These are not reorder/relabel.
- Untrusted descriptions. They carry no trusted signal, and the intervals cannot separate those options.
- Training-time poisoning: G15/G16 hold for fixed weights, as stated in `controlsafe/g16.py`.
- `DecisionModel` registration (`TrainConfig(head="g16sym")`) is not wired yet. The head and losses are ready for it.
