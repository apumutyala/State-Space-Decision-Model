# full8: all 8 attention layers of Qwen3.5-9B -> Gated DeltaNet (Q9-AF). Spec only, NOT submitted

**Trigger.** Issue #87. Job-5cmat (one-layer probe, issue #68) returned `proceed-all-8` under the 511e8dd rule.
`jobs/DECISION_TREE.md` branch A maps that verdict to this job: convert all 8 layers in KL-importance order, with
accept/rollback per layer.

**Status of this file.** This commit holds only the pre-registered decision rule (`jobs/full8/rule.json`,
`scripts/decide_full8.py`, `tests/test_full8.py`). It is committed on its own, before the runner, the pricing or any
run. Later commits may add sections to this spec. They may not change the rule.

## Decision rule (BINDING, pre-registered: `jobs/full8/rule.json`, applied by `scripts/decide_full8.py` only)

**Where every constant comes from.** Every constant comes from text that existed before job-5cmat ran
(2026-10-06T21:33Z). No threshold was tuned on 5cmat's numbers. There are two sources:
- the probe's 511e8dd `success_rule` and enforced gate (`configs/probe9b_h100.json` @ 511e8dd);
- the full-8 row of `HANDOFF.md` / `AGENT_HANDOFF.md` @ caa8f55 (2026-10-06 17:59Z): "per-layer gate ΔNLL ≤ 0.05,
  cumulative ≤ 0.15, hidden cos ≥ 0.95; on retrieval failure: no further spend, report the cost-of-attention-freedom
  curve, `decision-needed` on shipping a hybrid".

**Per-layer gate.** This is the 511e8dd gate, unchanged. It is enforced inside the run by `convert.progressive`, in
KL-importance order. Each layer passes or fails:
- step ΔNLL vs the last accepted model <= 0.05;
- cumulative ΔNLL vs the teacher <= 0.15;
- final-hidden cosine >= 0.95;
- 32 gate windows, as shipped in PR #73 and run in 5cmat.

A failing layer is rolled back to attention. Its GDN is reset, the earlier GDNs return to their pre-step weights, and
the loop goes on to the next layer. No layer is retried.

**Gate audit.** Before any verdict, `decide_full8` re-derives every step from its recorded numbers. It checks:
- the 8 steps are the 8 attention layers, in ascending single-swap-KL order;
- pass/fail re-computes from the constants above;
- `accepted == passed`, since the policy is enforce;
- each baseline is the last accepted nll, or the teacher's if none was accepted;
- the converted list equals the accepted layers;
- the final probe ran on exactly that list.

Any mismatch gives `no-measurement`.

**Final model vs stock.** Same code paths as the probe, same items. The 511e8dd bounds apply, with one rescaled:

| Check | Bound | Source |
|---|---|---|
| ppl ratio, wikitext-2 test, ctx 1024 and 4096 | <= **1.1618** | Rescaled. See below |
| MMLU subset drop | <= 3.0 points | 511e8dd, unchanged |
| Index DEV-subset paired diff (2,008 rows, ids sha256 `6e4cb8df...`) | point >= -1.0 and 95% CI low >= -2.5 | 511e8dd, unchanged |
| MQAR mean accuracy drop | <= 0.10 | 511e8dd, unchanged |
| S-NIAH | mean drop <= 0.10 and every length's drop <= 0.25 | 511e8dd, unchanged |

**Why only ppl is rescaled.** The ppl bound is exp(0.15) = 1.16183, rounded down to 1.1618. Of these checks, only
ppl has an 8-layer budget in the plan text: the gate's cumulative ΔNLL <= 0.15. That budget bounds the same
quantity the held-out ppl ratio measures, NLL vs the teacher on wikitext. The 511e8dd one-layer bound of 1.02 is the
per-layer share of that budget: ln 1.02 = 0.0198 ≈ 0.15 / 8. Keeping 1.02 for 8 layers would make the plan's own
cumulative budget unreachable. MMLU, Index, MQAR and S-NIAH have no 8-layer scaling anywhere in the plan, so their
one-layer bounds apply unchanged. That makes them stricter for 8 layers than for one.

**Verdicts.** They are evaluated in this order. N is the number of accepted layers.

| Verdict | When | Next step |
|---|---|---|
| `stopped` / `no-measurement` | Any of: a pipeline stop (exit 3/4); fewer than 8 layers attempted (deadline or crash); a gate-audit mismatch; a missing number | Exit 9. Never a verdict and authorises nothing. The checkpoint and the exports keep every finished layer |
| `no-conversion` | N = 0 | Fold: the m2 backbone is RWKV7. The Index arms are skipped |
| `partial-conversion` | 1 <= N <= 7 | See "When the gate rejects layers" below |
| `quality-fail` | N = 8, and ppl, MMLU or Index is past its bound | Fold: the m2 backbone is RWKV7 (the DECISION_TREE quality-fail mapping). The Q9-AF weights are archived, not pinned |
| `retrieval-fail` | N = 8, quality passes, and MQAR or S-NIAH is past its bound | See below |
| `q9af-backbone` | N = 8 and every bound passes | **Q9-AF becomes the m2 decision-cell backbone.** Pin `gdn_converted.pt`'s sha256 as `q9af.converted_sha256` in `jobs/m2_pilot/m2_pilot_cell_q9af.json`, in its own reviewed PR. Then the m2 parent arm and pilot cell run with `PARENT=q9af` |

**What happens on `retrieval-fail`.** This is the PLAN's pre-registered fallback:
- attention is **not** added back;
- the cost-of-attention-freedom curve is reported, with 8, 7, ... 0 attention layers taken from this job's
  per-layer probes;
- Q9-AF stays the pure-recurrent entrant, as PLAN says;
- the m2 backbone (Q9-AF anyway, RWKV7, or a hybrid) is `decision-needed` for the user;
- there is no further conversion spend.

**When the gate rejects layers.** A model with 1-7 converted layers is a hybrid with 8 - N attention layers. It is
**not attention-free**, so it can never be the Q9-AF backbone: `heads.backbone.q9af_from_stock` refuses any
conversion that does not cover exactly the 8 attention layers. Such a model:
- is reported as `Q9-H<8-N>`, with the converted and kept layer lists;
- gets the same ppl / MMLU / Index / MQAR / S-NIAH numbers and the curve, non-binding for Q9-AF;
- has its weights exported as `gdn_partial.pt`, never as `gdn_converted.pt`.

The next step is the same whatever those numbers say:
- the m2 backbone defaults to RWKV7, which is attention-free;
- shipping the hybrid is `decision-needed` for the user (PLAN: "we do not quietly add attention");
- a rejected layer is not re-converted without a new spend issue and a new pre-registered rule.

If the cumulative budget runs out on a late layer, the result is `partial-conversion` too. The verdict does not
depend on why a layer was rejected.

**Reported, never binding** (`decision.json: report`, computed after the verdict):
- the cost-of-attention-freedom curve;
- per-layer gate numbers and stage-2 receipts (tokens, tok/s, MFU, peak);
- ECE per arm and prefill throughput / call latency, stock vs converted;
- an MMLU noise note. The subset has 285 items, so a fail within 2 SE of the 3.0-point bound is labelled
  noise-limited. The rule is still applied as written.

Job-5cmat's single-layer numbers (dNLL 0.0068, ppl ratio 1.007, Index +1.66) are context only. No bound depends on
them.
