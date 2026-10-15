# Student v1: verified capability scenarios

## Registered experiment (2026-10-15)

Question: can a one-pass Qwen3.5-4B close most of the remaining public-231 Intelligence gap to H2O-Lightning-4B without losing v0's manipulation resistance? This is exploratory, not a sealed benchmark claim.

The author receives only family descriptions derived from `receipts/2026-10-11-jev3-draw/FAMILY_SPEC_JEV3.md`, `receipts/2026-10-10-jev2-draw/FAMILY_SPEC.md` and the family boundary in `harness/families.py`. No public item text is read during data creation. No public outcome selects prompts, data, hyperparameters or checkpoint.

- Planned draw: 7,200 items, seven families (documents/policies, temporal-numeric, rubric judging, multi-hop, state tracking, guardrails, support); 3,600 noul, 2,520 choice, 1,080 score. All noul items are single-fact contrastive pairs with exactly balanced labels. Both members survive or neither does.
- Open author: pinned Qwen3.5-9B. Thinking authorship followed by JSON generation; different narrative domains and seeded briefs, not generated template substitution. Most briefs request 180–350 words, with a longer documents/policies stratum requesting 650–900 words. These are requested lengths, not yet measured lengths.
- Blind verifier: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), revision `6cee5e81ee83917806bbde320786a8fb61efebee`, Apache-2.0. Use is conditional on a real H100/vLLM 0.30.0 smoke. gpt-oss-120b was considered; 20b leaves more KV headroom and a practical token budget under the $35 cap. No 120b compatibility claim is made.
- A second independent thinking verification is supplied by Qwen3.5-9B, trust mode, two samples, 1,024-token thinking budget. Both samples must match the authored answer and the blind cross-family verdict. The two samples also supply the mean soft distribution; this reuses verification as labelling, not an additional independent vote.
- Rebuild option keys from text. Reject ambiguity, duplicate options, unverifiable numeric audits, incomplete pairs and malformed output. Recompute supported extracted arithmetic/calendar/business-day facts with bounded deterministic code; this does not prove every prose inference.
- Every authored and final training-visible row passes every JevBench fingerprint matcher, exact-segment and draw-level document guard, plus attack_guard; unavailable/disabled guards fail closed. **The Decision Index document-overlap guard cannot run because its index is unavailable.**
- Union with all 8,105 v0 screened training labels. A group-disjoint synthetic dev split is reserved from the new draw before any student scoring. No public item enters the training mix. Restricted-label KL + 0.2 CE; no rationale-token loss.
- Training candidates are selected only using synthetic dev objective and accuracy, with fixed tie-breaks. The final artifact must be declared in a committed JSON before public scoring. Public Intelligence is scored once with the existing H2O shim (including its disclosed decisive-noul floor); Card T1/T2 use base/trust one-pass arms and the existing semantically audited evidence panel.
- Per-item text, outputs, labels and reasoning: `$SSDM_PRIVATE/distill_v1/`. Model output is parsed as data, never executed. Model weights: `keel-results` volume. Aggregate outputs only enter git.

### Synthetic-only selection rule

Reserve 160 capability source groups by a label-blind SHA256 ordering; keep both
members of a noul pair in the same split. Combine those with the already
group-disjoint v0 pilot labels for synthetic dev. Compare three prespecified
recipes on 1,536 union-source pilot rows:

| Candidate | LoRA rank | Epochs | Capability fraction |
|---|---:|---:|---:|
| r16-e1-balanced | 16 | 1 | 1/2 |
| r32-e1-balanced | 32 | 1 | 1/2 |
| r16-e2-capability | 16 | 2 | 2/3 |

Choose the lowest mean of the two source-specific dev objectives (capability
and v0); ties break by higher balanced accuracy, then table order. Train that
recipe from the pinned stock 4B on the complete training union. Mixing
oversamples the underweight source rather than discarding any v0 row.
Learning rate 1e-4, batch 4, accumulation 4, alpha twice rank, maximum 4,096
tokens with overlength rejection rather than truncation, seed 20261015.
Selection pilots do not export checkpoints; only the full-union artifact is
eligible for final public evaluation.

## Execution and results

The cross-family verifier passed a real H100/vLLM **0.30.0** load and
reasoning-plus-JSON arithmetic smoke: 1/1 correct, 121 thinking tokens, no
truncation. Load 226.75 s; generation 3.51 s. This establishes runtime
compatibility, not broad capability. Aggregate: `oss-smoke.json`.

CPU acceptance: `python -X utf8 -m pytest -q keel/tests` — **425 passed in
35.83 s** (2026-10-15). The new tests cover exact one-fact pair flips, pairwise
retention and noul balance, text-based option-key reconstruction, a planted
public-overlap fingerprint and disabled-guard failure, bounded arithmetic/date
audits, blind verifier inputs, two-sample agreement, and a no-public,
group-disjoint v0/capability training union.

## Commands

```bash
source ~/ssdm/env.sh
python -X utf8 -m keel.distill.v1_pipeline smoke-oss
python -X utf8 -m keel.distill.v1_pipeline author
python -X utf8 -m keel.distill.v1_pipeline verify
python -X utf8 -m keel.distill.v1_pipeline label
python -X utf8 -m pytest -q keel/tests
python -X utf8 -m keel.distill.assembly_v1
python -X utf8 -m keel.distill.train_v1 select
python -X utf8 -m keel.distill.train_v1 final
```

## Spend

Modal ceiling: **$35**. H100 estimate: $3.95/hour; pipeline charges conservative local wall time including startup/shutdown to its private append-only ledger. The final report will enumerate every paid attempt. No paid call has completed at registration.

| Date | Function | H100 minutes (upper estimate) | USD (upper estimate) |
|---|---|---:|---:|
| 2026-10-15 | gpt-oss-20b compatibility smoke | 4.345 | 0.286 |
