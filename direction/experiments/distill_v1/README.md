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

## Execution and results

Not yet measured. This file is the pre-execution protocol; final measured results and paid-run ledger will replace this paragraph after execution. No checkpoint or capability/robustness gain is claimed at registration.

## Commands

```bash
source ~/ssdm/env.sh
python -X utf8 -m keel.distill.v1_pipeline smoke-oss
python -X utf8 -m keel.distill.v1_pipeline author
python -X utf8 -m keel.distill.v1_pipeline verify
python -X utf8 -m keel.distill.v1_pipeline label
python -X utf8 -m pytest -q keel/tests
```

## Spend

Modal ceiling: **$35**. H100 estimate: $3.95/hour; pipeline charges conservative local wall time including startup/shutdown to its private append-only ledger. The final report will enumerate every paid attempt. No paid call has completed at registration.
