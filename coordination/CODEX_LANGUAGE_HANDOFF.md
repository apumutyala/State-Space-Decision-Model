# Codex language growth ownership and handoff

> **Rate note (2026-10-09):** the dollar figures here use the 2026-10-06 SF Compute rates ($2.97/h = $0.0495/min batch, $3.96/h interactive) and are left as written. From 2026-10-08 the H100 list price is $5.04/h and batch jobs bill $3.78/h ($0.063/min; Fridays $0.05628/min), so batch figures scale by x 1.27. Source and history: `jobs/common/rates.py`.

6 October 2026. The user asked Codex to critically review the concurrent Claude work and take the language-growth build-out. Codex accepts that lane. Claude's latest saved session proposed the same split; its formal acknowledgment of this specific interface is still pending.

## Ownership

| Lane | Owner or proposed owner | Scope |
| --- | --- | --- |
| System 1 parent | Claude, consistent with its current proposal | Backbone selection/conversion, recurrent decision heads, model training and primary decision benchmarks |
| Language growth | Codex, requested by user | Semantic programs, checker/realizer, multi-step language supervision, capability retention and semantic intervention evaluation |
| Shared boundary | Both review | Versioned decision adapter, evidence/policy semantics, parent checkpoint handoff and one spend ledger |

Codex files are `plans/codex/`, `analysis/CODEX_CRITICAL_COMPARISON.md`, `interfaces/DECISION_LANGUAGE_V0.md`, and this note. Future implementation is proposed under `language_growth/` and its tests. This does not reserve or rewrite Claude's existing `plans/claude/`, `reviews/`, `techniques/` or analysis outputs. Do not broadly stage one another's uncommitted files.

Read [the language plan](../plans/codex/LANGUAGE_GROWTH_PLAN.md) and [the draft interface](../interfaces/DECISION_LANGUAGE_V0.md). The minimum request to the decision lane is a versioned `decide` adapter with raw/served probabilities, limits, stateless operation, and optional opaque state handles. No particular backbone, hidden-state dimension, concept count or tokenizer is required by the language checker.

## Items for Claude to review

1. Accept or revise the proposal-only response contract and the separation of authoritative records, extracted beliefs and quotations.
2. Provide one frozen parent revision with adapter/config/tokenizer identifiers, a small replay fixture and limits. If the parent is not ready, Codex proceeds with oracle/replay backends; learned capability remains untested.
3. Agree how to learn semantic operation choices through the parent's native head. Initial experiments may make several bounded decision calls; report total answer latency rather than calling the language output single-pass System 1 inference.
4. Keep P2's first surface language entirely within checked productions. Unverified connective text must not silently inherit the guarantee.
5. Reconcile the $525 spending plan before paid jobs. There is one shared allowance, not one per agent.

## Proposed shared envelope

| Allocation | Proposed maximum |
| --- | ---: |
| Shared setup, baseline and throughput gate | $75 |
| Additional System 1 capability/conversion/head work | $250 |
| Language-growth learned pilot and retention checks | $100 |
| Shared confirmation, failures and overhead reserve | $100 |
| Total | $525 |

These are suggestions, not reservations or authorization to launch. Codex has spent $0 on this lane. Its immediate contract/data/checker work is local and CPU-only. At the currently published $2.97 batch rate, the proposed language envelope is at most 33.7 H100-hours before overhead. Prefer a first learned-language probe capped at $20, with the remainder conditional on useful decisions and a measured bottleneck. No external API budget is implied by SF credits.

Important corrections for the decision lane: Qwen's eight full-attention text layers are verified, but the Clef head/vision path also matter; conversion cost and retrieval preservation remain unknown; a held-out test slice must not become a hidden tuning set; a learned provenance embedding is not an exact noninterference proof. The [critical review](../analysis/CODEX_CRITICAL_COMPARISON.md) also contains corrections to Codex's earlier small-model allocation.

No commit, push, model download, paid run or external message was performed by Codex for this handoff. Communication is through these user-requested shared local artifacts; no acknowledgment is inferred from another process merely being active.
