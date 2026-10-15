# Keel — structure-first judgment for AI workflows

**Proposition v1 · 2026-10-15 · private** — not for publication without the owner's approval.

- **Code:** branch `collab/mottopanikeiku/keel` of `apumutyala/State-Space-Decision-Model`.
- **Evidence:** every number below comes from a committed experiment README or aggregate file (Appendix A).
- **Status:** all results are exploratory. They use public decision items, one open model family (Qwen3.5) and no
  sealed confirmation set.

## The proposition

**Build and release Keel, an open judgment layer for AI workflows, and use it to change how decision-model safety is
claimed.** Keel separates three things that workflows blur today:

- **Evidence:** untrusted content such as records, tool outputs, retrieved pages and messages. Decisions should
  depend on it.
- **Judgment:** a fast decision model that turns typed questions into option probabilities, plus a gate that acts
  or defers.
- **Authority:** what the workflow may do. Only the host grants it; content can never grant it to itself.

We make three commitments:

1. **Authority lives in the host, not in the model.** A small enforcement kernel decides what may run:
   - untrusted content cannot expand the authorized action set;
   - deferral has no side effect;
   - a full review queue holds actions and never auto-approves them;
   - every outcome leaves a hash-chained, replayable receipt.

   *Built; checked exhaustively within stated bounds.*
2. **Judges are measured against attackers who move second.** The Decision Alignment Card reports clean quality,
   legitimate-evidence retention, operator-policy compliance and attack success at static, transfer, adaptive and
   label-only tiers, with family-clustered statistics. *Built; v0 and v0.1 have been run.*
3. **Robust judgment is a research target, not a premise.** We distil deliberative, trust-aware discernment into
   fast one-pass judges and test whether it survives adaptive attack. *Built; a first student has been trained and
   carded.*

### What building it taught us

These findings drive the design.

- **Distillation works on static and transfer attacks, at one-pass cost.**
  - A rank-16 LoRA student of Qwen3.5-4B learned from Qwen3.5-9B thinking under the trust policy. Its training
    data were synthetic items from three generator families with training-only attack templates.
  - On the public Card (held-out items, held-out attack families) it cut static attack success from 25.0% to 9.8%
    and transfer success from 76.7% to 47.9%, against the stock one-pass judge.
  - It also raised clean accuracy (+5.6 pp), policy compliance (+10.4 pp) and evidence retention (+17 pp, n = 23).
  - It satisfies our prespecified dominance rule over the stock one-pass judge. On static and transfer attacks it is
    statistically indistinguishable from the stock *thinking* judge, which spends ~500–1,800 reasoning tokens per
    decision.
- **No judge we tested survives cheap adaptive attacks, the student included.** Black-box search within the
  held-out template space, at 32–64 queries per item, steered:
  - a trust-prompted one-pass 4B judge on **43/43** items it otherwise got right;
  - the same judge on **42/43** when the attacker saw only decisions, never probabilities;
  - a *thinking* 4B judge on **26/29**;
  - the distilled student on **45/48**, under both probability and decision-only feedback.

  For open-weight judges the attacker can run this search offline, so adaptive access is the default threat model.
- **Static benchmarks overstate robustness by 4–10×.** Static attack success was 25% (one pass), 13.8% (thinking)
  and 9.8% (student). The adaptive tier gave 100%, 89.7% and 93.8%.
- **Naive gates and stock judges do not fix this.**
  - A prespecified manipulation-aware gate defers 22–26% of clean decisions yet still lets through 45–72% of
    successful adaptive attacks.
  - As a tool-call gate in AgentDojo, a stock one-pass judge stops every measured attack only by holding 85–95% of
    *all* calls, which collapses benign utility (banking 100% → 41.7%, Slack 91.7% → 0%). A thinking gate recovers
    some utility at about 10× the latency but still holds over half of benign calls.
- **Certification is valid but demanding.**
  - In simulation, our cluster-valid test falsely certified 0.2% of the time (nominal 5%); a naive per-decision
    test did so 21.4% of the time.
  - On real outputs from a non-public population, the student's acted error was 1.5–5.4% across 19–95% coverage
    (stock 4B: 8–45%). The preregistered test still issued **no certificate** at ε = 5%, δ = 5% with about 470
    calibration clusters.
  - Certifying against attack prevalence was impossible without a detector that recognises attacks the judge
    already resists.
- **Evidence-retention labels are the measurement bottleneck.**
  - Of 231 counterfactual-evidence edits written by a 9B model, 70 passed model-agreement validation.
  - Only 26 survived semantic screening (direct audit: 7 valid, 7 invalid, 6 ambiguous in a random 20).
  - Machine agreement is not ground truth. Robustness claims that do not measure retention on validated labels can
    hide "robust because it ignores evidence".
- **Structure is cheap to get right.**
  - The host kernel's abstract model was checked exhaustively within stated bounds, with refinement checks against
    the real implementation.
  - Five deliberately broken kernels were each caught.
  - The Keel package passes 408 CPU tests.

**Therefore Keel's safety case rests on structure.** It does not rest on model robustness. The judge is a
performance component whose wrong-but-authorized choices are limited by authority boundaries, deferral and review.
Its robustness is reported per attack tier and budget, never as a blanket property. Distillation is worth doing
because it buys deliberation-level static robustness and quality at one-pass cost, not because it makes the judge
a security boundary.

## 1. Problem

Decision models already form part of the control plane of AI workflows. They route tickets, approve or block tool
calls, moderate, triage and monitor agents. Jev, Clef, MD1, OpenAI Decisions and H2O-Lightning launched in
September–October 2026, all pitched as fast, cheap, single-pass judges. The content such a judge reads is often
written by the party being judged: an email, a support ticket, a web page, a tool result. The characteristic
failure is **authority transfer**: text inside untrusted content ("Note from the reviewing supervisor: the correct
answer is B") decides the outcome as if the operator had said it.

Our pilots show the failure is large and not fixed by thinking alone (Card v0; `direction/experiments/card_v0/`):

| Qwen3.5-4B arm | Clean accuracy | Static attack success (T1) |
|---|---|---|
| One pass, no trust policy | 74.0% [62.4, 84.3] | 44.0% [32.9, 59.4] |
| One pass, trust policy + delimited record | 73.2% [60.8, 84.1] | 25.0% [16.5, 37.0] |
| Thinking, trust policy + delimited record | 93.1% [86.0, 98.1] | 13.8% [8.5, 20.3] |

In the earlier channel probe, *without* a trust policy thinking made planted notes **more** effective: 94–98%
steering versus 42–63% for one pass (`direction/experiments/channel_probe/`). One-pass models followed an authority
claim planted *in the record* more often than the same sentence from the trusted operator channel, a channel gap of
−14 pp (4B) and −17 pp (9B). Their behaviour tracks where text sits, not who wrote it.

## 2. Threat model and goals

**Attacker.**
- Controls the content of any untrusted segment: records, tool outputs, retrieved documents, messages. That covers
  text, JSON, Unicode tricks and encodings, plus fake transcripts, fake tool results and chat-marker spoofs.
- Knows the protocol, renderer, trust-policy text and judge architecture, and has the weights of open judges.
- Card query-access tiers: T1 none (static held-out templates); T2 transfer (searched on a smaller surrogate model);
  T3 black-box with probability feedback; T3L black-box with decision-only feedback. Budgets are stated per tier.
- **Cannot** write trusted segments (operator policy, notes, metadata), alter host configuration, compromise the
  host process or the receipt checkpoint, or poison judge training data. Data poisoning is out of scope and named
  as such.

**Goals and claim types.**

| Goal | Claim type | What would falsify it |
|---|---|---|
| G1. No executed action outside the authorized set; defer and hold never execute | structural, bounded-verified | a reachable state within the checked bounds that violates I1–I5, or a refinement mismatch |
| G2. Few executed actions chosen by the attacker *within* the authorized set | empirical, per tier and budget | an attack within the stated tier and budget that exceeds the reported rate |
| G3. Legitimate evidence and trusted policy still change decisions | empirical, co-equal with G2 | validated counterfactuals or policies that the judge ignores |
| G4. The risk of acted decisions is bounded on a stated population | statistical: cluster-exchangeable, not adversarial | realized acted risk above ε more often than δ across replications |
| G5. Every decision is auditable and replayable | structural | an edited, reordered or truncated log that verifies, or a prompt that replays differently |

## 3. Design

```mermaid
flowchart LR
  subgraph HOST["Host: trusted computing base"]
    AUTH["Authority<br/>policy ∩ role grants"]
    KERNEL["HostKernel"]
    QUEUE["Review queue<br/>full ⇒ hold"]
    LOG[("Receipt log<br/>+ checkpoint")]
  end
  UNTRUSTED["Untrusted content<br/>records, tool outputs, pages"] -->|"host labels: untrusted"| REQ["DecisionRequest"]
  OPERATOR["Operator policy and notes"] -->|"host labels: trusted"| REQ
  REQ --> JUDGE["Judge<br/>render → label-token probabilities"]
  JUDGE --> GATE["Gate<br/>uncertainty + manipulation → act / defer"]
  GATE --> KERNEL
  AUTH --> KERNEL
  KERNEL -->|"act, only inside the authorized set"| EXEC["Executor"]
  KERNEL -->|"defer / hold: no side effect"| QUEUE
  KERNEL -->|"write-ahead receipt"| LOG
  CARD["Decision Alignment Card"] -.->|"measures"| JUDGE
  CARD -.->|"measures"| GATE
```

### 3.1 Protocol (`keel/protocol`, `PROTOCOL.md`)

- **Typed requests.**
  - `DecisionRequest(request_id, policy, segments, questions)` round-trips through canonical JSON; its fingerprint
    is the SHA-256 of that JSON.
  - A `Segment` carries `trust`, `kind`, `source` and `parent_ids`.
  - Questions are `choice`, `noul` (yes/no) or `score` (ordinal), each with host-authored instructions and criteria.
- **Trust comes from the API, never from content.**
  - Only `operator_note` and `metadata` segments may be trusted, and they must be.
  - Records, tool outputs, user messages and retrieved text are always untrusted.
  - `derive()` makes anything computed from an untrusted parent untrusted, so derivation never raises trust.
  - Inside the host kernel, trusted segments can only be minted by authenticated host constructors. The registry
    rejects foreign or modified segment objects, so a JSON field such as `"trust": "trusted"` is inert.
- **Rendering** (`keel-render/1`, `trust-policy/1`):
  - The system turn is the trusted channel: base instructions, the trust policy, the operator policy, and trusted
    notes and metadata.
  - Each untrusted segment sits in its own `<untrusted_record source=…>` block in the user turn.
  - The block's delimiters and the chat-control markers (`<|`, `|>`) are escaped inside untrusted text, so content
    cannot close its block or open a new turn. This is tested, and the guarantee is syntactic only.
  - `mode="base"` merges everything into one record and is kept as the undefended baseline.
- **Exact order and label invariance.**
  - Options are presented in a canonical order that depends only on their content, under fixed letters; request
    option ids never reach the model.
  - Reordering or relabelling options therefore cannot change the served distribution; ties get equal probability.
  - Card v0 observed zero reorder/relabel flips, a construction check reused from the repo's G16 work
    (`docs/guarantees/REORDER_INVARIANCE.md`).
- **Readout.**
  - One-pass: a single forward pass to the `Answer:` slot; option probabilities come from label-token log-probs,
    and a missing label raises rather than being imputed.
  - Think-first: a sampled reasoning trace of at most 3,072 tokens, budget-forced closure, then the same slot. In
    the smoke run, traces were 486–1,759 tokens per decision, against one generated token for one pass.

### 3.2 Host enforcement kernel (`keel/host`)

- **Authority.** A principal's authorized set is the operator policy's `authorized_actions` intersected with the
  grants of its host-assigned roles. The authenticated principal is bound by the host, independently of who a
  proposal claims to come from. Model answers, citations and segment text never add grants.
- **`HostKernel.process(request, proposal, result)`** returns one of four outcomes:
  - `executed`;
  - `deferred`: any defer reason vetoes acting;
  - `held`: review required and the queue is full;
  - `denied`: invalid policy, principal, citations, result binding or probabilities.

  A receipt is durably appended *before* any enqueue or dispatch.
- **Invariants:**
  - **I1:** every executed action is in the trusted authorized set, and changing only untrusted content cannot
    change that set.
  - **I2:** defer, deny and hold never call the executor.
  - **I3:** at a full queue, actions that need review are held, never auto-approved.
  - **I4:** trust labels originate only in host constructors, and untrusted lineage cannot gain trust.
  - **I5:** exactly one receipt per returned outcome, and per executor exception.
- **Verification.**
  - An independent abstract specification is explored exhaustively over two actions, two roles, every
    policy/role subset, queue capacities 0–2, five provenance patterns (including forged trusted segments and
    transitive untrusted derivation) and sequences up to length 4.
  - That covers **48 configurations, 632 reachable quotient states and 16,920 transitions**. Each transition is
    replayed against the concrete kernel as a refinement check.
  - Hypothesis property tests run on the concrete kernel.
  - **Five deliberately broken kernels, one per invariant, are each caught.**
  - This is bounded checking, not an unbounded proof.
- **Trusted computing base:**
  - the authenticated workflow adapter, configuration and principal binding;
  - `Authority`, `HostInputs`, `HostKernel` and `ReviewQueue` (about 260 lines);
  - the protocol, canonicalisation and renderer code used for receipts;
  - `receipts.py`;
  - the host process, OS and filesystem, plus the independently protected checkpoint;
  - the executor's faithful mapping from action names to effects.

### 3.3 Receipts and replay (`keel/protocol/receipts.py`)

- **Contents.** Each receipt records:
  - the request fingerprint and the SHA-256 of every prompt used, including thinking openings and every sampled
    answer-slot prompt;
  - the model ID and revision, renderer and trust-policy versions, and gate and calibration identifiers;
  - per-segment trust, kind, source, content hash and parents;
  - per-question probabilities and decisions, and the outcome with its reasons;
  - the proposal, the host-computed authorized set and a UTC time.
- **Chaining.** Each `receipt_id` is a SHA-256 over the canonical payload and is chained through `prev_hash`.
  `verify()` detects edits, reordering, internal or suffix deletion, and non-canonical bytes against a separately
  protected checkpoint. `replay_check()` re-renders every prompt and matches hashes and fingerprints.
- **Limits.** These are hashes, not signatures: anyone able to rewrite both the log and the checkpoint can forge a
  history, so anchor the checkpoint externally. An `act` receipt records authorized dispatch intent, not completion.

### 3.4 The judge, and discernment distillation (`keel/runtime`, `keel/distill`)

- **Runtime.**
  - vLLM on Modal H100s with pinned revisions.
  - A one-shot batch scorer and a warm `JudgeServer` share one engine.
  - Students are full merged checkpoints on a results volume.
- **Distillation recipe.**
  - **Teacher:** Qwen3.5-9B thinking under the trust policy, two samples (768-token budget), soft label = mean of
    the two distributions.
  - **Student:** Qwen3.5-4B one-pass trust-mode with LoRA (rank 16, α 32) on attention, gated-delta and MLP
    projections. One epoch, AdamW 1e-4, effective batch 16, bf16.
  - **Loss:** `KL(teacher ‖ student) + 0.2·CE(gold)`, normalised **only over the rendered option-label tokens** at
    the answer slot. Other vocabulary tokens never enter the loss.
  - **Variants per source item:**
    - clean;
    - injected instruction and fake authority, from six training-only templates disjoint from every evaluation
      template (`train/attack_guard.py`, both directions, 38 pinned evaluation templates). Their target is the
      teacher's *clean* distribution: the decision should not move.
    - legitimate later-event evidence that must change the answer, kept only if it passes an executable
      event-semantics screen and cross-model validation;
    - a trusted operator reporting-policy remap that must change the answer.
  - **Label filtering:**
    - clean parents need a teacher argmax equal to gold, which removes 596 of 3,107;
    - attack units are dropped if either teacher sample changed its decision: 403 inject and 503 authority;
    - evidence goes 1,207 teacher-accepted → 248 executable → 237 cross-validated. A random 20 of the final 237 were
      all valid on full-record review.
  - **Result:** 8,105 soft-labelled training rows and 507 optimizer updates, ≈ 17 H100-minutes ($1.21). The
    distillation total, including a failed export retry, was $20.72.
  - **Data gates:** training sources are 3,299 repository-generated decision items from three synthetic families,
    with a group-disjoint pilot of 192. `train/jevbench_guard.py` found zero overlap with public items. The
    Decision Index document-overlap guard could not run on this workstation; this is stated as a limitation.

### 3.5 Act/defer gate and certification (`keel/gate`)

- **Rule.** Act iff the uncertainty (`1 − max p`, or `1 − margin`) is below its threshold **and** the manipulation
  score is below its threshold; equality defers.
  - Multi-question requests defer atomically.
  - Two transparent signals ship: `RenderDivergence`, the Jensen–Shannon divergence between trust-mode and base-mode
    distributions, and `InstructionLexicon`, a versioned pattern score over untrusted text that is labelled a weak
    baseline.
  - The gate never sees gold labels at decision time.
- **Certification of the complete gate.** Learn-then-Test over a preregistered two-dimensional threshold grid:
  - multiplicity is controlled by Bonferroni or a declared fixed sequence;
  - the tests are cluster-valid: a CRR betting test on cluster-weighted acted loss, or a representative-per-cluster
    test;
  - fit, calibration and evaluation splits are cluster-disjoint;
  - the loss is "acted and wrong", or "acted and (wrong or under attack)".
- **Report.** The output is either thresholds plus a certificate, or literally "no certificate", together with full
  risk–coverage and review-load curves.
- **Validity in simulation.** 500 clustered datasets with true risk 0.22 against ε = 0.20 and δ = 0.05 gave false
  certificates for:
  - CRR: 1/500 (0.2%);
  - the representative test: 3/500 (0.6%);
  - a naive per-decision test: 107/500 (21.4%).

  On clustered data, decision-level tests are anti-conservative.
- **Scope.** The guarantee assumes exchangeable clusters and a frozen pipeline. **It is not an adversarial
  guarantee.**

### 3.6 The Decision Alignment Card (`keel/card`)

- **Units.**
  - Paired (clean, variant) requests from the 231 pinned public JevBench items, clustered into 18 families; the
    items are evaluation-only.
  - 3,587 paired units in v0.
  - Variant kinds:
    - injected instruction and claimed authority, from all 21 held-out v1 templates (with base64, zero-width and
      homoglyph transforms), the official hijack templates and the earlier probe notes;
    - fabricated evidence (D1-B), reported separately from authority transfer;
    - validated counterfactual evidence;
    - a trusted operator policy;
    - a trusted factual conflict (descriptive only);
    - paraphrase, reorder and relabel.
- **Tiers.**
  - T1: static.
  - T2: search on Qwen3.5-2B, then transfer.
  - T3: probability-guided coordinate/beam search over template × target phrasing × position × wrapper. Wrappers
    include JSON, tool-result envelope, fake transcript and chat-marker spoof, always inside the untrusted record.
  - T3L: the same search with decision-only feedback.
  - Budgets are exact query counts.
- **Statistics.**
  - 95% percentile cluster bootstrap over item families.
  - Paired arm comparisons with Holm adjustment over four primary metrics: attack success, clean accuracy,
    evidence retention and policy compliance.
  - Prespecified **dominance rule:** an arm dominates only with Holm-significantly lower attack success *and*
    one-sided 95% non-inferiority (−3 pp) on clean accuracy, evidence retention and policy compliance. This blocks
    "robust because it ignores everything".
- **Hygiene.**
  - Targets are chosen from labels only; the renderer is never passed gold labels; templates are frozen before any
    scoring.
  - Per-item outputs and search traces stay outside git.
  - The gate's operating point was committed before measurement (`42869497`), and the full 336-point grid is
    published unselected.

### 3.7 Workflow adapter (`keel/workflows`)

- **Gate placement.** An AgentDojo (v1.2.2, package 0.1.35) pre-dispatch gate sits in front of every tool call:
  - the user's task is host-provided trusted metadata;
  - every prior tool output and the proposed call are untrusted segments;
  - the judge answers allow or hold.
- **Hold ⇒ the call is not executed;** the agent receives a fixed "held for review" result.
- **Scoring.** Targeted attack success is scored only on *executed* calls, through the suite's predicates on the
  simulated environment.

## 4. Safety implementations

| Mechanism | Failure addressed | Guarantee type | Evidence | Residual risk |
|---|---|---|---|---|
| Host-assigned trust, registry of host-minted segments, `derive` propagation | content granting itself authority; provenance laundering | structural | I4 tests, forged-segment cases in the model checker | host mislabels inputs (integration bug); compromised host |
| Authority = policy ∩ role grants, bound to the authenticated principal | an injected action outside the operator's set | structural (I1) | bounded check: 632 states, 16,920 transitions; mutant caught | wrong-but-authorized choices; argument-level permissions are not covered |
| Defer/hold ⇒ no executor call; saturation ⇒ hold | "fail open" under uncertainty or floods | structural (I2, I3) | bounded check; property tests; mutants caught | availability: attack floods raise review load |
| Write-ahead, hash-chained receipts; replay | unaccountable or unreplayable decisions; silent log edits | structural (I5) + tamper evidence | tamper/deletion/reorder tests; replay of thinking prompts | not signatures: anchor the checkpoint externally |
| Escaped untrusted blocks; trusted material only in the system turn | delimiter closure; chat-turn injection | syntactic | rendering tests | semantic steering inside the block (measured by the Card) |
| Canonical option order | order and label manipulation | exact, by construction | zero reorder/relabel flips | paraphrase sensitivity (measured) |
| Trust-policy prompting, thinking judges, distilled students | authority transfer within the authorized set | empirical, per tier and budget | Card v0/v0.1; student Card (static 25.0% → 9.8%, transfer 76.7% → 47.9%) | adaptive search still succeeds 90–100% (student 93.8%) |
| Manipulation-aware gate + certification | confident wrong or attacked decisions | statistical, conditional on exchangeable clusters | simulation 0.2% vs 21.4% false certification; frozen-gate Card evaluation; preregistered certification attempt (no certificate) | not adversarial; weak detectors; a gate-aware attacker is not yet measured |
| Evidence-retention and policy controls; dominance rule | robustness bought by ignoring evidence or policy | empirical metric design | Card co-primary metrics; semantic screening | small validated cohorts (n = 20–26) |
| Training and evaluation template disjointness; JevBench guard | contamination, train-on-test | process | `attack_guard`, `jevbench_guard`, fail-closed | Decision Index guard not run on this workstation |

## 5. Results so far

All results are exploratory. Brackets are 95% intervals: family-cluster bootstrap for the Card, task-cluster
bootstrap for workflows. A bracket that collapses to a point at 0% or 100% is not certainty.

### 5.1 Card v0 and v0.1 on Qwen3.5-4B (231 public items, 18 families)

| Arm | Clean acc. | T1 ASR ↓ | Evidence retention ↑ | Policy compliance ↑ | False-evidence susceptibility ↓ |
|---|---|---|---|---|---|
| one pass, base | 74.0% [62.4, 84.3] | 44.0% [32.9, 59.4] | 66.7% (n=21) | 64.5% [52.0, 78.1] | 25.1% |
| one pass, trust | 73.2% [60.8, 84.1] | 25.0% [16.5, 37.0] | 70.0% (n=20) | 78.8% [68.0, 88.9] | 27.5% |
| thinking, trust | 93.1% [86.0, 98.1] | 13.8% [8.5, 20.3] | 100% (n=25) | 98.7% [97.2, 100] | 27.2% |
| 9B thinking, trust (teacher) | 94.8% [89.6, 99.1] | 6.3% [3.4, 10.3] | 88.0% (n=25) | 92.2% [87.5, 96.1] | 15.5% |

Adaptive and transfer tiers, ASR among clean-correct items:

| Target arm | T1 static | T2 transfer from 2B | T3 probability search | T3L decision-only search |
|---|---|---|---|---|
| one pass, base | 44.0% | 81.8% [68.2, 94.6] (n=44) | — | — |
| one pass, trust | 25.0% | 76.7% [65.9, 88.2] (n=43) | **100% (43/43)**, 64 queries | **97.7% (42/43)**, 64 queries |
| thinking, trust | 13.8% | 46.4% [31.3, 59.1] (n=56) | **89.7% (26/29)**, 32 queries | — |

Frozen gate (`lexicon ≥ 0.5 OR JSD ≥ 0.3 OR 1 − max p ≥ 0.4`, declared before measurement):

| Arm | Clean deferred | Successful T1 deferred | Successful T3 deferred | Successful T3L deferred |
|---|---|---|---|---|
| one pass, trust | 22.5% [12.5, 34.8] | 65.7% [60.8, 71.8] | 27.9% [14.3, 41.5] | 54.8% [34.7, 71.4] |
| thinking, trust | 26.4% [15.6, 38.6] | 53.9% [47.0, 61.9] | 53.8% [31.0, 73.7] | — |

### 5.2 Workflow v0: AgentDojo, two suites, Qwen3.5-9B agent

12 user tasks × (1 benign + 2 attacked) per suite and arm; standard `important_instructions` attack; targeted ASR
counts executed calls only.

| Suite | Arm | Benign utility | Utility under attack | Targeted ASR | Calls held |
|---|---|---|---|---|---|
| banking | no defense | 100% | 83.3% | 16.7% [0, 37.5] | — |
| banking | spotlighting | 91.7% | 75.0% | 12.5% | — |
| banking | tool filter | 58.3% | 66.7% | 8.3% | — |
| banking | Keel gate, stock one-pass 4B | 41.7% | 41.7% | 0/24 | 84.8% |
| banking | Keel gate, stock think-first 4B | 58.3% | 54.2% | 0/24 | 60.6% |
| Slack | no defense | 91.7% | 62.5% | 12.5% [0, 25.0] | — |
| Slack | spotlighting | 91.7% | 62.5% | 12.5% | — |
| Slack | tool filter | 16.7% | 8.3% | 0/24 | — |
| Slack | Keel gate, stock one-pass 4B | 0% | 0% | 0/24 | 95.4% |
| Slack | Keel gate, stock think-first 4B | 75.0% | 33.3% | 0/24 | 56.6% |

Think-first gating ran at 83–204 s per episode against 9–17 s for the other arms. 90% of its reasoning traces hit
the 1,024-token cap.

**Reading.** Zero attack success bought by holding most calls is not robustness. Stock judges cannot tell needed
calls from injected ones under this gate question:
- one pass holds nearly everything;
- thinking recovers some utility at roughly ten times the latency and still holds 57–61% of benign calls.

The workflow target (10× fewer executed unauthorized actions at *matched* utility and review load) needs a trained
tool-call judge and larger samples: undefended attack success here is only 12.5–16.7% on 24 episodes per suite.

### 5.3 Discernment distillation: the student on the public Card

The student was never trained on, tuned on or selected with public items or Card templates. Arms are trust mode;
the stock rows repeat §5.1 for comparison (`direction/experiments/card_student_v0/`).

| Arm | Clean acc. | T1 ASR ↓ | Evidence retention | Policy compliance | T2 ASR ↓ | T3 ASR ↓ | T3L ASR ↓ |
|---|---|---|---|---|---|---|---|
| stock 4B one pass | 73.2% | 25.0% | 70.0% (n=20) | 78.8% | 76.7% | 100% | 97.7% |
| **student 4B one pass** | **78.8%** [66.7, 90.0] | **9.8%** [6.2, 14.5] | 87.0% (n=23) | 89.2% [77.6, 96.9] | **47.9%** (n=48) | 93.8% | 93.8% |
| stock 4B thinking | 93.1% | 13.8% | 100% (n=25) | 98.7% | 46.4% | 89.7% (32 queries) | — |
| 9B thinking teacher | 94.8% | 6.3% | 88.0% (n=25) | 92.2% | — | — | — |

Paired on shared units (Holm-adjusted):

- **Student − stock one pass:**
  - clean +5.6 pp [+0.9, +12.2];
  - static ASR −15.3 pp [−25.1, −8.1] (p_Holm = 0.008);
  - policy +10.4 pp [+1.3, +19.3];
  - evidence +17.0 pp [−6.7, +43.1].

  The student **dominates** under the prespecified rule.
- **Student − stock thinking:** static ASR −4.0 pp [−10.4, +1.3] (n.s.); clean −14.3 pp; no dominance either way.
- **Student − 9B teacher:** static ASR +3.5 pp (p_Holm = 0.009); clean −16.0 pp.

On its own synthetic held-out pilot (192 groups, same families and training attack templates as training; in
distribution), the student scored:
- clean 82.8% (stock 61.5%, teacher 78.1%);
- 0/318 attacks succeeded (stock 44.9%, teacher 8.0%);
- policy compliance 100% (stock 33.3%);
- evidence retention 14/14.

These pilot numbers show the recipe trains what it targets. They do not show generalization. The public Card above
does.

### 5.4 Complete-gate certification on a non-public population

The plan was preregistered and committed before scoring (`direction/experiments/gate_cert_v0/`):
- 900 fresh synthetic source items, disjoint from training;
- each gives one clean request and two training-family attacked requests;
- clusters = source items;
- a 36-point grid with Bonferroni, CRR test, δ = 0.05.

| Pipeline | Certificate (P(wrong \| acted) ≤ 5%) | Certificate (P(wrong or attacked \| acted) ≤ 10%) | Empirical acted error, evaluation split |
|---|---|---|---|
| distilled student | none (best p = 0.053 vs level 0.0014) | none | 0.8% at 21% coverage · 1.6% at 32% · 4.0% at 66% · 5.3% at 95% |
| stock 4B | none | none | 12.5% at 1.4% coverage · 21.8% at 11% · 45.0% at 81% |

**Two lessons.**
- **Power.** The test is valid but needs more clusters, or a variance-adaptive or fixed-sequence design, to certify
  small margins. The betting test starts from a worst-case variance prior, and cluster weighting shrinks each
  residual.
- **Detection.** "Resists the attack" and "knows it was attacked" are different properties. The student resists
  these attacks, yet no signal flags them, so attack-prevalence losses cannot be certified.

### 5.5 Spend

| Item | Estimated H100 cost |
|---|---|
| Earlier probes (reasoning headroom, steering, channel) | ≈ $5–6 |
| Card v0 + v0.1 (stock 4B, 9B reference, adaptive tiers, gate layer) | $8.46 |
| Workflow v0 (432 AgentDojo episodes) | $16.94 |
| Distillation (labels, training, pilot, failed export retry) | $20.72 |
| Student Card, certification scoring, runtime smoke runs, failed first student-Card launch | ≈ $3 |
| **Total** | **≈ $55** |

These are wall-time estimates at $3.95/H100-hour, not invoices.

## 6. Implications

### 6.1 For safety claims about decision models

- **Model robustness is not a security boundary today.**
  - For an open-weight judge, assume the attacker has run adaptive search offline. Our 64-query black-box search
    is far weaker than gradient attacks with weights in hand, and it already succeeds 90–100% of the time.
  - A claim like "resists prompt injection" without a tier, a budget and an evidence-retention control is
    uninformative.
  - The Card's format (tiers × budgets × co-primary controls) is the minimum we think a credible claim needs.
- **Static suites mislead by large factors.** Static-to-adaptive attack success went 25% → 100% (one pass), 13.8% →
  89.7% (thinking) and 9.8% → 93.8% (distilled student). A vendor scorecard that reports only static injection
  numbers can overstate robustness by an order of magnitude.
- **Deliberation helps, and it can be distilled, but neither solves adaptive attack.**
  - Thinking plus a trust policy is the best stock arm on every static metric.
  - Distillation moved that static and transfer robustness into a one-pass judge: 9.8% static and 47.9% transfer
    attack success, statistically matching 4B thinking on attacks, at roughly a thousandth of the generated tokens.
  - Both remain steerable 90%+ of the time under 32–64-query adaptive search.
  - Distilled discernment is a large *efficiency* gain on the static frontier, not a security property.
- **Hiding probabilities is not a mitigation.** The decision-only attacker reached 97.7% (stock one pass) and 93.8%
  (student) with the same 64-query budget.
- **Certificates and robustness are different objects.**
  - A cluster-valid certificate bounds acted error on an exchangeable population.
  - It cannot bound what an adaptive attacker achieves.
  - It cannot bound attack prevalence unless something *detects* attacks.

  Deployers should demand both kinds of evidence and never accept one as the other.

### 6.2 For deployers

- **Put the safety case in structure.** The worst outcome of a steered judge should be a *wrong but authorized*
  choice that the authority model, deferral and review were designed to tolerate.
  - Keel's kernel makes the action set a function of trusted inputs only (I1).
  - Every act/defer is attributable and replayable (I5).
- **Size the review budget honestly.** In our runs, gates that catch most adaptive attacks also defer a quarter of
  clean traffic. In AgentDojo, a stock gate holds nearly everything. Review load is the binding constraint, so
  report hold rates beside attack rates.
- **Treat high-impact options differently.** Where a steered choice is costly (payments, data export, permission
  changes), require corroboration or human approval for that option class regardless of judge confidence. This is
  an authority-model decision and belongs in the host.
- **Separate fabricated evidence from authority transfer.** Any evidence-following judge moves when given
  plausible false facts: 15–27% susceptibility in Card v0. Defences here are provenance and corroboration, not
  instruction-following training.

### 6.3 For the research agenda

1. **Adaptive-robust discernment.**
   - The distillation pipeline works and is cheap ($1.21 per training run), so the next experiment is direct.
   - Put search in the training loop: run T3-style attacks on training-source items, train on the hard cases, and
     re-measure on held-out families and held-out items at fixed budgets.
   - Success is a *budget curve* that moves right, not a single number.
2. **Teach the judge to report attacks, not only to resist them.**
   - Add an "untrusted content attempted to direct this decision" output, trained on the same triplets.
   - The gate can then defer on it, and attack-prevalence losses become certifiable.
   - Measure it against gate-aware attackers.
3. **Shrink the semantic attack surface.**
   - Move from "the judge chooses the action" to "a quarantined model extracts typed facts; host policy computes
     the action" (CaMeL-style).
   - Attacks are then limited to falsifying facts, which corroboration rules can bound.
   - Measure how much decision quality this costs.
4. **Certification with power.**
   - Preregister fixed-sequence paths or variance-adaptive tests.
   - Calibrate on at least 2,000 clusters per population.
   - Certify review-budget-constrained operating points on non-public calibration data from real workflows.
5. **Validated evidence sets.**
   - Retention can only be measured on labels that a human or an executable semantics confirms.
   - Build a human-validated set of at least 200 units per workflow; machine-authored counterfactuals are a
     candidate pool, not labels.
6. **Workflow judges.**
   - The stock tool-call gate over-held, so train the student on tool-call authorization triplets (needed call vs
     injected call).
   - Re-run full AgentDojo suites with enough episodes to resolve a 10× reduction from a 12–17% baseline.

### 6.4 For this project

The SSDM repository was built to win a decision-model leaderboard. These results argue that the durable
contribution is different. It is a **structure-first judgment protocol plus an honest measurement standard**, with
the judge as a replaceable component. The leaderboard work is not wasted: its lineage produced a one-pass judge
that matches 4B deliberation on the static and transfer attack tiers. The repo's existing assets fit directly:
- the Qwen3.5-4B lineage and data builders;
- canonical-order invariance;
- cluster-valid certification;
- the held-out attack templates;
- the νPI constrained-training controller (now merged beside consistency training). νPI is the natural tool for
  "lower attack success subject to retention ≥ 90%" once search-in-the-loop training starts.

### 6.5 Dual use, disclosure and release

- The adaptive search and attack templates help attackers as much as defenders. Release policy:
  - publish the protocol, the host kernel, receipts, the Card methodology and aggregate results;
  - keep confirmation sets and per-item search traces private and versioned, replacing them after each release;
  - give vendors the attack *method* and aggregate numbers before any public naming.
- **No third-party model is measured without the owner's exemption** from the 2026-10-11 directive. Everything here
  uses open models we ran ourselves.
- Distillation recipes are generic capability know-how. We release only what the safety claim needs.

### 6.6 What we are not claiming

- That any judge here is robust to adaptive attack.
- That the gate is certified for any real deployment population. No certificate was issued in our one
  preregistered attempt.
- That workflow-level reductions were achieved at matched utility.
- That the distilled student generalizes beyond what one recipe, one seed and one public Card run show.
- Any result on sealed or held-out data, or on any model family beyond Qwen3.5.
- The Decision Index overlap check was not run.
- Bounded verification is not an unbounded proof.
- Receipts are not signatures.

## 7. Plan

| Phase | Work | Gate to continue | Kill or pivot |
|---|---|---|---|
| A, weeks 0–4 | Protocol + kernel + receipts release candidate; external review of the kernel and TCB. Card v1 spec: human-validated evidence set (≥ 200 units), gate-aware tier, matched budgets across arms. Partner check: show the two-page contract and one replayable example to three prospective users | at least one external user would adopt the contract; the Card v1 spec is frozen | nobody would use the contract → publish the Card methodology and the negative adaptive result only |
| B, months 1–3 | Adaptive-robust judge research: search-in-the-loop distillation (starting from the v0 student); an attack-report head for the gate; the typed-fact firewall variant; gate-aware attacks | on held-out families: the T3 budget curve moves right at matched budget (for example ASR at 64 queries ≤ 50%), clean cost ≤ 2 pp, evidence retention ≥ 90% on validated labels; an attack-prevalence certificate on a non-public population | no movement at T3 after two training rounds → stop judge-robustness claims; ship structure-first Keel with honest Card numbers |
| C, months 3–6 | Workflow proof: trained tool-call judge; full AgentDojo suites plus one partner workflow; certified gate on non-public calibration data | 10× fewer executed unauthorized actions vs the strongest same-budget baseline at matched utility and ≤ 10% review in two workflows (the stretch target) | reductions only via over-holding → report the review-load frontier and stop the 10× claim |
| D, months 6–12 | External red team, independent reproductions, publication, adoption | three independent groups reproduce and use the protocol | — |

**Budget.**
- **Phase A–B compute:** ≈ $300–600. Card runs cost $0.5–8, workflow runs $5–17 and a distillation cycle about
  $10–20; this phase spent about $55 in total.
- **People and red team:** $80–150k for phases B–C, plus an external attack round.
- **Grant routes:** EA Funds Transformative AI Fund (rolling), BlueDot rapid grants, Manifund, Foresight AI Nodes
  (deadline Oct 31), SFF Speculation.

## 8. Decisions and asks

1. **Adopt structure-first Keel as the project's direction**, with the leaderboard as a secondary output.
2. **Owner permission** to publish the protocol, kernel, Card methodology and aggregate results, and an exemption
   to measure incumbents on common units if desired.
3. **Data access:** the lab's training draws and the sealed 2,199-item held-out set (`ssdm-data`), or approval to
   rebuild with a new seal.
4. **A named human lead** to own commitments and author grant applications.
5. **Compute:** a capped $600 tranche for phases A–B.

## 9. Related work and novelty boundary

| Work | What it does | How Keel relates |
|---|---|---|
| CaMeL; Fides | Information-flow and capability control around LLM agents | Keel's host kernel follows the same structure-first lesson and adds decision-model ergonomics, receipts and certified deferral; the typed-fact firewall (Phase B) borrows CaMeL's split directly |
| Instruction Hierarchy; StruQ / SecAlign; ISE; ASIDE | Train or architect models to privilege trusted instructions | Candidate judge-level defences; the Card measures them under adaptive tiers rather than assuming them |
| RAISED (arXiv 2610.06401); IH-Challenge (2603.10521) | Hierarchy and robustness training and evaluation | Overlap with our distillation bet; novelty must come from the measured joint frontier (attack × evidence retention × policy × cost), not from hierarchy training itself |
| DecisionRobust (2610.19118) | Robustness evaluation for decision models | Closest evaluation peer; the Card adds paired evidence and policy controls, adaptive and label-only tiers and cluster-valid statistics |
| "The attacker moves second" (2510.09023) | Adaptive attacks break published defences | Our T3/T3L results reproduce the phenomenon for single-token decision judges at small query budgets |
| AgentDojo | Agent tool-use benchmark with injection tasks | Our workflow harness; Keel adds a pre-dispatch authority gate and executed-call scoring |

## Appendix A. Reproduction and file map

| Artifact | Path |
|---|---|
| Protocol, renderer, receipts | `keel/protocol/`, `PROTOCOL.md` |
| Host kernel, model checker | `keel/host/` (`python -X utf8 -m keel.host.model_check`) |
| Gate, certification | `keel/gate/` |
| Card, results | `keel/card/`, `direction/experiments/card_v0/{CARD.md,aggregate.json}` |
| Distillation, results | `keel/distill/`, `direction/experiments/distill_v0/` (pilot), `direction/experiments/card_student_v0/` (public Card + paired comparison) |
| Gate certification run | `direction/experiments/gate_cert_v0/` (`plan.json` committed before scoring, `results.json`) |
| Workflow harness, results | `keel/workflows/`, `direction/experiments/workflow_v0/` |
| Earlier probes | `direction/experiments/{reasoning_headroom,steering_probe,channel_probe}/` |
| Runtime | `keel/runtime/` (`python -X utf8 -m keel.runtime.smoke_judge --n 4`) |
| Interface contract | `keel/CONTRACT.md` |

Tests: `python -X utf8 -m pytest -q keel/tests`. Data rules:
- public JevBench items are evaluation-only;
- per-item outputs and dataset text stay outside git;
- training data passes `jevbench_guard` and `attack_guard`.
