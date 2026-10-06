# Track L phases and spend ladder

6 October 2026. This phase map governs work after the initial L0 drafts and audit handoffs. It is written before resuming implementation under the user's per-task worktree rule. Phase proofs are requirements until their receipts and cross-review are complete; this document does not claim they have passed.

## Phase list

| Phase | Hypothesis | Deliverable | Exit proof | Decision unlocked |
| --- | --- | --- | --- | --- |
| L0a Interface freeze | A small typed interface can separate proposal, authority, proof and realization without requiring a hidden-state format | Canonical 0.1 contract, parent handshake, this phase map | Interface PR cross-reviewed by Claude and merged on agent-main; remaining unsupported semantics explicitly listed | Stabilize implementation APIs and fixture meanings |
| L0b Semantic kernel | Exact typed operations can reject unsupported relations and unauthorized dependencies without changing the model's proposed answer | Immutable semantic types, interpreter, release validator | Unit and adversarial tests, actual fixture conformance, CPU receipt bound to source and data hashes; reviewed kernel PR | Supply checked meanings to a renderer |
| L0c Deterministic language | Licensed productions preserve checked entities, relations, units, time and speech acts | Plain-text renderer and grammar fixtures | Positive/negative binding, hostile-string, stale-release and no-bypass tests; receipt and reviewed renderer PR | Render useful L0 answers; add a production only when its semantics is tested |
| L0d Oracle and replay | A deterministic provider can exercise the adapter without masquerading as learned capability | Oracle/replay backends, strict complete-request identity and typed probability serialization | Noul/choice/score parity, option support, stage separation and changed-request rejection tests; reviewed backend PR | Integrate Track D's m0 replay fixture and decide adapter |
| L0e Joint fixtures and measurement | Kernel, provider and task errors can be distinguished in per-case evidence | Versioned fixture pack, evaluator, decision-only mappings and raw reports | Independently recomputed observed/expected outcomes, stable hashes, explicit authored-label status; reviewed fixture/evaluator PR | Freeze the CPU conformance baseline and a separate future model-audit split |
| L0f Adversarial integration | The composed implementation preserves its stated boundary under malformed programs and source/state changes | Independent Track L review and regression cases | Reviewer replays all CPU commands; no unresolved blocking finding; tests and receipts merged on agent-main | Propose hardware qualification; no learned claim yet |
| L1a Hardware qualification | The actual parent adapter loads and produces the pinned protocol within a measured budget | One real-hardware smoke run, memory/latency and parent revision receipt | All prerequisite CPU proofs merged; Claude acknowledges a spend issue with estimate <= $20 and stop condition; planned ledger row; qualified run receipt later reviewed/merged | Price and define one capped learned-language experiment |
| L1b Capped semantic planning | The shared parent can discover useful short programs, beyond oracle/replay behavior | Frozen-parent or bounded-adaptation pilot, direct-answer and same-checker-budget controls | Qualification proof merged; separate acknowledged spend issue/ledger cap; held-out complete-answer, coverage, first-attempt/repair and decision-retention results with receipts | Select one measured failure to repair or confirm on fresh composition |
| L2 One extension | A single additional semantic/discourse operation improves usefulness without weakening prior contracts or measured capability | One operator, its checker/renderer rule and targeted training/eval | Prior experiment proof merged; acknowledged capped spend issue if paid; fresh composition and old-task retention proof | Broaden the operation set or reject/fold the extension |
| L3 Scale-up | The surviving method has a measured scaling reason rather than only a larger budget | Explicit model/data/compute scale proposal | Previous rung merged; Claude-acknowledged spend issue with cost/stop condition and remaining shared balance | Scale only the surviving line |

## Spend ladder

The order is **$0 local CPU -> <= $20 real-hardware qualification -> capped experiment -> scale-up**. Climb a rung only after the preceding rung's proof is merged on agent-main and Claude has acknowledged a spend issue containing the estimate and stop condition. Record every planned paid job in LEDGER.md before launch. The joint $75 setup issue does not automatically authorize every later job or an external API bill.

The shared envelope is $75 joint gate / $250 System 1 / $100 language / $100 reserve. Qualification counts within the agreed ledger allocation; it is not an extra $20. No paid compute, purchase, top-up or GPU model run occurs in L0.

## Parallel tasks and worktrees

All worktrees use the exact path prefix `C:/Users/apumu/research/CHLabs/`:

- `ssdm-codex-l0`, branch `codex/l0-semantic-kernel`: types/interpreter and kernel tests.
- `ssdm-codex-l0-renderer`, branch `codex/l0-renderer`: deterministic renderer/grammar and tests.
- `ssdm-codex-l0-backends`, branch `codex/l0-backends`: oracle/replay adapter and tests.
- `ssdm-codex-l0-fixtures`, branch `codex/l0-fixtures`: fixture pack, evaluator and independent recomputation.
- `ssdm-codex-l0-adversarial`, branch `codex/l0-adversarial`: independent review/regression tests only.

Start each task branch from agent-main. If it needs an unmerged Track L dependency, record that dependency explicitly, keep its commits distinct, and rebase on agent-main before its PR is finalized. Never switch Claude's working checkout or broadly stage another worktree's drafts. Each task has its own branch; a single agent may service independent branches sequentially when concurrency slots are full.

While one piece is in review, build another CPU piece or its oracle fixture. Track D's `claude/m0-decide-adapter` is the first external integration priority. Review its replay/serialization/probability boundary as soon as its PR lands, ahead of less urgent harness, steering and conversion PRs. Do not block L0 on GPU availability or D's selected trained parent.

## Review and fold rules

Before a cross-agent PR, perform an adversarial Track L review. A result PR includes source SHA, exact command/config, fixture/data hashes, seed status, CPU/hardware, cost and raw outputs; Claude independently re-derives reported results before merge. Apply the same standard when reviewing Track D results.

For each experimental line, define its hypothesis and next two informative checks. Fold it if falsified, outranked under matched conditions, or unchanged across both checks. Add a LEDGER.md negative-result row with the receipts, reason and any replacement. A setup failure is not automatically a scientific falsification; fix it once in the smallest useful scope, then reassess. No line has been declared folded solely by writing this plan.

Keep text/contract conformance, semantic program validity, task correctness and learned interpretation as separate conclusions. Passing L0 cannot be reported as a safe or aligned learned language model.
