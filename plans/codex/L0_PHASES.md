# Track L phases and spend ladder

6 October 2026. This phase map governs work after the initial L0 drafts and audit handoffs. It is written before resuming implementation under the user's per-task worktree rule. Phase proofs are requirements until their receipts and cross-review are complete; this document does not claim they have passed.

## Phase list

| Phase | Hypothesis | Deliverable | Exit proof | Decision unlocked |
| --- | --- | --- | --- | --- |
| L0a Interface freeze | A small typed interface can separate proposal, authority, proof and realization without requiring a hidden-state format | Canonical 0.1 contract, parent handshake, this phase map | Interface PR cross-reviewed by Claude and merged on agent-main; remaining unsupported semantics explicitly listed | Stabilize implementation APIs and fixture meanings |
| L0b Semantic kernel | Exact typed operations can reject unsupported relations and unauthorized dependencies without changing the model's proposed answer | Immutable semantic types, interpreter, release validator | Unit and adversarial tests, actual fixture conformance, CPU receipt bound to source and data hashes; reviewed kernel PR | Supply checked meanings to a renderer |
| L0c Deterministic language | Licensed productions preserve checked entities, relations, units, time and speech acts | Plain-text renderer and grammar fixtures | Positive/negative binding, hostile-string, stale-release and no-bypass tests; receipt and reviewed renderer PR | Render useful L0 answers; add a production only when its semantics is tested |
| L0d Oracle and replay | A deterministic provider can exercise the adapter without masquerading as learned capability | Oracle/replay backends, strict complete-request identity and typed probability serialization | Noul/choice/score parity, option support, stage separation and changed-request rejection tests; reviewed backend PR | Integrate Track D's m0 replay fixture and decide adapter |
| L0d2 Typed planner | The same parent can select operations and arguments through typed choices without a host oracle selecting the task answer | Bounded two-stage operation/argument planner and a Track L serialization bridge | CPU fake/oracle/replay traces show actual chosen slots, no gold-label inputs, complete call accounting, typed failures and no unchecked prose; later D fixture compatibility | Run actual parent semantic planning once qualification is permitted |
| L0e Joint fixtures and measurement | Kernel, provider and task errors can be distinguished in per-case evidence | Versioned fixture pack, evaluator, decision-only mappings and raw reports | Independently recomputed observed/expected outcomes, stable hashes, explicit authored-label status; reviewed fixture/evaluator PR | Freeze the CPU conformance baseline and a separate future model-audit split |
| L0f Adversarial integration | The composed implementation preserves its stated boundary under malformed programs and source/state changes | Independent Track L review and regression cases | Reviewer replays all CPU commands; no unresolved blocking finding; tests and receipts merged on agent-main | Propose hardware qualification; no learned claim yet |
| L0g Hardware qualification | The actual parent adapter loads and produces the pinned protocol within a measured budget | One real-hardware smoke run, memory/latency and parent revision receipt | All prerequisite CPU proofs merged; Claude acknowledges a spend issue with estimate <= $20 and stop condition; planned ledger row; qualified run receipt later reviewed/merged | Price and define one capped learned-language experiment |
| L0h Learned checked explanations | The shared parent can discover useful short programs and explain decisions, beyond oracle/replay behavior | Frozen-parent or bounded-adaptation pilot, strong unrestricted comparator and same-checker-budget controls | Qualification proof merged; separately acknowledged spend cap; all four level gates below with receipts and paired confidence intervals | Promote L0 only if its full gate passes; otherwise diagnose/fold the failing line |
| L1a Multi-record exact answers | Bound facts can compose through exact calculations without loss of useful capability or decision competence | Typed arithmetic/comparison module, multi-record answers and faithful productions | Prior level's proof merged before paid progression; all four gates on fresh L1 tasks and identical retention items | Promote L1 or repair expression, program discovery or realization |
| L2a Compositional reasoning | Checked inference, assumptions, questions and plans can expand useful reasoning without an unchecked output path | Bounded inference/assumption/speech-act/plan modules with explicit semantics | All four level gates, plan-effect simulation scope, independent recomputation and cross-review | Promote L2 or fold the failing rule/search strategy |
| L3a Rich licensed discourse | Licensed paraphrase, coreference and discourse choices preserve complete meaning while matching strong unrestricted usefulness | Richer grammar/realizer choices with semantic dependencies | All four level gates, whole-answer fidelity and omission checks, no free-connective bypass | Promote L3 or expand/fold the inadequate semantic/realizer representation |
| L4a New semantic domains | A new domain module transfers the learned planning method while preserving previous guarantees and competence | New ontology/module contracts, data and language productions | All four level gates in the new domain and earlier retention sets; cross-reviewed receipts and user-approved merge to main | Completion audit against GOAL.md, never completion from boundary tests alone |
| Scale-up gate | A surviving line has a measured scaling reason rather than only a larger budget | Explicit model/data/compute scale proposal | Previous rung merged; Claude-acknowledged spend issue with cost/stop condition and remaining shared balance | Scale only that surviving line |

## Spend ladder

At every semantic level L0 through L4, the standing GOAL.md requires useful-answer capability at least matching a strong unrestricted comparator, no regression on identical parent-decision items, relation-level soundness including wrong-question cases, and causal semantic-intervention fidelity beating random and output-only controls. Each needs receipts and paired confidence intervals, with the actual intervention port and scope stated. L0a–L0f are engineering prerequisites and do not satisfy those learned-level gates. Do not relabel input edits or deterministic-renderer tests as neural causal mediation. Final completion additionally requires the user-approved main merge; this map grants no authority to push or merge main.

The order is **$0 local CPU -> <= $20 real-hardware qualification -> capped experiment -> scale-up**. Climb a rung only after the preceding rung's proof is merged on agent-main and Claude has acknowledged a spend issue containing the estimate and stop condition. Record every planned paid job in LEDGER.md before launch. The joint $75 setup issue does not automatically authorize every later job or an external API bill.

The shared envelope is $75 joint gate / $250 System 1 / $100 language / $100 reserve. Qualification counts within the agreed ledger allocation; it is not an extra $20. No paid compute, purchase, top-up or GPU model run occurs in the CPU prerequisites L0a–L0f. L0g and later paid work require the spend gates above.

## Parallel tasks and worktrees

All worktrees use the exact path prefix `C:/Users/apumu/research/CHLabs/`:

- `ssdm-codex-l0`, branch `codex/l0-semantic-kernel`: types/interpreter and kernel tests.
- `ssdm-codex-l0-renderer`, branch `codex/l0-renderer`: deterministic renderer/grammar and tests.
- `ssdm-codex-l0-backends`, branch `codex/l0-backends`: oracle/replay adapter and tests.
- `ssdm-codex-l0-planner`, branch `codex/l0-planner`: typed parent operation/argument selection and transport bridge.
- `ssdm-codex-l0-fixtures`, branch `codex/l0-fixtures`: fixture pack, evaluator and independent recomputation.
- `ssdm-codex-l0-evaluation`, branch `codex/l0-evaluation`: observed runtime evaluation after the fixture pack is frozen.
- `ssdm-codex-l0-adversarial`, branch `codex/l0-adversarial`: independent review/regression tests only.

Start each task branch from agent-main. If it needs an unmerged Track L dependency, record that dependency explicitly, keep its commits distinct, and rebase on agent-main before its PR is finalized. Never switch Claude's working checkout or broadly stage another worktree's drafts. Each task has its own branch; a single agent may service independent branches sequentially when concurrency slots are full.

While one piece is in review, build another CPU piece or its oracle fixture. Track D's `claude/m0-decide-adapter` is the first external integration priority. Review its replay/serialization/probability boundary as soon as its PR lands, ahead of less urgent harness, steering and conversion PRs. Do not block L0 on GPU availability or D's selected trained parent.

## Review and fold rules

Before a cross-agent PR, perform an adversarial Track L review. A result PR includes source SHA, exact command/config, fixture/data hashes, seed status, CPU/hardware, cost and raw outputs; Claude independently re-derives reported results before merge. Apply the same standard when reviewing Track D results.

For each experimental line, define its hypothesis and next two informative checks. Fold it if falsified, outranked under matched conditions, or unchanged across both checks. Add a LEDGER.md negative-result row with the receipts, reason and any replacement. A setup failure is not automatically a scientific falsification; fix it once in the smallest useful scope, then reassess. No line has been declared folded solely by writing this plan.

Keep text/contract conformance, semantic program validity, task correctness and learned interpretation as separate conclusions. Passing L0 cannot be reported as a safe or aligned learned language model.
