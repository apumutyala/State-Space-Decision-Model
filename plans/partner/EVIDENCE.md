# Evidence for the decision first SSM plans

Research checked 5 October 2026 in the user's America/Los_Angeles date context. Primary literature/product sources support the claims below; architecture choices in PLAN.md are proposed designs. No new training, external benchmark run, endpoint call, paid compute allocation, or leaderboard submission was performed.

## Supplied research

- [Original decision first report](<C:/Users/apumu/.codex/attachments/c8a5e634-290a-4418-bd72-30f53ce7fe27/Pasted text.txt>): typed decision interface, diagonal/selective SSM ladder, semantic bottleneck, calibration/controller proposal, verified emission, LLC/Hankel hypothesis, and staged language growth.
- [Follow-up critique and budget discussion](<C:/Users/apumu/.codex/attachments/64920a61-7367-4760-9e81-6b9a91d35614/Pasted text.txt>): benchmark/scale objections, Bayes-integrator and truncation sketches, pretrained-backbone proposal, and a historical $5,000 research allowance.

Both were read. The new plan retains the first report's decision-to-language lineage while correcting unsupported mathematics. The $5,000 amount is used for planning, not as a new spend event.

## Local Codex logs read directly

The user clarified that prior research should be recovered from saved device logs. These files were parsed directly, with relevant final research messages and artifact references inspected:

- [Assess SSM control research log](C:/Users/apumu/.codex/sessions/2026/09/18/rollout-2026-09-18T01-39-02-01a0b306-7fe9-7dc1-8710-0da460fe236c.jsonl): knowledge-bank synthesis and limits of deterministic alignment, global controllability, semantic guarantees, and universal developmental claims.
- [Assess AI alignment research thesis log](C:/Users/apumu/.codex/sessions/2026/09/22/rollout-2026-09-22T16-08-01-01a0cabb-8236-70c2-8f51-bcc2f7408c49.jsonl): latest campaign completion and October 1 evidence location.
- [Prior adversarial reviewer log](C:/Users/apumu/.codex/sessions/2026/09/21/rollout-2026-09-21T22-02-30-01a0c6d9-af17-7dc3-89d7-f6d93bb2f588.jsonl): final review of state sufficiency, Jacobian overreach, CBF assumptions, and adaptive evaluation risk. Inherited fork-history messages were distinguished from the reviewer's own final report.
- [Native evaluator log](C:/Users/apumu/.codex/sessions/2026/09/28/rollout-2026-09-28T15-53-22-01a0e994-40d2-7d10-ae12-477c57178f5d.jsonl): synthetic data freeze, trained qualification, and executor simulation scope.

The logs locate evidence; their earlier plans, authorizations and summaries do not turn unexecuted work into results.

## Relevant local artifacts

| Artifact | Used conclusion |
| --- | --- |
| [Latest pilot results](../../overnight/LATEST_PILOT_RESULTS_20261001.md) | 21 Qwen retrains and 12 scratch models completed; useful-control gates failed; source/action boundaries held within scope |
| [Research reassessment](../../overnight/reassessment_20261001/NEXT_EXPERIMENTS_AFTER_SESSION_REVIEW.md) | Useful semantics and wrong-permitted-action behavior matter more than another legality demonstration |
| [Executed model research](../../chl-structured-models/docs/EXECUTED_RESEARCH_STATUS.md) | Existing registers and control ports are reusable; state correction/inheritance and composition failures are preserved |
| [Primary-source audit](../../chl-structured-models/docs/RESEARCH_SOURCE_AUDIT.md) | Prior art and corrected claims; pointers were treated as leads rather than blanket verification of every linked theorem |
| [White Box Labs memo](../../white-box-labs/WBL_1_Memo.md) | Separate runtime/continual-learning proposal, not the project requested here |
| [Bounded intelligence status](../../bounded-intelligence/docs/STATUS.md) | A scaffold/status note, not evidence of a trained controlled foundation model |

The October 1 study is a separate population from the September 28 BANKING77 and dialogue results. Their scores must not be pooled or directly compared to competitor zero-shot scores.

## Cloudflare and TypeSafe

[Cloudflare's full article](https://blog.cloudflare.com/clef-decision-models/) describes two open releases, a prefill-only Qwen-based decision system with attention routing, calibration-oriented post-training, multimodal input, and a proposed fine-tuning service. The serving workflow matters: there is already no autoregressive decision-text phase to remove. Its reported numbers are vendor measurements. The article's 64k versus Jev's 32k context claim is not evidence that either model accurately uses every token.

[Clef model card](https://huggingface.co/Cloudflare/clef) and [Clef-flash model card](https://huggingface.co/Cloudflare/clef-flash) identify 27B and 9B backbones and custom joint heads. Illustrative reported results:

| Vendor reported metric | Clef | Clef-flash | Jev |
| --- | ---: | ---: | ---: |
| BANKING77 macro-F1 | 94.2 | 90.9 | 79.7 |
| CLINC150 plus OOS macro-F1 | 97.4 | 66.8 | 89.3 |
| GPQA Diamond accuracy | 48.0 | 51.0 | 78.3 |
| Median request latency in ms | 209.3 | 38.8 | 524.1 |

These are orientation points, not matched measurements from this task. The varying task winners are a reason to keep the full target, not cherry-pick the smallest gap.

[Released head and serializer code](https://huggingface.co/Cloudflare/clef/raw/main/joint_schema_model.py) was inspected. The score is the expected raw level index; confidence is max class probability. The helper defaults to a 16,384-token maximum. Adapter limits therefore need explicit checking against claimed product context. Do not rely on automatically generated generic Hugging Face pipeline snippets for this custom model.

[TypeSafe's launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev) ties its zero-hallucination chart to guaranteed schema matching. Its workflow evaluation uses frontier-model reference probabilities, and it discloses benchmark and latency caveats. That supports a typed-output contract, not a guarantee that a chosen answer is true. The low-level model architecture and a reproducible RLCD algorithm were not established by this article. Do not classify Jev as a known transformer or a known PGM on this evidence.

The main text of both articles was read through the web extraction, including their limitations and deployment/service sections. A direct HTML read confirmed the Jev FAQ text exposed in the source. The browser inspection tool could not start; additional collapsed FAQ answers or video/demo contents were not independently inspected. The plans do not rely on unseen media or FAQ answers.

## Benchmark protocol

[Decision Index reproduction kit](https://github.com/apolinario/decision-index) is the primary benchmark source. At inspection, edition 0.2.1 counts 38 benchmarks with revised area/gold weights; 0.2's 40 equal-area formulation is different. The kit specifies no truncation, no option filtering, fixed rendering and penalties for unanswered requests. This corrects the pasted report's protocol/count assumptions. Use its pinned data hashes and scorer; do not copy old headline totals between editions.

Astra and Fable are added as direct baselines by the user's clarified target. Pin exact model identifiers, reasoning settings, dates, wrappers and budgets before execution. The source article's historical reference configurations do not define their current best operating point. Report factual ground truth separately from agreement with either reference model.

## Technical sources informing design choices

| Primary source | What it supports and does not support |
| --- | --- |
| [Mamba-3 paper](https://arxiv.org/html/2603.15569v1) | Selective recurrence, complex state, MIMO, capability/efficiency experiments; no general alignment theorem and no evidence of beating the requested decision baselines |
| [Official Mamba repository](https://github.com/state-spaces/mamba) | Available Mamba-3 code and enumerated Mamba-1/2 pretrained checkpoints; audit pure configurations rather than assuming every family member excludes attention |
| [Mamba-2 2.7B](https://huggingface.co/state-spaces/mamba2-2.7b) | Available pretrained SSM competence reference |
| [Falcon-Mamba 7B](https://huggingface.co/tiiuae/falcon-mamba-7b) | Larger pure-SSM reference with its own license; not a pretrained glass-box/alignment guarantee |
| [Concept Bottleneck LLMs](https://arxiv.org/abs/2412.07992) | Precedent for intrinsic concept interfaces in classification/generation; concept sufficiency still needs testing |
| [Concept leakage](https://openreview.net/forum?id=tglniD_fn9) | Predicted concept probabilities can carry unintended information and impair interventions |
| [Monet](https://arxiv.org/abs/2412.04139) | Sparse dictionaries integrated into pretraining; a candidate ingredient, not proof of universally monosemantic state |
| [Learn then Test](https://arxiv.org/abs/2110.01052) | Statistical risk calibration with explicit assumptions; no blanket adaptive-shift guarantee |
| [Decision Hijacking](https://arxiv.org/abs/2609.28613) | Direct Jev study showing allowed typed outputs still admit injection-related probability shifts; its small attack result is scoped to its setup |
| [CaMeL](https://arxiv.org/abs/2503.18813) | A relevant separation-of-authority/data baseline; permission enforcement and useful semantic decisions remain distinct |
| [Mamba in the Llama](https://arxiv.org/abs/2408.15237) | Evidence that conversion/distillation can preserve some capabilities; strongest cited result retains attention, so it is not a pure-SSM victory |

No claim of being the first attention-free entrant, the first LLC study on SSMs, or the first interpretable-by-design model is made. Those would require a dedicated novelty search and a precisely delimited claim.
