# direction/ — strategy work for SSDM (collab/mottopanikeiku)

| Path | What it is |
|---|---|
| `PROPOSITION.md` | **Current proposition (v1):** structure-first Keel — design, safety implementations, results, implications, plan, asks. Supersedes `MOONSHOT.md` where they differ |
| `MOONSHOT.md` | Keel v2 moonshot plan (targets, bets, phases, gates) that the implementation started from |
| `BRANCHES.md` | Triage of the 373 repo branches: what Keel builds on, what is reference, what is history |
| `GRANTS.md` | Is SSDM a good direction for AI-safety grants? Verdict, reframe, program, where to apply, next steps |
| `DIRECTION_MEMO.md` | Current direction memo (v1; git history holds v0 and the review-driven revision) |
| `research/ssdm-context.md` | Shared project facts given to every research workstream |
| `research/ssdm-W1-landscape.md`, `ssdm-W1-catalog.json` | Competitors, benchmarks, market (121-system catalog) |
| `research/ssdm-W2-capability.md`, `ssdm-W2-levers.json` | Capability levers; why SSDM's training data did not transfer |
| `research/ssdm-W3-alignment.md` | What "far more aligned" means; threat model; audit of SSDM's guarantee assets |
| `research/ssdm-W4-architecture.md` | Backbones, JevBench arithmetic, compute economics |
| `research/ssdm-W5-postmortem.md` | Spend/time accounting, failure taxonomy, asset inventory |
| `research/memo-review.md` | Independent adversarial review of memo v0 (recomputed every experimental number) |
| `research/ssdm-G1-funders.md` | AI-safety funding programmes open or opening, fit matrix, best routes |
| `research/ssdm-G2-safetyfit.md` | Safety relevance, novelty check, capabilities externality, fundable framings |
| `experiments/reasoning_headroom/` | Think-vs-one-pass probe on public-231 (code, jevkit reports, README with pins) |
| `experiments/steering_probe/` | Planted-note steering probe: one-pass vs think, Qwen3.5-4B/9B (code, results, README) |
| `experiments/channel_probe/` | Trusted vs untrusted channel discrimination with and without a trust policy |
| `experiments/card_v0/` | Decision Alignment Card v0/v0.1 on stock Qwen3.5-4B (+ 9B reference): static, transfer, adaptive, label-only tiers; frozen gate layer |
| `experiments/distill_v0/` | Discernment distillation pilot (synthetic held-out groups) |
| `experiments/card_student_v0/` | Public Card on the distilled student and paired comparison with stock/teacher arms |
| `experiments/gate_cert_v0/` | Preregistered complete-gate certification on a non-public population |
| `experiments/workflow_v0/` | AgentDojo banking + Slack: executed injection targets with and without Keel gates |

Conventions: the public JevBench items are EVAL-ONLY and never committed; per-item model outputs stay out of git
(workstation copy: `$SSDM_PRIVATE`, default `~/ssdm/private`). Paths such as `/tmp/...` or `local://...` inside
research reports are provenance from the session that produced them. On 2026-10-15 the workstation files moved into
the `~/ssdm` workspace: `/tmp/State-Space-Decision-Model` → `~/ssdm/repo`, `/home/alp/ssdm-direction` →
`~/ssdm/private`, `/home/alp/ssdm-data` → `~/ssdm/data`, `/tmp/ssdm-scratch` → `~/ssdm/scratch`,
`/home/alp/keel-wt/<slice>` → `~/ssdm/worktrees/keel-<slice>`.
