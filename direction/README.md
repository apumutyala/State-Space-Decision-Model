# direction/ — strategy work for SSDM (collab/mottopanikeiku)

| Path | What it is |
|---|---|
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
| `experiments/steering_probe/` | Planted-note steering probe (code; results added when landed) |

Conventions: the public JevBench items are EVAL-ONLY and never committed; per-item model outputs stay out of git
(workstation copy: `/home/alp/ssdm-direction/`). Paths such as `/tmp/...` or `local://...` inside research reports are
provenance from the session that produced them.
