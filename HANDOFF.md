# HANDOFF: the MD1 / JevBench objective (DRAFT, agent-b 2026-10-09; agent-a edits its lane; finalised before the operator starts the new agents)

Status: DRAFT. Research inputs pending: ssdm-wt/b-landscape/LANDSCAPE.md (JevBench + MD1 + competitors), ssdm-wt/b-landscape/CONTROL_RESEARCH.md (control-theoretic / mechanistic controllability SOTA). Harness evolution (eval kit for JevBench axes + MD1 battery, GOAL/AGENTS rewrite): agent-a, via PRs.

## The new objective (operator, 2026-10-09)
Outperform Microsoft-Decision-1 (MD1) and the best system on JevBench with OUR methodology: state-space-based decision models whose structure gives more controllability / interpretability, and use dynamical-systems theory, constrained state spaces and classical + modern control to CONTROL learning (sculpt it during training), with formal or threshold guarantees where achievable, and show where that is more robust / safer / more valuable than Jev, Clef, classifier guardrails and LLM decision models.

## Targets (to be pinned exactly by the harness work)
- MD1 (2026-10-09; Qwen3.5-9B post-trained, single pass, calibrated per-option probabilities): 83.5% avg on its own 36 blind benchmarks (147,137 q); p50 85 ms / p95 125 ms; calibration 92.2; 1.3% flip rate over 8 perturbation types (0 on option reorder/paraphrase); safety 5,250 requests / 11 benchmarks; $0.042 per M input tokens. Its full benchmark list/protocol is NOT fully public: the kit reproduces whatever is.
- JevBench v1.6.1 open-weights board: official Score = equal-weight harmonic mean of Intelligence, Calibration, Speed, Cost (0–100). #1 H2O-Lightning-4B v1.1 (Qwen3.5-4B FT) 72.5; Jev 1.13.0 reference 71.5; Capability leader Quyet-1.0-Large 81.7 (Capability = mean(Intel, Calib)). Jev-class caps: ≤ $0.065 / 1k decisions, p50 ≤ 1.23 s.
- Carry-over evidence: RWKV7 best D13 = 41.07 proxy600 (Index 0.2.1) vs Clef-flash 59.03; stock Qwen3.6-35B-A3B (10/40 attention) 43.56; Q9-AF v2 strict conversion collapsed recall (35.82 → 14.05). Open question for the operator: strict attention-free vs hybrid-SSM.

## Claim rules (carry over, adapt)
Pre-register on the state BOARD before results; paired CIs; readout parity; train-split-only targets; leak audits (doc_overlap_guard, attack_guard, Fingerprint); frozen scorers; independent rescore of every keep; no TEST / held-out / cohort-v3 training; open-weight teachers only; board submissions only with operator approval; never create top-ups or payment links.

## Agent prompts (two new sessions)
### NEW AGENT 1: "lead": training + architecture + control-of-learning
(agent-a drafts this section)

### NEW AGENT 2: "evidence": eval kit, data, safety/controllability guarantees, claims
(agent-b drafts after the research lands)

## Goals to set for them (/goal text)
(drafted with the prompts)

## In-flight jobs at handoff
D13 full DEV shards job-3z2m7 + job-hc6wp (merge with fulldev_eval merge, then analysis/land_fulldev.py); D13 safety job-4yedh; heldout_v1 Clef-27B job-gmbqi (done), Clef-flash (resubmitting); RAGTruth polarity (resubmitting); D18 job-i8mcp, D19 job-fi5xf (agent-a). Builders: bschema-draw-v4 (claude/b-schema-v4), Q9-AF v3 (claude/b-q9af-v3).
