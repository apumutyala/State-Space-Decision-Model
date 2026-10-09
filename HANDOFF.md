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
DRAFT v1 (agent-a, 2026-10-09 ~21:00Z). PROVISIONAL until (i) the operator answers the two open questions below, (ii) agent-a's JevBench/MD1 spec pin lands, (iii) CONTROL_RESEARCH.md lands.

**Role.** You are the lead for training, architecture and control-of-learning on the State-Space Decision Model project (repo apumutyala/State-Space-Decision-Model, private; branch per variant, agent-main = clean best; state branch = BOARD/ledger/PROGRESS/DEAD_ENDS). You own what gets trained and how training is controlled. Agent 2 ("evidence") owns the eval kit, data, safety and claims, and reviews every win you claim.

**Objective.** Build a state-space decision model that beats (a) the JevBench v1.6.1 open-weights leader on the OFFICIAL score (equal-weight harmonic mean of Intelligence, Calibration, Speed, Cost; currently 72.5, H2O-Lightning-4B) and (b) Microsoft-Decision-1 on its published axes that we can reproduce (accuracy on the reproducible part of its 36-benchmark protocol, calibration, flip rate under its 8 perturbation types, safety, p50/p95 latency, cost). Use the architecture's structure to CONTROL learning, not just monitor it, with stated guarantees where they are provable.

**What the score rewards (design from this).** The harmonic mean is dominated by the weakest axis. The leader is a 4B model at I60/C90/S93/Cost60. Speed is log-scale on p50 (100 ≈ 100 ms; 20 points per 10×). Small, fast, cheap and well calibrated beats big and accurate. An SSM's constant memory and linear prefill should be won on Speed and Cost, but only if Intelligence stays ≥ ~60.

**Evidence you inherit (receipts on the state ledger).**
- RWKV7-13.3B best trained cell: D13 (attention-free joint record head) 41.07 on the Index proxy600 screen. Clef-flash scores 59.03 and Clef 27B 64.56 on the same rows. RWKV7's projected ceiling is ~45–50.
- Stock Qwen3.6-35B-A3B (hybrid, 10/40 attention layers, ~3B active) scores 43.56 untrained, above our best trained attention-free cell. Stock Qwen3.5-9B scores 35.82.
- Strict attention-free conversion of Qwen3.5-9B (8 attention layers → GDN, 196M distillation tokens) collapsed: 35.82 → 14.05. KL looked fine while recall died (MQAR −59%, NIAH −53%). The attention layers carry recall. A recall-targeted v3 conversion is being built (claude/b-q9af-v3).
- What worked on RWKV7: v3w readout (+3.6), Clef-style LS-CE+Brier+RPS loss, data coverage (+9 from 100k targeted rows), the joint record head (+3.4 screen), the anti-collapse guard (cosine LR + dev-card collapse guard + best-checkpoint). What failed: distillation from Clef-flash (−0.5), RLCD without a temperature fix (ECE 0.03 → 0.16), over-training past one epoch at constant LR.
- Data levers (B): missing families ~7.1 pts, untrained formats ~4.3 pts; bschema v4 is being built.
- Platform: SF Compute H100, ≤ 899 min per job. Chained multi-job training is being built (claude/chained-cells PR). Disk on the laptop is tight; delete own reproducible files.

**Open operator questions (do not decide these yourself).** (1) Is "state-space" strictly attention-free, or are hybrid SSMs (a few attention layers) allowed? (2) Is MD1 to be beaten on its own 36-bench protocol (only partly public) or via a pre-registered proxy?

**Control-of-learning program (your differentiator; start with what has guarantees).**
1. Invariances by construction: an option-permutation-equivariant head gives 0 flips on option reorder/reverse/shuffle as a theorem, not a measurement. Paraphrase invariance via consistency regularisation, measured with MD1's 8 perturbation types.
2. Calibration with guarantees: post-hoc temperature on non-eval rows, plus split-conformal prediction sets / conformal risk control for abstention. These give distribution-free coverage guarantees per family. Report the guarantee and the empirical coverage.
3. Constrained training as control: treat ECE, flip rate and per-family regression bounds as constraints solved with primal-dual (Lagrangian) updates, i.e. a feedback controller on loss weights. This generalises the existing collapse guard (a bang-bang controller) to PID/Lagrangian control with stated setpoints.
4. Stability of the state: bound the SSM transition spectrum (decay gates in (0,1), spectral-norm constraints) for Lyapunov-style stability of the recurrent state, and use state-space observability/controllability analysis of the trained readout for interpretability claims.
Source of truth for this program: ssdm-wt/b-landscape/CONTROL_RESEARCH.md (11 ranked mechanisms, 54 sources; builds on the existing controlsafe/ G4/G15/G16 guards). Highest-value item there is NOVEL: injection-robust conformal sets. The G16 protected judge bounds each option's score to a trusted-only interval, so the worst-case prediction set is computable from trusted text alone and (by G15) identical for any untrusted text, i.e. coverage holds under arbitrary injection given correct trust labels. Use νPI (PI-controlled Lagrange multipliers) for item 3, including a RECALL constraint for any attention-to-SSM conversion.
5. Each control mechanism ships with an ablation (on/off, paired) and a written statement of what is guaranteed and under which assumptions.

**First 48 hours.**
(a) Land in-flight: D16, D18, D19, the D13 full-DEV merge, Q9-AF v3 probe.
(b) With agent 2's kit, measure stock candidates on the JevBench-style axes (Intelligence proxy, Calibration, p50 latency, $/1k) on our own hardware path: RWKV7-13.3B, RWKV7 smaller G1 sizes, Qwen3.5-4B/9B, Qwen3.6-35B-A3B, gemma-4-12B (the board is dominated by gemma-4 fine-tunes). Pick the host by projected harmonic-mean score, not Intelligence alone.
(c) Port the best RWKV7 recipe (v3w + LS-CE/Brier/RPS + joint head + anti-collapse + bschema data) to the chosen host.
(d) Implement control items 1 and 2 first (cheapest provable guarantees).

**Rules.** Everything from "Claim rules" above; pre-register on BOARD; paired CIs; independent rescore by agent 2 before any claim; never train on TEST, held-out or blind sets; never create top-ups or payment links; keep GPUs busy, pipeline the next job while one runs; on a permission denial stop that outcome and report.

### NEW AGENT 2: "evidence": eval kit, data, safety/controllability guarantees, claims
(agent-b drafts after the research lands)

## Goals to set for them (/goal text)
(drafted with the prompts)

## In-flight jobs at handoff
D13 full DEV shards job-3z2m7 + job-hc6wp (merge with fulldev_eval merge, then analysis/land_fulldev.py); D13 safety job-4yedh; heldout_v1 Clef-27B job-gmbqi (done), Clef-flash (resubmitting); RAGTruth polarity (resubmitting); D18 job-i8mcp, D19 job-fi5xf (agent-a). Builders: bschema-draw-v4 (claude/b-schema-v4), Q9-AF v3 (claude/b-q9af-v3).
