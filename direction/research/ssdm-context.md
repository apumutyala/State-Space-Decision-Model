# SSDM shared context for research agents (chief of staff, 2026-10-14)

## What the project is
State-Space-Decision-Model (SSDM; private repo apumutyala/State-Space-Decision-Model) builds an open **"System One" decision model**: input = a state (text) + typed questions (`noul` yes/no, `choice` over named options with descriptions, `score` on an ordinal scale); output = a calibrated probability for every allowed option, single prefill pass, no generated text. Same API category as TypeSafe AI's **Jev** (launched 2026-09-15), Cloudflare's **Clef / Clef-flash** (2026-10-01; Qwen3.8-27B / Qwen3.5-9B frozen + LoRA r256 + ~128M joint routing head, LS-CE + Brier, "RLCD"), **Microsoft-Decision-1 (MD1)** (2026-10-09; Qwen3.5-9B post-trained), H2O-Lightning-4B v1.1 (Qwen3.5-4B FT, Apache-2.0), Perplexity Decider, Nimble-9B, Quyet, Sage, decider-12b, torchcast-12b, LFM d1, Laya, Kev-9B and others.

SSDM's thesis: use **state-space (SSM) architectures** and **control theory to control learning**, with stated guarantees (exact reorder/relabel invariance by construction, conformal coverage + certified abstention, injection-robust conformal sets via a trust gate G15 and protected judge G16, state-stability bounds, νPI-constrained training). Track L (Codex, CPU-only) plans to grow a checked language model from the decision model; it is secondary.

## Targets (operator, 2026-10-09; binding text in repo GOAL.md / TARGETS.tsv)
1. JevBench v1.6.1 **Composite** (Benchmark Heaven; github.com/fstandhartinger/jevbench): beat open-weights leader H2O-Lightning-4B v1.1 = 72.5 (API leader Sage 1.3.0 = 74.0; Jev 1.13.0 ref 71.5). Composite = 4/(1/I+1/C+1/S+1/K) × min(1,I/50)² × min(1,S/50)² × min(1,K/50)². Speed S = 80 − 10·log10(p50) − 10·log10(p95), adjusted latency (self-hosted ×2 + 0.15 s). Cost K = 100 − 30·log10(USD per 1k decisions / 0.001), USD = input tokens × declared price class (parameters/tariff, NOT FLOPs). Board = 300 open + 1,200 sealed items; only 231 older public items are downloadable.
2. JevBench Capability = mean(I, C): beat Quyet-1.0-Large 81.7.
3. MD1's reported numbers: 83.5% avg on 36 private benchmarks, calibration 92.2, p50 85 ms (Foundry), 1.3% flip rate over 8 perturbations, $0.042/M input tokens.
4. Decision Index (multimodalart HF space "jev-decision-index"): beat Clef-flash and Clef 27B with paired CI > 0. Edition 0.3 board (2026-10-06): Perplexity Decider v1.1 62.75 (#1), Jev 60.11, Nimble-9B-v3 54.67, Clef 53.08, Clef-flash 47.61 (Full = 20% public + 50% private same-skill + 30% private new-domain).

## Where it stands (2026-10-13; all receipts in the repo)
- **Budget:** $785 of SF Compute H100 credit granted; ~$11.67 left; GPU work stopped 2026-10-12. Spend ~ $30 (10-07), $87 (10-08), $337 (10-09), $181 (10-10), $130 (10-11), $106 (10-12).
- **Every capability target is FALSIFIED / not met.**
  - Decision Index 0.2.1 full DEV: D19 (RWKV7-13.3B attention-free + LoRA + joint record head, our best trained) 49.49 vs Clef-flash 57.29 (−7.8) and Clef 27B 61.89 (−12.4). Stock Qwen3.5-9B ~38; Clef-flash's +17.5 over stock comes entirely from post-training a frozen backbone.
  - JevBench local (public-231, paired vs H2O): best of ours = untrained stock Qwen3.5-4B + H2O's prompt shim, ΔComposite −9.4 [−23.1, −4.2]; D19 −14.4; our trained 4B cells score BELOW the untrained stock 4B (JEV1 −4.1, JEV3-SYM −36.7 from noul hedging + real score/routing losses); H2O + our data = flat (−0.4, −1.0); JEV3-LM-S (answer-slot readout on human-labelled data) −0.78 n.s. with exact invariance. Gap is mostly Intelligence (50–67% by Shapley), concentrated in yes/no, temporal_numeric, judge_hard, long_policy, multi_hop, hard score items (~10 I pts each). Our TRAIN-split + rule-engine data raised in-distribution slice accuracy but did not transfer to public items (transfer corr +0.15, −0.32, −0.50).
  - Strict attention-free conversion of Qwen3.5-9B (8 attention layers → GDN, 196M distill tokens) collapsed recall: 35.8 → 14.1 Index (KL looked fine).
  - KD from Clef-flash: no gain. Constant LR past 1 epoch: collapse. ≤2B hosts: I ratio 2B/4B 0.68.
- **Proven:** jevkit reproduces the published JevBench v1.6.1 board arithmetic exactly and 3 open-weights rows on public-231; exact reorder/relabel invariance by construction (sym-fork; 0/565 flips on GPU); conformal coverage; injection-robust conformal construction (CPU, real BPE 1000/1000); leak-free data gates; independent rescores. Held-out attacks: D13 more robust than Clef-flash (hijack 0.025 vs 0.115; steer 0.003 vs 0.089, Holm-significant).
- **Falsified/open guarantees:** LTT certified abstention certifies nothing on our stacks (and on any model under cluster-valid tests on DEV); RWKV-7 stability certificate fails in bf16; νPI controller and trained G16 head never run.
- **Process facts:** 3 agent lanes (lead, evidence, Codex Track L), ~370 branches, ~250 PRs, ~700 CPU tests, heavy pre-registration + independent rescore of every claim; ~50 LESSONS entries, mostly infrastructure (OOMs, timeouts, kernel numerics, caps, CRLF, artifact limits); custom SF Compute job runner with ≤899-min jobs; single seeds; n=231 public items gives ~±8 Composite CI half-width.

## Operator constraints (binding unless the user changes them)
- Main line hybrid SSM (mostly SSM layers + a few attention layers); research line strictly attention-free.
- Operator directive 2026-10-11: build/engineer OUR custom stacks from state-space/dynamics-based architectures with control/safety guarantees; do not benchmark or engineer third-party models further (display existing numbers only).
- Never train/tune on test, held-out, sealed rows or JevBench/Index eval items. Open-weight teachers only (never Jev or frontier APIs). Nothing public / no board submission without user approval. No AI attribution in commits.

## The new user brief (2026-10-14)
The user (new ruling chief of staff's principal) wants the project **better, faster, cheaper, and far more aligned**. Phase now: opine, research, form pathways before committing.

## Local paths (read-only for you; never commit, push or modify)
- `/tmp/State-Space-Decision-Model` — clone, branch `agent-main` checked out; every remote branch readable via `git -C /tmp/State-Space-Decision-Model show origin/<branch>:<path>`.
- `/tmp/ssdm-state` — `state` branch: BOARD.md, PREREG.md, PROGRESS.md, DEAD_ENDS.md, HANDOFF.md, INBOX.md, results/ledger.tsv.
- `/tmp/ssdm-docs/EVIDENCE_REPORT_V2.md` — target-by-target verdicts with receipts (2026-10-12).
- Key repo docs: GOAL.md, AGENTS.md, TARGETS.tsv, LESSONS.md, PREMORTEM.md, INDEX_0.3.md, techniques/10-clef-recipe-and-gap-attribution.md, techniques/CATALOG.md, jevkit/README.md; main-branch onboarding: `git show main:onboarding/RESEARCH_DIRECTIONS.md`, `git show main:plans/partner/PLAN.md`.

## Rules for every research output
- Primary sources first (papers, model cards, official blogs, repos, board data files); link every claim; give dates. Mark anything unverified `[UNVERIFIED]` and your own extrapolation `[INFERENCE]`.
- Today is 2026-10-14; much relevant material is from Sept–Oct 2026 and beyond any training cutoff: search, do not recall.
- No GPU jobs, no spending, no accounts, no pushes. Read-only on the repo.
- Lead with conclusions; quantify (effect sizes, costs, latency) wherever sources allow.
