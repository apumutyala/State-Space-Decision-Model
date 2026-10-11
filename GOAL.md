# SSDM goal (operator decisions 2026-10-09; detail in HANDOFF.md on the `state` branch)

Build state-space decision models that win on the public decision benchmarks with our methodology: the state-space structure is used to CONTROL learning, and every control comes with a stated guarantee. Work autonomously toward every target below and do not stop at an intermediate milestone. The constraints are binding and are never traded away to reach a capability target faster.

## Stage 1: decision core (targets in priority order; numbers in `TARGETS.tsv`)

1. **PRIMARY: JevBench v1.6.1 Composite.** Beat the open-weights leader (H2O-Lightning-4B v1.1, 72.5) on the official 4-axis Composite: the harmonic mean of Intelligence, Calibration, Speed and Cost, with min(·/50)² penalties on I, S and K. The API leader (Sage 1.3.0, 74.0) is reported alongside. The score rewards small, fast, cheap and well-calibrated models; design from that.
2. **Capability** = mean(I, C): beat the leader (Quyet-1.0-Large, 81.7). Pursued after the Composite.
3. **Microsoft-Decision-1's REPORTED numbers:** 83.5% average on 36 benchmarks, calibration 92.2, p50 85 ms (Foundry; compare on JevBench's adjusted scale), 1.3% average flip rate. MD1 is never re-run; we compare against its published numbers and any public benchmark it cites.
4. **Clef:** beat Clef-flash and Clef 27B on the Decision Index with a paired CI > 0, under PREREG.md B-PREREG-CLAIM (full DEV + full_excl51, the 0.3 public edition reported, G-proxy tier for any "beats Clef" statement).

**Architecture: both tracks.** The main line is a hybrid state-space model (mostly state-space layers plus a few attention layers) to win the benchmarks. The research line is strictly attention-free, for the strongest guarantees.

**Methodology: control of learning, with guarantees.** Use reorder/relabel invariance by construction; conformal coverage with abstention, including injection-robust sets; νPI-constrained training (ECE, flip rate, recall and per-family regression as constraints); and state stability bounds. Source: `ssdm-wt/b-landscape/CONTROL_RESEARCH.md`. Each mechanism ships with a paired ablation and a statement of what it guarantees and under which assumptions.

**Evidence.** jevkit (PR #243) is the JevBench scoring path. It reproduces the v1.6.1 board exactly, per `receipts/2026-10-09-jevkit-board/`. Kit parity comes first: reproduce ≥ 3 published open-weights systems within their CIs before any of our numbers count. Every win needs the evidence lane's independent rescore. Report failures, uncertainty and the exact scope of every claim, including the sealed-split and price-class caveats of local JevBench numbers.

## Stage 2: language growth (Agent C, Codex, CPU-only until Stage 1 passes)

Grow language modularly from the qualified decision model through checked semantic programs L0–L4. At every level: useful-answer performance at least matching a strong unrestricted comparator, with paired CIs; no regression in the parent's decision competence on identical items; soundness on joint and adversarial fixtures; and semantic interventions that beat random and output-only controls. Abstain when support is missing. Do not claim zero hallucinations beyond the tested conditions.

## Constraints

- **Directive (operator 2026-10-08):** dismiss self-imposed limitations you have put on yourself or think you have and proceed confidently while other subagents check your work and correct/suggest as needed in phases.
- **Spend:** compute is limited but is not the binding constraint right now. The live balance comes from `autor billing`. About $440 of promo credit expires 2026-10-27; use it on the objective. Never create top-ups or payment links, never buy credits, never enable auto-topup. Other providers need the user's OK.
- **Data:** never train or tune on TEST, held-out or sealed rows, JevBench eval items, or retention cohort v3. Every draw passes the leak audits (doc_overlap_guard, attack_guard, fingerprints). Open-weight teachers only.
- **Git and visibility:** commit as Apuroop with no AI or co-author trailers. Never push to `main`. Merges to `agent-main` are allowed after the peer lane's review APPROVE (operator 2026-10-09). The repository stays private. Nothing outward-facing, and no board submission, without the user's explicit approval.
- **Permission denials:** stop pursuing that outcome and report it.

## Completion test

The goal is complete when targets 1–4 are each met with receipts, the evidence lane's rescore, and merges to `agent-main`, or when a target is shown to be unreachable within the budget with a receipted gap attribution and a funding request. A failed experiment folds that line and unlocks the next justified one; it does not pause the goal. If a paid step is blocked, continue with independent $0 work and resolve the blocker. Never claim success without its gates. Pause only if the user asks, or if no permissible work remains; in that case, provide receipts plus the exact next action that would resume progress.
