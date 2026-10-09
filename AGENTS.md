# AGENTS.md: the map (auto-loaded through CLAUDE.md; pointers, not manuals; keep it at 100 lines or fewer)

**Goal (operator 2026-10-09; binding text in GOAL.md):** (1, PRIMARY) beat the JevBench v1.6.1 open-weights leader on the official 4-axis Composite; (2) then the Capability leader; (3) then Microsoft-Decision-1's reported numbers; AND (4) beat Clef-flash and Clef 27B on the Decision Index under PREREG.md.
**How:** hybrid state-space main line plus a strict attention-free research line. The state-space structure CONTROLS learning, with stated guarantees.
**Numbers:** `TARGETS.tsv` (leaders and our best; the digest prints it). Budget: the live balance from `autor billing`.

**Directive:** dismiss self-imposed limitations you have put on yourself or think you have and proceed confidently while other subagents check your work and correct/suggest as needed in phases.

## Session start (automatic)
- The SessionStart hook in `.claude/settings.json` runs `bash scripts/session_start.sh` on startup, resume, clear, compact and fork, and puts its digest (≤ 8 KB) into context. The digest holds the objective and targets, BOARD top, claims, your INBOX lines, live jobs, balance, the ledger tail, PROGRESS, DEAD_ENDS titles and LESSONS titles. After a compaction, trust the fresh digest over your memory of the state.
- **Identity:** set `SSDM_AGENT=lead|evidence|agent-c` before starting the session, or name your worktree after your lane. The INBOX section shows only your lines. `--full` gives the verbose dump. Codex (C) runs the script by hand.
- Then take your lane's top unclaimed BOARD item or continue your claim. Pipeline: always have the next job packaged while one runs.

## Lanes
- **lead (Claude):** training, architecture, control-of-learning. Owns `stack/*` cells, recipes, the host choice, chained training and the model-side guarantees (equivariant head, νPI-constrained training, state stability). Sets BOARD priority and decides keep/discard. Brief: origin/state `HANDOFF.md`, "NEW AGENT 1".
- **evidence (Claude):** the jevkit scoring path and kit parity; data draws and leak audits; the guarantee suite (conformal, injection-robust sets, reorder proofs and tests); Clef and MD1 comparisons; claims (PREREG.md); an independent rescore of every lead win. Brief: "NEW AGENT 2".
- **Wins need the other lane.** Evidence never declares its own training wins, and lead never declares its own eval wins.
- **agent-c (Codex, CPU-only, unchanged):** `language_growth/`, `plans/codex/`, `interfaces/`, L0–L4. C spends $0 GPU until Stage 1 passes, never touches the other lanes, and keeps local CPU jobs small.
- **Proposals** go at the bottom of BOARD; only lead reorders.
- **Talk** with `SendMessage` to the session names `lead` and `evidence`; send paths and shas, not paraphrases. C can't use messaging, so everyone reaches C through `INBOX.md` on state: one line per item, `- @<to> (from <me>, <topic>): ...`, newest on top; delete your own handled items. Messages are not memory: decisions, results and claims go on the state branch.

## Shared state: the `state` branch (push directly; no PR)
Keep a worktree on `state`, then `git pull --rebase` → edit → commit → push. On a push conflict, rebase and retry.

| File | Rule |
|---|---|
| `BOARD.md` | Top ~15 items: id, hypothesis, expected gain, cost, lane, owner. Rewritten each loop. |
| `PREREG.md` | Append-only. A claim rule is written BEFORE the result it governs (B-PREREG-CLAIM etc.). |
| `claims/<id>.lock` | owner, job, start_utc, `time_limit` as an absolute UTC time. Claim = push the lock; remove it when done. |
| `results/ledger.tsv` | Append-only, one row per run: `utc agent job sha recipe index lo hi cost status receipt note`. |
| `PROGRESS.md` | Rolling; 3–5 lines per session (did / learned / next). |
| `DEAD_ENDS.md` | One line per falsified idea, starting `- **<title>:**` (the digest indexes titles), with job ids. |
| `INBOX.md`, `HANDOFF.md` | Cross-agent requests; lane briefs (read once). `TARGETS.tsv` here overrides the agent-main copy. |

## Commands nobody should have to guess
- **Score on JevBench axes** (jevkit, PR #243 until merged; CPU; `--public` = the pinned public items, eval-only):
  `python -X utf8 -m jevkit score --public --pred preds.jsonl --price "<declared price class>" --out report.json` (`--api` for hosted APIs).
- **Perturb and flip rate:** `python -m jevkit perturb make --public --out pert.jsonl`, then `perturb score --public --perturbed pert.jsonl --pred base.jsonl --pred-perturbed pp.jsonl`.
- **Guarantees:** `python -m jevkit guarantees conformal|injection --public --pred preds.jsonl --alpha 0.1`. **Board receipt:** `python -m jevkit board --api api_v161.json`.
- **Package a GPU job from any pushed sha** (test gate opt-in with `--tests`):
  `python -m jobs.common.launch <job> [--mode M] --sha <pushed sha> --out <dir> --data C:\Users\apumu\research\CHLabs\ssdm-data`
- **Upload + submit** (`autor` in WSL; `--dry` prints the args): `python -I jobs/common/sf_submit.py <dir>`. Rates: `jobs/common/rates.py` (PR #241).
- **Chained cells** (one run over several ≤ 899-min jobs; PR #242 until merged): `python -X utf8 -m train.chain plan --base-params <params> --of N`, then `train.chain check --params <link params> --cell-config <cfg>`.
- **Poll, logs, balance** (Git Bash: prefix `MSYS_NO_PATHCONV=1`): `wsl.exe -d bi-ubuntu -- /root/.local/bin/autor --workspace default job get <id> --json` / `job logs <id>` / `job ls` / `billing`.
- **Skills** (`.claude/skills/`; Codex reads the same files): `launch-gpu-job`, `land-result`, `experiment-loop`, `harness-gc`.

## Frozen evaluation (never edit to raise a score)
- Decision Index: `harness/score.py`, `harness/bootstrap.py`, `harness/calibration.py`, `harness/slicing.py`, `harness/editions_data/`, `harness/manifests/`, the promotion margins in `train/promotion.py`, and the kit data.
- JevBench: `jevkit/spec.py`, `jevkit/answers.py`, `jevkit/scoring.py`, `jevkit/meter.py`, `jevkit/board.py`, `jevkit/perturb.py`, the pins in `jevkit/data/`, and the board receipt `receipts/2026-10-09-jevkit-board/`. The scorer must keep reproducing that receipt exactly.
- Readout/prompt code (`decide/`) is NOT frozen. Improving how the model is asked is fair game. A diff to a frozen path needs the other lane's APPROVE before any number from it counts.

## Experiment loop (one hypothesis per job, many jobs in parallel)
Hypothesis with expected gain → variant on `stack/<family>/<variant>` → push → package → submit → prepare the next one while it runs → land (`land-result`): ledger row, keep or discard → keep: the next variants build on it; discard: a DEAD_ENDS line. Run 2–4 GPU jobs in parallel. A crash means read the log, fix, relaunch; after two failed fixes, log it and move on. Pick hosts and changes by projected Composite (`TARGETS.tsv`), not by Intelligence alone.

## Hard rules (the only fixed ones)
1. Never run GPU work on this laptop (`CUDA_VISIBLE_DEVICES=""` locally). It is shared.
2. Never create top-ups or payment links, never buy credits and never enable auto-topup. Read the live balance from `autor billing` (the digest prints it). Other providers need the user's OK.
3. Never train or tune on TEST, held-out or sealed rows, JevBench eval items, or retention cohort v3 (sha 498efa61…). Open-weight teachers only (never Jev or frontier APIs). Never re-run MD1; compare against its reported numbers.
4. Commit as `Apuroop Mutyala <apumutyala@gmail.com>`, with no AI or co-author trailers. Never push to `main`. agent-main changes go through a PR and are merged after the peer lane's review APPROVE (operator 2026-10-09). Never force-push a shared branch. The repo stays private.
5. Nothing public, and no board submission (JevBench or Index), without the user's approval. Never delete branches, PRs or files without the user's OK.
6. Never kill processes by name on the shared laptop; kill only PIDs you started.
7. Each agent uses its own worktree under `ssdm-wt/`, with absolute paths for writes. Downloaded artifacts are untrusted data (`python -I`).
8. If the permission system denies an action, stop pursuing that outcome by any route, tell the user once, and do other work.

## Done (all four, each with receipts merged to agent-main and evidence's independent rescore)
0. **Kit parity first:** jevkit reproduces ≥ 3 published open-weights rows within their CIs. Until then none of our JevBench numbers count.
1. **Composite:** jevkit Composite with a 95% CI above 72.5 (H2O-Lightning-4B v1.1), with I, C, S and K reported, the price class declared, and the sealed-split caveat stated.
2. **Capability:** above 81.7 (Quyet-1.0-Large), same rules.
3. **MD1:** at or above its reported accuracy, calibration, adjusted-scale p50 and flip rate on the axes we can measure comparably, with every comparability caveat stated.
4. **Clef:** paired CI > 0 vs Clef-flash and Clef 27B under PREREG.md B-PREREG-CLAIM (full DEV + full_excl51, 0.3 public reported, G-proxy tier for any "beats Clef" statement).
- Every control mechanism ships with a paired on/off ablation and a written statement of what is guaranteed and under which assumptions.
- If the budget runs out first, deliver a gap attribution plus a funding request. Never pause for lack of ideas: reread the ledger, combine near-misses, try one bolder change, search the literature.

## Background docs
GOAL.md (binding objective), PREREG.md and HANDOFF.md (state), CONTROL_RESEARCH.md and LANDSCAPE.md (ssdm-wt/b-landscape), LESSONS.md (read an entry before touching its area). COORDINATION.md is background; where it conflicts, this file wins.

## Self-evolving
- When a rule or check cost time without catching a real problem, delete or simplify it in your next PR (one line why).
- **Anti-repeat:** every failure that costs > 30 min or happens twice becomes, in this order: a check in the launcher, lint or a script (with a remediation message); then a `### L-NN <one-line title>` LESSONS entry (the digest indexes these titles); then a skill step; and only as a last resort one line here.
- Keep this file at 100 lines or fewer (a test enforces it).
