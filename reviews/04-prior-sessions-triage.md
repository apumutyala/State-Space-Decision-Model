# 04 — Triage of two prior Claude Code sessions (transcript-only read)

Written 2026-10-05. Source: JSONL transcripts under `C:\Users\apumu\.claude\projects\C--Users-apumu-research-CHLabs\` (user messages + assistant text extracted by script; tool results read only when short or numeric). Nothing in the repos was read for this note; the other agent covers `chl-typed-control/RESULTS_OVERNIGHT.md` etc. Numbers below are as reported in-session (most were independently recomputed by a verifier agent at the time, noted where so). Treat as "what the session believed", not as re-verified by me.

Goal being served: (1) attention-free SSM "System 1" decision model that beats Clef / Jev with interpretability + alignment built in; (2) a controlled, aligned LLM built off it; $5k budget.

---

## Session e169e9d2 (67 KB, 2026-09-30 04:13Z) — verdict: DISTRACTION

Contents: the same "Overnight GPU run: typed-action control under injection + LLM-scale patterning" plan prompt (pasted by the user, authored by a Cowork session) followed by one assistant line: "Your organization has disabled Claude subscription access for Claude Code · Use an Anthropic API key instead". Dead session; zero work. The plan text itself is repeated verbatim as the first message of session 93cb1cee, so nothing unique here.

---

## Session 93cb1cee (19 MB, 2026-09-30 04:14Z → 2026-10-03 13:50Z) — verdict: PARTLY RELEVANT (two distinct phases)

### Phase A (Sep 30 – Oct 1): `chl-typed-control` overnight study + follow-up round — RELEVANT
Measures exactly the failure modes a Jev/Clef-style typed decision model has under prompt injection, on 17 open LLMs scored as 7-option typed decisions, plus patterning/attribution experiments. Nothing here is an SSM, and the planned "stretch" Jev-style non-autoregressive decision head on a frozen backbone was **never built**. Value to us: (a) a verified, reusable typed-decision eval harness and threat model, (b) hard evidence that "typed outputs" and "mask + alarm" do not deliver alignment by themselves (the thing our v1 plan risks overclaiming), (c) a working, cheap GPU-rental + kill-safety + pre-registration + independent-verification process, (d) the user's working style and corrections.

### Phase B (Oct 1 22:43Z – Oct 3): `bounded-intelligence` Trainium2 build-out — DISTRACTION (with reusable infra facts)
User asked to build out `white-box-labs/WBL_3_Trainium2_Build_Plan.md` into a new private repo and start testing on Trainium2 (ask before spending). Eight-plus subagents wrote NKI kernels (fail-closed action gate, slot hash/rollback, memory scrub, budgeted LoRA-expert updates, router, Mamba-2 chunked scan with telemetry, gradient-alignment safety screen, own Qwen3 serving path without NxDI), a host control plane, CPU emulator, three rounds of adversarial review, a 968k-decision toy sandbox. **Nothing ever ran on Trainium**: both AWS quota requests (L-64569A79 Trn2 Capacity Blocks, L-6B0D517C Trn spot vCPUs, account 932693968295, Melbourne ap-southeast-4) were still `CASE_OPENED` ~30 h later when the transcript ends with a failing status check. Only an r7i.4xlarge build box ran (2 h, ≈$2.15). Relevance to our goal: low (continual-learning/self-update plumbing, not a decision model), but the Trainium pricing/quota facts and the Mamba-2 NKI scan kernel are reusable if we ever go Trainium. The repo contains a Mamba-2 SSD scan kernel + BIRSim findings (silent wrong values past cumulative decay ≈ −29; NaN in state reported as "nothing written") — a hardware-numerics gotcha for SSMs on Trn2.

---

## Phase A findings that are probably NOT fully in the repo docs (or worth restating)

### A1. Typed decision setup (reusable harness)
- InjecAgent DH+DS rebuilt as 1,054 typed-decision cases (A/train 255, B/dev 126, B/test 129, DS/dev 276, DS/test 268), 7 lettered options, 27 variants each (clean, base, enhanced, 24 attack-template families split 12/12), 2 option orders. Gold action varied (respond 40% / follow-up tool 20% / clarify 20% / retry 20%) so a LoRA cannot win by "always respond". One DS case excluded because attacker tool == user tool.
- Scoring = one forward pass, next-token logits restricted to option letters (logsumexp over "A"/" A"), fp32 letter logits from final hidden state, **RIGHT padding with no padding mask** (bf16 left-padding moved probabilities by up to 0.12 — a real bug they hit), mean over two permutations to debias position, letter-mass validity check (99.9–100% for Qwen3; Muse-Glimmer forced-answer only 0.16 → excluded). Throughput: Qwen3-0.6B ≈140 prompts/s on A100-40GB; Qwen3-32B whole sweep 51 min on H100; Qwen3.8-27B 55 min; Gemma-4-31B 60 min; gpt-oss-20b 19 min.
- Hybrid-attention models (Qwen3.5/3.6/3.8, Gemma-4 SWA) are fine for scoring but broke their LoRA "cache the frozen prefix" trick, which assumes full attention in the last 8 layers → training stayed on Qwen3-1.7B/4B.

### A2. EXP-A results (verified by an independent agent: 1,461 quantities, 0 discrepancies >0.5 pp; then corrected after adversarial review, reanalysis re-verified: 280 point estimates exact)
- Trusted mask is exact: 0 attacker-tool selections in every unit of every model.
- Qwen3-32B within-mask accuracy loss per attacked input: 12.8 pp [11.1, 14.6] (cluster CI [8.9, 17.7]). **Decomposition (the correction):** 9.2 pp are "renormalised hijacks" (free argmax was the attacker tool; mask silently handed choice to 2nd-best) and only 3.6 pp [2.8, 4.5] is direct within-mask steering, ~70% of which is "ask user to clarify". **Mask-and-alarm** (alarm if free argmax ∉ allowed set) flags 24.6% [22.4, 27.0] of attacked inputs with 1/1,053 false alarms on clean.
- Hijack rates, Qwen3-32B free arm: single injection 24.5% [22.2, 26.8]; best-of-24 templates 72.6% [68.8, 76.3]; best-of-23 without chatml 72.2%. Qwen3 ladder: 0.6B 14.0 / 1.7B 40.4 / 4B 74.8 / 32B 72.6 (best-of-24) — hijack **rises then plateaus ~70% from 4B up**; it does not vanish with scale. OLS slope +35.4 pp/decade (best-of-24) or +17.6 pp/decade (single injection, no chatml).
- ECE (Qwen3-32B, free arm, x100): clean 0.7 → base 4.0 → enhanced 14.7 → adaptive 17.8. Calibration degrades badly under attack.
- A3 conformal abstention (α=0.05, calibrate on attacked split A): held-out templates realised risk 1.8% [0.8, 2.9] (passes with naive CI, **fails under clustered CI**), held-out goal (DS) 5.3% [2.9, 8.3] → **fails**; coverage collapses to 38–46%. Calibrating on clean data (τ=0.675): risk 11.7% / 20.2%. "Calibration needs attack data and does not transfer to new attack goals."
- A4 (Qwen3-4B): LoRA with injection augmentation: clean 100%, B-family hijack 0.3%, best-of-24 2.3%; LoRA on clean data only: hijack 25.4% / 75.7% — **clean fine-tuning made models more hijackable** (base "none" 18.9% / 72.5%). Sandwich prompt: 2.6% / 27.2%; spotlighting 15.9% / 64.2%. Gate passes, but the reviewer notes the adapter may simply learn the mask rule (benchmark is solvable by a trusted rule).
- **chatml confound**: one of 24 templates writes literal `<|im_start|>`/`<|im_end|>`; Qwen tokenizers (all Qwen3.x/3.6/3.8/SecOPD, ids 151644/151645) parse them as real role tokens; Gemma/gpt-oss/Mistral/Muse see plain text; OLMo-2 has the special tokens but not as role tokens. Without it Qwen3.6-27B 80.3→34.3%, Qwen3.8-27B 44.2→24.7%, Qwen3-1.7B 40.4→12.0%. "Qwen3.8 halves Qwen3.6" and "8× spread at ~30B" were withdrawn (now 2.6× best-of-K, 3.4× single-injection, diff −9.7 pp [−19.6, +0.8]). Lesson for us: any tokenizer-level decision model must `split_special_tokens` on untrusted spans.
- Other models (best-of-24 hijack): gpt-oss-20b ≈9%, Gemma-4-31B ≈13%, Mistral-Small-24B 75.5%, Qwen3.6-27B 80%; ordering matched Meta's published AgentDojo ASRs (Qwen3.6-27B 40.3%, Muse 28.4%, Gemma-4-31B 25.6%).

### A3. Follow-up items (Oct 1; each independently verified "confirmed with notes")
- **Item 2 (SecOPD + sanitisation, H100 ≈5 h):** sanitize (U+200B after first char of special-token strings) or split-tokenise removes 96–100% of the chatml hijack (Qwen3.6-27B 78.7→2.0/3.4%, Qwen3.8 42.6→0%, SecOPD 6.9→0%); Gemma ZWSP control +0.0 pp. SecOPD (security-trained Qwen3.6-27B) vs untrained Gemma-4-31B worst-case hijack 4.7% vs 6.5%, CI [−7.4, +3.3] → not reliably better. SecOPD single-injection hijack 0.09× base, direct within-mask steering 0.60× [0.21, 1.26] → inconclusive. Separate `input` role cost SecOPD clean accuracy 97.0→91.7%.
- **D1 (erase-and-check certificate, A100 ≈2 h + gpt-oss on H100):** one inserted sentence steers to attacker-preferred *allowed* option 68.9% (Qwen3-8B) / 67.6% (4B) with alarm 0%; excluding ties 68.6/65.4, excluding clarify 61.4/57.2. Certificate (erase each sentence, k=1) let 0 attack-changed decisions execute vs CausalArmor-style LOO baseline 27/896 (8B) and 496/896 (gpt-oss). Cost: escalation 94.5% (8B) / 87.1% (4B) under insertion; clean auto-execute only ~50% (90% when evidence repeated, 10–23% when unique). Edits/merges outside the threat model give 30–35% auto-wrong. On InjecAgent (injection in one field) Qwen3-8B auto-executes 64.5% with 0 wrong. Novelty check: not scooped but incremental (erase-and-check 2309.02705 already has the proof; CausalArmor 2602.07918 is the baseline; 2609.34245 certifies multi-source only).
- **Item 3 (GCG optimising attacker, A100 9.9 h vs 8 h prereg cap — deviation):** hardened Qwen3-4B hijacked in 2/15 (bar was ≥50%), but 13/15 attacks still improving at the 100-step cap → "held for 100 steps", not robust. **Steering to a wrong allowed action beat hijacking in 5/6 model×defence cells (4 significant), 50–80% success with 0 alarms**; a universal suffix optimised on hardened 4B steered 31.5% of 397 held-out cases, 0 alarms. Caveat: successes win by a hair (p≈0.5), both-orderings criterion drops hijacks to 0 and steering to 0–3 per cell; 15 cases/cell.
- **Item 4 (EXP-B v2, 2×A100):** gradient alignment (TracIn-style) predicts real retrains: median LDS 0.72 [0.56, 0.81], near noise ceiling; GGN influence −0.14, empirical-Fisher 0.05, random 0.07 → influence variants fail here; SGLD inconclusive (0.03; R-hat 1.16–2.05, ~42 draws/chain, one of 4 chains lost to a sync race — copied mid-replace). Earlier overnight EXP-B was a null caused by setup (saturated fit, median train loss 3.7e-5; empirical-Fisher collapse is known, Kunstner 2019; SGLD at nβ=N, γ=1e4 vs Resolution's guide ranges nβ 1–100, γ 10–1000 → 90% of χ on 10/3,825 examples; RMSProp-SGLD R-hat 9.4–16.7 despite passing loss rule; plain SGLD mb=32 R-hat 1.01).
- **Item 5 (CPU, convex head, `chl-acquisition-gate`):** SGLD minibatch bias predicted by noise model at cosine 0.9988; correction cuts error 3.46→0.196 (bar 0.2, borderline, ~4× more samples, needs exact Hessian, partly circular); extrapolation fix fails (0.34 vs 0.3).

### A4. External-anchor facts verified in-session (useful for our Jev/Clef comparison)
- Jev (TypeSafe, 15 Sep 2026, $40M seed led by DCVC): types Choice / Score / Noul (0–1 prob); claimed latency 70–500 ms; $0.042 per M input tokens, output free; speed claim 193.6× consistent, cost claim conflicting (homepage 244.6× vs Wikipedia 444.6×); no quantitative calibration metric found at source.
- Decision Hijacking (Wu & Lim, 2609.28613, NTU): Jev state fields `user_request, actions_taken, latest_tool_result`; 3-option set (final_answer / user tool / attacker tool) plus secondary 8-option set; `jev-1.13.0`; 1.8% (9/510) base, 3.5% adaptive at B=24 from a 72-config grammar, 0.2% at confidence ≥0.7, enhanced 0/510.
- Resolution (ex-Timaeus) susceptibility primer 2605.07980 defines χ = −Cov[φ, ℓ_z − L] (1/nβ factored out); the "Latent Posterior" paper's real title is "Elicitation without Backpropagation" 2607.18804; BIF paper 2509.26544 admits underperforming EK-FAC and blames sampling bias.
- Open-weight landscape (Sep 2026, from HF API): only attention-free-ish open models that fit 80 GB were **Nemotron 3.5 Lightning 30B-A3B (Mamba-2 + MoE, training data includes injection sets)** and hybrid Gated-DeltaNet Qwen3.8-27B (3:1 linear:full). Kimi-Linear-48B too big. **No pure Mamba/RWKV/xLSTM multi-B backbone was surveyed** — the v2 plan's "largest open pretrained attention-free backbone" assumption is still unverified by these sessions. Also: Gemma-4 ungated Apache-2.0; Llama still gated; Qwen3.8 needs flash-linear-attention + causal-conv1d or runs on a slow fallback.
- There is a `C:\Users\apumu\research\CHLabs\mechanistic-ssm-control\` directory (created 18 Sep 2026) and `chl-structured-models` (coordinator-owned core/, CONTRACT.md, DESIGN.md) that were never opened in these transcripts — worth a look by the repo-reading agent.

---

## Costs actually spent and rates

| Item | Rate | Spent |
|---|---|---|
| Lambda H100 80GB PCIe (us-west-3 only had capacity) | $3.29/h | overnight "big" ≈5.5 h; item 2 ≈6.5 h |
| Lambda A100 40GB SXM4 (3 regions) | $1.99/h | expa, a4, expb/expb2, d23, item3 (9.9 h), item4 ×2, d1 |
| Lambda H100 SXM (none available) | $4.29/h | — |
| Overnight run total (approved worst case $128.31) | | **≈$78.73** |
| Follow-up round (items 2,3,4,D1; cap raised to $200) | | **≈$68.22** |
| **Phase A total** | | **≈$147** (upper bound; dashboard authoritative) |
| AWS r7i.4xlarge build box, us-east-1 (Phase B) | ≈$1.06/h | ≈$2.15 (2 h) |
| Trainium2 (never purchased) | trn2.3xlarge CB $2.235/h ($53.64/day; Melbourne + São Paulo only); spot $2.27/h Melbourne, $5.09 São Paulo; trn2.48xlarge CB $35.76/h Ohio ($858/day), spot $14.27/h; no trn2 on-demand exists; trn1.2xlarge OD $1.344/h | $0 |
| RunPod | plugin installed, kill-safe tooling built (49 tests), never used | $0 |

Other public Lambda prices they fetched (2026-09-30): 8×H100 SXM $31.92/h, A100 80GB $2.79, GH200 $2.29, B200 $6.99, A10 $1.29, A6000 $1.09. Implication for our $5k: ≈1,500 H100-PCIe-hours or ≈2,500 A100-40GB-hours on Lambda; the user's own machine (HP Omen laptop, small GPU, RAM pressure at 87% with one 10 GB smoke) is not a training box.

## Reusable infra (all in `chl-typed-control/infra/` and `bounded-intelligence/infra/`)
- Lambda: `lambda_ctl.py` (types/launch/list/terminate/spend/push-key, dry-run mock, refuses wrong price/prefix/no-capacity/over-budget), `start_instance.py` (per-instance git bundle), `on_instance/deadline_runner.sh` (self-terminate via API after `sync_hook` gets a pull ack; OS shutdown does NOT stop Lambda billing), `sync.py` (tar-over-ssh puller, `PULLED_AT`/`FINAL_READY`), `watchdog.py` + `install_watchdog.ps1` (two wake-enabled scheduled tasks), fleet `health.sh`, smoke-then-GO gate, resume-safe JSONL results. Lambda API: 1 req/s, 1 launch/12 s; billing address must be valid (all first launches got HTTP 403 `global/invalid-address`).
- AWS: `launch_trn2.py`, `cost_guard.py`, `teardown.sh`, `stages.py` (approval gate `BI_SPEND_APPROVED=<stage>`), locked to profile `bi-personal` / account 932693968295 / IAM user `hp-omen-agent`; SSH key `bi-trn2` (`~/.ssh/bi-trn2.pem`), SG `sg-0b2ec1d1a4ea7ec99` (port 22 from 146.70.202.29/32, a VPN exit), VPC `vpc-0d513fa8d9bba84b8`, us-east-1 SG `sg-0684fa65bff0bda22`; Neuron 2.32 DLAMI located; 348 precompiled NEFFs cached (`hw_cache/`, 10.6 MB).
- Process scaffolding: `PREREG_*.md` with numbered amendments/deviations, `verify/` independent recompute agents, fresh-context adversarial reviewer, 20-min cron loop watching GitHub Issue #1 / draft PR on branch `agent/work` for teammate (pranavdulepet) feedback, `overnight/mirror_scrubbed.py` redacting names, `scratchpad/build_brief.py` → `MORNING_BRIEF.html` (artifact https://claude.ai/artifact/MYPJNYqxFhR5CAodU3appH).

## Gotchas logged (hardware/infra/analysis)
- Key leak: user pasted the Lambda key as the `Read-Host` prompt text → rotated; use `-AsSecureString`.
- H100 boot can exceed 15 min; retry `start_instance`. Capacity errors in us-west-2 → retry other regions.
- `pgrep -f`/`pkill -f` self-match killed the SSH session twice; kill by PID.
- Rewriting `job.sh` under a running bash corrupts it; restore clone-time bytes; append-only edits to running scripts (md5-check existing bytes).
- Parallel `start_instance` collided on one temp bundle path.
- GitHub 100 MB limit hit by `H_ggn.npy` (512 MB); ignore `*.npy`, `*.pt`, `grads_*.npz`; 45 MB of duplicate pulled-back results remain in history.
- Windows CRLF in `.pem` files breaks SSH; Windows paths are case-insensitive so `expb2` ignore rule swallowed `expB2`.
- Smoke tests on tiny dev sets missed OOMs (expb, d23, Qwen3-4B on 40 GB) → micro-batching.
- GPU nondeterminism broke same-seed retrain comparisons → `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Result-sync race copied a chain file mid-replace → lost one SGLD chain.
- gpt-oss chat template embeds the current date (reproduces only with run date); Muse-Glimmer loads only as ImageTextToText in transformers 5.17; Qwen3.8 hybrid needs `causal_conv1d`.
- Laptop RAM: agents' local model smoke tests pushed RAM to 87% / free <1 GB repeatedly; user complained; keep all model runs off the laptop.

## User priorities and corrections (should shape the new plan)
- Priority order stated by the design: "practical, convincing evidence about controlling AI decisions > breadth > methodological fanciness. Nothing is included just because it sounds sophisticated." User's own framing: "get on the critical important path of controlling language generation / solving AI alignment"; "pursue the highest-leverage and opportunity experiments".
- Wants frontier-strength models, not stale ones ("why Qwen3-32B and Mistral-24B instead of the latest frontier models… at the same param size?") and asks for epistemic humility + "is it worth the money" reports before spend.
- Wants a fresh-context adversarial reviewer over all work before trusting it; acted on its corrections and the reanalysis.
- Results presentation: "remember to graph and clearly break down and explain results for me in the morning"; wants an HTML brief he can open on device/iPad; plain-language "why we ran it / how it works / what we learned" panels; later asked "is any of this worth actually sharing" (answer: in meetings yes, publicly not yet).
- Budget behaviour: $150 hard cap → "don't need to cap or limit the cost… as long as under $200 total"; for Trainium "ask me for money spend before you do", then set a $200/month AWS budget and said "execute autonomously… set a goal for yourself with your subagents". Every instance must be guaranteed to die (three independent kill paths).
- Git/identity: commits authored and committed as Apuroop Mutyala <apumutyala@gmail.com> only, no AI trailers; repos stay **private** (rejected making them public); teammate ps.dulepet@gmail.com (GitHub `pranavdulepet`) gets write access + a step-by-step HANDOFF.md with an agent-specific section; **remove person/meeting names** (Jesse, Esben) from anything shareable but keep org relevance (Resolution/Timaeus, Apart/Seldon/Juniper).
- AWS: personal account only (profile `bi-personal`, 932693968295); **never** the Infinity AI account (apuroop@infinity.inc). No InfinityAI work in these repos.
- Subagent style: "spawn subagents to help you parallelize, pipeline, and verify your work"; pre-register before results; report every gate whichever way it falls.
- Two assistant refusals the user should know about: it declined to write a plan for engineering the model to break its own oversight boundaries and declined to delete the oversight boundaries from `docs/BOUNDARIES.md` (left them in place; renting hardware didn't depend on them).

## Open threads at end of transcript
- Trainium: quota requests pending since 2026-10-02 07:16Z; nothing bought; `bounded-intelligence` repo fully simulator-tested but zero hardware results; the last status tick (Oct 3 13:50Z) errored (exit 107) — unknown whether the local watchdog/cron still runs.
- Never done: Jev-style non-autoregressive decision head (stretch), reasoning-on comparison for thinking models, AgentDojo port (benchmark limited to InjecAgent's 17 user tasks), EK-FAC baseline for EXP-B, an attacker who deliberately drives up escalations for D1, k>1 certificate at scale, Gemma-4-31B on D1 (needs 80 GB).
- Teammate feedback loop (Issue #1 / draft PR #7) received no comments; cron deleted at user's request on Oct 1 18:32Z.
- `chl-acquisition-gate` now private on GitHub (Oct 1); one uncommitted local edit to `scripts/run_dev_stage.sh` not made by the agent.
- Watchdog scheduled tasks `chl-overnight-terminate*` and modified power settings may still be on the laptop (`Unregister-ScheduledTask`, `powercfg /restoredefaultschemes` were suggested).

## What this means for the SSM decision-model plan
1. The sessions give us a ready typed-decision injection benchmark and harness (scoring recipe, splits, 24 attack families, mask/alarm/certificate code, GCG attacker on letter logits, cluster bootstrap). Reuse it as the alignment gate for any decision model we train; it is far more adversarial than Decision Index.
2. Evidence says "schema conformance" and "mask + alarm" are necessary but not sufficient: silent steering within the allowed set is the live failure, calibration collapses under attack, conformal guarantees don't transfer across attack goals, and clean fine-tuning increases hijackability. Our "alignment built in" claims must be scoped to what the certificate/erase-and-check family can actually guarantee, with the escalation cost measured.
3. Nothing in these sessions tested an attention-free backbone; the v2 plan's first gate (frozen attention-free backbone vs same-size transformer on zero-shot schema decisions) is still the right first experiment, and the model survey suggests Nemotron 3.5 Lightning 30B-A3B (Mamba-2+MoE) and Qwen3.8-27B (DeltaNet hybrid) as the only sub-80GB near-attention-free candidates found so far; pure-SSM multi-B backbones still need a survey.
4. Patterning/influence for data-driven control: at this scale gradient alignment (cheap) predicted retrains (LDS 0.72) and curvature/SGLD methods did not; budget accordingly if "controlled LLM" work plans to use susceptibilities.
5. Costs: a full 17-model scoring sweep + hardening + two attribution studies cost ≈$147 on Lambda; $5k is ~30× that. Lambda A100-40GB at $1.99/h was the workhorse; H100 PCIe capacity was scarce (one region).
