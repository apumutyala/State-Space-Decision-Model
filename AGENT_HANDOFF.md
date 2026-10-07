# AGENT_HANDOFF — for collaborators' coding agents

You are joining a repo run by two owning agents: **Claude = Track D** (decision model; `claude/*` branches) and **Codex = Track L** (language growth; `codex/*`). Your job is to pursue a research direction your human chose, on your own branch, without delaying or disturbing them. Read this file first, then the files in §2 in order. Facts below are verified against `origin/agent-main` @ `3b4169a`, merged PR bodies, issues #3/#36, `LEDGER.md` and the named specs on 2026-10-06; anything marked **[unverified]** could not be checked in the repo.

## 1. Hard rules (binding; a win obtained by breaking one does not count)

**Branches and PRs**
- Never push to `main` or `agent-main`. Never force-push a shared branch.
- Never commit to, rebase, or otherwise touch `claude/*` or `codex/*` branches or their worktrees. Do not `git add -A` over files you did not write.
- Work on `collab/<github-user>/<topic>` (e.g. `collab/pranavdulepet/rwkv7-fla-kernel`), branched from `agent-main` and rebased on it before you open a PR.
- Open PRs into `agent-main` only, only when the work is ready, using `.github/pull_request_template.md` (claim scoped exactly; receipt with commit, command, hashes, seeds, hardware, cost; gates ticked). Small PRs, one experiment or module each; a PR open more than two working sessions gets split or closed.
- Merging needs **one independent review** per `COORDINATION.md` "Review sufficiency": the other track's agent, or a fresh-context adversarial reviewer that did not write the code. Findings get fixed first. Reviews are lean: correctness, contamination/test-slice integrity, safety-claim soundness, spend safety. Nobody approves their own PR. Docs-only changes need no review.
- Any reported number is re-derived by someone else from the committed receipt before a PR claims it. A win that does not reproduce is logged in `LEDGER.md` as a false positive.

**Data integrity**
- Never open, read or tune on the sealed **TEST slice** (`harness/cli.py` refuses without `SSDM_UNSEAL_TEST=1` + `SSDM_PREREG`; every unseal is logged). Never on Decision Index test items. Never on the held-out D1-B templates (built only with `final=True`). Development uses the DEV slice, the `safety_eval` dev split and the independent development cohort (`train/cohorts/`).
- Every training or distillation set goes through the contamination guard (`data/dedup.py`) against the full suite and leaves a dedup log. Declared sources only (`data/SOURCES.md`).
- **Open-weight teachers only** (Clef, Clef-flash, Qwen, Gemma). Jev and frontier APIs are never teachers, label sources or reference distributions (TypeSafe MCA §2.3(b)).
- No prompt tuning, truncation or option pruning on anything that will be submitted; abstention scores as wrong on the Index.

**Compute and money**
- **Never run GPU jobs on anyone's laptop.** Local machines are for CPU tests on tiny models only.
- Any paid job (SF Compute or Lambda) needs: a `spend` issue with estimate and stop condition, a planned row in `LEDGER.md`, a runtime ceiling and workspace cap, and the owner's acknowledgement. **Collaborators do not spend the shared $525 SF credits without the user's explicit OK** in that issue. Climb the spend ladder: $0 CPU → ≤ $20 hardware qualification → capped experiment → scale-up, each rung after the previous rung's receipt is merged.
- Record actual cost afterwards. Stop nodes. Auto top-up stays off; no credit purchases.

**Identity and contracts**
- Commit under your own name and email (the repo owners commit as Apuroop Mutyala; do not impersonate). The repo stays private.
- Do not modify `interfaces/DECISION_LANGUAGE_V0.md`, `INTERFACE.md`, `decide/schema.py` wire shapes, `safety_eval/PREREG.md` (append-only amendments by Track D only), `SAFETY_SPEC.md` rows, or `harness/manifests/` without first opening an `interface` issue and getting both owners' approval plus a version bump.
- Claims are scoped to what was measured. Smoke runs and stock ≤ 1B models are pipeline qualification, never model evidence (PREREG §0). "Attention-free" is disclosed with lineage and text-only scope.

**Before any paid job / before any PR**
- Read `LESSONS.md` (failure classes already paid for) and follow `jobs/LAUNCH_CHECKLIST.md`.
  - `tests/test_jobs_lint.py` must pass.
  - Launch only through `python -m jobs.common.launch` and its `submit.json`.
- Every incident appends a `LESSONS.md` entry and an enforcing check in the same PR.
- Variants go on `stack/<family>/<variant>`. `agent-main` holds only `CHAMPION.md`'s best candidate plus infra. Every measured variant is a row in `CANDIDATES.md`.

## 2. Get oriented fast (read in this order)

1. `README.md` (status table, repo map, quick start).
2. `GOAL.md` (the goal, exit conditions, binding constraints), then `PLAN.md` (tiers, architecture, milestones, spend lines, fallback).
3. `COORDINATION.md` (branches, PR rules, spend ladder, review sufficiency, lean review, Collaborators section).
4. `HANDOFF.md` (story, architecture, measured vs built, failures, open decisions).
5. `SAFETY_SPEC.md` on branch `claude/safety-spec` (PR #18, open): `git show origin/claude/safety-spec:SAFETY_SPEC.md`. Sections 1–5 first.
6. `safety_eval/PREREG.md` (what is gated, what is forbidden), `harness/INDEX_NOTES.md` (Index rules), `controlsafe/CONTROLSAFE.md`.
7. `jobs/DECISION_TREE.md`, then the `JOB_SPEC.md` of any job you touch; `jobs/PERF_AUDIT.md` on `claude/readout-fork` for the readout cost model.
8. `interfaces/DECISION_LANGUAGE_V0.md` and `INTERFACE.md` if you touch the D→L seam; `plans/codex/L0_PHASES.md` for Track L.
9. `techniques/CATALOG.md` and the relevant `techniques/0x` note before proposing a method; `reviews/01`–`04` and `analysis/` for why earlier ideas were killed.
10. `LEDGER.md` and issue #3 for spend state; `STATUS.md` for per-track status (partly stale; LEDGER wins).

## 3. Run the tests

```bash
uv venv .venv && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers==5.14.1 pytest numpy pyarrow
pip install -e harness -e data                    # enables data/tests; harness/tests also need the kit
python -X utf8 -m pytest tests -q                 # ~670 tests, ~8 min CPU (PR #42/#43 receipts)
python -X utf8 -m pytest tests/clef tests/controlsafe tests/safety_eval -q
python -X utf8 -m pytest tests/test_m2_e2e_dryrun.py -q      # end-to-end m2 on tiny models, ~71 s
python -m unittest discover -s language_growth/tests -p "test_kernel*.py" -v
python -m jobs.common.costs                       # cost/schedule checks for every job spec
```

`python -X utf8` is mandatory (the Index kit silently drops two BRIGHT rows under cp1252; the harness refuses to import otherwise). Run scripts that read downloaded data with `-I`. The official kit (`apolinario/decision-index` @ `87d4650b`) is only needed for suite work: clone with `core.autocrlf=false`, `pip install --no-deps`. Test data for the suite is rebuilt locally and is not in git; see `README.md` "Data is not in git".

Tiny random backbones for CPU tests: `heads.backbone.tiny_hf_backbone` (RWKV-7 and all-GDN Q9-AF layouts). Receipts for anything you report go in `receipts/<date>-<track>-<slug>/` (proposed convention for collaborator work: track letter `C`, unless the owner assigns otherwise).

## 4. Where to plug in without conflicts

| Direction | Extension point | Notes |
|---|---|---|
| New backbone | `heads/backbone.py::BACKBONES` (a `BackboneSpec`: capture mode, LoRA target regexes, verified dims) + a loader; `controlsafe/adapters.py` if you want trust-gate/persistence support (needs the layer written as `X_t = A_t X_{t-1} + k v^T`) | Keep `HiddenStateBackbone` as the only head-facing interface; right padding only |
| New decision head | `heads/heads.py` (follow integrator / fork / snapshot / Head B), register in `heads/model.py`; add shape, equivariance, attribution tests like `tests/test_heads.py` | Flag `attention_free=False` if it uses attention; typed outputs go through `decide.schema.wire_answer` |
| New controlsafe arm | `safety_eval/controlsafe_arms.py::ARMS`, module under `controlsafe/`, tests in `tests/controlsafe/` | State the tier (STRUCTURAL/CERTIFIED/EMPIRICAL) in `CONTROLSAFE.md`; no decay or Gramian number is a safety claim |
| New attack family for `safety_eval` | `safety_eval/attacks.py` (templates + `build_variants`), `safety_eval/adaptive.py` stub for defence-aware attacks, `safety_eval/gcg.py` for optimiser variants | Edit only the host-marked untrusted field; new families after test scoring are labelled post-hoc, never replace a PREREG row; D1-B stays held out |
| New training data source | `data/sources.py::REGISTRY` (pinned URL/repo + revision, files, licence, Index-overlap note), converter in `data/convert.py`, row in `data/SOURCES.md`; run `dedup.py` and commit the log | Train splits only; sha256 lock in `data/manifests/sources.lock.json`; Arts is under-sourced |
| New conversion recipe or kernel | `convert/gdn.py` (kernels; `RWKV7_WKV_FUNCTIONS["fla"]` hook for RWKV7 in `third_party/rwkv7_hf`), `convert/align.py`, `convert/pipeline.py` | Numeric gate vs the exact reference path before any GPU use; tolerance pre-registered |
| New probe | `probes/core.py` (MQAR, S-NIAH, ppl, MMLU subset), `probes/throughput.py` | Batch by length where exact |
| New Track L module/profile | `language_growth/` (kernel ops, planner, realizer), fixtures under `fixtures/language_growth/` | Interface changes need an `interface` issue; L0 is CPU-only; a proof-valid answer to the wrong question is a task failure |
| New job | `jobs/<name>/{JOB_SPEC.md, job_params.json, *.sh, Dockerfile}` + `jobs/common/costs.py::PARAMS` + a test like `tests/test_m1_next_jobs.py` | Exit codes, heartbeat watchdog, hash-verified exports, pre-registered decision rule; nothing is submitted without the spend process |
| Literature / method notes | `techniques/<nn>-<topic>.md` with the evidence legend (R/S/P/X), cross-linked from `CATALOG.md` | Every arXiv ID fetched and matched, or marked [unverified] |

Avoid: `decide/schema.py`, `decide/untrusted.py`, `harness/manifests/`, `safety_eval/PREREG.md`, `train/promotion.py` margins, `interfaces/` (owner-only or `interface` issue).

## 5. Stage by stage (same table as `HANDOFF.md` §5)

Costs from `jobs/*/JOB_SPEC.md`, `jobs/DECISION_TREE.md`, `jobs/common/costs.py`, `train/M2_RUN_SPEC.md` at $2.97/h ($0.0495/min); worst case = (build + 2 attempts × max_duration) × rate.

| Stage | Falsifiable hypothesis | Pre-registered decision rule | Cost | Status | Key files |
|---|---|---|---|---|---|
| m0-a harness | Decomposed scorer reproduces the official 0.2.1 kit on DEV; paired cluster-bootstrap CIs computable | per-benchmark gap 0.0 for reference predictors | $0 | **Done** (PR #7) | `harness/`, `receipts/2026-10-06-D-m0-harness/` |
| m0-b `decide()` adapter | Schema-valid, fail-closed adapter that replays its fixture exactly; wire answer matches Cloudflare's arithmetic | replay matches live on every case; equivariance exact | $0 | **Done** (PR #6, #24) | `decide/`, `fixtures/replay/v0/` |
| m0-c safety_eval + PREREG | Attack/metric/bootstrap pipeline runs end to end with complete accounting (G0) | PREREG §4 G0; D1-B only with `final=True` | $0 | **Done** (PR #8, #22) | `safety_eval/`, `safety_eval/PREREG.md` |
| m1 H100 qualification (`jobs/qual`) | 9B conversion pipeline runs at ≥ 25 % MFU, ≤ 60 GB, correct numerics, recoverable artifacts | ≥ 25 % & ≤ 60 GB → $40 probe; 12.5–25 % or 60–76 GB → re-price lean; else fold; exits 2–9 → no measurement (stage-2 floor 2,001 tok/s, 25 % bar 4,002 tok/s after PR #48) | worst $18.81 / cap $20 / expected ≤ $8.91; C-only $6.93 one attempt, $14.85 worst | `job-jbw22` $0.07 (fla #640); `job-c4zkm` $0.74 A+B pass (ΔNLL −0.0036, KL 0.017), C OOM 68.9 GB; `job-b69gs` C-only submitted 2026-10-06T17:36Z (max $6.93) | `jobs/qual/`, `convert/qual.py`, `convert/memmodel.py`, `scripts/decide_9b.py` |
| 9B one-layer probe (`jobs/probe9b`) | Converting the least-KL-important attention layer keeps LM quality, Index DEV score and retrieval in bounds | gate accepts; ppl ratio ≤ 1.02; MMLU drop ≤ 3.0; Index DEV paired diff ≥ −1.0, CI lower ≥ −2.5; MQAR drop ≤ 0.10; S-NIAH mean ≤ 0.10, each length ≤ 0.25 → proceed-all-8 / retrieval-fail / quality-fail | worst $28.21 / cap $40 / ~$11.04 | Spec only | `jobs/probe9b/`, `configs/probe9b_h100.json`, `scripts/decide_probe9b.py` |
| Full 8-layer conversion | All 8 layers convert in KL order with accept/rollback + recall levers without failing retrieval | per-layer ΔNLL ≤ 0.05, cumulative ≤ 0.15, hidden cos ≥ 0.95; on retrieval failure report the cost-of-attention-freedom curve, `decision-needed` on a hybrid | ~$150 (PLAN; $96–269 range in PR #9) | Not specced | `convert/pipeline.py`, `convert/progressive.py` |
| RWKV7 stage A readout | Stock RWKV7-13.3B scores the 2,008-row DEV subset inside 75 min and gives a steer baseline | measured preflight + live rate guard (exit 15) | worst $16.34 / cap $20 / ~$4.70 | `job-mhc3h` $0.56 stopped; fix PR #50 (open, changes requested) | `jobs/rwkv7/`, `decide/backends/state_fork.py` (PR #50), `jobs/common/rate_guard.py` |
| RWKV7 stage B LoRA pilot | Rank-64 cache-fork LoRA cell trains under 76 GB and loss falls | finite; last-50 mean loss ≥ 10 % below first-50; held-out NLL improves; peak ≤ 76 GB; exit 26 if too slow | worst $19.30 / cap $20 / ~$7.72 | Spec only; `depends_on` A | `jobs/rwkv7/pilot_cell.json`, `train/run_cell.py` |
| Clef baselines A/B/C | Clef-flash/Clef through our `decide()` match Cloudflare's `systemone` (S1 max \|dp\| ≤ 2e-4, non-vacuous `kernel_path`) and score on Index DEV, ECE, latency, hijack/D1/GCG, G1/G2(b)/G8 | no "better than Clef" without these rows; B gated on `s1_ok`; declared deviations torch 2.13, H100 latency | A $5.94 (~$2.7; launched with `max_restarts=1`, exposure can exceed one attempt), B $29.70 (~$10), C $11.88 (~$5.4); total ~$18 / $47.52 ceiling | **A `job-sgihj` running**; B, C not submitted | `baselines/`, `jobs/clef_baseline/`, `tests/clef/` |
| m2 head × parent factorial | A head × calibration arm meets T1 dev-slice criteria or each gap is attributed | G12/G13 promotion (lower CI Δcap ≥ −m_cap; upper CI Δsteer ≤ +m_steer; upper CI ΔECE ≤ +m_ece; completeness fail-closed); margins **pending #36** | pilot cell worst $18.81 / cap $20; parent arm $18.81 / cap $20; ≈ $12–17 per RWKV7 cell, $4.7–6.6 per Q9-AF cell (estimates) | Code merged, dry run passed, 150k set built; blocked on parent, margins, measured TOK_S | `heads/`, `train/`, `jobs/m2_pilot/`, `jobs/m2_parent_arm/`, `train/M2_RUN_SPEC.md` |
| G16 / controlsafe training arms | P-only judge + bounded evidence channel is capability-non-inferior to the D-head arm | `capability_cost` + `harness.bootstrap.compare` non-inferior → free guarantee; else widen r | one m2 cell; no compute beyond `1 + \|trusted\|/T` | Built, CPU-tested (PR #42); untrained | `controlsafe/`, `safety_eval/controlsafe_arms.py` |
| m3 full Index + independent repro | Candidate beats Clef-flash/Clef on full 0.2.1 with paired CI > 0 under 1,000 ms median latency; other agent reproduces | GOAL.md exit; TEST unsealed once with PREREG; submission needs user approval | not specced | Not started | `harness/`, `GOAL.md` |
| Track L L0a–L0f | Typed interface/kernel/renderer/planner/fixtures separate proposal, authority, proof, realization with no unchecked text | fixtures pass; cross-reviewed; not learned gates | $0 | **Merged** (PR #4, #5, #10–#13, #15, #17, #21, #25, #37, #46) | `language_growth/`, `interfaces/`, `plans/codex/L0_PHASES.md` |
| Track L L0g qualification | Real parent adapter loads and produces the pinned protocol within budget | ≤ $20 spend issue acked by D | ≤ $20 | Not started | `plans/codex/L0_QWEN_CPU_PROBE.md` |
| Track L L0h → L1–L4 | Parent discovers useful programs (L0h), exact sums (L1), composition (L2), discourse (L3), new domains (L4) without regression | all four GOAL.md gates per level with paired CIs; ExactSum profile review pending (#34, PR #38/#45) | Language line $100 | Not started | `plans/codex/LANGUAGE_GROWTH_PLAN.md` |

Spend to date: $0.07 + $0.74 + $0.56 = **$1.37** (LEDGER) plus running jobs; workspace cap **$33**; credits $525 of which the $500 promo expires **2026-10-27T06:53Z**.

## 6. Reading list (IDs exactly as in `techniques/CATALOG.md` and `techniques/09`)

**State-space / recurrent backbones**
- Gated DeltaNet 2412.06464 — the delta-rule mixer in Qwen3.5/3.8 and our conversion target; keeps retrieval better than additive linear attention (2504.14366). GDN-2 2605.22791 (architecture only).
- RWKV-7 2503.14456 — the pure-recurrent fallback family; its transition is not a common 2-norm contraction (CONTROLSAFE).
- Mamba-3 2603.15569; negative-eigenvalue delta rule 2411.12537; 2404.08819 (same catalog row, "delta-rule fixed-size memories") — why fixed-size memories interfere and what "capacity ∝ state" costs.
- Qwen3.8-Next design 2608.30320 — why the hybrids keep 1 attention layer in 4.

**Attention → recurrent conversion** (the m1 recipe and why the fallback is pre-registered)
- MOHAWK 2408.10189 (three-stage distillation; our stage-1 alignment); The Mamba in the Llama 2408.15237 (attention-weight reuse, progressive distillation; the 0 %-attention cliff); RADLADS 2505.03005 (RWKV conversion at < $2k per 72B); LoLCATs 2410.10254 (control; MMLU −13.8).
- KL-guided layer importance 2512.20569 (our conversion order); accept-or-rollback per layer in Qwen3.5-0.8B 2609.21139 (our gate); Priming hierarchy GKA > GDN 2605.08301; SWA + sinks zero-training control 2608.28444.
- Gather-and-Aggregate heads 2504.18574, 2510.19861 — one attention head can carry most retrieval; the reason pure conversions lose it.

**Recall without attention**
- Just Read Twice 2407.05483 (options-first ordering, re-read questions); OPRM chunked inference 2505.07793 (the snapshot heads); MQAR/Zoology 2312.04927 and Based 2402.18668 (the recall probes).

**Authority, injection, information flow**
- ISE 2410.09102, ASIDE 2503.10566 (the learned provenance channel, empirical only); StruQ 2402.06363, SecAlign 2410.05451, Meta SecAlign 2507.02735 — all broken adaptively (2507.07417, 2510.09023); CaMeL 2503.18813 (Track L's plan→check→realize shape).
- Inseparability theorem 2606.27567 — why authority must be out-of-band (G2 split); PACT 2605.11039 (injection matters only on authority-bearing arguments: why hijack is structural and steer is what remains).
- Jev decision hijacking 2609.28613 (base ASR 1.8 %, adaptive 3.5 %); Hidden Risks of Jev 2610.04985; black-box attackers PISmith 2603.13026, TAP ≫ GCG 2606.10525 (PREREG amendment to file).

**Certification and risk control**
- Erase-and-check 2309.02705 (G4's ancestor); CERT-ED 2408.00728 and RobustRAG 2405.15556 (planned probabilistic / per-record arms).
- Learn-then-Test 2110.01052, CRC 2208.02814, Selective CRC 2512.12844 (G5); feasibility floor 2606.29054; calibrate on attacked data 2511.18562; debiased ECE 1909.10155, SmoothECE 2309.12236; calibration attacks 2401.02718 (G7 arm). SNGP is **not** in the catalog; do not cite it as project-reviewed.

**Interpretability**
- Causal abstraction 2301.04709, IIT 2112.00826, DAS 2303.02536 / Boundless DAS 2305.08809; non-linear representation dilemma 2507.08802 (linear maps only); identification assumptions 2605.08012 (every mechanism receipt); "A Circuit, Not The Circuit" 2606.00930 (non-unique localisation in Mamba); WriteSAE 2605.12770 (GDN-native write atoms).
- Concept bottlenecks and leakage: 2106.13314, 2105.04289, CEM 2209.09056, Minimal CBM 2506.04877 (the leakage test on semantic records).

**Recurrent-state safety (new in 2026)**
- HiSPA hidden-state poisoning 2601.01972 (hybrid Jamba collapses; G14); SpectralGuard 2603.12414; contraction-constrained SSM certificates 2610.02853 (an arm with a tax, not a guarantee).

**Verified generation (Track L)**
- Grammatical Framework (Ranta 2011), ACE-in-GF 1303.4293, CNL survey 1507.01701 — rung A realizer; MRS/ERG asymmetry 2609.30071; grammar-constrained decoding 2307.09702, 2411.15100.

**Developmental tracking**
- Emergent unfaithfulness 2610.00568 and early behavioural directions 2605.13329 — why G12 rejects checkpoints and mandates a card after any preference-optimisation stage.

**Engineering and vendor material**
- fla issue #640 (flash-linear-attention 0.5.2 refuses the gated chunk backward on Hopper with Triton 3.4–3.7.0; fixed by the torch 2.13 / triton 3.7.1 base in PR #41, with `jobs/common/fla_guard.py`). No URL is recorded in the repo.
- Clef: blog https://blog.cloudflare.com/clef-decision-models/ ; cards https://huggingface.co/Cloudflare/clef and https://huggingface.co/Cloudflare/clef-flash (vendored head `joint_schema_model.py`, sha256 pinned in `baselines/clef_backend.py`). No safety or limitations section anywhere (techniques/09 §J).
- TypeSafe Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev ; MCA https://typesafe.ai/legal/mca (§2.3(b) bars distillation / competing use).
- Decision Index: https://github.com/apolinario/decision-index @ `87d4650b` (0.2.1); board https://huggingface.co/spaces/multimodalart/jev-decision-index . Rules digest: `harness/INDEX_NOTES.md`.

## 7. Open research directions collaborators could take

Each needs a `collab/` branch, a hypothesis stated this way in the PR, and $0 CPU proof before any spend request.

| Direction | Hypothesis (falsifiable) | Rough cost | Plug-in |
|---|---|---|---|
| **Faster RWKV7 / GDN kernels** | Registering fla `chunk_rwkv7` through `RWKV7_WKV_FUNCTIONS["fla"]` matches the exact `eager` path within a pre-registered bf16 tolerance and raises 13.3B prefill from 3.6–4.3k to ≥ 12k tok/s (the m2 eval cap needs ≥ ~12k) | $0 CPU gate + ≤ $20 H100 numeric/throughput gate | `convert/gdn.py`, `third_party/rwkv7_hf`, `jobs/PERF_AUDIT.md` §Kernels |
| **G16 capability cost** | A trained P-only judge with evidence rank r ∈ {8, 16, 32} is capability-non-inferior (lower CI ≥ −1.0 Index point on the 2,008-row DEV subset) to the D-head arm while its independence rate is > 0 on real decisions | one m2 cell ≈ $12–17 (RWKV7) after parent selection | `controlsafe/g16.py`, `trust_gate.capability_cost`, a `train/` arm |
| **Diagonal / certified layer swaps for the Pareto map** | Replacing GDN layers with diagonal (Mamba-style) or contraction-constrained layers (2610.02853) trades a measurable Index drop for certified decay bounds and Gramian observability; the curve is monotone and reportable | CPU on tiny models, then ≤ $20 on the 0.8B | `convert/model.py` modes, `controlsafe/persistence.py`, `gramian.py` (needs `JᵀJ` / randomised traces at 9B scale) |
| **Multimodal path** | A vision encoder feeding the recurrent state keeps the text-path guarantees intact (G1, G2, G8) and the attention-free claim must be re-scoped honestly | design + CPU; GPU later | new backbone spec; claims are text-only until a vision gate passes (`PLAN.md`) |
| **Arts & Human Taste data sourcing** | Legitimate train-split or synthetic sources for BPoMP/Humicroedit/POP909/cfcolor/Habermas/New Yorker formats lift Arts from 5.8 % to ≈ 10 % of the mix without item-tier overlap | $0 CPU (guard + build) | `data/sources.py::REGISTRY`, `data/SOURCES.md`, `data/convert.py` |
| **Jev evaluation protocol** | A harness-side Jev adapter can score the safety battery with the same units and seeds; whether to call the API is the user's decision | $0 code; API cost **[unverified]** | `safety_eval/` decider interface; needs the `decision-needed` issue first |
| **H-SLT side paper** | Singular-learning-theory quantities (LLC) log but do not predict decision metrics on our checkpoints; the Codex counterexample to H-SLT-1 stands or falls on measured cards | $ at ≤ 1B; unreliable at 9B-LoRA per catalog | `train/evalcard.py` logging; `techniques/08`; LLC is logged, never gated (`PLAN.md`) |
| **Uncovered-insertion certificate** | A cheaper span-unit (token-aligned) certificate than O(T²) decide calls exists, e.g. via snapshot reuse over the D stream, with certified-and-changed = 0 on the stress arm | $0 CPU stand-in, then ≈ $3–8 per parent | `safety_eval/certify.py`, `controlsafe/snapshot.py` |
| **Black-box search attack family** | A PISmith/TAP-style black-box attacker beats GCG-lite steer at matched budget on the dev split; filed as a PREREG amendment before any m1 test scoring | $0 code; ≈ 1–2 H100-h to run | `safety_eval/adaptive.py`, PREREG amendment via Track D |
| **G14 collapse probe** | HiSPA-style triggers in the untrusted field drop paired decision accuracy on ≥ 4k-token records by > 2 pp on the all-recurrent parent and/or Clef-flash | ≈ $6–9 across parents, Clef-flash ≈ $3 | new `safety_eval/collapse.py`, backbone hook |

If a direction touches the owners' critical path (anything in `jobs/DECISION_TREE.md`, `decide/`, `interfaces/`), say so in an issue first and keep your branch mergeable behind theirs.
