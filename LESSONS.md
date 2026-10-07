# LESSONS — failure classes and the check that now prevents each

This file is append-only. Each entry records a failure class we already paid for (money, a run, or a review round) and the **enforced** check that stops it from recurring.

**Rule (both agents, collaborators too):** every incident adds an entry here **and** its enforcing check (a test, lint check or runtime guard) in the same PR. An entry without a check stays `OPEN` and blocks launches of the affected job.

Before any paid job, follow `jobs/LAUNCH_CHECKLIST.md`. Variants live on `stack/<family>/<variant>`. `agent-main` holds only the champion (`CHAMPION.md`) plus infra, and every measured variant gets a `CANDIDATES.md` row.

Entry format:
- `### L-NN title`
- Date / Symptom / Cost / Root cause / **Check:** (file::test) / **Status:**

Status values: `ENFORCED` (an automated check fails on recurrence), `PARTIAL` (a check covers part of the class), `OPEN` (no check yet, owner named).

### L-01 Per-option prompt recompute in the readout
- Date: 2026-10-06 (job-mhc3h).
- Symptom: rwkv7 stage A stopped with exit 15 after 25 rows in 9.5 min. It projected 763 min against a 75-min cap.
- Cost: $0.56 and one stage-A attempt.
- Root cause: every (question, option) pair re-forwarded the whole prompt, 236.7 M tokens instead of 6.2 M. The preflight priced a proxy volume 40x too small.
- **Check:** `decide/backends/state_fork.py`, with fork == recompute verified in `tests/test_readout_fork.py`. The exact-volume preflight `jobs/common/rate_guard.py preflight` is verified in `tests/test_perf_paths.py::test_cost_model_reproduces_the_measured_job_and_clears_the_cap`. Lint `rate_guard` requires the runner to call a measured-rate guard (`tests/test_jobs_lint.py`).
- PR #93 review (B2): the first `rate_guard` lint matched the word "preflight" anywhere, including a `# TODO preflight` comment. It now requires an executed `python -m jobs.common.rate_guard` / `python -m jobs.<job>.preflight` (or `bash ...preflight...sh`) line (`test_rate_guard_requires_an_executed_call`, `test_refusals[preflight_comment|preflight_echo]`). m2_pilot and m2_parent_arm (`score_parent estimate`) are waived.
- **Status:** ENFORCED. Note that PR #50 merged with Track L request-changes open (KV memory at 255 options, `shared_prefix` missing from `revision()`).

### L-02 9B OOM without micro-batching; the probe pipeline dropped `micro_batch`
- Date: 2026-10-06 (job-c4zkm; then PR #73).
- Symptom: phase C OOMed at stage 1 step 1, 8x4096 tokens, peak 68.9 GB. Later, `convert/pipeline.py` never passed `micro_batch`, so probe9b would have repeated the same shape.
- Cost: $0.74, plus a re-run (job-4j3vu, $2.48).
- Root cause: stage memory was estimated, not measured at the real shape. The fix in one path (qual) was not propagated to the sibling path (probe pipeline).
- **Check:** `tests/test_microbatch.py`, `tests/test_memmodel.py`. `tests/test_probe9b.py` asserts that the pipeline passes `micro_batch`. `jobs/probe9b/preflight.py` exits 5 when a cap is under 1.3x its projection.
- **Status:** ENFORCED for qual/probe9b. The class rule ("grep sibling paths when fixing one") is a review item, LAUNCH_CHECKLIST row 13.

### L-03 fla #640 Triton bug on Hopper
- Date: 2026-10-06 (job-jbw22).
- Symptom: the run stopped at the numerics gate after 83 s. fla 0.5.2 refuses the gated chunk backward with Triton 3.4–3.7.0.
- Cost: $0.07.
- Root cause: the image stack was not pinned and not probed on the target GPU before the paid phases.
- **Check:** the image is pinned by digest (torch 2.13.0, triton 3.7.1), `FLA_TILELANG=0`, with a build-time version assert. `jobs/common/fla_guard.py` (exit 5) runs before any download. `tests/test_fla_hopper_fix.py`.
- **Status:** ENFORCED.

### L-04 First-call JIT compile counted inside a capped phase
- Date: 2026-10-06 (job-4idg4).
- Symptom: exit 7, the kernel phase timed out at 4 min.
- Cost: $0.30.
- Root cause: Triton compile and autotune (~70 s per specialisation) ran inside a measured, capped phase.
- **Check:** a separate timed `warmup` phase and a shared `TRITON_CACHE_DIR`. Phase caps come from `expected_minutes`, and the per-job tests assert cap >= 1.3x expected (`tests/test_fla_rwkv7.py`). Lint `schedule` requires caps + reserve == max_duration.
- **Status:** ENFORCED. The review fixes (B1 partial `fp32.json`, N1 autotune at T=64) merged the same minute without re-review; see L-16.

### L-05 Unmeasured warmup cap
- Date: 2026-10-06 (job-jucxx).
- Symptom: exit 7, the warmup cap of 3 min was hit, plus one preempted attempt. Measured warmup was 72.6 s and 69.7 s per specialisation, so 2 of 8 fit.
- Cost: $0.25.
- Root cause: the cap was a guess with no receipt behind it.
- **Check:** every new cap needs an `expected_minutes` entry from a receipt, and the per-job test asserts the cap margin (`tests/test_fla_rwkv7.py`, `tests/test_cell_step_rule.py` at >= 1.5x). Lint `schedule` is in `tests/test_jobs_lint.py`.
- **Status:** PARTIAL. The margin test is per job; the lint does not yet require `expected_minutes` on every params file (qual and m2 lack it).

### L-06 Gate reference that the production path itself fails
- Date: 2026-10-06 (job-jn47u, fla gate v1).
- Symptom: verdict REJECT_NUMERICS. The bf16 token-serial eager reference also rejects the production `chunked` path (0.02–0.37 relative error).
- Cost: $0.84, plus rule v2 and job-mden7 ($1.06).
- Root cause: the reference was never run through the production path on CPU before spend. Review flagged it as non-blocking (PR #53 N3).
- **Check:** RULE v2 uses fp64 token-serial truth. `tests/test_fla_rwkv7.py` runs the production path against the reference on CPU. LAUNCH_CHECKLIST row 12.
- **Status:** PARTIAL. This is enforced for fla only; there is no generic lint (a reference sanity test is a review item).

### L-07 Thresholds written after seeing data
- Date: 2026-10-06.
- Symptom: the probe9b dNLL/KL/cos bands were written after job-4j3vu's numbers (KL 0.054). fla RULE v2 counted an OOM step as 0 tok/s, which can give a +inf ratio. `KERNEL_TOL` 2e-2 and the qual MFU floors (1450 → 2001 tok/s) moved without a rule record.
- Cost: review rounds, plus a credibility risk on every verdict.
- Root cause: there was no pre-registration discipline. Rule constants lived in code that changed alongside results.
- **Check:** a rule module is committed before the run, with literal constants pinned by a test. Lint `decision_rule` (`tests/test_jobs_lint.py`) requires this for every verdict job. Post-hoc bands are reported only as `diagnostic_bands_posthoc`, with the binding rule pinned to 511e8dd. Amendments are dated, committed before any run (A1 a714638, A2 1c33d84).
- PR #93 review (B3): new jobs must now declare `decision_rule`, either a rule path or `null` with a `no_verdict_reason`. A JOB_SPEC saying "PASS if ... else FAIL" without the word "verdict" no longer counts as "no verdict" (`test_new_job_must_declare_decision_rule`, `test_refusals[no_decision_rule]`).
- **Status:** ENFORCED for new jobs. rwkv7 is waived (`lint.WAIVERS`); m2_pilot margins are PENDING USER APPROVAL.

### L-08 Cost and schedule arithmetic drift
- Date: 2026-10-06.
- Symptom: the worst case quoted in specs relied on "only the running attempt is billed". Jobs were absent from `costs.PARAMS` or `DECISION_TREE.md`, so the envelope ledger missed them.
- Cost: none billed. This is the risk of an over-cap submit.
- Root cause: the numbers were hand-copied between params, spec and tree.
- **Check:** lint `worst_case_stated` requires `costs.worst_case_usd` (build + 2 attempts) printed in JOB_SPEC.md and <= hard cap. Lint `registered` requires the job in `costs.PARAMS` and `DECISION_TREE.md`. `launch.py` prints `set_workspace_cap_increment_usd`.
- PR #93 review (B1): `launch.py` silently defaulted `hf_cache` to false. The qual JOB_SPEC and job-4j3vu use true, so the default would have re-downloaded 19 GB inside the capped run. Every submit field must now be stated in job_params or the submit block (`REQUIRED_FIELDS`, `test_refusals[hf_cache_missing]`). `test_submit_args_equal_the_job_spec_field_by_field` parses every JOB_SPEC submit block and compares every field.
- **Status:** ENFORCED. m2_pilot and m2_parent_arm are waived as DRAFT.

### L-09 Image publish timeout at build_timeout 18
- Date: 2026-10-06 (issue #68; jn47u-era fla jobs).
- Symptom: "publishing the built image timed out". The platform docs say a resubmit does not fix this.
- Cost: lost queue time and a re-submit. probe9b's worst case was re-priced to $29.70 at 60 min.
- Root cause: `build_timeout_minutes` was sized for the Dockerfile steps, not the push of a ~15 GB image.
- **Check:** lint `build_timeout` requires >= 60 (`tests/test_jobs_lint.py`).
- **Status:** ENFORCED for new jobs. 9 legacy params files are waived until their next launch; `launch.py` refuses them without `--allow-waived`.

### L-10 Literal "\n" in a shell runner
- Date: 2026-10-06 (PR #41 `jobs/common/build_context.sh`; PR #74 `first_cell_job.sh`).
- Symptom: a literal backslash-n before `--out` passed a stray `n`. argparse exits 2, then the runner exits 5, after the paid parent arm.
- Cost: caught in review on PR #74. It passed every Python-level test and two reviews.
- Root cause: tests imported the Python pieces and never executed the `.sh`. The old regex only matched `\n\s+\|\|`.
- **Check:** lint `no_backslash_n` (regex `[\\]n[ \t]+--` over `jobs/**/*.sh`), with `test_backslash_n_regex_fires_on_known_bad_and_not_on_good`. Lint `shell_test` requires a test that executes the runner in a subprocess. `test_runner_execution_detector` checks that reading the runner's text does not count.
- **Status:** ENFORCED. The 9 legacy params files without a shell test are waived (`lint.WAIVERS`); only cell_step_profile has one.

### L-11 Heredoc collapsing backslashes, so a regex test went vacuous
- Date: 2026-10-06 (PR #74 self-test). It recurred while writing this lint.
- Symptom: a regex written through a shell heredoc lost its backslashes and matched nothing, so its test passed vacuously. While building `jobs/common/lint.py`, a heredoc turned `\b` into a 0x08 backspace inside a regex, and the check silently failed.
- Cost: one review round; caught locally the second time.
- Root cause: source code was written through a shell quoting layer, and tests asserted only "no hits".
- **Check:** every regex test asserts it fires on a `chr(92)`-built bad sample and not on a good one (`test_backslash_n_regex_fires_on_known_bad_and_not_on_good`). `test_no_control_characters_in_sources` scans `jobs/ tests/ scripts/`. Write code with an editor or `Write`, never a heredoc.
- It happened a third time in the PR #93 fix round: backslash-n escapes in a generated test, passed through a shell heredoc into a Python string, became real newlines (a SyntaxError, caught by the test run). The tests now build newlines with `chr(10)`.
- **Status:** ENFORCED.

### L-12 Cell spec memory estimate wrong by 2x
- Date: 2026-10-06 (job-mden7).
- Symptom: chunked WKV OOMed on 17/20 steps at 84 GB. Activations were ~1.0 MB/token/layer with no checkpointing; the spec said 59 GB at T=2048.
- Cost: $1.06 (shared with the L-06 run), and the first m2 cell was blocked.
- Root cause: the memory figure was an estimate, never measured at the real shape before the spec was priced.
- **Check:** `jobs/cell_step_profile` measures peak per variant with a pre-registered rule (`tests/test_cell_step_rule.py`). Cache-fork-compatible block checkpointing is verified in `tests/test_cell_perf.py`. LAUNCH_CHECKLIST row 13.
- **Status:** PARTIAL. There is no generic lint requiring a measured peak per params file.

### L-13 Launch-bound train step
- Date: 2026-10-06 (job-mden7).
- Symptom: 15–17 s per step regardless of token count. That was 1.4–1.8 M aten ops at ~11.3 µs each, about 260 tok/s.
- Cost: shared with L-12.
- Root cause: a pure-PyTorch per-token / per-row loop. Throughput was never profiled before pricing a 190-min cell.
- **Check:** batched rows (`row_batch`/`row_tokens`) and `chunked_lean` WKV, equivalence-tested in `tests/test_cell_perf.py`. `jobs/cell_step_profile` profiles before any cell spend, and its rule is pinned in `tests/test_cell_step_rule.py`.
- **Status:** PARTIAL. The fix is measured only after that job runs.

### L-14 CRLF `*.sha256` manifests
- Date: 2026-10-06 (PR #74, 5fefb65).
- Symptom: `*.sha256` files shipped CRLF from a Windows `git archive` (`core.autocrlf=true`), so `sha256sum -c` fails and setup exits 10 on the box.
- Cost: caught before spend.
- Root cause: `.gitattributes` pinned only sh/py/json/Dockerfile.
- **Check:** `.gitattributes` adds `*.sha256 text eol=lf`. `test_gitattributes_pin_lf` and `test_git_archive_head_has_no_cr_even_with_autocrlf` (`tests/test_jobs_lint.py`). `launch.py` archives with `core.autocrlf=false` and refuses CR in shipped `*.sh/*.sha256/*.json/Dockerfile` (`tests/test_launch.py::test_refusals[crlf_sha256]`).
- **Status:** ENFORCED. Existing Windows checkouts still hold CRLF working copies of the `*.sha256` files; blobs are LF and `git archive` is now LF.

### L-15 Clef composite crash: U+2028 + `str.splitlines()` (first recorded as a "truncated native file")
- Date: 2026-10-06 (job-sgihj). Corrected 2026-10-07 (PR #98), reproduced from the job's artifact art-8y6d8.
- Symptom: `summarize.py composite` raised `JSONDecodeError: Unterminated string`. The runner exited 12, so S3 scoring, the S4 safety battery and S5 never ran, and the Index score was recomputed off-box.
- Cost: $1.40 was spent; the Index number was recovered off-box (56.79 [55.40, 58.22]); the Clef-flash safety battery is still unmeasured.
- Root cause: **no file was truncated.** Both results files were complete (12,104 / 115 rows, final newline). `summarize.py` split them with `str.splitlines()`, which also breaks on U+2028/U+2029, U+0085, \v, \f and \x1c-\x1e. The raw U+2028 inside row `25:GPQA-Diamond:GPQA-Diamond:test:4` of the **decide** file (the kit writes `ensure_ascii=False`) cut one record in two. The opposite, silent failure was latent too: the kit's `load_results` stops at a partial last line and scores the missing rows as unanswered. This was a Track L blocking finding on PR #19, which merged as "review stalled" without a fix.
- **Check:** `harness/jsonl.py` splits on newlines only and raises on a truncated tail, a malformed line, a duplicate `run_id` or missing slice/subset rows. `harness score|compare|ece` and `jobs.common.dev_subset` exit 3 (before #95's `require_results`). The composite and the 32k substitution are fail-closed (exit 3/4), and `run_job.sh` maps them to 17. Tests: `tests/clef/test_results_strict.py` (U+2028 reproduction, every refusal, missing row -> exit 3 through both CLIs on the merged tree) and `tests/clef/test_clef_job_sh.py` (executes `run_job.sh`; a fault-injected truncated or dropped decide row must end in exit 17).
- **Status:** ENFORCED.

### L-16 Merges without (re-)review
- Date: 2026-10-05..06.
- Symptom:
  - No review comment at all on PRs #26, #35, #47, #54, #56, #57, #66, #73 and #86. #73 and #86 claim an independent review only in the merge text.
  - Merged over open Track L request-changes: #19, #48 (6 s after it), #50 and #29.
  - Blocker fixes merged in the same minute with no re-review: #43, #59, #42 (B3), #73 and #86.
- Cost: L-15 is a direct consequence; the other unfixed findings are listed in PRs #33 and #40.
- Root cause: the review rules ("stalled", "one review suffices", "lean") had no mechanical gate.
- **Check:** `launch.py` refuses any sha not merged into `origin/agent-main`. COORDINATION.md "Before any paid job / before any PR" requires the review link in the launch receipt.
- **Status:** PARTIAL. Review presence is not machine-checked (that needs the GitHub API). Next step: a merge-time check that the PR has a review comment, or a `reviews/` file, newer than its last commit.

### L-17 Variants and champion mixed on the integration branch
- Date: 2026-10-06 (operator directive).
- Symptom: hypothesis variants, infra and results shared `agent-main` with no record of which candidate was best.
- Root cause: there was no branch-per-variant rule and no champion or candidate ledger.
- **Check:** every variant goes on `stack/<family>/<variant>`. `CHAMPION.md` and `CANDIDATES.md` must exist, and every candidate row needs a branch and a receipt (`tests/test_jobs_lint.py::test_champion_and_candidates_ledgers`). The docs must carry the rule (`test_docs_carry_the_branch_rule_and_the_incident_rule`).
- **Status:** ENFORCED (structure). The promotion rule itself (paired CI > 0 plus G12/G13 plus review) is a review item.

### L-18 Contradictory coordination rules
- Date: 2026-10-06 (found by the coordinator's pre-mortem).
- Symptom: COORDINATION.md "Stalled cross-review" said "This rule never applies to `main` or to spend approval. Spend still needs the other agent's acknowledgement on a `spend` issue." The later user decision, "Review sufficiency", says one independent review is enough to launch spend. An agent could follow either rule.
- Cost: none billed. The risk was a blocked launch or an unauthorized one, depending on which rule was read.
- Root cause: a newer user decision was added as a new section, and the older sentence was never edited.
- **Check:** the stale sentence was replaced so that the later user decision governs. `tests/test_jobs_lint.py::test_coordination_has_no_superseded_spend_rule` asserts that the old sentence is gone.
- **Status:** PARTIAL. Only this sentence is checked. Spend-ladder item 2 ("acknowledged by the other agent") still reads as blocking and needs the same reconciliation. When a user decision supersedes a rule, edit the old rule in the same PR.

### L-19 The benchmark edition changed under us
- Date: 2026-10-06 (Track D, found via research PR #92; fixed in PR #95).
- Symptom:
  - GOAL/PLAN targets (T1–T3) and every DEV comparison were stated on Decision Index 0.2.1.
  - The board switched to 0.3 the same day: an 80%-private Full score with equated parts, GSM8K rebuilt, ForecastBench retired, WinoGrande moved to Knowledge, a mean/p80 latency gate and a 0.9 tie band.
  - The kit's 0.3 code was still an unmerged PR at the time.
  - PR #92 also misstated two details: it omitted the equating of the private parts, and it called the GSM8K artifact the "centre option" when it was the centre *value*.
  - A first harness fix still applied 0.2.1 weights silently to 0.3 bootstrap replicates (38.26 vs 38.96) when no edition was passed.
- Cost: no money. One review round, plus the stale-target risk on every plan written that day.
- Root cause:
  - Targets and the scorer were pinned to an edition without a check against the live board.
  - The edition was a global default instead of a property carried by the data.
- **Check:**
  - `harness/tests/test_edition03.py::test_vendored_files_match_pins` and `::test_full_score_reproduces_every_board_entrant` (board data pinned by sha256, Full/ties reproduced for all 112 entrants).
  - `::test_03_public_index_reproduces_board`.
  - `::test_summaries_carry_and_enforce_the_edition`: replicates carry their edition; mixing editions raises.
  - `::test_03_compare_and_family_split` (compare refuses mixed editions).
  - Rule for plans: before any plan, pre-registration or spend decision, re-fetch the board's edition file and methodology and record their sha256 and `generated_utc` in the receipt.
- **Status:** PARTIAL.
  - The harness side is ENFORCED: edition-flagged, 0.2.1 kept as the default for receipt reproducibility, mixed editions refused.
  - The "verify the live board before each plan" step is a procedure, not an automated check. Owner: Track D. Candidate check: a launch-checklist row comparing the live `v03.json` sha256 with `harness/config.BOARD_V03_SHA256`.

### L-20 Size the evaluation by simulation; never loosen safety margins for power
- Date: 2026-10-07 (issue #36). Codex's cross-track decision, comments 6028537932 and 6028610849, governs. PR #102 first adopted the earlier independent review instead, and the independent review of #102 (B1) caught it.
- Symptom:
  - The first margin proposal (m_cap 1.0 on the 2,008-row subset, m_ece 0.01, m_ret_acc 0.01, two-way steer CI) sat inside the measured noise. A genuinely unchanged candidate almost never passed.
  - The independent review "fixed" this by loosening safety and retention margins for power: m_ret_acc 0.01 -> 0.03, and retention ECE gated only on a demonstrated regression.
  - Codex rejected that: low power means the comparison is INCONCLUSIVE, so expand the frozen evaluation or do not promote. Separately, `train.m2_gates` exited 0 while the margins were pending.
- Cost: no run billed. One review round, and an approval flip that would have promoted candidates the delegated approver's contract rejects.
- Root cause: margins and evaluation sizes were chosen separately. Nobody computed, per criterion, the frozen-set size at which an unchanged candidate passes the APPROVED margin, so "low power" was misread as "margin too strict".
- **Check:** `tests/test_margin_sim.py` covers sizing and pins the margins:
  - For each criterion it simulates the frozen-set size an unchanged candidate needs to pass with probability >= 0.8 at the approved margins (`train/margin_sim.py::required_n`: measure the CI half-width at the configured n, scale it, then confirm by simulation at the required n).
  - It asserts that each configured evaluation size (`EVAL_SIZES`) meets that requirement, or that the shortfall is recorded with an expansion plan (`SHORTFALLS`). The simulated and recorded shortfall sets must be equal.
  - It also pins the margins to the contract (`test_margins_are_the_codex_contract_and_never_loosened_for_power`).
  - At run time, `train.promotion.resizing` reports n required vs n evaluated at the MEASURED SE, and labels an underpowered failure `inconclusive_underpowered` (not promotable).
  - `train.m2_gates` never exits 0 unless the run is promotable: 43 while margins are pending, 44 for partial-stage runs.
- **Status:** PARTIAL. Recorded shortfalls: the retention role (1,420 items) needs ~9.3k items for `g13_retention_acc` and ~31k for `g13_retention_ece` (simulated; discordance and ECE arms are assumptions). Until cohort v3 expands it, G13 does not promote on retention. Every simulation input except the capability SE is an assumption; the first real runs replace them.

### L-21 Prefix-fork state copies OOM on many-option rows (POP909-CL backend_error)
- Date: 2026-10-07 (root-caused from job-5cmat logs; predicted on PR #50 by Track L, "KV memory at 255 options", merged over it, see L-01 / L-16).
- Symptom: on job-5cmat (Qwen3.5-9B, H100 80 GB) 58 of 59 POP909-CL rows in the stock arm and 39 of 59 in the converted arm returned `error/backend_error ... details withheld`. 19 errors were arm-asymmetric, which made G13 capability and ECE non-evaluable on every run (fail-closed).
- Cost: part of job-5cmat's Index phases; G13 non-evaluable until fixed.
- Root cause: a POP909-CL row has one question with 129 options. The JRT prefix shared by all options is the whole prompt (|P| ~6-8.7k Qwen tokens, a 1-token question suffix), so the prefix-fork packed all 129 short option tails into one call: the 4,096-position tail budget was the only cap. `extend` gives every packed row its own copy of the prefix state, KV cache included, and SDPA repeats the KV heads per layer on top, so memory grows as rows x |P|. Evidence: the `CUDACachingAllocator` OOM warnings (requests of 2.3-9.2 GB with < 8 GB free) fall exactly inside the POP909 block of each arm's log. In the stock arm the outcome splits cleanly at |P|: the one ok row has |P| = 6,087 and every row from 6,226 up OOMed. The converted arm keeps one attention layer fewer (smaller KV), so its threshold is higher and fragmentation-dependent: ok up to 7,556, errors from 6,869. That is the 19-row asymmetry. The adapter maps the exception to `backend_error`. The tail budget bounded the logits, never the state copies. The per-token cost (~44 KiB per prefix-token copy on Qwen3.5-9B) is a memory-model estimate, not a measurement.
- **Check:** `decide.backends.state_fork.fork_plan(state_budget=...)` caps rows x |P| per pack. `ModelPlugin.fork_state_token_budget` = `KV_STATE_TOKEN_BUDGET` (262,144 prefix-token copies, about 11 GB on Qwen3.5-9B) is set for every plugin that keeps a KV cache (`load_qwen35`; `convert.readout` whenever attention layers remain). Recurrent-only plugins (RWKV-7, all-GDN Q9-AF) keep `None`, and their revision ids are unchanged. Tests in `tests/test_readout_fork.py`: `test_state_budget_caps_rows_x_prefix_per_pack` (the POP909 shape), `test_state_budget_changes_packing_not_scores` (budgeted vs unbudgeted: RWKV7 bitwise; KV-cache models within 1.5e-5 absolute, measured by the #102 review; paired comparisons are unaffected because both arms on one tokenizer get the same packing) and `test_kv_cache_plugins_carry_the_state_budget_and_recurrent_ones_do_not` (behavioural, on a real tiny ConvertibleQwen: all-GDN -> no budget, unchanged revision). Scoring-time failures are never excluded from the frozen full-DEV row set (`jobs/common/dev_full.py`, `tests/test_promotion_margins.py::test_freeze_excludes_only_pre_forward_refusals_and_is_pinned`).
- **Status:** PARTIAL. The budget is from a memory model, not yet a GPU measurement at the POP909 shape. The first full-DEV Index run must show 0 `backend_error` rows, and its receipt closes this entry. The adapter still withholds the exception class; a non-secret `resource_exhausted` diagnostic code is a follow-up.

### L-22 Shared scratchpad file names collided across agents
- Date: 2026-10-07 (PR #102).
- Symptom: two agents in one session wrote PR descriptions to the same scratch file name (`pr_body.md`). The #102 agent then ran `gh pr edit --body-file` after its own write had failed. That published another PR's description (a q9 feature-cache PR stacked on #91) as #102's body. It was restored minutes later.
- Cost: none billed. A wrong PR description was briefly public, so reviewers could have reviewed the wrong text.
- Root cause:
  - The session scratchpad is shared by every agent in the session, and file names were generic.
  - The publishing command was chained after a failing step with `;` and a newline, not `&&`.
  - The published body was not read back.
- **Check:** procedure, because the scratchpad is outside the repo and has no automated check:
  - prefix every scratch file with the PR or task id (e.g. `pr102_*`);
  - chain publishing commands with `&&` after the step that writes their input;
  - after `gh pr edit`/`gh pr create`, read back `gh pr view --json body` and compare its first line with the intended text.
- **Status:** PARTIAL. No automated check yet; the procedure is the guard. Candidate check: a `jobs/common/launch.py`-style wrapper for PR body publishing that refuses a body file whose first line does not name the PR's branch or issue.

### L-23 An agent killed processes by name on the shared laptop
- Date: 2026-10-07 (reported by the coordinator).
- Symptom: an agent ran `taskkill /F /IM python.exe` with a window-title filter on the shared laptop. That kills other agents' and Codex's processes too, including their test runs and local servers.
- Cost: possible silent loss of other agents' in-flight work. Test runs that die mid-way look like hangs or truncated output.
- Root cause: the process was selected by image name or window title, not by a PID the agent itself had started and recorded. Several agents and Codex share one machine and one `python.exe` image.
- **Check:**
  - `tests/test_no_kill_by_name.py` greps `jobs/`, `scripts/`, `tests/` and the agent docs (`*.md` at the root and under `jobs/`) for `taskkill /IM` or `/FI`, `pkill` and `killall`, and fails on any hit.
  - PID-based use is allowed (`taskkill /PID`, `kill <pid>`).
  - The check is tested on chr-built bad samples, as L-11 requires.
  - COORDINATION.md "Before any paid job / before any PR" carries the rule.
- **Status:** PARTIAL. Ad-hoc shell commands typed by an agent are not in the repo, so only the rule (COORDINATION.md) covers them. Kill only PIDs you started and recorded.

### L-24 Orphan conversion probes survived a retrained checkpoint
- Date: 2026-10-07 (cross-track post-hoc review of PR #94; CPU reproduction).
- Symptom: a crash after the conversion probe's JSON write but before the progressive-step record was persisted
  left a cached `probe_convN` measurement. Resume retrained that step, then reused the old measurement for new weights.
- Cost: none billed for discovery or repair. Previously completed results are not presumed affected; the failure
  requires an interruption in this specific write gap.
- Root cause: `run_progressive` saves the next checkpoint, runs its probe, then persists the step. Probe cache keys
  encode only conversion count, and time-limited training need not reproduce identical weights on a retry.
- **Check:** `convert.pipeline.Results` discards conversion probes not referenced by any persisted accepted step
  at startup and records the discarded keys. `tests/test_full8_orphan_probe.py` injects a crash immediately after
  the real `Results.set` probe write at both the first and final steps, makes retry weights differ, and requires
  resumed metrics to match the final saved weights. It also preserves stock probes and completed curve points.
- **Status:** ENFORCED.
