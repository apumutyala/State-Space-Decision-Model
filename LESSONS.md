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
- PR #93 post-hoc cross-track review (2026-10-06): the pin detector accepted `assert R.MIN_GAIN` or an equality for only one constant, leaving other thresholds free to drift. **Check:** `decision_rule` now parses import-bound AST equality assertions and requires every module-level literal constant to match its exact literal; tuple assertions and required `and` clauses count. `test_decision_rule_rejects_unpinned_constants` rejects truthiness, partial coverage, wrong literals, self-comparisons, optional `or` clauses, comments and unrelated imports. `test_decision_rule_accepts_direct_and_tuple_equality_pins` preserves the existing rule-test styles; `test_remaining_legacy_rule_constants_are_pinned` fills existing omissions without changing thresholds. No paid job was launched for this fix.
- PR #99 second review (2026-10-06): merging import bindings across scopes left stale aliases after reimports and let helper-local imports or assertions falsely pin a rule. **Check:** bindings are now resolved separately at each direct assertion in a collected test function/method; reimports, assignments, parameters and lexical local names invalidate shadowed aliases. Helpers and nested/conditional assertions do not supply pins. `test_rule_pin_bindings_respect_test_scope_and_shadowing` reproduces 13 false-positive cases, and `test_rule_pin_bindings_keep_module_and_test_local_imports_separate` preserves valid module/local imports and test methods. No paid job was launched.
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

### L-15 Clef composite crash on a truncated native file
- Date: 2026-10-06 (job-sgihj).
- Symptom: the composite step crashed on a truncated JSONL line, and the run was scored locally.
- Cost: $1.40 was spent; the result was recovered off-box.
- Root cause: `jobs/clef_baseline/summarize.py` parses JSONL with no tolerance for a partial last line, and the composite silently falls back. This was a Track L blocking finding on PR #19, which merged as "review stalled" without that fix.
- **Check:** none yet.
- **Status:** OPEN (owner: Track D). Needed: a truncated-line test for `summarize.py` and a fail-closed composite, before any further Clef job.

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
