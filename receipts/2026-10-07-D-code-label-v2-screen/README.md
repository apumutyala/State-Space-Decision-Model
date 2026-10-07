# CPU code-label v2 matched-screen proof

Implementation commit: `1ff42ce3974c9569d9ce2296b8a4e52ae3b88d9b`, rebased onto
`agent-main` `6ab62f5b8f43f9f916900fb1963d37a36657dda4`. Author/committer:
Apuroop Mutyala <apumutyala@gmail.com>. No push or PR was opened.

The real `screen_job.sh` ran on Windows CPU with CUDA hidden, using scripted
vocabulary tensors and no model/checkpoint load. Actual v1/v2 backends, Decider,
production engine response mapping, strict raw JSONL readers, production Index
decomposition and 2,000-replicate paired bootstrap, calibration diagnostic and
existing raw-probability audit completed. The fixed synthetic cohort has 12
rows: 11 answered and one actual adapter input-limit refusal in each arm. All
12 remain in both score denominators. The synthetic v2-v1 interval is exactly
zero. These are fixture outputs, not learned efficacy or performance results.

`report.json`, both raw prediction files, the synthetic rows, and complete
backend identities are retained. `receipt.json` records commands, environment,
source/input/output hashes, the pinned kit commit, bootstrap seed, CPU scope,
zero billed rate/cost and N=1 dry run. No performance median/spread was measured.
The raw prediction hashes cover timing fields as well, so repeated runs can
have different file hashes without changing the fixture's score.

CPU suite: **172 passed, one upstream failure** on this rebased tree. The 33 new
screen checks and 70 inherited code-label checks passed. The remaining 69
jobs-lint checks passed. The sole failure is the unchanged
`tests/test_jobs_lint.py::test_coordination_has_no_superseded_spend_rule`, which
still asserts `requested but not blocking` in COORDINATION after PR #156 made
spend acknowledgement binding. Both the test and COORDINATION are unchanged
from the integration base; the coordinator was informed to align this stale
assertion separately. `pytest.txt` preserves the exact output. This receipt does
not claim all required repository checks are green.

Negative fixtures exercise missing/extra/duplicate/truncated rows, raw absence,
NaN/negative/unnormalized probabilities, missing/extra option IDs, all three
adapter option types and K=53 support, calibration/masking/revision/request
drift, invalid refusals, changed source/tokenizer/renderer/model/runtime identity,
inadequate per-arm/paired counts, inadequate calibration fields and absent
comparator. Hash drift is refused before row parsing. The shell rejects a
launch request and a reused output directory cannot overwrite prior evidence.

Every report fixes `promotable=false`, `beats_clef=false` and
`spend_authorized=false`; all safety, calibrated-usefulness, abstention, control,
interpretability and both Clef floors are explicitly unmeasured. No private
DEV or TEST items were opened. The real-data artifact audit was not run; only
committed DEV IDs/manifests were checked.

Next: fresh adversarial review, upstream stale-test fix and green integration.
Then separately prepare real H100 kernel parity/peak-memory evidence and a
price accounting for both padded positions and increased call count. Paid
launch registration, merged proof and an acknowledged spend issue remain
required. This is CPU preparation only and has no paid-launch configuration.
