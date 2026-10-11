# ev_pool_sample: CPU-only build of the B-PREREG-GUAR-FSCRR unseen split (evidence lane)

Why a job: building the sample streams the whole 0.2.1 suite twice and holds a union-find over the pool's link keys
(a few hundred MB); the shared laptop had < 0.5 GB free (2026-10-11). Pure CPU; no model, no prediction, no GPU.

What: `receipts/2026-10-11-ev-guar-fscrr/pool_sample.py`. Pool = scoreable 0.2.1 index rows in neither the DEV nor the
TEST manifest (TEST is read as run ids only). Eligible = pool rows with >= 1 gold decision. Per benchmark, quota = DEV
rows of that benchmark with >= 1 gold decision; source cases (harness.cases over the pool) with more than
max(1, int(0.25 x quota)) rows are skipped; cases in sha256("B-PREREG-GUAR-FSCRR|pool|{b}:{case}") order are taken
whole until the quota is met. Fixed by the prereg before any prediction exists on these rows.

Inputs (sha256-pinned in `data.sha256`, checked at launch AND in the job): the four 0.2.1 suite files under
`suite-0.2/` (selected-rows 5510346a… = slices.json's pin, added-rows 921e245c…, excluded-questions 331df32d…,
manifest ff77c78f…) and the DEV rows `slices/dev-rows.jsonl.gz` d1202ab8… (for the quotas).

Phases (caps, min): setup 3 (sha256sum -c), preflight 2 (DEV-row stream probe x suite/DEV byte ratio x 2 passes x 2
must fit the 20-min build cap, else exit 15), build 20, final 1; reserve 4. max_duration 30 min. Outputs under
/output/ev_pool_sample/: pool-sample-rows.jsonl.gz, pool-sample-manifest.jsonl.gz, pool_sample.json, run.log,
SHA256SUMS; plus /output/MANIFEST.jsonl and EV_POOL_SAMPLE_DONE.

Cost: worst case (60 build + 1 x 30) x $0.063 = $5.67 = hard_cap_usd; platform hold (30 x $0.063) = $1.89; expected
~$0.57 (9 min).

No verdict (decision_rule null): it only builds rows.
