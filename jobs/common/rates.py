"""SF Compute (Autoresearch) H100 prices: the one place the repo's $/min constants live.

Every cost path reads from here: jobs.common.costs (worst case, quoted max_cost, expected), the lint's `rate_current`
check (each job_params.json `batch_usd_per_min` must equal BATCH_USD_PER_MIN, so the per-job copy the in-job scripts
read cannot drift), and the scorers / pricers that used to hard-code a literal.

Current rates, effective 2026-10-08 (verified 2026-10-09):

- LIST_USD_PER_GPU_HOUR = 5.04 (interactive h100 node, $0.084/min; was $3.96 = $0.066/min). Source: `autor docs`
  "Pricing" (the 1x idle window "15 min ... $1.26 at the H100 list rate" = $0.084/min) and agent B's 2026-10-08 report.
- BATCH_USD_PER_MIN = 0.063 per GPU-minute ($3.78/h = list x 0.75, the batch-job discount). What a job is BILLED
  Monday-Thursday and on weekends (the 20 % weekend discount is smaller than the job discount, so it does not move a
  job). Source: the billing journal (`autor job get <id> --json` receipt.attempts[].rate_usd_micros_per_gpu_minute):
  every attempt that ran 2026-10-08 16:59Z -> 2026-10-09 06:57Z bills 63,000 micro-$/GPU-min (charged / billed s
  62,842-63,000 after per-attempt rounding), e.g. job-2gpiv 25,883 s -> 27,176,800, job-bs5we 24,948 s -> 26,195,400,
  job-gi58k 23,864 s -> 25,057,200, job-35mak 10,408 s -> 10,928,400.
- QUOTE_USD_PER_MIN = 0.063: validate_job / submit_job preflight worst_case_usd_micros = max_duration x 63,000, build
  excluded, exact for every 2026-10-08/09 submit (job-b7xax 637 min -> 40,131,000; job-hjwpq 492 -> 30,996,000;
  job-a6vt6 742 -> 46,746,000; job-gen87 90 -> 5,670,000; job-8cd26 828 -> 52,164,000).
- FRIDAY_BATCH_USD_PER_MIN = 0.05628 ($3.3768/h = list x 0.67): Fridays (US Pacific) are 33 % off list
  (node rate.calendar: "Fridays 33% off, weekends 20% off (US Pacific)"), which beats the job discount, so minutes
  run on a Friday bill lower: job-b7xax 24,420 s -> 22,905,400 (56,278/min), job-hjwpq 15,006 s -> 14,075,600,
  job-afpc2, job-w3c2e, job-gen87 (56,168-56,279/min), all attempts started after 2026-10-09 07:00Z (Friday 00:00 PT).
  Budgets and caps never use it: a calendar day only ever lowers the rate.

Before 2026-10-08 the list price was $3.96/h and the batch rate $0.0495/min ($2.97/h; job-sgihj receipt 49,500
micro-$/GPU-min, last seen on job-2xe5w 2026-10-07). Historical analyses, LEDGER rows, receipts and job results priced
at that rate are left as they were; HISTORY and batch_usd_per_min_on() reproduce them.
"""
from __future__ import annotations

import datetime as _dt

#: effective date of the current price set (list $3.96 -> $5.04 per H100-hour)
EFFECTIVE = "2026-10-08"
VERIFIED = "2026-10-09"

LIST_USD_PER_GPU_HOUR = 5.04
LIST_USD_PER_MIN = 0.084
#: billed per GPU-minute for a batch job on a non-Friday (list x 0.75); the rate every budget and cap uses
BATCH_USD_PER_MIN = 0.063
BATCH_USD_PER_GPU_HOUR = 3.78
#: the platform's quoted max_cost_usd / preflight worst case per GPU-minute of max_duration (build excluded)
QUOTE_USD_PER_MIN = 0.063
#: billed per GPU-minute for minutes run on a Friday (US Pacific): list x 0.67. Reporting only, never a budget.
FRIDAY_BATCH_USD_PER_MIN = 0.05628

#: (effective_from UTC date, list $/GPU-h, batch $/GPU-min billed, quote $/GPU-min, source)
HISTORY = (
    ("2026-10-06", 3.96, 0.0495, 0.0495,
     "rate card 2026-10-06 ($3.96/h interactive, $2.97/h batch); job-sgihj receipt 49,500 micro-$/GPU-min; "
     "validate_job max_cost = max_duration x 0.0495 (qual 180 min -> $8.91)"),
    ("2026-10-08", LIST_USD_PER_GPU_HOUR, BATCH_USD_PER_MIN, QUOTE_USD_PER_MIN,
     "autor docs Pricing ($1.26 per 15 min at list); receipts 63,000 micro-$/GPU-min (job-2gpiv, job-bs5we, "
     "job-gi58k, job-35mak); preflight worst_case = max_duration x 63,000 (job-b7xax, job-hjwpq, job-a6vt6)"),
)


def batch_usd_per_min_on(date: str | _dt.date) -> float:
    """The non-Friday batch rate in force on a UTC date (YYYY-MM-DD), so an old cost analysis can be re-derived."""
    d = date if isinstance(date, str) else date.isoformat()
    rate = HISTORY[0][2]
    for since, _list, batch, _quote, _src in HISTORY:
        if d >= since:
            rate = batch
    return rate
