# Fixed-sequence CRR LTT on an unseen split (B-PREREG-GUAR-FSCRR): status

Pre-registered as **B-PREREG-GUAR-FSCRR** (state @b0bb96c0, pushed before any prediction on the split existed; the
sample ids are in the addendum on state). Code: this branch, `claude/ev-guar-fscrr` (child of
`claude/ev-guarantee-proofs` @01738528).

## Takeaway

- **No unseen split with predictions exists, so the table is still empty.** None of the four models has
  per-decision predictions on any non-DEV, non-TEST row. The test is fully specified, implemented and tested. The
  unseen split is built and pinned. What remains is the prediction jobs, which need a spend decision of about $35
  without D13 or about $57 with it (full-DEV $/row × 10,365 rows, plus builds).
- **Why the candidates the brief named do not qualify:**
  - `full_excl51` is DEV minus 51 rows (12,053 of 12,104), so it was already seen.
  - The local "0.3 public" index is the same DEV rows re-weighted under 0.3. Its only new rows are 262 rebuilt
    GSM8K twins, which no model has run and which are far too few for LTT.
  - TEST is sealed, and no PREREG entry allows a one-shot guarantee evaluation on it.
  - The JevBench public set has 231 items and no Clef predictions.
- **Chosen split: the Decision Index 0.2.1 POOL.** These are the scoreable index rows in neither DEV nor TEST. Slicing
  leaves at least 10% of every benchmark there, and no model or analysis has touched them.

## The unseen split (job-9na9j, CPU-only, art-stkva sha256 c3b98bb0…48af1, billed 60 s ≈ $0.06)

| What | Value |
|---|---|
| Pool rows (0.2.1 scoreable, not DEV, not TEST) | 95,961; ids sha256 `10573bb8…2c21a8` |
| Sample rows / source cases | **10,365 / 6,780**; ids sha256 **`990b5911…38e04b`** |
| Overlap with DEV / TEST manifests (re-checked locally) | 0 / 0 |
| Rows file (kept outside git: `ssdm-data/pool-sample-fscrr/`) | `a500a31a…1bef83` |
| Manifest (in `job-9na9j/`) | file `6f68a03b…51e9c`, text `c1c9e3c2…b8d5cb` |
| DEV manifest / TEST manifest used for exclusion | `1362c23e…073a` / `76a6b135…be12` |

The sample mirrors DEV's per-benchmark quotas wherever the pool allows. Eleven benchmarks have a smaller pool than
DEV's quota, so all their eligible pool rows are taken:

| Benchmark | Sample / DEV quota |
|---|---|
| BFCL | 171 / 761 |
| ToolRet | 69 / 308 |
| API-Bank | 52 / 228 |
| Home appliances | 10 / 39 |
| ContractNLI | 13 / 55 |
| GPQA | 21 / 87 |
| MuSR | 228 / 262 |
| BRIGHT | 22 / 99 |
| CRUXEval | 58 / 256 |
| HLE | 51 / 225 |
| New Yorker | 54 / 237 |

So the Tools area is under-represented relative to DEV. This changes the population the guarantee is about, not its
validity. Per `pool_sample.json`, every DEV row has at least one gold decision, so the eligibility filter removed
nothing on the DEV side.

## Code (tested)

- **`jevkit/guarantees.py`.** `ltt_select(..., method="fixed_sequence", fs_start=λ0)` is opt-in.
  - It tests the grid from λ0 downward, each λ at level δ.
  - It STOPS at the first λ that is not rejected, and also at the first λ that acts on nothing. The second stop
    fixes CONFORMAL.md D2.
  - It reports `fs_start`, `fs_stop` and `fs_tested`.
  - It works for both the binomial path and the CRR path.
  - The default output is byte-identical with and without the new keyword.
- **`tests/guarantees/test_fscrr.py`.** 39 passed, 1 skipped across `tests/guarantees`, in about 38 s.
  - **Shared-fate clusters** (sizes 1 and 16, 120 seeds, true boundary λ = 0.75): fixed-sequence CRR at c = 8
    certified in 114 of 120 seeds and violated in 4 (3.3%, below δ = 10%). Bonferroni CRR on the same data
    certified in 96 seeds. A per-decision binomial run as the same fixed sequence violated in 20 seeds (16.7%,
    above δ).
  - **Global null** (R just above ε at the start λ): 16 of 200 seeds certified (8%, at most δ).
  - **The D2 stop:** a λ that acts on nothing now ends the sequence, while the legacy code would skip it.
- **`pool_sample.py`** builds the split. **`jobs/ev_pool_sample`** runs that build as a CPU job: dry-run test, pinned
  data, and an entry in `costs.py` and `DECISION_TREE.md`.
- **`guarantees_fscrr.py`** is the analysis driver.
  - It takes the PRIMARY test fs8 plus the secondaries fs1, fs_max, bf8, cluster and published.
  - Marginal and per-area are both covered, in both configurations, with the paired bootstrap.
  - It has been smoke-tested end to end on synthetic inputs.

## What runs next (each step pre-registered; nothing changes after the fact)

1. Produce predictions on the 10,365 rows from the pinned arms, each with the 64-row DEV identity gate.
   - Arms: D19 (art-p3gpi step 2800, v3w untempered, cap 65,536), Clef 27B and Clef-flash (reference GDN,
     decide_32k_substituted). D13 is optional.
   - Cost at the full-DEV $/row: D19 ≈ $24.6, Clef 27B ≈ $6.5, Clef-flash ≈ $3.2, D13 ≈ $22.0. Builds come on top.
   - Needs new job modes that read `pool-sample-rows.jsonl.gz` instead of the DEV slice, on each arm's own lineage.
2. Run the analysis:
   `python -X utf8 -I guarantees_fscrr.py <checkout> inputs.json GUARANTEES_FSCRR.json`
   Then fill the headline table: TEST act fraction at α = 0.10, PRIMARY, marginal and per area. Add the paired
   contrasts D19 − Clef-flash and D19 − Clef 27B.

## What counts as a guarantee claim (verbatim rule in PREREG)

- **A model has an FS-CRR guarantee** only if the marginal fixed-sequence CRR test at c = 8 certifies some λ*. This
  must happen in the PRIMARY configuration (run_id split, cluster = row) at α = 0.10.
- **The claim reads:** with probability ≥ 0.90 over the LTT half of the pool sample, conditional on its FIT half,
  R_8(λ*) ≤ 0.05.
  - This holds for rows exchangeable with that sample, under any within-row dependence.
- **"D19 is more actionable than X under a guarantee"** needs two things: D19's own certificate, and a paired Δ act
  CI above 0.
- **Not claims:**
  - anything on DEV;
  - the case variant alone;
  - the secondaries;
  - R_∞, unless the uncapped secondary certifies;
  - behaviour under distribution shift;
  - the board.
