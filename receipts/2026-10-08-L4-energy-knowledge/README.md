L4 electricity knowledge expressibility slice, based on `5fec2b2`.

Implemented in `language_growth/domains/energy.py`, with one typed host table
and one authored development example. No generic kernel/types/realizer,
neural model, held-out data, training data or paid resource changed.

The source is the published IPCC AR5 WGIII (2014) Annex III, Table A.III.2,
[official PDF](https://archive.ipcc.ch/pdf/assessment-report/ar5/wg3/ipcc_wg3_ar5_annex-iii.pdf).
Official indexed PDF text was retrieved on 2026-10-08 and used to check the
five rows' lifecycle minimum/median/maximum literals. Direct browser PDF opens
returned HTTP 403; no downloaded PDF or visual page verification is claimed.
The module records the source, 2014 assessment/table scope, gCO2eq/kWh unit,
lifecycle boundary including albedo effect, statistic identity and explicit
non-confidence-interval interpretation. Importing the extraction remains a
trusted-host premise-admission decision.

Observed local CPU validation:

- `python -m language_growth.domains.energy`: accepted the proposed Wind
  onshore unique minimum, median 11, reported minimum 7.0 and maximum 56.
  The released sentence lists the exact five-technology comparison domain.
- A direct `ReadTechnology` call for Nuclear: accepted median 12 and reported
  minimum 3.7, maximum 110.
- Direct boundary probes: wrong winner, current/unscoped question, inline
  answer value, missing competitor, revoked policy, changed unit, changed time,
  private competitor, stale fact, relabelled uncertainty and tied minimum all
  returned exactly `{"status": "unavailable"}`. A previously checked proof
  failed kernel release after its selected record version changed.
- `python -m unittest discover -s language_growth/tests -p 'test_kernel*.py' -q`:
  all existing 20 tests passed. No test-suite files were added or changed.
- `git diff --check`: passed.

This establishes a runnable, source-grounded semantic operation and deterministic
realization boundary. It is not learned L4 qualification, comparator usefulness,
decision retention, safety/alignment qualification or an intervention result.
