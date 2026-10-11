# ev_cal_land: CPU-only landing of the D19 calibrated full-DEV variant (evidence lane)

Why a job: `analysis/land_fulldev.py` needs ~4 GB RAM; the shared laptop had < 1 GB free (2026-10-10). The work is
pure CPU analysis of shipped prediction files on the SF box; no model is loaded and nothing runs on the GPU.

Inputs (sha256-pinned in `data.sha256`, checked at launch AND in the job):
- DEV rows 0.2.1 `slices/dev-rows.jsonl.gz` d1202ab8… (same pin as jobs/clef_baseline/data.sha256).
- `cand/calibrated.results.jsonl` 7c40c115…: D19 full-DEV rows (b6bab803…, job-y6ste+job-uq5p5) with T = 0.8062114
  applied by the evidence rescore's own code (softmax(log p / T) per decision; 0 argmax flips; max |dp| vs the lead's
  d6e4b10c… file 5.8e-7).
- `cand/untempered.results.jsonl` b6bab803… (identity check: must reproduce 49.49 [47.77, 50.80]).
- Clef 27B job-zfexa decide_32k_substituted f5ad788b…, Clef-flash job-t5yg2 decide_32k_substituted 2cb2e961…
  (land_fulldev's own pins).

Phases (caps, min): setup 3 (sha256sum -c), preflight 3 (`python -m jobs.ev_cal_land.preflight`: measured DEV-row load
probe x 6 x 3 passes must fit the 22-min landing cap, else exit 15), land 22 (land_fulldev for both candidates: full +
full_excl51, B = 2000, seed 0), area_ece 6 (`area_ece.py`: overall/per-area ECE for all four files), final 1; reserve 5.
max_duration 40 min. Outputs under /output/ev_cal_land/ + SHA256SUMS + MANIFEST.jsonl + EV_CAL_LAND_DONE.

Cost: worst case (60 build + 1 x 40) x $0.063 = $6.30 (costs.worst_case_usd, max_restarts 0, floor 0) = hard_cap_usd;
platform hold (max_duration x $0.063) = $2.52; expected ~$1.39.

No verdict (decision_rule null): the calibrated row is a pre-registered SECONDARY under B-PREREG-D19-TEMP; the claim
rule (B-PREREG-CLAIM + addenda) is unchanged and land_fulldev's beats_both is reported as is.
