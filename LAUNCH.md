# LAUNCH: JEV3-LM-R (modes `jev3_lm_r_s0`, `jev3_lm_r_s1`)

Prereg (draft): `docs/preregs/A-PREREG-JEV3-LM-R.md`. Spec: `jobs/m2_first_cell/JOB_SPEC.md` (modes jev3_lm_r_s*).
NOTHING here has been submitted. Run these steps in order once credit is available.

## Money per seed

| | minutes | $ |
|---|---|---|
| Platform hold at submit (quoted max_cost = max_duration 161 x $0.063) | 161 | **$10.14** |
| Worst case (build 60 + 2 attempts x 161) x $0.063 | 382 | **$24.07** |
| Expected (zbmhc phase rates: setup 3 + guard 1 + heldout 1 + cell 84 + screen 16 + report 1) + the build | 106 + build | **$6.68** + <= $3.78 |

The 954-step clock (73.1 min expected) uses the job-zbmhc gross rate of 4.6 s/step, with dev cards every 50 steps
included. The 96-min clock cap covers the slow end: job-wm8b5's 5.07 s/step plus 41-s cards. The screen cap is 33 min:
the no-proxy projection of 22.0 plus 9.0 for the one jev-heldout scoring (2,199 items) gives 31.0, within 0.95 x 33.
For both seeds the worst case is $48.14, the holds total $20.28 if both are in flight at once, and the expected cost
is about $13.4 plus two builds.

## 0. Before the first submit (once, $0 except the CPU preflight)

1. Register the prereg on state: append `docs/preregs/A-PREREG-JEV3-LM-R.md`'s bullet to `PREREG.md`. The jev-heldout
   pins are filled from pointer 34b176fa.... Get the evidence lane's APPROVE.
2. Data: `ssdm-data/m2/build/bschema-draw-jev3-lmr/` must exist and match `jobs/m2_first_cell/data_jev3_lmr.sha256`. To
   rebuild it, which is deterministic:
   `python -X utf8 -I scripts/jev3_lm_r_draw.py --data C:\Users\apumu\research\CHLabs\ssdm-data`
   then `sha256sum` the three .gz files against the pin file.
3. **Preflight** (`scripts/cell_preflight.py`). It runs the real pre-train path on the lmr draw: load_cell_data with all
   guards, the K / T filter with assert_no_truncation, the lm_label single-token labels, one tiny-model step and one dev
   card. The laptop has no torch, so run it as the cpu-8 job (about $0.27 per 30 min):
   ```
   PREFLIGHT_PARAMS=jobs/m2_first_cell/job_params_jev3_lm_r_s0.json TESTS=tests/test_jev3_lm_r.py MAX_MIN=40 \
     uv run --no-project --with zstandard python -I jobs/jev3_lm_cpu/build.py . <pushed sha> C:\Users\apumu\research\CHLabs\ssdm-data <out-pf>
   python -I jobs/common/sf_submit.py <out-pf>
   ```
   The preflight package ships the sealed `jev-heldout/items.jsonl` because it is in the params' context data. Before
   submitting, append a line to `ssdm-data/m2/build/jev-heldout/ACCESS.log` (purpose "packaging copy, preflight; not
   scored"). Do the same for each `jobs.common.launch` run below, since the launcher hashes and ships the file.
   PASS means `preflight.json` has `fail: []` and the tests passed. The setup gates `screen_projection` and `cell_budget`
   are inside it. s1 differs only by seed, so one preflight covers both.

## 1. Seed 0

```
python -m jobs.common.launch m2_first_cell --mode jev3_lm_r_s0 --sha <pushed sha> --out <out-s0> --data C:\Users\apumu\research\CHLabs\ssdm-data
python -I jobs/common/sf_submit.py <out-s0> --dry      # check: worst 24.07, key m2-first-cell-jev3-lm-r-s0-r1
python -I jobs/common/sf_submit.py <out-s0>
```
Add a planned row to `results/ledger.tsv` on state. Poll with
`MSYS_NO_PATHCONV=1 wsl.exe -d bi-ubuntu -- /root/.local/bin/autor --workspace default job get <id> --json`.

## 2. Seed 1: only after seed 0's hold releases

Submit s1 only after s0 reaches a terminal state (succeeded or failed) and `autor billing` shows the $10.14 hold
released. If s0 crashed on infrastructure, fix it, relaunch s0 under the same prereg, and only then run s1.
```
python -m jobs.common.launch m2_first_cell --mode jev3_lm_r_s1 --sha <same sha> --out <out-s1> --data C:\Users\apumu\research\CHLabs\ssdm-data
python -I jobs/common/sf_submit.py <out-s1>
```

## 3. After each landing

- Land the result (skill `land-result`): read `screen.json` `jevkit.primary` (arm native_shimcal vs stock_shim, CI), the
  G1 and score_inv KEEP gates, `slice_floor.pass` and `letter_prior` (the A-share flag).
- Replication verdict after s1: both point estimates > 0 AND s0's CI > 0.
- jev-heldout (scored ONCE in-job, in the screen): read `screen.json` `jevkit.jev_heldout`, which holds the delta
  Composite A of native_shimcal vs stock + shim with its CI. Append `jevkit/jev_heldout/ACCESS.json` to
  `ssdm-data/m2/build/jev-heldout/ACCESS.log`. Never re-score the set, and never use it for selection or fitting.
