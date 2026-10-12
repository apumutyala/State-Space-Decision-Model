# Independent rescore: job-6zs79 (A-PREREG-JEV3-SYM seed 0, eval-only screen of job-5x4vr step 200)

Evidence lane, 2026-10-12. Kit: `scripts/ev_cell_rescore.py` at claude/ev-cell-kit @ef25e482 (this branch is cut from it).
Governing rules (origin/state PREREG.md): A-PREREG-JEV3-SYM plus Amendment 1 (8ae5c82f) and Amendment 2 (e839cb61),
EV-PREREG-COMPOSITE-LOCAL, EV-PREREG-SECONDARY-AXES.

**PRIMARY verdict: NEGATIVE. The rescore AGREES with the lead.** The paired delta Composite A of native minus the in-job stock+shim
control is -36.66 [-47.97, -19.19]. The 95% CI excludes 0 on the wrong side. One-sided p = 1.000, and Holm does not allow a claim.
**G1 verdict: KEEP PASS, bitwise. AGREES.**
**Single seed, replication pending.** Seed 1 is pre-registered but unfunded, so this result is not a claim either way.

## Inputs (all downloaded by me and checked against the job records)
| item | sha256 | check |
|---|---|---|
| job-6zs79 artifact art-xkhwy.tar (6,451,200 B) | 146a6996c675d6cd... | = the job record's artifact.sha256 |
| phases setup/guard/checkpoint/screen/final.tgz | 0c73a252 / 1b2dbe18 / fd9523dd / 740da701 / 0b16d6ed | = MANIFEST.jsonl |
| package (CONTEXT_MANIFEST branch_commit) | f591d84e (stack/qwen35/jev3-sym) | = c4d5ccc0 (Amendment 2) + 2 budget-only commits (build_timeout 60, screen cap 44; worst case $11.34 instead of $7.94). Charged $1.92 (1,827 s) |
| screen.json | 10198ebc | kit diff: 97 fields, max abs d 0.0 |
| job-5x4vr artifact art-8xs32.tar (505,600,000 B) | 933cc0ac1748304b... | streamed, never saved to disk; = the job record and = checkpoint.json artifact_sha256 |
| step-200 member trainable-step0000200.safetensors | 5cb7da4246efce72... | streamed out of the tar (no disk copy); = the sidecar .sha256, latest.json best, 6zs79 checkpoint.json, identity-child.json |
| cell.json (5x4vr, copied into 6zs79) | 03ff0ba0 | identical in both artifacts; = checkpoint.json cell_json_sha256 |
| H2O target preds (job-4g6m8), my copy from claude/lead-h2o-eval @f5615683 receipts/2026-10-11-h2o-host-eval/arms/h2o/preds.jsonl | e00d5ac7 | = the in-artifact target_ref.preds.jsonl |
| price (Qwen3.5-4B, deepinfra 0.03/0.15) from claude/ev-x7c42-rescore @bf706854 | c3fd78b3 | = the in-job jevkit.price |
| flip baseline receipt claude/ev-flip-rate @4a65dd18 run2/flip_ci_8op.json | c7f7c930 | qwen35_4b_shim 7.55% |

The command is below. `--mtimes` is landing.json converted to the kit's {name: epoch} map by `tools/landing_to_mtimes.py`.
```
python -I -X utf8 scripts/ev_cell_rescore.py --art <6zs79 final.tgz extracted> --control stock_shim --primary native \
  --out primary_native --price inputs/price_qwen35_4b.json \
  --slice-ids-sha 844233f228b718edeffb198f622fc9c8a605eed3a346c128993f8aac80453078 \
  --target "h2o=<h2o preds e00d5ac7>::inputs/price_qwen35_4b.json" --mtimes inputs/mtimes_6zs79.json \
  --identity-child <art>/out/screen/identity-child.json --g1 native=<art>/out/screen/jevkit/g1/native \
  --g1 stock_shim=<art>/out/screen/jevkit/g1/stock_shim --g1-max-dp 0 --flip8 <art>/out/screen/jevkit/flip8 \
  --flip-baseline-receipt "inputs/flip_ci_8op.json::qwen35_4b_shim"
```
I ran the same command again with `--primary native_shimcal` (no g1/flip8) and saved it in `secondary_native_shimcal/`. That run is only a SECONDARY readout.

## Lead's claims, number by number
| claim (lead) | rescore | verdict |
|---|---|---|
| PRIMARY delta Comp A -36.66 [-47.97, -19.19] | -36.6632 [-47.9664, -19.1925]; 97.5% [-49.42, -16.28]; 99% [-51.73, -14.02]; p 1.000 | **AGREES** |
| Comp A 29.98 (native) vs 66.64 (stock-shim) | 29.981 vs 66.645 | **AGREES** |
| delta I -19.21 | -19.21 [-28.88, -8.72] (36.02 vs 55.23) | **AGREES** |
| G1 KEEP PASS: 0 flips / 0 invalid on 565 reverse/shuffle/relabel pairs, max abs dp 0, bitwise | 213 + 213 + 139 = 565; 0 flips, 0 invalid, max abs dp 0.0 per op; bitwise | **AGREES** |
| 8-op flips mean 4.14% vs stock-shim 7.55% | 4.14% [2.94, 5.35] vs receipt 7.55% [5.71, 9.67] (not paired: the baseline comes from another job) | **AGREES** |
| per op: whitespace 10.4, option_para 8.7, markup 6.9, instr 5.2, state 1.9, reorder/relabel 0 | 10.39 (24/231), 8.66 (20/231), 6.93 (16/231), 5.19 (12/231), 1.92 (2/104), 0/0/0 | **AGREES** |
| slice floor +1.08 pt, McNemar p 0.53 | +1.083 pp (0.6783 vs 0.6675, n 1,200; 193 vs 180 discordant), McNemar p 0.534 | **AGREES** |
| slice floor CI [-2.08, +4.25] | iid rows [-1.83, +4.09]; family-stratified [-2.00, +4.00] | **DOES NOT REPRODUCE** (the bounds differ by <= 0.25 pt; the lead's bootstrap scheme is not stated). Either way the CI spans 0 |
| order OK: slice_first 08:55-09:00Z < first public pred 09:10:24Z | slice_first 08:55:06Z to 09:00:43Z; tfit.json 08:59:02Z; slice_floor 09:00:43Z; first public pred native.preds.jsonl 09:10:24Z | **AGREES** |
| gap to the H2O target (e00d5ac7) -46.80 | -46.80 [-59.36, -30.87]; control minus H2O -10.14 [-26.64, -4.85]. Cross-job: S 85.00 vs 92.12 and K 61.28 vs 61.13 come from different jobs | **AGREES** |
| cause: the noul band gate FAILS (65% vs 0%) | native 48/74 = 64.9%; native_tfit 48/74; stock_shim 0/74; native_shimcal 0/74; H2O 0/74 | **AGREES** |
| native_shimcal 65.57, delta -1.07 [-22.47, +7.47] | 65.573; -1.0713 [-22.4652, +7.4682]; p 0.772; delta I -3.36 [-12.70, +6.21] | **AGREES** (SECONDARY; see below) |
| T stayed 1.0 | T = 1.0 (constrained). The NLL minimiser T 1.297 would raise the slice's yes/no band rate from 0.498 to 0.587, so the pre-registered rule returns T 1.0. native_tfit = native (Comp A equal to 4e-11) | **AGREES** |
| on I: fact 83.3 -> 0, adequacy 50 -> 0 | fact (n 12) I 83.33 -> 0.00; adequacy (n 12) I 50.0 -> 0.0 | **AGREES** |

## Checks the coordinator asked for
1. **LINEAGE: PASS.** `primary_native/lineage_check.json` gives `lineage.ok = true`.
   - The scored ckpt sha 5cb7da42 matches every record: it is the member I streamed out of art-8xs32, the sidecar, 5x4vr latest.json `best` (step 200), 6zs79 checkpoint.json and identity-child.json.
   - 5x4vr cell.json `best` = step 200, which is the NLL argmin over the dev cards (step 0 0.9013, step 100 0.6910, step 200 0.6150).
   - The dev cards are 360 rows, hash-stratified from the jev3 held-out slice (ids sha 844233f2, 1,200 rows, 0 states shared with the 5,960 training states). Cards ran every 100 steps. Step 258 (the last step, clock-bound) has no dev card, so step 200 was the last selectable step.
   - Selection order: 5x4vr dev_cards/card-step0000200.json 06:39:49Z; cell.json 06:50:21Z.
   - 5x4vr wrote **no** public pred: its screen stopped at identity (exit 10, units_done 0), and none of its 6 phase tgz contains a preds or results file.
   - The first public pred in either job is 6zs79's native.preds.jsonl at 09:10:24Z. Selection therefore came before any public pred.
   - The kit raises a flag here ("scored != selected"). It is only a naming difference: training names the file ckpt-step0000200.pt and the export names it trainable-step0000200.safetensors, both at step 200, and the sha chain above links them.
   - Also on record: one collapse event in 5x4vr at step 138 (train-loss trigger; it reloaded the best step 100 at lr x 0.5). Training then continued, and step 200 beat step 100 on the slice.
2. **Reorder-only views (Amendment 1): view_reordered_filtered = 55** (5x4vr cell.json `data.consistency_views`).
   - Of the pool of 5,824: 1,409 rows have a view, 4,360 were not sampled and 55 were reorder-filtered (1,409 + 4,360 + 55 = 5,824). view_filtered (option set changed, fail closed) = 0; no_transform = 0.
   - Retained views by family: format 577, instruction 599, option 233.
   - cell.json does not split the 55 filtered views by family, and neither does cell.log. Under Amendment 1 only an option-paraphrase view can reorder, so all 55 should be option-family. I inferred this; it is not recorded.
3. **G1 used real perturbations: PASS.**
   - Every op has n > 0: reverse 213, shuffle 213, relabel 139.
   - On disk, 213/213, 213/213 and 139/139 perturbed rows differ from the base presentation, and the file order matches the in-memory order.
   - Both arms used the same perturbed.jsonl (sha 1fab827a).
   - The result is bitwise: max abs dp 0.0 against a declared bound of 0.
   - Non-vacuity check: the stock_shim control **FAILS** G1 on the same files, with 28 / 21 / 12 flips and max abs dp 0.84. These perturbations do move a model that is not invariant.
4. **Noul band diagnosis.**
   - Fraction of public yes/no answers in the 0.2-0.8 band: native 64.9% (48/74); native_tfit 64.9%; stock_shim 0%; native_shimcal 0%; H2O target 0%. On the slice at T 1 the band rate is 49.8%.
   - Per-type delta I, native minus stock_shim (cc component, paired):
     - choice: -3.86 [-14.12, +5.67]; top-1 -2.9 pp (10/14, p 0.54).
     - **noul: -80.83 [-101.88, -55.63]**; top-1 **+5.4 pp** (9/5, p 0.42).
     - score: -16.18 [-35.31, +5.71]; top-1 -33.3 pp (0/6, p 0.031).
   - The noul collapse is a confidence/readout effect, not a ranking effect. noul top-1 does not drop, and the same weights read through shimcal give noul cc +10.78 [-10.28, +31.20].
   - The band does **not** explain everything. The losses on score items, and the public top-1 losses on families outside the slice (routing -58.3 pp 0/7, ordinal -50 pp 0/6), are genuine ranking regressions. With shimcal, delta I is still -3.36 and top-1 is -2.6 pp.
5. **native_shimcal is a pre-registered SECONDARY, not the primary.** A-PREREG-JEV3-SYM names native vs the stock control as PRIMARY, and lists native_shimcal / native_tfit under SECONDARY.
   - Its delta of -1.07 [-22.47, +7.47] cannot replace the primary. Even if it could, its CI spans 0 and p = 0.77.
   - Any readout fix chosen after seeing these public-231 results (shimcal, a band-aware T, a yes/no floor) is **adaptive**. It can only be confirmed on the jev3 held-out slice or on a fresh split, under a new pre-registration written before that run.
6. **SINGLE SEED.** Seed 1 is not funded, so the result reads "single seed, replication pending". Under the prereg CLAIM rule (seed 0 CI > 0 AND both point estimates > 0), seed 0 alone already rules out a positive claim.

## Other things noticed (none changes a number)
- screen.json `jevkit.primary.rule` reads "A-PREREG-JEV1-4B: SUCCESS iff ...", a label carried over from the older prereg. The rule text is the same as the JEV3 PRIMARY. Cosmetic.
- The kit flags `target_ref` as an arm that is not in `screen.json jevkit.arms`. This is benign: it is the H2O target file, and its sha is the same e00d5ac7 as my independent copy.
- Step-0 floor: the in-job rule is a point estimate (pass = true). The kit's strict rule needs the lower CI bound > 0 and gives "not beaten". Both say the selected checkpoint is only +1.08 pp over step 0 on the slice, which is not significant. The adequacy family drops -8.0 pp on the slice. This does not matter for the verdict because the primary fails.
- Transfer: corr(slice delta, public delta correct) is +0.58, but corr(slice, public top-1) is -0.50. The slice gains did not carry over to the public items.

## Files
- `primary_native/`: cell_rescore.json, CELL_RESCORE.md, core.rescore.json, stdout.txt, lineage_check.json.
- `secondary_native_shimcal/`: the same files for the shimcal readout.
- `job-6zs79/`: screen.json, landing.json, checkpoint.json, cell.json, identity-child.json, identity.json, slice_first.json, tfit.json, slice_floor.json, the G1 and flip8 perturb reports, MANIFEST.jsonl, run.log, job_record.json.
- `job-5x4vr/`: landing.json, dev_cards/, card-step0000258.json, latest.json, ckpt sidecars, MANIFEST.jsonl, tar member list, streamed tar and member shas, run.log, screen.log, job_record.json.
- `inputs/`: price, mtimes, flip baseline. `tools/`: landing_to_mtimes.py, lineage_check.py.

Preds are not committed. Their shas are in the kit output: native ddc5dfb8, stock_shim a8ff9184, native_shimcal 614a4184, native_tfit 6a0c8677, target e00d5ac7.
