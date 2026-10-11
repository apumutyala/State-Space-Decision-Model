# ev-cell-kit: one rescore + guarantee report per landed SSDM cell (evidence lane, 2026-10-11)

`scripts/ev_cell_rescore.py` combines the evidence lane's pieces into ONE report (`cell_rescore.json` + `CELL_RESCORE.md`):
(1) PRIMARY paired delta Composite A vs the in-job control (B 1000, seed 16) with 95% CI, one-sided p, Holm levels, and the
screen.json diff; (2) EV-PREREG-SECONDARY-AXES; (3) the target gap vs existing preds (paired on ids, S/K at each system's own
value, cross-job caveat); (4) integrity; (5) selection, step-0 floor, order of events; (6) G1 from files (fail closed,
file-order check, bitwise max|dp|); (7) the 8-op flip battery vs a baseline; (8) per-family transfer.

Sources folded in: `scripts/ev_rescore_jevkit.py` (claude/ev-j1-rescore @a8ca3e7e = claude/ev-x7c42-rescore @bf706854; two
edits: find the arm-preds dir when there is no `identity.json`, e.g. a lead landing dir), the extras of
`scripts/ev_rescore_x7c42.py` (@bf706854), `flip_ci.py` (claude/ev-flip-rate @4a65dd18), and the G1 rule of
`jobs/sym_eval/perturb_eval.py` (stack/qwen35/sym-invariant @55bfdd39: n > 0, 0 invalid, 0 flips, max|dp| bound, presented
order changed, file order = in-memory order), here recomputed from the files on disk.

## Usage (the jev3 cell, as soon as it lands)
```
python -I -X utf8 scripts/ev_cell_rescore.py --art <extracted artifact> --control stock_shim --primary <arm> --out <dir> \
  --price <price.json> --expect-steps N --slice-ids-sha <sha> \
  --target "h2o=<job-4g6m8 or job-zbtbj H2O preds>::<price.json>" [--holm-arm <arm>] [--holm-ext NAME=P] \
  [--g1 NAME=<dir>] [--g1-max-dp 0] [--flip8 <dir> | <base.jsonl>,<pert.jsonl>] \
  [--flip-baseline "qwen35_4b_shim=<preds.jsonl>,<preds_pert.jsonl>"] \
  [--flip-baseline-receipt "<run2/flip_ci_8op.json>::qwen35_4b_shim"] [--mtimes landing.json] [--identity-child <json>]
```
Default G1 dirs: `out/screen/jevkit/g1/` (or one sub-dir per arm). Exit 0 = no flag, 3 = flags, 2 = inputs missing.

## Tests (this directory)
- `repro_x7c42.out.txt`: the kit on the job-x7c42 landing files (stack/qwen35/jev2-h2o receipts/2026-10-11-jev2-h2o-x7c42,
  preds shas f63858d1 / 296cf7d3 / e65160ae / 3555841e, rows d1bf3590) with the job-4g6m8 H2O preds (e00d5ac7) as the target
  and `--mtimes landing.json`. **795/795 values exactly equal** to the committed independent rescore (rescore.json +
  rescore-extra.json @bf706854): delta Comp A -0.40914 [-1.75376, +1.39471], 97.5% [-2.05957, +1.57901], p 0.565, Holm not
  claimable (with q668q p 0.81); 95 screen.json fields, max |d| 0; floor +4.44 pp [+3.47, +5.31], McNemar 3.1e-19; slice ids
  sha 142a593a; per-family transfer table; selection 14:51:16Z < first public pred 14:55:38Z.
- `repro_flip.out.txt`: the (7) battery on job-z8jcd (H2O as the cell, stock Qwen3.5-4B shim as the paired baseline):
  **77/77 values exactly equal** to claude/ev-flip-rate run2 `flip_ci_8op.json` (baseline 7.55% [5.71, 9.67]; H2O 2.44%,
  paired -5.11 pp [-7.27, -3.04]).
- `pytest.out.txt`: tests/test_ev_cell_rescore.py, 9 passed: synthetic invariant arm = G1 PASS bitwise; position-biased arm =
  FAIL (flips); sort_keys perturbed.jsonl = FAIL (vacuous) even for the invariant arm; 1e-6 noise FAILS at 0 and PASSES at
  1e-3 flagged NON-BITWISE; a missing operator fails closed; in-job report diff; Holm; McNemar; one end-to-end CLI run.
- Not reproducible from the landing files alone (reported as warnings, not numbers): the scored checkpoint (no
  identity-child.json in the landing; pass `--identity-child`), G1 (x7c42 landed no g1 preds).
