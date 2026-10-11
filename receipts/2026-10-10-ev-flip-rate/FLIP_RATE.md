# FLIP_RATE: the MD1 robustness axis, measured on our candidates (B-PREREG-FLIP)

Status: LANDED 2026-10-11 (run 2). Rule: state `PREREG.md` B-PREREG-FLIP + amendment 1. Receipts: branch
`claude/ev-flip-rate`, `receipts/2026-10-10-ev-flip-rate/` (run2/ = governing; run1/ = partly void, kept).

## Bottom line
- **H2O-Lightning-4B v1.1** has the lowest 8-op average: **2.44 % [1.26, 3.75]**. MD1 reports 1.3 % on its own items
  and operators, so the two numbers are not the same estimand.
- **Stock Qwen3.5-4B + H2O shim** (the baseline the sym-invariant head must beat) averages **7.55 % [5.71, 9.67]**.
  Reorder flips are 12.2 % on reverse and 9.4 % on shuffle; relabel is 8.6 %. The reorder family is its largest
  weakness. H2O sits −5.11 pp [−7.27, −3.04] below it.
- **RWKV7-13.3B stock and D19** get exactly 0 reverse and 0 shuffle flips, **by construction**. The Decider readout
  presents options sorted by id (`decide/schema.py:98`, canonical options), and the base and reordered probabilities are
  byte-identical (213 / 213). These readouts are not invariant to relabel, though: stock flips 22.3 % on relabel, the
  worst cell in the table. D19 cuts relabel flips to **8.6 %** (−13.67 pp [−20.59, −7.48] vs stock).

## Results (run 2: jobs z8jcd + andnb; per-operator flip %, bootstrap 95 % CI on the average, B = 1000, seed 16)

| System | state | instr | opt-para | reverse | shuffle | relabel | whitespace | markup | avg (8) [95 % CI] | reorder flips |
|---|---|---|---|---|---|---|---|---|---|---|
| MD1 (reported, own items / operators) | ? | ? | 0 | 0 | 0 | ? | ? | ? | 1.3 % | 0 |
| H2O-Lightning-4B v1.1 (shim) | 1.0 | 0.9 | 1.3 | 2.8 | 2.3 | 4.3 | 3.0 | 3.9 | **2.44 [1.26, 3.75]** | 17 / 565 |
| Qwen3.5-4B + H2O shim | 2.9 | 3.5 | 6.9 | 12.2 | 9.4 | 8.6 | 8.2 | 8.7 | **7.55 [5.71, 9.67]** | 58 / 565 |
| RWKV7-G1k-13.3B stock (v3w) | 2.9 | 2.6 | 3.5 | 0.0* | 0.0* | 22.3 | 7.8 | 3.9* | **5.37 [3.95, 6.99]** | 31 / 565 |
| D19 (reverse + shuffle + relabel only) | — | — | — | 0.0* | 0.0* | 8.6 | — | — | 3-op 2.88 [1.36, 4.44] | 12 / 565 |

\* Invariant by construction in the Decider path: options are canonicalized (sorted by id) and the state is JSON
canonicalized with sorted keys, so the reordered input equals the base input. markup_noise therefore reaches this path
only for string states. n per operator: state 104, instr / opt-para / whitespace / markup 231, reverse / shuffle 213,
relabel 139. Invalid pairs: 0 for every system.

Paired deltas vs Qwen3.5-4B + shim (average over 8): H2O −5.11 pp [−7.27, −3.04]; RWKV7 stock −2.18 pp [−4.31, +0.11].
On the 3 reorder-family operators vs RWKV7 stock: D19 −4.56 pp [−6.86, −2.49]; Qwen-shim +2.64 [−0.21, +5.98];
H2O −4.27 [−7.05, −1.60].

## What this means for the sym-invariant head
- A head that is invariant to reorder AND relabel would put 0 in all three reorder-family columns. On Qwen-shim that is
  worth about 3.8 pp of the 8-op average (12.2 + 9.4 + 8.6 over 8). That alone would take it from 7.55 to about 3.8.
- Sorting options (the Decider's canonicalization) already gives reorder invariance for free. The relabel column is the
  real test: sorting by label makes a relabel into a reorder plus new names, and stock RWKV flips 22 % of those.
- Getting below H2O (2.44) also needs paraphrase and noise robustness, which the head does not target: Qwen-shim scores
  6.9 / 8.2 / 8.7 on opt-para / whitespace / markup against H2O's 1.3 / 3.0 / 3.9.

## Operator coverage vs MD1's 8 types (jevkit/perturb.py, frozen)
| MD1 type | jevkit operator | Coverage |
|---|---|---|
| State paraphrase | `state_paraphrase` (contraction / connective table) | yes, but shallow: changes only 104 / 231 states |
| Instruction paraphrase | `instruction_paraphrase` | yes |
| Option-description paraphrase | `option_paraphrase` | yes |
| Option reverse | `option_reverse` | yes (score items n/a) |
| Option shuffle | `option_shuffle` | yes |
| Option reorder (third reorder) | none distinct | **missing** |
| Key change | `key_relabel` (choice only) | yes |
| Formatting noise | `whitespace_noise` + `markup_noise` | yes, split into two operators |

## Defect found and fixed (amendment 1)
`jevkit perturb make` wrote rows with `sort_keys=True` (jevkit/cli.py:133). That made reverse and shuffle no-ops on
disk, undid markup's dict-state reversal, and turned relabel into relabel + reorder. Run 1 (file 655d028a) reported
0 reorder flips for every model as an artifact. The fix: an order-preserving file be4abaa3 (`gen_pert_nosort.py`), and
cli.py on this branch now writes without sort_keys and reproduces be4abaa3 exactly (11 cli / perturb tests pass). The
cli fix needs the lead's APPROVE in a PR before it reaches agent-main. The run-1 paraphrase and whitespace numbers are
unchanged in run 2 (identical inputs).

## Comparability caveats (MD1)
1. The operators differ: MD1's are unpublished, ours are rule-based. Our state paraphrase is shallow, formatting noise
   counts twice, and there is no third reorder. The 8-op averages are not the same estimand as MD1's 1.3 %.
2. The items differ: MD1's set is private. Ours are the 231 public JevBench items, long and multi-hop ones included.
3. MD1 was not run; we compare against its reported numbers only.
4. Zeros by construction (canonical sorting) are a property of the readout, not of robustness learned by the model.
   MD1's 0 reorder flips may be the same kind of property.
5. One perturbation seed: the CI covers item sampling, not operator randomness.

## Receipts
- run 2: job-z8jcd (claude/ev-flip-hs @a7c431b0, art-cnp6a sha 957621540bc5…, $0.55) and job-andnb
  (claude/ev-flip-d19 @552f2143, art-bx392 sha 84ac5e1725d5…, $2.27). Preds in `run2/`, scores in
  `run2/flip_ci_8op.json` and `run2/flip_ci_3op_d19.json`; flip_ci points match `jevkit perturb score` exactly (the
  script exits 3 otherwise).
- run 1 (partly void): job-uqbv7 ($0.59) and job-d3y4c ($2.07). Their base arms reproduce job-zbtbj and job-9j3uu
  (D19 I 52.6, Composite A 62.4).
