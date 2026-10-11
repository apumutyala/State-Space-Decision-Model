# Consistency-view training (stack/qwen35/consistency)

## What it targets

jevkit's flip-rate battery (8 perturbation operators on the 231 public items; claude/ev-flip-rate @4a65dd18) measured
H2O-Lightning-4B at 2.44% and stock Qwen3.5-4B + shim at 7.55%. The sym-fork layout (stack/qwen35/sym-invariant) removes
the three REORDER operators by construction (option reverse 12.2%, shuffle 9.4%, key relabel 8.6%). What is left can't be
fixed by construction, because the prompt text itself changes:

| flip type (stock + shim) | rate | covered by view family |
|---|---|---|
| option description paraphrase | 6.9% | `option` |
| whitespace noise | 8.2% | `format` |
| markup noise | 8.7% | `format` |
| instruction paraphrase | 3.5% | `instruction` |
| state paraphrase | 2.9% | none (already low; not targeted) |

This branch trains the model to give the same answer when only the surface changes. For a sampled share of training rows
it builds a meaning-preserving VIEW of the row (`train/consistency_views.py`) and adds a consistency term:

    L = L_task(orig) + lambda_c * symKL(p(orig) || p(view))

Here `symKL(p||q) = KL(p||q) + KL(q||p) = sum_k (p_k - q_k)(log p_k - log q_k)` (`train.losses.symmetric_kl`), computed
over the option distribution of the row's question. A view keeps the option ids, their order (and so the shim labels),
the question type and the gold label, so the two distributions share one option axis. The task loss is computed on the
original row only. Per optimiser step the objective is `mean_rows[L_task + lambda_c * 1[row has a view] * symKL]`, and
it does not change with grouping (`row_batch`).

## The view generator (CPU, deterministic, seeded, train rows only)

The generator is stdlib only. It makes no LLM calls and has no model or network access. Every choice is
`sha256(version, seed, item id, salt)`. One family per view:

- `format`: string leaves of the STATE only. Ops: bullet style (`- ` <-> `* `), numbered markers (`1.` <-> `1)`),
  markdown -> plain (strip `**bold**`, `# Header` -> `Header:`), plain -> markdown (`Key:` -> `**Key:**`), paragraph
  breaks (`\n` <-> `\n\n`), whitespace normalisation (collapse interior space runs, strip line ends and edges, tabs ->
  spaces). A flat dict state can also be rendered as `key: value` lines, keeping its key order.
- `instruction`: a frame from an 8-template bank (`INSTRUCTION_FRAMES`) and/or verb synonyms
  (`INSTRUCTION_SYNONYMS`: choose/select/pick, determine/decide, classify/categorize, ...).
- `option`: a light rule-based paraphrase of every option description (choice values, score levels, noul true/false
  text). It uses `OPTION_SYNONYMS` (issue/problem, buy/purchase, required/needed, ...), `&` <-> `and`, and first-letter
  case.

`view_frac` (default 0.25) picks the rows by a sha256 threshold, independent of data order. A row with no applicable
transform, or whose view fails the cell's K/T filter or the guard, stays single-view. Views attach only to the training
list (`train.run_cell.attach_consistency_views`). The Trainer refuses eval items that carry a view, and the guard
re-checks every view row (`items_from_records(..., guard=guard)`).

## Why it isn't fitted to the test operators

The test operators are in `jevkit/perturb.py`, which is frozen evaluation. The generator never imports jevkit and
reuses none of its rules. `tests/test_consistency.py::test_independent_of_the_jevkit_test_operators` checks this:

- The jevkit rewrite table is contractions plus e.g./i.e./approximately/in order to. Our tables share no word with it,
  and we never contract or expand.
- jevkit's three frames (`Task: ...`, `Decide the following. ...`, `... Answer from the options given.`) aren't in our
  bank.
- jevkit's option operator toggles the final period. We never touch final punctuation.
- jevkit's whitespace operator doubles space runs and adds a leading and trailing newline. We only collapse runs and
  strip edges.
- jevkit's markup operator quotes the state with `> ` and reverses dict-key order. We never quote, and we keep the key
  order.

So we train on the same perturbation FAMILIES as the test (the same families MD1 discloses) with disjoint operators. A
flip-rate gain on the battery is then evidence of transfer, not memorisation. The state-paraphrase operator isn't
targeted at all, so it serves as a within-battery control.

## Config (`consistency:` cell block; OFF by default)

    "consistency": {"view_frac": 0.25, "lambda": 0.5, "seed": 0,
                    "families": ["format", "instruction", "option"], "view_grad": true,
                    "controller": null}

- If the block is absent or `{}`, the feature is OFF: `TrainConfig.consistency` stays `{}`, and the key is left out of
  `to_json` and the config hash. The step never reads it, history rows and checkpoint keys are unchanged, and
  `cont_plan` "single" behaves as before. `test_off_path_bit_identical_to_the_parent_tree` runs the same toy training
  (micro-batch and row_batch paths) in the parent tree (a5b4cb33) and in this tree, with and without an explicit
  `consistency: {}`. It compares losses, trainable weights, config hash, to_json, lineage and checkpoint keys, bit for
  bit.
- `controller`: `{epsilon, kappa_p, kappa_i, nu, lambda_init, lambda_max}`. lambda_c is the nuPI multiplier of the
  constraint "mean symKL over the step's paired rows <= epsilon". It uses `train/pi_control.py`, taken verbatim from
  stack/qwen35/nupi @54d34df9. A full merge of that branch is NOT clean: it forks from 5e159437 and rewrites
  trainer/losses/run_cell. So only the standalone controller file is taken. The multiplier updates once per optimiser
  step, takes effect from the next step, and is saved in the checkpoint. `integral_min = -lambda_init / kappa_i`, so
  `lambda_init` is a warm start that can decay to 0.
- `view_grad: false` runs the view forward under no_grad. That's cheaper, but it only pulls p(orig) toward p(view).
- consistency and order_consistency are separate arms. Enabling both is refused.

## Cost

Each view row costs one extra forward (plus its backward when `view_grad`) on a row of about the same length. Views
change the token count by a few percent. Per-row work is roughly the same for base and view rows (forward + act-ckpt
recompute + backward ~ 4 forward units), and the optimiser step is small. The step-time factor at `view_frac` v is about
`1 + v * (L_view / L_base)`, which gives **~1.25x at view_frac 0.25** (`view_grad: false`: `1 + 0.25 * 1/4` ~ 1.06x).
Peak memory with `row_batch` 1 grows too, because both graphs are alive at the backward: up to about 2x activations on a
paired row. The cell's ladder and cont_chunk probe already step rows WITH their views (`item_context_len` counts
`cview`), and the step-10 rate check re-measures the real s/step in-job.

For the JEV2-4B budget (1,281 steps at 16.6 s/step, clock 462 min), 1.25x means about 20.8 s/step. Either keep the clock
and accept ~1,025 steps (above min_steps 600; the projection continues), or raise max_minutes by about 115 (worst-case
hold +$15, still <= $80 only if the other phases shrink). The proposed cell keeps the clock.

## Proposed ablation (one hypothesis per job, paired on the same draw and seed)

| arm | consistency | purpose |
|---|---|---|
| A0 | absent (JEV2-4B as packaged) | baseline: flip rate per operator, Composite |
| A1 | `{view_frac 0.25, lambda 0.5}` | fixed lambda_c |
| A2 | `{view_frac 0.25, controller {epsilon 0.01, kappa_p 1.0, kappa_i 0.05, nu 0.9, lambda_init 0.25, lambda_max 4.0}}` | constrained (nuPI) |

Readouts:

- jevkit flip rate per operator on public-231 (paired vs A0). The targets are option paraphrase, whitespace, markup and
  instruction. State paraphrase is the control, and the reorder ops must stay at 0 (sym-fork).
- Composite and Capability with CIs (the consistency term must not cost accuracy).
- In-job `cons_kl` and `cons_lambda` traces from the history.

Success: overall flip rate < 2.44% (H2O) with Capability not lower than A0 beyond its CI. Kill: no per-operator gain on
the four targets, or a Capability loss beyond the CI.
