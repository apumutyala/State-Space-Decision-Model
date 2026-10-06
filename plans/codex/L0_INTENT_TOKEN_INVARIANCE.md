# One fixed operation-token intervention

Stacked dependency: `origin/codex/l0-transparent-cpu-baseline` at
`64f473bb143b367d655d4929217f16f3413cbdaf` (PR #63). This separate branch and
worktree preserve that reviewed baseline unchanged. The fitting pool, generator,
task/program logic, feature slots outside operation, optimizer, seeds and scoring
remain fixed.

Hypothesis from the preserved operation contributions: numeric/date and common
function words, including `2026` and `for`, introduce spurious operation-head
correlations that outweigh the `compare` cue in changed wording. One predeclared
transform drops purely numeric tokens, twelve month-name tokens, and this fixed
function-word set from operation features only:

`a an the and or of for to in on at by with from as is are was were be been being do does did has have had it its this that these those`

Intent words (`compare`, `show`, `checked`, `derivation`, `what`, `how`), entity
names and field tokens remain. Exact entity/field grounding and Decimal-sign
features are unchanged. The feature version identifies the intervention and the
receipt records the full frozen filter. A tight test checks retained/dropped
tokens and the actual operation-feature keys.

Run exactly once: fit the same 13,332 original `development_train` records,
200 epochs, learning rate 2, L2 1e-4, seeds 17/29/43; evaluate the same 300
`development_validation` tasks, seed 314159, with no repair. This validation
domain is now an iterative development intervention set, not a final held-out
or promotion set. Do not tune the filter or repeat variants after observing
outcomes. Compare the per-seed free-running counts with the frozen baseline's
200/300 overall and 0/100 comparison tasks.

The exit proof is a separate receipt with source/config/data hashes, full raw
predictions/requests/responses and feature contributions, task/slot/status
counts, seeds and median/range. If comparison remains 0/100 or overall regresses,
report the negative result and recommend folding this intervention. If it
improves, freeze the apparent winner before later untouched synthetic audit
data is generated for a separate gate. Never access Index held-out data.

CPU/NumPy only, no GPU, paid service or checkpoint downloads. No PR is opened
before the receipt and coordinator's lean internal review. No L0 gate, learned
raw arithmetic, shared-parent retention or causal mediation is claimed.
