# Optional grounded candidate intervention ($0 CPU)

The two earlier development diagnostics selected opaque record-1 price operands
for both sides and `greater_than`, leaving only 3/8 expected slots correct.
The bounded hypothesis here is that placing the entity, human-readable field,
typed value, unit and time beside each candidate reduces the model's grounding
burden. The relation question also repeats the model-selected operands and
defines LEFT-relative-to-RIGHT relation orientation.

`plan_checked_answer(..., candidate_presentation="grounded")` enables this
representation. `reference` remains the default. Both representations offer
the same candidate IDs and domains. Candidate truth is never used to filter
the domain; the kernel still rejects a model-selected false relation. Added
information comes only from the validated audience snapshot. No gold slots,
reference programs or excluded catalog records enter requests.

The language suite passed 114 tests. The focused additions cover correct
grounded comparison construction, unchanged domains, explicit selected
operand values, false-relation rejection, excluded private records and invalid
presentation refusal. This is boundary evidence with scripted providers, not
a model performance result.

The paired Qwen CPU runner refused before importing Torch or loading weights:
7,522,709,504 bytes available versus the existing 8 GiB start floor. The raw
refusal is `preflight.txt`; no model predictions were produced. No GPU, paid
resource, held-out slice or Index was accessed. The existing 3 GiB running
floor and 300-second ceiling remain in the prepared runner.

The pending diagnostic compares reference/grounded free-running planning on
development seed 271828, indices 2 and 5. Those failures informed the change,
so any future result is exploratory and cannot establish generalization or an
L0 gate. The runner records all requests/responses, data and source hashes,
the checkpoint revision, configuration, seed, call counts and outcomes; labels
are consulted for scoring only after all provider calls. Mode order is
counterbalanced between tasks, with one deterministic repeat. A larger fresh
development check would be needed after any apparent gain.
