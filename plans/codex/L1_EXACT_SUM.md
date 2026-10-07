# L1 exact-sum groundwork

6 October 2026. CPU language expressivity work, based on `agent-main`
`6e39d9b`. This is not a learned L1 qualification or parent-selection change.

**Hypothesis.** Host-authorized, already checked numeric facts of the same quantity kind,
unit and effective scope can be added as an exact typed computation while
retaining both source dependencies. Explicit object projections and a
registry-gated operation prevent a model from injecting a scalar or prose
answer.

**Deliverable.** Draft semantic-program profile `ssdm.semantic_program.v0.2`
adds `ExactSum` with `rule_ref` and 2..16 prior Fact object projections under
host `sum_rules` (initial four-record work uses at most four). It returns a typed
exact integer/decimal sum and can be asserted or explained. The kernel checks
types, units, quantity kinds, scopes and current authority; the plain-text
renderer states only that checked sum. Existing v0.1 programs remain accepted,
and a v0.1 program cannot use `ExactSum`. The registry must explicitly list
`ExactSum` in `allowed_ops`, authorize the exact additive signature and require
explicit quantity-kind metadata. Duplicate semantic premises reject. Host constants and free strings are outside this
first arithmetic operation.

**Exit proof.** CPU tests cover integer and decimal sums under a deliberately
low ambient Decimal precision, including 4,300-digit inputs and 4,302-digit
result carries, up to 5,000 fractional places, negative values
and cancellation; both source dependencies; deterministic text;
registry/schema opt-in, nonnumeric operands, unit/kind/scope mismatches,
unbound constants/free prose, stale or forbidden evidence, and unchanged v0.1
fixture behavior. The canonical interface profile change is included on this
branch for joint Track D/L review before any merge.

**Decision unlocked.** If the checker/renderer profile is approved, a later
separate parent-choice and development-evaluation change can measure whether
the model selects useful sums. A passing deterministic kernel test says
nothing about task relevance, learning, real-world premise truth or L1 gates.

2026-10-07 local issue #175 hardening: bounded lexemes before Decimal parsing,
canonical duplicate rejection, fixed-point scale preservation/positive zero,
262,144-byte complete serialized/rendered limits and release binding to all
registry rules. This is an unpushed draft based on #38+#129, not integration,
joint approval, a provenance transport implementation or a new planner arm.
