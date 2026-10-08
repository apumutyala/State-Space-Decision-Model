This module adds historical electricity lifecycle knowledge through typed
semantic programs. It can answer a reported-range question for a listed
technology or check a proposed unique lowest median across its entire table.
It composes the merged `ReadFact`, `Compare` and `Assert` operations; no generic
kernel, arithmetic, planner or neural model behavior changes.

The shipped host-owned extraction contains five rows from [IPCC AR5 WGIII
(2014), Annex III, Table A.III.2](https://archive.ipcc.ch/pdf/assessment-report/ar5/wg3/ipcc_wg3_ar5_annex-iii.pdf):
Coal - PC, Gas - Combined Cycle, Nuclear, Solar PV - utility and Wind onshore.
Values are the **lifecycle including albedo effect** column, in gCO2eq/kWh.
Each numeric premise retains this assessment/table temporal scope, source
reference, unit and statistic-specific quantity kind. Source metadata carries
the boundary and uncertainty interpretation. Technology punctuation is
normalized to ASCII; numbers preserve the reported decimal literals.

The extraction was manually transcribed from the official published table's
indexed text on 2026-10-08. It is not an automatically authenticated scientific
database. `energy_environment` is a trusted-host import and premise-admission
decision; the model may supply only the separate four-key operation proposal.
Reported minima and maxima summarize the assessed literature; they are not
confidence intervals. Answers describe this 2014 assessment and this explicit
five-technology subset, not current products, individual plants, causal
superiority, latest evidence or an energy procurement recommendation. The
reported ranges overlap even where medians differ.

Run the single development example without a provider, model or paid service:

```text
python -m language_growth.domains.energy
```

The example proposes Wind onshore as the lowest median. Checking expands it
to its three reported-statistic facts and four strict, same-unit/scope/statistic
comparisons against **every** other module row. A missing competitor, incorrect
winner, tie, changed evidence, source metadata mismatch, disallowed operation,
private dependency or revoked policy prevents release. The kernel revalidates
live authority before deterministic realization. Failure has the single public
shape `{"status": "unavailable"}`.

Use the same entrypoint for one row:

```python
from language_growth.domains.energy import (
    MODULE_REF, PROGRAM_SCHEMA, energy_environment,
    load_energy_module, public_energy_answer,
)

module = load_energy_module()
host, catalog = energy_environment(module)
proposal = {"schema": PROGRAM_SCHEMA, "module_ref": MODULE_REF,
            "op": "ReadTechnology", "record_ref": "nuclear-ar5"}
answer = public_energy_answer(
    "In IPCC AR5 (2014), what lifecycle emissions are reported for Nuclear?",
    proposal, module, host, catalog,
)
```

Only that exact row-question grammar and the exact `LOWEST_TASK` grammar are
supported. `check_energy_program` exposes the existing kernel's accepted proof
for inspection; `public_energy_answer` separately checks question relevance.
The model chooses an operation and row reference, never a value, competitor
subset, comparison relation, source or output sentence. The example is authored
development material, not training or held-out evaluation. This is new grounded
expressibility, not learned L4 capability, qualification, parent retention,
alignment or an intervention result.
