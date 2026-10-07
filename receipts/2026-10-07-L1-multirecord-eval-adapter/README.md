# L1 evaluation-adapter infrastructure receipt

2026-10-07. Starting/source fixture commit:
`bd31c9051732ddfa325b17f6ad986c2d08574f49` (PR #127).
Branch: `codex/l1-multirecord-eval-adapter`. Local Windows laptop,
Python 3.12.13, standard-library CPU checks; billed cost $0. No weights,
model execution, GPU, paid service, Index or sealed/held-out data. This is
preparation only. Tests use scripted evidence oracles, not learned providers;
no L1, performance, calibration, comparator or causal result is claimed.
Repeated-run median/spread are inapplicable to these deterministic controls.

Data pins (seed 1707, one world per split, mixed authored choice format):

| Split | JSONL SHA-256 | Manifest SHA-256 |
| --- | --- | --- |
| l1_development_train | `2d455c7ad7e2a1116d026bb5801dec0c821de24602375ccecf66b2569011f007` | `c5d94f90e4c41daab59fedb6745ecac758bce48b2724acfa7ca13e1dc47e9e2d` |
| l1_development_validation | `613d5b293cebbb883c46df6e8d0f361b61d81e65277d8ba8a3eeb8baacca141d` | `6a6d3f8012c8b6089c6c8d40a2c6eb6ded60320ba7551d332fdd668bdd9d1cc1` |

Commands run from the adapter worktree:

```text
python -m unittest language_growth.tests.test_multirecord_eval language_growth.tests.test_multirecord_data language_growth.tests.test_free_running_dev language_growth.tests.test_training_records language_growth.tests.test_kernel language_growth.tests.test_kernel_adversarial language_growth.tests.test_decision_view -v
python -X utf8 -m pytest -q tests/test_jobs_lint.py tests/test_no_kill_by_name.py
git diff --check
```

Passed: 51 focused tests, including six adapter tests; 70 repository job/process
lint tests. The evidence-oracle control runs each 40-task cohort through the
actual planner/scorer, with 128 provider calls per cohort and all tasks accepted
and correctly scored. This only proves the plumbing works for the scripted
control. The runner builds production requests itself; authored augmented slot
IDs are discarded, so these are not separate shuffled-provider measurements.
Raw summaries and all per-task outputs are preserved in
`infrastructure-controls.jsonl`. To reproduce them without model execution:

```python
import json
from pathlib import Path
from language_growth.multirecord_eval import load_multirecord_evaluation
from language_growth.tests.test_multirecord_eval import EvidenceOracle, run, SPLITS
lines = []
for split in SPLITS:
    provider = EvidenceOracle()
    report = run(load_multirecord_evaluation(split), provider)
    summary = {k: v for k, v in report.items() if k != "observations"}
    summary.update(scope="scripted infrastructure control only; no learned L1 result",
                   provider_requests=len(provider.requests))
    lines.append(json.dumps({"split": split, "summary": summary}, sort_keys=True))
    lines.extend(json.dumps({"split": split, "observation": row}, sort_keys=True)
                 for row in report["observations"])
Path("receipts/2026-10-07-L1-multirecord-eval-adapter/infrastructure-controls.jsonl").write_bytes(
    ("\n".join(lines) + "\n").encode("ascii"))
```

Controls retain wrong-but-valid answers, false relations and errors as failures;
reversed comparisons receive equivalent-meaning credit including equality.

Provider-request equality under poisoned evaluator labels/programs is checked
with fixed test request IDs. Tests also prove training export rejection,
independent row/manifest byte pins, identity/evidence guards and fresh copies.
No targets are exported. The existing scorer's `training_only` key is used
solely for compatibility with its evaluator metadata access.

Plan: `plans/codex/L1_MULTIRECORD_EVALUATION_ADAPTER.md`. Learned L0, future
causal/adversarial/comparator/retention gates and v0.2 exact calculations remain
outstanding. This adapter does not satisfy or replace those gates. Internal
adversarial review and Claude cross-review status are recorded on the PR.
