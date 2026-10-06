# Development slot diagnostics CPU receipt

2026-10-06; branch `codex/l0-slot-diagnostics`. Synthetic scripted providers only; no model load, Torch, GPU, paid run, held-out data or Index access.

`language_growth/free_running_dev.py` adds `slot_diagnostics` to each observation and the report. Slots include the generated target ID, selected ID per attempt, runtime expected ID, status, missing-choice flag and a denominator of one. Aggregate slot totals use each task's final attempt; earlier selections cannot fill a missing final choice. Repairs keep all attempts. Shuffled training IDs resolve by generated option description after provider calls finish.

Targets are read only after all planner calls across the batch. Poisoned labels and generated gold prefixes leave provider states/questions and primary task/kernel scores unchanged. Existing output fields remain intact. Missing target metadata marks diagnostics unavailable without changing primary scores. An equivalent reversed comparison remains task-correct while its orientation slots differ.

Focused command: `python -m unittest language_growth.tests.test_free_running_dev language_growth.tests.test_paired_comparison -v`.

Full output is in `tests.txt`; 24 tests passed. Checks cover isolated wrong left/right/relation choices, correct read/explain/compare slots including shuffled IDs, pre-provider and partial provider errors, successful and interrupted repairs, primary score compatibility, and no label leakage.
