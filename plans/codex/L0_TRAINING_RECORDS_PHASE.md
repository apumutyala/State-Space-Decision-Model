# L0 training-record integration

Phase L0d3, 2026-10-06; follows the typed planner and development generator. CPU implementation only.

- Hypothesis: the parent's existing decision trainer can consume the language generator's typed slot supervision without changing model architecture or passing gold answers as current-slot inputs.
- Deliverable: a small exporter under language_growth/ to Track D's existing `{id, request, labels}` record format, keeping training and iterative development-validation domains explicit. It copies each existing request rather than reconstructing its state. Previous gold slot choices remain explicitly teacher-forced training context; this is not free-running evaluation.
- Exit proof: synthetic CPU tests reject split misuse, malformed/misaligned targets and duplicate identities; changing current labels does not change serialized model input. A CPU integration check passes exported records to Track D's actual `items_from_records` with a byte encoder and independently checks option-index labels. Exact committed dependencies and generated data hashes accompany the receipt.
- Decision unlocked: use the shared trainer for bounded supervised language adaptation after the frozen-parent probe, hardware qualification and separately acknowledged spend gate. Export success does not establish learning, retention, task capability or causal interpretation.

No Index or held-out data, weights, GPU, paid execution or new training framework is needed. Full L0–L4 goal gates remain unchanged.

## Bounded CPU optimizer-to-planner check

Before a paid adaptation, test the same exported slots through Track D's existing Trainer and DecisionModelBackend using its tiny randomly initialized recurrent test backbone. One optimizer step, one synthetic development task and one separate synthetic validation task, two CPU threads, no downloaded checkpoint and no CUDA. Exit proof: finite loss, changed trainable parameters, and real head decisions reaching the semantic planner without a transport failure. This unlocks an executable adaptation path, not a learned L0 gate or a claim about a qualified Track D checkpoint. Keep all decisions and failures in the test output; do not tune this tiny model for a score.
