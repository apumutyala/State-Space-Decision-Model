# Parent checkpoint load exactness prerequisite

## Hypothesis

Before a decision checkpoint can serve as Track L's frozen parent or resume training, a checkpoint whose trainable tensor set/shape/dtype does not exactly match the current model must fail before changing any model, optimizer, scheduler, counter or RNG state.

## Deliverable

Preflight canonical `named_parameters()` entries with `requires_grad=True` against checkpoint `trainable` keys exactly, preserving PyTorch's duplicate-eliding names for shared parameters. Validate every tensor's type, shape, dtype, strided layout and finiteness before copying any weight. Preflight optimizer group topology/state tensors, scheduler structure, counters/history and CPU/Python/NumPy RNG payloads before restoring mutable state. Strictness flags for config/lineage must not disable these integrity checks. Add same-PR regression tests and LESSONS L-34. Keep existing valid-format resume behavior unchanged.

## Exit proof

- Existing exact save/load and interrupted/resumed training tests pass.
- Missing, extra/frozen, unknown, non-tensor, broadcastable/wrong shape, dtype and nonfinite trainable entries fail before any mutation, even with config/lineage strictness disabled.
- Malformed optimizer-group, scheduler or CPU/Python/NumPy RNG payloads fail preflight before model-weight copy.
- Tests snapshot trainable parameters, optimizer/scheduler state, counters/history and RNG; all are byte/value-identical after each rejected checkpoint.
- Focused tests and static checks pass; internal adversarial review finds no blocker. No real model/checkpoint/GPU/Index/held-out/spend.

## Decision unlocked

Whether Track L may rely on a reviewed Track D checkpoint loader for a future immutable parent handoff. This is resume correctness only; it does not qualify or identify a parent, establish model capability, or authorize training/spend. PR #141 supplies LESSONS L-33; this follow-up uses L-34 and should integrate after that append-only lesson PR.
