# L0 causal semantic-state intervention

## L0i-a — identity-bound experimental state edit

- **Hypothesis:** one pre-registered recurrent-state direction can causally bias a later checked Assert-versus-Explain operation choice while preserving the original parent, selected fact, and release checks.
- **Deliverable:** an experimental-only RWKV-7 capture/clone/edit/replay wrapper over the existing state-fork API; exact parent/tokenizer/backend/template/context/prefix/state-tensor identity; a fixed allowlist of recurrent tensor paths; registered semantic and control directions; no external provider handle or arbitrary tensor callback.
- **Exit proof:** CPU fake-cache tests for exact sham replay, prefix-plus-suffix parity, tensor-path/shape/finite checks, registration and lineage mismatch rejection, clone isolation, and unchanged parent bytes; independent exact-head review. L0i-a does not prove planner/kernel/renderer integration; L0i-b must score outputs only after the ordinary checked path.
- **Decision unlocked:** whether this narrow protocol is safe to run on the frozen native RWKV CPU parent after the existing RAM guard clears. This proof is interface/contract evidence only, not causal fidelity.

## L0i-b — frozen-parent assert/explain pilot

- **Hypothesis:** a discovery-fitted assert↔explain state direction has a larger, more persistent within-request effect on later operation choice than sham, tensorwise norm-matched random directions and a frozen output-only correction fit on the same discovery pairs.
- **Deliverable:** one preregistered direction and scale, fixed discovery/validation worlds, identical neutral and trusted-countermand suffixes, raw paired traces and checksummed parent state before/after. No layer or scale sweep.
- **Exit proof:** complete raw traces, no missing pairs, ordinary checked output path, unchanged direct parent decisions and weights, countermand/overwrite behavior, and paired source-world CIs against every control. Report every stopped/error unit.
- **Decision unlocked:** advance this specific causal mechanism, revise/fold it, or conclude that it only matches an output override.

## Scope limits

The wrapper currently captures caller-supplied exact prefix IDs and does not authenticate their correspondence to a `LayoutRow` or prove a schema → state boundary. A capture at the current layout's state boundary has already consumed the request schema; a real parent adapter must validate that boundary before any run, and claims remain within-request recurrence, not query-independent semantic memory. Changing the layout requires a separate registered experiment. Full donor-state replacement is only a positive control; persistence and countermand behavior are required to localize the recurrent effect. No Index, held-out or paid compute is part of L0i-a/b. The actual parent run requires a fresh ≥6 GiB start guard and ≥3 GiB runtime floor.
