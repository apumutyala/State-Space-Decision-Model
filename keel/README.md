# Keel — an auditable judgment layer for AI workflows

Keel separates three things that AI workflows usually blur:

- **Evidence:** untrusted content (records, tool outputs, retrieved text, user messages) that a decision may and
  should depend on.
- **Judgment:** a fast decision model that turns typed questions over that evidence into option probabilities, plus
  a gate that decides whether to act on them or defer.
- **Authority:** what the workflow is permitted to do. Only the host assigns it. Content can never grant it to
  itself.

The proposition, technical design, safety analysis and results are in
[`direction/PROPOSITION.md`](../direction/PROPOSITION.md). The integrator contract is in [`PROTOCOL.md`](../PROTOCOL.md).

## Layout

| Package | What it does | Guarantee type |
|---|---|---|
| `keel/protocol` | Trust-labelled `DecisionRequest`s, trust-aware rendering with escaping and canonical option order, hash-chained receipts and replay | structural (tested) |
| `keel/host` | Enforcement kernel: host-held authority, defer ⇒ no side effect, review-queue saturation ⇒ hold, write-ahead receipts; bounded model checker | structural (bounded exhaustive check + property tests) |
| `keel/gate` | Act/defer gate on uncertainty **and** manipulation signals; complete-gate certification with cluster-valid Learn-then-Test | statistical (conditional on exchangeable clusters) |
| `keel/card` | Decision Alignment Card: paired units, static/transfer/adaptive attacks, evidence-retention and policy-compliance controls, cluster-bootstrap statistics | empirical evaluation |
| `keel/distill` | Discernment distillation: trust-policy thinking teacher → one-pass student (LoRA), guarded training data | empirical |
| `keel/workflows` | AgentDojo tool-call gate: executed unauthorized actions vs utility in agent workflows | empirical |
| `keel/runtime` | Modal images, pinned models, vLLM judge (one-shot `score`, warm `JudgeServer`) | — |

## Quick start

```bash
# CPU tests (protocol, host, gate, card, distill, workflow)
python -X utf8 -m pytest -q keel/tests

# bounded verification of the host kernel
python -X utf8 -m keel.host.model_check

# GPU smoke check of the judge runtime on Modal (public items, eval-only)
SSDM_DATA=/home/alp/ssdm-data python -X utf8 -m keel.runtime.smoke_judge --n 4
```

Each sub-package README states exactly what is implemented, what is verified or measured, and what is not claimed.
The interface rules between sub-packages are in [`CONTRACT.md`](CONTRACT.md).

## Verification record

| Date | Commit | Command | Result |
|---|---|---|---|
| 2026-10-15 | `d64e9698` | `python -X utf8 -m pytest -q keel/tests` | 414 passed |
| 2026-10-15 | `d64e9698` | `python -X utf8 -m pytest -q keel/tests tests/guarantees jevkit/tests tests/test_nupi.py tests/test_consistency.py tests/test_order_views.py tests/test_order_consistency_loss.py tests/test_jev3_sym.py` | 717 passed, 4 skipped |
| 2026-10-15 | `d64e9698` | `python -X utf8 -m keel.host.model_check` | 48 configurations, 632 states, 16,920 transitions, depth 4, 16,920 concrete refinement transitions |

Environment: Python 3.12, torch 2.14.1+cpu, transformers 5.14.1, CPU only. Results of paid GPU runs are recorded in
each experiment's README under `direction/experiments/`.
