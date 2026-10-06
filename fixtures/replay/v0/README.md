# Replay fixture v0 (Track D → Track L handoff, m0)

37 synthetic requests recorded from the stock **Qwen3.5-0.8B** readout (`decide.backends.hf_readout`, JRT prompt `jrt-chatml-v1`, identity calibration). These are not Decision Index items.

| File | Content |
|---|---|
| `requests.jsonl` | `{case_id, category, description, decider:{enable_v0_1_deltas}, timing_dependent, request}` |
| `responses.jsonl` | `{case_id, response}`: the full live response (SystemOne `model/answers/usage` plus the `ssdm` envelope) |
| `recordings.jsonl` | Per-option logits and plans, keyed by `request_fingerprint` |
| `manifest.json` | Parent `revision` (with `revision_id`), calibration identity, generator command and git SHA, hardware, `files_sha256` |

What the 37 cases cover:

- Question types: noul ×4, choice ×12, score ×4, and multi-question records ×4.
- Choice edge cases: 1 option, 20 options, option ids that are prefixes of each other, null descriptions, unicode text, a prompt injection inside the state.
- Track L shapes: a semantic-operation choice, an evidence-ref choice, ask-vs-assert.
- A policy restriction (v0.1 deltas on) and a pinned `parent_revision`.
- Every status/reason code: `error/{unknown_question_type, empty_criteria, invalid_request}`, `unsupported/{too_many_options, media_unsupported, parent_revision_mismatch, policy_restriction_requires_v0_1_deltas}`, `exhausted/{input_tokens_exceeded, choices_exceeded, inference_calls_exceeded, deadline_exceeded}`.

Usage:

```python
from decide import Decider
from decide.backends import ReplayBackend
from decide.backends.replay import normalize_response

d = Decider(ReplayBackend("fixtures/replay/v0"), enable_v0_1_deltas=case["decider"]["enable_v0_1_deltas"],
            backend_max_input_tokens=manifest["revision"]["max_context"])
resp = d.decide(case["request"])
```

The contract is that `normalize_response(resp)` equals the recorded response exactly. It drops only `latency_ms`, the backend call timings and the `served_by` note. The one exception is `status-09-deadline` (`timing_dependent: true`). A request that was never recorded returns `unsupported/replay_miss`; there is no model fallback. Option or question order in the request does not matter, because lookup is order-insensitive.

Scope: the fixture proves boundary behaviour (wire format, statuses, limits, envelope). It says nothing about decision competence. A 0.8B readout is near chance on hard items; for example, it answers `osaka` for the capital of Japan.
