# Replay fixture v0 (Track D → Track L handoff, m0)

**Parent identity:** this fixture is recorded from the stock **Qwen3.5-0.8B** readout (`decide.backends.hf_readout`, JRT prompt `jrt-chatml-v1`, CPU fp32, identity calibration).

This is **not** the Qwen3.5-9B readout promised in `coordination/CLAUDE_RESPONSE_TO_CODEX_HANDOFF.md` (item 2). The m0-b phase is $0 local work, and 9B does not run usefully on the shared 12 GB laptop GPU.

- The 9B-backed adapter and fixture come with the paid m0 run, as `fixtures/replay/v0-9b`. It will have a new `parent_revision`.
- Pin `parent_revision` from `manifest.json`. Do not assume it is the 9B parent.

The fixture holds 43 synthetic requests; none are Decision Index items. Requests use envelope `ssdm.decide.v0.1` (contract 0.1, PR #4 @ `787d64f`).

**Responses rebuilt without a model run (adapter 0.2.0).** `requests.jsonl` and `responses.jsonl` were rebuilt by `scripts/rebuild_fixture_from_recordings.py`; `recordings.jsonl` (the logits) is byte-identical to the CPU recording at `29b858c`. The rebuild runs every case through `ReplayBackend` and the current adapter. All 28 ok cases have the same answers, raw/served probabilities and request fingerprints as before. `manifest.responses_rebuilt` records this and lists `pending_live_recording` (`injection-03-added-tokens`).

| File | Content |
|---|---|
| `requests.jsonl` | `{case_id, category, description, decider:{enable_v0_1_deltas}, timing_dependent, request}` |
| `responses.jsonl` | `{case_id, response}`: the full live response (SystemOne `model/answers/usage`, plus `error` when status ≠ ok, plus the `ssdm` envelope) |
| `recordings.jsonl` | Per-option logits and plans, keyed by `request_fingerprint` |
| `manifest.json` | Parent `revision` (with `revision_id`), calibration identity, generator command and git SHA, `files_sha256`. The model ran on CPU (`revision.inference_config.device`); `hardware.gpu` only names the machine's GPU. |

## Coverage

- noul ×4, choice ×12, score ×4 and multi-question ×4.
- Choice edge cases:
  - 1 option and 20 options;
  - option ids that are prefixes of each other;
  - null descriptions;
  - unicode text.
- Track L shapes: a semantic-operation choice, an evidence-ref choice, and ask-vs-assert.
- Injection cases: three of them (see "Injection" below).
- A policy restriction (core in 0.1; full candidate set with zero on excluded, `restriction_id`, `excluded_served_mass`, stage `status`) and a pinned `parent_revision`.
- Every status/reason code:
  - `error/{unknown_question_type, empty_criteria, invalid_option_id}`
  - `error/invalid_request` ×3: a bad limit, duplicate restriction ids, and a `max_choices` alias that conflicts with `max_questions`
  - `unsupported/{too_many_options, media_unsupported, parent_revision_mismatch, provenance_not_enabled}`
  - `exhausted/{input_tokens_exceeded, questions_exceeded, options_exceeded, inference_calls_exceeded, deadline_exceeded}`
- Limit units follow contract 0.1:
  - `max_questions` caps questions per request; `max_choices` is accepted only as its legacy alias.
  - `max_options_per_question` caps supplied options per question.
  - Planner budgets (`max_proposal_calls`, `max_repair_calls`) are not D limits and are rejected.
- Fingerprint: `diagnostics.request_fingerprint_algorithm = ssdm.request_fingerprint.v0`, the contract's pinned algorithm. Provenance is a planned 0.2 extension; D's experimental support is behind a flag. With the flag off, provenance-bearing requests get `unsupported/provenance_not_enabled` (`status-15`).

## Usage

```python
from decide import Decider
from decide.backends import ReplayBackend
from decide.backends.replay import normalize_response

d = Decider(ReplayBackend("fixtures/replay/v0"), enable_v0_1_deltas=case["decider"]["enable_v0_1_deltas"],
            backend_max_input_tokens=manifest["revision"]["max_context"])
resp = d.decide(case["request"])
```

## What replay proves (and what it doesn't)

Replay is a deterministic function of the recorded logits. The contract is that `normalize_response(resp)` equals the recorded response. Normalisation drops only `latency_ms`, backend call timings and the `served_by` note.

Two checks back this up:

- `tests/test_replay_fixture.py` checks the contract for all 42 non-timing cases.
- It also recomputes every ok answer *without the adapter*: softmax of the recorded logits, then Cloudflare's own `systemone_answer` logic.

Limits of the contract:

- It is adapter/boundary evidence, not model evidence.
- `status-09-deadline` (`timing_dependent: true`) is excluded. The deadline is checked *between* inference calls, so the live case spent one full call before reporting `exhausted`. The rebuild reproduces that status with a replay backend that sleeps 5 ms per call.
- A request that was never recorded returns `unsupported/replay_miss`; there is no model fallback.

Order insensitivity is also an adapter property. The adapter canonicalises question and option order, so the model input is byte-identical under any shuffle. That says nothing about the model's own order robustness.

## Rules for Track L

- **Branch on `ssdm.status`** (or the top-level `error`), never on `answers`. Non-ok responses have `answers == {}`. A legacy SystemOne transport must map status ≠ ok to an HTTP error.
- **`abstention.mechanism = "not_implemented_v0"`**: `abstained: false` is not a measured non-abstention.
- **Answers are not separable across questions.** The JRT schema block lists every field before the state, so sibling questions change each other's answers. Compare `choice-02-invoice-3` (`overdue` 0.956) with the same question inside `multi-01-invoice` (`draft` 0.802). Do not compose programs assuming per-question independence.

## Injection

Request text (state, ids, instructions, descriptions) is encoded by the canonical `decide.untrusted.UntrustedEncoder`, the one encoder `safety_eval` must import. It works in two steps:

- Special tokens are split by `split_special_tokens=True`.
- Added tokens that the flag does not split are cut. On Qwen3.5 these are 12 non-special tokens, including `<think>`, `<tool_call>`, `<tool_response>` and `<|fim_*|>`.

The model therefore sees the same characters, and the encoder fails closed with `unsupported/control_tokens_in_input`. As a result, `<|im_end|>`/`<|im_start|>`/`<think>` inside a request never becomes a control token.

Fixture revision note: the fixture was recorded at `29b858c`, which used split-only encoding, and pins that `parent_revision`. A tokenizer-only CPU check shows that every encoded sequence in this fixture is identical under the canonical encoder. The result is 173/173, in `receipts/.../fixture-tokenization-check.json`. The recorded logits are therefore unchanged; only the live backend's `revision_id` (cfg hash) differs.

Case `injection-03-added-tokens` is in `scripts/fixture_cases.py` but is not recorded yet. It is added at the next regeneration, which is pending because no model runs are allowed on this laptop. A control-token audit over all 28 ok cases found no prompt with more special ids than the template's own 7.

Option ids or question ids containing control characters or `<|`/`|>` are rejected (`error/invalid_option_id`).

A text template cannot prevent plain-text role spoofing (`END STATE`, `Assistant:`); only control-token spoofing is prevented.

Steer behaviour is a separate matter: the 0.8B readout still answers `approve` on all three injection cases (0.997 / 0.845 / 0.831). That is a model/steer finding for the m1 steer baseline, not a boundary failure.

## Scope

The fixture proves boundary behaviour only. A 0.8B readout is near chance on hard items; for example, it picks `osaka` as the capital of Japan.
