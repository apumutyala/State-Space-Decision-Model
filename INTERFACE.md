# D → L interface

The canonical contract is [`interfaces/DECISION_LANGUAGE_V0.md`](interfaces/DECISION_LANGUAGE_V0.md), version 0.1 (Codex, PR #4). This file keeps only what is not yet in that contract.

Claude's original three requests (attribution, semantic_records, policy-conditioned probabilities) have been incorporated into the canonical 0.1 text, so they are no longer listed here.

## Track D implementation notes (D adapter `decide/`, adapter 0.2.0)

These document how D realises the contract. They do not change it.

- **Transport.** The legacy SystemOne `model` / `state` / `questions` fields sit at top level. The D fields go in a versioned `ssdm` envelope, `schema: "ssdm.decide.v0.1"`. Unknown envelope or limit keys are rejected as `error/invalid_request`.
- **Limits.**
  - `max_questions` and `max_options_per_question` are D keys.
  - `max_choices` is accepted only as a legacy alias of `max_questions`. A conflicting alias is `error/invalid_request`.
  - Exceeding a limit gives `exhausted/questions_exceeded` or `exhausted/options_exceeded`. The wire maximum is 255 options (`unsupported/too_many_options`).
- **Policy restriction.**
  - Duplicate ids are rejected.
  - The policy-conditioned stage covers the full candidate set, with zero on excluded options. It carries `allowed`, `restriction_id`, `excluded_served_mass` and `status` (`ok` | `unavailable_zero_served_mass`; probabilities are `null` when unavailable).
- **Request fingerprint.** `diagnostics.request_fingerprint` is computed with algorithm `ssdm.request_fingerprint.v0.1`:

  ```
  sha256(canonical_json({"state": state, "questions": {qid: {"type", "instructions", "options": canonical_options(q)}}}
                        + {"provenance": ssdm.provenance} only if supplied))
  ```

  - `canonical_json` is `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
  - `canonical_options` orders noul as (true, false), choice by sorted id, and score by level.
  - Without provenance, the result equals the v0 algorithm.
- **Failures.** A non-ok response carries a top-level `error {status, code, message}` and `answers == {}`.
- **Abstention.** `abstention.mechanism = "not_implemented_v0"`.
- **Untrusted text.** It is encoded only by `decide/untrusted.py`, which fails closed as `unsupported/control_tokens_in_input`.

## Pending delta: request provenance (proposed on PR #4; D accepts it now)

The optional `ssdm.provenance` field has schema `ssdm.provenance.v0` and is populated by the L bridge from `serialization_ref`. Its keys are `serializer`, `default_trust` and `segments{field: [{start, end, trust ∈ {instruction, data}, origin ∈ {host_template, record, task_input, program_prefix, other}, source_ref?, audience?, epistemic_class?}]}`.

Field names are `state`, `questions.<qid>.instructions` and `questions.<qid>.criteria.<option_id>`.

Rules:

- Offsets are code points into the exact scored string.
- Segments tile each string, with no gaps or overlaps.
- `state` must be a string.
- D tokenises each segment separately. Both trust classes go through `decide/untrusted.py`.
- The provenance is echoed in `diagnostics.provenance` (`supplied`, `applied`, segments, per-label token counts, `encoder_id`) and is covered by the request fingerprint.
- Malformed provenance is `error/invalid_provenance`.

Full rules are in `decide/provenance.py`. This delta becomes canonical once the provenance row lands in the contract.

## Change rules

Interface changes need both agents' approval and a version bump (`COORDINATION.md`).
