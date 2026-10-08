# D → L interface

The canonical contract is [`interfaces/DECISION_LANGUAGE_V0.md`](interfaces/DECISION_LANGUAGE_V0.md), profile 0.1.4 (base 0.1: Codex, PR #4 @ `787d64f`, approved by Track D). Profile 0.1.2 adds host-opted-in bounded exact difference under interface issue [#201](https://github.com/apumutyala/State-Space-Decision-Model/issues/201); Profile 0.1.3 adds registered conditional hypotheses under issue #205. Profile 0.1.4 adds registered non-executing plans under issue #207. D transport remains v0.1. This file keeps only what is not yet in that contract.

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
- **Request fingerprint.** D implements the contract's pinned `ssdm.request_fingerprint.v0` exactly, and `diagnostics.request_fingerprint_algorithm` says so on every provenance-free `ok` response.
  - A provenance-bearing request is possible only with the experimental flag. It uses the separately versioned `ssdm.request_fingerprint.provenance.experimental-0.2`: the same payload plus `"provenance"`.
- **Failures.**
  - A non-ok response carries a top-level `error {status, code, message}` and `answers == {}`.
  - Messages are fixed text per code; raw exception text is never returned.
  - `diagnostics.inference_calls` counts attempted calls, with an accounting note. It is `null` when unknown, never a false zero.
- **Abstention.** `abstention.mechanism = "not_implemented_v0"`.
- **Untrusted text.** It is encoded only by `decide/untrusted.py`, which fails closed as `unsupported/control_tokens_in_input`.

## Planned 0.2: request provenance (experimental in D, behind a flag)

The contract lists `ssdm.provenance` as a planned optional 0.2 extension. D carries experimental support only behind `Decider(enable_experimental_provenance=True)`. Without the flag, a request that carries it is `unsupported/provenance_not_enabled`. The 0.1 L0 bridge sends none.

The shape follows the contract: schema `ssdm.provenance.v0`, plus `serializer`, `default_trust` and `segments{field_path: [{start, end, trust, origin, source_ref?, audience?, epistemic_class?}]}`.

D's experimental choices, proposed for ratification in 0.2:

- **Field paths.** These use RFC 6901 JSON Pointer (`path_encoding: "rfc6901-json-pointer"`), with `~` escaped as `~0` and `/` as `~1` inside ids. The paths are `/state`, `/questions/<qid>/instructions` and `/questions/<qid>/criteria/<option_id>`. The encoding is unambiguous for ids containing `.` or `/`; the earlier dotted form was not.
- **Trust default.** `default_trust` may only be `data`. Every unlisted field defaults to `data`, including question instructions, as the contract's tightened rule requires. Instruction authority comes only from explicitly declared spans. Verifying the host-template binding of those spans is the bridge's responsibility.
- **Strings and spans.** Offsets are code points into the exact scored string, and `state` must be a string. Spans tile each field. Malformed provenance is `error/invalid_provenance`.
- **Tokenisation and echo.** Each span is tokenised separately, and both trust classes go through `decide/untrusted.py`. `diagnostics.provenance` echoes `applied` (false for the stock readout), the segments, per-label token counts and the encoder id.

Full rules are in `decide/provenance.py`. Both tracks must approve 0.2 before any measurement uses this extension.

## Change rules

Interface changes need both agents' approval and a version bump (`COORDINATION.md`).
