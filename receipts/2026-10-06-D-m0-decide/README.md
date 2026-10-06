# Receipt: m0-b, the `decide` adapter, backends and replay fixture (Track D)

Date: 2026-10-06 · Branch: `claude/m0-decide-adapter` · Cost: **$0 billed**.

- Hardware: local RTX 5070 Ti Laptop GPU (12 GB, shared) and CPU, Windows 11.
- Software: Python 3.12.13, torch 2.8.0+cu128, transformers 5.17.0.

Revision history: v1 at `45a399a`/`a497d6b`, then the internal review's fixes at `29b858c` (PR #6 comment "Track D internal adversarial review (pre-Codex)").

- The fixture was regenerated on **CPU fp32** at `29b858c`.
  - The manifest says `worktree_dirty: true` only because the generator writes the fixture files before reading git status.
  - `git diff 29b858c -- decide scripts` is empty for the generation inputs.
- The backbone smoke JSONs (`smoke-*.json`) are from v1 code at `45a399a`, on GPU. Their prompt hash predates the segmented prompt builder.

## Takeaway

1. `decide(request, state_handle=None)` works end to end on the stock Qwen3.5-0.8B readout.
2. The handoff fixture (40 cases) replays exactly. Replay is a deterministic function of the recorded logits, and an adapter-free recompute with Cloudflare's own answer logic matches.
3. Request text can no longer inject control tokens. The audit found 0 excess special ids over 28 ok cases.
4. All three small backbone families load locally.
5. Model-level evidence in this receipt is limited to:
   - determinism on one stack (Δp = 0 across runs);
   - the smoke answers;
   - the Qwen debias controls below.

## 1. Tests

`python -I -m pytest -p no:cacheprovider tests -rs` (CPU, no `SSDM_QWEN_DIR`) gives **111 passed, 10 skipped** (`pytest.txt`).

- The skips are 8 unused parametrised indices, the timing-dependent deadline case, and the GPU real-model equivariance test.
- With `SSDM_QWEN_DIR` set, the real-model test also runs. The v1 run had it at 90/4 on GPU.

What the suite covers:

- **Conformance:** canonical fields, Clef wire semantics, legacy `error` on non-ok, abstention label.
- **Status/limits:** every reason code.
- **Injection** (`test_injection.py`):
  - untrusted segments cover every request field;
  - a fake tokenizer shows that a leaky encoder is refused with `unsupported/control_tokens_in_input`;
  - the **real Qwen3.5 tokenizer on CPU** gives the template-only `<|im_start|>`×3 and `<|im_end|>`×2 with the injected state;
  - option/question-id validation.
- **Debias controls** (`test_debias.py`, CPU): a position-biased oracle gets TV > 0.5 and is id-sticky; a description-reading oracle gets TV = 0 and follows the description; the controls also run through the HF path on a tiny CPU LM.
- **Equivariance:** an *adapter* property. The prompt is byte-identical under shuffles, so any deterministic backend passes. It is not evidence of model order-robustness.
- **Replay:** contract plus an adapter-free recompute using Clef's `systemone_answer`, verbatim.

## 2. Replay fixture (`fixtures/replay/v0`)

**Parent:** Qwen3.5-0.8B, *not* the 9B promised in the handoff. The 9B fixture comes with the paid m0 run.

- Command: `python -I scripts/gen_replay_fixture.py --qwen-dir <Qwen3.5-0.8B snapshot 2fc0636…> --out fixtures/replay/v0 --runs 2 --device cpu --dtype float32`
- Revision id: `qwen3.5:Qwen/Qwen3.5-0.8B@2fc06364715b;w=04b1c301231d;tok=5f9e4d4901a9;prompt=jrt-chatml-v1.f7d108c4;cfg=9c55f30e;adapter=0.1.0`. The template identity now includes a hash of rendered canary prompts.
- Cases: 40 in total. 28 are ok; 12 are non-ok, covering every reason code once, including the new `error/invalid_option_id`.
- Determinism (model evidence, this CPU stack only): served Δp = 0.0 across 2 runs.
- Replay: 39/39 non-timing cases match, and the adapter-free recompute matches. Both are adapter evidence.
- Control-token audit: template = 7 special ids per prompt; max excess over all ok prompts = 0.
- Injection cases:
  - `injection-01-role-tokens`: real `<|im_end|>`/`<|im_start|>` in state, instructions and a description;
  - `injection-02-plaintext-role`: `END STATE` / `Assistant:` (cannot be prevented by any text template);
  - `choice-12-injection`: a textual override.
  - The readout answers `approve` in all three (0.997 / 0.845 / 0.831). This is a steer finding, not a boundary failure.
- Latency is not a claim (CPU fp32, torch fallback kernels): median 6.2 s per ok case.
- Raw output: `fixture-generation-summary.json`.

## 3. Backbone smoke tests (`smoke-*.json`, v1 code, GPU)

| Backbone (smallest) | Loads? | Stack | Peak mem |
|---|---|---|---|
| Qwen3.5-0.8B (18 GDN + 6 attn) | yes, GPU bf16 (and CPU fp32 for the fixture) | transformers 5.17 native, torch fallback kernels | 1.6 GB |
| RWKV7 G1 0.1B | yes, **GPU only** (fla kernels are Triton-only) | `fla` 0.5.2 + triton-windows 3.4 (TinyCC launcher), local World tokenizer | 0.8 GB |
| Mamba-3 SISO 187M | yes, GPU | `mamba_ssm` 2.3.2.post1 @ git `e9594ce` (no CUDA build) + triton-windows 3.5.1, Llama-3.1 tokenizer (ungated mirror) | 0.65 GB |

Since v1 the RWKV7 and Mamba-3 plugins cap input at the trained context (2048). Recurrent models do not truncate; they degrade silently past it.

## 4. Option-order bias: downgraded, confounded

Tally from the smoke receipts (choice argmax = sorted-first id; score argmax = level 0):

| Backbone | Smoke questions with sorted-first argmax |
|---|---|
| RWKV7-0.1B | 5/5 |
| Mamba-3-187M | 2/5 |
| Qwen3.5-0.8B | 1/5 |

The v1 probe (`billing`/`technical` vs `a_technical`/`b_billing`) changed three things at once: sorted position, continuation token count, and an MCQ-letter-like `a_` prefix. It is **confounded** and must not drive readout design.

The controls are now in code (`decide/debias.py`, `scripts/run_debias_controls.py`):

- **(a) description swap with fixed ids:** moves only the meaning-to-id mapping.
- **(b) neutral length-matched ids `opt_01..opt_k` under all k cyclic presentations:** position-bias TV vs uniform, plus rotation-averaged probabilities as a labelled diagnostic variant (never served).
- **(c) content-free prior** (`state = "N/A"`).

Qwen3.5-0.8B results (CPU fp32; `debias-qwen3.5-0.8b-cpu.json`; 3 probes):

| Probe | Position-bias TV | Description swap | Content-free p(sorted-first) |
|---|---|---|---|
| billing/technical | 0.003 | id-sticky | 0.551 |
| paid/overdue/draft | 0.028 | id-sticky | 0.995 (`draft`) |
| 4 capitals | 0.061 | id-sticky | 0.006 |

How to read this:

- Qwen-0.8B shows little position bias under neutral ids.
- It reads the semantic *ids*, not the descriptions. This is expected when the ids carry the meaning, and it means descriptions alone do not steer it.
- Its content-free prior is strongly option-dependent (e.g. 0.995 on `draft` with no state). That supports contextual calibration as a diagnostic.

RWKV7 and Mamba-3 controls need the GPU and are **not run yet**; they are required before any stock-readout m0 number is used for design.

## 5. Canonical untrusted-text encoder (`decide/untrusted.py`, commits `ce7ae36`+)

The PR #8 review found that `split_special_tokens` alone leaves 12 *non-special* added Qwen tokens as single ids (`<think>`, `</think>`, `<tool_call>`, `<tool_response>`, `<|fim_*|>`, `<|repo_name|>`, `<|file_sep|>`). As a result, PR #6 refused input that PR #8 scored.

There is now one encoder, `UntrustedEncoder(tokenizer).encode(text)`, and `safety_eval` must import it. It works in three steps:

1. Split special tokens with the flag.
2. Cut, at the midpoint, every added token that still parses as one id under the flag.
3. Fail closed with `ControlTokenError` (`status = unsupported`, `code = control_tokens_in_input`) if any special or added id survives.

Unlike ZWSP, the model reads the same characters (`decode(ids) == text`).

Tests (`tests/test_untrusted.py`, tokenizers only, CPU):

- Every added token of Qwen3.5 (33) and the Llama-3.1/Mamba-3 tokenizer (256) is neutralised and round-trips.
- A realistic `<|im_end|>`/`<think>`/`<tool_response>` spoof is neutralised.
- An uncuttable 1-character added token fails closed.
- The Qwen leak is confirmed: `<think>` survives the flag alone.

Fixture compatibility (`fixture-tokenization-check.json`, `scripts/check_fixture_tokenization.py`, tokenizer only): all 173 encoded sequences of fixture v0 are identical under split-only (the fixture recording) and the canonical encoder. The recorded logits stay valid. The fixture keeps its recorded `parent_revision` (`cfg=9c55f30e`); the live backend's cfg hash now differs because the encoder id is part of it. Regeneration, which adds `injection-03-added-tokens`, is deferred: no model loads are allowed on this laptop.

Tests: `pytest.txt` is now **115 passed, 10 skipped** (CPU).

## 6. Other review items addressed

- **Legacy wire failure signal:** top-level `error {status, code, message}` on non-ok. This is documented in `decide/schema.py`, and transports must map it to HTTP errors.
- **Documented deliberate differences from Clef:**
  - stricter validation (1-level score, extra noul criteria keys, id charset);
  - `usage.input_tokens` = total forwarded tokens across option rows (the longest single input is in diagnostics).
- **Template identity:** now includes a hash of rendered canary prompts, so body-builder edits change `revision_id`.
- **Backend diagnostics:**
  - per-option `continuation_tokens`;
  - `length_normalized_logits`, a diagnostic for the token-length prior of sum-logprob scoring. It is not served.
- **Abstention:** labelled `mechanism: not_implemented_v0`.
- **Questions are not separable:** documented in the fixture README with the `choice-02` vs `multi-01` pair.

## Inputs read as text only (never executed)

- `Cloudflare/clef-flash/joint_schema_model.py`, sha256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`, plus its README. Both are stored outside the repo.

## 7. Pre-alignment with contract 0.1 (PR #4 @ `8109b13`; adapter 0.2.0; CPU only, no model loads)

| Change | Where |
|---|---|
| Envelope `ssdm.decide.v0.1` (v0 rejected as `envelope_version_unsupported`) | `decide/versions.py` |
| `max_questions` and `max_options_per_question` limits (`exhausted/questions_exceeded`, `exhausted/options_exceeded`). `max_choices` is accepted only as an alias of `max_questions`; a conflicting alias is `error/invalid_request`. Planner budgets are rejected | `decide/adapter.py` |
| `policy_restriction` is core, no longer behind the flag. Duplicate ids are rejected. The stage covers the full candidate set (zero on excluded) with `restriction_id`, `status` `ok` or `unavailable_zero_served_mass` | `decide/adapter.py` |
| `diagnostics.request_fingerprint_algorithm = ssdm.request_fingerprint.v0.1` (equal to v0 without provenance), plus `diagnostics.probability_stages` | `decide/versions.py`, `decide/schema.py` |
| Optional `ssdm.provenance` (schema `ssdm.provenance.v0`, exactly as proposed on #4): code-point tiling, per-segment tokenisation through `decide/untrusted.py`, echo in `diagnostics.provenance`, covered by the fingerprint, `error/invalid_provenance` | `decide/provenance.py`, `decide/prompt.py`, HF readout |
| Root `INTERFACE.md` reduced to a pointer at the canonical contract, plus D notes and the pending provenance delta | `INTERFACE.md` |

Fixture: rebuilt from the unchanged recordings by `scripts/rebuild_fixture_from_recordings.py`, with no model run.

- Cases: 43. Added: duplicate restriction, `options_exceeded`, conflicting alias, `invalid_provenance`. Removed: the flag-off restriction case.
- All 28 ok cases have the same answers, raw/served probabilities and fingerprints as before.
- Replay is exact for 42/42 non-timing cases.
- The template identity is unchanged (`jrt-chatml-v1.f7d108c4`).
- Tokenizer check: 173/173 sequences are unchanged.

Tests: **141 passed, 7 skipped** (CPU), including `tests/test_provenance.py`.
