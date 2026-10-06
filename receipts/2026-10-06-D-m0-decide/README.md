# Receipt: m0-b, the `decide` adapter, backends and replay fixture (Track D)

Date: 2026-10-06 · Branch: `claude/m0-decide-adapter` · Code commit used for generation: `45a399a`.

The fixture was generated while the worktree was dirty, but only from untracked files the generator does not import (tests, smoke script). `git diff 45a399a -- decide scripts/gen_replay_fixture.py scripts/fixture_cases.py` is empty.

- Hardware: local RTX 5070 Ti Laptop GPU (12 GB, shared with other processes), Windows 11.
- Software: Python 3.12.13, torch 2.8.0+cu128, transformers 5.17.0.
- Cost: **$0 billed**.

## Takeaway

1. `decide(request, state_handle=None)` works end to end on the stock Qwen3.5-0.8B readout.
2. The replay fixture reproduces the live run exactly (36/36 non-timing cases).
3. Live-model permutation equivariance holds exactly: max |Δp| = 0.0 over 52 shuffles.
4. Live runs are deterministic: Δp = 0.0 across 3 runs.
5. All three backbone families load and score locally on GPU: Qwen3.5 hybrid, RWKV7 and Mamba-3 SISO.
6. Two of the three show a strong first-listed-option bias, which matters for the readout design (see below).

## 1. Tests

`python -I -m pytest -p no:cacheprovider tests -rs` gives **90 passed, 4 skipped** (`pytest.txt`).

- The skips are 3 unused parametrised indices plus the timing-dependent deadline case.
- With `SSDM_QWEN_DIR` set, the run includes the real-model equivariance test.

What the suite covers:

- **Conformance:** canonical fields, wire semantics (noul = p_true; choice confidence = selected p; score = raw expected ordinal index; score confidence = max p), entropy diagnostic, versions.
- **Status/limits:** every reason code; 255 options allowed and 256 rejected; no truncation; no pruning; no fallback; deadline returns no partial answers.
- **Policy restriction:** policy-conditioned probabilities are separate and never substituted.
- **Equivariance:** checked on the oracle, a tiny CPU LM through the real HF readout path, Qwen3.5-0.8B on GPU, and the replay fixture.
- **Replay:** reproduction and manifest hashes.

## 2. Replay fixture (`fixtures/replay/v0`)

- Command: `python -I scripts/gen_replay_fixture.py --qwen-dir <Qwen3.5-0.8B snapshot 2fc0636…> --out fixtures/replay/v0 --runs 3`
- Parent revision id: `qwen3.5:Qwen/Qwen3.5-0.8B@2fc06364715b;w=04b1c301231d;tok=5f9e4d4901a9;prompt=jrt-chatml-v1.3b45ac33;cfg=d2735c84;adapter=0.1.0`
- Weights sha256: `04b1c301…4696`.
- Content: 37 cases. 26 are `ok`; 11 cover each non-ok reason code once.
- Determinism: max served-probability difference across 3 runs = 0.0.
- Live equivariance: 52 shuffles (question and option order), max |Δp| = 0.0, 0 choice flips.
- Latency (ok cases, median of 3 runs per case): median 624 ms, range 266 ms to 26 s. The 26 s is the 1.5k-token needle with 4 options.
  - These numbers are not a latency claim. Qwen3.5 Gated DeltaNet ran on the transformers *reference PyTorch fallback* (no `fla`, no `causal_conv1d`), and the laptop GPU is shared.
- Raw output: `fixture-generation-summary.json`, which includes per-case answers.

## 3. Backbone smoke tests (`smoke-*.json`)

Each run is `python -I scripts/smoke_backbones.py --family … --model-dir …`, with 4 requests × 3 repeats plus an equivariance check and an option-order probe.

| Backbone (smallest) | Loads? | Stack | Peak mem | Equivariance | France capital | First-listed bias probe |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (18 GDN + 6 attn) | yes, GPU bf16 | transformers 5.17 native, torch fallback kernels | 1.6 GB | 0.0 | paris | **no** (p(technical) 0.61 / 0.71 under both orders) |
| RWKV7 G1 0.1B (`fla-hub/rwkv7-0.1B-g1`) | yes, GPU only | `fla` 0.5.2 + triton-windows 3.4 (TinyCC launcher), local World tokenizer | 0.8 GB | 0.0 | berlin (wrong) | **yes** (whichever id sorts first wins: 0.96 → 0.59) |
| RWKV7 on CPU | no | fla kernels are Triton/GPU-only (`Pointer argument cannot be accessed from Triton (cpu tensor?)`) | — | — | — | — |
| Mamba-3 SISO 187M (`state-spaces/mamba3-siso-187m`) | yes, GPU | `mamba_ssm` 2.3.2.post1 from git `e9594ce` (no CUDA build) + triton-windows 3.5.1, Llama-3.1 tokenizer from an ungated mirror | 0.65 GB | 0.0 | paris | **yes** (0.96 → 1.00 for the first-sorted id) |

How the loads were done:

- The `fla` and `mamba_ssm` loads use no `trust_remote_code`. The RWKV World tokenizer is reimplemented locally (`decide/rwkv_world_tokenizer.py`, `ast.literal_eval` only).
- Extra packages were installed `--no-deps --target` into gitignored worktree dirs (`.venv-extras*`); the shared venv is untouched.
- Mamba-3 fails with the triton 3.4 that fla uses (`CompilationError` at `tl.dot`), because `mamba_ssm` requires triton ≥ 3.5. The two stacks therefore need separate triton dirs.

## Findings for Track D (adversarial)

- **Order bias in recurrent readouts.** Canonical (sorted-id) option order makes outputs exactly invariant to how the caller orders the request. It does *not* remove position bias: for RWKV7-0.1B and Mamba-3-187M, the option that sorts first wins regardless of meaning. Equivariance under relabelling is therefore not implied. The stock-readout m0 numbers need either a relabel/debias control (e.g. average over K cyclic option orders) or the trained head.
- **Clef truncates; we refuse.** Clef's `encode_record` silently truncates state to `max_length`. Our adapter returns `exhausted/input_tokens_exceeded` instead, as the interface requires. Parity runs against Clef must exclude or flag truncated records.
- **Clef's argmax tie-break follows insertion order.** Ours breaks ties by sorted id. Ties are measure-zero in practice.
- **Deadline granularity.** The deadline is checked between inference calls, not by preemption, so a single long call can overrun it before `exhausted/deadline_exceeded` is returned.
- **Choice of interpretation (flag to Codex).** `limits.max_choices` is read as the number of questions per request. Per-question options are capped at 255 by the wire cardinality.

## Inputs read as text only (never executed)

- `Cloudflare/clef-flash/joint_schema_model.py`, sha256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`, plus its README. Both are stored outside the repo in `ssdm-data/clef-flash-src/`.
