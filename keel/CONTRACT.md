# Keel interface contract (v0.1, 2026-10-15)

Shared rules for every Keel subpackage. Owned by the chief of staff; propose changes by message, do not edit.

## Package map and ownership

| Path | Owner | Status |
|---|---|---|
| `keel/protocol/types.py`, `keel/protocol/render.py`, `keel/protocol/__init__.py` | chief of staff | frozen v1 — request changes |
| `keel/runtime/modal_app.py`, `keel/runtime/vllm_judge.py` | chief of staff | frozen v1 — request changes |
| `keel/protocol/receipts.py`, `keel/host/`, `PROTOCOL.md` | protocol/host slice | new |
| `keel/gate/` | gate slice | new |
| `keel/card/` | card slice | new |
| `keel/distill/` | distill slice (v1: capability data authoring extends it; keep v0 modules working) | active |
| `keel/workflows/` | workflow slice | active |
| `keel/monitor/` | monitor slice (Card for trusted monitors in a control setting) | new |
| `keel/paths.py` | chief of staff | frozen — request changes |
| `keel/tests/test_<slice>*.py` | each slice | new |

## Core types (`keel.protocol`)

- `Segment(segment_id, content, trust, kind, source, parent_ids)`. `trust` ∈ {trusted, untrusted}. Only `operator_note` and
  `metadata` kinds may be trusted, and they must be. Trust comes from the host API that built the segment, never from
  content. `derive(...)` propagates trust: anything derived from an untrusted parent is untrusted.
- `Question(qid, type, instructions, criteria)`, where type ∈ {choice, noul, score}.
- `OperatorPolicy(policy_id, text, version, authorized_actions)`.
- `DecisionRequest(request_id, policy, segments, questions)`. It round-trips through `to_json()` / `from_json()`.
  `fingerprint()` is the sha256 of the canonical JSON.
- `from_jevbench_item(item, policy=None, extra=())` builds a request whose state is one untrusted `record` segment.
- `Answer`, `DecisionResult` hold the per-question option probabilities, the act flag and defer reasons.

## Rendering (`keel.protocol.render`)

- `render(request, qid, mode=("trust"|"base"), think=bool, canonical=True) -> Rendered`.
  - `Rendered.prompt` ends at the answer slot `Answer:` (one-pass) or at `<think>\n` (think).
  - `option_ids` and `labels` are in read order; labels are single tokens " A", " B", … for Qwen3.5.
  - `canonical=True`: options ordered by content (reorder and relabel invariant by construction).
  - `mode="trust"`: the system turn holds base instructions + `TRUST_POLICY_V1` + operator policy + trusted notes and
    metadata. Each untrusted segment is in an escaped `<untrusted_record source=...>` block in the user turn.
  - `mode="base"`: all segments merged into one record (the undefended baseline).
- `close_think(rendered, reasoning, truncated)` builds the post-reasoning answer-slot prompt.
- `probs_from_logprobs(rendered, logprobs, temperature)` gives option-id probabilities, with ties equalised.
- `decision_of(probs, qtype)` returns the argmax; yes/no maps to "yes"/"no".

## GPU runtime (`keel.runtime`)

- Modal image `vllm_image` (vLLM 0.30.0); model cache volume `ssdm-hf-cache` at `/root/.cache/huggingface`.
- `MODELS` holds the pinned revisions (`qwen3.5-4b`, `qwen3.5-9b`, `qwen3.5-2b`, `h2o-lightning-4b`). A model
  argument is a `MODELS` key or a checkpoint path on the `keel-results` volume, which is mounted at `/results`. A
  merged student, for example, lives at `/results/students/<run>/merged` and must be a full HF checkpoint vLLM can load.
- `vllm_judge.score.remote(model, jobs, think=, n_samples=, max_think=, temperature=, seed=, keep_reasoning=)` is a
  one-shot batch call:
  - `jobs` = `[{"id", "prompt", "labels"}]`;
  - returns per job `samples[{logprobs, missing, think_tokens, truncated, reasoning}]` plus run metadata.
- `vllm_judge.JudgeServer(model=...)` is a warm Modal class (`.score.remote(jobs, **same kwargs)`). Use it for
  adaptive search and other repeated calls; it stays up for 10 minutes after the last call.
- `vllm_judge.judge(requests, model, mode=, think=, canonical=, n_samples=, server=None)`:
  - call it inside `with vllm_judge.APP.run():`;
  - returns per request `{"probs", "samples", "option_ids", "think_tokens", "reasoning", "meta"}`;
  - it raises if a label log-prob is missing.
- Smoke check: `python -X utf8 -m keel.runtime.smoke_judge --n 4`.
- Training images and functions belong to the slice that needs them (e.g. `keel/distill/modal_train.py`). They reuse
  `hf_cache` and write checkpoints to `results_vol`.

## Data rules

- **JevBench public items are EVAL-ONLY.** They live at `$SSDM_DATA/jevbench/b6b8fff7e345b98c060ad26c13308860ddc67004/`
  (`SSDM_DATA=~/ssdm/data`) and are loaded with `jevkit.items.load_public()`. Never train on them, tune on
  them, select on them or prompt-engineer against them; Card evaluation only.
- **Training data passes the repo gates:**
  - `train/jevbench_guard.py` (JevBench overlap);
  - `train/attack_guard.py` (no evaluation attack templates in training);
  - provenance per row.
  The Decision Index document-overlap index is not available on this workstation; record that it was not run.
- **Attack templates used in training are disjoint from Card evaluation templates.** The Card owns the held-out
  families (including `analysis/safety_heldout/heldout_v1_templates.json`).
- **Per-item model outputs and dataset text stay out of git.** Use `$SSDM_PRIVATE` (default `~/ssdm/private`,
  resolved by `keel.paths.private_root()`) or the Modal volume `keel-results`. Commit code, configs, aggregates and
  README results.

## Engineering rules

- Workspace: `~/ssdm` (see its `README.md`); `source ~/ssdm/env.sh` sets `SSDM_DATA`, `SSDM_PRIVATE` and the venv.
- One git worktree per slice under `~/ssdm/worktrees/keel-<slice>`, on branch `collab/mottopanikeiku/keel-<slice>`,
  cut from `collab/mottopanikeiku/keel`. Commit every change (small commits, descriptive messages, no AI or co-author
  trailers) and push the branch.
- Python venv: `~/ssdm/.venv` (3.12; pinned in `~/ssdm/requirements.txt`: torch CPU, transformers 5.14.1, pytest,
  hypothesis, numpy, scipy, modal, agentdojo). Run from the repo root with `-X utf8`. Add packages with
  `uv pip install --python ~/ssdm/.venv/bin/python <pkg>`, refresh `requirements.txt` with `uv pip freeze`, and list
  them in your README.
- Tests: CPU-only, deterministic, under `keel/tests/`. Test behaviour and invariants, not wiring.
- Modal spend: each slice has its own cap (in its brief). Log every paid run (date, function, GPU time, cost estimate)
  in the slice README.
