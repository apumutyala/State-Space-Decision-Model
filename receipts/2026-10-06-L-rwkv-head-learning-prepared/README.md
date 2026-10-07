# RWKV CPU head-learning diagnostic: prepared, not run

One fixed recipe tests whether Track D's existing snapshot decision head can
learn useful semantic choices over the cached frozen RWKV7-0.1B reference.
This is an exploratory surrogate, not an L0 gate, a trained Track D parent,
adapted-head decision retention, or a strong unrestricted-comparator result.
No cached checkpoint was inspected or loaded and no real model was trained.

The variant branch is `stack/rwkv7/l0-head-only`. Distinct dependency merge
commits preserve PR #85 `e9447e64e97fac8a2e0b3377acdacc4e10d343e9` and PR #71
`14aa0daa486804ca82bf332dd0469717cc7a040b` on top of agent-main
`0c8eaf7276dd7680aedd7d7b3b7d6814eada5f41`; their independent review/merge
gates remain intact. These local dependency merges do not integrate them into
agent-main.

`registration.json` freezes source, expected checkpoint/tokenizer, corpus and
split hashes before any real run. `corpus.json.gz` preserves exactly the
generated dry-run corpus. Both domains are synthetic iterative development;
the generator separates names, numeric ranges and question templates. All
24 train worlds and 12 validation worlds remain in their respective cohorts,
with equal read/explain/compare counts. No TEST or Index data is involved.

The single recipe uses seed 20261006, 24 optimizer steps, batch size 2,
unsmoothed CE, AdamW, learning rate 0.001, a four-channel/two-snapshot head,
and no LoRA. Only existing `head.*` and `typed.*` parameters can train; the
backbone remains frozen. This is deliberately a small fit. A failure applies
to this exact recipe, and cannot falsify the broader recurrent-parent approach.

Primary outcomes are free-running complete useful answers and operation/
argument-slot errors. The paired CI resamples the 12 generated worlds, keeping
all slots of each world together. One seed and 12 worlds cannot establish
generalization. Kernel acceptance alone is never credited as a useful answer.
The unchanged LM-head direct comparison path is measured before and after,
on identical requests, with exact wire/full-probability parity required. This
is a frozen-path integrity check; it is not adapted-head D-parent retention.
No causal intervention, adversarial soundness or strong-comparator gate is
claimed by this pilot.

The launcher sets offline CPU mode and two threads, refuses less than 8 GiB
available at start, and monitors a 3 GiB running floor plus a 900-second hard
cap. It kills only its owned child. A default/dry CLI run imports no Torch,
opens no checkpoint, and requires a fresh output directory. Real execution
also refuses drift from this committed registration. Checkpoint restoration
requires exact trained-state keys/shapes/dtypes plus trainer config/lineage.

Preparation command, already completed:

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_head_learning --dry-run --out C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-learning-prepared-20261006
```

The output was `prepared_not_run`. Runner source SHA256:
`74fc73bcad7d336c890e7c4627da8f56716a83f1bf578d30e5f0ab4b0de5944d`.

Proposed guarded command, **not authorized or executed**; coordinator approval
and a current RAM/resource check must precede it:

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_head_learning --execute --out C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-learning-run1-20261006
```

Run either command from the variant checkout. The launcher handles child
environment settings. No paid job or GPU is involved. Paid jobs elsewhere
must continue to use `python -m jobs.common.launch` and the launch checklist.

Preparation verification uses tiny random CPU RWKV only. It covers strict
save/restore, frozen backbone, current labels outside requests, no gold
program/prefix leakage in free-running inference, typed-choice attribution,
fresh-output preservation, preregistration drift and all monitor stop paths.
The dependency and runner suite is recorded in `verification.json`.
