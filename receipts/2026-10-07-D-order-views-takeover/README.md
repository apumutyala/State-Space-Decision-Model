# Permutation-view data WIP takeover (CPU only)

Base: fetched `origin/agent-main` at `ee0adb70029fc80c65ee6072c307873034052362`.
Branch: `codex/rwkv7-order-views-takeover`.
Worktree: `C:/Users/apumu/research/CHLabs/ssdm-codex-rwkv7-order-views`.

Copied only the dirty `train/data.py`, dirty `train/lineage.py` and untracked `train/order_views.py` from
`C:/Users/apumu/research/CHLabs/ssdm-wt/rwkv7-cell2` (source HEAD `ee0adb7`). The source worktree was not edited.
Source SHA256 values, checked before copying and again before committing:

| Source file | SHA256 |
| --- | --- |
| `train/data.py` | `83fa6e04827b61077c26cd0e1fd02d8c8d01d32bfcbf4af3b2e849ec349b8f21` |
| `train/lineage.py` | `f49c94e4021d8c54b769d519391e703857dbe5caa7ff39f97a09f28a48c7fbc1` |
| `train/order_views.py` | `5ab244b00a3bed6ac0130b49425d6d1181a1807f08f4379b9f0d8168e06c54f9` |

The inherited helper reverses descriptions under fixed canonical option ids and moves the gold label with its
description; `view_perm[k]` maps view index k to the base index. The takeover fixes source aliasing in the returned
target question and skips choices with any `None` description: swapping `None` can leave identical prompts with
contradictory labels. The module documentation now scopes the helper to data construction without claiming an
enabled cell, preregistration, or unverified model results.

Validation used `C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m pytest -q -p no:cacheprovider`,
with `CUDA_VISIBLE_DEVICES` empty and `PYTHONDONTWRITEBYTECODE=1`:

| Arguments after the command prefix | Result |
| --- | --- |
| `tests/test_order_views.py --tb=short` | 23 passed; 1.75 s |
| `tests/test_order_views.py tests/test_guard.py --tb=short` | 27 passed; 1.46 s; four existing PytestReturnNotNoneWarning warnings in `test_guard.py` |

Synthetic cases prove nested/flat-record text and gold mapping, the exact self-inverse permutation, unchanged
non-target values, no source aliasing, choice-only eligibility, rejection of id-only descriptions, filter and
guard application, explicit attachment to a training list while separately built evaluation items stay unchanged,
unchanged base LayoutRows, and legacy-exact lineage that changes with view labels, tokens or permutation.

Limitations: no callsite currently invokes `attach_views`; trainer/loss/cell integration remains separate work.
Training-only use is an explicit caller contract, not automatic role enforcement. The generic helper cannot prove
that descriptions carry the entire answer semantics: state/instruction references to literal option ids can
invalidate reassignment. Any integration must restrict eligibility to appropriate training records. No model was
executed, no GPU or paid resources were used, and nothing was pushed.

## Independent review follow-up: duplicate descriptions

Review of `de4dede5dda7292435cd958120b9ead76ec91545` found that duplicate non-None descriptions could still produce
identical model inputs with contradictory labels. Reversal now returns no view when gold would move and any
descriptions have identical canonical JSON renders. This also conservatively skips moved-gold duplicates when
the complete prompt changes. A fixed-center gold remains eligible because its label does not change.

The nine added cases cover equal strings, palindromic descriptions, structured descriptions equal after canonical
key sorting, a non-palindromic duplicate case, both ordered/symmetric layouts, rejection at attachment, and the
fixed-gold case. They reconstruct the unsafe prior transformation to prove identical token inputs and changed
labels. The same CPU-only command prefix with `tests/test_order_views.py --tb=short` gave **32 passed; 1.39 s**.
`git diff --check` passed. Source WIP hashes were verified unchanged again before this follow-up commit.

Remaining limitation: distinct renders can still express the same meaning, and literal option-id references can
invalidate reassignment. Training integration must establish appropriate record eligibility; this helper cannot
prove semantic equivalence from arbitrary descriptions. There remains no trainer callsite or model execution.
