# Synthetic L1 fixture preparation receipt

2026-10-07. Branch `codex/l1-multirecord-data`, exact starting commit
`abf23a7354295a575f1b99cb1ad3ef77a0c2ac41` (`origin/agent-main` at creation).
Source commit: `a3f7f5984b66048c08a080bf89026554bcbe7b4d`. Exact generator/test
file hashes below identify the implementation independently of the commit
envelope. Subsequent receipt/phase-note edits do not change generator or tests.
Local Windows laptop, Python 3.12.13, standard-library CPU generation/checking,
billed cost $0. No provider calls, model execution, weights, GPU, Index or
sealed/held-out data. There are no predictions or performance measurements;
median/spread are not applicable.

Each split contains one world, seed 1707, 40 tasks, mixed choice format
(20 production/20 augmented). Both contain eight reads, eight explanations,
12 price comparisons and 12 warranty comparisons. Relations total four equal,
ten less-than and ten greater-than. Task/world/template IDs, label status,
catalog hashes and context identities are recorded in each committed manifest.
The full canonical JSONL is committed as `l1_development_train.jsonl` and
`l1_development_validation.jsonl` for direct inspection; generation originally
wrote the exact temporary paths below.

| Artifact | SHA-256 |
| --- | --- |
| `language_growth/multirecord_data.py` | `6fbd913c4b4ecb18d84d80112f423ae6d5a6daf502e85ac88a02580ab9ace089` |
| `language_growth/tests/test_multirecord_data.py` | `6f2e705b1cb5026ddcc372b559194644c4f73b672d669c67b4a126a6be402306` |
| `l1_development_train-manifest.json` | `c5d94f90e4c41daab59fedb6745ecac758bce48b2724acfa7ca13e1dc47e9e2d` |
| `l1_development_validation-manifest.json` | `6a6d3f8012c8b6089c6c8d40a2c6eb6ded60320ba7551d332fdd668bdd9d1cc1` |
| train `tasks.jsonl` | `2d455c7ad7e2a1116d026bb5801dec0c821de24602375ccecf66b2569011f007` |
| validation `tasks.jsonl` | `613d5b293cebbb883c46df6e8d0f361b61d81e65277d8ba8a3eeb8baacca141d` |

Commands run from `C:/Users/apumu/research/CHLabs/ssdm-codex-l1-multirecord-data`:

```text
python -m language_growth.multirecord_data --seed 1707 --world-count 1 --split l1_development_train --choice-format mixed --out C:/Users/apumu/AppData/Local/Temp/ssdm-l1-multirecord-data-20261007-train
python -m language_growth.multirecord_data --seed 1707 --world-count 1 --split l1_development_validation --choice-format mixed --out C:/Users/apumu/AppData/Local/Temp/ssdm-l1-multirecord-data-20261007-validation
python -m unittest language_growth.tests.test_multirecord_data language_growth.tests.test_development_data language_growth.tests.test_kernel language_growth.tests.test_kernel_adversarial language_growth.tests.test_decision_view -v
python -X utf8 -m pytest -q tests/test_jobs_lint.py tests/test_no_kill_by_name.py
git diff --check
```

The generation commands each succeeded; the focused data/kernel/bridge suite
passed 43 tests, and repository job/process-safety lint passed 70 tests. The
CLI requires a fresh output directory; use fresh paths on a repeat. Files are
written as sorted compact ASCII JSON with LF lines and final LF, using byte
writes to avoid Windows newline conversion. The manifest `sample_sha256`
hashes those exact JSONL bytes. Tests independently recompute catalog/row
hashes and relations using Decimal, and verify all candidate values survive.

See `plans/codex/L1_MULTIRECORD_FIXTURE_PREP.md` for the fixture-only compatibility
boundary: this schema cannot yet feed the merged training/export/evaluation
paths. Checker-valid authored labels are not independent task adjudication,
learned performance, causal fidelity or L1 proof. L0 gating remains in force.
The normal-task fixtures do not cover or satisfy the required adversarial
splicing, negation, stale-fact, unit-confusion, injected-authority or wrong-question
gates, and do not establish a strong comparator, parent retention or
causal-intervention fidelity. Independent review is recorded separately from
authorship; cross-track review remains required before integration.
