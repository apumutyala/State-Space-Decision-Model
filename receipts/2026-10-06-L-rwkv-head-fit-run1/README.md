# RWKV7 step-24 teacher-forced fit/transfer audit — run 1

This evaluation-only run used the final head checkpoint from the earlier 24-step pilot. It made zero optimizer updates and measured each preserved teacher-forced slot request once: 64 training rows and 32 disjoint development-validation rows. It diagnoses whether the frozen checkpoint fit the supplied slot mapping; it is not a capability gate, does not establish transfer by itself, and does not qualify an L0 parent.

## Result

The final head selected the exact reference choice on **22/64 training rows (34.38%)** and **10/32 validation rows (31.25%)**. Mean cross-entropy was 1.369076 on training and 1.142761 on validation. Both are well below the preregistered strong-fit threshold of 80%, so the mapping is not strongly fit. Since both splits are weak, this does not isolate a transfer-only gap; the short training schedule and representation remain unresolved.

| Task / slot | Training | Validation |
|---|---:|---:|
| All rows | 22/64 (34.38%), CE 1.369076 | 10/32 (31.25%), CE 1.142761 |
| Read | 9/16 (56.25%), CE 1.118721 | 5/8 (62.50%), CE 0.915570 |
| Explain | 4/16 (25.00%), CE 1.745153 | 1/8 (12.50%), CE 1.204639 |
| Compare | 9/32 (28.13%), CE 1.306215 | 4/16 (25.00%), CE 1.225417 |
| Operation | 11/24 (45.83%), CE 1.074493 | 6/12 (50.00%), CE 0.801149 |
| Fact | 4/16 (25.00%), CE 1.850759 | 2/8 (25.00%), CE 1.233372 |
| Left operand | 3/8 (37.50%), CE 1.416856 | 0/4, CE 1.952668 |
| Right operand | 1/8 (12.50%), CE 1.598899 | 2/4 (50.00%), CE 1.265768 |
| Relation | 3/8 (37.50%), CE 1.011854 | 0/4, CE 1.053462 |

There were exactly 96 unique requests, 96 responses and 96 retained outcomes; no row failed. The audit runner reported zero optimizer steps and cost $0. The prior free-running validation remains 2/12→1/12 useful answers and 12/12→10/12 kernel acceptance. Read operation choice was 8/8 on both splits, but fact binding was only 4/16 train and 2/8 validation; explanation operation choice was 1/8 train and 0/4 validation; comparison left operand and relation were 0/4 each on validation. This pattern points to poor learned semantic slot selection, not just a surface renderer issue.

The supported diagnosis is “mapping not strongly fit; representation versus optimization unresolved.” The 24-step run saw 48 of 64 teacher-forced training slots (75%, less than one full exposure pass), and step loss fell over the run, so a fixed longer-exposure test is a more informative next learning experiment than changing the architecture and data at once. This audit alone does not authorize or claim that follow-up.

## Reproduction and provenance

- Audit source commit: `aac12b461c7e8567ffdf0e6cb1e80dbcb7114677`; audit runner SHA-256 `dbcc1d1bf26dad35b394f84d3cc76e346e05ca849d70dffdbbeea8f8f76d1663`.
- Command, from `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-head-fit-audit`:

  ```powershell
  & 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -I -X utf8 language_growth/rwkv_head_fit_audit.py --execute --out 'C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-fit-run1-20261006'
  ```

- Input: `fla-hub/rwkv7-0.1B-g1` frozen parent plus the existing step-24 head checkpoint at `C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-learning-run1-20261006/checkpoint-after/ckpt-step0000024.pt`. Checkpoint SHA-256 `3a896a00dc06fdd79aa8656f5272fcd4307038633efb0e0d642773280affe13c`; prior source run and full receipt are in [the head-learning result](../2026-10-06-L-rwkv-head-learning-run1/README.md).
- Data: exact corpus and split hashes match [the prepared v2 receipt](../2026-10-06-L-rwkv-head-learning-prepared-v2/registration.json): canonical corpus `cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466`, train split `4dfd05d495cdcc5bf38c0db857ac31fbd6367fcf96c2df64d8791ad77978ceaf`, validation split `e8847f275c4001d53c8d2a4bb88863faec4b0de1c0645dd9605cba7559f7d4cb`.
- Execution: Windows 11 AMD64 local CPU, two threads, offline; 175.65 seconds, cost $0. There were no provider errors and no optimizer steps. The runner verified the prior parent/source identity and unchanged full-parent hash before/after; raw model tensors were not part of the saved audit artifacts.
- Independent remeasurement recomputed all task/slot accuracies and cross-entropies from raw choice scores plus frozen labels; no mismatches. The reviewer did not load tensors or rerun the model.

## Preserved artifacts

`raw.jsonl` preserves all choice-score requests, responses and outcomes. `rows.json` and `summary.json` contain the measured rows and aggregates; `receipt.json` and `registration.json` pin the run and its inputs. The corpus, parent evidence, prior receipt and train config are linked to the preceding prepared/run receipts instead of duplicated here.

| File | SHA-256 |
|---|---|
| `raw.jsonl` | `ed2e20bdbabfb6bd01f242dca88253b5c3732241c278266bc86af7c0bafe5ba5` |
| `rows.json` | `0feeac867d8028ebf3001f406b0ab70669e2423c1c5641ce86af9a1bf7129a98` |
| `summary.json` | `e29c19a9257b84291d48d145637f735bb7b48b2d0b43b1ea9c243c208b50c71a` |
| `receipt.json` | `7fee4cc382b63c030683de37eae2d5c5f114686ed3de1c2b1d9f036cb3141dfa` |
| `registration.json` | `d4f74759f35c097c3dfa9d4f53c27fd6d29a6f340887b891a6aeac04791fa950` |
