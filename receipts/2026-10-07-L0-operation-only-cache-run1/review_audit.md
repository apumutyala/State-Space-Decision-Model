# L0 operation-only cached q_end diagnostic — run 1

**Verdict: fold this line; no L0 advancement.** This was a $0 CPU screening diagnostic on the pinned RWKV7-0.1B parent, not a capability or qualification result. The q_end head memorized the 24 training rows (24/24 operation choices; CE 0.03440) but did not improve useful outputs on the 12 reused development worlds. Trained useful-answer accuracy was 1/12 versus 1/12 stock and 2/12 at step 0. Its paired useful-answer difference was 0.00 versus stock (95% bootstrap CI [0, 0]) and -1/12 versus step 0 (95% CI [-0.25, 0]). Kernel acceptance fell to 11/12 from 12/12 in both controls. Development operation CE was 2.69680 after training versus 1.48877 at step 0. The preregistered advancement condition is not met.

## Frozen recipe and execution

- Source registration base: `018585548104dfbcb9d65a22214ec51405529711`; #112 integration is merged at `64ece2e6f80bff8a80f8086b8d860ed404f30e23`. The registration pins 101 source file hashes, including runner SHA-256 `e133c73d12585aeab9764d630049759ef6255d34dcebff4a22d77a807a424d08`. The run receipt does not preserve the exact runtime checkout SHA; source-file hashes and registration are the available identity evidence.
- Parent: `fla-hub/rwkv7-0.1B-g1`, snapshot `7450ee04eb9c66baa7161b7a5c0fe09e5b2817d8`; in-memory parent state digest `5fbc0ff4572bf5b76fecc4efa262e96c93f6162a34eabbe264bae4db36c13dc8`; parent weight-file SHA-256 `051b3684ffa896aa842522e5bbc2f355ac00783e12e373a8eb7e4907dd5ead70`.
- Training/dev: 24 synthetic training operation rows and 12 reused-development worlds, frozen corpus SHA-256 `cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466`; task targets SHA-256 `0d8e9bdfae7242363a9963978a52fef4396eddadbf1b3d97f692b4633ecd2bbf`.
- Three paired arms: stock free-running planner, exact step-0 q_end head, fixed step-96 q_end head. Training used CE only, seed 20261006, AdamW lr 0.001, batch 2, 96 updates/eight passes; no tuning or selection. Paired bootstrap: seed 9203, 2,000 draws over the 12 independent development worlds.
- Command: `C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_operation_cache --out C:/Users/apumu/research/CHLabs/ssdm-data/l0-operation-only-cache-run1-20261007 --execute`.
- Runtime: Windows AMD64 CPU, `CUDA_VISIBLE_DEVICES=-1`, two threads, 493.813 seconds in the receipt (about 495.235 seconds wall-clock including process exit), $0. No GPU, Index, held-out set, or paid compute.

## Independent raw remeasurement

The reviewer parsed all 36 world/arm responses from `raw.jsonl`, independently recomputed aggregate and per-kind paired bootstrap intervals, and matched the saved arrays. Useful answers: stock 1/12, step 0 2/12, trained 1/12. Operation selection: 4/12, 4/12, 5/12. The trained head corrected one comparison's operands, then selected a false relation which the checked kernel rejected. Read useful/accepted: 1/4 and 4/4 for all arms; explain useful: 0/4, 1/4, 0/4; compare useful: 0/4 for all, with trained acceptance 3/4 versus 4/4 controls. The 12-world sample is too small and reused development data cannot support generalization claims.

The stock-parent direct path had 0/4 correct before and after, and saved direct arrays are exactly equal. The raw log does not separately capture direct calls, and the runner did not persist independent before/after backbone digests. The independent reviewer therefore verified saved arrays and the frozen-backbone run path, not raw direct-response equality or a separately hashed backbone diff.

Process note: the coordinator saw one stock response in the raw file while the process was still running. This was disclosed; no code, configuration, data, selection, or rerun changed in response. The independent reviewer re-derived final outcomes from the complete logs.

## Artifact hashes

- `receipt.json.gz` uncompressed SHA-256: `483aa33d222ae00700b45f58bf8187a9b6b3bc9587259d92020ba623ea1a7a38`
- Raw JSONL: `1d81eda6270be603e621be18dca7bbd2ee36415ac9b5d7c93a4c4ff30293e7fe`
- `registration.json.gz` uncompressed SHA-256: `8f72a091516c268c3bc915a926c2d6da90afca02bd34804fd2be67f1f216f997`
- `operation-records.json.gz` uncompressed SHA-256: `9b32ee60ff3bd89d9de2c3995b17633da5badf463c00f22507a8195d1288af43`
- Feature cache: `b56911e6d1d073d4336c94fcd9a42697ee19d2c5b600793288bbab6f9dfbfc54`
- Initial head: `4d4184073a2426962f4471719cba889fe2d150855ad3a0e87838a37be8dcd4ed`
- Trained head: `0ac583634c93e82fce3698f098fcba99cdc059b94439f8d9be0af08be2e9f60a`

The registration's runner and all 100 registered Python source hashes match the merged `64ece2e` tree. Its prepared-plan hash differs from the merged plan blob only by CRLF versus LF line endings; the plan content is otherwise identical. The JSON run artifacts are gzip-compressed without changing their uncompressed bytes; `artifact_hashes.json` records compressed and uncompressed hashes, and `tests/test_operation_cache_receipt.py` checks both after checkout.

This receipt folds only the tested CE-only operation-head line. It does not falsify the semantic program interface, other head losses, other parents, or language growth more broadly.
