# Independent curriculum-result review

Independent raw-artifact recomputation completed by `/root/baseline_receipt_recompute` on 2026-10-06. Audited frozen HEAD `37adfc98c8f7702c3f23854e829c1261b02cdeee` and measured source `2b7863290542b06bcd1d29f92ffcb4650160d238`. No inference, fitting, edits, GPU, held-out/Index or posting during review.

All seven transfer arms match raw task/call reconstruction:

| Arm | Semantic /300 | Exact /300 | Kernel /300 | Compare /100 |
|---|---:|---:|---:|---:|
| PR64 frozen 20261006 | 63 | 37 | 196 | 0 |
| Old seed 17 | 77 | 46 | 221 | 0 |
| Curriculum seed 17 | 66 | 56 | 225 | 0 |
| Old seed 29 | 58 | 40 | 230 | 0 |
| Curriculum seed 29 | 61 | 39 | 237 | 0 |
| Old seed 43 | 44 | 29 | 218 | 0 |
| Curriculum seed 43 | 58 | 56 | 250 | 0 |

Every comparison task fails in every arm. Curriculum raises mean operation-slot accuracy by **11.333 points [8.000, 14.556]**, but meaning accuracy changes only **+0.667 points [−3.222, 4.447]**. Exact-program, kernel-accepted, teacher-slot and free-slot paired intervals reproduce as recorded. CIs resample task IDs stratified by task kind (10,000 draws, seed 8128), average the three fixed training seeds within task, and do not treat 900 repeated task/seed observations as independent.

The reviewer verified 13,332 old/curriculum records have identical labels, questions/choices and complete state after removing only `task_input`; 11,632 task-wording fields differ. All records enter both pools. Seeded initial model states and paired row-sampling sequences match. The PR67 source constants were read only for exact-template exclusion; no retired evaluation tasks were loaded. Exact template overlap is zero; shared vocabulary is disclosed. **51 indexed artifacts plus the index file match committed Git bytes.**

**Verdict:** result and receipt verified within this synthetic transfer cohort. This does not establish a useful-answer gain, comparison competence, an L0 gate, parent retention, causal semantic fidelity, or broad language generalization. Fold scaling of this 70,081-parameter SSM curriculum as an L0 candidate; retain it as a control.

Reproduce the compact independent audit from the worktree with:

```powershell
python -X utf8 -m language_growth.curriculum_audit --root receipts/2026-10-06-l-ssm-curriculum --old-records C:/Users/apumu/research/CHLabs/ssdm-data/l0-training-cohort-20261006-seed271828-5k/training_records.jsonl
```

Key hashes:

```text
artifact-index.json: 3a01ff94d2f398c4d84f40863bbcd201b7a75bdd6336fbf8d268b3330de94e84
run receipt: 8c21220ed82d30363f8169f0f52e93bd8fc16a3df4aff85a83b0b41087d560a8
transfer task JSONL: d62c1ec9d2a23177d671fdc5f5355fb8e1c08745852ed4a7708f7cdedd1eb6af
transfer record JSONL: 357b9a063f8d7f783da3966bdc60ff031178838e6d3e00851fd6af30a363a5d7
old training records: eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c
PR64 checkpoint: 0252d53cdebb6d1d0074542b375cc375badd3ee43c1ba06ce6712aa0f5eab0b4
```
