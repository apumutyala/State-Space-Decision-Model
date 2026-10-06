# CPU paraphrase curriculum: operation gains without useful-answer transfer

The curriculum raises average operation-slot accuracy by 11.33 percentage
points (paired task-cluster 95% CI 8.00 to 14.56), but average useful-answer
accuracy changes by only 0.67 points (CI -3.22 to 4.45). Every arm remains
0/100 on comparison tasks. This bounded wording intervention does not establish
a useful-answer transfer gain or justify scaling.

Dependency: PR #64 at `115c4535ce9db716d3dde70ad000d206fdd2a4b0`.
Source implementation: `fc33b84e52bd77f9008ee0df7db1ffbec6833da9`.
Pre-training frozen data and run source: `2b7863290542b06bcd1d29f92ffcb4650160d238`.
The new commits are distinct from PR #64. Nothing has been pushed and no PR
has been opened. All six fits finished before transfer evaluation; there was
no result-dependent tuning, retraining, or prompt change.

| Model / seed | Semantic /300 | Exact program /300 | Kernel accepted /300 | Read /100 | Explain /100 | Compare /100 | Teacher slots /800 | Free slots /800 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PR64 frozen / 20261006 | 63 | 37 | 196 | 32 | 31 | 0 | 321 | 225 |
| Old cohort / 17 | 77 | 46 | 221 | 40 | 37 | 0 | 334 | 248 |
| Curriculum / 17 | 66 | 56 | 225 | 32 | 34 | 0 | 389 | 309 |
| Old cohort / 29 | 58 | 40 | 230 | 32 | 26 | 0 | 325 | 232 |
| Curriculum / 29 | 61 | 39 | 237 | 36 | 25 | 0 | 345 | 240 |
| Old cohort / 43 | 44 | 29 | 218 | 30 | 14 | 0 | 295 | 171 |
| Curriculum / 43 | 58 | 56 | 250 | 37 | 21 | 0 | 371 | 258 |

Semantic scoring credits an explanation of the requested fact for a read
task and operand-reversed equivalent comparisons. Exact-program scoring requires
the canonical reference program. Kernel acceptance alone is not task usefulness.
Free-slot denominators retain all gold slots; when a wrong operation omits a
required slot, that slot fails. Teacher-forced slot diagnostics use gold prior
slots and are reported separately from free-running outcomes.

| Teacher slot | Old mean accuracy | Curriculum mean accuracy | Difference, percentage points | Task-cluster 95% CI |
| --- | ---: | ---: | ---: | --- |
| Operation | 40.56% | 51.89% | +11.33 | [8.00, 14.56] |
| Fact | 36.33% | 39.83% | +3.50 | [-3.33, 10.33] |
| Left operand | 41.67% | 44.00% | +2.33 | [-5.67, 10.33] |
| Right operand | 40.67% | 41.67% | +1.00 | [-7.33, 9.00] |
| Relation | 41.33% | 47.33% | +6.00 | [-1.67, 14.00] |

Across matched seeds, semantic accuracy is 19.89% old / 20.56% curriculum.
Exact-program accuracy improves 4.00 points [0.44, 7.56], kernel acceptance
4.78 [2.11, 7.67], teacher slots 6.29 [3.75, 8.83], and free slots 6.50
[4.00, 8.88]. These percentile CIs use 10,000 bootstrap draws, seed 8128,
resampling whole tasks within kind and retaining all slots in each task.
The mean comparison first averages the three seed outcomes within each task,
then resamples the same 300 clusters. It never treats 900 repeated-seed tasks
or 2,400 slots as independent. Per-seed paired CIs against both matched old
models and the separate PR64 frozen model are in `run/receipt.json`.

The 5,000-task training pool has 1,667 read, 1,667 explain and 1,666 compare
tasks, eight balanced templates per kind (208/209 examples each). It exactly
matches the old cohort's latent worlds, typed values, labels and choices:
13,332 paired slot records match after stripping only `task_input`, with
11,632 changed wording records. Equal initialization hashes and equal record
counts prove each old/curriculum seed pair uses matched initializations and
batch index draws. The architecture is unchanged: width 64, 70,081 parameters,
400 steps, batch 32, Adam 0.003, gradient clip 1.0.

The fresh transfer split has 300 tasks, 100 per kind, six templates per kind
(16/17 examples each), seed 141421, names Marigold/Nettle/Orchid/Plover/Quince/Rowan,
warranties 97–132 months and prices 45001–60000 cents. Phrases, names and
ranges are disjoint from training and both earlier evaluation domains.
The retired PR #67 source constants were read only for exclusion: zero retired
tasks were loaded and no exact template was reused. Shared vocabulary includes
training `contrast`, `retrieve`, `tell`, `trace`, `walk`, `evidence`, `reasoning`,
and transfer `evidence`, `recorded`, `whether`, `which`. This tests novel
templates with some shared vocabulary, not unseen language. The full token
intersection and retired-source hash are frozen in `data/manifest.json`.

The entire six-fit / seven-evaluation experiment took 207.391 seconds under
one 300-second watchdog, with minimum available RAM 6,007,746,560 bytes
(5.59 GiB), above the 3-GiB floor. CPU AMD Ryzen AI 9 HX 375, two intra-op
threads, one inter-op thread, float32 PyTorch 2.8.0; CUDA hidden, no GPU run,
Index data, official held-out data or spend. Slot records were sampled uniformly,
so comparison tasks contribute twice as many records as read/explain tasks.
Three seeds show substantial training variation; CIs condition on these fixed
seeds and do not estimate a distribution of training initializations.

Hashes of uncompressed canonical JSONL (sorted compact ASCII, LF):

- Curriculum tasks: `0f031413a953a2fe471c8a887a6da387b42df38ad6b5df65cccb6b13301603a1`
- Curriculum training records: `2375c6dc939b274c0f4d9b6cb93e5ca9546dd5554750d58c1ff92a636b42fe2a`
- Transfer tasks: `d62c1ec9d2a23177d671fdc5f5355fb8e1c08745852ed4a7708f7cdedd1eb6af`
- Transfer records: `357b9a063f8d7f783da3966bdc60ff031178838e6d3e00851fd6af30a363a5d7`
- Old training records: `eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c`
- Pre-training manifest: `b6d30f58d070b96739e2014a82f305aa48a11a17f6e2b8ab76681d583ce06cbe`
- Exact PR64 trained checkpoint: `0252d53cdebb6d1d0074542b375cc375badd3ee43c1ba06ce6712aa0f5eab0b4`
- Run receipt: `8c21220ed82d30363f8169f0f52e93bd8fc16a3df4aff85a83b0b41087d560a8`

`run/` retains all six new checkpoints, loss histories, every free-running
request/logit/output, teacher/free slot rows, task outcomes, template and slot
counts, model state hashes and file hashes. The PR64 checkpoint is preserved
under the inherited `receipts/2026-10-06-l-tiny-ssm-slots-run1/` directory.
`audit.json` recomputes raw program meaning, exactness, kernel status, counts,
paired intervals and latent-world equality; it passed for all seven models.
There are no free-running top-logit ties. `tests.txt` records 24 passing focused
tests. An independent reviewer still needs to recompute the evidence before a PR.

Commands, from this worktree (the freeze and run outputs must be fresh paths
to replay, because the CLI refuses to overwrite evidence):

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.curriculum_probe freeze --out receipts/2026-10-06-l-ssm-curriculum/data --retired-spec C:/Users/apumu/research/CHLabs/ssdm-codex-l0-frozen-wording-eval/language_growth/paraphrase_data.py
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.curriculum_probe run --data receipts/2026-10-06-l-ssm-curriculum/data --old-records C:/Users/apumu/research/CHLabs/ssdm-data/l0-training-cohort-20261006-seed271828-5k/training_records.jsonl --out receipts/2026-10-06-l-ssm-curriculum/run
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.curriculum_audit --root receipts/2026-10-06-l-ssm-curriculum --old-records C:/Users/apumu/research/CHLabs/ssdm-data/l0-training-cohort-20261006-seed271828-5k/training_records.jsonl
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m unittest language_growth.tests.test_curriculum_probe language_growth.tests.test_development_data language_growth.tests.test_training_records language_growth.tests.test_tiny_ssm_slots
```
