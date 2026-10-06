# Comparison paraphrase condition: negative CPU result

The single preregistered comparison-wording condition worsened exact requested
target accuracy on the fresh development validation set. Keep the original
training recipe; this condition does not justify planner integration.

| Fixed final-step exact target | Original-wording baseline | Comparison paraphrases |
| --- | ---: | ---: |
| Train overall | 563/600 | 576/600 |
| Train comparison | 164/200 | 179/200 |
| Fresh validation overall | 126/180 (70.0%) | 114/180 (63.3%) |
| Validation read | 37/60 | 38/60 |
| Validation explain | 37/60 | 32/60 |
| Validation compare | 52/60 (86.7%) | 44/60 (73.3%) |
| Validation operation alone | 179/180 | 165/180 |
| Validation field alone | 180/180 | 180/180 |

Random initialization scored 16/180, including 0/60 comparisons. The treatment
improved 17 cases and regressed 29 relative to the trained baseline; comparisons
improved 5 and regressed 13. The original-wording baseline predicted comparison
on 59/60 comparison rows, versus 45/60 for the treatment; the other predictions
were reads. Comparison left-entity binding scored 52/60 versus 54/60, so the
treatment's primary comparison regression was operation selection. All 180
semantic predictions in each arm survived option-ID relabeling. IDs are excluded
from the model by design; this is not learned invariance.

The prior source probe scored 0/60 comparisons on its different validation
wording, `For October 2026, compare ...`. This experiment preregistered a fresh
seed and `At October 2026, how does ... compare with ...?` before training. The
baseline's 52/60 therefore shows strong wording sensitivity. It does not repair
or retest that prior failure. The comparison treatment increased training fit
while reducing fresh-validation accuracy. One seed and one new template are
insufficient to identify all generalization failures or establish a reliable
curriculum benefit. No validation-driven change or second condition was run.

Both models use the exact same width-32 byte diagonal SSM (27,937 parameters),
random seed 20261006, Adam 0.003, batch 32, clip norm 1, 300 steps and two CPU
threads. Training took 12.375 and 13.235 seconds, below the fixed 120-second
per-arm bound. Initialization and sampled row indices match. Only the
comparison question text differs: 200 train worlds evenly receive the four
preregistered templates. Labels and candidates match, read/explain rows match,
and both arms score the same validation rows. Models are reloaded from saved
final checkpoints before scoring. Fourteen lightweight tests passed before run.

The train/validation world counts are 600/180. Their name pools and numeric
ranges are disjoint. The fresh validation seed is 20261008. Numeric values never
enter the model; temporal scope is constant. Gold targets are authored synthetic
programs used only for offline dataset construction and supervision. The model
receives question and candidate descriptions, with no gold labels, program,
request ID or external teacher features. The treatment uses words with the
same meaning but does not add classes or examples. This is iterative synthetic
development data, not an untouched audit slice. No Index, external teacher,
download, GPU, network or paid compute was used. No numerical answering,
unsupported/negated questions or whole-program correctness was tested.

Preregistration and exact runnable source were committed at
`e5a1ea61a7979927f998ab5bcbf0e87e1c377cd3` before the run. This worktree branches
from `origin/agent-main` at `4733f785e5284ea959d4d640fff884136487bd05`.
The copied model/target recipe provenance is recorded in the receipt; no
unmerged runtime/import/Git dependency exists. Committed receipt includes source,
row and checkpoint hashes, metrics and loss histories. The committed raw paired
predictions are in `observations.json`. Both checkpoints and originals remain at
`C:/Users/apumu/research/CHLabs/ssdm-data/l0-compare-curriculum-20261006-run1/`.

Reproduce once in a fresh output directory using the existing environment:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.compare_curriculum_probe --out C:/Users/apumu/research/CHLabs/ssdm-data/l0-compare-curriculum-reproduction
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m unittest language_growth.tests.test_compare_curriculum_probe language_growth.tests.test_development_data -v
```
