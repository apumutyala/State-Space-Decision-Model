# Score-item invariance (separate from G1): PASS

18 score items, 108 perturbed rows, base requests faithful 18/18; bitwise True, max|dp| 0.0 (bound 0.0). Rendering: ev-score-level-as-content/1: content = 'level v of K: <rubric>' (v = 0..K-1); scale canonical; no handles in scale.

| op | n | invalid | flips | max abs dp | max abs dE[level] | mean abs dE | on-disk in-memory order | changed vs base | content = base |
|---|---|---|---|---|---|---|---|---|---|
| score_reverse | 18 | 0 | 0 | 0.0 | 0.0 | 0.0 | 18/18 | 18/18 | 18/18 |
| score_shuffle | 54 | 0 | 0 | 0.0 | 0.0 | 0.0 | 54/54 | 54/54 | 54/54 |
| score_relabel | 18 | 0 | 0 | 0.0 | 0.0 | 0.0 | 18/18 | 18/18 | 18/18 |
| score_permute | 18 | 0 | 0 | 0.0 | 0.0 | 0.0 | 18/18 | 18/18 | 18/18 |


Rule: SCORE-INV PASS iff every required op n>0, 0 invalid, 0 flips, max|dp| <= bound (0 = bitwise), max|dE| <= bound*(K-1), every on-disk row in the in-memory order, changed vs base, content-identical to base; separate from jevkit G1 (which never perturbs score items).
jevkit G1 (option_reverse / option_shuffle / key_relabel) never perturbs score items; this check does not change or replace the G1 verdict.
