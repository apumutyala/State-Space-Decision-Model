# Score-item invariance (separate from G1): FAIL

18 score items, 108 perturbed rows, base requests faithful 18/18; bitwise False, max|dp| 0.5049349011472439 (bound 0.0). Rendering: ev-score-level-as-content/1: content = 'level v of K: <rubric>' (v = 0..K-1); scale canonical; no handles in scale.

| op | n | invalid | flips | max abs dp | max abs dE[level] | mean abs dE | on-disk in-memory order | changed vs base | content = base |
|---|---|---|---|---|---|---|---|---|---|
| score_reverse | 18 | 0 | 2 | 0.3911928857503181 | 0.48504055182464634 | 0.11153445284530454 | 18/18 | 18/18 | 18/18 |
| score_shuffle | 54 | 0 | 8 | 0.3911928857503181 | 0.6942245267399799 | 0.12163515580028704 | 54/54 | 54/54 | 54/54 |
| score_relabel | 18 | 0 | 3 | 0.5049349011472439 | 0.7087641909685967 | 0.09544766621466498 | 18/18 | 18/18 | 18/18 |
| score_permute | 18 | 0 | 4 | 0.47316935252209646 | 0.4744006533667209 | 0.07996776893941487 | 18/18 | 18/18 | 18/18 |

- FAIL: score_reverse: 2 flips
- FAIL: score_reverse: max|dp| 0.3911928857503181 > 0.0
- FAIL: score_reverse: max|dE[level]| 0.48504055182464634 > 0.0*(K-1)
- FAIL: score_shuffle: 8 flips
- FAIL: score_shuffle: max|dp| 0.3911928857503181 > 0.0
- FAIL: score_shuffle: max|dE[level]| 0.6942245267399799 > 0.0*(K-1)
- FAIL: score_relabel: 3 flips
- FAIL: score_relabel: max|dp| 0.5049349011472439 > 0.0
- FAIL: score_relabel: max|dE[level]| 0.7087641909685967 > 0.0*(K-1)
- FAIL: score_permute: 4 flips
- FAIL: score_permute: max|dp| 0.47316935252209646 > 0.0
- FAIL: score_permute: max|dE[level]| 0.4744006533667209 > 0.0*(K-1)

Rule: SCORE-INV PASS iff every required op n>0, 0 invalid, 0 flips, max|dp| <= bound (0 = bitwise), max|dE| <= bound*(K-1), every on-disk row in the in-memory order, changed vs base, content-identical to base; separate from jevkit G1 (which never perturbs score items).
jevkit G1 (option_reverse / option_shuffle / key_relabel) never perturbs score items; this check does not change or replace the G1 verdict.
