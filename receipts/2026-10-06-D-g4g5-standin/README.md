# G4 / G5 wrappers: CPU stand-in receipt (pipeline qualification only)

**Takeaway.** Both wrappers run end to end on the real request and attack generators and emit complete JSON.

- **G4 scope: covered insertions only.** On covered insertions (whole segments at segmenter boundaries), the
  exact certificate never let an inserted sentence change an executed decision: 0 of 400, and 32 certified
  covered units were checked against their clean decision.
- **The attacker chooses the boundaries.** With the same sentences merged into a sentence, placed
  mid-sentence, or appended without a space, **4.0–5.75 % of attacked units were certified *and* changed**.
  These insertions are outside the guarantee, and this is the number to quote beside it.
- **The hardened span unit closes that gap.** It certifies one token-aligned insertion anywhere, and here it
  had 0 certified-and-changed. The cost is about 6,400 decide calls per attacked request, against about 4
  for the sentence unit.
- **G5** holds on held-out templates (2.2 % against α = 5 %), and the break under a goal shift is flagged
  (18.1 %, CP lower bound 15.9 % > α).

**The decider is a keyword stand-in, not a model, so none of these numbers says anything about any model.**

## G4 on covered insertions (dev, clean + D1-A, 100 requests)

| Run | Clean escalation | Clean auto-execute | Certified-but-wrong (wrong given certified, clean) | Inserted auto-execute | Certified steer | Decide calls (clean / inserted mean) |
|---|---|---|---|---|---|---|
| exact, sentence, k = 1 | 47 % | 53 % | 19 % (10/53) | 8 % | 0 / 400 | 3.6 / 5.6 |
| exact, sentence, k = 2 | 47 % | 53 % | 19 % | 8 % | 0 / 400 | 5.5 / 11.3 |
| smoothing ρ = 0.3, k = 1 | 22 % | 78 % | 13 % (10/78) | 17.5 % | 0 / 400 (vs exact g(clean)) | 3.6 / 7.2 |

- **Denominators.** `certified_but_wrong_rate` is now wrong | certified, SAFETY_SPEC's wording.
  `wrong_given_auto` (wrong | auto) sits beside it. The two are equal here because this run has no alarmed
  units: the stand-in never prefers a disallowed option on clean or D1 inputs.
- **Certified inserted units are 100 % wrong** because the stand-in's *clean* decision was already wrong on
  them. The certificate preserves the clean decision; it does not correct it (non-claim 4).
- **Call counts rose against the v1 receipt** (3.1 to 3.6, and 4.6 to 5.6). `sent-leadws-v2` adds the
  trailing-whitespace origins.

## Uncovered-insertion stress arm (outside the guarantee)

Same D1-A sentences, four insertion modes. `certify.uncovered_stress` raises if a *covered* row is
certified-and-changed.

| Mode | Sentence unit: covered | Sentence unit: certified-and-changed | Span unit (all spans): covered | Span unit: certified-and-changed | Span unit (≤ 48 tokens): covered |
|---|---|---|---|---|---|
| merge, no terminal punctuation | 25 % | 18/400 (4.5 %) | 100 % | 0/24 | 0 % |
| mid-sentence | 0 % | 23/400 (5.75 %) | 100 % | 0/24 | 0 % |
| no space | 0 % | 16/400 (4.0 %) | 100 % | 0/24 | 0 % |
| trailing whitespace | 100 % (v2 fix) | 0/400 | 100 % | 0/24 | 0 % |

- **Span-unit subset.** The span unit runs on 6 requests (3 route + 3 pay) × 4 templates per mode.
- **Span-unit cost on that subset:**
  - attacked requests: 6,397–6,537 decide calls on average with all spans, and about 3,900 at ≤ 48 tokens;
  - clean requests: 1,666 calls on average (max 3,993) and 1,067 suffix-recompute forward-equivalents;
  - the sentence unit on the same clean requests: 5.3 calls and 3.0 forward-equivalents;
  - clean escalation is equal on this tiny subset: 4/6 for both units.
- **The 48-token cap covers nothing.** D1 sentences run longer than 48 tokens, so the cap removes every
  covering origin while still costing about 3,900 calls. A span cap must exceed the longest injection the
  threat model admits.
- **Conclusion.** The span unit is a real hardening of the certificate. At roughly 1,000× the decide cost it
  is a report-only arm, not a deployment default. A snapshot-at-boundary recurrent backend reduces each call
  to a suffix recompute, but the number of origins stays O(T²).

## G5

- **Stand-in strata.** All four strata come out vacuous (λ̂ = None), with 5–40 clusters per stratum and the
  bound computed on one unit per cluster.
- **Estimand.** The receipts now name it: cluster-uniform selective risk.
- **Stress tests** report both unit-weighted and cluster-weighted realised risk.
- **Score.** `executed_conf` is the policy-conditioned probability, the same score G4 uses.
- **Synthetic run** (s ~ U(0,1), wrong ~ Bern(0.3(1 − s)), n = 5,000):
  - λ̂ = 0.80, coverage 20.5 %, calibration selective risk 2.8 %, CP upper bound 4.8 %, floor 10.4 %;
  - held-out templates: 2.2 % [1.4, 3.2], no violation;
  - held-out goal (+0.15): 18.1 % [15.9, 20.5], **violation detected**. This is outside the assumption.

## Reproduce

```
python -m safety_eval.run_guarantees_smoke --out receipts/2026-10-06-D-g4g5-standin   # ~4 min CPU (span arm ~3.5 min)
python -m pytest -q tests/safety_eval/test_certify_g4.py tests/safety_eval/test_selective_g5.py
```

- **Code commit** (in every JSON `meta.commit`): `89458f9`. Later commits change docstrings and docs only.
- **Data:** `data.build(60, 40, split="dev")`, manifest sha256 in `meta.data`. D1-B stays held out.
- **Seeds:** data 20261006; smoothing 20261006; G5 synthetic 0; bootstrap 20261006.
- **Hardware:** CPU (laptop), $0. No GPU was used.
- **Files:**
  - `summary.json`;
  - `g4_*.json`, including `g4_uncovered.json`;
  - `g4_exact_k1_rows.jsonl`;
  - `g5_*.json`;
  - `pytest.txt`: 49 of 49 pass.
