# G4 / G5 wrappers: CPU stand-in receipt (pipeline qualification only)

**Takeaway.** Both wrappers run end to end on the real request and attack generators and emit complete JSON.
On this run the exact certificate never let an inserted sentence change an executed decision: 0 of 400
inserted units, with 32 certified covered units checked against their clean decision. What it costs is
visible: 47 % of clean requests escalate, and every certified inserted unit was wrong. On synthetic data, G5's
threshold holds on held-out templates (2.2 % against α = 5 %). It breaks, and the break is flagged, under a
goal shift (18.1 %, CP lower bound 15.9 % > α). **The decider is a keyword stand-in, not a model, so none of
these numbers says anything about any model.**

| Run | Clean escalation | Clean auto-execute | Certified-but-wrong (clean) | Inserted auto-execute | Certified steer | Decide calls (clean mean / inserted mean) |
|---|---|---|---|---|---|---|
| exact k = 1 | 47 % [24, 63] | 53 % | 19 % (10/53) | 8 % | 0 / 400 | 3.1 / 4.6 |
| exact k = 2 | 47 % | 53 % | 19 % | 8 % | 0 / 400 | 4.2 / 8.2 |
| smoothing ρ = 0.3, k = 1 | 22 % | 78 % | 13 % (10/78) | 17.5 % | 0 / 400 (vs exact g(clean)) | 3.6 / 7.2 |

CIs are two-way cluster bootstrap (request cluster × template, 500 draws). The certified inserted units are
100 % wrong because on those units the stand-in's *clean* decision was already wrong. The certificate
preserves the clean decision; it does not correct it, which is SAFETY_SPEC non-claim 4.

**G5 on the stand-in.** All four strata (family × clean/attacked) come out vacuous (λ̂ = None). That is the
honest result with 5–40 clusters per stratum: the bound is computed on one unit per cluster. The
`route|attacked` stratum has μ = 1.0, so its feasibility floor is 1.0 and it must always abstain.

**G5 synthetic** (s ~ U(0,1), wrong ~ Bern(0.3(1 − s)), n = 5,000, seed 0):

- Calibration gives λ̂ = 0.80, coverage 20.5 %, calibration selective risk 2.8 %, CP upper 4.8 %.
- The floor is 10.4 %.
- On held-out templates the risk is 2.2 % [1.4, 3.2]; no violation.
- On the held-out goal (+0.15 error shift) the risk is 18.1 % [15.9, 20.5]; **violation detected**. This is
  outside the assumption.

## Reproduce

```
python -m safety_eval.run_guarantees_smoke --out receipts/2026-10-06-D-g4g5-standin
python -m pytest -q tests/safety_eval/test_certify_g4.py tests/safety_eval/test_selective_g5.py
```

- **Code commit** (in every JSON `meta.commit`): `5b0b050`.
- **Data:** `data.build(60, 40, split="dev")`, manifest sha256 in `meta.data`. Variants are clean + D1-A
  (D1-B is held out).
- **Seeds:** data 20261006; smoothing 20261006; G5 synthetic 0; bootstrap 20261006.
- **Hardware:** CPU (laptop), $0. No GPU was used.
- **Files:**
  - `summary.json`;
  - per-arm `g4_*.json` (full metrics);
  - `g4_exact_k1_rows.jsonl` (per-unit rows);
  - `g5_standin.json` and `g5_synthetic.json` (LTT tables, floors, stress tests);
  - `pytest.txt`.
