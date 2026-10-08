# Finite L3 coordination primitive: CPU boundary receipt

Tested implementation commit: `f950a657d05b024bb6ceff840e00694a0051b2bd`,
branched from `agent-main@5fec2b2`. Python 3.12.13 on the local Windows CPU;
no GPU, network provider, model weights, held-out data or paid work.

Exact command:

```text
python -m unittest language_growth.tests.test_licensed_realizer language_growth.tests.test_realize language_growth.tests.test_public_output language_growth.tests.test_public_conjoin -v
```

The committed-source run passed 31 tests, including the 7 new primitive checks.
Raw validation output is `tests.txt`; data/inline synthetic-world source hashes
are in `input-sha256.json` (working-file SHA-256). This is deterministic contract
validation, not a model measurement: seeds, repeated-run median/spread and model
predictions are not applicable. Billed cost: $0.

The new production renders exactly one checked two-Fact Conjoin as a sentence
sharing its identical scope. Tests assert exact literals and clause order,
unchanged source/dependency bindings, release refusal after source, version,
scope, audience or policy changes, forged-result refusal, unsupported-shape
refusal without fallback, quoted clause-like literals, ignored planner prose,
missing/wrong requested fact refusal and unlicensed connective refusal before
any scripted call. Existing renderer/public-boundary tests validate compatibility.

Claim scope: a deterministic licensed-discourse L3 primitive only. No learned
fluent capability, useful-answer comparator win, parent decision retention,
neural semantic intervention, language-rung qualification or general natural-
language soundness result is claimed. Independent nonauthor review remains
required before merge.
