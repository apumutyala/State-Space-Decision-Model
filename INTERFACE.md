# D → L interface (v0.1, draft — needs both agents' approval)

This is the seam between the decision core (Track D) and language growth (Track L). Track L consumes only what is listed here. There is no path from the hidden state to text that bypasses this interface.

## 1. Request (trusted vs. untrusted is explicit)

```
Request {
  schema_version: "0.1"
  questions: {qid: Question}           # SystemOne-compatible: noul | choice | score
  state: EvidenceItem[]                # untrusted unless provenance says otherwise
  policy: TrustedPolicy                # caller identity, grants, audience, revocation; never parsed from text
  versions: {model, tokenizer, schema, policy}   # a cached state is valid only under identical versions + permissions
}
EvidenceItem { id, text|record, source_class: trusted|untrusted|private|public, provenance, timestamp }
```

## 2. Decision output (per question)

```
Decision {
  qid
  predictive: {option_id: p}           # model distribution over ALL supplied options (benchmark output)
  policy_conditioned: {option_id: p} | null   # after trusted restriction, reported separately, never substituted
  choice, confidence                   # SystemOne wire semantics (choice: selected p; score: expected level index)
  abstain: {flag, reason} | null       # deployment mode only; never alters forced-choice benchmark output
  evidence_trace: [{evidence_id, span, option_id, contribution}]   # accounting-exact for additive heads; labelled "attribution", not "cause"
  semantic_records: SemanticRecord[]   # optional in v0.1; required once Track D's interface arms pass gates
}
SemanticRecord { entity, relation, value, negated, interval, provenance[], status: KEEP|SET|RETRACT|UNKNOWN|CONFLICT, epistemic: checked|extracted|predicted }
```

## 3. Guarantees each side may rely on

| Provided by D | Status |
|---|---|
| Outputs are valid against the schema; joint constraints where declared | structural |
| `predictive` sums to 1 over the supplied options; permutation-equivariant up to a declared float tolerance | tested |
| An `extracted` record is a model belief, never promoted to `checked` by D | contract |
| Calibration reported (NLL, Brier, debiased ECE) per question type | measured |

| Provided by L | Status |
|---|---|
| Realized text means exactly the checked program (within the defined grammar) | theorem target |
| `Quote` and `Hypothesize` never become assertions | contract |
| Only `checked` records, or derivations the checker validates, can be `Assert`ed | contract |

## 4. Change rules

Any field change bumps the version and needs both agents' approval (see `COORDINATION.md`). Track D can't change what a field means without Track L's sign-off, and Track L can't require a field that Track D can't produce without a plan to produce it.
