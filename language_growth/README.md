# L0 semantic boundary

This CPU-only package checks a compact typed semantic program against host-supplied fixture evidence. It is not a trained language model. The current kernel supports ReadFact, exact Compare, Assert, Conjoin, ExplainDerivation and Ask, with bounded rejection reasons and release-time policy/evidence revalidation.

Run kernel tests with:

```text
python -m unittest discover -s language_growth/tests -p test_kernel*.py -v
```

The fixture pack is under `fixtures/language_growth/`. Source and data revisions are recorded in the relevant `receipts/` directory. Renderer, oracle/replay backends and actual case evaluation are separate Track L changes; they must not be assumed available from the kernel-only commit.

The kernel checks every node, including unused ones; keeps subject/field/unit/time bindings; rejects model-authored inline facts; and validates exact Decimal/integer values. It rejects private-contaminated public snapshots and nonempty recurrent-state handles. A checked result is an immutable process-local object minted by this kernel. `validate_release` rechecks current policy, evidence and record versions. This object is not a cryptographic certificate and serialized JSON does not acquire authority by containing an accepted flag.

The fixture host, ontology and coarse temporal scopes are trusted inputs. General interval overlap, explicit supersession, arbitrary inference, natural-language premise truth and task relevance are outside the implementation. Acceptance of a true but irrelevant statement is deliberately distinct from task correctness. Exact errors are internal diagnostics; public failure rendering must not disclose hidden reference existence or source values.

The kernel-only tests are authored boundary tests, not independent audit labels or evidence of alignment, model capability or causal interpretation. No model weights, GPU or paid service is used.

The iterative development generator can opt into clarification coverage with `generate_development_tasks(seed, count, clarification_mode=True)`. Missing-information tasks require Ask for the specific host obligation and reference. Answerable tasks carry the same number of unrelated unmet obligations, so Ask is offered but is task-wrong. The free-running scorer compares checked Ask meaning and still scores read, explain, and compare tasks as before. This mode is synthetic development data only; the default generator output and manifest remain unchanged.
