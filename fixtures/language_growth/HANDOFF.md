# Fixture handoff — 2026-10-06

Completed by the original fixture subagent; ready for the coordinator's Sol-high successor. All edits are confined to `language_growth/fixtures/`. No commit, model/GPU work, paid run or external message was performed.

- `manifest.json`: 20 explicit cases, stable `LG01`–`LG20` case IDs, `LGW001`–`LGW004` evidence-world IDs, decision-question IDs and scope disclaimers.
- `catalog.json`: nine small trusted-in-fixture records, immutable snapshots, policies, exact integer/decimal values, field predicates and explicit belief/conflict/private cases.
- `cases/`: proposals aligned to `ssdm.semantic_program.v0.1`, explicit Compare object projections, expected provider/kernel/task/coverage outcomes, and same-question decision-only choices with evaluator-only targets.
- `README.md`: interpretation and use; `validate.py`: standard-library packaging/reference validation only.

Validation completed successfully with:

```text
python language_growth/fixtures/validate.py
PASS: 20 fixture files, 9 records; JSON shape and declared references only.
```

No checker, planner, renderer or adapter is implemented here. No expected outcome has been demonstrated by kernel execution or model inference. Labels were written during this design pass and are not independent audit labels.

Agreed conventions: ReadFact returns a subject-bound Fact; Assert.proposition references its node; Compare uses `{node, project: "object"}`; Conjoin uses `speech_acts`; Ask uses host-registered `category`, `obligation_ref`, and `references`; revoked policy uses `stale_policy`.

Remaining implementation/review points:

- Catalog storage is a local fixture format; normalize it at the L0 loader boundary without granting model-authored metadata authority.
- LG04 tests literal relation splicing at the invalid Assert-input boundary. Add a semantic binding counterexample if a richer binding operation is introduced.
- LG10 explicitly chooses equal temporal scopes as the pilot registry's compatibility rule; a future cross-period operation needs its own contract.
- LG13's injection/renderer behavior is an expected test, not a result of the packaging validator.
- LG20 combines an unknown operation and arbitrary text; split those checks during L0 implementation. Exact precedence for multiply-invalid proposals is not frozen.
- LG15 must fail the provider precondition before inference/checking; its direct decision target applies only after fresh public recomputation.
- LG18 preserves first-attempt failure and charges one repair. LG19 blocks a second call. Counts are illustrative full-proposal calls, not model/slot measurements; compare repair arms separately from one-shot arms.
- Every case includes decision-only choices for the identical host task. Keep the target and all program/expected data out of model inputs. Cases sharing a world must stay together if a split is later constructed.

The coordinator said the next work belongs in an isolated Codex worktree, with actual L0 code under `language_growth/` and final fixtures at root `fixtures/`. These drafts have intentionally not been moved. Update README/manifest relative contract links and validator paths when relocating. Preserve these design expectations while implementing and running the boundary, then report actual outcomes separately.
