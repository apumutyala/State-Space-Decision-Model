# CPU preparation for native language learning

The following paths prepare actual model inputs and fitted intervention
artifacts. They do not qualify a language level. Real champion fitting waits
for Stage 1 and an A-ordered run on suitable resources.

## Supervised slots

```text
python -m language_growth.gate_training --worlds 32 --out FRESH_DIRECTORY
```

This replays only `development_train` references through the actual GatePlanner
and exports its exact offered domains and separate target IDs. Earlier target
choices are diagnostic teacher forcing; the current target and reference
program never enter its request. `group_id` binds all slots of a source-world.
The gzip export is accepted by `_cell_records(..., development=True)` and
`items_from_records` with the synthetic-only guard. The protected production
training path rejects it; this is explicitly development supervision.

L4 training contains public historical `ReadTechnology` examples only. Its
validation-only `LowestMedian` targets are excluded, and its single shared
source-world does not establish independent knowledge generalization.

## Native semantic directions

`gate_direction_fit.development_relation_pairs` constructs fresh empty-history
development pairs. `fit_native_directions` captures actual native fork hidden
features and raw logits, fits fixed paired mean directions on discovery-train,
and measures four arms on separate development-validation. Labels remain
outside provider inputs. `write_direction_fit` preserves source pairs, fitting
evidence, exact tensor identities and a loadable direction/registration pair.

Use `python -m language_growth.gate_direction_fit --help` for the qualified
parent pins and pair files. An explicit `--target-slots comparison-relation`
selects v2 registered applicability: non-targeted calls still run and record
unchanged native captures, while targeted calls require genuinely fitted
controls. At least one valid target capture in every arm is required across
the cohort. No task is removed from evaluation. Legacy v1 keeps its original
all-slot behavior and requires controls at every encountered candidate count.

The current relation helper covers K=3. Other semantic concepts/target slots
need genuine train-only paired requests and their fitted controls. Tiny
functional-head tests establish mechanics; the observed 0/1 effect-target
selection in every arm is not semantic improvement.

## Checkpoint header handoff

```text
python -m language_growth.checkpoint_metadata --checkpoint CHECKPOINT --sha256 KNOWN_SHA256 --out FRESH_JSON
```

This uses restricted mmap loading to export saved config/lineage and tensor
names/shapes/dtypes. It reconstructs no model, restores no optimizer, validates
no tensor values, and supplies no Stage 1 qualification. A/B can preserve this
small artifact alongside the full serving/calibration/runtime export. D1's
locally available `cell.json` alone is insufficient for strict admission.
