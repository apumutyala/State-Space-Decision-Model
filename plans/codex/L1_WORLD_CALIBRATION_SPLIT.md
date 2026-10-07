# L1 whole-world fit/calibration preparation

2026-10-07. Preregistered before generating the partition: fixed split seed
1708; rank all 32 train world IDs by the ascending SHA-256 hex digest of
compact ASCII JSON `["ssdm.l1_world_split.v1",1708,world_id]`, breaking any
digest tie by world ID. Assign the first eight worlds to calibration and the
remaining 24 to fit. The method takes only world IDs and the fixed seed; it
does not inspect labels, evidence values, outcomes or task kinds. No seed
search, label balancing or caller-selected seed/count is supported.

Base: exact PR #133 head `e94c79793082fc8643bae34ac52813760f1950e8`.
Load only the pinned `l1_multiworld_development_train` source and manifest
through the existing checked export. Require the exact canonical 4,096-row
export SHA-256 `a7806d12672be658f66f75d57a0d53a888cb00b93074b183a0a7c5bceb52d489`
before assignment. The old one-world smoke and all validation inputs fail
closed. The separate 32-world `l1_multiworld_development_validation` cohort
is evaluation-only and cannot be a source for fitting or calibration.

Track D's generic Item conversion drops world metadata. Perform this split
before that conversion and preserve its two files as separate partitions;
recombining them and applying a random task/slot split defeats the contract.
This addition does not change the generic trainer or add training behavior.

Each output preserves every original exported row and metadata field, with
relative source order unchanged. Each world remains 40 complete tasks and
128 slots. Expected totals: fit 24 worlds / 960 tasks / 3,072 rows;
calibration 8 worlds / 320 tasks / 1,024 rows. Tests will check whole-world
disjointness, exact row union and unique slot identities, per-world slot
coverage, identical task-kind/field/relation distributions, deterministic
label-independent assignment and fail-closed input handling. Publish both
JSONLs plus their hash/count/provenance manifest by staging and one directory
rename; a failed write must leave no partial final bundle or staging files.

Only eight synthetic calibration worlds are available. This preparation
does not fit a calibration parameter, estimate calibration risk or establish
calibrated performance. It is a narrow six-name, fixed-template development
distribution with authored teacher-forced prefixes. Learned L0, strong
comparator, decision retention, adversarial and causal gates remain required.

Scope is local no-model CPU data preparation, cost $0. No weights, optimizer,
GPU, paid service, Index or heldout access. Preserve existing sources and
receipts. Internal adversarial and coordinator exact-head reviews precede
any push/PR, followed by Track D review stacked on #133.
