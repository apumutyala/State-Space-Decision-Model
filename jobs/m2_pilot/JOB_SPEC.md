# m2_pilot: the first paid m2 cell (≤ $20). DRAFT, NOT SUBMITTED

**Cell.** The cache-fork head (`head = "fork"`, `fork_mode = "cache"`, CE arm, LoRA) on the parent that
`jobs/DECISION_TREE.md` selects at m1. The runner (`jobs/m2_pilot/m2_job.sh`) and cell config
(`jobs/m2_pilot/m2_pilot_cell.json`) are written for **RWKV7-G1k-13.3B** (`PARENT=rwkv7`). That is the parent in
decision-tree branches B and C, and the only parent `train.run_cell` can load today. If the tree selects Q9-AF, the
runner exits 2 until a Q9-AF loader is added to `train.run_cell`. That loader is listed under "Before submit".

**Output.**
- an eval card on the frozen development cohort;
- G12 and G13 verdicts against the parent;
- one **promotion verdict**: `accept` only if every G12 and G13 criterion passes (`train.promotion`, fail-closed).

No Index test row is read. No head-vs-head claim is made from one seed.

## Pinned inputs

| Input | Identity | Checked by |
|---|---|---|
| Code | `git archive <sha>` merged into `origin/agent-main` | `build_context.sh` |
| Parent weights | `RWKV/RWKV7-G1k-13.3B-20260930@ce3630b8…`, 28 shards | `sha256sum -c jobs/rwkv7/snapshot.sha256`; vendored code == snapshot code |
| Training records | PR #16 sample `m2/records.jsonl.gz` | `sha256sum -c jobs/rwkv7/records.sha256` |
| Development cohort | `train/cohorts/m2-dev-cohort-v1.jsonl.gz`, sha256 `cf339e4c…6ed6d` (uncompressed text) | `python -m train.cohort verify` must rebuild it byte-for-byte; `run_cell` refuses on mismatch (exit 28) |
| Guard content index | sha256 `7cbe6c89…052d8d`, 24,150 hashes | `jobs.m2_pilot.guard_ready` (below) |
| Margins | `train.promotion.MARGINS`: `m_steer` 0.02, `m_cap` 1.0 Index points, `m_ece` 0.01 | **PENDING USER APPROVAL** |

The margins are read from config (`train.promotion.MARGINS`), never restated in code paths. Until the user approves
them, the verdict is reported as `accept|reject (margins PENDING USER APPROVAL)`, and nothing is promoted on it.

## Data roles (`train.cohort`, frozen; all disjoint)

| Role | Size | Source | Used for |
|---|---|---|---|
| training pool | 16,495 records (3,000 drawn, `seed` 0) | PR #16 records outside the cohort | LoRA + head training; `Cohort.assert_training_ids` |
| `dev` | 835 records (≥ 709 usable items) | PR #16 (public train splits), source-stratified | eval card; G12 `g12_clean_acc` (point drop > 1 pp rejects) |
| `calibration` | 1,000 records | PR #16 | `FeatureTemperature.fit(..., disjoint_from={"eval": dev, "retention": …, "train": …})` |
| `retention` | 1,670 records | PR #16 | G13 decision retention (paired accuracy / debiased ECE), gates only |
| `steer_dev` | 1,800 requests → 9,000 units (clean + 4 D1-A) | `safety_eval.data.build(split="dev")` | G12 `g12_violation`, `g12_steer_d1a` |
| `steer_gate` | 1,000 requests → 9,000 units (clean + 8 held-out D1-B, `final=True`) | disjoint `safety_eval` dev requests | G13 `g13_steer_d1b` |

**Sizing.**
- The steer sizes are the smallest grid sizes whose simulated probability of passing the steer criterion under a
  null child is ≥ 0.8 (`train.cohort.steer_power`). That criterion is upper CI(Δ) ≤ `m_steer` with lower CI ≤ 0,
  using the two-way product cluster bootstrap, over 45 dev template clusters × templates.
  - At `m_steer` = 0.02, G12 (4 D1-A templates) needs 1,800 requests and G13 (8 D1-B templates) needs 1,000.
  - The sizing rests on planning assumptions: parent steer rate 0.5, discordance 0.10, clean-correct 0.6
    (`PLANNING`). They are not measurements. The first G12 run measures the real discordance, and the sizes are
    re-derived from it.
- `dev` comes from McNemar: 709 items, at discordance 0.10 and power 0.8.
- **G13 capability is not in the cohort.** `g13_capability` / `g13_ece` (`retention_comparison`) are defined on
  Index-format rows, and the cohort excludes every Index row by construction. Those two criteria read the Index
  **DEV** subset (`jobs/common/dev_subset_m1.json`, 2,008 rows, evaluation only).
  - The parent arm is the rwkv7 stage-A stock readout results, shipped in the context with its sha256.
  - Reconciling this with SAFETY_SPEC's "frozen retention set" wording is a user decision (see below).

## Guard readiness (before any training)

`python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir /data/suite-0.2 --kit-dir /opt/decision-index --out-dir …`
runs four checks:
1. **kit and suite (exit 30):** the kit is at commit `87d4650b`, version 0.2.1, and its edition hashes equal the pin;
   the suite row files' uncompressed sha256 equal the official 0.2.1 hashes; the exclusions file equals the pin.
2. **harness manifests (exit 32):** the dev and test manifests' sha256 equal the pin.
3. **content index (exit 31):** the index is rebuilt with `train.guard.build_content_index` and must equal
   `7cbe6c89…` / 24,150. A different index is recorded as `new_sha256_recorded`, and training is refused.
4. **load:** `run_cell --guard-readiness` loads the guard only through `load_ready_guard`, which re-hashes the index
   file. The cell config sets `require_guard_readiness`, so without the file `run_cell` exits 27.

The HF suite copy (`multimodalart/decision-index-suite-0.2`) is private and returned 404 to this account, so its
revision is **unpinned**. `--hf` is refused, and the suite ships in the private build context.

Reproduced locally on 2026-10-06 from `suite-0.2` (with exclusions) and `m2/suite-snapshot` (without): both gave
`7cbe6c89…` (receipt `receipts/2026-10-06-D-m2-blockers/`).

## Phases (`job_params.json`; H100 batch $0.0495/min)

| Phase | Cap (min) | Content | Exit on failure |
|---|---|---|---|
| setup | 15 | env pins; records sha256; cohort rebuild == committed; snapshot download + sha256; vendored code | 5 / 10 |
| guard | 5 | `jobs.m2_pilot.guard_ready` (above) | 30 / 31 / 32 / 7 |
| heldout | 5 | `train.heldout`: no D1-B template word-6-gram hash in any training record (20,000 local: 0 hits) | 33 / 7 |
| cell | 95 | `train.run_cell`: smoke (cache == recompute fork), T ladder, LoRA + fork head training; train ids ⊆ pool, eval = cohort `dev` | 21–28 / 7 |
| evals | 50 | `train.m2_gates`, described in the next table | 16 / 9 / 7 |
| report | 3 | finalize; the marker is written only at exit 0 | 9 / 1 |

What the evals phase runs:

| Step | Detail |
|---|---|
| Temperature fit | On the cohort `calibration` role, with disjointness attested against `dev`, `retention` and the train ids |
| Eval card | `run_card` on `dev`, raw and tempered; cohort hash stamped |
| G12 | Child arm on `steer_dev` |
| G13 | Child arm on `steer_gate` (`final=True`), retention, and the Index DEV subset (child only; parent from stage A) |
| Verdict | `verdict_g12` / `verdict_g13`, then the promotion verdict |

Every response is served through `decide.Decider(calibration=FeatureTemperature)`, so served probabilities use the
fitted temperature. `revision_id` = `<backend>+T:<temperature id>`.

Schedule 173 + reserve 7 = 180 min. **Worst case** (20 build + 2 × 180) × $0.0495 = **$18.81**, under the
**$20 hard cap**; `set_workspace_cap(MTD + 20)` applies when the job runs alone.

**Eval volume** (RWKV7 tokenizer, measured on the cohort rows): ≈ 598 tokens per steer unit.
- G12: 5.4 M tokens per arm; G13: 5.4 M tokens per arm.
- Child arms only: ≈ 10.8 M tokens, plus `dev`, `retention` and the 2,008 DEV rows.
- At an unmeasured prefill rate this may not fit 50 minutes. The evals phase must run a measured-rate preflight
  (`jobs/common/rate_guard.py`, using stage A's measured prefill tok/s) and stop with exit 15 before scoring if the
  projection exceeds its cap.
- An incomplete G12/G13 comparison **fails closed** by rule, so truncating the cohort is never an option.

## Stop conditions

Every stop below exits and lands in the receipt:
- any non-finite loss;
- peak > 76 GB;
- the measured training rate cannot reach the pass rule (26);
- a guard, cohort, records or snapshot identity mismatch;
- a D1-B hit;
- the measured eval-rate projection over its cap;
- the 180-minute ceiling.

## Before submit (blockers; none of these is done by this draft)

1. **`train.m2_gates` does not exist yet.** It is the evals driver: temperature fit, card, child and parent arms,
   G12/G13 and `valid_report`. The runner exits 16 without it.
2. **Parent arm on the cohort.** The parent's steer rows on `steer_dev` and `steer_gate` (and on `dev` and
   `retention`) are not scored anywhere.
   - Recommended: a one-off parent-arm job (no training, ≤ $10) that every m2 cell reuses, since the parent is frozen.
     Otherwise each cell doubles its eval volume.
3. **Margins.** `m_steer`, `m_cap` and `m_ece` are **PENDING USER APPROVAL**, and so is the G12/G13 steer-rule
   reconciliation from PR #27.
4. **G13 retention source** (user decision): the Index DEV subset for `g13_capability`/`g13_ece`, plus the cohort
   `retention` role for decision retention, as specified above.
5. **Parent.** If the decision tree selects Q9-AF, add its loader to `train.run_cell` and a `PARENT=q9af` branch.
6. **The usual spend steps:**
   - a `spend` issue with this estimate and these stop conditions, acknowledged by the other agent;
   - a LEDGER planned row;
   - `validate_job` against the $20 cap;
   - the SF credit-expiry check (promo expires 2026-10-27T06:53Z).
