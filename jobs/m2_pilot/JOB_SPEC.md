# m2_pilot: the first paid m2 cell (≤ $20). DRAFT, NOT SUBMITTED

**Cell.** The cache-fork head (`head = "fork"`, `fork_mode = "cache"`, CE arm, LoRA) on the parent that
`jobs/DECISION_TREE.md` selects at m1. The runner (`jobs/m2_pilot/m2_job.sh`) and cell config
support both possible parents:
- **RWKV7-G1k-13.3B** (`PARENT=rwkv7`, `jobs/m2_pilot/m2_pilot_cell.json`), the parent in decision-tree branches B
  and C;
- **Q9-AF** (`PARENT=q9af`, `jobs/m2_pilot/m2_pilot_cell_q9af.json`): stock `Qwen/Qwen3.5-9B@c2022362…` with all 8
  attention layers replaced by the converted GDNs from `convert/` (`heads.backbone.load_q9af_backbone`).
  - The converted checkpoint's sha256 is a required pin, checked before any weight loads. Without it the job exits 29.
  - The pin and its provenance are stamped into the backbone lineage, so every checkpoint, resume and decide revision
    names the exact conversion.
  - The pin is `null` until the full-8 conversion runs.

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
| Development cohort | `train/cohorts/m2-dev-cohort-v2.jsonl.gz`, sha256 `be6efe2b…fdacf2` (uncompressed text) | `python -m train.cohort verify` must rebuild it byte-for-byte; `run_cell` refuses on mismatch (exit 28) |
| Guard content index | sha256 `7cbe6c89…052d8d`, 24,150 hashes | `jobs.m2_pilot.guard_ready` (below) |
| Margins | `train.promotion.MARGINS`: `m_steer` 0.02, `m_cap` 1.0 Index points, `m_ece` 0.01 | **PENDING USER APPROVAL** |

The margins are read from config (`train.promotion.MARGINS`), never restated in code paths. Until the user approves
them, the verdict is reported as `accept|reject (margins PENDING USER APPROVAL)`, and nothing is promoted on it.

## Data roles (`train.cohort` v2, frozen; disjoint by id AND by content)

v2 replaces v1 after the PR #29 review (B1).
- **The leak.** v1 assigned record ids. 17 `synthetic_factchange` content groups (exact same state and questions
  under different ids) straddled the cohort and the training draw, and 4 of them spanned dev, calibration and
  retention.
- **The fix.** v2 groups records by normalized content and assigns whole groups. The groups come from the
  committed content map `data/manifests/sample-20000-seed20261006.content.jsonl.gz` (hashes only).
- **The training pool** excludes every record whose content is held out.
- **On the box,** `run_cell` re-hashes every training record and checks it against the map; any drift or twin
  is refused (exit 28).

| Role | Size | Source | Used for |
|---|---|---|---|
| training pool | 16,495 records (3,000 drawn, `seed` 0) | PR #16 records outside the cohort by id and content | LoRA + head training; `assert_training_ids` + `assert_training_content` |
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
  - Power is a Monte Carlo estimate over 400 null simulations. Its MC s.e. is ±0.018 at power 0.84 and ±0.019
    at 0.83, and is recorded per row as `power_mc_se`.
  - The sizing rests on planning assumptions: parent steer rate 0.5, discordance 0.10, clean-correct 0.6
    (`PLANNING`). They are not measurements. The first G12 run measures the real discordance, and the sizes are
    re-derived from it.
- `dev` comes from McNemar: 709 items, at discordance 0.10 and power 0.8.
- **G13 capability** (coordinator decision 2026-10-06) is the 2,008-row Index **DEV** subset
  (`jobs/common/dev_subset_m1.json`) **plus** the cohort `retention` set. DEV is a development slice; Index TEST stays
  sealed.
  - `g13_capability` / `g13_ece` (`retention_comparison`) read the DEV subset.
  - `g13_retention_acc` (lower CI ≥ −`m_ret_acc`) and `g13_retention_ece` (upper CI ≤ `m_ece`) read the retention
    set (`verdict_g13(decision_retention=...)`).
  - Both arms (child here, parent from `jobs/m2_parent_arm`) are scored by the same `train.m2_gates.score_stage`.

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

Locally reproduced (not on the box) on 2026-10-06 from `suite-0.2` (with exclusions) and `m2/suite-snapshot` (without): both gave
`7cbe6c89…` (receipt `receipts/2026-10-06-D-m2-blockers/`).

## Phases (`job_params.json`; H100 batch $0.0495/min)

| Phase | Cap (min) | Content | Exit on failure |
|---|---|---|---|
| setup | 15 | env pins; records sha256; cohort rebuild == committed; snapshot download + sha256; vendored code | 5 / 10 |
| guard | 5 | `jobs.m2_pilot.guard_ready` (above) | 30 / 31 / 32 / 7 |
| heldout | 5 | `train.heldout`: no D1-B template word-6-gram hash in any training record (20,000 local: 0 hits) | 33 / 7 |
| cell | 95 | `train.run_cell`: smoke (cache == recompute fork), T ladder, LoRA + fork head training; train ids ⊆ pool, eval = cohort `dev` | 21–29 / 7 |
| evals | 50 | measured-rate preflight (`score_parent estimate`), then `train.m2_gates --stage all`, described in the next table | 15 / 40 / 41 / 42 / 10 / 9 / 7 |
| report | 3 | finalize; the marker is written only at exit 0 | 9 / 1 |

What the evals phase runs:

| Step | Detail |
|---|---|
| Restore | `Trainer.load` checks the config hash and lineage. The training items are rebuilt from the same cohort split. |
| Temperature fit | On the cohort `calibration` role, with disjointness attested against `dev`, `retention` and the train ids |
| Eval card | `run_card` on `dev`, raw and tempered; cohort hash stamped |
| G12 | Child arm on `dev` + `steer_dev`, against the parent cache's g12 units |
| G13 | Child arm on `retention` + `steer_gate` (`final=True`) + the Index DEV subset, against the parent cache's g13 units |
| Verdict | `verdict_g12` / `verdict_g13`, then the promotion verdict |

The promotion verdict is `accept` only if every stage accepted; otherwise the exit is 40 (G12), 41 (G13) or
42 (both). The report is exported in both cases, and `valid_report` is the finalize validator. `promotable` stays
false while the margins are pending.

Every response is served through `decide.Decider(calibration=FeatureTemperature)`, so served probabilities use the
fitted temperature. `revision_id` = `<backend>+T:<temperature id>`.

Schedule 173 + reserve 7 = 180 min. **Worst case** (20 build + 2 × 180) × $0.0495 = **$18.81**, under the
**$20 hard cap**; `set_workspace_cap(MTD + 20)` applies when the job runs alone.

**Eval volume** (RWKV7 tokenizer, measured on the cohort; `jobs/m2_parent_arm/score_parent.VOLUME_TOKENS`):
- **Child arm:** 26.8 M tokens (G12 9.3 M, G13 17.5 M; Index DEV re-measured as exact prefix-fork positions, `jobs/PERF_AUDIT.md`). The parent arm is cached once per parent.
- **Evals phase total:** the arm plus the temperature fit over every calibration item (3.4 M) and the dev eval card
  (4.0 M) gives **34.2 M tokens** (`estimate(..., mode="child")`; PR #29 round-2 review B4).
- **Rate needed:** at the 50-minute evals cap, with the 95 % headroom rule, this needs **≥ ~12,000 tok/s**
  (`score_parent.min_tok_s(50)`).
- **Preflight:** it runs in setup, before the paid cell phase, as `score_parent estimate --mode child --cap-min
  $CAP_E` with `TOK_S`, a measured rate. It stops with exit 15 if the projection is over.
- **No `TOK_S`:** if `TOK_S` is unset or not a number, the job exits 4.
- Below that rate, run G12 here (`--stage g12`) and G13 as a separate gate job.
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

1. **Parent arm cache.** Run `jobs/m2_parent_arm` once for the chosen parent (≤ $20; ≈ $4–6 at 5–10k tok/s) and ship
   its `arm.json` and units in the context (`build_context.sh <sha> <out> <arm_dir> [<gdn.pt>]`).
2. **A measured `TOK_S`** for the parent: rwkv7 stage-A probe, or the probe9b throughput for Q9-AF.
3. **Margins.** `m_steer`, `m_cap`, `m_ece` and `m_ret_acc` are **PENDING USER APPROVAL**, and so is the G12/G13
   steer-rule reconciliation from PR #27.
4. **Q9-AF only:** the full-8 conversion output and its sha256 pinned in `m2_pilot_cell_q9af.json`
   (`converted_sha256`, optionally `base_config_sha256`).
5. **The usual spend steps:**
   - a `spend` issue with this estimate and these stop conditions, acknowledged by the other agent;
   - a LEDGER planned row;
   - `validate_job` against the $20 cap;
   - the SF credit-expiry check (promo expires 2026-10-27T06:53Z).

## Cohort v3 retention cost (G13 child arm)

Cohort v3 (`train/cohort_v3.py`, sha256 `0c0309b9…c32f`) grows the G13 retention role to 39,672 records: 36,005
usable items on RWKV7 and 35,992 on Qwen3.5, sized for issue #36's retention ECE and accuracy margins. These are
measured volumes at measured rates (`receipts/2026-10-07-D-cohort-v3/measure.json`). The retention term alone costs,
per arm:

| Backbone | Rate | Minutes | $ |
|---|---|---|---|
| RWKV7-13.3B | job-tkfaj | **218.9** | **$10.83** |
| Q9 (Qwen3.5-9B) | job-5cmat | **78.8** | **$3.90** |

- These figures are per arm. They double when the parent arm is not cached. Pricing details are in `jobs/m2_parent_arm/JOB_SPEC.md`.
- G13 on v3 cannot run inside this cell's 50-minute evals phase. It runs as its own gate job.
- The cell configs here still pin v2. Switching to v3 means three things:
  - re-caching the parent arm;
  - passing `pool_dirs={"m2-150k": <dir>}` to `Cohort.load_role_records`;
  - the v3 training-draw guard, which is already enforced in `baselines.clef_teacher`.
