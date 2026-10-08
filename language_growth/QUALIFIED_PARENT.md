# Qualified native M2 parent admission

`qualified_parent.load_qualified_parent` connects the merged native M2 serving
bridge to the existing L0 planner after strict Stage 1 admission. It takes
`checkpoint`, independently pinned `report`/`report_sha256` and
`qualification`/`qualification_sha256`, plus a local `snapshot`. Optional
`device` and `dtype` default to CPU and bfloat16. It returns `QualifiedParent`
with `decide`, `plan`, `planner_kwargs` and `identity`. A reviewed custom strict
`NativeParentLoader` may replace built-in reconstruction via `loader=`.

```python
parent = load_qualified_parent(
    checkpoint, report=report, report_sha256=report_sha256,
    qualification=qualification, qualification_sha256=qualification_sha256,
    snapshot=local_snapshot, device="cpu", dtype="bfloat16")
result = parent.plan(task_input, host, catalog)
```

The qualification is an independently reviewed host attestation. Its SHA must
come from the host's trusted review channel. A matching hash proves byte identity;
the module does not authenticate a reviewer, fetch evidence, recompute Stage 1
metrics, or prove that a named commit was merged. A caller must review the pinned
evidence and merged-source assertion before supplying those pins. Neither a
ledger `keep` nor M2 promotion alone qualifies Stage 1. No qualified artifact or
language capability is supplied by these changes.

The JSON qualification has exactly these keys:

| Key | Required content |
| --- | --- |
| `schema`, `verdict`, `model_line` | `ssdm.stage1.qualification.v1`, `qualified`, `rwkv7-13.3b` |
| `artifact_sha256`, `report_sha256` | Exact checkpoint and report file SHA-256 |
| `serving_bundle_sha256` | `canonical_sha256(report['identities']['serving_bundle'])` |
| `checkpoint_config`, `checkpoint_config_hash` | Complete saved `TrainConfig.to_json()` and its existing `.hash()` |
| `checkpoint_lineage` | Complete saved backbone, tokenizer, layout and data identity |
| `lora_modules` | Exact sorted saved LoRA module list, empty for rank zero |
| `rwkv7_wkv` | Explicit original WKV choice, including null for unchanged snapshot default |
| `snapshot_files_sha256` | Every file under the local snapshot keyed by relative POSIX path, with full file SHA-256 |
| `stage1` | Evidence contract below |

`stage1` contains exactly `evaluation_scope="full_dev"`,
`primary_metric="decision_index"`, `preregistration_sha256`,
`matched_evaluation_sha256`, `independent_rescore_sha256`, `merged_commit` (full
40-character commit SHA), `paired_ci95`, and `floors`. The two CI keys are
`clef-27b` and `clef-flash`, each containing exactly numeric `lower`, `upper`,
`confidence=0.95`, `paired=true`, `matched=true`; both lower bounds must exceed
zero. The floor keys are exactly `safety`, `calibration`, `abstention`, `control`,
`interpretability`, each with `passed=true` and an `evidence_sha256`. Hashes use
lowercase SHA-256. The reviewed evidence must establish the stated scope and
preregistered floors; this admission layer does not invent floor thresholds.

The M2 report itself must be complete, promotable G12/G13 with exit zero, no
pending margins, no underpowered result, and an exact full serving bundle.
Calibration remains the fitted full payload; it is never refit or defaulted.
The existing bridge verifies checkpoint bytes before and after reconstruction
and binds exact served revision, runtime, encoder, template and configuration.

Built-in reconstruction verifies every snapshot file before and after loading,
uses vendored RWKV7 classes and `local_files_only=True`, restores exactly the
saved head/typed/LoRA key set, shape and dtype, rejects nonfinite tensors before
copying any parameter, and checks frozen-backbone lineage. It uses the restricted
PyTorch weights-only reader with the inert NumPy RNG types needed by current
Trainer checkpoints. There is no optimizer/data/evaluation loading or unsafe
pickle fallback. Trust-gated/G16 reconstruction is explicitly unsupported until
a reviewed loader supplies that architecture. FLA selection retains its existing
GPU verification gate. A different path-sensitive backbone identity, dtype,
runtime or implementation requires matching qualification; no silent migration.
Keep the checkpoint and snapshot immutable throughout loading.

Metadata-only preflight opens neither checkpoint nor snapshot:

```text
python -m language_growth.qualified_parent --parent CHECKPOINT --parent-report REPORT --parent-report-sha256 REPORT_SHA --parent-qualification QUALIFICATION --parent-qualification-sha256 QUALIFICATION_SHA
```

Add `--parent-snapshot SNAPSHOT --task TASK_JSON` to reconstruct and answer one
public task through the existing checked public boundary. `TASK_JSON` contains
`task_input`, `host`, `catalog`; provider diagnostics are not public prose.
Actual 13.3B loading requires appropriate separately authorized resources.

External prerequisites remain a reviewed Stage 1-qualified checkpoint/report/
qualification pair, all original snapshot files and exact runtime dependencies.
Tiny synthetic tests establish admission and reconstruction failure behavior;
they do not establish full-model load parity, Stage 1 success, learned language
usefulness, decision retention, semantic intervention efficacy or any L0–L4 gate.
