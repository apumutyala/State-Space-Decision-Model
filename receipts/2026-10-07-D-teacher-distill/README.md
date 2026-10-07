# Clef-flash teacher export: fetch and verification (CPU, $0)

This receipt verifies the teacher data used by the knowledge-distillation (KD) arm: `train/losses.py` KD, `train/teacher.py`, and proxy-rule-v1 amendment A2 (recipes R16 and R17).

## Source

- Job: job-tnrwu (`clef_baseline/teacher_clef_flash` @ 5b478e3; mission clef-baseline-safety-spec-s7).
- Status: succeeded, $1.76 billed.
- Artifact: art-ggrs8, a tar of `/output`. sha256 `c20d13da05c83f171dd7b5414d01d2c3404ac8786cb35238b4f4bfe2ba9ef57f`, 26,808,320 bytes.

## Fetch steps (reproducible)

1. `get_job job-tnrwu` returns artifact `art-ggrs8`.
2. `get_artifact art-ggrs8` returns a presigned URL, valid about 7 days. The object itself is kept about 30 days.
3. `curl -o art-ggrs8.bin <url>`, then check `sha256sum art-ggrs8.bin` = `c20d13da...`.
4. `tar -xf art-ggrs8.bin`, then run `cd clef-flash-teacher && sha256sum -c SHA256SUMS`.

## SHA256SUMS result

- Every file matches except `stages.log`, as expected by construction. `jobs/clef_baseline/run_job.sh` writes SHA256SUMS before its last call, `stage "done"`, which appends one line to `stages.log`.
- The first 5 lines of `stages.log` hash to the listed `6fcf95ae...`, so nothing else changed.
- The teacher files all match:

| File | sha256 |
|---|---|
| `teacher/teacher-clef-flash.jsonl` | `a88b4fe0f6a79a8fc48bdea51391524bd4f5da885705461a543becf81c81e7f7` (= identity.json `teacher_sha256` = A2 pin) |
| `teacher/teacher-clef-flash.identity.json` | `f0394421...` |
| `teacher/selected-ids.txt` | `1c7dda1e...`; ids sha256 (newline-joined, draw order) `2bbbf180...` = the pinned m2 first-cell / q9-cache draw |
| `teacher/selection.json` | `fa666cec...` |

## Verification (`teacher_verify.json`)

Command:

```
python -X utf8 scripts/teacher_verify.py --teacher-dir <art>/clef-flash-teacher/teacher \
    --train-dir <ssdm-data>/m2/build/sample-150000-seed20261006 --out receipts/2026-10-07-D-teacher-distill/teacher_verify.json
```

Run under the repo's uv test env (torch CPU, transformers 5.17.0) in 131 s. Results:

1. **Identity.** `load_teacher` accepts the export against the A2 pin:
   - release, repo and revision `17f0b0ad`, plus the backend revision;
   - 24,000 rows: 23,458 ok, 542 exhausted (over 16,384 tokens, no probabilities);
   - per-row content hash = selection.json;
   - every distribution is finite and non-negative, and sums to 1 within 1e-6;
   - 30,727 questions carry probabilities.
2. **Draw alignment.** `train.train_set.draw` + `jobs/q9_feature_cache/cache_config.json` was recomputed with this tree, which excludes cohort v3 (38,202 records) before `split_ids`. It returns exactly the teacher's `selected-ids.txt`, same order, sha `2bbbf180...`.
   - The stock_train cache (job-wuw7g) keeps its 20,000 rows from these records in draw order, so every cached training row is a teacher record or an exhausted one.
   - `proxy_loop.check_cache_draw` re-checks this on the cache's own `summary_*.json` before any KD recipe trains.
3. **Contamination.** All checks report 0 hits:
   - no teacher id is a DEV/TEST manifest id;
   - no teacher record is in cohort v2 (`be6efe2b...`, by id and content) or in cohort v3 (`498efa61...`, merged PR #105, by id, normalized content and source item);
   - all 24,000 drawn records pass the production DEV/TEST `SplitGuard`, i.e. the content index pinned by `jobs/m2_pilot/guard_pin.json`;
   - all 24,000 pass `harness.heldout.check_record` (frozen families 9, 20, 22, 31, 32, 33, 44, 58, 64; heldout-draw-v1 `c3172003...`).
   - The job itself (at 5b478e3, before #105) did not run the cohort v3 check; this receipt does.
4. **Option alignment.** All 31,269 training items (`items_from_records`: choice 27,839, score 1,968, noul 1,462) were mapped by record id, question id and option id:
   - 30,727 items get a teacher distribution with an identical option set, with no refusals;
   - 542 fall back to the clef arm; these are exactly the exhausted records, one question each.
   - Teacher argmax equals the gold label on 87.7% of the taught training items. This is reported only; it is not a gate.

The teacher files are not committed; they are 23 MB and come from the artifact above. `train.teacher.load_teacher` refuses any other bytes.
