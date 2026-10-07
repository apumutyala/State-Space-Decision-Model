#!/usr/bin/env bash
# Assemble the Clef job build context (run locally; CPU only). Nothing is submitted.
#   jobs/clef_baseline/build_context.sh <index|teacher|all> [<sha> [<out_dir>]]
# Code: `git archive <sha>` of the WHOLE repo (decide/, safety_eval/, baselines/, third_party/, jobs/, harness/, train/,
# heads/, data/manifests/ ...). The sha must be merged into origin/agent-main (LESSONS L-16) unless ALLOW_UNMERGED=1
# (local dry runs only; never for a context that is uploaded). harness/ comes from the SAME sha: the strict results
# reader (harness/jsonl.py, the job-sgihj fix) must ship, so the old f3c7791 harness pin is retired.
# Data (PRIVATE; the context goes only to our own SF workspace via create_context, never to a public place; delete the
# context after the jobs end):
#   index:   slices/dev-rows.jsonl.gz (sha256 pinned in data.sha256)
#            + OPTIONAL reference/clef_flash.decide.results.jsonl: job-sgihj's decide results (art-8y6d8), pinned below;
#              enables the in-job paired compare Clef - Clef-flash on identical rows
#   teacher: m2-150k/ (records + manifest + content map; each checked against the committed pointer) and suite-0.2/
#            (official row files for jobs.m2_pilot.guard_ready's content index; hashes pinned in guard_pin.json)
set -euo pipefail
MODE=${1:?index|teacher|all}; SHA=${2:-HEAD}; OUTD=${3:-clef-baseline-context-$MODE}
DATA=${SSDM_DATA:-$HOME/research/CHLabs/ssdm-data}
REF_FLASH=${REF_FLASH:-}   # path to job-sgihj index/decide/results.jsonl (optional)
REF_FLASH_SHA=463f21ff492fc792da4d40ecc501cffab0498d2458715f8effd3270614cbd0aa
B150=$DATA/m2/build/sample-150000-seed20261006
case "$MODE" in index|teacher|all) ;; *) echo "mode must be index, teacher or all" >&2; exit 2 ;; esac
cd "$(git rev-parse --show-toplevel)"
SHA=$(git rev-parse "$SHA")
if [ "${ALLOW_UNMERGED:-0}" != 1 ]; then
  git fetch -q origin agent-main
  git merge-base --is-ancestor "$SHA" origin/agent-main || { echo "$SHA is not merged into agent-main" >&2; exit 1; }
fi
rm -rf "$OUTD"; mkdir -p "$OUTD/app" "$OUTD/data"
git -c core.autocrlf=false archive "$SHA" | tar -x -C "$OUTD/app"
cp "$OUTD/app/jobs/clef_baseline/Dockerfile" "$OUTD/Dockerfile"
if [ "$MODE" != teacher ]; then
  mkdir -p "$OUTD/data/slices"
  cp "$DATA/slices/dev-rows.jsonl.gz" "$OUTD/data/slices/"
  [ "$(sha256sum "$OUTD/data/slices/dev-rows.jsonl.gz" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$OUTD/app/jobs/clef_baseline/data.sha256")" ] \
    || { echo "DEV rows sha256 mismatch" >&2; exit 1; }
  if [ -n "$REF_FLASH" ]; then
    mkdir -p "$OUTD/data/reference"
    cp "$REF_FLASH" "$OUTD/data/reference/clef_flash.decide.results.jsonl"
    [ "$(sha256sum "$OUTD/data/reference/clef_flash.decide.results.jsonl" | cut -d' ' -f1)" = "$REF_FLASH_SHA" ] \
      || { echo "job-sgihj reference sha256 mismatch" >&2; exit 1; }
  fi
fi
if [ "$MODE" != index ]; then
  mkdir -p "$OUTD/data/m2-150k" "$OUTD/data/suite-0.2"
  cp "$B150/records.jsonl.gz" "$B150/sample-150000-seed20261006.manifest.jsonl.gz" \
     "$B150/sample-150000-seed20261006.content.jsonl.gz" "$OUTD/data/m2-150k/"
  cp "$DATA/suite-0.2/selected-rows.jsonl.gz" "$DATA/suite-0.2/added-rows.jsonl.gz" \
     "$DATA/suite-0.2/excluded-questions.json" "$OUTD/data/suite-0.2/"
  # local dry run of the CPU guard on exactly the files that ship (same exit codes as on the box: 20 / 10)
  ( cd "$OUTD/app" && python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir "$OUTD/data/suite-0.2" \
      --kit-dir "$DATA/decision-index" --out-dir "$OUTD/guard-local" ) > /dev/null || { echo "guard not ready" >&2; exit 1; }
  ( cd "$OUTD/app" && python -X utf8 -m baselines.clef_teacher select --train-dir "$OUTD/data/m2-150k" \
      --guard-readiness "$OUTD/guard-local/readiness.json" --out "$OUTD/teacher-local" ) || { echo "teacher select refused" >&2; exit 1; }
fi
python - "$OUTD" "$SHA" "$MODE" <<'PY'
import hashlib, json, sys, pathlib
d, sha, mode = sys.argv[1:4]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
a = f"{d}/app"
files = ["jobs/clef_baseline/run_job.sh", "jobs/clef_baseline/summarize.py", "jobs/clef_baseline/price.py",
         "jobs/clef_baseline/teacher_scope.json", "harness/jsonl.py", "baselines/clef_teacher.py",
         "jobs/clef_baseline/job_params_index_subset.json", "jobs/clef_baseline/job_params_index_full.json",
         "jobs/clef_baseline/job_params_teacher_clef_flash.json", "jobs/clef_baseline/job_params_teacher_clef.json"]
tl = pathlib.Path(d, "teacher-local", "selection.json")
man = {"branch_commit": sha, "mode": mode, "dockerfile_sha256": h(f"{d}/Dockerfile"),
       "files_sha256": {f: h(f"{a}/{f}") for f in files},
       "data_sha256": {str(p.relative_to(d)): h(p) for p in sorted(pathlib.Path(d, "data").rglob("*")) if p.is_file()},
       "teacher_select_local": {k: json.loads(tl.read_text())[k] for k in ("n_drawn", "drawn_ids_sha256", "guard_refused")}
       if tl.exists() else None}
json.dump(man, open(f"{a}/CONTEXT_MANIFEST.json", "w"), indent=1)
print(json.dumps({k: man[k] for k in ("branch_commit", "mode", "teacher_select_local")}, indent=1))
PY
rm -rf "$OUTD/guard-local" "$OUTD/teacher-local"
tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -C "$OUTD" -c . | zstd -19 -o "$OUTD.tar.zst"
sha256sum "$OUTD.tar.zst"; stat -c %s "$OUTD.tar.zst"
