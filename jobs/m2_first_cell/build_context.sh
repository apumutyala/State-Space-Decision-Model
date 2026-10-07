#!/usr/bin/env bash
# Build the immutable build context for the first trained m2 cell (CPU only, local). DRAFT; nothing is submitted.
#   jobs/m2_first_cell/build_context.sh <sha> <out_dir> [<tkfaj stock results.jsonl> [<clef-flash results.jsonl>]]
# Code: `git archive <sha>` (must be merged into origin/agent-main). Data (PRIVATE: our own SF workspace only, via
# create_context; delete the context after the job):
#   m2/records.jsonl.gz          PR #16 sample: the cohort's dev / calibration / retention records (eval only)
#   m2-150k/                     PR #44 training set: records.jsonl.gz + manifest + content map, each pinned by the
#                                pointer data/manifests/sample-150000-seed20261006.pointer.json (verified here and on the box)
#   suite-0.2/                   the official 0.2.1 row files for the guard's content index (jobs.m2_pilot.guard_ready)
#   slices/dev-rows.jsonl.gz     Index DEV rows (the job cuts the 2,008-row subset; TEST stays sealed)
#   reference/<name>.results.jsonl  OPTIONAL, same subset rows: job-tkfaj stock readout, Clef-flash. Pin their sha256 in
#                                job_params.json reference_results_sha256 to have the report compare against them.
# No parent arm cache ships: the job scores the parent itself (phases parent_g12, parent_index).
set -euo pipefail
SHA=$1; OUTD=$2; TKFAJ=${3:-}; CLEF=${4:-}
DATA=${SSDM_DATA:-$HOME/research/CHLabs/ssdm-data}
B150=$DATA/m2/build/sample-150000-seed20261006
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin agent-main
git merge-base --is-ancestor "$SHA" origin/agent-main || { echo "$SHA is not merged into agent-main" >&2; exit 1; }
rm -rf "$OUTD"; mkdir -p "$OUTD/code/m1" "$OUTD/data/m2" "$OUTD/data/m2-150k" "$OUTD/data/suite-0.2" "$OUTD/data/slices"
git archive "$SHA" | tar x -C "$OUTD/code/m1"
cp "$OUTD/code/m1/jobs/rwkv7/Dockerfile" "$OUTD/Dockerfile"
sed -i 's#jobs/rwkv7/rwkv7_job.sh#jobs/m2_first_cell/first_cell_job.sh#' "$OUTD/Dockerfile"
cp "$DATA/m2/build/sample-20000-seed20261006/records.jsonl.gz" "$OUTD/data/m2/records.jsonl.gz"
( cd "$OUTD/data" && sha256sum -c "$OUTD/code/m1/jobs/rwkv7/records.sha256" )
cp "$B150/records.jsonl.gz" "$B150/sample-150000-seed20261006.manifest.jsonl.gz" \
   "$B150/sample-150000-seed20261006.content.jsonl.gz" "$OUTD/data/m2-150k/"
( cd "$OUTD/code/m1" && python -X utf8 -m train.train_set verify --config jobs/m2_first_cell/m2_first_cell_rwkv7.json \
    --train-dir "$OUTD/data/m2-150k" ) || { echo "150k set != pointer" >&2; exit 1; }
cp "$DATA/suite-0.2/selected-rows.jsonl.gz" "$DATA/suite-0.2/added-rows.jsonl.gz" "$DATA/suite-0.2/excluded-questions.json" \
  "$OUTD/data/suite-0.2/"
cp "$DATA/slices/dev-rows.jsonl.gz" "$OUTD/data/slices/"
( cd "$OUTD/data" && sha256sum -c "$OUTD/code/m1/jobs/clef_baseline/data.sha256" )
if [ -n "$TKFAJ" ] || [ -n "$CLEF" ]; then mkdir -p "$OUTD/data/reference"; fi
[ -n "$TKFAJ" ] && cp "$TKFAJ" "$OUTD/data/reference/rwkv7_stock_tkfaj.results.jsonl"
[ -n "$CLEF" ] && cp "$CLEF" "$OUTD/data/reference/clef_flash.results.jsonl"
# local dry runs on exactly the files that ship (same exit codes as on the box)
( cd "$OUTD/code/m1" && python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir "$OUTD/data/suite-0.2" \
    --out-dir "$OUTD/guard-local" ) || { echo "guard not ready on the shipped suite" >&2; exit 1; }
( cd "$OUTD/code/m1" && python -X utf8 -m train.train_set heldout --config jobs/m2_first_cell/m2_first_cell_rwkv7.json \
    --train-dir "$OUTD/data/m2-150k" ) || { echo "held-out hits != pinned exclusions" >&2; exit 1; }
python - "$OUTD" "$SHA" <<'PY'
import hashlib, json, sys, pathlib
d, sha = sys.argv[1:3]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c = f"{d}/code/m1"
files = ["jobs/m2_first_cell/first_cell_job.sh", "jobs/m2_first_cell/job_params.json",
         "jobs/m2_first_cell/m2_first_cell_rwkv7.json", "jobs/m2_pilot/guard_pin.json",
         "data/manifests/sample-150000-seed20261006.pointer.json", "train/cohorts/m2-dev-cohort-v2.jsonl.gz",
         "jobs/rwkv7/snapshot.sha256"]
man = {"job": "m2_first_cell", "sha": sha, "dockerfile_sha256": h(f"{d}/Dockerfile"),
       "files_sha256": {f: h(f"{c}/{f}") for f in files},
       "data_sha256": {str(p.relative_to(d)): h(p) for p in sorted(pathlib.Path(d, "data").rglob("*")) if p.is_file()},
       "guard_local": json.load(open(f"{d}/guard-local/readiness.json"))["observed"]}
json.dump(man, open(f"{d}/CONTEXT_MANIFEST.json", "w"), indent=1)
print(json.dumps(man, indent=1))
PY
rm -rf "$OUTD/guard-local"
tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -C "$OUTD" -c . | zstd -19 -o "$OUTD.tar.zst"
sha256sum "$OUTD.tar.zst"; stat -c %s "$OUTD.tar.zst"
