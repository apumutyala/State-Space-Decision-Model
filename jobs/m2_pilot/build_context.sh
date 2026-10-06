#!/usr/bin/env bash
# Build the immutable build context for the m2 pilot job (CPU only, local). DRAFT; nothing is submitted.
#   jobs/m2_pilot/build_context.sh <sha> <out_dir> <stageA_results.jsonl>
# Code: `git archive <sha>` (must be merged into origin/agent-main). Data (PRIVATE: goes only to our own SF workspace via
# create_context, never anywhere public; delete the context after the job):
#   m2/records.jsonl.gz       PR #16 sample (sha256 pinned in jobs/rwkv7/records.sha256)
#   suite-0.2/{selected-rows.jsonl.gz, added-rows.jsonl.gz, excluded-questions.json}
#                             the official 0.2.1 row files; jobs.m2_pilot.guard_ready checks their uncompressed sha256
#                             against jobs/m2_pilot/guard_pin.json and rebuilds the guard's content index from them
#   stageA/index/stock/results.jsonl   the parent's stock-readout Index DEV results from rwkv7 stage A (parent arm of
#                             g13_capability; its sha256 goes into CONTEXT_MANIFEST.json)
set -euo pipefail
SHA=$1; OUTD=$2; STAGEA=$3
DATA=${SSDM_DATA:-$HOME/research/CHLabs/ssdm-data}
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin agent-main
git merge-base --is-ancestor "$SHA" origin/agent-main || { echo "$SHA is not merged into agent-main" >&2; exit 1; }
rm -rf "$OUTD"; mkdir -p "$OUTD/code/m1" "$OUTD/data/m2" "$OUTD/data/suite-0.2" "$OUTD/data/stageA/index/stock"
git archive "$SHA" | tar x -C "$OUTD/code/m1"
cp "$OUTD/code/m1/jobs/rwkv7/Dockerfile" "$OUTD/Dockerfile"
sed -i 's#jobs/rwkv7/rwkv7_job.sh#jobs/m2_pilot/m2_job.sh#' "$OUTD/Dockerfile"
cp "$DATA/m2/build/sample-20000-seed20261006/records.jsonl.gz" "$OUTD/data/m2/records.jsonl.gz"
( cd "$OUTD/data" && sha256sum -c "$OUTD/code/m1/jobs/rwkv7/records.sha256" )
cp "$DATA/suite-0.2/selected-rows.jsonl.gz" "$DATA/suite-0.2/added-rows.jsonl.gz" "$DATA/suite-0.2/excluded-questions.json" \
  "$OUTD/data/suite-0.2/"
cp "$STAGEA" "$OUTD/data/stageA/index/stock/results.jsonl"
# local readiness dry run on exactly the files that ship (same exit codes as on the box)
( cd "$OUTD/code/m1" && python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir "$OUTD/data/suite-0.2" \
    --out-dir "$OUTD/guard-local" ) || { echo "guard not ready on the shipped suite" >&2; exit 1; }
python - "$OUTD" "$SHA" <<'PY'
import hashlib, json, sys, pathlib
d, sha = sys.argv[1:3]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c = f"{d}/code/m1"
files = ["jobs/m2_pilot/m2_job.sh", "jobs/m2_pilot/job_params.json", "jobs/m2_pilot/m2_pilot_cell.json",
         "jobs/m2_pilot/guard_pin.json", "train/cohorts/m2-dev-cohort-v1.jsonl.gz", "jobs/rwkv7/snapshot.sha256"]
man = {"job": "m2_pilot", "sha": sha, "dockerfile_sha256": h(f"{d}/Dockerfile"),
       "files_sha256": {f: h(f"{c}/{f}") for f in files},
       "data_sha256": {str(p.relative_to(d)): h(p) for p in sorted(pathlib.Path(d, "data").rglob("*")) if p.is_file()},
       "guard_local": json.load(open(f"{d}/guard-local/readiness.json"))["observed"]}
json.dump(man, open(f"{d}/CONTEXT_MANIFEST.json", "w"), indent=1)
print(json.dumps(man, indent=1))
PY
rm -rf "$OUTD/guard-local"
tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -C "$OUTD" -c . | zstd -19 -o "$OUTD.tar.zst"
sha256sum "$OUTD.tar.zst"; stat -c %s "$OUTD.tar.zst"
