#!/usr/bin/env bash
# Build context for jobs/m2_parent_arm (CPU only, local; DRAFT, nothing submitted).
#   jobs/m2_parent_arm/build_context.sh <sha> <out_dir> [<q9af gdn_converted.pt>]
# PRIVATE data (own SF workspace only; delete the context after the job): PR #16 records, the official 0.2.1 suite
# row files (guard readiness rebuilds the content index from them), Index DEV rows (the job cuts the 2,008-row subset),
# and for PARENT=q9af the converted GDN checkpoint (sha256 pinned in jobs/m2_pilot/m2_pilot_cell_q9af.json).
set -euo pipefail
SHA=$1; OUTD=$2; GDN=${3:-}
DATA=${SSDM_DATA:-$HOME/research/CHLabs/ssdm-data}
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin agent-main
git merge-base --is-ancestor "$SHA" origin/agent-main || { echo "$SHA is not merged into agent-main" >&2; exit 1; }
rm -rf "$OUTD"; mkdir -p "$OUTD/code/m1" "$OUTD/data/m2" "$OUTD/data/suite-0.2" "$OUTD/data/slices"
git archive "$SHA" | tar x -C "$OUTD/code/m1"
cp "$OUTD/code/m1/jobs/rwkv7/Dockerfile" "$OUTD/Dockerfile"
sed -i 's#jobs/rwkv7/rwkv7_job.sh#jobs/m2_parent_arm/parent_arm_job.sh#' "$OUTD/Dockerfile"
cp "$DATA/m2/build/sample-20000-seed20261006/records.jsonl.gz" "$OUTD/data/m2/records.jsonl.gz"
( cd "$OUTD/data" && sha256sum -c "$OUTD/code/m1/jobs/rwkv7/records.sha256" )
cp "$DATA/suite-0.2/selected-rows.jsonl.gz" "$DATA/suite-0.2/added-rows.jsonl.gz" "$DATA/suite-0.2/excluded-questions.json" \
  "$OUTD/data/suite-0.2/"
cp "$DATA/slices/dev-rows.jsonl.gz" "$OUTD/data/slices/"
( cd "$OUTD/data" && sha256sum -c "$OUTD/code/m1/jobs/clef_baseline/data.sha256" )
if [ -n "$GDN" ]; then mkdir -p "$OUTD/data/q9af"; cp "$GDN" "$OUTD/data/q9af/gdn_converted.pt"; fi
( cd "$OUTD/code/m1" && python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir "$OUTD/data/suite-0.2" \
    --out-dir "$OUTD/guard-local" ) || { echo "guard not ready on the shipped suite" >&2; exit 1; }
rm -rf "$OUTD/guard-local"
python - "$OUTD" "$SHA" <<'PY'
import hashlib, json, sys, pathlib
d, sha = sys.argv[1:3]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
man = {"job": "m2_parent_arm", "sha": sha, "dockerfile_sha256": h(f"{d}/Dockerfile"),
       "data_sha256": {str(p.relative_to(d)): h(p) for p in sorted(pathlib.Path(d, "data").rglob("*")) if p.is_file()}}
json.dump(man, open(f"{d}/CONTEXT_MANIFEST.json", "w"), indent=1)
print(json.dumps(man, indent=1))
PY
tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -C "$OUTD" -c . | zstd -19 -o "$OUTD.tar.zst"
sha256sum "$OUTD.tar.zst"; stat -c %s "$OUTD.tar.zst"
