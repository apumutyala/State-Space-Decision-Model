#!/usr/bin/env bash
# Build the immutable build context for the rung-2 qualification job (CPU only, local). Usage:
#   jobs/qual/build_context.sh <m1_sha> <out_dir>
# The SHA must be reachable from origin/agent-main (merged code only; issue #3 items 3/4).
set -euo pipefail
M1=$1; OUTD=$2
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin agent-main
for s in $M1; do
  git merge-base --is-ancestor "$s" origin/agent-main || { echo "$s is not merged into agent-main" >&2; exit 1; }
done
rm -rf "$OUTD"; mkdir -p "$OUTD/code/m1"
git archive "$M1"  | tar x -C "$OUTD/code/m1"
cp "$OUTD/code/m1/jobs/qual/Dockerfile" "$OUTD/Dockerfile"
python - "$OUTD" "$M1" <<'PY'
import hashlib, json, sys, pathlib
d, m1 = sys.argv[1:3]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
man = {"m1_sha": m1,
       "config_sha256": {"configs/qual_h100.json": h(f"{d}/code/m1/configs/qual_h100.json")},
       "dockerfile_sha256": h(f"{d}/Dockerfile"), "runner_sha256": h(f"{d}/code/m1/jobs/qual/qual_job.sh"),
       "job_params_sha256": h(f"{d}/code/m1/jobs/qual/job_params.json")}
json.dump(man, open(f"{d}/CONTEXT_MANIFEST.json", "w"), indent=1)
print(json.dumps(man, indent=1))
PY
tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -czf "$OUTD.tar.gz" -C "$OUTD" .
sha256sum "$OUTD.tar.gz"
