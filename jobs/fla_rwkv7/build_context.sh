#!/usr/bin/env bash
# Build the immutable build context for jobs/fla_rwkv7 (CPU only, local). DRAFT; nothing is submitted.
#   jobs/fla_rwkv7/build_context.sh <sha> <out_dir>
# Code only: `git archive <sha>` (must be merged into origin/agent-main). No private data ships (the gate uses synthetic
# WKV inputs, the snapshot's own tokenizer and the vendored LICENSE text; truth_v2's train_mix_v2.json is row SHAPES only:
# lengths, spans, option counts, question types -- no ids, no text, no labels). The image is the jobs/rwkv7 one; fla-core
# 0.5.2 is installed by the runner at setup from jobs/fla_rwkv7/requirements-fla.txt (wheel sha256 pinned).
set -euo pipefail
SHA=$1; OUTD=$2
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin agent-main
git merge-base --is-ancestor "$SHA" origin/agent-main || { echo "$SHA is not merged into agent-main" >&2; exit 1; }
rm -rf "$OUTD"; mkdir -p "$OUTD/code/m1" "$OUTD/data"
git archive "$SHA" | tar x -C "$OUTD/code/m1"
cp "$OUTD/code/m1/jobs/rwkv7/Dockerfile" "$OUTD/Dockerfile"
sed -i 's#jobs/rwkv7/rwkv7_job.sh#jobs/fla_rwkv7/fla_job.sh#' "$OUTD/Dockerfile"
grep -q 'jobs/fla_rwkv7/fla_job.sh' "$OUTD/Dockerfile" || { echo "CMD rewrite failed" >&2; exit 1; }
: > "$OUTD/data/.keep"  # the rwkv7 Dockerfile does COPY data/ /data/
python - "$OUTD" "$SHA" <<'PY'
import hashlib, json, sys, pathlib
d, sha = sys.argv[1:3]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c = f"{d}/code/m1"
files = ["jobs/fla_rwkv7/fla_job.sh", "jobs/fla_rwkv7/job_params.json", "jobs/fla_rwkv7/fla_equivalence.py",
         "jobs/fla_rwkv7/requirements-fla.txt", "heads/rwkv7_wkv.py", "jobs/rwkv7/snapshot.sha256",
         "jobs/fla_rwkv7/job_params_truth_v2.json", "jobs/fla_rwkv7/train_mix.py", "jobs/fla_rwkv7/train_mix_v2.json",
         "train/trainer.py", "train/run_cell.py"]
man = {"job": "fla_rwkv7", "m1_sha": sha, "dockerfile_sha256": h(f"{d}/Dockerfile"),
       "files_sha256": {f: h(f"{c}/{f}") for f in files}, "data_sha256": {}}
json.dump(man, open(f"{d}/CONTEXT_MANIFEST.json", "w"), indent=1)
print(json.dumps(man, indent=1))
PY
tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -C "$OUTD" -c . | zstd -19 -o "$OUTD.tar.zst"
sha256sum "$OUTD.tar.zst"; stat -c %s "$OUTD.tar.zst"
