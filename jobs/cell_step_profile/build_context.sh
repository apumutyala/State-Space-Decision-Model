#!/usr/bin/env bash
# Build the immutable build context for jobs/cell_step_profile (CPU only, local). DRAFT; nothing is submitted.
#   jobs/cell_step_profile/build_context.sh <sha> <out_dir>
# Code only: `git archive <sha>` (must be merged into origin/agent-main). No private data ships: the step runs on
# jobs/fla_rwkv7/train_mix_v2.json, row SHAPES only (lengths, spans, option counts, question types; no ids, text or
# labels), with token ids drawn from a fixed generator. Image = the jobs/rwkv7 one (torch 2.13.0, transformers 5.17.0,
# pytest); no extra packages (the WKV variants are pure PyTorch).
set -euo pipefail
SHA=$1; OUTD=$2
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin agent-main
git merge-base --is-ancestor "$SHA" origin/agent-main || { echo "$SHA is not merged into agent-main" >&2; exit 1; }
rm -rf "$OUTD"; mkdir -p "$OUTD/code/m1" "$OUTD/data"
git archive "$SHA" | tar x -C "$OUTD/code/m1"
cp "$OUTD/code/m1/jobs/rwkv7/Dockerfile" "$OUTD/Dockerfile"
sed -i 's#jobs/rwkv7/rwkv7_job.sh#jobs/cell_step_profile/cell_step_job.sh#' "$OUTD/Dockerfile"
grep -q 'jobs/cell_step_profile/cell_step_job.sh' "$OUTD/Dockerfile" || { echo "CMD rewrite failed" >&2; exit 1; }
: > "$OUTD/data/.keep"  # the rwkv7 Dockerfile does COPY data/ /data/
python - "$OUTD" "$SHA" <<'PY'
import hashlib, json, sys, pathlib
d, sha = sys.argv[1:3]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c = f"{d}/code/m1"
files = ["jobs/cell_step_profile/cell_step_job.sh", "jobs/cell_step_profile/job_params.json",
         "jobs/cell_step_profile/cell_step.py", "jobs/cell_step_profile/rule.py", "jobs/fla_rwkv7/train_mix.py",
         "jobs/fla_rwkv7/train_mix_v2.json", "jobs/rwkv7/snapshot.sha256", "heads/act_ckpt.py", "heads/rwkv7_lean.py",
         "heads/rwkv7_wkv.py", "heads/backbone.py", "heads/features.py", "train/trainer.py", "train/run_cell.py",
         "tests/test_cell_perf.py", "tests/test_cell_step_rule.py"]
man = {"job": "cell_step_profile", "m1_sha": sha, "dockerfile_sha256": h(f"{d}/Dockerfile"),
       "files_sha256": {f: h(f"{c}/{f}") for f in files}, "data_sha256": {}}
json.dump(man, open(f"{d}/CONTEXT_MANIFEST.json", "w"), indent=1)
print(json.dumps(man, indent=1))
PY
tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -C "$OUTD" -c . | zstd -19 -o "$OUTD.tar.zst"
sha256sum "$OUTD.tar.zst"; stat -c %s "$OUTD.tar.zst"
