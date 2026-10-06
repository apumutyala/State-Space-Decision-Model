#!/usr/bin/env bash
# Build the immutable build context for an m1 batch job (CPU only, local). Usage:
#   jobs/common/build_context.sh <probe9b|rwkv7> <m1_sha> <out_dir>
# Code: `git archive <m1_sha>` (the SHA must be merged into origin/agent-main). Data (PRIVATE; the context goes only to
# our own SF workspace via create_context, never to a public place; delete the context after the jobs end):
#   probe9b, rwkv7: DEV rows slices/dev-rows.jsonl.gz (sha256 pinned in jobs/clef_baseline/data.sha256)
#   rwkv7:          + the m2 pilot records m2/records.jsonl.gz (sha256 pinned in jobs/rwkv7/records.sha256)
# Writes CONTEXT_MANIFEST.json (SHA + sha256 of Dockerfile, runner, params, configs, data) and a reproducible tar.zst.
set -euo pipefail
JOB=$1; M1=$2; OUTD=$3
DATA=${SSDM_DATA:-$HOME/research/CHLabs/ssdm-data}
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin agent-main
git merge-base --is-ancestor "$M1" origin/agent-main || { echo "$M1 is not merged into agent-main" >&2; exit 1; }
case "$JOB" in probe9b|rwkv7) ;; *) echo "job must be probe9b or rwkv7" >&2; exit 2 ;; esac
rm -rf "$OUTD"; mkdir -p "$OUTD/code/m1" "$OUTD/data/slices"
git archive "$M1" | tar x -C "$OUTD/code/m1"
cp "$OUTD/code/m1/jobs/$JOB/Dockerfile" "$OUTD/Dockerfile"
cp "$DATA/slices/dev-rows.jsonl.gz" "$OUTD/data/slices/"
[ "$(sha256sum "$OUTD/data/slices/dev-rows.jsonl.gz" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$OUTD/code/m1/jobs/clef_baseline/data.sha256")" ] || { echo "DEV rows sha256 mismatch" >&2; exit 1; }  # data.sha256 names the in-image path /data/slices/...
if [ "$JOB" = rwkv7 ]; then
  mkdir -p "$OUTD/data/m2"
  cp "$DATA/m2/build/sample-20000-seed20261006/records.jsonl.gz" "$OUTD/data/m2/records.jsonl.gz"
  ( cd "$OUTD/data" && sha256sum -c "$OUTD/code/m1/jobs/rwkv7/records.sha256" )
fi
python - "$OUTD" "$M1" "$JOB" <<'PY'
import hashlib, json, sys, pathlib
d, m1, job = sys.argv[1:4]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c = f"{d}/code/m1"
files = {"probe9b": ["jobs/probe9b/probe_job.sh", "jobs/probe9b/job_params.json", "configs/probe9b_h100.json",
                     "configs/probe9b_h100_lean.json", "jobs/common/dev_subset_m1.json", "jobs/probe9b/preflight.py",
                     "scripts/decide_probe9b.py", "convert/pipeline.py", "convert/align.py"],
         "rwkv7": ["jobs/rwkv7/rwkv7_job.sh", "jobs/rwkv7/job_params_readout.json", "jobs/rwkv7/job_params_lora.json",
                   "jobs/rwkv7/pilot_cell.json", "jobs/rwkv7/snapshot.sha256", "jobs/common/dev_subset_m1.json"]}[job]
man = {"job": job, "m1_sha": m1, "dockerfile_sha256": h(f"{d}/Dockerfile"),
       "files_sha256": {f: h(f"{c}/{f}") for f in files},
       "data_sha256": {str(p.relative_to(d)): h(p) for p in sorted(pathlib.Path(d, "data").rglob("*")) if p.is_file()}}
json.dump(man, open(f"{d}/CONTEXT_MANIFEST.json", "w"), indent=1)
print(json.dumps(man, indent=1))
PY
tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -C "$OUTD" -c . | zstd -19 -o "$OUTD.tar.zst"
sha256sum "$OUTD.tar.zst"; stat -c %s "$OUTD.tar.zst"
