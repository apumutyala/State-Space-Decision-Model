#!/usr/bin/env bash
# Build the immutable build context for jobs/q9_feature_cache (CPU only, local). Nothing is submitted here.
#   jobs/q9_feature_cache/build_context.sh <stock|extra> <sha> <out_dir> [<gdn_converted.pt>]
# (Equivalent to `python -m jobs.common.launch q9_feature_cache --mode <m>` once the launcher lands: the job_params
# "context" block lists the same files and pins.)
# Code: `git archive <sha>` (must be merged into origin/agent-main). Image: jobs/probe9b/Dockerfile (torch 2.13.0 /
# triton 3.7.1 / fla 0.5.2 / transformers 5.17.0, digest-pinned base; the stack job-5cmat measured), CMD rewritten.
# Data (PRIVATE: our own SF workspace only, via create_context; delete the context after the job):
#   slices/dev-rows.jsonl.gz   Index DEV rows (sha256 pinned in jobs/q9_feature_cache/data.sha256); the job reads only
#                              the SELECT and HOLDOUT run ids; TEST is never shipped
#   m2-150k/                   PR #44 training set (records + manifest + content map, pinned by the pointer)
#   suite-0.2/                 official 0.2.1 row files for the guard's content index (jobs.m2_pilot.guard_ready)
#   gdn/gdn_converted.pt       extra mode only: job-5cmat art-2bybh member, sha256 pinned in gdn_converted.sha256
set -euo pipefail
MODE=$1; SHA=$2; OUTD=$3; GDN=${4:-}
DATA=${SSDM_DATA:-$HOME/research/CHLabs/ssdm-data}
B150=$DATA/m2/build/sample-150000-seed20261006
case "$MODE" in stock|extra) ;; *) echo "mode must be stock or extra" >&2; exit 2 ;; esac
[ "$MODE" = stock ] || [ -n "$GDN" ] || { echo "extra mode needs the gdn_converted.pt path" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin agent-main
git merge-base --is-ancestor "$SHA" origin/agent-main || { echo "$SHA is not merged into agent-main" >&2; exit 1; }
rm -rf "$OUTD"; mkdir -p "$OUTD/code/m1" "$OUTD/data/m2-150k" "$OUTD/data/suite-0.2" "$OUTD/data/slices"
git -c core.autocrlf=false archive "$SHA" | tar x -C "$OUTD/code/m1"
cp "$OUTD/code/m1/jobs/probe9b/Dockerfile" "$OUTD/Dockerfile"
sed -i 's#jobs/probe9b/probe_job.sh#jobs/q9_feature_cache/cache_job.sh#' "$OUTD/Dockerfile"
grep -q 'jobs/q9_feature_cache/cache_job.sh' "$OUTD/Dockerfile" || { echo "CMD rewrite failed" >&2; exit 1; }
cp "$DATA/slices/dev-rows.jsonl.gz" "$OUTD/data/slices/"
cp "$B150/records.jsonl.gz" "$B150/sample-150000-seed20261006.manifest.jsonl.gz" \
   "$B150/sample-150000-seed20261006.content.jsonl.gz" "$OUTD/data/m2-150k/"
( cd "$OUTD/code/m1" && python -X utf8 -m train.train_set verify --config jobs/q9_feature_cache/cache_config.json \
    --train-dir "$OUTD/data/m2-150k" ) || { echo "150k set != pointer" >&2; exit 1; }
cp "$DATA/suite-0.2/selected-rows.jsonl.gz" "$DATA/suite-0.2/added-rows.jsonl.gz" "$DATA/suite-0.2/excluded-questions.json" \
  "$OUTD/data/suite-0.2/"
( cd "$OUTD/data" && sha256sum -c "$OUTD/code/m1/jobs/q9_feature_cache/data.sha256" )  # every shipped data file
if [ "$MODE" = extra ]; then
  mkdir -p "$OUTD/data/gdn"
  cp "$GDN" "$OUTD/data/gdn/gdn_converted.pt"
  ( cd "$OUTD/data" && sha256sum -c "$OUTD/code/m1/jobs/q9_feature_cache/gdn_converted.sha256" )
fi
# local dry runs on exactly the files that ship (same exit codes as on the box)
( cd "$OUTD/code/m1" && python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir "$OUTD/data/suite-0.2" \
    --out-dir "$OUTD/guard-local" ) || { echo "guard not ready on the shipped suite" >&2; exit 1; }
( cd "$OUTD/code/m1" && python -X utf8 -m train.train_set heldout --config jobs/q9_feature_cache/cache_config.json \
    --train-dir "$OUTD/data/m2-150k" ) || { echo "held-out hits != pinned exclusions" >&2; exit 1; }
( cd "$OUTD/code/m1" && python -m jobs.q9_feature_cache.preflight --mode "$MODE" > /dev/null ) \
  || { echo "preflight fails on the shipped params" >&2; exit 1; }
python - "$OUTD" "$SHA" "$MODE" <<'PY'
import hashlib, json, sys, pathlib
d, sha, mode = sys.argv[1:4]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c = f"{d}/code/m1"
files = ["jobs/q9_feature_cache/cache_job.sh", f"jobs/q9_feature_cache/job_params_{mode}.json",
         "jobs/q9_feature_cache/cache_config.json", "jobs/q9_feature_cache/extract.py", "jobs/q9_feature_cache/cache.py",
         "jobs/q9_feature_cache/dev_holdout.json", "jobs/common/dev_subset_m1.json", "train/proxy_rule.py",
         "jobs/m2_pilot/guard_pin.json", "data/manifests/sample-150000-seed20261006.pointer.json",
         "train/cohorts/m2-dev-cohort-v2.jsonl.gz"]
man = {"job": "q9_feature_cache", "mode": mode, "sha": sha, "dockerfile_sha256": h(f"{d}/Dockerfile"),
       "files_sha256": {f: h(f"{c}/{f}") for f in files},
       "data_sha256": {str(p.relative_to(d)): h(p) for p in sorted(pathlib.Path(d, "data").rglob("*")) if p.is_file()},
       "guard_local": json.load(open(f"{d}/guard-local/readiness.json"))["observed"]}
json.dump(man, open(f"{d}/CONTEXT_MANIFEST.json", "w"), indent=1)
print(json.dumps(man, indent=1))
PY
rm -rf "$OUTD/guard-local"
tar --sort=name --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -C "$OUTD" -c . | zstd -19 -o "$OUTD.tar.zst"
sha256sum "$OUTD.tar.zst"; stat -c %s "$OUTD.tar.zst"
