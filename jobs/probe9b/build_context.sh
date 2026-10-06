#!/usr/bin/env bash
# Build the immutable build context for jobs/probe9b (CPU only, local). Spec only; nothing is submitted.
#   jobs/probe9b/build_context.sh <m1_sha> <out_dir>
# Thin wrapper over jobs/common/build_context.sh (the one builder for the m1 jobs): `git archive <m1_sha>` (the SHA
# must be merged into origin/agent-main), the PRIVATE DEV rows (sha256 pinned in jobs/clef_baseline/data.sha256; the
# context goes only to our own SF workspace via create_context and is deleted after the job), CONTEXT_MANIFEST.json
# (SHA + sha256 of Dockerfile, runner, params, configs, preflight, decision rule, DEV subset) and a reproducible
# tar.zst. Before calling it, the measured-rate preflight must pass on the same tree (no GPU needed):
#   python -m jobs.probe9b.preflight && python -m pytest -q tests/test_probe9b.py tests/test_m1_next_jobs.py
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <m1_sha> <out_dir>" >&2; exit 2; }
exec bash "$(git rev-parse --show-toplevel)/jobs/common/build_context.sh" probe9b "$1" "$2"
