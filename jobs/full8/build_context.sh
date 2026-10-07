#!/usr/bin/env bash
# Build the immutable build context for jobs/full8 (CPU only, local). Spec only; nothing is submitted.
#   jobs/full8/build_context.sh <m1_sha> <out_dir>
# Thin wrapper over jobs/common/build_context.sh (git archive of a SHA merged into origin/agent-main, the PRIVATE DEV
# rows, CONTEXT_MANIFEST.json with the full8 runner / params / config / rule / decide / export / hook hashes, and a
# reproducible tar.zst). Before calling it, the measured-rate preflight and the CPU tests must pass on the same tree:
#   python -m jobs.full8.preflight && python -m pytest -q tests/test_full8.py tests/test_probe9b.py tests/test_m1_next_jobs.py
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <m1_sha> <out_dir>" >&2; exit 2; }
exec bash "$(git rev-parse --show-toplevel)/jobs/common/build_context.sh" full8 "$1" "$2"
