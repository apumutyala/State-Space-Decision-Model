#!/usr/bin/env bash
# Assemble the job build context (run locally; CPU only). Produces clef-baseline-context.tar.zst + sha256.
# Contents: this branch at HEAD (decide/, safety_eval/, baselines/, third_party/, jobs/), harness/ from
# claude/m0-harness @ f3c7791, and the PRIVATE DEV rows (Decision Index suite is not redistributable:
# the tar goes only to our own SF Compute workspace via create_context, never to a public place).
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
HARNESS_COMMIT=f3c7791bcbe6455effc7cfce019f3f3f99435843
DEV_ROWS="${SSDM_DATA:-$HOME/research/CHLabs/ssdm-data}/slices/dev-rows.jsonl.gz"
WORK=$(mktemp -d)
mkdir -p "$WORK/app" "$WORK/data/slices"
git -C "$REPO" archive HEAD decide safety_eval baselines third_party jobs | tar -x -C "$WORK/app"
git -C "$REPO" archive "$HARNESS_COMMIT" harness | tar -x -C "$WORK/app"
cp "$DEV_ROWS" "$WORK/data/slices/dev-rows.jsonl.gz"
(cd "$WORK/data" && sha256sum -c "$WORK/app/jobs/clef_baseline/data.sha256" 2>/dev/null) || \
  (echo "$(sha256sum "$WORK/data/slices/dev-rows.jsonl.gz")"; grep -q d1202ab8 <(sha256sum "$WORK/data/slices/dev-rows.jsonl.gz"))
cp "$WORK/app/jobs/clef_baseline/Dockerfile" "$WORK/Dockerfile"
printf '{"branch_commit":"%s","harness_commit":"%s","dev_rows_sha256":"%s"}\n' "$(git -C "$REPO" rev-parse HEAD)" \
  "$HARNESS_COMMIT" "$(sha256sum "$DEV_ROWS" | cut -d' ' -f1)" > "$WORK/app/CONTEXT_MANIFEST.json"
tar --sort=name --mtime='2026-10-06 00:00Z' --owner=0 --group=0 --numeric-owner -C "$WORK" -c . | zstd -19 -o clef-baseline-context.tar.zst
sha256sum clef-baseline-context.tar.zst; stat -c %s clef-baseline-context.tar.zst
