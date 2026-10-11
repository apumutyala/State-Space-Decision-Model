#!/usr/bin/env bash
# Evidence lane, CPU only: build the UNSEEN split of B-PREREG-GUAR-FSCRR, a seeded answer-free sample of the Decision
# Index 0.2.1 POOL (scoreable index rows in neither the DEV nor the TEST manifest), with
# receipts/2026-10-11-ev-guar-fscrr/pool_sample.py. Streams the 0.2.1 suite twice; no model, no prediction, no GPU.
# TEST is read only as manifest run ids (membership), never as rows.
#   DRY=1 bash jobs/ev_pool_sample/ev_pool_sample_job.sh   prints the plan and exits 0 (tests).
set -euo pipefail
OUT=${OUT:-/output}
DATA=${DATA:-/data}
CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
W=$OUT/ev_pool_sample
CAP_BUILD_MIN=${CAP_BUILD_MIN:-20}
run() { if [ "${DRY:-0}" = 1 ]; then echo "DRY: $*"; else "$@"; fi; }
say() { echo "[ev_pool_sample $(date -u +%FT%TZ)] $*"; }
man() { [ "${DRY:-0}" = 1 ] || echo "{\"phase\": \"$1\", \"ok\": $2, \"t\": \"$(date -u +%FT%TZ)\"}" >> "$OUT/MANIFEST.jsonl"; }

if [ "${DRY:-0}" != 1 ]; then
  mkdir -p "$W"
  date -u +%FT%TZ > "$OUT/heartbeat"
  ( while true; do date -u +%FT%TZ > "$OUT/heartbeat"; sleep 60; done ) &
  HB=$!
  trap 'kill $HB 2>/dev/null || true' EXIT
  exec > >(tee -a "$W/run.log") 2>&1
fi
cd "$CODE"
export SSDM_DATA=$DATA CUDA_VISIBLE_DEVICES=""
[ -d /opt/decision-index ] && export SSDM_DI_KIT=/opt/decision-index

say "setup: data pins (jobs/ev_pool_sample/data.sha256)"
if [ "${DRY:-0}" = 1 ]; then echo "DRY: sha256sum -c jobs/ev_pool_sample/data.sha256"
else sed "s#  /data/#  $DATA/#" jobs/ev_pool_sample/data.sha256 | sha256sum -c - | tee "$W/data-verify.txt"; fi
man setup true

say "preflight: projected build time vs cap $CAP_BUILD_MIN min"
if [ "${DRY:-0}" != 1 ]; then
  if ! python -m jobs.ev_pool_sample.preflight --cap-min "$CAP_BUILD_MIN" --data "$DATA" | tee "$W/preflight.json"; then
    say "preflight refused"; man preflight false; exit 15
  fi
else echo "DRY: python -m jobs.ev_pool_sample.preflight --cap-min $CAP_BUILD_MIN"; fi
man preflight true

say "build: pool sample"
run python -I -X utf8 receipts/2026-10-11-ev-guar-fscrr/pool_sample.py "$CODE" "$W"
man build true

if [ "${DRY:-0}" != 1 ]; then
  ( cd "$W" && sha256sum pool-sample-rows.jsonl.gz pool-sample-manifest.jsonl.gz pool_sample.json preflight.json \
      data-verify.txt > SHA256SUMS )
  cp "$W/SHA256SUMS" "$OUT/SHA256SUMS"
  man final true
  date -u +%FT%TZ > "$OUT/EV_POOL_SAMPLE_DONE"
fi
say "done"
