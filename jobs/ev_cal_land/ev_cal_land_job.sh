#!/usr/bin/env bash
# Evidence lane, CPU only: the full-DEV Decision Index of the D19 CALIBRATED variant (B-PREREG-D19-TEMP, T = 0.806211
# applied independently by the evidence rescore to job-y6ste+uq5p5's merged rows) paired vs Clef 27B and Clef-flash
# through analysis/land_fulldev.py (frozen scorer + case-cluster bootstrap B = 2000 seed 0), the UNTEMPERED rows as an
# identity check (must reproduce 49.49), and per-area ECE (jobs/ev_cal_land/area_ece.py). No model, no GPU work.
#   DRY=1 bash jobs/ev_cal_land/ev_cal_land_job.sh   prints the plan and exits 0 (tests).
set -euo pipefail
OUT=${OUT:-/output}
DATA=${DATA:-/data}
CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
W=$OUT/ev_cal_land
CAP_LAND_MIN=${CAP_LAND_MIN:-22}
CANDS="calibrated untempered"
run() { if [ "${DRY:-0}" = 1 ]; then echo "DRY: $*"; else "$@"; fi; }
say() { echo "[ev_cal_land $(date -u +%FT%TZ)] $*"; }
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

say "setup: data pins (jobs/ev_cal_land/data.sha256)"
if [ "${DRY:-0}" = 1 ]; then echo "DRY: sha256sum -c jobs/ev_cal_land/data.sha256"
else sed "s#  /data/#  $DATA/#" jobs/ev_cal_land/data.sha256 | sha256sum -c - | tee "$W/data-verify.txt"; fi
man setup true

say "preflight: projected landing time vs cap $CAP_LAND_MIN min"
if [ "${DRY:-0}" != 1 ]; then
  if ! python -m jobs.ev_cal_land.preflight --cap-min "$CAP_LAND_MIN" --passes 3 --data "$DATA" | tee "$W/preflight.json"; then
    say "preflight refused"; man preflight false; exit 15
  fi
else echo "DRY: python -m jobs.ev_cal_land.preflight --cap-min $CAP_LAND_MIN --passes 3"; fi
man preflight true

for c in $CANDS; do
  say "land: $c"
  run python -I -X utf8 analysis/land_fulldev.py --candidate "$DATA/cand/$c.results.jsonl" \
    --clef "$DATA/clef/clef27b.results.jsonl" --clef-flash "$DATA/clef/clef_flash.results.jsonl" \
    --job "job-y6ste+job-uq5p5 ($c)" --artifact "art-utxjp+art-6qppy" --out "$W/land_$c.json"
  man "land_$c" true
done

say "area ECE"
run python -I -X utf8 jobs/ev_cal_land/area_ece.py --out "$W/area_ece.json" \
  "calibrated=$DATA/cand/calibrated.results.jsonl" "untempered=$DATA/cand/untempered.results.jsonl" \
  "clef27b=$DATA/clef/clef27b.results.jsonl" "clef_flash=$DATA/clef/clef_flash.results.jsonl"
man area_ece true

if [ "${DRY:-0}" != 1 ]; then
  ( cd "$W" && sha256sum land_*.json area_ece.json preflight.json data-verify.txt > SHA256SUMS )
  cp "$W/SHA256SUMS" "$OUT/SHA256SUMS"
  man final true
  date -u +%FT%TZ > "$OUT/EV_CAL_LAND_DONE"
fi
say "done"
