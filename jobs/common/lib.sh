#!/usr/bin/env bash
# Shared helpers for the m1 batch runners (jobs/probe9b/probe_job.sh, jobs/rwkv7/rwkv7_job.sh). Same patterns as
# jobs/qual/qual_job.sh: gates recorded as JSON, per-phase export verified by re-reading + double hash into
# MANIFEST.jsonl (also POSTed as a sample), caps read from job_params. Needs: OUT WORK T0 META PARAMS.
# Heartbeat: jobs/clef_baseline/watchdog.sh (progress-gated: no file progress for STALL_S -> heartbeat withheld).
say() { echo "[$(date -u +%FT%TZ) +$(( ($(date +%s) - T0) / 60 ))m] $*"; }
gate() {  # gate <name> <rc>
  GATES=$(python - "$GATES" "$1" "$2" <<'EOF'
import json, sys
g = json.loads(sys.argv[1]); g[sys.argv[2]] = int(sys.argv[3]); print(json.dumps(g))
EOF
)
}
sample() { curl -s -m 5 -X POST -H 'Metadata-Flavor: givemeanode' -H 'Content-Type: application/json' \
  -d "{\"path\": \"$1\", \"kind\": \"json\", \"label\": \"$2\"}" "$META/v1/sample" > /dev/null 2>&1 || true; }
export_phase() {  # tar WORK/out + WORK/logs, verify by re-reading + hashing twice, record in MANIFEST.jsonl
  local ph=$1 f="$OUT/phases/$1.tgz" ok=false sha="" bytes=0 outb inb max=${ARTIFACT_MAX_BYTES:-4500000000}
  mkdir -p "$OUT/phases"
  # artifact size guard: the platform rejects an artifact >= 5 GB (one PUT; D5a's 14 GB export was lost). Refuse to
  # write a phase tgz that could take $OUT past 4.5 GB (uncompressed out+logs as the upper bound) and fail LOUDLY.
  outb=$(du -sb "$OUT" 2>/dev/null | cut -f1); inb=$(du -sbc "$WORK/out" "$WORK/logs" 2>/dev/null | tail -1 | cut -f1)
  if [ "$(( ${outb:-0} + ${inb:-0} ))" -ge "$max" ]; then
    say "ARTIFACT SIZE GUARD: $OUT is ${outb:-0} B + phase $ph up to ${inb:-0} B >= $max B: phase NOT exported (exit 6 at finalize)"
    echo "{\"phase\": \"$ph\", \"file\": null, \"ok\": false, \"size_guard\": {\"out_bytes\": ${outb:-0}, \"phase_bytes_max\": ${inb:-0}, \"max_bytes\": $max}, \"t\": \"$(date -u +%FT%TZ)\"}" >> "$OUT/MANIFEST.jsonl"
    EXPORT_FAIL=1
    return 0
  fi
  if tar czf "$f" -C "$WORK" out logs 2>/dev/null && tar tzf "$f" > /dev/null 2>&1; then
    sha=$(sha256sum "$f" | cut -d' ' -f1); bytes=$(stat -c %s "$f")
    [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ] && [ "$bytes" -gt 0 ] && ok=true
  fi
  echo "{\"phase\": \"$ph\", \"file\": \"phases/$ph.tgz\", \"sha256\": \"$sha\", \"bytes\": $bytes, \"ok\": $ok, \"t\": \"$(date -u +%FT%TZ)\"}" >> "$OUT/MANIFEST.jsonl"
  [ "$ok" = true ] || EXPORT_FAIL=1
  sample "$OUT/MANIFEST.jsonl" "export:$ph"
  say "export $ph ok=$ok sha256=$sha"
}
cap() { python -c "import json,sys;print(json.load(open(sys.argv[1]))['phase_caps_minutes'][sys.argv[2]])" "$PARAMS" "$1"; }
