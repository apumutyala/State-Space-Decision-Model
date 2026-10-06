#!/usr/bin/env bash
# Progress-gated heartbeat watchdog for run_job.sh (semantics documented at the top of run_job.sh).
# Needs: OUT, CK, HF_DIR, MAIN_PID, HB_EVERY_S, STALL_S. Defines hb, progressed, watchdog, start_watchdog.
hb() { date -u +%FT%TZ > "$OUT/heartbeat"; }
progressed() {  # any file other than the heartbeat / watchdog log modified within STALL_S seconds?
  find "$OUT" "$CK" "$HF_DIR" -type f ! -name heartbeat ! -name watchdog.log        -newermt "@$(( $(date +%s) - STALL_S ))" -print -quit 2>/dev/null | grep -q .
}
watchdog() {
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    if progressed; then
      hb
    else
      echo "$(date -u +%FT%TZ) watchdog: no progress for ${STALL_S}s; heartbeat withheld" >> "$OUT/watchdog.log"
    fi
    sleep "$HB_EVERY_S"
  done
}
start_watchdog() {
  watchdog &
  WD_PID=$!
  trap 'kill "$WD_PID" 2>/dev/null' EXIT
}
