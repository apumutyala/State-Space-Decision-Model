#!/usr/bin/env bash
# Progress-gated heartbeat watchdog for run_job.sh (semantics documented at the top of run_job.sh).
# Needs: OUT, CK, HF_DIR, MAIN_PID, HB_EVERY_S, STALL_S. Optional: WATCH_EXTRA (space-separated extra dirs a runner
# writes to, e.g. its /tmp work dir). Defines hb, progressed, watchdog, start_watchdog.
# find -H follows symlinked START dirs (an HF cache linked into a work dir), never symlinks below them.
hb() { date -u +%FT%TZ > "$OUT/heartbeat"; }
progressed() {  # any file other than the heartbeat / watchdog log modified within STALL_S seconds?
  # shellcheck disable=SC2086  (WATCH_EXTRA is a word list of paths without spaces)
  find -H "$OUT" "$CK" "$HF_DIR" ${WATCH_EXTRA:-} -type f ! -name heartbeat ! -name watchdog.log        -newermt "@$(( $(date +%s) - STALL_S ))" -print -quit 2>/dev/null | grep -q .
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
