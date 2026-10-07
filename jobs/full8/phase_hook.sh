#!/usr/bin/env bash
# PHASE_HOOK of convert.pipeline in jobs/full8 (called as `<hook> <results key>` after every results.json write).
# After a DURABLE phase lands in $FULL8_CK (stage1 -> gdn/stage1.pt; progressive -> gdn/prog_<n>.pt, written before
# results.json; importance; throughput) it declares a platform checkpoint (POST /v1/checkpoint), so a preempted or
# restarted attempt (resume "checkpoint") restores the slot and convert.pipeline resumes after the last finished layer.
# Best effort and never fatal: a 429 (one capture per 300 s) or a network error leaves the previous slot standing; the
# pipeline ignores the hook's exit code. Needs FULL8_OUT, FULL8_META (exported by full8_job.sh).
k=${1:-}
case "$k" in stage1|importance|progressive|throughput) ;; *) exit 0 ;; esac
code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Metadata-Flavor: givemeanode' \
  -H 'Content-Type: application/json' -d '{}' "${FULL8_META}/v1/checkpoint" 2> /dev/null) || code=err
echo "{\"key\": \"$k\", \"t\": \"$(date -u +%FT%TZ)\", \"http\": \"$code\"}" >> "${FULL8_OUT}/checkpoints.jsonl"
exit 0
