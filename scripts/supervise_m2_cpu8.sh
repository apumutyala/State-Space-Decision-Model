#!/usr/bin/env bash
# Local WSL controller for a freshly reviewed Issue #171 attempt. Never creates a bucket.
# Usage: bash scripts/supervise_m2_cpu8.sh PACKAGE OUT WORKSPACE EXISTING_PRIVATE_BUCKET IMAGE
set -Eeuo pipefail
OUT="${2:-./cpu8-controller-failure}"
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
PHASE=arguments CURRENT_FILE=none NODE_ID=none NODE_STATE=not_created
NODE=none OWNED=0 STOPPED=0 STOP_ATTEMPTS=0 STOP_RESULT=not_needed RUN_RC=not_run EXPORT_RC=not_run IMPORT_OUTCOME=not_started
START=$SECONDS
STAGE=""
AUTOR="${AUTOR:-/root/.local/bin/autor}"
JSON_HELPER="$(dirname "$(realpath "$0")")/m2_cpu8_json.py"
REMOTE=/home/dev/m2-cpu8-validation
TOTAL_SECONDS=$((175 * 60))
SETUP_SECONDS=$((20 * 60))

stop_node() {
  (( OWNED == 1 && STOPPED == 0 )) || return 0
  (( STOP_ATTEMPTS < 2 )) || return 70
  STOP_ATTEMPTS=$((STOP_ATTEMPTS + 1))
  STOP_RESULT=failed
  local stop_rc=0
  timeout --kill-after=5s 55s "${AUTOR_CMD[@]}" node stop "$NODE" --json > /dev/null 2>&1 || stop_rc=$?
  if (( stop_rc == 0 )); then
    STOPPED=1 NODE_STATE=stopped STOP_RESULT=stopped
  else
    NODE_STATE=unknown
    return "$stop_rc"
  fi
}
on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if [[ -f "$OUT/transfer.stdout.tmp" && -f "$OUT/transfer.stderr.tmp" ]]; then
    python3 "$JSON_HELPER" event "$OUT/transfer-events.json" "$PHASE" "$CURRENT_FILE" "$rc" \
      "$OUT/transfer.stdout.tmp" "$OUT/transfer.stderr.tmp"
  fi
  if ! stop_node; then
    # One bounded cleanup retry only; never retry the workload or an upload.
    stop_node
    (( STOPPED == 1 )) || rc=70
  fi
  python3 "$JSON_HELPER" receipt "$OUT/controller.json" "$PHASE" "$CURRENT_FILE" "$rc" \
    "$NODE" "$NODE_ID" "$NODE_STATE" "$STOP_RESULT" "$RUN_RC" "$EXPORT_RC" "$((SECONDS - START))" "$IMPORT_OUTCOME"
  local receipt_rc=$?
  rm -f -- "$OUT/node.tmp.json" "$OUT/node.checked.txt" "$OUT/transfer.stdout.tmp" "$OUT/transfer.stderr.tmp"
  [[ -z "$STAGE" ]] || rm -f -- "$STAGE"/*
  [[ -z "$STAGE" ]] || rmdir -- "$STAGE"
  (( receipt_rc == 0 )) || rc=70
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
remaining() { echo $((TOTAL_SECONDS - (SECONDS - START) - 135)); }
bounded() {
  local ceiling="$1"; shift
  local left; left=$(remaining)
  (( left > 0 )) || return 124
  (( ceiling < left )) && left="$ceiling"
  timeout --kill-after=10s "${left}s" "$@"
}
setup_bounded() {
  local left=$((SETUP_SECONDS - (SECONDS - START)))
  (( left > 0 )) || return 124
  bounded "$left" "$@"
}
transfer() {
  local ceiling="$1"; shift
  local transfer_rc=0
  bounded "$ceiling" "$@" > "$OUT/transfer.stdout.tmp" 2> "$OUT/transfer.stderr.tmp" || transfer_rc=$?
  if ! python3 "$JSON_HELPER" event "$OUT/transfer-events.json" "$PHASE" "$CURRENT_FILE" "$transfer_rc" \
    "$OUT/transfer.stdout.tmp" "$OUT/transfer.stderr.tmp"; then return 70; fi
  rm -f -- "$OUT/transfer.stdout.tmp" "$OUT/transfer.stderr.tmp"
  return "$transfer_rc"
}
setup_transfer() {
  local left=$((SETUP_SECONDS - (SECONDS - START)))
  (( left > 0 )) || return 124
  transfer "$left" "$@"
}

(( $# == 5 )) || exit 64
PKG=$(realpath "$1")
WORKSPACE="$3" BUCKET="$4" IMAGE="$5"
AUTOR_CMD=("$AUTOR" --workspace "$WORKSPACE")
[[ "$WORKSPACE" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 64
[[ "$BUCKET" =~ ^[a-z0-9][a-z0-9-]{1,62}$ && "$IMAGE" =~ ^[a-zA-Z0-9._-]+$ ]] || exit 64
# A fresh unpredictable name/prefix prevents cleanup targeting the stopped incident node.
NODE="ssdm-codex-m2cpu8-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
PREFIX="m2-cpu8/$NODE"
LIST_SHA=$(python3 "$JSON_HELPER" list-sha)
printf '%s\n' "$BUCKET/$PREFIX" > "$OUT/storage-prefix.txt"
PHASE=package_verify CURRENT_FILE=package.sha256
python3 "$JSON_HELPER" package "$PKG" "$OUT/staging.json"
STAGE=$(mktemp -d "$OUT/stage.XXXXXXXX")
while IFS= read -r file; do
  CURRENT_FILE="$file"
  cp -- "$PKG/$file" "$STAGE/$file"
done < <(python3 "$JSON_HELPER" names "$OUT/staging.json")
# Verify the copied bytes too; no mutable package content crosses the upload boundary.
python3 "$JSON_HELPER" package "$STAGE" "$OUT/staging.json"
PHASE=storage_upload
while IFS= read -r file; do
  CURRENT_FILE="$file"
  setup_transfer "${AUTOR_CMD[@]}" storage cp "$STAGE/$file" "$BUCKET/$PREFIX/input/$file" --concurrency 1 --json
  python3 "$JSON_HELPER" uploaded "$OUT/staging.json" "$file" "$BUCKET/$PREFIX/input/$file"
done < <(python3 "$JSON_HELPER" names "$OUT/staging.json")

PHASE=node_create CURRENT_FILE=none
# From this point any create/parse/inspect/import failure stops only this fresh owned name.
OWNED=1 NODE_STATE=create_requested
setup_bounded "${AUTOR_CMD[@]}" node create --chip cpu-8 --name "$NODE" --image "$IMAGE" \
  --max-wait 10m --wait --json > "$OUT/node.tmp.json"
python3 "$JSON_HELPER" identity "$OUT/node.tmp.json" "$NODE" > "$OUT/node.checked.txt"
read -r NODE_ID NODE_STATE < "$OUT/node.checked.txt"
PHASE=node_validate
setup_bounded "${AUTOR_CMD[@]}" node get "$NODE" --json > "$OUT/node.tmp.json"
python3 "$JSON_HELPER" node "$OUT/node.tmp.json" "$NODE" > "$OUT/node.checked.txt"
read -r NODE_ID NODE_STATE < "$OUT/node.checked.txt"
rm -f -- "$OUT/node.tmp.json" "$OUT/node.checked.txt"
PHASE=storage_import CURRENT_FILE=package_directory IMPORT_OUTCOME=unknown
setup_transfer "${AUTOR_CMD[@]}" import "$NODE" --connection "storage/$BUCKET" --source "$PREFIX/input/" \
  --dest "$REMOTE/input" --wait --json
IMPORT_OUTCOME=completed
PHASE=remote_hash CURRENT_FILE=package.sha256
setup_bounded "${AUTOR_CMD[@]}" run "$NODE" -- bash -lc "cd '$REMOTE/input' && printf '%s  package.sha256\\n' '$LIST_SHA' | sha256sum -c - && sha256sum -c package.sha256" > /dev/null
python3 "$JSON_HELPER" verified "$OUT/staging.json"
PHASE=park
stop_node
PHASE=register_run
# Registration wakes the parked node. Preserve the frozen server and subreaper deadlines.
STOPPED=0 STOP_ATTEMPTS=0 NODE_STATE=wake_requested
bounded 120 "${AUTOR_CMD[@]}" run "$NODE" -d --on-wake --max-duration 145m --max-wait 10m -- \
  bash -lc "cd '$REMOTE/input' && exec python3 ./run_deadline.py --seconds 8610 --kill-grace-seconds 30 -- bash ./run_m2_cpu8_tests.sh" > /dev/null
PHASE=workload
RUN_RC=0
bounded 8700 "${AUTOR_CMD[@]}" logs "$NODE" -f > "$OUT/remote.log" 2>&1 || RUN_RC=$?
PHASE=export CURRENT_FILE=launch-output
EXPORT_RC=0
EXPORT_PHASE=export
transfer 600 "${AUTOR_CMD[@]}" export "$NODE" --connection "storage/$BUCKET" --source "$REMOTE/work/launch-output" \
  --dest "$PREFIX/launch-output.tar" --wait --json || EXPORT_RC=$?
stop_node
if (( EXPORT_RC == 0 )); then
  PHASE=download EXPORT_PHASE=download
  transfer 600 "${AUTOR_CMD[@]}" storage cp "$BUCKET/$PREFIX/launch-output.tar" "$OUT/launch-output.tar" \
    --concurrency 1 --json || EXPORT_RC=$?
fi
if (( RUN_RC != 0 )); then PHASE=workload; exit "$RUN_RC"; fi
if (( EXPORT_RC != 0 )); then PHASE="$EXPORT_PHASE"; exit "$EXPORT_RC"; fi
PHASE=complete CURRENT_FILE=none
exit 0
