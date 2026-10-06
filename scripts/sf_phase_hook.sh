#!/usr/bin/env bash
# PHASE_HOOK for convert.pipeline on the paid node: snapshot the run's receipt dir after every results.json phase.
# usage: sf_phase_hook.sh <run-name> <results-key>
ROOT=/home/dev/m1q
tar czf $ROOT/export/$1-$2.tgz -C $ROOT out/$1 2>/dev/null && \
  echo "$(date -u +%FT%TZ) $1-$2 $ROOT/export/$1-$2.tgz" >> $ROOT/export/MANIFEST
