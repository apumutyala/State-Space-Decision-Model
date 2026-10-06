#!/bin/bash
# Rebuild the Decision Index 0.2/0.2.1 suite (minus gated HLE) inside WSL.
set -e
cd ~/ssdm-data
mkdir -p work logs
ONLY=$(.venv/bin/python -I -c 'from decision_index.suite.build import rebuild, adapters_added as a; print(" ".join(str(n) for n in sorted(set(rebuild.BUILDERS))+list(a.ORDER) if n!=45))')
echo "$ONLY" > logs/only.txt
export HF_HUB_DISABLE_XET=1
.venv/bin/python -I -m decision_index suite rebuild --work work --only $ONLY > logs/rebuild.log 2>&1
echo "EXIT $?" >> logs/rebuild.log
