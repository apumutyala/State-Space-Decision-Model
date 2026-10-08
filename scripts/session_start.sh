#!/usr/bin/env bash
# Session-start digest for both agents (AGENTS.md). Read-only: prints the shared state, the ledger tail, live GPU jobs and
# the balance. Usage: bash scripts/session_start.sh
cd "$(git rev-parse --show-toplevel)" || exit 0
git fetch -q origin agent-main state 2>/dev/null
show() { git show "origin/state:$1" 2>/dev/null || echo "(no $1 on origin/state)"; }
echo "== BOARD (origin/state) =="; show BOARD.md | head -40
echo; echo "== PROGRESS (last 15 lines) =="; show PROGRESS.md | tail -15
echo; echo "== CLAIMS =="; git ls-tree --name-only origin/state claims/ 2>/dev/null | sed 's#claims/##' || true
echo; echo "== LEDGER (last 12 runs) =="; show results/ledger.tsv | tail -12
echo; echo "== agent-main =="; git log --oneline -5 origin/agent-main
AUTOR=(wsl.exe -d bi-ubuntu -- /root/.local/bin/autor --workspace default)
if command -v wsl.exe >/dev/null 2>&1; then
  echo; echo "== GPU JOBS =="; MSYS_NO_PATHCONV=1 "${AUTOR[@]}" job ls 2>/dev/null | head -8
  echo; echo "== BALANCE =="; MSYS_NO_PATHCONV=1 "${AUTOR[@]}" billing 2>/dev/null | grep -E '"available_usd"|"burn_rate' | head -2
fi
exit 0
