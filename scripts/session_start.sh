#!/usr/bin/env bash
# Session digest (AGENTS.md). Read-only for the repo, idempotent, ALWAYS exits 0, never blocks a session.
#   bash scripts/session_start.sh          compact digest (<= 8 KB). The SessionStart hook in .claude/settings.json runs
#                                          it on startup / resume / clear / compact / fork and adds stdout to context.
#   bash scripts/session_start.sh --full   the old verbose dump (BOARD head, PROGRESS 15, ledger 12, whole INBOX).
# Agent identity (filters INBOX): $SSDM_AGENT (lead | evidence | agent-a | agent-b | agent-c), else the worktree
#   directory name, else the branch name. INBOX lines that mention @<id>, @agent-<id> or @all are shown.
# Few processes on purpose: process spawns cost ~1 s each on the loaded shared laptop. One `git archive` of the state
#   branch, one awk pass, and two bounded autor calls in the background (their output goes to files, never to our
#   stdout, so a hung child cannot hold the hook's pipe open).
# Test knobs: SSDM_WSL (default wsl.exe), SSDM_AUTOR_TIMEOUT (s, 15), SSDM_FETCH_TIMEOUT (s, 10),
#   SSDM_DIGEST_MAX_BYTES (8000).
FULL=0; [ "${1:-}" = "--full" ] && FULL=1
MAXB=${SSDM_DIGEST_MAX_BYTES:-8000}
WSL=${SSDM_WSL:-wsl.exe}
export LC_ALL=C.UTF-8 GIT_TERMINAL_PROMPT=0
tmo() { if command -v timeout >/dev/null 2>&1; then timeout "$@"; else shift; "$@"; fi; }

src=${BASH_SOURCE[0]//\\//}   # works when the hook passes a C:\... path
cd "${src%/*}/.." 2>/dev/null || true

# A core.worktree in the shared .git/config breaks git in every worktree (2026-10-08 incident). Warn only.
cw=$(git --work-tree="$PWD" config --get core.worktree 2>/dev/null)
if [ -n "$cw" ]; then
  echo "!!!! WARNING: shared .git/config has core.worktree=$cw -- this breaks git in EVERY worktree."
  echo "!!!! Fix: git --work-tree=. config --unset core.worktree. Never export GIT_DIR/GIT_WORK_TREE."
fi
top=$(git rev-parse --show-toplevel 2>/dev/null) && cd "$top" 2>/dev/null

T=${TMPDIR:-/tmp}/ssdm-digest-$$-$RANDOM
mkdir -p "$T/s" 2>/dev/null || exit 0
trap 'rm -rf "$T" 2>/dev/null' EXIT

if command -v "$WSL" >/dev/null 2>&1; then
  AUTOR=("$WSL" -d bi-ubuntu -- /root/.local/bin/autor --workspace default)
  MSYS_NO_PATHCONV=1 tmo "${SSDM_AUTOR_TIMEOUT:-15}" "${AUTOR[@]}" job ls >"$T/jobs" 2>/dev/null </dev/null &
  MSYS_NO_PATHCONV=1 tmo "${SSDM_AUTOR_TIMEOUT:-15}" "${AUTOR[@]}" billing >"$T/bill" 2>/dev/null </dev/null &
fi
tmo "${SSDM_FETCH_TIMEOUT:-10}" git fetch -q origin agent-main state >/dev/null 2>&1 </dev/null
git archive origin/state 2>/dev/null | tar -x -C "$T/s" 2>/dev/null

if [ "$FULL" = 1 ]; then
  S=$T/s
  echo "== BOARD (origin/state) =="; head -40 "$S/BOARD.md" 2>/dev/null
  echo; echo "== PROGRESS (last 15 lines) =="; tail -15 "$S/PROGRESS.md" 2>/dev/null
  echo; echo "== CLAIMS =="; ls "$S/claims" 2>/dev/null
  echo; echo "== LEDGER (last 12 runs) =="; tail -12 "$S/results/ledger.tsv" 2>/dev/null
  echo; echo "== INBOX =="; cat "$S/INBOX.md" 2>/dev/null
  echo; echo "== agent-main =="; git log --oneline -5 origin/agent-main 2>/dev/null
  wait
  echo; echo "== GPU JOBS =="; head -12 "$T/jobs" 2>/dev/null
  echo; echo "== BALANCE =="; grep -E '"available_usd"|"burn_rate|"runway_minutes"' "$T/bill" 2>/dev/null | head -3
  exit 0
fi

me=${SSDM_AGENT:-}
if [ -z "$me" ]; then
  br=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
  for cand in "${PWD##*/}" "${br##*/}"; do
    case "$cand" in agent-a|agent-b|agent-c|lead|evidence|lead-*|evidence-*) me=$cand; break;; esac
  done
fi
[ -n "$br" ] || br=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
targets=$T/s/TARGETS.tsv; [ -f "$targets" ] || targets=TARGETS.tsv
: >>"$T/jobs"; : >>"$T/bill"
wait  # the background autor calls; each is bounded by its own timeout


S=$T/s
files=()  # only existing inputs, no forks: awk must never fall back to stdin or die on a missing file
for f in "$targets" "$S/BOARD.md" "$S"/claims/*.lock "$S/results/ledger.tsv" "$S/PROGRESS.md" "$S/DEAD_ENDS.md" \
         "$S/INBOX.md" LESSONS.md "$T/jobs" "$T/bill"; do [ -f "$f" ] && files+=("$f"); done
# Sections are ordered by value, so a size overflow drops the least useful tail (DEAD_ENDS, LESSONS) first.
awk -v me="$me" -v br="$br" -v now="$(date -u +%Y-%m-%dT%H:%MZ)" '
function cut(s, n) { gsub(/\t/, " ", s); return length(s) > n ? substr(s, 1, n - 1) "~" : s }
function sec(t) { print "## " t }
function tail0(n, k) { return n > k ? n - k + 1 : 1 }
BEGIN { id = tolower(me); sub(/^agent-/, "", id); if (id ~ /^lead/) id = "lead"; if (id ~ /^evidence/) id = "evidence" }
{ sub(/\r$/, "") }  # a CRLF checkout (core.autocrlf) must not leak CRs into the digest
FILENAME ~ /TARGETS.tsv$/ {
  if ($0 ~ /^# objective: /) { obj = substr($0, 14); next }
  if ($0 ~ /^#/ || $0 ~ /^rank\t/) next
  n = split($0, f, "\t"); if (n >= 5) tg[++ntg] = "- " f[1] " " f[2] " | " f[3] " " f[4] " | ours: " f[5]
  next
}
FILENAME ~ /BOARD.md$/ {
  if ($0 ~ /^\| *[0-9]+ *\|/) { s = $0; gsub(/ *\| */, " | ", s); sub(/^ \| /, "", s); sub(/ \| $/, "", s); bd[++nbd] = s }
  if ($0 ~ /^Proposals:/) inprop = 1; else if (inprop && $0 ~ /^- /) nprop++
  next
}
FILENAME ~ /\/claims\/.*\.lock$/ {
  c = FILENAME; sub(/.*\/claims\//, "", c); sub(/\.lock$/, "", c)
  if (!(c in seen)) { seen[c] = 1; cl[++ncl] = c }
  if ($0 ~ /^owner:/) { o[c] = $0; sub(/^owner: */, "", o[c]) }
  if ($0 ~ /^job:/) { j[c] = $0; sub(/^job: */, "", j[c]); sub(/ .*/, "", j[c]) }
  if ($0 ~ /^time_limit:/) { l[c] = $0; sub(/^time_limit: */, "", l[c]) }
  next
}
FILENAME ~ /ledger.tsv$/ { if (FNR > 1 && $0 !~ /^[ \t]*$/) led[++nled] = $0; next }
FILENAME ~ /PROGRESS.md$/ { if ($0 !~ /^[ \t]*$/) pr[++npr] = $0; next }
FILENAME ~ /DEAD_ENDS.md$/ {
  # Title = the **bold** lead if present, else the text up to the first ". " / ": " after the date/author prefix.
  if ($0 !~ /^- /) next
  s = substr($0, 3)
  if (match(s, /\*\*[^*]+\*\*/)) s = substr(s, RSTART + 2, RLENGTH - 4)
  else { r = substr(s, 26); if (match(r, /(\. |: )/)) s = substr(s, 1, 25 + RSTART - 1) }
  sub(/[:.]$/, "", s); de[++nde] = s
  next
}
FILENAME ~ /INBOX.md$/ {
  if ($0 !~ /^- /) next
  ninb++
  if (id != "" && (tolower($0) ~ ("@(agent-)?" id "([^a-z0-9_-]|$)") || $0 ~ /@(all|everyone)([^a-z]|$)/)) ib[++nib] = $0
  next
}
FILENAME ~ /LESSONS.md$/ { if ($0 ~ /^### L-/) ls[++nls] = substr($0, 5); next }
FILENAME ~ /\/jobs$/ {
  if ($1 !~ /^job-/) next
  lab = ""; for (i = 5; i < NF; i++) lab = lab (lab == "" ? "" : " ") $i
  if ($2 ~ /^(running|queued|pending|building|submitted|starting|provisioning|validat)/) live[++nlive] = "- " $1 " " $2 " " substr($NF, 6, 11) " | " lab
  else if (nend < 4) ended = ended (nend++ ? ", " : " ") $1 " " $2
  next
}
FILENAME ~ /\/bill$/ {
  if (av == "" && $0 ~ /"available_usd":/) { av = $2; sub(/,$/, "", av) }
  if (bu == "" && $0 ~ /"burn_rate_usd_per_hour":/) { bu = $2; sub(/,$/, "", bu) }
  if (rw == "" && $0 ~ /"runway_minutes":/) { rw = $2; sub(/,$/, "", rw) }
  if (ex == "" && $0 ~ /"expires_at": "/) ex = substr($2, 2, 10)
  next
}
END {
  print "# SSDM digest " now " | agent=" (me == "" ? "unknown (set SSDM_AGENT)" : me) " | " br
  print "Verbose: bash scripts/session_start.sh --full. Read once, on demand: origin/state HANDOFF.md, PREREG.md; GOAL.md; LESSONS.md entries."
  sec("Objective"); print (obj == "" ? "(no TARGETS.tsv)" : cut(obj, 300))
  sec("Targets: target | leader score | ours (TARGETS.tsv)"); for (i = 1; i <= ntg; i++) print cut(tg[i], 160)
  sec("BOARD top 8 (origin/state; read it in full before reordering)")
  if (!nbd) print "(no BOARD on origin/state)"
  for (i = 1; i <= nbd && i <= 8; i++) print cut(bd[i], 120)
  if (nbd > 8 || nprop) print "(+" (nbd > 8 ? nbd - 8 : 0) " items, " nprop + 0 " proposals)"
  sec("Claims"); if (!ncl) print "(none)"
  for (i = 1; i <= ncl && i <= 8; i++) { c = cl[i]; print cut("- " c " | " o[c] " | " j[c] " | until " l[c] ((l[c] != "" && l[c] < now) ? " EXPIRED" : ""), 100) }
  sec("INBOX for " (me == "" ? "?" : me) " (newest first; " ninb + 0 " entries in all)")
  if (id == "") print "(identity unknown: set SSDM_AGENT=lead|evidence|agent-a|agent-b|agent-c)"
  else if (!nib) print "(nothing addressed to you)"
  else { for (i = 1; i <= nib && i <= 3; i++) print cut(ib[i], 180); if (nib > 3) print "(+" nib - 3 " older: git show origin/state:INBOX.md)" }
  sec("GPU jobs live (autor job ls)")
  if (nlive) { for (i = 1; i <= nlive && i <= 8; i++) print cut(live[i], 110); if (nlive > 8) print "(+" nlive - 8 " more)" }
  else print "(none live, or autor unavailable)"
  if (ended != "") print cut("recent ended:" ended, 170)
  sec("Balance (autor billing; never create top-ups or payment links)")
  print (av == "" ? "(unavailable)" : "available $" av " | burn $" bu "/h | runway " rw " min | first credit expiry " ex)
  sec("Ledger last 8: utc agent job | index [lo,hi] | $ | status | recipe")
  for (i = tail0(nled, 8); i <= nled; i++) {
    split(led[i], f, "\t"); ci = (f[7] != "" || f[8] != "") ? " [" f[7] "," f[8] "]" : ""
    print cut("- " substr(f[1], 6, 11) " " f[2] " " f[3] " | " (f[6] == "" ? "-" : f[6]) ci " | " (f[9] == "" ? "-" : f[9]) " | " f[10] " | " f[5], 115)
  }
  sec("PROGRESS last 5 lines"); for (i = tail0(npr, 5); i <= npr; i++) print cut(pr[i], 130)
  sec("DEAD_ENDS titles, last 15 (do not retry without new evidence)")
  for (i = tail0(nde, 15); i <= nde; i++) print cut("- " de[i], 88)
  sec("LESSONS titles, last 10 of " nls + 0 " (read the entry before touching that area)")
  for (i = tail0(nls, 10); i <= nls; i++) print cut(ls[i], 100)
}' "${files[@]}" 2>/dev/null >"$T/digest"

if [ "$(wc -c <"$T/digest")" -le "$MAXB" ]; then cat "$T/digest"
else head -c "$((MAXB - 80))" "$T/digest"; echo; echo "(digest truncated at $MAXB bytes; bash scripts/session_start.sh --full)"; fi
exit 0
