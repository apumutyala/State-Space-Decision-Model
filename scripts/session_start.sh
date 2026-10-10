#!/usr/bin/env bash
# Session digest (AGENTS.md). Never touches the working tree, idempotent, ALWAYS exits 0, finishes by its own deadline.
#   bash scripts/session_start.sh          compact digest (<= 8000 bytes). The SessionStart hook in .claude/settings.json
#                                          runs it on startup / resume / clear / compact / fork; stdout goes to context.
#   bash scripts/session_start.sh --full   the old verbose dump (BOARD head, PROGRESS 15, ledger 12, whole INBOX).
# Agent identity (filters INBOX): $SSDM_AGENT (lead | evidence | agent-a | agent-b | agent-c), else the worktree
#   directory name, else the branch name. INBOX lines that mention @<id>, @agent-<id> or @all are shown.
#
# Speed (the shared laptop is CPU-starved: a process spawn can cost 1-2 s, so the old script took 40-110 s):
#   * the state branch is read once per commit: `git archive` into a cache keyed by the origin/state sha, shared by
#     all worktrees of this clone (TMPDIR/ssdm-digest-cache); unchanged state = no git archive, no tar;
#   * `git fetch` runs at most once per SSDM_FETCH_MAX_AGE seconds (default 120) across all worktrees;
#   * no date/wc/cat/sed: bash builtins and two awk passes do everything;
#   * autor (via WSL) starts first, in the background, capped at SSDM_AUTOR_TIMEOUT (15 s), output to files only.
# Deadline: everything is bounded by SSDM_DIGEST_DEADLINE (45 s, well inside the hook's 180 s timeout). State
#   sections print first; autor sections print only if ready by the deadline; whatever was skipped is named on a final
#   "partial digest: ..." line. A hook kill can therefore never drop the whole digest.
# Test knobs: SSDM_WSL (default wsl.exe), SSDM_AUTOR_TIMEOUT, SSDM_FETCH_TIMEOUT (10), SSDM_FETCH_MAX_AGE,
#   SSDM_DIGEST_DEADLINE, SSDM_DIGEST_MAX_BYTES (8000), SSDM_DIGEST_CACHE (cache root).
SECONDS=0
FULL=0; [ "${1:-}" = "--full" ] && FULL=1
MAXB=${SSDM_DIGEST_MAX_BYTES:-8000}
DL=${SSDM_DIGEST_DEADLINE:-45}
WSL=${SSDM_WSL:-wsl.exe}
export GIT_TERMINAL_PROMPT=0
skipped=()
tmo() { if [ -n "$HAVE_TIMEOUT" ]; then timeout "$@"; else shift; "$@"; fi; }
type -P timeout >/dev/null 2>&1 && HAVE_TIMEOUT=1

src=${BASH_SOURCE[0]//\\//}   # the hook may pass a C:\... path
cd "${src%/*}/.." 2>/dev/null || true

# 1. autor first (slowest source), in the background, output to files only (a hung child cannot hold our stdout).
CROOT=${SSDM_DIGEST_CACHE:-${TMPDIR:-/tmp}/ssdm-digest-cache}; CROOT=${CROOT//\\//}
jobs_f=$CROOT/jobs.$$; bill_f=$CROOT/bill.$$
[ -d "$CROOT" ] || mkdir -p "$CROOT" 2>/dev/null
p_jobs=; p_bill=
if type -P "$WSL" >/dev/null 2>&1 && [ -d "$CROOT" ]; then
  AUTOR=("$WSL" -d bi-ubuntu -- /root/.local/bin/autor --workspace default)
  MSYS_NO_PATHCONV=1 tmo "${SSDM_AUTOR_TIMEOUT:-15}" "${AUTOR[@]}" job ls >"$jobs_f" 2>/dev/null </dev/null & p_jobs=$!
  MSYS_NO_PATHCONV=1 tmo "${SSDM_AUTOR_TIMEOUT:-15}" "${AUTOR[@]}" billing >"$bill_f" 2>/dev/null </dev/null & p_bill=$!
fi

# 2. One git call for the branch and the common dir. core.worktree is read from the shared config with builtins
#    (a set core.worktree breaks git in every worktree: 2026-10-08 incident; warn only).
g=$(git --work-tree="$PWD" rev-parse --abbrev-ref HEAD --git-common-dir 2>/dev/null)
br=${g%%$'\n'*}; common=${g#*$'\n'}; [ "$common" = "$g" ] && common=
[ -n "$common" ] && [ "${common#/}" = "$common" ] && [ "${common#?:}" = "$common" ] && common=$PWD/$common
if [ -f "$common/config" ]; then
  sect=
  while IFS= read -r ln || [ -n "$ln" ]; do
    ln=${ln%$'\r'}
    case "$ln" in "["*) sect=$ln;; *) if [ "$sect" = "[core]" ]; then
      k=${ln%%=*}; k=${k//[[:space:]]/}
      [ "$k" = worktree ] && echo "!!!! WARNING: shared .git/config has core.worktree=${ln#*=} -- this breaks git in EVERY worktree. Fix: git --work-tree=. config --unset core.worktree; never export GIT_DIR/GIT_WORK_TREE."
    fi;; esac
  done <"$common/config"
fi
key=${common:-$PWD}; C=$CROOT/${key//[^A-Za-z0-9]/_}
[ -d "$C" ] || mkdir -p "$C" 2>/dev/null

# 3. Fetch at most every SSDM_FETCH_MAX_AGE s (stamp shared by all worktrees), never past the deadline budget.
last=0; [ -f "$C/fetch.stamp" ] && read -r last <"$C/fetch.stamp" 2>/dev/null
age=999999; [ -n "$EPOCHSECONDS" ] && age=$(( EPOCHSECONDS - ${last:-0} ))
budget=$(( DL - SECONDS - 12 )); ft=${SSDM_FETCH_TIMEOUT:-10}; [ "$budget" -lt "$ft" ] && ft=$budget
if [ "$age" -ge "${SSDM_FETCH_MAX_AGE:-120}" ]; then
  if [ "$ft" -ge 2 ] && tmo "$ft" git fetch -q origin agent-main state >/dev/null 2>&1 </dev/null; then
    printf '%s\n' "$EPOCHSECONDS" >"$C/fetch.stamp" 2>/dev/null
  else skipped+=("state refresh (git fetch failed or timed out; showing the last fetched origin/state)"); fi
fi

# 4. State branch: read origin/state's sha with builtins when the ref is loose; extract once per sha.
sha=
[ -f "$common/refs/remotes/origin/state" ] && read -r sha <"$common/refs/remotes/origin/state"
[ -n "$sha" ] || sha=$(git rev-parse -q --verify origin/state 2>/dev/null)
S=
if [ -n "$sha" ]; then
  S=$C/state-$sha
  if [ ! -f "$S/.complete" ]; then
    tmp=$C/tmp-$$; mkdir -p "$tmp" 2>/dev/null
    if tmo "$(( DL - SECONDS > 3 ? DL - SECONDS - 2 : 1 ))" git archive "$sha" 2>/dev/null | tar --force-local -x -C "$tmp" 2>/dev/null; then  # "C:/..." is a path, not host:path
      : >"$tmp/.complete"; mv -T "$tmp" "$S" 2>/dev/null || rm -rf "$tmp" 2>/dev/null  # -T: never nest into a racing copy
      # keep the cache small: drop other state snapshots (own reproducible files), in the background
      ( for d in "$C"/state-*; do [ "$d" = "$S" ] || rm -rf "$d"; done ) >/dev/null 2>&1 </dev/null &
    else rm -rf "$tmp" 2>/dev/null & fi
    [ -f "$S/.complete" ] || S=
  fi
fi
[ -n "$S" ] || skipped+=("state sections (origin/state unavailable)")

if [ "$FULL" = 1 ]; then
  echo "== BOARD (origin/state) =="; head -40 "$S/BOARD.md" 2>/dev/null
  echo; echo "== PROGRESS (last 15 lines) =="; tail -15 "$S/PROGRESS.md" 2>/dev/null
  echo; echo "== CLAIMS =="; ls "$S/claims" 2>/dev/null
  echo; echo "== LEDGER (last 12 runs) =="; tail -12 "$S/results/ledger.tsv" 2>/dev/null
  echo; echo "== INBOX =="; cat "$S/INBOX.md" 2>/dev/null
  echo; echo "== agent-main =="; git log --oneline -5 origin/agent-main 2>/dev/null
  wait $p_jobs $p_bill 2>/dev/null
  echo; echo "== GPU JOBS =="; head -12 "$jobs_f" 2>/dev/null
  echo; echo "== BALANCE =="; grep -E '"available_usd"|"burn_rate|"runway_minutes"' "$bill_f" 2>/dev/null | head -3
  rm -f "$jobs_f" "$bill_f" >/dev/null 2>&1
  exit 0
fi

me=${SSDM_AGENT:-}
if [ -z "$me" ]; then
  for cand in "${PWD##*/}" "${br##*/}"; do
    case "$cand" in agent-a|agent-b|agent-c|lead|evidence|lead-*|evidence-*) me=$cand; break;; esac
  done
fi
TZ=UTC0 printf -v now '%(%Y-%m-%dT%H:%MZ)T' -1
targets=$S/TARGETS.tsv; [ -f "$targets" ] || targets=TARGETS.tsv
AUTOR_BYTES=1500   # reserved for the autor sections and the partial line
sf=(); [ -n "$S" ] && sf=("$S/BOARD.md" "$S"/claims/*.lock "$S/results/ledger.tsv" "$S/PROGRESS.md" "$S/DEAD_ENDS.md" "$S/INBOX.md")

# 5. State sections now (they are the critical ones). Byte mode (-b) so the cap is in bytes; cut() never splits a
#    UTF-8 character. Sections are ordered by value: an overflow drops the DEAD_ENDS / LESSONS tail first.
LC_ALL=C awk -b -v me="$me" -v br="$br" -v now="$now" -v cap="$((MAXB - AUTOR_BYTES))" -v S="$S" '
function cut(s, n) { gsub(/\t/, " ", s); if (length(s) > n) { s = substr(s, 1, n - 1); sub(/[\300-\377][\200-\277]*$/, "", s); s = s "~" } return s }
function emit(s) { if (full) return; if (used + length(s) + 1 > cap) { full = 1; return } out = out s "\n"; used += length(s) + 1 }
function sec(t) { emit("## " t) }
function tail0(n, k) { return n > k ? n - k + 1 : 1 }
BEGINFILE { if (ERRNO != "") nextfile }
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
END {
  emit("# SSDM digest " now " | agent=" (me == "" ? "unknown (set SSDM_AGENT)" : me) " | " br)
  emit("Verbose: bash scripts/session_start.sh --full. Read once, on demand: origin/state HANDOFF.md, PREREG.md; GOAL.md; LESSONS.md entries.")
  sec("Objective"); emit(obj == "" ? "(no TARGETS.tsv)" : cut(obj, 300))
  sec("Targets: target | leader score | ours (TARGETS.tsv)"); for (i = 1; i <= ntg; i++) emit(cut(tg[i], 160))
  if (S != "") {
    sec("BOARD top 8 (origin/state; read it in full before reordering)")
    if (!nbd) emit("(no BOARD on origin/state)")
    for (i = 1; i <= nbd && i <= 8; i++) emit(cut(bd[i], 120))
    if (nbd > 8 || nprop) emit("(+" (nbd > 8 ? nbd - 8 : 0) " items, " nprop + 0 " proposals)")
    sec("Claims"); if (!ncl) emit("(none)")
    for (i = 1; i <= ncl && i <= 8; i++) { c = cl[i]; emit(cut("- " c " | " o[c] " | " j[c] " | until " l[c] ((l[c] != "" && l[c] < now) ? " EXPIRED" : ""), 100)) }
    sec("INBOX for " (me == "" ? "?" : me) " (newest first; " ninb + 0 " entries in all)")
    if (id == "") emit("(identity unknown: set SSDM_AGENT=lead|evidence|agent-a|agent-b|agent-c)")
    else if (!nib) emit("(nothing addressed to you)")
    else { for (i = 1; i <= nib && i <= 3; i++) emit(cut(ib[i], 180)); if (nib > 3) emit("(+" nib - 3 " older: git show origin/state:INBOX.md)") }
    sec("Ledger last 8: utc agent job | index [lo,hi] | $ | status | recipe")
    for (i = tail0(nled, 8); i <= nled; i++) {
      split(led[i], f, "\t"); ci = (f[7] != "" || f[8] != "") ? " [" f[7] "," f[8] "]" : ""
      emit(cut("- " substr(f[1], 6, 11) " " f[2] " " f[3] " | " (f[6] == "" ? "-" : f[6]) ci " | " (f[9] == "" ? "-" : f[9]) " | " f[10] " | " f[5], 115))
    }
    sec("PROGRESS last 5 lines"); for (i = tail0(npr, 5); i <= npr; i++) emit(cut(pr[i], 130))
    sec("DEAD_ENDS titles, last 15 (do not retry without new evidence)")
    for (i = tail0(nde, 15); i <= nde; i++) emit(cut("- " de[i], 88))
  }
  sec("LESSONS titles, last 10 of " nls + 0 " (read the entry before touching that area)")
  for (i = tail0(nls, 10); i <= nls; i++) emit(cut(ls[i], 100))
  printf "%s", out
  if (full) print "(state sections truncated at the byte cap; bash scripts/session_start.sh --full)"
}' "$targets" "${sf[@]}" LESSONS.md 2>/dev/null

# 6. autor sections only if ready by the deadline (a watchdog kills the bounded calls at the deadline).
ok_jobs=0; ok_bill=0; wd=
if [ -n "$p_jobs" ]; then
  rem=$(( DL - SECONDS - 1 ))
  if kill -0 "$p_jobs" 2>/dev/null || kill -0 "$p_bill" 2>/dev/null; then
    if [ "$rem" -gt 0 ]; then ( sleep "$rem"; kill "$p_jobs" "$p_bill" ) >/dev/null 2>&1 </dev/null & wd=$!
    else kill "$p_jobs" "$p_bill" 2>/dev/null; fi
  fi
  wait "$p_jobs" 2>/dev/null && ok_jobs=1
  wait "$p_bill" 2>/dev/null && ok_bill=1
  [ -n "$wd" ] && kill "$wd" 2>/dev/null
  [ "$ok_jobs" = 1 ] || skipped+=("GPU jobs (autor not ready by the ${DL}s deadline or failed)")
  [ "$ok_bill" = 1 ] || skipped+=("balance (autor not ready by the ${DL}s deadline or failed)")
else
  skipped+=("GPU jobs + balance (no wsl/autor here)")
fi
fj=/dev/null; fb=/dev/null; [ "$ok_jobs" = 1 ] && fj=$jobs_f; [ "$ok_bill" = 1 ] && fb=$bill_f  # /dev/null keeps awk off stdin
part=; for s in "${skipped[@]}"; do part+=${part:+; }$s; done
LC_ALL=C awk -b -v part="$part" -v jf="$fj" -v bf="$fb" -v hj="$ok_jobs" '
function cut(s, n) { if (length(s) > n) { s = substr(s, 1, n - 1); sub(/[\300-\377][\200-\277]*$/, "", s); s = s "~" } return s }
FILENAME == jf && hj {
  if ($1 !~ /^job-/) next
  lab = ""; for (i = 5; i < NF; i++) lab = lab (lab == "" ? "" : " ") $i
  if ($2 ~ /^(running|queued|pending|building|submitted|starting|provisioning|validat)/) live[++nlive] = "- " $1 " " $2 " " substr($NF, 6, 11) " | " lab
  else if (nend < 4) ended = ended (nend++ ? ", " : " ") $1 " " $2
  next
}
FILENAME == bf && bf != "/dev/null" {
  if (av == "" && $0 ~ /"available_usd":/) { av = $2; sub(/,$/, "", av) }
  if (bu == "" && $0 ~ /"burn_rate_usd_per_hour":/) { bu = $2; sub(/,$/, "", bu) }
  if (rw == "" && $0 ~ /"runway_minutes":/) { rw = $2; sub(/,$/, "", rw) }
  if (ex == "" && $0 ~ /"expires_at": "/) ex = substr($2, 2, 10)
}
END {
  print "## GPU jobs live (autor job ls)"
  if (nlive) { for (i = 1; i <= nlive && i <= 8; i++) print cut(live[i], 110); if (nlive > 8) print "(+" nlive - 8 " more)" }
  else print (hj ? "(none live)" : "(unavailable)")
  if (ended != "") print cut("recent ended:" ended, 170)
  print "## Balance (autor billing; never create top-ups or payment links)"
  print (av == "" ? "(unavailable)" : "available $" av " | burn $" bu "/h | runway " rw " min | first credit expiry " ex)
  if (part != "") print cut("partial digest: " part, 400)
}' "$fj" "$fb" 2>/dev/null
rm -f "$jobs_f" "$bill_f" >/dev/null 2>&1 </dev/null &
exit 0
