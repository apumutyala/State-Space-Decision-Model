"""Source cases: the unit that must never straddle DEV and TEST, and the bootstrap cluster.

The kit's `group_id` links only requests scored together (ToolRet/BRIGHT chunks, ACOS
reviews, GSM8K tracks, BPoMP poems). Rows that share a *source* (one cartoon contest,
clinical trial, forecast event, song, rater, HotpotQA question, ...) carry different group
ids, so a group-level split leaks: a model tuned on DEV has seen TEST's cartoons, trials,
users and events, and a group-level bootstrap ignores within-source correlation.

A source case is a connected component of rows linked by any of:
- the same kit `group_id`;
- the same per-benchmark source key(s) below (list-valued keys link every element);
- a byte-identical engine payload (`payload_sha256`);
- a byte-identical non-trivial `state` (>= MIN_STATE_CHARS of canonical JSON).
Links never cross benchmarks.

The component is at least as coarse as every clustering the kit uses (`scoring/report.py`:
POP909 by song, cfcolor by user; BPoMP's group is the poem).
"""

import collections
import hashlib

from decision_index.suite.io import dumps

MIN_STATE_CHARS = 64

# catalog_id -> [(section, key)]; section "m" = metadata, "p" = provenance
SOURCE_KEYS = {
    5: [("p", "source_id")],  # CLINC150: repeated source utterance ids
    22: [("m", "song_id")],  # POP909: song (kit clusters by song)
    23: [("m", "user_id")],  # cfcolor: rater (kit clusters by user)
    37: [("m", "query_id")],  # ESCI: shopping query
    40: [("m", "source_id")],  # iSarcasmEval: same tweet across tracks
    42: [("m", "trial_ids")],  # NLI4CT: clinical trial(s)
    44: [("p", "source_question_id")],  # CLadder
    48: [("m", "underlying_event_groups")],  # ForecastBench: underlying event
    50: [("m", "question_id")],  # Habermas: debate question
    59: [("m", "source_task_id")],  # RAGTruth: source context / task
    61: [("m", "hpqa_id")],  # HoVer: HotpotQA question
    62: [("m", "bfcl_source_id")],  # When2Call: BFCL scenario
    64: [("m", "contest_number")],  # New Yorker: cartoon contest
}


def _scalars(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _scalars(x)
    elif isinstance(v, dict):
        for k in sorted(v):
            yield from _scalars(v[k])
    elif v is not None:
        yield str(v)


def link_keys(row):
    e = row["_evaluation"]
    n = e["catalog_id"]
    keys = [("group", str(e["group_id"])), ("payload", e["payload_sha256"])]
    state = dumps(row.get("state"))
    if len(state) >= MIN_STATE_CHARS:
        keys.append(("state", hashlib.sha256(state.encode()).hexdigest()))
    for sect, k in SOURCE_KEYS.get(n, ()):
        src = row.get("metadata" if sect == "m" else "provenance") or {}
        for v in _scalars(src.get(k)):
            keys.append((k, v))
    return [(n,) + k for k in keys]


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def case_ids(rows):
    """{(catalog_id, group_id): case_id}. Case id = catalog + smallest group id in the component,
    so it does not depend on row order."""
    uf = _UF()
    groups = set()
    for r in rows:
        e = r["_evaluation"]
        g = ("G", e["catalog_id"], str(e["group_id"]))
        groups.add(g)
        for k in link_keys(r):
            uf.union(g, ("K",) + k)
    members = collections.defaultdict(list)
    for g in groups:
        members[uf.find(g)].append(g)
    out = {}
    for comp in members.values():
        cid = f"{comp[0][1]}:{min(m[2] for m in comp)}"
        for m in comp:
            out[(m[1], m[2])] = cid
    return out


def case_of(rows):
    """Row -> case id helper over a fixed row set."""
    ids = case_ids(rows)
    return lambda r: ids[(r["_evaluation"]["catalog_id"], str(r["_evaluation"]["group_id"]))]


def overlap_report(a_rows, b_rows):
    """Residual source overlap between two slices, by link key type (should be empty)."""
    def keys(rows):
        out = collections.defaultdict(set)
        for r in rows:
            for k in link_keys(r):
                out[(k[0], k[1])].add(k[2])
        return out

    ka, kb = keys(a_rows), keys(b_rows)
    rep = {}
    for key in set(ka) & set(kb):
        shared = ka[key] & kb[key]
        if shared:
            rep[f"{key[0]}:{key[1]}"] = len(shared)
    return dict(sorted(rep.items()))
