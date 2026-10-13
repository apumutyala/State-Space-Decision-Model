"""jev-heldout gate 1: disjointness from EVERY jev draw (prereg EV-PREREG-JEV-HELDOUT; 0 shared required on the final set).

    python -X utf8 -I scripts/jevho_disjoint.py --records <pool or final records.jsonl.gz> --out <json> [--ids-out <txt>]

Against every ``ssdm-data/m2/build/bschema-draw-jev*/records.jsonl.gz`` (jev1, jev2a, jev2a-v2, jev2b, jev2c, jev2c-v2,
jev2h*, jev2-mix*, jev3, jev3-slice, jev3-lmr, jev3-typed) and the candidate pools / checked pools on disk
(``bschema-draw-jev*-work/{pool,checked}.jsonl.gz``):

  keys (prereg): id, source item (parent id; dataset + source row key), content hash (train.cohort.record_content_hash),
                 normalised state (whitespace collapsed, casefolded, dict -> canonical JSON)
  text (extra):  exact normalised LINES of >= 4 words (and the line after its first ':' label) of every string leaf;
                 13-word shingles of human-source rows: a heldout row is hit when one jev row shares >= 3 of its
                 shingles AND >= 30 % of them. Lines / shingles present in >= 3 distinct heldout groups are template
                 text (our own headers, notes banks) and are not probed.
Prints counts only. Writes the hit ids (one per line) to --ids-out. Rule-engine rows (source jevho_synth) hit ONLY by
exact_line are template overlap with the reused engines' own banks (jev2b / jev1 rule texts), not a shared item: they
are reported separately and not dropped; every key hit and every human-source text hit is dropped.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BUILD = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data\m2\build")
N = 13


def canonical(v) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def state_key(st) -> str:
    s = st if isinstance(st, str) else canonical(st)
    return hashlib.sha256(re.sub(r"\s+", " ", s).strip().casefold().encode()).hexdigest()


def leaves(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from leaves(x)
    elif isinstance(v, list):
        for x in v:
            yield from leaves(x)


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.casefold())


def line_keys(st) -> set[str]:
    out = set()
    for leaf in leaves(st):
        for ln in [leaf] + leaf.split("\n"):
            for t in (ln, ln.split(":", 1)[1] if ":" in ln else ""):
                w = words(t)
                if len(w) >= 4:
                    out.add(" ".join(w))
    return out


def shingles(st) -> set[int]:
    out = set()
    for leaf in leaves(st):
        w = words(leaf)
        for i in range(len(w) - N + 1):
            out.add(hash(" ".join(w[i:i + N])))
    return out


def jev_files() -> list[Path]:
    fs = sorted(BUILD.glob("bschema-draw-jev*/records.jsonl.gz"))
    fs += [p for d in sorted(BUILD.glob("bschema-draw-jev*-work")) for p in (d / "pool.jsonl.gz", d / "checked.jsonl.gz")
           if p.exists()]
    return fs


def main(argv=None) -> int:
    from train.cohort import record_content_hash
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-out")
    a = ap.parse_args(argv)
    H = []  # heldout rows: (id, family, source)
    k_id, k_parent, k_rowkey, k_content, k_state = {}, {}, {}, {}, {}
    line_groups, sh_groups = collections.defaultdict(set), collections.defaultdict(set)
    row_lines, row_sh = [], []
    with gzip.open(a.records, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            j = r["aug"]["jev1"]
            i = len(H)
            H.append((r["id"], j["family8"], r["source"]))
            k_id[r["id"]] = i
            k_parent[r["aug"]["b1_view"]["parent_id"]] = i
            pv = j.get("prov") or {}
            if pv.get("row_key"):
                k_rowkey[(pv.get("dataset"), pv["row_key"])] = i
            k_content[record_content_hash(r)] = i
            k_state[state_key(r["state"])] = i
            g = j.get("group") or r["id"]
            lk = line_keys(r["state"])
            sh = shingles(r["state"]) if r["source"] != "jevho_synth" else set()
            for x in lk:
                line_groups[x].add(g)
            for x in sh:
                sh_groups[x].add(g)
            row_lines.append(lk)
            row_sh.append(sh)
    tmpl_l = {x for x, gs in line_groups.items() if len(gs) >= 3}
    tmpl_s = {x for x, gs in sh_groups.items() if len(gs) >= 3}
    line_map, sh_map = collections.defaultdict(list), collections.defaultdict(list)
    for i, (lk, sh) in enumerate(zip(row_lines, row_sh)):
        for x in lk - tmpl_l:
            line_map[x].append(i)
        clean = sh - tmpl_s
        row_sh[i] = len(clean)
        for x in clean:
            sh_map[x].append(i)
    del line_groups, sh_groups, row_lines
    hits = collections.defaultdict(set)  # heldout row -> reasons
    per_file = {}
    for f in jev_files():
        c = collections.Counter()
        n = 0
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                n += 1
                r = json.loads(line)
                checks = [("id", k_id.get(r["id"])), ("source_item", k_parent.get(r["aug"]["b1_view"]["parent_id"])),
                          ("content", k_content.get(record_content_hash(r))), ("state", k_state.get(state_key(r["state"])))]
                pv = (r["aug"].get("jev1") or {}).get("prov") or {}
                if pv.get("row_key"):
                    checks.append(("source_row_key", k_rowkey.get((pv.get("dataset"), pv["row_key"]))))
                for x in line_keys(r["state"]):
                    for i in line_map.get(x, ()):
                        checks.append(("exact_line", i))
                cnt = collections.Counter()
                for x in shingles(r["state"]):
                    for i in sh_map.get(x, ()):
                        cnt[i] += 1
                for i, m in cnt.items():
                    if m >= 3 and m >= 0.3 * row_sh[i]:
                        checks.append(("shingle_13", i))
                for why, i in checks:
                    if i is not None:
                        hits[i].add(why)
                        c[why] += 1
        per_file[str(f.relative_to(BUILD))] = {"rows": n, "hits": dict(c)}
        print(json.dumps({"file": str(f.relative_to(BUILD)), "rows": n, "hits": dict(c)}), flush=True)
    by_fam = collections.defaultdict(collections.Counter)
    for i, ws in hits.items():
        for w in ws:
            by_fam[H[i][1]][w] += 1
        by_fam[H[i][1]]["rows_hit"] += 1
    tmpl_only = {i for i, ws in hits.items() if H[i][2] == "jevho_synth" and ws == {"exact_line"}}
    drop = sorted(set(hits) - tmpl_only)
    rep = {"gate": "jev-heldout disjointness vs every jev draw (ids, source items, row keys, content, state, exact "
                   "lines, 13-word shingles)", "n_heldout_rows": len(H), "n_rows_hit": len(hits),
           "hits_by_family": {k: dict(v) for k, v in sorted(by_fam.items())}, "files": per_file,
           "n_rows_dropped": len(drop),
           "engine_rows_template_line_overlap_not_dropped": len(tmpl_only),
           "key_hits": sum(1 for ws in hits.values() if ws & {"id", "source_item", "content", "state", "source_row_key"}),
           "template_lines_excluded": len(tmpl_l), "template_shingles_excluded": len(tmpl_s),
           "records_sha256": hashlib.sha256(gzip.open(a.records).read()).hexdigest(), "pass": not drop}
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    if a.ids_out:
        Path(a.ids_out).write_text("".join(H[i][0] + "\n" for i in sorted(hits)), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_heldout_rows", "n_rows_hit", "n_rows_dropped", "key_hits",
                                         "engine_rows_template_line_overlap_not_dropped", "hits_by_family", "pass")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
