"""Independent re-gate of bschema-draw-jev2a (evidence lane, adversarial data review).

Streams the draw one line at a time; never prints or writes eval text (only counts, hashes and TRAINING row ids).

  python -X utf8 scripts/regate_jev2a.py hashes   --draw <dir> --out hashes.json
  python -X utf8 scripts/regate_jev2a.py format   --draw <dir> --out format.json
  python -X utf8 scripts/regate_jev2a.py overlap  --draw <dir> --out overlap.json [--no-index]
  python -X utf8 scripts/regate_jev2a.py cohort   --draw <dir> --out cohort.json
  python -X utf8 scripts/regate_jev2a.py fingerprint --draw <dir> --out fp.json
  python -X utf8 scripts/regate_jev2a.py provenance --draw <dir> --out prov.json     (needs pyarrow)

``overlap``: own word 13-gram + exact-leaf matcher (NOT train.doc_overlap_guard): NFKC + casefold, ``\\w+`` words, a
13-gram per leaf string (state / instructions / option texts), 64-bit hashes. Eval sides: the 231 jevkit public
items, every Index suite row (DEV / TEST / HELDOUT split as the harness manifests say), proxy600 ids flagged, and the
GSM8K official test split (local parquet, sha256 recorded).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
K = 13
EXACT_MIN_WORDS = 6
_W = re.compile(r"\w+", re.UNICODE)
PRIME = 0x100000001B3


def lines(p):
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line.rstrip("\n")


def recs(p):
    for line in lines(p):
        yield json.loads(line)


def words(s: str) -> list[str]:
    return _W.findall(unicodedata.normalize("NFKC", s).casefold())


def leaves(o, out=None):
    """String leaves (dict VALUES; option ids are slugs of their text and are skipped)."""
    out = [] if out is None else out
    if isinstance(o, str):
        out.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            leaves(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            leaves(v, out)
    elif o is not None and not isinstance(o, bool):
        out.append(str(o))
    return out


def row_leaves(state, questions) -> list[str]:
    out = leaves(state)
    for q in (questions or {}).values():
        if not isinstance(q, dict):
            out += leaves(q)
            continue
        out += leaves(q.get("instructions"))
        for key in ("criteria", "options", "choices"):
            c = q.get(key)
            if isinstance(c, dict):
                for k, v in c.items():
                    out.append(v if isinstance(v, str) and v.strip() else str(k))
            elif c is not None:
                out += leaves(c)
    return out


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def grams(ws: list[str]):
    import numpy as np
    n = len(ws)
    if n < K:
        return np.zeros(0, dtype=np.uint64)
    wh = np.fromiter((h64(w) for w in ws), dtype=np.uint64, count=n)
    g = np.zeros(n - K + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(K):
            g = g * np.uint64(PRIME) + wh[j:n - K + 1 + j]
    return g


def row_grams(lvs):
    import numpy as np
    parts = [grams(words(s)) for s in lvs]
    parts = [p for p in parts if len(p)]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)


def norm_leaf(s: str) -> str:
    return " ".join(words(s))


def sha_uncompressed(p: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(p, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fam(r):
    return r["aug"]["jev1"]["family8"]


# ------------------------------------------------------------------------------------------------ hashes
def cmd_hashes(a):
    d = Path(a.draw)
    out = {}
    for name in ("records.jsonl.gz", f"{d.name}.manifest.jsonl.gz", f"{d.name}.content.jsonl.gz"):
        p = d / name
        out[name] = {"gz_sha256": sha_file(p), "uncompressed_sha256": sha_uncompressed(p),
                     "n_lines": sum(1 for _ in lines(p))}
    return out


# ------------------------------------------------------------------------------------------------ format
def p_chi2(x: float, k: int) -> float:
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, k))
    except Exception:  # Wilson-Hilferty
        z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
        return 0.5 * math.erfc(z / math.sqrt(2))


def cmd_format(a):
    d = Path(a.draw)
    n = 0
    ids = set()
    dup_ids = []
    by_fam = collections.Counter()
    by_fam_type = collections.defaultdict(collections.Counter)
    by_src = collections.Counter()
    split_by_src = collections.defaultdict(collections.Counter)
    lic_by_src = collections.defaultdict(collections.Counter)
    missing_lic = []
    state_type = collections.defaultdict(collections.Counter)
    gold_bad = collections.defaultdict(list)
    gold_pos = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # fam -> K -> pos
    keys_sorted = collections.defaultdict(lambda: [0, 0, 0.0])  # fam -> [n_choice, n_keys_sorted, expected]
    empty_crit = collections.defaultdict(list)
    empty_desc = collections.defaultdict(list)
    noul_desc_variants = collections.defaultdict(collections.Counter)
    label_dist = collections.defaultdict(collections.Counter)
    dup_content = collections.Counter()
    dup_state = collections.Counter()
    state_subs = {}
    corr_label = collections.defaultdict(collections.Counter)
    dup_nodigit = collections.Counter()
    words_by_fam = collections.defaultdict(list)
    content_first = {}
    dup_examples = collections.defaultdict(list)
    for r in recs(d / "records.jsonl.gz"):
        n += 1
        f = fam(r)
        by_fam[f] += 1
        rid = r["id"]
        if rid in ids:
            dup_ids.append(rid)
        ids.add(rid)
        src = r.get("source")
        by_src[src] += 1
        split_by_src[src][r.get("source_split")] += 1
        lic = r.get("license")
        lic_by_src[src][lic] += 1
        if not lic:
            missing_lic.append(rid)
        st = r["state"]
        state_type[f]["dict" if isinstance(st, dict) else "str" if isinstance(st, str) else type(st).__name__] += 1
        qs = r["questions"]
        if list(qs) != ["q"]:
            gold_bad[f].append((rid, "questions keys != [q]"))
        q = qs["q"]
        g = r["labels"]["q"]
        t = q["type"]
        by_fam_type[f][t] += 1
        crit = q.get("criteria")
        label_dist[f"{f}/{r['aug']['jev1'].get('sub')}"][str(g)] += 1
        if "corruption" in r["aug"]["jev1"]:
            corr_label[f"{r['aug']['jev1'].get('sub')}/{r['aug']['jev1']['corruption']}"][str(g)] += 1
        if t == "score" and isinstance(crit, list):  # score schema: an ordered list of level descriptions
            if not crit or any((not isinstance(v, str)) or not v.strip() for v in crit):
                empty_crit[f].append(rid)
        elif not isinstance(crit, dict) or not crit:
            empty_crit[f].append(rid)
        else:
            if any((not isinstance(v, str)) or not v.strip() for v in crit.values()):
                empty_desc[f].append(rid)
            if t == "noul":
                noul_desc_variants[f][json.dumps(crit, sort_keys=True)] += 1
        if t == "choice":
            ks = list(crit or {})
            if g not in ks:
                gold_bad[f].append((rid, "gold not in options"))
            else:
                k = len(ks)
                gold_pos[f][k][ks.index(g)] += 1
                kc = keys_sorted[f]
                kc[0] += 1
                kc[1] += int(ks == sorted(ks))
                kc[2] += 1 / math.factorial(k)
                nt = [" ".join(words(v if isinstance(v, str) and v.strip() else o)) for o, v in crit.items()]
                if nt.count(nt[ks.index(g)]) != 1:
                    gold_bad[f].append((rid, "gold text not unique"))
        elif t == "noul":
            if g not in ("true", "false"):
                gold_bad[f].append((rid, f"noul label {g!r}"))
            if set(crit or {}) != {"true", "false"}:
                gold_bad[f].append((rid, "noul criteria keys != {true,false}"))
        elif t == "score":
            try:
                if not 0 <= int(g) < len(crit):
                    gold_bad[f].append((rid, "score label out of range"))
            except Exception:
                gold_bad[f].append((rid, f"score label {g!r}"))
        else:
            gold_bad[f].append((rid, f"type {t!r}"))
        lv = row_leaves(st, qs)
        full = " | ".join(norm_leaf(x) for x in lv)
        words_by_fam[f].append(len(full.split()))
        hc = hashlib.sha256(full.encode()).hexdigest()[:16]
        dup_content[hc] += 1
        if hc in content_first and len(dup_examples["content"]) < 10:
            dup_examples["content"].append([content_first[hc], rid])
        content_first.setdefault(hc, rid)
        sh = hashlib.sha256(" | ".join(norm_leaf(x) for x in leaves(st)).encode()).hexdigest()[:16]
        dup_state[(f, sh)] += 1
        state_subs.setdefault((f, sh), collections.Counter())[r["aug"]["jev1"].get("sub")] += 1
        nd = re.sub(r"\d+", "#", full)
        dup_nodigit[(f, hashlib.sha256(nd.encode()).hexdigest()[:16])] += 1
    gp = {}
    for f, byk in gold_pos.items():
        x = df = m = 0
        per_k = {}
        for k, c in sorted(byk.items()):
            tot = sum(c.values())
            per_k[k] = {"n": tot, "positions": [c.get(i, 0) for i in range(k)]}
            if tot < 5 * k:
                continue
            e = tot / k
            x += sum((c.get(i, 0) - e) ** 2 / e for i in range(k))
            df += k - 1
            m += tot
        gp[f] = {"n_tested": m, "chi2": round(x, 2), "df": df, "p": round(p_chi2(x, df), 4) if df else None,
                 "per_k": per_k}

    def dupstats(cnt):
        extra = sum(v - 1 for v in cnt.values() if v > 1)
        return {"groups_gt1": sum(1 for v in cnt.values() if v > 1), "extra_rows": extra}

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None

    dn = collections.defaultdict(collections.Counter)
    for (f, _), v in dup_nodigit.items():
        if v > 1:
            dn[f]["groups"] += 1
            dn[f]["extra_rows"] += v - 1
    ds = collections.defaultdict(collections.Counter)
    for (f, _), v in dup_state.items():
        if v > 1:
            ds[f]["groups"] += 1
            ds[f]["extra_rows"] += v - 1
    dup_state_sub = collections.defaultdict(collections.Counter)
    for (f, sh), subs in state_subs.items():
        if dup_state[(f, sh)] > 1:
            for sb, c in subs.items():
                dup_state_sub[f"{f}/{sb}"]["rows_in_dup_groups"] += c
    return {
        "n_records": n, "unique_ids": len(ids), "dup_ids": dup_ids[:20],
        "by_family": dict(by_fam), "type_by_family": {k: dict(v) for k, v in by_fam_type.items()},
        "by_source": dict(by_src), "split_by_source": {k: dict(v) for k, v in split_by_src.items()},
        "license_by_source": {k: dict(v) for k, v in lic_by_src.items()}, "missing_license": len(missing_lic),
        "missing_license_ids": missing_lic[:20],
        "state_type_by_family": {k: dict(v) for k, v in state_type.items()},
        "gold_failures": {k: {"n": len(v), "examples": v[:10]} for k, v in gold_bad.items()},
        "gold_position_presented": gp,
        "choice_keys_already_sorted": {f: {"n_choice": v[0], "n_sorted": v[1], "expected_if_shuffled": round(v[2], 1)}
                                       for f, v in keys_sorted.items()},
        "empty_criteria_rows": {k: {"n": len(v), "examples": v[:5]} for k, v in empty_crit.items()},
        "empty_option_description_rows": {k: {"n": len(v), "examples": v[:5]} for k, v in empty_desc.items()},
        "noul_criteria_variants": {f: {"n_variants": len(c), "top": c.most_common(5)} for f, c in noul_desc_variants.items()},
        "label_dist_by_family_sub": {k: (dict(v) if len(v) <= 6 else {"n_distinct": len(v), "top5": v.most_common(5)})
                                     for k, v in label_dist.items()},
        "dup_state_by_sub": {k: dict(v) for k, v in dup_state_sub.items()},
        "judge_corruption_vs_label": {k: dict(v) for k, v in corr_label.items()},
        "words_by_family": {f: {"p50": pct(v, .5), "p99": pct(v, .99), "max": max(v)} for f, v in words_by_fam.items()},
        "dup_exact_text": dupstats(dup_content), "dup_examples": dict(dup_examples),
        "dup_state_within_family": {k: dict(v) for k, v in ds.items()},
        "near_dup_digits_masked_within_family": {k: dict(v) for k, v in dn.items()},
    }


# ------------------------------------------------------------------------------------------------ overlap
def free_mb():
    from train.jevbench_guard import free_ram_mb
    return free_ram_mb()


def cmd_overlap(a):
    import numpy as np
    d = Path(a.draw)
    t0 = time.time()
    G, R = array("Q"), array("I")
    ids, srcs, fams = [], [], []
    leafset = {}  # norm leaf hash (>= EXACT_MIN_WORDS words) -> first row idx
    state_hash = {}
    for i, r in enumerate(recs(d / "records.jsonl.gz")):
        ids.append(r["id"]); srcs.append(r.get("source")); fams.append(fam(r))
        lv = row_leaves(r["state"], r["questions"])
        g = row_grams(lv)
        G.frombytes(g.astype("<u8").tobytes()); R.extend([i] * len(g))
        for s in lv:
            ns = norm_leaf(s)
            if len(ns.split()) >= EXACT_MIN_WORDS:
                leafset.setdefault(h64(ns), i)
        sh = h64(" ".join(norm_leaf(x) for x in leaves(r["state"])))
        state_hash.setdefault(sh, i)
    G = np.frombuffer(G, dtype="<u8").astype(np.uint64)
    R = np.frombuffer(R, dtype=np.uint32)
    o = np.argsort(G, kind="stable")
    G, R = G[o], R[o]
    del o
    print(f"train grams {len(G):,} rows {len(ids)} leaves {len(leafset):,} {time.time()-t0:.0f}s free {free_mb()} MB",
          flush=True)
    sides = {}

    def new_side(name):
        sides[name] = {"mask": np.zeros(len(G), dtype=bool), "eval_rows": 0, "eval_rows_hit_13gram": 0,
                       "exact_leaf_train_rows": set(), "exact_state_train_rows": set(), "eval_rows_exact": 0,
                       "by_family_hit": collections.Counter()}
        return sides[name]

    def probe(side, state, questions, family):
        side["eval_rows"] += 1
        lv = row_leaves(state, questions)
        e = row_grams(lv)
        if len(e):
            lo = np.searchsorted(G, e, "left")
            hi = np.searchsorted(G, e, "right")
            sel = hi > lo
            if sel.any():
                side["eval_rows_hit_13gram"] += 1
                side["by_family_hit"][str(family)] += 1
                for l, h in zip(lo[sel], hi[sel]):
                    side["mask"][l:h] = True
        hit = False
        for s in lv:
            ns = norm_leaf(s)
            if len(ns.split()) >= EXACT_MIN_WORDS and h64(ns) in leafset:
                side["exact_leaf_train_rows"].add(leafset[h64(ns)]); hit = True
        sh = h64(" ".join(norm_leaf(x) for x in leaves(state)))
        if sh in state_hash and len(" ".join(norm_leaf(x) for x in leaves(state)).split()) >= 1:
            side["exact_state_train_rows"].add(state_hash[sh]); hit = True
        side["eval_rows_exact"] += int(hit)

    # JevBench public items
    from jevkit.items import load_public, overlap_guard_rows, read_pin
    s = new_side("jevbench_public_231")
    for _, _, family, row in overlap_guard_rows(load_public()):
        probe(s, row["state"], row["questions"], family)
    s["identity"] = {"repo": read_pin()["repo"], "commit": read_pin()["commit"]}
    print(f"jevbench done {s['eval_rows']} {time.time()-t0:.0f}s", flush=True)
    # GSM8K official test split
    gp = Path(a.gsm8k_test)
    if gp.exists():
        import pyarrow.parquet as pq
        s = new_side("gsm8k_official_test")
        s["identity"] = {"path": str(gp), "sha256": sha_file(gp)}
        tbl = pq.read_table(gp, columns=["question", "answer"]).to_pylist()
        for x in tbl:
            probe(s, x["question"], {"q": {"instructions": x["answer"]}}, "gsm8k_test")
        del tbl
        print(f"gsm8k test done {s['eval_rows']} {time.time()-t0:.0f}s", flush=True)
    # Index suite (DEV / TEST / HELDOUT), proxy600 flagged
    if not a.no_index:
        from train import doc_overlap_guard as DG
        px = set()
        pp = REPO / "jobs" / "common" / "dev_proxy600.json"
        if pp.exists():
            pj = json.loads(pp.read_text(encoding="utf-8"))
            cand = pj.get("run_ids") or pj.get("ids") or (pj if isinstance(pj, list) else [])
            px = {str(x) for x in cand}
        for nm in ("index_DEV", "index_TEST", "index_HELDOUT", "index_proxy600"):
            new_side(nm)
        k = 0
        for split, rid, family, row in DG.iter_eval_rows():
            probe(sides[f"index_{split}"], row.get("state"), row.get("questions"), family)
            if rid in px:
                probe(sides["index_proxy600"], row.get("state"), row.get("questions"), family)
            k += 1
            if k % 20000 == 0:
                print(f"index rows {k} {time.time()-t0:.0f}s free {free_mb()} MB", flush=True)
        sides["index_proxy600"]["identity"] = {"path": str(pp), "n_ids": len(px)}
    out = {"rule": f"own matcher: NFKC+casefold \\w+ words, word {K}-grams per leaf, blake2b-64 word hash, polynomial "
                   f"gram hash; exact = normalised leaf >= {EXACT_MIN_WORDS} words, or whole normalised state",
           "n_train_rows": len(ids), "n_train_grams": int(len(G)), "sides": {}}
    for nm, s in sides.items():
        m = s.pop("mask")
        hit_rows = np.unique(R[m]) if m.any() else np.zeros(0, dtype=np.uint32)
        frac = collections.Counter()
        tot = np.bincount(R, minlength=len(ids))
        hits = np.bincount(R[m], minlength=len(ids)) if m.any() else np.zeros(len(ids), dtype=np.int64)
        cov = {thr: int(((hits / np.maximum(tot, 1)) >= thr).sum() - ((tot == 0) & (thr <= 0)).sum())
               for thr in (0.1, 0.5, 0.9)}
        for i in hit_rows:
            frac[f"{fams[i]}/{srcs[i]}"] += 1
        ex_leaf = sorted(s.pop("exact_leaf_train_rows"))
        ex_state = sorted(s.pop("exact_state_train_rows"))
        out["sides"][nm] = {**{k: v for k, v in s.items() if k != "by_family_hit"},
                            "eval_rows_hit_by_eval_family": dict(s["by_family_hit"].most_common(30)),
                            "train_rows_with_13gram_hit": int(len(hit_rows)),
                            "train_rows_13gram_coverage_ge": {str(k): v for k, v in cov.items()},
                            "train_rows_hit_by_family_source": dict(frac),
                            "train_hit_ids_examples": [ids[i] for i in hit_rows[:25]],
                            "train_hit_ids_sha256": hashlib.sha256("\n".join(sorted(ids[i] for i in hit_rows)).encode()).hexdigest(),
                            "exact_leaf_train_rows": len(ex_leaf), "exact_leaf_ids": [ids[i] for i in ex_leaf[:25]],
                            "exact_state_train_rows": len(ex_state), "exact_state_ids": [ids[i] for i in ex_state[:25]]}
        if a.dump_hit_ids:
            Path(a.dump_hit_ids).mkdir(parents=True, exist_ok=True)
            (Path(a.dump_hit_ids) / f"{nm}.ids.txt").write_text("\n".join(ids[i] for i in hit_rows) + "\n",
                                                               encoding="utf-8")
    out["seconds"] = round(time.time() - t0)
    return out


# ------------------------------------------------------------------------------------------------ cohort / heldout
def base_id(r):
    return (r.get("aug") or {}).get("b1_view", {}).get("base_id") or r["id"].split("~", 1)[0]


def cmd_cohort(a):
    from harness import heldout as HO
    from train.cohort import Cohort, record_content_hash, record_item_keys, record_text_keys
    d = Path(a.draw)
    out = {}
    cm = {}
    for line in lines(d / f"{d.name}.content.jsonl.gz"):
        e = json.loads(line)
        cm[e.get("id")] = e.get("content_sha256") or e.get("sha256") or e.get("content")
    out["content_map_rows"] = len(cm)
    cohorts = {"v2": Cohort(expected_sha256="be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2"),
               "v3": Cohort(expected_sha256="498efa617c931a5ea073bc5c62ce94158270888f64b9bbdb027982cde43876da")}
    prep = {}
    for k, c in cohorts.items():
        prep[k] = {"ids": c.held_out(), "content": c.held_out_content(), "items": c.held_out_items(),
                   "text": c.text_keys, "identity": c.identity()}
    res = {k: collections.Counter() for k in cohorts}
    ex = {k: collections.defaultdict(list) for k in cohorts}
    cm_mismatch = []
    ho_hits = []
    n = 0
    for r in recs(d / "records.jsonl.gz"):
        n += 1
        rid = r["id"]
        b = base_id(r)
        ch = record_content_hash(r)
        if cm.get(rid) not in (None, ch) and len(cm_mismatch) < 20:
            cm_mismatch.append(rid)
        tk = record_text_keys(r)
        items = record_item_keys(b) | record_item_keys(rid)
        for k, p in prep.items():
            for what, hit in (("id", rid in p["ids"] or b in p["ids"]), ("content", ch in p["content"]),
                              ("source_item", bool(items & p["items"])), ("key_free_text", bool(tk & p["text"]))):
                if hit:
                    res[k][what] += 1
                    if len(ex[k][what]) < 10:
                        ex[k][what].append(rid)
        try:
            HO.check_record(r)
        except Exception as e:  # noqa: BLE001
            ho_hits.append([rid, type(e).__name__])
    out.update({"n_records": n, "cohorts": {k: {"identity": p["identity"], "hits": dict(res[k]),
                                                "examples": {w: v for w, v in ex[k].items()}} for k, p in prep.items()},
                "content_map_mismatch": cm_mismatch, "frozen_heldout_check_record_refused": len(ho_hits),
                "frozen_heldout_examples": ho_hits[:10], "heldout_frozen_families": sorted(map(str, HO.frozen()))[:50]})
    return out


# ------------------------------------------------------------------------------------------------ Index fingerprint
def cmd_fingerprint(a):
    from data import config
    from data.dedup import Fingerprint
    from train import jevbench_guard as JG
    JG.wait_for_ram(a.min_free_mb, log=lambda m: print(m, flush=True), poll_s=30, max_wait_s=3600)
    t0 = time.time()
    fp = Fingerprint.load(config.FINGERPRINT)
    try:
        fp.assert_production()
        prod = True
    except Exception as e:  # noqa: BLE001
        prod = f"assert_production failed: {e}"
    print(f"fingerprint loaded {time.time()-t0:.0f}s free {free_mb()} MB", flush=True)
    d = Path(a.draw)
    item = collections.Counter(); full = collections.Counter()
    ex_item, ex_full = [], []
    by_m = collections.Counter()
    n = 0

    def canonical(v):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    for r in recs(d / "records.jsonl.gz"):
        n += 1
        f = fam(r)
        ks, st = r["aug"]["jev1"].get("item_keys"), r["state"]
        it = {k: st[k] for k in ks} if ks else (st if isinstance(st, dict) else {"state": st})
        probe = {k: (v if isinstance(v, str) else [x["text"] for x in v]
                     if isinstance(v, list) and v and all(isinstance(x, dict) and "text" in x for x in v)
                     else canonical(v)) for k, v in it.items()}
        h = {k: v for k, v in fp.check(probe, {}).items() if v}
        if h:
            item[f] += 1
            if len(ex_item) < 20:
                ex_item.append([r["id"], sorted(h)])
        h2 = {k: v for k, v in fp.check(st if isinstance(st, dict) else {"state": st}, r["questions"]).items() if v}
        if h2:
            full[f] += 1
            for k in h2:
                by_m[k] += 1
            if len(ex_full) < 20:
                ex_full.append([r["id"], sorted(h2)])
        if len(getattr(fp, "_seg_cache", ())) > 100_000:
            fp._seg_cache.clear(); fp._txt_cache.clear()
        if n % 5000 == 0:
            print(f"{n} rows {time.time()-t0:.0f}s free {free_mb()} MB", flush=True)
    return {"identity": {k: fp.meta.get(k) for k in ("version", "npz_sha256", "arrays_sha256", "rows")},
            "assert_production": prod, "npz_path": str(config.FINGERPRINT), "n_records": n,
            "item_tier_probe_hits_by_family": dict(item), "item_tier_examples": ex_item,
            "full_row_probe_hits_by_family": dict(full), "full_row_hits_by_matcher": dict(by_m),
            "full_row_examples": ex_full, "seconds": round(time.time() - t0)}


# ------------------------------------------------------------------------------------------------ split provenance
def cmd_provenance(a):
    import pyarrow.parquet as pq
    d = Path(a.draw)
    want = collections.defaultdict(dict)  # source -> source_id -> [row ids]
    texts = {}
    for r in recs(d / "records.jsonl.gz"):
        src = r.get("source")
        if src in ("musique", "boolq", "gsm8k", "glaive_fc"):
            want[src].setdefault(str(r.get("source_id")), []).append(r["id"])
            if src == "boolq":
                texts[r["id"]] = (norm_leaf(r["state"]) if isinstance(r["state"], str) else None,
                                  norm_leaf(r["questions"]["q"]["instructions"]))
            elif src == "gsm8k":
                st = r["state"]
                req = st.get("request") if isinstance(st, dict) else st
                q = r["questions"]["q"]
                gold = (q.get("criteria") or {}).get(r["labels"]["q"]) if q["type"] == "choice" else None
                texts[r["id"]] = (norm_leaf(req) if isinstance(req, str) else None, gold)
            elif src == "glaive_fc":
                texts[r["id"]] = (norm_leaf(r["state"]) if isinstance(r["state"], str) else None, None)
            elif src == "musique":
                st = r["state"]
                docs = st.get("documents") if isinstance(st, dict) else None
                texts[r["id"]] = ({norm_leaf(x.get("text", "")) for x in docs} if docs else None, None)
    out = {"source_files": {}}
    root = Path(a.data)
    files = {"musique": root / "raw/jev2/musique/musique_ans_v1.0_train.jsonl",
             "gsm8k": root / "m2/sources/gsm8k/train-00000-of-00001.parquet",
             "boolq": root / "m2/sources/boolq/train-00000-of-00001.parquet",
             "glaive_fc": root / "m2/sources/glaive_fc/0000.parquet"}
    for src, p in files.items():
        out["source_files"][src] = {"path": str(p), "sha256": sha_file(p)}
    res = {}
    # MuSiQue: every id must be a TRAIN id; and the row's docs must come from that train record's paragraphs
    w = want["musique"]
    found, docs_ok, docs_bad = set(), 0, []
    with open(files["musique"], encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            sid = str(j.get("id"))
            if sid in w:
                found.add(sid)
                paras = {norm_leaf(p.get("paragraph_text", "")) for p in j.get("paragraphs", [])}
                for rid in w[sid]:
                    ds = texts[rid][0] or set()
                    if ds and ds <= paras:
                        docs_ok += 1
                    elif len(docs_bad) < 10:
                        docs_bad.append(rid)
    dev_like = [s for s in w if not s.split("__")[0] in ("2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3")]
    res["musique"] = {"n_rows": sum(len(v) for v in w.values()), "n_source_ids": len(w),
                      "ids_found_in_train_file": len(found), "ids_missing": sorted(set(w) - found)[:20],
                      "rows_docs_subset_of_train_paragraphs": docs_ok, "rows_docs_mismatch_examples": docs_bad,
                      "unexpected_id_prefix": dev_like[:10]}
    # parquet sources: source_id = row index in the train parquet (checked by text equality)
    for src, cols, key in (("gsm8k", ["question", "answer"], "question"), ("boolq", ["question", "passage"], None),
                           ("glaive_fc", None, None)):
        w = want[src]
        pf = pq.ParquetFile(files[src])
        names = pf.schema_arrow.names
        res[src] = {"n_rows": sum(len(v) for v in w.values()), "n_source_ids": len(w), "parquet_rows": pf.metadata.num_rows,
                    "columns": names}
        idx_ok = idx_bad = 0
        bad = []
        gold_chk, gold_bad_ex = [0, 0], []
        base = 0
        need = {int(s) for s in w if s.isdigit()}
        res[src]["non_integer_source_ids"] = sum(1 for s in w if not s.isdigit())
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=[c for c in (cols or names) if c in names]).to_pylist()
            for i, x in enumerate(t):
                gi = base + i
                if gi not in need:
                    continue
                for rid in w[str(gi)]:
                    a_, b_ = texts[rid]
                    if src == "gsm8k":
                        ok = a_ is not None and a_ == norm_leaf(x["question"])
                        if ok and b_ is not None:
                            ans = x["answer"].split("####")[-1].strip().replace(",", "")
                            gold_chk[0 if b_.replace(",", "").strip() == ans else 1] += 1
                            if b_.replace(",", "").strip() != ans and len(gold_bad_ex) < 10:
                                gold_bad_ex.append(rid)
                    elif src == "boolq":
                        ok = (a_ == norm_leaf(x["passage"])) and (b_.rstrip() == norm_leaf(x["question"]).rstrip()
                                                                   or norm_leaf(x["question"]) in b_)
                    else:
                        blob = norm_leaf(json.dumps(x, ensure_ascii=False))
                        ok = a_ is not None and (a_ in blob)
                    if ok:
                        idx_ok += 1
                    else:
                        idx_bad += 1
                        if len(bad) < 10:
                            bad.append(rid)
            base += len(t)
            del t
        res[src].update({"rows_text_match_train_row": idx_ok, "rows_text_mismatch": idx_bad,
                         "mismatch_examples": bad, "max_source_id": max(need) if need else None})
        if src == "gsm8k":
            res[src]["choice_gold_equals_source_final_answer"] = gold_chk[0]
            res[src]["choice_gold_differs"] = gold_chk[1]
            res[src]["choice_gold_differs_examples"] = gold_bad_ex
    out["checks"] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hashes", "format", "overlap", "cohort", "fingerprint", "provenance"])
    ap.add_argument("--draw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--gsm8k-test", default=r"C:\Users\apumu\research\CHLabs\ssdm-data\gsm8k-v3-src\test-00000-of-00001.parquet")
    ap.add_argument("--dump-hit-ids")
    ap.add_argument("--builder-data")
    ap.add_argument("--data", default=r"C:\Users\apumu\research\CHLabs\ssdm-data")
    ap.add_argument("--min-free-mb", type=int, default=900)
    a = ap.parse_args(argv)
    rep = globals()[f"cmd_{a.cmd}"](a)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(rep, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
