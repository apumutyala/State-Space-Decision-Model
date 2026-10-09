"""6-gram train/eval contamination audit (CPU, streaming, low-RAM). Used for analysis/NGRAM_TEST_AUDIT_2026-10-09.md.

  python -I ngram_test_audit.py build-eval OUTDIR                       # hash DEV+TEST rows -> OUTDIR/eval.npz (+ meta; no eval text)
  python -I ngram_test_audit.py scan OUTDIR NAME FILE [FILE ...] [--ids IDFILE]   # one training draw -> OUTDIR/scan-NAME.json

Normalisation: lowercase, non-alphanumerics -> space, collapse whitespace, split. 6-grams never cross a field leaf.
Hash: blake2b-64 per token, polynomial combine mod 2^64, splitmix finaliser. Field types: S state leaves, Q question
instructions, O option (criteria) texts. Boilerplate = eval df > 1% AND the draw's train df > 1% (per draw, listed).
DEV = slices/dev-rows*.jsonl.gz; TEST = suite rows selected by harness/manifests/test-manifest*.jsonl.gz run_id
(read directly; the harness unseal gate is not invoked and no TEST text is printed or written).
"""
import gzip, hashlib, json, re, sys, os
from collections import Counter, defaultdict
import numpy as np

DATA = r"C:\Users\apumu\research\CHLabs\ssdm-data"
REPO = r"C:\Users\apumu\research\CHLabs\ssdm-wt\gap-b"
N = 6
P = np.uint64(0x9E3779B97F4A7C15)
_NONALNUM = re.compile(r"[\W_]+", re.UNICODE)
_tok_cache = {}
FIELD_BIT = {"S": 1, "Q": 2, "O": 4}


def norm_tokens(s):
    return _NONALNUM.sub(" ", s.lower()).split()


def tok_hash(t):
    h = _tok_cache.get(t)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little")
        if len(_tok_cache) < 3_000_000:
            _tok_cache[t] = h
    return h


def mix(x):
    x = x ^ (x >> np.uint64(30)); x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27)); x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def leaves(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v)


def row_fields(r):
    """-> dict field -> list of token lists (one per leaf)."""
    out = {"S": [], "Q": [], "O": []}
    for s in leaves(r.get("state")):
        out["S"].append(norm_tokens(s))
    qs = r.get("questions") or {}
    if isinstance(qs, dict):
        for q in qs.values():
            if not isinstance(q, dict):
                continue
            if isinstance(q.get("instructions"), str):
                out["Q"].append(norm_tokens(q["instructions"]))
            crit = q.get("criteria")
            if isinstance(crit, dict):
                for v in crit.values():
                    for s in leaves(v):
                        out["O"].append(norm_tokens(s))
            elif isinstance(crit, list):
                for s in leaves(crit):
                    out["O"].append(norm_tokens(s))
    return out


def ngram_hashes(tok_lists, want_text=False):
    """unique 6-gram hashes of a list of token lists (no window crosses a leaf). Optionally hash->text."""
    toks, seg = [], []
    for i, tl in enumerate(tok_lists):
        if len(tl) >= N:
            toks.extend(tl); seg.extend([i] * len(tl))
    if len(toks) < N:
        return np.empty(0, np.uint64), None
    th = np.fromiter((tok_hash(t) for t in toks), np.uint64, len(toks))
    seg = np.asarray(seg, np.int32)
    m = len(toks) - N + 1
    h = np.zeros(m, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(N):
            h = h * P + th[k:k + m]
        h = mix(h)
    valid = seg[:m] == seg[N - 1:N - 1 + m]
    h = h[valid]
    if not want_text:
        return np.unique(h), None
    idx = np.nonzero(valid)[0]
    return h, (toks, idx)


def read_jsonl_gz(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def manifest_ids(path):
    return {m["run_id"] for m in read_jsonl_gz(path)}


def eval_rows():
    """yield (split, run_id, family, row). DEV from materialised slice files; TEST from the suite by manifest id."""
    for p in [os.path.join(DATA, "slices", "dev-rows.jsonl.gz"), os.path.join(DATA, "slices", "dev-rows-0.3-gsm8k.jsonl.gz")]:
        for r in read_jsonl_gz(p):
            yield "DEV", r["_evaluation"]["run_id"], r.get("family"), r
    tid = manifest_ids(os.path.join(REPO, "harness", "manifests", "test-manifest.jsonl.gz"))
    tid |= manifest_ids(os.path.join(REPO, "harness", "manifests", "test-manifest-0.3-gsm8k.jsonl.gz"))
    seen = set()
    for p in ["suite-0.2/selected-rows.jsonl.gz", "suite-0.2/added-rows.jsonl.gz", "suite-0.3-gsm8k/gsm8k-rows.jsonl.gz"]:
        for r in read_jsonl_gz(os.path.join(DATA, p)):
            rid = (r.get("_evaluation") or {}).get("run_id")
            if rid in tid and rid not in seen:
                seen.add(rid)
                yield "TEST", rid, r.get("family"), r
    sys.stderr.write(f"TEST ids in manifests {len(tid)}, found {len(seen)}\n")


def build_eval(outdir):
    os.makedirs(outdir, exist_ok=True)
    chunks_h, chunks_r, chunks_f = [], [], []
    meta = []
    for i, (split, rid, fam, r) in enumerate(eval_rows()):
        f = row_fields(r)
        per = []
        for fld, bit in FIELD_BIT.items():
            h, _ = ngram_hashes(f[fld])
            per.append((h, bit))
        allh = np.concatenate([h for h, _ in per]) if per else np.empty(0, np.uint64)
        fb = np.concatenate([np.full(len(h), b, np.uint8) for h, b in per])
        # unique per row, OR field bits
        order = np.argsort(allh, kind="stable"); allh = allh[order]; fb = fb[order]
        if len(allh):
            u, start = np.unique(allh, return_index=True)
            fbu = np.bitwise_or.reduceat(fb, start) if len(start) else fb
        else:
            u, fbu = allh, fb
        chunks_h.append(u); chunks_f.append(fbu.astype(np.uint8)); chunks_r.append(np.full(len(u), i, np.int32))
        meta.append([split, rid, fam, int(len(u))] + [int(len(h)) for h, _ in per])
        if i % 2000 == 0:
            sys.stderr.write(f"eval {i} {split}\n")
    H = np.concatenate(chunks_h); del chunks_h
    R = np.concatenate(chunks_r); del chunks_r
    F = np.concatenate(chunks_f); del chunks_f
    order = np.argsort(H, kind="stable")
    H = H[order]; R = R[order]; F = F[order]; del order
    u, start, cnt = np.unique(H, return_index=True, return_counts=True)
    first = R[start]
    fb = np.bitwise_or.reduceat(F, start)
    del H, F
    np.savez(os.path.join(outdir, "eval.npz"), h=u, df=cnt.astype(np.uint32), first=first, fb=fb,
             post=R, pstart=start.astype(np.int64))
    with open(os.path.join(outdir, "eval-meta.json"), "w", encoding="utf-8") as fo:
        json.dump(meta, fo)
    sys.stderr.write(f"eval rows {len(meta)} unique 6-grams {len(u)}\n")


SAME_DS = ["banking77", "clinc", "when2call", "bfcl", "apibank", "cruxeval", "gsm8k", "acos", "mmlu", "hover",
           "anli", "contractnli", "esci", "toolret", "bright", "winogrande", "hellaswag", "bbh", "ragtruth", "phish",
           "nli4ct", "habermas", "vast", "hle", "musr", "sata", "cladder", "chess", "newyorker", "pop909", "sgd", "forecast"]


def _nn(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def same_dataset(src, fam):
    a, b = _nn(src), _nn(fam)
    return any(x in a and x in b for x in SAME_DS)


def bits(b):
    return "".join(v for k, v in ((1, "S"), (2, "Q"), (4, "O")) if b & k)


def scan(outdir, name, files, ids_file=None, post_cap=500):
    z = np.load(os.path.join(outdir, "eval.npz"))
    EH, EDF, EFB, POST, PSTART = z["h"], z["df"], z["fb"], z["post"], z["pstart"]
    meta = json.load(open(os.path.join(outdir, "eval-meta.json"), encoding="utf-8"))
    n_eval = len(meta)
    ev_n = np.array([m[3] for m in meta], np.int64)
    ev_s = np.array([m[4] for m in meta], np.int64)
    cand_thr = 0.01 * n_eval
    want = None
    if ids_file:
        want = set(l.strip() for l in open(ids_file, encoding="utf-8") if l.strip())
    n_rows, found = 0, set()
    cand_df = Counter()
    hit_rows = []
    gram_text = {}
    for path in files:
        for r in read_jsonl_gz(path):
            rid = r.get("id")
            if want is not None:
                if rid not in want or rid in found:
                    continue
                found.add(rid)
            n_rows += 1
            f = row_fields(r)
            row_hits, runs = {}, []
            for fld, bit in FIELD_BIT.items():
                for leaf in f[fld]:
                    if len(leaf) < N:
                        continue
                    h, ti = ngram_hashes([leaf], want_text=True)
                    if ti is None or not len(h):
                        continue
                    pos = np.searchsorted(EH, h)
                    pos[pos >= len(EH)] = 0
                    ok = EH[pos] == h
                    if not ok.any():
                        continue
                    toks, idx = ti
                    js = np.nonzero(ok)[0]
                    for j in js.tolist():
                        p_ = int(pos[j])
                        row_hits[p_] = row_hits.get(p_, 0) | bit
                        if p_ not in gram_text and len(gram_text) < 600_000:
                            gram_text[p_] = " ".join(toks[idx[j]:idx[j] + N])
                    runs.append((pos[js].astype(np.int64), js.astype(np.int64)))
            if row_hits:
                for p_ in row_hits:
                    if EDF[p_] > cand_thr:
                        cand_df[p_] += 1
                hit_rows.append((rid, r.get("source") or r.get("family"), r.get("family"), row_hits, runs))
            if n_rows % 20000 == 0:
                sys.stderr.write(f"{name}: {n_rows} rows, {len(hit_rows)} with any hit\n")
    boiler = {p_ for p_, c in cand_df.items() if c > 0.01 * n_rows}
    rows_any, gram_rows_nb, field_pairs, split_rows = [], Counter(), Counter(), Counter()
    for rid, src, fam, hits, runs in hit_rows:
        nb = [p_ for p_ in hits if p_ not in boiler]
        if not nb:
            continue
        nbset = set(nb)
        for p_ in nb:
            gram_rows_nb[p_] += 1
            field_pairs[f"train:{bits(hits[p_])}->eval:{bits(int(EFB[p_]))}"] += 1
        longest = 0
        for ps, js in runs:
            keep = np.fromiter((int(p) in nbset for p in ps.tolist()), bool, len(ps))
            js = js[keep]
            if len(js):
                br = np.nonzero(np.diff(js) != 1)[0]
                starts = np.r_[0, br + 1]
                ends = np.r_[br, len(js) - 1]
                longest = max(longest, int((js[ends] - js[starts]).max()) + N)
        arr = np.array(nb, np.int64)
        arr = arr[EDF[arr] <= post_cap]
        best, nbest, cov = -1, 0, 0.0
        sbest, s_n, s_cov = -1, 0, 0.0
        if len(arr):
            segs = [POST[PSTART[p]:PSTART[p] + EDF[p]] for p in arr.tolist()]
            cnt = np.bincount(np.concatenate(segs), minlength=n_eval)
            covv = cnt / np.maximum(ev_n, 1)
            best = int(np.argmax(covv))
            nbest, cov = int(cnt[best]), float(covv[best])
            sarr = arr[(EFB[arr] & 1) > 0]
            if len(sarr):
                scnt = np.bincount(np.concatenate([POST[PSTART[p]:PSTART[p] + EDF[p]] for p in sarr.tolist()]), minlength=n_eval)
                scov = np.where(ev_s > 0, scnt / np.maximum(ev_s, 1), 0.0)
                sbest = int(np.argmax(scov)); s_n = int(scnt[sbest]); s_cov = min(1.0, float(scov[sbest]))
            for s_ in {meta[t][0] for t in np.nonzero(cnt)[0].tolist()}:
                split_rows[s_] += 1
        bm = meta[best] if best >= 0 else ["?", "?", "?", 0]
        rows_any.append({"id": rid, "source": src, "family": fam, "n_grams": len(nb), "longest_span_tokens": longest,
                         "best_eval": bm[1], "best_eval_split": bm[0], "best_eval_family": bm[2],
                         "best_eval_ngrams": bm[3], "n_to_best": nbest, "best_eval_coverage": round(cov, 4),
                         "same_dataset": same_dataset(src, bm[2]),
                         "state_best_eval": meta[sbest][1] if sbest >= 0 else None,
                         "state_best_split": meta[sbest][0] if sbest >= 0 else None,
                         "state_best_family": meta[sbest][2] if sbest >= 0 else None,
                         "state_best_ngrams": meta[sbest][4] if sbest >= 0 else 0,
                         "state_n": s_n, "state_coverage": round(s_cov, 4),
                         "state_same_dataset": same_dataset(src, meta[sbest][2]) if sbest >= 0 else False,
                         "eval_fields": sorted({bits(int(EFB[p])) for p in nb}),
                         "sample": [gram_text.get(p_, "?") for p_ in sorted(nb, key=lambda q: int(EDF[q]))[:5]]})
    rows_any.sort(key=lambda d: (-d["best_eval_coverage"] * min(1.0, d["n_to_best"] / 5.0), -d["longest_span_tokens"]))

    def hist(key, edges):
        h = Counter()
        for d in rows_any:
            v = d[key]
            for e in edges:
                if v >= e:
                    h[f">={e}"] += 1
                    break
            else:
                h[f"<{edges[-1]}"] += 1
        return dict(h)

    strong = [d for d in rows_any if (d["best_eval_coverage"] >= 0.5 and d["n_to_best"] >= 3)
              or (d["state_coverage"] >= 0.8 and d["state_n"] >= min(3, d["state_best_ngrams"]) and d["state_best_ngrams"] >= 1)]
    for d in strong:
        d["strong_reason"] = ("row" if d["best_eval_coverage"] >= 0.5 and d["n_to_best"] >= 3 else "") +                              ("state" if d["state_coverage"] >= 0.8 and d["state_n"] >= 1 else "")
    out = {"draw": name, "files": files, "ids_file": ids_file, "n_rows": n_rows,
           "ids_requested": len(want) if want is not None else None, "n_eval_rows": n_eval,
           "rows_any_raw_hit": len(hit_rows),
           "boilerplate": sorted([{"ngram": gram_text.get(p_, "?"), "eval_df": int(EDF[p_]), "train_df": cand_df[p_]}
                                  for p_ in boiler], key=lambda d: -d["train_df"]),
           "rows_any_nonboiler": len(rows_any),
           "rows_by_ngram_count": hist("n_grams", [50, 20, 5, 2, 1]),
           "rows_by_longest_span": hist("longest_span_tokens", [50, 25, 12, 7, 6]),
           "rows_by_best_eval_coverage": hist("best_eval_coverage", [0.8, 0.5, 0.2, 0.05, 0.0]),
           "rows_touching_split": dict(split_rows),
           "field_pairs": dict(field_pairs.most_common(20)),
           "strong_rows": len(strong), "strong_same_dataset": sum(d["same_dataset"] or d["state_same_dataset"] for d in strong),
           "strong_eval_rows_n": len({d["best_eval"] for d in strong} | {d["state_best_eval"] for d in strong if d["state_coverage"] >= 0.8}),
           "strong_by_pair": Counter(f'{d["source"]}->{d["state_best_family"] if "state" in d["strong_reason"] else d["best_eval_family"]}' for d in strong).most_common(40),
           "top_ngrams": [{"ngram": gram_text.get(p_, "?"), "train_rows": c, "eval_rows_df": int(EDF[p_]),
                           "eval_fields": bits(int(EFB[p_]))} for p_, c in gram_rows_nb.most_common(40)],
           "strong_list": strong[:3000],
           "span25_by_pair": Counter(f'{d["source"]}->{d["best_eval_family"]}' for d in rows_any if d["longest_span_tokens"] >= 25).most_common(60),
           "span12_by_pair": Counter(f'{d["source"]}->{d["best_eval_family"]}' for d in rows_any if d["longest_span_tokens"] >= 12).most_common(60),
           "same_dataset_rows": [{k: d[k] for k in ("id", "source", "n_grams", "longest_span_tokens", "best_eval", "best_eval_split",
                                                    "best_eval_family", "n_to_best", "best_eval_coverage", "state_best_eval",
                                                    "state_coverage", "sample")}
                                 for d in rows_any if d["same_dataset"] or d["state_same_dataset"]][:20000],
           "top_rows": rows_any[:150]}
    with open(os.path.join(outdir, f"scan-{name}.json"), "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=1, ensure_ascii=False)
    sys.stderr.write(f"{name}: rows {n_rows}, raw {len(hit_rows)}, nonboiler {len(rows_any)}, strong {len(strong)}, boiler {len(boiler)}\n")



if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "build-eval":
        build_eval(a[0])
    elif cmd == "scan":
        ids = None
        if "--ids" in a:
            i = a.index("--ids"); ids = a[i + 1]; a = a[:i] + a[i + 2:]
        scan(a[0], a[1], a[2:], ids)
