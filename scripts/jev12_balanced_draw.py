"""JEV2-4B training + calibration draws: family-balanced, from one or more reviewed draw dirs (jev1, jev2, ...).

Each source dir is a finished, gated draw (``records.jsonl.gz`` + ``<build>.manifest.jsonl.gz`` +
``<build>.content.jsonl.gz``, as scripts/jev1_build.py ``final`` writes it). Per JevBench family (``aug.<draw>.family8``
or ``aug.<draw>.family``) the rows are ranked by sha256("<seed>:<id>") (label-blind, deterministic); the first
``--train-per-family`` go to the TRAIN draw and the next ``--calib-per-family`` to the CALIB draw (post-hoc temperature
fit only; never trained on). A family with fewer rows than train + calib fails closed unless ``--allow-short`` (then
it contributes what it has, train first, and the report says so). Content disjointness train vs calib is checked on the
source content maps (record_content_hash), so a content twin under another id cannot straddle the two.

Two passes, O(families x (train + calib)) memory: pass 1 ranks ids per family (heaps), pass 2 streams the lines out.
Writes <out>/<name>/ and <out>/<name>-calib/ in the jev1 three-file format + a pointer JSON + draw-report.json.

    python -X utf8 -I scripts/jev12_balanced_draw.py --src <ssdm-data>/m2/build/bschema-draw-jev1 \
        --src <ssdm-data>/m2/build/bschema-draw-jev2 --train-per-family 1000 --calib-per-family 60 \
        --name bschema-draw-jev12-bal1000 --out <ssdm-data>/m2/build
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import heapq
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def canonical(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _lines(p):
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line.rstrip("\n")


def _gz(p):
    return io.TextIOWrapper(gzip.GzipFile(p, "wb", mtime=0), encoding="utf-8", newline="\n")


def family_of(r: dict) -> str | None:
    for v in (r.get("aug") or {}).values():
        if isinstance(v, dict) and (v.get("family8") or v.get("family")):
            return v.get("family8") or v.get("family")
    return None


def rank(seed: int, rid: str) -> str:
    return hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest()


def _empty_levels(q: dict) -> bool:
    """A score question whose level criteria are all empty (no rubric text: the model sees bare level numbers)."""
    crit = q.get("criteria")
    vals = list(crit.values()) if isinstance(crit, dict) else list(crit or [])
    return q.get("type") == "score" and not any(str(v).strip() for v in vals if v is not None)


def _request_key(r: dict) -> str:
    return hashlib.sha256(canonical(r.get("state")).encode()).hexdigest()[:24]


def plan(srcs: list[Path], n_train: int, n_calib: int, seed: int,
         family_src: dict[str, str] | None = None, exclude: set[tuple[str, str]] | None = None,
         family_cap: dict[str, int] | None = None, score_share: float | None = None,
         pool_exclude: dict[str, set[str]] | None = None, dedupe_sources: set[str] | None = None,
         deprioritize_empty_score: bool = False, stats: dict | None = None,
         exclude_subs: set[str] | None = None, force_calib: set[str] | None = None,
         global_exclude: set[str] | None = None) -> tuple[dict, dict]:
    """Pass 1: per family the (n_train_f + n_calib) smallest-rank ids; returns ({id: (family, role)}, family counts).
    ``family_src`` {family: source dir name}: that family is drawn ONLY from that source. ``exclude`` {(family, source
    dir name)}: that family's rows of that source are never drawn (e.g. jev1's long_policy). ``family_cap`` {family:
    n}: at most n TRAIN rows for that family (down-weighting; its calib rows are unchanged). ``force_calib`` (JEV2-H2O):
    these ids are never trained and always go to CALIB; a family with any forced id gets ONLY its forced ids as calib
    (e.g. the H2O eval's fixed jev2a held-out slice, so the step-0 floor is comparable to that eval)."""
    family_cap = family_cap or {}
    force_calib = force_calib or set()
    forced: dict[str, tuple[str, str]] = {}
    forced_skey: dict[str, str] = {}
    heaps: dict[str, list] = collections.defaultdict(list)
    full: dict[str, list] = collections.defaultdict(list)
    seen: set[str] = set()
    counts = collections.Counter()
    family_src = family_src or {}
    exclude = exclude or set()
    pool_exclude = pool_exclude or {}
    dedupe_sources = dedupe_sources or set()
    stats = stats if stats is not None else {}
    stats.setdefault("pool_rows_after_exclude", collections.Counter())
    stats.setdefault("pool_excluded", collections.Counter())
    stats.setdefault("deduped", collections.Counter())
    for d in srcs:
        for line in _lines(d / "records.jsonl.gz"):
            r = json.loads(line)
            rid, fam = r["id"], family_of(r)
            if fam is None:
                raise SystemExit(f"{d}: {rid} has no aug.<draw>.family8 / family")
            if global_exclude and rid in global_exclude:  # cross-pool exclusion lists (sha-pinned files)
                stats["global_excluded"] = stats.get("global_excluded", 0) + 1
                continue
            if rid in pool_exclude.get(d.name, ()):  # per-pool verbatim-leaf / leak exclusion (sha-pinned file)
                stats["pool_excluded"][d.name] += 1
                continue
            stats["pool_rows_after_exclude"][d.name] += 1
            if exclude_subs:  # whole sub-families dropped (e.g. jev2b prob_indep: constant gold)
                sub = next((v.get("sub") for v in (r.get("aug") or {}).values() if isinstance(v, dict) and v.get("sub")),
                           None)
                if sub in exclude_subs:
                    stats.setdefault("sub_excluded", collections.Counter())[sub] += 1
                    continue
            if (fam in family_src and family_src[fam] != d.name) or (fam, d.name) in exclude:
                continue
            keep = min(n_train, family_cap.get(fam, n_train)) + n_calib
            if rid in seen:
                raise SystemExit(f"duplicate id across sources: {rid}")
            seen.add(rid)
            counts[fam] += 1
            if rid in force_calib:
                forced[rid] = (fam, "calib")
                forced_skey[rid] = _request_key(r)
                continue
            if score_share is not None:  # per-type quota: keep every (rank, id, type) of the family (ids only, small)
                q = r["questions"]["q"]
                full[fam].append((int(rank(seed, rid)[:16], 16), rid, (q.get("type") or "choice"), _empty_levels(q),
                                  _request_key(r) if r.get("source") in dedupe_sources else None, _request_key(r)))
                continue
            item = (-int(rank(seed, rid)[:16], 16), rid)  # max-heap of the smallest ranks
            h = heaps[fam]
            if len(h) < keep:
                heapq.heappush(h, item)
            elif item > h[0]:
                heapq.heapreplace(h, item)
    roles = {}
    forced_fams = {f for f, _ in forced.values()}
    stats["forced_calib_found"] = len(forced)
    stats["forced_calib_by_family"] = dict(collections.Counter(f for f, _ in forced.values()))
    if score_share is not None:
        for fam in list(full):  # dedupe: at most one row per request text (dedupe sources), the lowest rank wins
            rows, keys = sorted(full[fam]), set()
            kept = []
            for x in rows:
                if x[4] is not None:
                    if x[4] in keys:
                        stats["deduped"][fam] += 1
                        continue
                    keys.add(x[4])
                kept.append(x)
            full[fam] = kept
        nts = {fam: min(n_train, family_cap.get(fam, n_train)) for fam in full}
        quota = score_quotas({f: sum(1 for x in v if x[2] == "score") for f, v in full.items()}, nts, score_share,
                             n_calib, {f: sum(1 for x in v if x[2] != "score") for f, v in full.items()})
        trains: dict[str, list] = {}
        calib_skeys: set[str] = set()
        for fam, rows in full.items():
            rows.sort()
            # score rows with rubric text first; empty-level rows (bare numbers) only if needed
            sc = sorted((x for x in rows if x[2] == "score"), key=lambda x: (x[3] and deprioritize_empty_score, x[0]))
            ns = [x for x in rows if x[2] != "score"]
            q = quota[fam]
            if deprioritize_empty_score:  # down-weight: empty-level rows fill at most half of the family's score quota
                full_lv = [x for x in sc if not x[3]]
                empty = [x for x in sc if x[3]]
                take = full_lv[:q] + empty[: max(0, min(q - len(full_lv[:q]), q // 2))]
                ts = set(take)
                rest_sc = [x for x in sc if x not in ts]
                stats.setdefault("empty_score_capped", collections.Counter())[fam] += max(0, q - len(take))
            else:
                take, rest_sc = sc[:q], sc[q:]
            train = take + ns[: nts[fam] - len(take)]
            if len(train) < nts[fam]:  # not enough non-score rows: top up with more score rows
                train += rest_sc[: nts[fam] - len(train)]
            trains[fam] = train
        # JEV3-SYM: the selection slice shares NO state text with any training row (state-level dedupe, any family);
        # slice candidates whose state is a training state are skipped, then the disjointness is asserted (fail closed)
        train_skeys = {x[5] for t in trains.values() for x in t}
        for fam, rows in full.items():
            chosen = {x[1] for x in trains[fam]}
            pool = [x for x in rows if x[1] not in chosen]
            calib, skipped = [], 0
            if fam not in forced_fams:
                for x in pool:
                    if len(calib) >= n_calib:
                        break
                    if x[5] in train_skeys:
                        skipped += 1
                        continue
                    calib.append(x)
            stats.setdefault("calib_state_overlap_skipped", collections.Counter())[fam] += skipped
            for x in trains[fam]:
                roles[x[1]] = (fam, "train")
            for x in calib:
                roles[x[1]] = (fam, "calib")
            calib_skeys.update(x[5] for x in calib)
        calib_skeys.update(forced_skey.values())
        clash = calib_skeys & train_skeys
        if clash:
            raise SystemExit(f"selection slice shares {len(clash)} state text(s) with training rows (state-level dedupe)")
        stats["calib_state_disjoint"] = True
        roles.update(forced)
        return roles, dict(counts)
    for fam, h in heaps.items():
        ids = [rid for _, rid in sorted(h, key=lambda x: -x[0])]  # ascending rank
        nt = min(n_train, family_cap.get(fam, n_train))
        for k, rid in enumerate(ids):
            if k < nt or fam not in forced_fams:
                roles[rid] = (fam, "train" if k < nt else "calib")
    roles.update(forced)
    return roles, dict(counts)


def score_quotas(avail: dict[str, int], nts: dict[str, int], share: float, n_calib: int,
                 avail_ns: dict[str, int] | None = None) -> dict[str, int]:
    """Per-family number of SCORE rows among the train rows so that the whole train set is ~``share`` score rows:
    one common fraction q (bisection) with quota_f = clip(round(q x n_f), floor_f, cap_f). cap_f = the family's score
    rows minus a calib reserve; floor_f = the score rows the family MUST supply because it lacks non-score rows (e.g. a
    rating-only family). Families without score rows contribute 0, so the others carry more. If the floors alone exceed
    the target, the share cannot go lower than they force (reported by the caller's type mix)."""
    avail_ns = avail_ns or {}
    target = share * sum(nts.values())
    cap = {f: max(0, min(nts[f], avail.get(f, 0) - min(n_calib, avail.get(f, 0) // 4))) for f in nts}
    floor = {f: min(cap[f], max(0, nts[f] - max(0, avail_ns.get(f, nts[f]) - min(n_calib, avail_ns.get(f, nts[f]) // 4))))
             for f in nts}

    def quota(q):
        return {f: max(floor[f], min(cap[f], round(q * nts[f]))) for f in nts}

    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(quota(mid).values()) < target else (lo, mid)
    return quota(hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--train-per-family", type=int, required=True)
    ap.add_argument("--calib-per-family", type=int, default=200)  # EV-PREREG-AREA-TEMP addendum: >= 200 per family
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--family-src", action="append", default=[],
                    help="FAMILY=SRC_DIR_NAME: draw FAMILY only from that source (repeatable)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="FAMILY=SRC_DIR_NAME: never draw FAMILY from that source (repeatable)")
    ap.add_argument("--pool-exclude", action="append", default=[], help="SRC_DIR_NAME=PATH: ids to drop from that pool")
    ap.add_argument("--pool-exclude-sha", action="append", default=[], help="SRC_DIR_NAME=SHA256 of that exclusion file")
    ap.add_argument("--pool-rows", action="append", default=[], help="SRC_DIR_NAME=N: assert N rows after exclusion")
    ap.add_argument("--exclude-sub", action="append", default=[],
                    help="sub-family name (aug.<draw>.sub): never drawn (repeatable)")
    ap.add_argument("--dedupe-source", action="append", default=[],
                    help="record source (e.g. glaive): at most one row per request text (state) in the draw")
    ap.add_argument("--deprioritize-empty-score", action="store_true",
                    help="score rows with empty level criteria fill at most half of a family's score quota, last")
    ap.add_argument("--score-share", type=float, default=None,
                    help="target share of SCORE-type rows among the train rows (per-family quotas; total unchanged)")
    ap.add_argument("--family-cap", action="append", default=[],
                    help="FAMILY=N: at most N train rows for FAMILY (repeatable; calib unchanged)")
    ap.add_argument("--force-calib", default=None,
                    help="ids file: never trained, always calib (its families get only these as calib)")
    ap.add_argument("--force-calib-sha", default=None, help="sha256 of the --force-calib file (required with it)")
    ap.add_argument("--global-exclude", action="append", default=[],
                    help="PATH=SHA256: ids dropped from EVERY pool (repeatable; sha asserted; every id must be found)")
    a = ap.parse_args(argv)
    gex, gex_rec = set(), []
    for spec in a.global_exclude:
        path, want = spec.rsplit("=", 1)
        b = Path(path).read_bytes()
        got = hashlib.sha256(b).hexdigest()
        if got != want:
            raise SystemExit(f"--global-exclude {path}: sha256 {got} != pinned {want}")
        ids_ = {ln.strip() for ln in b.decode("utf-8").splitlines() if ln.strip()}
        gex |= ids_
        gex_rec.append({"file": path, "sha256": got, "n_ids": len(ids_)})
    fcal, fcal_rec = set(), None
    if a.force_calib:
        b = Path(a.force_calib).read_bytes()
        got = hashlib.sha256(b).hexdigest()
        if got != a.force_calib_sha:
            raise SystemExit(f"--force-calib file sha256 {got} != pinned {a.force_calib_sha}")
        fcal = {ln.strip() for ln in b.decode("utf-8").splitlines() if ln.strip()}
        fcal_rec = {"file": str(a.force_calib), "sha256": got, "n_ids": len(fcal)}
    from train.cohort import record_content_hash

    srcs = [Path(s) for s in a.src]
    fsrc = dict(x.split("=", 1) for x in a.family_src)
    if any(v not in {s.name for s in srcs} for v in fsrc.values()):
        raise SystemExit(f"--family-src names a source that is not a --src: {fsrc}")
    excl = {tuple(x.split("=", 1)) for x in a.exclude}
    caps = {k: int(v) for k, v in (x.split("=", 1) for x in a.family_cap)}
    # per-pool exclusion files (verbatim-leaf / leak rule), each pinned by sha256; fail closed on any mismatch
    pex, pex_rec = {}, {}
    shas_given = dict(x.split("=", 1) for x in a.pool_exclude_sha)
    for name, path in (x.split("=", 1) for x in a.pool_exclude):
        b = Path(path).read_bytes()
        got = hashlib.sha256(b).hexdigest()
        if name not in shas_given or got != shas_given[name]:
            raise SystemExit(f"pool exclusion file for {name}: sha256 {got} != pinned {shas_given.get(name)}")
        ids_ = {ln.strip() for ln in b.decode("utf-8").splitlines() if ln.strip()}
        pex[name] = ids_
        pex_rec[name] = {"file": str(path), "sha256": got, "n_ids": len(ids_)}
    stats: dict = {}
    roles, counts = plan(srcs, a.train_per_family, a.calib_per_family, a.seed, fsrc, excl, caps, a.score_share,
                         pex, set(a.dedupe_source), a.deprioritize_empty_score, stats, set(a.exclude_sub), fcal, gex)
    if gex_rec:
        if stats["global_excluded"] != len(gex):
            raise SystemExit(f"--global-exclude: {stats['global_excluded']} of {len(gex)} ids found in the pools")
        gex_rec = {"files": gex_rec, "n_ids": len(gex), "found": stats["global_excluded"]}
    if fcal_rec is not None:  # forced ids missing from the pools (e.g. removed by a pool exclusion) are reported
        fcal_rec.update(found=stats["forced_calib_found"], by_family=stats["forced_calib_by_family"])
    for name, n in (x.split("=", 1) for x in a.pool_rows):
        if stats["pool_rows_after_exclude"][name] != int(n):
            raise SystemExit(f"pool {name}: {stats['pool_rows_after_exclude'][name]} rows after exclusion != expected {n}")
    for name, rec in pex_rec.items():
        rec["excluded_found"] = stats["pool_excluded"][name]
        if rec["excluded_found"] != rec["n_ids"]:
            raise SystemExit(f"pool {name}: {rec['excluded_found']} of {rec['n_ids']} exclusion ids found in the pool")
    short = {f: n for f, n in counts.items()
             if n < min(a.train_per_family, caps.get(f, a.train_per_family)) + a.calib_per_family}
    if short and not a.allow_short:
        raise SystemExit(f"families below train + calib: {short} (pass --allow-short to accept)")
    out = Path(a.out)
    dirs = {"train": out / a.name, "calib": out / f"{a.name}-calib"}
    for d in dirs.values():
        if (d / "records.jsonl.gz").exists():
            raise SystemExit(f"refusing to overwrite {d}")
        d.mkdir(parents=True, exist_ok=True)
    fh = {role: (_gz(d / "records.jsonl.gz"), _gz(d / f"{d.name}.manifest.jsonl.gz"), _gz(d / f"{d.name}.content.jsonl.gz"))
          for role, d in dirs.items()}
    hs = {role: [hashlib.sha256(), hashlib.sha256(), hashlib.sha256()] for role in dirs}
    per = {role: collections.Counter() for role in dirs}
    content = {role: set() for role in dirs}
    ids = {role: [] for role in dirs}
    types = {role: collections.Counter() for role in dirs}
    try:
        for d in srcs:
            for line in _lines(d / "records.jsonl.gz"):
                r = json.loads(line)
                got = roles.get(r["id"])
                if got is None:
                    continue
                fam, role = got
                ch = record_content_hash(r)
                content[role].add(ch)
                parts = (line + "\n",
                         canonical({"family": r["family"], "id": r["id"],
                                    "sha256": hashlib.sha256(line.encode()).hexdigest(), "source": r["source"]}) + "\n",
                         canonical({"content_sha256": ch, "id": r["id"]}) + "\n")
                for f, h, p in zip(fh[role], hs[role], parts):
                    f.write(p)
                    h.update(p.encode())
                per[role][fam] += 1
                ids[role].append(r["id"])
                types[role][r["questions"]["q"].get("type") or "choice"] += 1
    finally:
        for fs in fh.values():
            for f in fs:
                f.close()
    clash = content["train"] & content["calib"]
    if clash:
        raise SystemExit(f"{len(clash)} content hash(es) in both train and calib (rebuild with another seed)")
    rep = {"name": a.name, "seed": a.seed, "sources": [str(s) for s in srcs], "family_src": fsrc,
           "exclude": sorted("=".join(x) for x in excl), "family_cap": caps, "score_share_target": a.score_share,
           "pool_exclusions": pex_rec, "global_exclude": gex_rec or None, "force_calib": fcal_rec, "pool_rows_after_exclude": dict(stats["pool_rows_after_exclude"]),
           "dedupe_sources": sorted(a.dedupe_source), "deduped_by_family": dict(stats["deduped"]),
           "excluded_subs": dict(stats.get("sub_excluded", {})),
           "calib_state_disjoint": stats.get("calib_state_disjoint"),
           "calib_state_overlap_skipped": dict(stats.get("calib_state_overlap_skipped", {})),
           "deprioritize_empty_score": a.deprioritize_empty_score,
           "empty_score_short_by_family": dict(stats.get("empty_score_capped", {})),
           "train_type_mix": {t: round(n / max(1, sum(types["train"].values())), 4) for t, n in types["train"].items()},
           "train_type_counts": dict(types["train"]), "calib_type_counts": dict(types["calib"]),
           "source_family_rows": counts,
           "train_per_family": a.train_per_family, "calib_per_family": a.calib_per_family, "short_families": short,
           "train_rows": dict(per["train"]), "calib_rows": dict(per["calib"]),
           "n_train": sum(per["train"].values()), "n_calib": sum(per["calib"].values()), "content_disjoint": True,
           # sha256 of the sorted record ids, newline-joined (== jobs.m2_first_cell.screen.slice_ids_sha256)
           "train_ids_sha256": hashlib.sha256("\n".join(sorted(ids["train"])).encode()).hexdigest(),
           "calib_ids_sha256": hashlib.sha256("\n".join(sorted(ids["calib"])).encode()).hexdigest(),
           "rule": "per family, ascending sha256('<seed>:<id>'): first train_per_family -> train, next calib_per_family "
                   "-> calib (label-blind)"}
    for role, d in dirs.items():
        h = hs[role]
        ptr = {"build": f"{d.name} (scripts/jev12_balanced_draw.py over {', '.join(s.name for s in srcs)})",
               "location": f"ssdm-data/m2/build/{d.name}/ (outside git; deterministic)",
               "records_uncompressed_sha256": h[0].hexdigest(),
               "manifest": f"{d.name}.manifest.jsonl.gz", "manifest_uncompressed_sha256": h[1].hexdigest(),
               "content_map": f"{d.name}.content.jsonl.gz", "content_map_uncompressed_sha256": h[2].hexdigest(),
               "n_records": sum(per[role].values()), "role": role,
               "ids_sha256": hashlib.sha256("\n".join(sorted(ids[role])).encode()).hexdigest(),
               "schema": "ssdm-train-v0, one question per row (qid 'q'), JevBench request shape",
               "excludes_cohort": "inherited from every source draw's own gates (each source pointer's excludes_cohort)",
               "builder": {"module": "scripts/jev12_balanced_draw.py", "seed": a.seed},
               "templates_verified": ["jrt-h2o-shim:qwen3.5-4b"]}
        (d / "pointer.json").write_text(json.dumps(ptr, indent=1) + "\n", encoding="utf-8")
        rep[f"{role}_pointer"] = ptr
    (dirs["train"] / "draw-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_train", "n_calib", "train_rows", "train_type_mix", "short_families")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
