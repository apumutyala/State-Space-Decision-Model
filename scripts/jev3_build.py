"""Build bschema-draw-jev3 (data/jev3.py, FAMILY_SPEC_JEV3.md): gen -> check (jev1 cmd_check, unchanged) -> final.

    python -X utf8 -I scripts/jev3_build.py gen   --work <dir> [--families f ...]
    python -X utf8 -I scripts/jev3_build.py check --work <dir>
    python -X utf8 -I scripts/jev3_build.py final --work <dir> --lens <lens.jsonl> --receipt <dir> [--drop-ids f ...]

final, per family (rows in sha256(id) order, label-blind except the stated noul balance):
  1. token window: long_policy 2,000-4,096 shim tokens, every other family <= 4,096 (shim lens file; recorded per row)
  2. STATE dedupe (coordinator 2026-10-11): normalised state text (whitespace collapsed, casefolded; dict states as
     canonical JSON) is unique over the WHOLE draw (train + slice, all families); the first row by rank keeps it
  3. noul balance: the family's 1,200 rows keep the pool's noul share; noul true share 45-55 % where both strata allow
  4. held-out selection slice: whole groups (aug.jev1.group) in sha256('jev3-slice|<group>') order until 200 rows (rows
     of the last group beyond 200 are dropped, never trained); TRAIN = the other groups' rows, first 1,000 by rank
  5. disjointness asserted (fail closed): slice vs train by id, source item (parent), group, content hash, state text
Writes ssdm-data/m2/build/bschema-draw-jev3/ and bschema-draw-jev3-slice/ (records / manifest / content / lens), the
pointers data/manifests/bschema-draw-jev3.pointer.json + bschema-draw-jev3-slice.pointer.json, build-report.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import jev1_build as JB  # noqa: E402

BUILD, SLICE = "bschema-draw-jev3", "bschema-draw-jev3-slice"
N_TRAIN, N_SLICE = 1000, 200
T_WIN = {"long_policy": (2000, 4096)}
T_DEFAULT = (0, 4096)
#: FAMILY_SPEC_JEV3 type-mix targets where the candidate pool's own mix departs from the spec
TYPE_MIX = {"long_policy": {"choice": 0.60, "noul": 0.28, "score": 0.12}}
RECEIPT_ROOT = "receipts/2026-10-11-jev3-draw"


def state_key(st) -> str:
    s = st if isinstance(st, str) else JB.canonical(st)
    return hashlib.sha256(re.sub(r"\s+", " ", s).strip().casefold().encode()).hexdigest()


def rk(rid: str) -> str:
    return hashlib.sha256(rid.encode()).hexdigest()


def cmd_gen(a) -> int:
    from data import augment as A, jev3
    w = Path(a.work)
    w.mkdir(parents=True, exist_ok=True)
    fams = tuple(a.families or jev3.FAMILIES)
    out = w / f"pool-{'-'.join(fams)}.jsonl.gz" if a.families else w / "pool.jsonl.gz"
    ids, c = set(), collections.Counter()
    h = hashlib.sha256()
    with JB._gz(out) as fo:
        for r in jev3.generate(fams):
            A.validate(r)
            if r["id"] in ids:
                raise SystemExit(f"duplicate id {r['id']}")
            ids.add(r["id"])
            line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            fo.write(line + "\n")
            h.update((line + "\n").encode())
            c[r["aug"]["jev1"]["family8"]] += 1
    rep = {"n": len(ids), "by_family": dict(c), "pool_uncompressed_sha256": h.hexdigest(), "plan": jev3.PLAN,
           "seed": jev3.SEED, "families": fams, "file": out.name}
    (w / f"gen-report{'-' + '-'.join(fams) if a.families else ''}.json").write_text(json.dumps(rep, indent=1) + "\n",
                                                                                     encoding="utf-8")
    print(json.dumps(rep))
    return 0


def cmd_merge(a) -> int:
    """Concatenate per-family pools (gen --families) into <work>/pool.jsonl.gz in the canonical family order."""
    from data import jev3
    w = Path(a.work)
    h, n = hashlib.sha256(), collections.Counter()
    reps = {}
    with JB._gz(w / "pool.jsonl.gz") as fo:
        for f in jev3.FAMILIES:
            p = w / f"pool-{f}.jsonl.gz"
            reps[f] = json.loads((w / f"gen-report-{f}.json").read_text(encoding="utf-8"))
            for line in JB._lines(p):
                fo.write(line + "\n")
                h.update((line + "\n").encode())
                n[f] += 1
    rep = {"n": sum(n.values()), "by_family": dict(n), "pool_uncompressed_sha256": h.hexdigest(), "plan": jev3.PLAN,
           "seed": jev3.SEED, "per_family_gen": reps}
    (w / "gen-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n", "by_family", "pool_uncompressed_sha256")}))
    return 0


def tq(xs, win):
    xs = sorted(xs)
    if not xs:
        return None
    hist = collections.Counter(min(x // 500, 9) * 500 for x in xs)
    return {"n": len(xs), "min": xs[0], "p10": xs[int(0.1 * (len(xs) - 1))], "p50": xs[len(xs) // 2],
            "p90": xs[int(0.9 * (len(xs) - 1))], "p99": xs[int(0.99 * (len(xs) - 1))], "max": xs[-1],
            "t_window": list(win), "histogram_500": {f"{k}-{k + 499}": v for k, v in sorted(hist.items())}}


def cmd_final(a) -> int:
    from data import config, jev3
    from train.cohort import record_content_hash

    w = Path(a.work)
    lens, ident = {}, None
    for f in a.lens:
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if "_identity" in d:
                ident = d
                continue
            lens[d["id"]] = d
    drop = set()
    for f in a.drop_ids or []:
        drop |= {x.strip() for x in Path(f).read_text(encoding="utf-8").split() if x.strip()}
    st = collections.defaultdict(collections.Counter)
    by_fam = collections.defaultdict(list)
    for line in JB._lines(w / "checked.jsonl.gz"):
        r = json.loads(line)
        fam = r["aug"]["jev1"]["family8"]
        d = lens.get(r["id"])
        if d is None:
            raise SystemExit(f"{r['id']}: no token length")
        lo, hi = T_WIN.get(fam, T_DEFAULT)
        if "error" in d:
            st[fam]["shim_row_refused"] += 1
            continue
        if d["t_shim"] > hi:
            st[fam][f"over_{hi}"] += 1
            continue
        if d["t_shim"] < lo:
            st[fam][f"under_{lo}"] += 1
            continue
        if r["id"] in drop:
            st[fam]["dropped_by_draw_gate"] += 1
            continue
        by_fam[fam].append((rk(r["id"]), r, line))
    # 2. draw-wide STATE dedupe (all families, rank order inside a family; families in canonical order)
    seen_state: dict[str, str] = {}
    for fam in jev3.FAMILIES:
        kept = []
        for x in sorted(by_fam[fam], key=lambda t: t[0]):
            k = state_key(x[1]["state"])
            if k in seen_state:
                st[fam]["state_dup_dropped"] += 1
                continue
            seen_state[k] = x[1]["id"]
            kept.append(x)
        st[fam]["after_state_dedupe"] = len(kept)
        by_fam[fam] = kept
    # 3. per family: 1,200 rows with the noul balance
    N = N_TRAIN + N_SLICE
    chosen = {}
    for fam in jev3.FAMILIES:
        xs = by_fam[fam]
        T = [x for x in xs if x[1]["questions"]["q"]["type"] == "noul" and x[1]["labels"]["q"] == "true"]
        F = [x for x in xs if x[1]["questions"]["q"]["type"] == "noul" and x[1]["labels"]["q"] == "false"]
        O = [x for x in xs if x[1]["questions"]["q"]["type"] != "noul"]
        if fam in TYPE_MIX:  # spec type-mix target (long_policy: the pool's view rotation over-supplies noul / score)
            mix = TYPE_MIX[fam]
            C_ = [x for x in O if x[1]["questions"]["q"]["type"] == "choice"]
            S_ = [x for x in O if x[1]["questions"]["q"]["type"] == "score"]
            n_c = min(len(C_), round(N * mix["choice"]))
            n_s = min(len(S_), round(N * mix["score"]))
            n_noul = min(len(T) + len(F), N - n_c - n_s)
            n_t = min(len(T), max(round(0.5 * n_noul), n_noul - len(F)))
            n_f = min(len(F), n_noul - n_t)
            rest = N - n_c - n_s - n_t - n_f  # shortfall: more choice first, then score
            add_c = min(rest, len(C_) - n_c)
            add_s = min(rest - add_c, len(S_) - n_s)
            sel = T[:n_t] + F[:n_f] + C_[:n_c + add_c] + S_[:n_s + add_s]
        else:
            share = (len(T) + len(F)) / max(1, len(xs))
            n_noul = min(len(T) + len(F), round(min(N, len(xs)) * share))
            n_t = min(len(T), max(round(0.5 * n_noul), n_noul - len(F)))
            n_f = min(len(F), n_noul - n_t)
            n_o = min(len(O), N - n_t - n_f)
            sel = T[:n_t] + F[:n_f] + O[:n_o]
        st[fam].update({"pool_noul_true": len(T), "pool_noul_false": len(F), "pool_other": len(O),
                        "selected": len(sel)})
        chosen[fam] = sorted(sel, key=lambda t: t[0])
    # 4. slice by group
    rows = {"train": [], "slice": []}
    fam_split = {}
    for fam in jev3.FAMILIES:
        groups = collections.defaultdict(list)
        for x in chosen[fam]:
            groups[x[1]["aug"]["jev1"]["group"]].append(x)
        order = sorted(groups, key=lambda g: hashlib.sha256(f"jev3-slice|{g}".encode()).hexdigest())
        sl, tr, dropped = [], [], 0
        for g in order:
            if len(sl) < N_SLICE:
                take = groups[g][: N_SLICE - len(sl)]
                dropped += len(groups[g]) - len(take)
                sl += take
            else:
                tr += groups[g]
        tr.sort(key=lambda t: t[0])
        st[fam]["slice_group_overflow_dropped"] = dropped
        if len(tr) > N_TRAIN:
            st[fam]["train_unused_over_target"] = len(tr) - N_TRAIN
            tr = tr[:N_TRAIN]
        st[fam]["train"], st[fam]["slice"] = len(tr), len(sl)
        rows["train"] += tr
        rows["slice"] += sl
        fam_split[fam] = {"train": tr, "slice": sl}
    # 5. disjointness (fail closed)
    def keys(xs):
        return {"id": {x[1]["id"] for x in xs},
                "source_item": {x[1]["aug"]["b1_view"]["parent_id"] for x in xs},
                "group": {x[1]["aug"]["jev1"]["group"] for x in xs},
                "content": {record_content_hash(x[1]) for x in xs},
                "state": {state_key(x[1]["state"]) for x in xs}}
    kt, ks = keys(rows["train"]), keys(rows["slice"])
    disjoint = {k: len(kt[k] & ks[k]) for k in kt}
    if any(disjoint.values()):
        raise SystemExit(f"slice / train overlap: {disjoint}")
    within = {role: len(xs) - len({state_key(x[1]['state']) for x in xs}) for role, xs in rows.items()}
    if any(within.values()):
        raise SystemExit(f"duplicate state inside a split: {within}")
    out_dirs = {"train": config.BUILD_DIR / BUILD, "slice": config.BUILD_DIR / SLICE}
    shas, ids_sha = {}, {}
    for role, d in out_dirs.items():
        if (d / "records.jsonl.gz").exists() and not a.overwrite:
            raise SystemExit(f"refusing to overwrite {d}")
        d.mkdir(parents=True, exist_ok=True)
        name = d.name
        xs = sorted(rows[role], key=lambda t: t[1]["id"])
        hs = [hashlib.sha256() for _ in range(4)]
        with JB._gz(d / "records.jsonl.gz") as fr, JB._gz(d / f"{name}.manifest.jsonl.gz") as fm, \
                JB._gz(d / f"{name}.content.jsonl.gz") as fc, JB._gz(d / f"{name}.lens.jsonl.gz") as fl:
            for _, r, line in xs:
                parts = (line + "\n",
                         JB.canonical({"family": r["family"], "id": r["id"], "sha256": hashlib.sha256(line.encode()).hexdigest(),
                                       "source": r["source"]}) + "\n",
                         JB.canonical({"content_sha256": record_content_hash(r), "id": r["id"]}) + "\n",
                         JB.canonical({"family8": r["aug"]["jev1"]["family8"], "id": r["id"],
                                       "t_shim": lens[r["id"]]["t_shim"]}) + "\n")
                for f, h, p in zip((fr, fm, fc, fl), hs, parts):
                    f.write(p)
                    h.update(p.encode())
        shas[role] = dict(zip(("records", "manifest", "content", "lens"), (h.hexdigest() for h in hs)))
        ids_sha[role] = hashlib.sha256("\n".join(sorted(x[1]["id"] for x in xs)).encode()).hexdigest()
    # reports
    fam_rep = {}
    for fam in jev3.FAMILIES:
        rep = {}
        for role in ("train", "slice"):
            xs = fam_split[fam][role]
            types, K, lab = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
            src, lic, subs, pos, toks, strs = collections.Counter(), collections.Counter(), collections.Counter(), \
                collections.defaultdict(collections.Counter), [], 0
            for _, r, _ in xs:
                q, g = r["questions"]["q"], r["labels"]["q"]
                types[q["type"]] += 1
                src[r["source"]] += 1
                lic[f"{r['source']}: {r['license']}"] += 1
                subs[r["aug"]["jev1"]["sub"]] += 1
                toks.append(lens[r["id"]]["t_shim"])
                strs += isinstance(r["state"], str)
                if q["type"] == "choice":
                    ids = list(q["criteria"])
                    K[len(ids)] += 1
                    pos[len(ids)][ids.index(g)] += 1
                else:
                    lab[q["type"]][g] += 1
            rep[role] = {"rows": len(xs), "types": dict(types), "choice_K": dict(sorted(K.items())),
                         "label_rates": {t: {k: round(v / sum(c.values()), 4) for k, v in sorted(c.items())}
                                         for t, c in lab.items()},
                         "sources": dict(src), "licenses": dict(lic), "subfamilies": dict(sorted(subs.items())),
                         "string_state_share": round(strs / max(1, len(xs)), 4),
                         "gold_position_chi2_presented": JB.chi(pos),
                         "tokens_shim_qwen35_4b": tq(toks, T_WIN.get(fam, T_DEFAULT))}
        rep["filter"] = dict(st[fam])
        fam_rep[fam] = rep
    gate_pos = {f: (fam_rep[f]["train"]["gold_position_chi2_presented"]["p"] or 1.0) >= 0.001 for f in jev3.FAMILIES}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    rd = Path(a.receipt)
    rd.mkdir(parents=True, exist_ok=True)
    gen_rep = json.loads((w / "gen-report.json").read_text(encoding="utf-8"))
    chk_rep = json.loads((w / "check-report.json").read_text(encoding="utf-8"))
    dedupe = {f: st[f]["state_dup_dropped"] for f in jev3.FAMILIES}
    report = {"build": BUILD, "slice": SLICE, "n_train": len(rows["train"]), "n_slice": len(rows["slice"]),
              "families": fam_rep, "state_dedupe_dropped_by_family": dedupe,
              "slice_train_disjoint_counts": disjoint, "within_split_duplicate_states": within,
              "gold_position_gate_p_ge_0.001": gate_pos,
              "token_template": {"template": "jrt-h2o-shim:qwen3.5-4b", "identity": ident, "windows": {
                  f: list(T_WIN.get(f, T_DEFAULT)) for f in jev3.FAMILIES}},
              "uncompressed_sha256": shas, "ids_sha256": ids_sha, "pool": gen_rep,
              "check": {k: chk_rep[k] for k in ("n_pool", "n_kept", "checked_uncompressed_sha256", "reasons")},
              "drop_ids_files": {str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in a.drop_ids or []},
              "rule": __doc__.split("final, per family")[1].split("Writes")[0].strip(), "commit": commit}
    (rd / "build-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    ptrs = {}
    for role, d in out_dirs.items():
        name = d.name
        ptr = {"build": f"{name} (data/jev3.py + scripts/jev3_build.py: HUMAN-labelled TRAIN-split rows for judge_hard, "
                        f"adequacy, policy, ambiguous, temporal_numeric, long_policy; role {role})",
               "role": "train" if role == "train" else "held-out selection slice (checkpoint selection only; never "
                                                       "trained on)",
               "location": f"ssdm-data/m2/build/{name}/ (outside git; deterministic, a rerun reproduces it)",
               "records_uncompressed_sha256": shas[role]["records"], "manifest": f"{name}.manifest.jsonl.gz",
               "manifest_uncompressed_sha256": shas[role]["manifest"], "content_map": f"{name}.content.jsonl.gz",
               "content_map_uncompressed_sha256": shas[role]["content"], "lens": f"{name}.lens.jsonl.gz",
               "lens_uncompressed_sha256": shas[role]["lens"],
               "ids_sha256": ids_sha[role], "ids_sha256_rule": "sha256 of the sorted record ids, newline-joined (no "
                                                                "trailing newline)",
               "n_records": len(rows[role]), "families": {f: len(fam_split[f][role]) for f in jev3.FAMILIES},
               "schema": "ssdm-train-v0, one question per row (qid 'q'), JevBench request shape; jev1 row layout "
                         "(aug.jev1.family8; aug.jev1.prov = per-row dataset / revision / licence / train split / row key)",
               "loader_contract": "option keys are NOT re-sorted (seeded per-row shuffle = presented order); no state "
                                  "text (normalised) is shared by any two rows of train + slice",
               "slice_disjoint": {"rule": "slice vs train: 0 shared ids, source items, groups, content hashes, "
                                          "normalised states", "counts": disjoint, "slice_ids_sha256": ids_sha["slice"]},
               "excludes_cohort": "m2-dev-cohort-v2 be6efe2b... and m2-dev-cohort-v3 498efa61..., Index DEV/TEST "
                                  "rows, JevBench public items (eval-only), every jev1 source item of HelpSteer2 / "
                                  "HS2-preference / Aegis2, MATH test problems (PRM800K)",
               "builder": {"module": "scripts/jev3_build.py", "generator": "data/jev3.py", "commit": commit},
               "receipt": f"{RECEIPT_ROOT}/", "templates_verified": ["jrt-h2o-shim:qwen3.5-4b"],
               "token_windows": {f: list(T_WIN.get(f, T_DEFAULT)) for f in jev3.FAMILIES}}
        pp = config.MANIFEST_DIR / f"{name}.pointer.json"
        pp.write_bytes((json.dumps(ptr, indent=1) + "\n").encode("utf-8"))
        ptrs[name] = hashlib.sha256(pp.read_bytes()).hexdigest()
    print(json.dumps({"n_train": len(rows["train"]), "n_slice": len(rows["slice"]),
                      "rows": {f: {r_: len(fam_split[f][r_]) for r_ in ("train", "slice")} for f in jev3.FAMILIES},
                      "state_dedupe": dedupe, "disjoint": disjoint, "gate_pos": gate_pos,
                      "filter": {f: dict(st[f]) for f in jev3.FAMILIES}, "shas": shas, "ids_sha": ids_sha,
                      "pointer_sha256": ptrs}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--families", nargs="*")
    m = sub.add_parser("merge")
    m.add_argument("--work", required=True)
    c = sub.add_parser("check")
    c.add_argument("--work", required=True)
    f = sub.add_parser("final")
    f.add_argument("--work", required=True)
    f.add_argument("--lens", nargs="+", required=True)
    f.add_argument("--drop-ids", nargs="*")
    f.add_argument("--receipt", required=True)
    f.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    return {"gen": cmd_gen, "merge": cmd_merge, "check": JB.cmd_check, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
