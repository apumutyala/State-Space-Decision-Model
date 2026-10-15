"""Build jev-heldout (data/jev_heldout.py, FAMILY_SPEC_HELDOUT.md, PREREG EV-PREREG-JEV-HELDOUT): gen -> merge -> check
(jev1 cmd_check, unchanged) -> pre-gates (drop ids) -> final (quotas, seal). Prints counts and shas only, never item text.

    python -X utf8 -I scripts/jevho_build.py gen   --work W --families f [f ...]
    python -X utf8 -I scripts/jevho_build.py merge --work W
    python -X utf8 -I scripts/jevho_build.py check --work W          (jev1 cmd_check: waits for >= 1 GB free RAM)
    python -X utf8 -I scripts/jevho_build.py final --work W --lens L [L ...] --drop-ids D [D ...] --receipt R
        [--stage] [--overwrite]

final, per family (label-blind sha256 rank of the id, then the stated quotas):
  1. token window: long_policy 2,000-4,096 shim tokens, every other family <= 4,096 (rows without a length are dropped)
  2. drop-ids from the pre-gates (disjointness vs every jev draw, provenance, strict full-row Index fp, verbatim leaves,
     doc_overlap, attack_guard, jevbench rescan)
  3. set-wide STATE dedupe (normalised state text, all families, canonical family order, first by rank keeps it)
  4. K filter: a choice row's K must be in the public K support of its family
  5. type quotas (TYPE_QUOTA); noul quota balanced 45-55 % true where both strata allow; a short type is filled from
     the family's other types in rank order (reported)
  6. tier: label-blind sha256('jevho-tier|<id>') in the public tier proportions (72 standard / 48 easy / 111 hard)
Writes ssdm-data/m2/build/jev-heldout/ (records, items, manifest, lens, SEAL.json, ACCESS.log) and the pointer
data/manifests/jev-heldout.pointer.json; with --stage, jev-heldout-stage/ + jev-heldout-stage.pointer.json instead
(role "stage, NOT sealed for use").
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
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

FP = REPO / "receipts" / "2026-10-10-jev2-draw" / "public-shape-fingerprint.json"
T_WIN = {"long_policy": (2000, 4096)}
T_DEFAULT = (0, 4096)
TIER_SHARE = (("standard", 72), ("easy", 48), ("hard", 111))


def state_key(st) -> str:
    s = st if isinstance(st, str) else JB.canonical(st)
    return hashlib.sha256(re.sub(r"\s+", " ", s).strip().casefold().encode()).hexdigest()


def rk(rid: str) -> str:
    return hashlib.sha256(rid.encode()).hexdigest()


def type_quota() -> dict:
    from data import jev_heldout as H
    fp = json.loads(FP.read_text(encoding="utf-8"))["families"]
    out = {}
    for f, n in H.TARGET.items():
        ty = fp[f]["types"]
        tot = sum(ty.values())
        q = {t: round(n * c / tot) for t, c in ty.items()}
        d = n - sum(q.values())
        q[max(q, key=q.get)] += d
        out[f] = {"types": q, "K": sorted(int(k) for k in fp[f]["choice_K"]), "string_share": fp[f]["string_state_share"]}
    return out


def cmd_gen(a) -> int:
    from data import augment as A, jev_heldout as H
    w = Path(a.work)
    w.mkdir(parents=True, exist_ok=True)
    for f in a.families:
        out = w / f"pool-{f}.jsonl.gz"
        ids, c = set(), collections.Counter()
        h = hashlib.sha256()
        with JB._gz(out) as fo:
            for r in H.generate((f,)):
                A.validate(r)
                if r["id"] in ids:
                    raise SystemExit(f"duplicate id {r['id']}")
                ids.add(r["id"])
                line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
                fo.write(line + "\n")
                h.update((line + "\n").encode())
                q = r["questions"]["q"]
                c[q["type"] + (f":K{len(q['criteria'])}" if q["type"] == "choice" else "")] += 1
        rep = {"family": f, "n": len(ids), "by_type": dict(sorted(c.items())), "pool_uncompressed_sha256": h.hexdigest(),
               "plan": H.PLAN[f], "seed": H.SEED, "file": out.name}
        (w / f"gen-report-{f}.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(rep), flush=True)
    return 0


def cmd_merge(a) -> int:
    from data import jev_heldout as H
    w = Path(a.work)
    h, n, reps = hashlib.sha256(), collections.Counter(), {}
    with JB._gz(w / "pool.jsonl.gz") as fo:
        for f in H.FAMILIES:
            p = w / f"pool-{f}.jsonl.gz"
            if not p.exists():
                continue
            reps[f] = json.loads((w / f"gen-report-{f}.json").read_text(encoding="utf-8"))
            for line in JB._lines(p):
                fo.write(line + "\n")
                h.update((line + "\n").encode())
                n[f] += 1
    rep = {"n": sum(n.values()), "by_family": dict(n), "pool_uncompressed_sha256": h.hexdigest(), "seed": H.SEED,
           "families_missing": [f for f in H.FAMILIES if f not in n], "per_family_gen": reps}
    (w / "gen-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n", "by_family", "pool_uncompressed_sha256", "families_missing")}))
    return 0


def to_item(r: dict) -> dict:
    """ssdm-train-v0 row -> jevkit Item dict (split 'heldout' = the sealed side)."""
    q, g = r["questions"]["q"], r["labels"]["q"]
    question = {"type": q["type"], "instructions": q["instructions"]}
    if q.get("criteria") is not None:
        question["criteria"] = q["criteria"]
    if q["type"] == "choice":
        labels, expected = list(q["criteria"]), g
    elif q["type"] == "noul":
        labels, expected = ["no", "yes"], ("yes" if g == "true" else "no")
    else:
        labels, expected = [str(i) for i in range(len(q["criteria"]))], int(g)
    hsh = int(hashlib.sha256(f"jevho-tier|{r['id']}".encode()).hexdigest()[:8], 16) % 231
    acc, tier = 0, "hard"
    for t, c in TIER_SHARE:
        acc += c
        if hsh < acc:
            tier = t
            break
    j = r["aug"]["jev1"]
    return {"id": r["id"], "family": j["family8"], "state": r["state"], "question": question, "labels": labels,
            "expected": expected, "split": "heldout", "tier": tier, "group": j.get("group"),
            "provenance": {"source": r["source"], "source_split": r["source_split"], "license": r["license"],
                           "draw": j["draw"], "prov": j.get("prov")}}


def cmd_final(a) -> int:
    from data import config, jev_heldout as H
    from jevkit.items import Item
    from train.cohort import record_content_hash

    w = Path(a.work)
    TQ = type_quota()
    lens, ident = {}, None
    for f in a.lens:
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if "_identity" in d:
                ident = d
                continue
            lens[d["id"]] = d
    drop, drop_src = set(), {}
    for f in a.drop_ids or []:
        ids = {x.strip() for x in Path(f).read_text(encoding="utf-8").split() if x.strip()}
        drop |= ids
        drop_src[str(f)] = {"n": len(ids), "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()}
    st = collections.defaultdict(collections.Counter)
    by_fam = collections.defaultdict(list)
    src = w / "checked.jsonl.gz" if (w / "checked.jsonl.gz").exists() and not a.unchecked else w / "pool.jsonl.gz"
    for line in JB._lines(src):
        r = json.loads(line)
        fam = r["aug"]["jev1"]["family8"]
        st[fam]["in"] += 1
        d = lens.get(r["id"])
        lo, hi = T_WIN.get(fam, T_DEFAULT)
        if d is None or "error" in d:
            st[fam]["no_token_length"] += 1
            continue
        if not lo <= d["t_shim"] <= hi:
            st[fam]["outside_token_window"] += 1
            continue
        if r["id"] in drop:
            st[fam]["dropped_by_pre_gate"] += 1
            continue
        q = r["questions"]["q"]
        if q["type"] not in TQ[fam]["types"]:
            st[fam]["type_not_in_public"] += 1
            continue
        if q["type"] == "choice" and len(q["criteria"]) not in TQ[fam]["K"]:
            st[fam]["K_not_in_public"] += 1
            continue
        by_fam[fam].append((rk(r["id"]), r, line))
    seen_state: dict[str, str] = {}
    for fam in H.FAMILIES:
        kept = []
        for x in sorted(by_fam[fam], key=lambda t: t[0]):
            k = state_key(x[1]["state"])
            if k in seen_state:
                st[fam]["state_dup_dropped"] += 1
                continue
            seen_state[k] = x[1]["id"]
            kept.append(x)
        by_fam[fam] = kept
    chosen, short = {}, {}
    for fam in H.FAMILIES:
        xs = by_fam[fam]
        quota = dict(TQ[fam]["types"])
        pools = {t: [x for x in xs if x[1]["questions"]["q"]["type"] == t] for t in quota}
        sel = []
        for t, n_t in quota.items():
            pool = pools[t]
            if t == "noul":
                T_ = [x for x in pool if x[1]["labels"]["q"] == "true"]
                F_ = [x for x in pool if x[1]["labels"]["q"] == "false"]
                n_n = min(n_t, len(T_) + len(F_))
                n_true = min(len(T_), max(round(0.5 * n_n), n_n - len(F_)))
                n_false = min(len(F_), n_n - n_true)
                take = T_[:n_true] + F_[:n_false]
            else:
                take = pool[:n_t]
            sel += take
            st[fam][f"quota_{t}"] = n_t
            st[fam][f"selected_{t}"] = len(take)
        missing = H.TARGET[fam] - len(sel)
        if missing > 0:  # fill a short type from the family's other types (rank order), reported
            used = {x[1]["id"] for x in sel}
            rest = [x for x in xs if x[1]["id"] not in used]
            fill = rest[:missing]
            sel += fill
            st[fam]["filled_from_other_types"] = len(fill)
        if len(sel) < 0.7 * H.TARGET[fam]:
            short[fam] = {"rows": len(sel), "target": H.TARGET[fam]}
        st[fam]["selected"] = len(sel)
        chosen[fam] = sorted(sel, key=lambda t: t[0])
    rows = [x for f in H.FAMILIES for x in chosen[f]]
    within = len(rows) - len({state_key(x[1]["state"]) for x in rows})
    if within:
        raise SystemExit(f"duplicate states inside the set: {within}")
    # shape / tolerance gate (two-sided, prereg) -------------------------------------------------------------------
    tol = {}
    for fam in H.FAMILIES:
        xs = chosen[fam]
        n = len(xs)
        types = collections.Counter(x[1]["questions"]["q"]["type"] for x in xs)
        q = TQ[fam]
        pub_share = {t: c / sum(q["types"].values()) for t, c in q["types"].items()}
        ss = sum(isinstance(x[1]["state"], str) for x in xs) / max(1, n)
        nl = [x for x in xs if x[1]["questions"]["q"]["type"] == "noul"]
        tshare = (sum(x[1]["labels"]["q"] == "true" for x in nl) / len(nl)) if nl else None
        tol[fam] = {"n": n, "target": H.TARGET[fam], "count_ok": abs(n - H.TARGET[fam]) <= 0.15 * H.TARGET[fam],
                    "types": dict(types), "type_share_ok": all(abs(types.get(t, 0) / max(1, n) - s) <= 0.10
                                                               for t, s in pub_share.items()),
                    "types_subset": set(types) <= set(q["types"]),
                    "string_share": round(ss, 4), "string_share_public": q["string_share"],
                    "string_share_ok": abs(ss - q["string_share"]) <= 0.15,
                    "noul_true_share": None if tshare is None else round(tshare, 4),
                    "noul_criteria_all": all(x[1]["questions"]["q"].get("criteria") for x in nl)}
    # gold position chi2 on presented order
    pos_rep = {}
    for fam in H.FAMILIES:
        pos = collections.defaultdict(collections.Counter)
        for _, r, _ in chosen[fam]:
            q = r["questions"]["q"]
            if q["type"] == "choice":
                ids = list(q["criteria"])
                pos[len(ids)][ids.index(r["labels"]["q"])] += 1
        pos_rep[fam] = JB.chi(pos) if pos else {"p": None}
    gate_pos = {f: (pos_rep[f].get("p") or 1.0) >= 0.001 for f in H.FAMILIES}
    # write ----------------------------------------------------------------------------------------------------
    name = "jev-heldout-stage" if a.stage else "jev-heldout"
    d = config.BUILD_DIR / name
    if (d / "records.jsonl.gz").exists() and not a.overwrite:
        raise SystemExit(f"refusing to overwrite {d}")
    d.mkdir(parents=True, exist_ok=True)
    xs = sorted(rows, key=lambda t: t[1]["id"])
    hs = {k: hashlib.sha256() for k in ("records", "manifest", "lens", "items")}
    with JB._gz(d / "records.jsonl.gz") as fr, JB._gz(d / f"{name}.manifest.jsonl.gz") as fm, \
            JB._gz(d / f"{name}.lens.jsonl.gz") as fl, open(d / "items.jsonl", "w", encoding="utf-8", newline="\n") as fi:
        for _, r, line in xs:
            it = to_item(r)
            Item.from_dict(it)  # validates (fail closed)
            parts = {"records": line + "\n",
                     "manifest": JB.canonical({"content_sha256": record_content_hash(r), "family": r["aug"]["jev1"]["family8"],
                                               "id": r["id"], "sha256": hashlib.sha256(line.encode()).hexdigest(),
                                               "source": r["source"], "source_split": r["source_split"]}) + "\n",
                     "lens": JB.canonical({"family8": r["aug"]["jev1"]["family8"], "id": r["id"],
                                           "t_shim": lens[r["id"]]["t_shim"]}) + "\n",
                     "items": json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n"}
            for f, k in ((fr, "records"), (fm, "manifest"), (fl, "lens"), (fi, "items")):
                f.write(parts[k])
                hs[k].update(parts[k].encode())
    shas = {k: h.hexdigest() for k, h in hs.items()}
    ids_sorted = sorted(x[1]["id"] for x in xs)
    ids_sha = hashlib.sha256("\n".join(ids_sorted).encode()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fam_counts = {f: len(chosen[f]) for f in H.FAMILIES}
    type_counts = {f: dict(collections.Counter(x[1]["questions"]["q"]["type"] for x in chosen[f])) for f in H.FAMILIES}
    K_counts = {f: dict(sorted(collections.Counter(len(x[1]["questions"]["q"]["criteria"]) for x in chosen[f]
                                                   if x[1]["questions"]["q"]["type"] == "choice").items()))
                for f in H.FAMILIES}
    src_counts = {f: dict(collections.Counter(f"{x[1]['source']}:{x[1]['source_split']}" for x in chosen[f]))
                  for f in H.FAMILIES}
    tiers = dict(collections.Counter(to_item(x[1])["tier"] for x in xs))
    seal = {"set": name, "sealed_utc": utc, "prereg": "EV-PREREG-JEV-HELDOUT (state d48e500c)",
            "role": "stage, NOT sealed for use" if a.stage else "SEALED secondary confirmation set (score at most once "
                                                                "per pre-registered cell; never train / select / tune)",
            "n_items": len(xs), "records_uncompressed_sha256": shas["records"], "items_jsonl_sha256": shas["items"],
            "manifest_uncompressed_sha256": shas["manifest"], "lens_uncompressed_sha256": shas["lens"],
            "ids_sha256": ids_sha, "ids_sha256_rule": "sha256 of the sorted record ids, newline-joined, no trailing "
                                                       "newline", "builder_commit": commit}
    (d / "SEAL.json").write_text(json.dumps(seal, indent=1) + "\n", encoding="utf-8")
    with open(d / "ACCESS.log", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"utc": utc, "agent": "evidence data worker", "purpose": "build + seal (writer)",
                             "prereg": "EV-PREREG-JEV-HELDOUT", "files": ["records.jsonl.gz", "items.jsonl"]}) + "\n")
    rd = Path(a.receipt)
    rd.mkdir(parents=True, exist_ok=True)
    report = {"set": name, "n": len(xs), "families": fam_counts, "types": type_counts, "choice_K": K_counts,
              "sources": src_counts, "tiers": tiers, "filter": {f: dict(st[f]) for f in H.FAMILIES},
              "short_families": short, "tolerance_gate": tol, "gold_position_chi2": pos_rep,
              "gold_position_gate_p_ge_0.001": gate_pos, "type_quota": TQ,
              "token_template": {"identity": ident, "windows": {f: list(T_WIN.get(f, T_DEFAULT)) for f in H.FAMILIES}},
              "drop_ids_files": drop_src, "seal": seal, "input": src.name,
              "rule": __doc__.split("final, per family")[1].split("Writes")[0].strip()}
    (rd / f"{'stage-' if a.stage else ''}build-report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    ptr = {"build": f"{name} (data/jev_heldout.py + scripts/jevho_build.py; PREREG EV-PREREG-JEV-HELDOUT)",
           "role": seal["role"], "location": f"ssdm-data/m2/build/{name}/ (outside git; deterministic from the pinned "
                                             "sources and this commit)",
           "n_records": len(xs), "families": fam_counts, "types": type_counts, "choice_K": K_counts, "tiers": tiers,
           "records_uncompressed_sha256": shas["records"], "items_jsonl_sha256": shas["items"],
           "manifest": f"{name}.manifest.jsonl.gz", "manifest_uncompressed_sha256": shas["manifest"],
           "lens": f"{name}.lens.jsonl.gz", "lens_uncompressed_sha256": shas["lens"], "ids_sha256": ids_sha,
           "ids_sha256_rule": seal["ids_sha256_rule"], "seal_file": "SEAL.json", "access_log": "ACCESS.log",
           "use_rule": "secondary confirmation only; scored at most once per pre-registered cell with jevkit score "
                       "--items items.jsonl; never training, checkpoint selection, temperature / calibration fit or "
                       "tuning; every read is appended to ACCESS.log",
           "builder": {"generator": "data/jev_heldout.py", "builder": "scripts/jevho_build.py", "commit": commit},
           "receipt": (str(rd.relative_to(REPO)) if rd.is_relative_to(REPO) else str(rd)).replace("\\", "/") + "/"}
    pp = config.MANIFEST_DIR / f"{name}.pointer.json"
    pp.write_bytes((json.dumps(ptr, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({"set": name, "n": len(xs), "families": fam_counts, "types": type_counts, "short": short,
                      "records_sha256": shas["records"], "items_sha256": shas["items"], "ids_sha256": ids_sha,
                      "pointer_sha256": hashlib.sha256(pp.read_bytes()).hexdigest(),
                      "tolerance_fail": {f: [k for k, v in t.items() if k.endswith("_ok") or k == "types_subset"
                                             if v is False] for f, t in tol.items()},
                      "gate_pos_fail": [f for f, v in gate_pos.items() if not v]}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--families", nargs="+", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--work", required=True)
    c = sub.add_parser("check")
    c.add_argument("--work", required=True)
    f = sub.add_parser("final")
    f.add_argument("--work", required=True)
    f.add_argument("--lens", nargs="+", required=True)
    f.add_argument("--drop-ids", nargs="*")
    f.add_argument("--receipt", required=True)
    f.add_argument("--stage", action="store_true")
    f.add_argument("--unchecked", action="store_true", help="stage only: build from pool.jsonl.gz (check not run)")
    f.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "final" and a.unchecked and not a.stage:
        raise SystemExit("--unchecked is only allowed with --stage")
    return {"gen": cmd_gen, "merge": cmd_merge, "check": JB.cmd_check, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
