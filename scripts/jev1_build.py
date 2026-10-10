"""Build bschema-draw-jev1 (data/jev1.py) in streamed stages; every stage is deterministic and writes a receipt.

  gen    data.jev1.generate() -> <work>/pool.jsonl.gz (structure-validated: data.augment.validate, unique ids)
  check  per row, in generation order (first owner wins the dedup):
           gold_once        choice gold in the criteria and its normalised text unique among the options
           checker          data.b_schema.Checker (the v4 P3 row gate): structure + D1-B templates, cohort v2 / v3
                            (498efa61...) id / source item / content / key-free text, within-draw exact content and
                            cross-item near-duplicate text, SplitGuard, harness.heldout frozen families, Index DEV/TEST
                            content / text / item text
           index_fp         production data.dedup.Fingerprint (index-fp-v3; exact / ngram13 / minhash / edit /
                            contain) on the row's ITEM-tier texts (aug.jev1.item), as the v4 lever rows
           jevbench_fp      train.jevbench_guard fingerprint (all five matchers) + exact-segment gate on the FULL row
                            (state + questions): JevBench public items are eval-only
           attack           train.attack_guard.record_hits (eval attack templates)
         -> <work>/checked.jsonl.gz + check-report.json
  final  drop rows over T_MAX shim tokens (scripts/jev1_tokcheck.py lens) or refused by the shim row builder, drop
         ``--drop-ids`` (later draw-level gates), trim each family to its target in sha256(id) order (label-blind), write
         the draw dir (records / manifest / content, deterministic gzip), the pointer and the receipt files
         (build-report.json: per family sources, licences, rows, type mix, label rates, gold-once + gold-position chi2
         on the PRESENTED option order, token stats; target_provenance.json)

    python -X utf8 -I scripts/jev1_build.py gen --work <dir>
    python -X utf8 -I scripts/jev1_build.py check --work <dir>
    python -X utf8 -I scripts/jev1_build.py final --work <dir> --lens <lens.jsonl> [--drop-ids f ...] --receipt <dir>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BUILD = "bschema-draw-jev1"
T_MAX = 2048
#: per-family row target: the brief's 50-150k draw balanced over the 8 JevBench families -> 96,000 / 8 = 12,000 per
#: family (mid-range; no rate or count from any JevBench item or eval row)
TARGET = 12000


def _lines(p):
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line.rstrip("\n")


def _gz(p):
    import io
    return io.TextIOWrapper(gzip.GzipFile(p, "wb", mtime=0), encoding="utf-8", newline="\n")


def canonical(v) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def norm(s) -> str:
    import re
    return re.sub(r"\W+", " ", str(s).casefold()).strip()


def gold_once(r: dict) -> bool:
    q, g = r["questions"]["q"], r["labels"]["q"]
    if q["type"] == "noul":
        return g in ("true", "false")
    if q["type"] == "score":
        return 0 <= int(g) < len(q["criteria"])
    t = {o: norm(d if d not in (None, "") else o) for o, d in q["criteria"].items()}
    return g in t and sum(1 for o in t if t[o] == t[g]) == 1


def p_chi2(x: float, k: int) -> float:  # Wilson-Hilferty upper tail
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi(d: dict) -> dict:
    x = df = n = 0
    for K, c in d.items():
        m = sum(c.values())
        if m < 5 * K:
            continue
        e = m / K
        x += sum((c.get(i, 0) - e) ** 2 / e for i in range(K))
        df += K - 1
        n += m
    return {"n": n, "chi2": round(x, 1), "df": df, "chi2_per_df": round(x / df, 3) if df else None,
            "p": round(p_chi2(x, df), 6) if df else None}


def cmd_gen(a) -> int:
    from data import augment as A, jev1
    w = Path(a.work)
    w.mkdir(parents=True, exist_ok=True)
    ids, c = set(), collections.Counter()
    h = hashlib.sha256()
    with _gz(w / "pool.jsonl.gz") as fo:
        for r in jev1.generate():
            A.validate(r)
            if r["id"] in ids:
                raise SystemExit(f"duplicate id {r['id']}")
            ids.add(r["id"])
            # insertion order, never key-sorted: the shim letters options in the request's criteria order, so the
            # seeded per-row option shuffle must survive serialisation (sorted keys put the gold at its id's rank)
            line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            fo.write(line + "\n")
            h.update((line + "\n").encode())
            c[r["aug"]["jev1"]["family8"]] += 1
    rep = {"n": len(ids), "by_family": dict(c), "pool_uncompressed_sha256": h.hexdigest(), "plan": jev1.PLAN,
           "seed": jev1.SEED}
    (w / "gen-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep))
    return 0


def item_probe(r: dict) -> tuple[dict, dict]:
    """ITEM-tier texts for the Index fingerprint: the state keys named in aug.jev1.item_keys (None = whole state)."""
    ks, st = r["aug"]["jev1"].get("item_keys"), r["state"]
    it = {k: st[k] for k in ks} if ks else (st if isinstance(st, dict) else {"state": st})

    def leaf(v):  # a string as is; dialogue turns as their texts; any other structure as ONE canonical string
        if isinstance(v, str):
            return v
        if isinstance(v, list) and v and all(isinstance(x, dict) and "text" in x for x in v):
            return [x["text"] for x in v]
        return canonical(v)
    return {k: leaf(v) for k, v in it.items()}, {}


def cmd_check(a) -> int:
    from data import config
    from data.b1 import exclusion_sets
    from data.b_schema import Checker, h64, root_items
    from data.dedup import Fingerprint
    from train import attack_guard as AG
    from train import jevbench_guard as JG

    w = Path(a.work)
    t0 = time.time()
    JG.wait_for_ram(1024, log=lambda m: print(m, flush=True), poll_s=30)
    ex = exclusion_sets()
    chk = Checker(ex)
    fp = Fingerprint.load(config.FINGERPRINT)
    fp.assert_production()
    _, jfp = JG.load()
    matcher = AG.eval_matcher()
    print(f"loaded gates in {time.time() - t0:.0f}s", flush=True)
    reasons = collections.defaultdict(collections.Counter)
    fp_by = collections.defaultdict(collections.Counter)
    jev_by = collections.Counter()
    jev_ex, atk_ex = [], []
    n = kept = 0
    h = hashlib.sha256()
    with _gz(w / "checked.jsonl.gz") as fo:
        for line in _lines(w / "pool.jsonl.gz"):
            r = json.loads(line)
            n += 1
            fam = r["aug"]["jev1"]["family8"]
            if not gold_once(r):
                reasons[fam]["gold_once"] += 1
                continue
            hit = {k: v for k, v in fp.check(*item_probe(r)).items() if v}
            if hit:
                reasons[fam]["index_fingerprint"] += 1
                for k in hit:
                    fp_by[fam][k] += 1
                continue
            jh = {k: v for k, v in JG.fingerprint_item_hits(jfp, r["state"], r["questions"]).items() if v}
            if JG.exact_segment_hits(jfp, r["state"], r["questions"]):
                jh["jevbench_exact_segment"] = True
            if jh:
                reasons[fam]["jevbench_fingerprint"] += 1
                for k in jh:
                    jev_by[k] += 1
                if len(jev_ex) < 30:
                    jev_ex.append({"id": r["id"], "matchers": sorted(jh)})
                continue
            ah = AG.record_hits(r, matcher)
            if ah:
                reasons[fam]["attack_template"] += 1
                if len(atk_ex) < 30:
                    atk_ex.append({"id": r["id"], "hits": sorted(ah)})
                continue
            if not chk.accept(r, fam, h64(sorted(root_items(r))[0])):
                continue  # Checker counts its own reason under chk.reasons[fam]
            reasons[fam]["kept"] += 1
            fo.write(line + "\n")
            h.update((line + "\n").encode())
            kept += 1
            if len(getattr(jfp, "_seg_cache", ())) > 200_000:
                jfp._seg_cache.clear()
                jfp._txt_cache.clear()
            if len(getattr(fp, "_seg_cache", ())) > 200_000:
                fp._seg_cache.clear()
                fp._txt_cache.clear()
            if n % 10000 == 0:
                print(f"{n} rows, kept {kept}, {time.time() - t0:.0f}s", flush=True)
    for fam, c in chk.reasons.items():
        for k, v in c.items():
            if k != "kept":
                reasons[fam][f"checker:{k}"] += v
    rep = {"n_pool": n, "n_kept": kept, "checked_uncompressed_sha256": h.hexdigest(),
           "reasons": {k: dict(v) for k, v in reasons.items()},
           "index_fingerprint": {"identity": {k: fp.meta.get(k) for k in ("version", "npz_sha256", "arrays_sha256",
                                                                           "rows")},
                                 "hits_by_family_matcher": {k: dict(v) for k, v in fp_by.items()}},
           "jevbench_fingerprint": {"identity": JG.identity(), "hits_by_matcher": dict(jev_by), "examples": jev_ex},
           "attack_guard": {"examples": atk_ex},
           "checker": {"guard_identity": str(chk.guard_identity)[:2000], "hit_examples": chk.hit_examples,
                       "exclusion_identity": ex["identity"]},
           "seconds": round(time.time() - t0)}
    (w / "check-report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_pool", "n_kept", "reasons")}, indent=1))
    return 0


def cmd_final(a) -> int:
    from data import config, jev1
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
    by_fam = collections.defaultdict(list)
    st = collections.defaultdict(collections.Counter)
    for line in _lines(w / "checked.jsonl.gz"):
        r = json.loads(line)
        fam = r["aug"]["jev1"]["family8"]
        d = lens.get(r["id"])
        if d is None:
            raise SystemExit(f"{r['id']}: no token length")
        if "error" in d:
            st[fam]["shim_row_refused"] += 1
            continue
        if d["t_shim"] > T_MAX:
            st[fam]["over_2048"] += 1
            continue
        if r["id"] in drop:
            st[fam]["dropped_by_draw_gate"] += 1
            continue
        st[fam]["fit"] += 1
        by_fam[fam].append((hashlib.sha256(r["id"].encode()).hexdigest(), r["id"], line))
    rows = []
    for fam in jev1.FAMILIES:
        xs = sorted(by_fam[fam])
        if len(xs) > TARGET:
            st[fam]["trimmed"] = len(xs) - TARGET
            xs = xs[:TARGET]
        st[fam]["kept"] = len(xs)
        rows += [(rid, line) for _, rid, line in xs]
    rows.sort()
    out = config.BUILD_DIR / BUILD
    if (out / "records.jsonl.gz").exists() and not a.overwrite:
        raise SystemExit(f"refusing to overwrite {out} (pass --overwrite for a rebuild of this unpublished draw)")
    out.mkdir(parents=True, exist_ok=True)
    rec_h, man_h, con_h = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    fams = collections.defaultdict(lambda: {"rows": 0, "sources": collections.Counter(), "subs": collections.Counter(),
                                            "types": collections.Counter(), "labels": collections.defaultdict(
                                                collections.Counter), "tokens": [], "licenses": collections.Counter()})
    pos = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    pos_sorted = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    gold_fail = 0
    with _gz(out / "records.jsonl.gz") as fr, _gz(out / f"{BUILD}.manifest.jsonl.gz") as fm, \
            _gz(out / f"{BUILD}.content.jsonl.gz") as fc:
        for rid, line in rows:
            r = json.loads(line)
            fr.write(line + "\n")
            rec_h.update((line + "\n").encode())
            m = canonical({"family": r["family"], "id": rid, "sha256": hashlib.sha256(line.encode()).hexdigest(),
                           "source": r["source"]}) + "\n"
            fm.write(m)
            man_h.update(m.encode())
            c = canonical({"content_sha256": record_content_hash(r), "id": rid}) + "\n"
            fc.write(c)
            con_h.update(c.encode())
            j = r["aug"]["jev1"]
            F = fams[j["family8"]]
            q, g = r["questions"]["q"], r["labels"]["q"]
            F["rows"] += 1
            F["sources"][r["source"]] += 1
            F["subs"][j["sub"]] += 1
            F["types"][q["type"]] += 1
            F["licenses"][f"{r['source']}: {r['license']}"] += 1
            F["tokens"].append(lens[rid]["t_shim"])
            if not gold_once(r):
                gold_fail += 1
            if q["type"] == "choice":
                ids = list(q["criteria"])
                pos[j["family8"]][len(ids)][ids.index(g)] += 1
                si = sorted(ids)
                pos_sorted[j["family8"]][len(si)][si.index(g)] += 1
                F["labels"]["choice_K"][len(ids)] += 1
            else:
                F["labels"][q["type"]][g] += 1
    shas = {"records": rec_h.hexdigest(), "manifest": man_h.hexdigest(), "content": con_h.hexdigest()}

    def tq(xs):
        xs = sorted(xs)
        return {"n": len(xs), "p50": xs[len(xs) // 2], "p99": xs[int(0.99 * (len(xs) - 1))], "max": xs[-1],
                "n_over_2048": sum(1 for x in xs if x > T_MAX)} if xs else None

    fam_rep = {}
    for fam in jev1.FAMILIES:
        F = fams[fam]
        fam_rep[fam] = {"rows": F["rows"], "target": TARGET, "sources": dict(F["sources"]),
                        "licenses": dict(F["licenses"]), "subfamilies": dict(sorted(F["subs"].items())),
                        "types": dict(F["types"]),
                        "label_rates": {t: {k: round(v / sum(c.values()), 4) for k, v in sorted(c.items(), key=str)}
                                        for t, c in F["labels"].items() if t != "choice_K"},
                        "choice_K": dict(sorted(F["labels"]["choice_K"].items())),
                        "gold_position_chi2_presented": chi(pos[fam]),
                        "gold_position_chi2_sorted_ids": chi(pos_sorted[fam]),
                        "tokens_shim_qwen35_4b": tq(F["tokens"]), "filter": dict(st[fam])}
    gate_pos = {f: (fam_rep[f]["gold_position_chi2_presented"]["p"] or 1.0) >= 0.001 for f in jev1.FAMILIES}
    types = collections.Counter()
    for f in fam_rep.values():
        types.update(f["types"])
    n = len(rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    rd = Path(a.receipt)
    rd.mkdir(parents=True, exist_ok=True)
    gen_rep = json.loads((w / "gen-report.json").read_text(encoding="utf-8"))
    chk_rep = json.loads((w / "check-report.json").read_text(encoding="utf-8"))
    rep = {"build": BUILD, "n_records": n, "type_mix": {k: round(v / n, 4) for k, v in sorted(types.items())},
           "families": fam_rep, "gold_once_fail": gold_fail, "gold_position_gate_p_ge_0.001": gate_pos,
           "token_template": {"template": "jrt-h2o-shim:qwen3.5-4b", "identity": ident, "t_max": T_MAX},
           "uncompressed_sha256": shas, "pool": gen_rep, "check": {k: chk_rep[k] for k in ("n_pool", "n_kept",
                                                                                            "checked_uncompressed_sha256",
                                                                                            "reasons")},
           "drop_ids_files": {str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in a.drop_ids or []},
           "commit": commit}
    (rd / "build-report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tp = target_provenance(fam_rep)
    (rd / "target_provenance.json").write_bytes((json.dumps(tp, indent=1, ensure_ascii=False) + "\n").encode())
    ptr = {"build": f"{BUILD} (data/jev1.py + scripts/jev1_build.py: JevBench-family draw from public train splits "
                    "and deterministic generators)",
           "location": f"ssdm-data/m2/build/{BUILD}/ (outside git; deterministic, a rerun reproduces it)",
           "records_uncompressed_sha256": shas["records"], "manifest": f"{BUILD}.manifest.jsonl.gz",
           "manifest_uncompressed_sha256": shas["manifest"], "content_map": f"{BUILD}.content.jsonl.gz",
           "content_map_uncompressed_sha256": shas["content"], "n_records": n,
           "schema": "ssdm-train-v0, one question per row (qid 'q'), JevBench request shape: state + "
                     "question{type choice|noul|score, instructions, criteria}",
           "excludes_cohort": "m2-dev-cohort-v2 be6efe2b... and m2-dev-cohort-v3 498efa61... (id, content, source "
                              "item, key-free text), Index DEV/TEST rows (content, key-free text, item text), JevBench "
                              "public items (eval-only)",
           "builder": {"module": "scripts/jev1_build.py", "generator": "data/jev1.py", "commit": commit},
           "receipt": f"receipts/2026-10-10-jev1-draw/{BUILD}/",
           "templates_verified": ["jrt-h2o-shim:qwen3.5-4b"],
           "target_provenance": {"path": f"receipts/2026-10-10-jev1-draw/{BUILD}/target_provenance.json",
                                 "sha256": hashlib.sha256((rd / "target_provenance.json").read_bytes()).hexdigest()}}
    pp = config.MANIFEST_DIR / f"{BUILD}.pointer.json"
    pp.write_bytes((json.dumps(ptr, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({"n": n, "type_mix": rep["type_mix"], "rows": {f: fam_rep[f]["rows"] for f in fam_rep},
                      "filter": {f: fam_rep[f]["filter"] for f in fam_rep}, "gold_once_fail": gold_fail,
                      "gate_pos": gate_pos, "shas": shas,
                      "pointer_sha256": hashlib.sha256(pp.read_bytes()).hexdigest()}, indent=1))
    return 0


def target_provenance(fam_rep: dict) -> dict:
    from data import jev1
    return {
        "rule": "Brief (agent-b, 2026-10-10): a JevBench-family draw of ~50-150k rows balanced across the 8 families, "
                "from public TRAIN splits or deterministic generators only; no rate or count taken from JevBench items "
                "or any eval row. Per-family target T = 96,000 / 8 = 12,000 rows (the brief's range midpoint, split "
                "evenly over the 8 families). Within a family, sub-source shares come from data.jev1.PLAN (fixed "
                "before any gate ran: roughly one third public-train / two thirds synthetic where a public train "
                "source exists, capped by availability); label rates are the natural train-split rates for public "
                "rows and the generator's own rule outcomes for synthetic rows (reported, never tuned toward any "
                "eval rate). Question-type mix per family follows the source's natural answer type; score items were "
                "added where an ordinal answer is natural (requirement counts, reimbursed share, priority levels). "
                "Option presentation order is a seeded per-row shuffle. A family short of T after the gates is "
                "reported short, never topped up from another family.",
        "target_rows_per_family": TARGET, "plan": jev1.PLAN, "seed": jev1.SEED,
        "families": {f: {"rows": v["rows"], "sources": v["sources"], "licenses": v["licenses"],
                         "label_rates": v["label_rates"], "types": v["types"]} for f, v in fam_rep.items()},
        "public_sources": {
            "massive": {"file": "dl-2026-10-10/massive/amazon-massive-dataset-1.0.tar.gz (1.0/data/en-US.jsonl)",
                        "split": "partition == train", "license": "CC-BY-4.0",
                        "pools": "sha256('jev1-part|massive:<id>') % 3: 0 routing, 1 intent, 2 enum (no utterance in "
                                 "two families)"},
            "helpsteer2": {"file": "dl-2026-10-10/helpsteer2/train.jsonl.gz", "split": "train", "license": "CC-BY-4.0",
                           "note": "one row per prompt; noul bar drops the 0-4 midpoint 2 (not decisive)"},
            "hs2pref": {"file": "dl-2026-10-10/helpsteer2/preference.jsonl.gz", "split": "split == train",
                        "license": "CC-BY-4.0", "note": "preference_strength 0 (ties) dropped; prompts not reused"},
            "aegis2": {"file": "dl-2026-10-10/aegis2/train.json", "split": "train", "license": "CC-BY-4.0",
                       "note": "REDACTED prompts skipped; prompt labels are human; response labels human or "
                               "llm_jury (recorded per row as aug.jev1.label_source)"},
            "banking77": {"file": "banking77/train.csv", "split": "train", "license": "CC-BY-4.0"},
            "clinc150": {"file": "clinc150/data_full.json", "split": "train + oos_train", "license": "CC-BY-3.0"},
            "sst5": {"file": "sst5/train.jsonl", "split": "train", "license": "unspecified on HF (research)"},
            "multiwoz22": {"file": "dl-2026-10-10/multiwoz22/dialogues_001..017.json", "split": "train folder",
                           "license": "MIT (repo LICENSE)", "repo_head": "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b"}},
        "not_used": {"RouterBench": "never a train source (embeds other benchmarks' items; data/SOURCES.md)",
                     "SGD": "display-only Index row (test in the guard); MultiWOZ used instead",
                     "WildGuardMix": "gated (terms acceptance); not accepted", "BeaverTails": "CC-BY-NC-4.0; Aegis 2.0 "
                     "(CC-BY-4.0) covers the family", "UltraFeedback / Skywork / PRM800K": "not needed: HelpSteer2 "
                     "fills adequacy", "Few-NERD / CoNLL": "parquet-only / restrictive licence; MASSIVE slots used",
                     "HWU64": "overlaps MASSIVE (same base corpus)"},
        "generators": jev1.GEN_SPECS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    c = sub.add_parser("check")
    c.add_argument("--work", required=True)
    f = sub.add_parser("final")
    f.add_argument("--work", required=True)
    f.add_argument("--lens", nargs="+", required=True)
    f.add_argument("--drop-ids", nargs="*")
    f.add_argument("--receipt", required=True)
    f.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    return {"gen": cmd_gen, "check": cmd_check, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
