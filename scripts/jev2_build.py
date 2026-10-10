"""Build bschema-draw-jev2a / jev2b (data/jev2.py) with the jev1 builder's stages and gates.

Same three stages as scripts/jev1_build.py; ``check`` IS jev1_build.cmd_check (identical row gates: gold-once, the
v4 P3 Checker with cohort v2 / v3 + SplitGuard + harness.heldout + Index DEV/TEST, the production Index Fingerprint
on item-tier texts, the JevBench fingerprint + exact-segment gate on the full row, attack_guard). ``gen`` and
``final`` differ only in the generator module (data.jev2), the stage's family list, the per-family target and the
draw name.

    python -X utf8 -I scripts/jev2_build.py gen   --stage a --work <dir>
    python -X utf8 -I scripts/jev2_build.py check --work <dir>
    python -X utf8 -I scripts/jev2_build.py final --stage a --work <dir> --lens <lens.jsonl> --receipt <dir>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import jev1_build as JB  # noqa: E402  (shared helpers + the identical check stage)

T_MAX = JB.T_MAX
#: per-family row target: the brief (evidence, 2026-10-10) asks ~6,000 rows per missing family (60k over 10)
TARGET = 6000
STAGES = {"a": "bschema-draw-jev2a", "b": "bschema-draw-jev2b"}
RECEIPT_ROOT = "receipts/2026-10-10-jev2-draw"


def fams_of(stage: str):
    from data import jev2
    return jev2.STAGE_A if stage == "a" else jev2.STAGE_B


def cmd_gen(a) -> int:
    from data import augment as A, jev2
    w = Path(a.work)
    w.mkdir(parents=True, exist_ok=True)
    ids, c = set(), collections.Counter()
    h = hashlib.sha256()
    with JB._gz(w / "pool.jsonl.gz") as fo:
        for r in jev2.generate(fams_of(a.stage)):
            A.validate(r)
            if r["id"] in ids:
                raise SystemExit(f"duplicate id {r['id']}")
            ids.add(r["id"])
            # insertion order, never key-sorted: the per-row option shuffle must survive serialisation
            line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            fo.write(line + "\n")
            h.update((line + "\n").encode())
            c[r["aug"]["jev1"]["family8"]] += 1
    rep = {"n": len(ids), "by_family": dict(c), "pool_uncompressed_sha256": h.hexdigest(),
           "plan": {f: jev2.PLAN[f] for f in fams_of(a.stage)}, "seed": jev2.SEED, "stage": a.stage}
    (w / "gen-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep))
    return 0


def cmd_final(a) -> int:
    from data import config, jev2
    from train.cohort import record_content_hash

    BUILD = STAGES[a.stage]
    FAMS = fams_of(a.stage)
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
    for line in JB._lines(w / "checked.jsonl.gz"):
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
    for fam in FAMS:
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
    with JB._gz(out / "records.jsonl.gz") as fr, JB._gz(out / f"{BUILD}.manifest.jsonl.gz") as fm, \
            JB._gz(out / f"{BUILD}.content.jsonl.gz") as fc:
        for rid, line in rows:
            r = json.loads(line)
            fr.write(line + "\n")
            rec_h.update((line + "\n").encode())
            m = JB.canonical({"family": r["family"], "id": rid, "sha256": hashlib.sha256(line.encode()).hexdigest(),
                              "source": r["source"]}) + "\n"
            fm.write(m)
            man_h.update(m.encode())
            c = JB.canonical({"content_sha256": record_content_hash(r), "id": rid}) + "\n"
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
            if not JB.gold_once(r):
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
    for fam in FAMS:
        F = fams[fam]
        fam_rep[fam] = {"rows": F["rows"], "target": TARGET, "sources": dict(F["sources"]),
                        "licenses": dict(F["licenses"]), "subfamilies": dict(sorted(F["subs"].items())),
                        "types": dict(F["types"]),
                        "label_rates": {t: {k: round(v / sum(c.values()), 4) for k, v in sorted(c.items(), key=str)}
                                        for t, c in F["labels"].items() if t != "choice_K"},
                        "choice_K": dict(sorted(F["labels"]["choice_K"].items())),
                        "gold_position_chi2_presented": JB.chi(pos[fam]),
                        "gold_position_chi2_sorted_ids": JB.chi(pos_sorted[fam]),
                        "tokens_shim_qwen35_4b": tq(F["tokens"]), "filter": dict(st[fam])}
    gate_pos = {f: (fam_rep[f]["gold_position_chi2_presented"]["p"] or 1.0) >= 0.001 for f in FAMS}
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
           "uncompressed_sha256": shas, "pool": gen_rep, "check": {k: chk_rep[k] for k in (
               "n_pool", "n_kept", "checked_uncompressed_sha256", "reasons")},
           "drop_ids_files": {str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in a.drop_ids or []},
           "commit": commit}
    (rd / "build-report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tp = target_provenance(fam_rep, a.stage)
    (rd / "target_provenance.json").write_bytes((json.dumps(tp, indent=1, ensure_ascii=False) + "\n").encode())
    ptr = {"build": f"{BUILD} (data/jev2.py + scripts/jev2_build.py: TRAIN-ONLY rows for the JevBench families jev1 "
                    f"does not cover, stage {a.stage}: {', '.join(FAMS)})",
           "location": f"ssdm-data/m2/build/{BUILD}/ (outside git; deterministic, a rerun reproduces it)",
           "records_uncompressed_sha256": shas["records"], "manifest": f"{BUILD}.manifest.jsonl.gz",
           "manifest_uncompressed_sha256": shas["manifest"], "content_map": f"{BUILD}.content.jsonl.gz",
           "content_map_uncompressed_sha256": shas["content"], "n_records": n,
           "families": {f: fam_rep[f]["rows"] for f in FAMS},
           "schema": "ssdm-train-v0, one question per row (qid 'q'), JevBench request shape: state + "
                     "question{type choice|noul|score, instructions, criteria}; jev1 row layout (aug.jev1.family8)",
           "loader_contract": "option keys are NOT re-sorted: criteria order = the seeded per-row shuffle (the "
                              "presented gold position); never json.dumps(sort_keys=True) a row before rendering",
           "excludes_cohort": "m2-dev-cohort-v2 be6efe2b... and m2-dev-cohort-v3 498efa61... (id, content, source "
                              "item, key-free text), Index DEV/TEST rows (content, key-free text, item text), JevBench "
                              "public items (eval-only)",
           "builder": {"module": "scripts/jev2_build.py", "generator": "data/jev2.py", "commit": commit},
           "receipt": f"{RECEIPT_ROOT}/{BUILD}/",
           "templates_verified": ["jrt-h2o-shim:qwen3.5-4b"],
           "target_provenance": {"path": f"{RECEIPT_ROOT}/{BUILD}/target_provenance.json",
                                 "sha256": hashlib.sha256((rd / "target_provenance.json").read_bytes()).hexdigest()}}
    pp = config.MANIFEST_DIR / f"{BUILD}.pointer.json"
    pp.write_bytes((json.dumps(ptr, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({"n": n, "type_mix": rep["type_mix"], "rows": {f: fam_rep[f]["rows"] for f in fam_rep},
                      "filter": {f: fam_rep[f]["filter"] for f in fam_rep}, "gold_once_fail": gold_fail,
                      "gate_pos": gate_pos, "chi2_p": {f: fam_rep[f]["gold_position_chi2_presented"]["p"] for f in FAMS},
                      "shas": shas, "pointer_sha256": hashlib.sha256(pp.read_bytes()).hexdigest()}, indent=1))
    return 0


def target_provenance(fam_rep: dict, stage: str) -> dict:
    from data import jev2
    return {
        "rule": "Brief (evidence lane, 2026-10-10): TRAIN-ONLY rows for the 10 JevBench families with no jev1 rows, "
                "~6,000 per family (60k over 10), from public TRAIN splits or deterministic generators only; no rate "
                "or count taken from JevBench items or any eval row. Stage a = multi_hop, judge_hard, "
                "temporal_numeric, fact, tool_selection; stage b = probability, trap, ambiguous, tradeoff, "
                "adversarial. Within a family, sub-source shares come from data.jev2.PLAN (fixed before any gate ran; "
                "generated with ~35% headroom over the target for gate drops); the final trim is sha256(id) order "
                "(label-blind). Label rates are the natural train-split rates for public rows and the generator's own "
                "rule outcomes for synthetic rows (reported, never tuned toward any eval rate). Question-type mix per "
                "family follows the family's public FORMAT (type counts of the public items are format, not labels). "
                "Option presentation order is a seeded per-row shuffle. A family short of T after the gates is "
                "reported short, never topped up from another family.",
        "target_rows_per_family": TARGET, "stage": stage, "plan": {f: jev2.PLAN.get(f) for f in fam_rep},
        "seed": jev2.SEED,
        "families": {f: {"rows": v["rows"], "sources": v["sources"], "licenses": v["licenses"],
                         "label_rates": v["label_rates"], "types": v["types"]} for f, v in fam_rep.items()},
        "public_sources": {
            "musique": {"file": "ssdm-data/raw/jev2/musique/musique_ans_v1.0_train.jsonl (HF bdsaglam/musique @ "
                                "22873a405dd809893b22ada0b499299fb612d2df)", "split": "answerable train",
                        "license": "CC-BY-4.0", "selection": "sha256('jev2-part|musique:<id>') % 5 == 0, supporting "
                                                             "paragraphs <= 3,200 chars, first n in file order"},
            "gsm8k": {"file": "sources/gsm8k main/train (sources.lock)", "split": "train", "license": "MIT",
                      "pools": "sha256('jev2-part|gsm8k:<i>') % 2: 0 judge_hard, 1 temporal_numeric (no problem in "
                               "two families)"},
            "boolq": {"file": "sources/boolq data/train (sources.lock)", "split": "train", "license": "CC-BY-SA-3.0",
                      "selection": "hash % 3 == 0, passage <= 1,400 chars"},
            "glaive_fc": {"file": "sources/glaive_fc default/train/0000.parquet (sources.lock)", "split": "train",
                          "license": "Apache-2.0", "selection": "hash % 6 == 0; first user turn <= 240 chars whose "
                                                                 "first assistant turn calls a listed function"}},
        "not_used": {"HotpotQA / 2WikiMultihopQA": "HotpotQA underlies HoVer (an Index benchmark); MuSiQue covers the "
                                                    "family (its paragraphs still pass the Index guards per row)",
                     "NaturalQuestions / TriviaQA": "fact rows are short explicit statements; BoolQ + generator cover "
                                                    "it", "ToolBench / APIGen / xLAM": "licence / gating; Glaive FC v2 "
                     "(Apache-2.0) is on disk", "HelpSteer2 / UltraFeedback": "HelpSteer2 is already used by jev1 "
                     "adequacy; judge_hard uses GSM8K solutions + rule-checked synthetic responses",
                     "TimeQA / TempReason / FinQA": "deterministic rule engines give exact gold"},
        "generators": jev2.GEN_SPECS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--stage", choices=sorted(STAGES), required=True)
    c = sub.add_parser("check")
    c.add_argument("--work", required=True)
    f = sub.add_parser("final")
    f.add_argument("--work", required=True)
    f.add_argument("--stage", choices=sorted(STAGES), required=True)
    f.add_argument("--lens", nargs="+", required=True)
    f.add_argument("--drop-ids", nargs="*")
    f.add_argument("--receipt", required=True)
    f.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    return {"gen": cmd_gen, "check": JB.cmd_check, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
