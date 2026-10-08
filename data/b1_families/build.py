"""Build the B1 add-on pool: generate/load -> guards -> write -> independent re-verification -> manifest.

    CUDA_VISIBLE_DEVICES="" python -X utf8 -m data.b1_families.build [--out DIR] [--seed N]

Guards per record (a record is dropped and counted if any fires; the pass fails if a SYNTHETIC family drops more than
``SYNTH_MAX_DROP`` of its candidates, which would mean a template itself overlaps the Index):

1. ``harness.heldout``: every family's generator spec passes ``check_generator`` and every record ``check_record``;
   the three requested families that heldout-draw-v1 FREEZES (MuSR 32, SATA-Bench 33, CLadder 44) are refused here
   by asking ``check_generator`` (fail-closed, not a hard-coded skip).
2. ``data.augment.validate`` (schema) and ``data.injection.assert_no_heldout`` (D1-B / eval hijack 6-grams).
3. Cohort v3 (``train/cohorts/m2-dev-cohort-v3``, sha 498efa61..., which contains every v2 role): id, normalized
   content (``train.cohort.record_content_hash``), source item, key-free text (``record_text_keys``).
4. ``train.guard.SplitGuard`` over the harness DEV + TEST manifests + pinned content index (``data.build.content_guard``).
5. ``data.dedup.Fingerprint`` v3 (all Index rows, DEV and TEST, every catalog) on the record's ITEM-tier texts:
   exact / ngram13 / minhash / edit / contain. Whole-payload hits that are not item hits are logged as context.
6. ``FamilyOverlap`` against the eval rows of the SAME family: normalized exact text and word 13-gram.
7. In-pool duplicate normalized content.

After writing, every output file is read back and re-checked with guards 1-7 from scratch; any hit raises.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import time
from pathlib import Path

from .. import augment as A
from .. import config
from .. import injection as I
from ..text import canonical_json
from . import apibank, bfcl, cruxeval
from .common import FamilyOverlap, iter_family_rows, probe_texts

OUT_DIR = config.BUILD_DIR / "b1-families"
MANIFEST_DIR = config.MANIFEST_DIR / "b1-families"
# Max share of a SYNTHETIC family's candidates the guards may drop before the build fails (a higher rate would mean a
# template itself reproduces Index text). CRUXEval is higher: the fingerprint's ``contain`` matcher drops any program
# that contains a 6+-token code line of any Index CRUXEval row (``for i in range(len(nums)):``), ~20% of programs.
SYNTH_MAX_DROP = {"bfcl": 0.05, "cruxeval": 0.35}
OVERSAMPLE = {"bfcl": 1.15, "cruxeval": 1.5}  # synthetic candidates generated per kept record wanted

FAMILIES = {  # name -> (module, catalog id, task family tag, index family names, provenance)
    "api_bank": (apibank, 3, "api_tool_selection", {"API-Bank-tool-selection"}, "train-split"),
    "bfcl": (bfcl, 1, "function_tool_selection", {"BFCL-tool-selection"}, "synthetic"),
    "cruxeval": (cruxeval, 43, "code_execution", {"CRUXEval-output-choice"}, "synthetic"),
}
REQUESTED_FROZEN = {"musr": (32, "narrative_multistep"), "sata": (33, "select_all"), "cladder": (44, "causal_query")}
TARGETS = {"api_bank": 3000, "bfcl": 2400, "cruxeval": 2400}


def eval_item_texts(fam: str, rows) -> list[list[str]]:
    out = []
    for r in rows:
        st = r.get("state") or {}
        if fam == "api_bank":
            out.append([t.get("text", "") for t in st.get("dialogue", []) if t.get("text")])
        elif fam == "bfcl":
            out.append([m.get("content", "") for turn in st.get("conversation", []) for m in turn
                        if isinstance(m, dict) and m.get("content")])
        elif fam == "cruxeval":
            out.append([st.get("code", ""), st.get("input", "")])
    return out


class Guards:
    def __init__(self, log=print):
        from data.build import content_guard
        from data.dedup import Fingerprint
        from train.cohort import record_text_keys
        from train.cohort_v3 import cohort_v3
        from harness import heldout

        self.heldout = heldout
        t0 = time.time()
        self.co = cohort_v3()
        self.held_ids, self.held_c = self.co.held_out(), self.co.held_out_content()
        self.held_i, self.held_t = self.co.held_out_items(), self.co.text_keys
        self.text_keys = record_text_keys
        self.guard, self.guard_ident = content_guard()
        log(f"  cohort v3 + split guard loaded ({time.time() - t0:.0f}s)")
        self.fp = Fingerprint.load(config.FINGERPRINT)
        self.fp.assert_production()
        log(f"  fingerprint loaded ({time.time() - t0:.0f}s)")
        self.fam: dict[str, FamilyOverlap] = {}
        rows = collections.defaultdict(list)
        names = {n: fam for fam, (_, _, _, ns, _) in FAMILIES.items() for n in ns}
        for r in iter_family_rows(set(names)):
            rows[names.get(r.get("family")) or names.get(r.get("benchmark"))].append(r)
        for fam in FAMILIES:
            self.fam[fam] = FamilyOverlap(fam, eval_item_texts(fam, rows[fam]))
            if self.fam[fam].n_rows == 0:
                raise RuntimeError(f"{fam}: no eval rows found for the family overlap check")
        log(f"  family overlap indexes: {[f.identity() for f in self.fam.values()]}")

    def identity(self) -> dict:
        return {"cohort_v3": self.co.identity(), "split_guard": self.guard_ident, "fingerprint": {
            k: self.fp.meta.get(k) for k in ("version", "rows", "npz_sha256", "suite_files_uncompressed_sha256")},
            "family_overlap": {k: v.identity() for k, v in self.fam.items()}}

    def check(self, fam: str, rec: dict, content: str) -> tuple[list[str], dict]:
        """Reasons the record must not be trained on (empty = clean) and context-tier hits."""
        from train.cohort import record_item_keys
        from train.guard import ForbiddenRow

        why = []
        self.heldout.check_record({k: v for k, v in rec.items() if k != "_probe"})
        A.validate(rec)
        I.assert_no_heldout(rec["state"], rec["questions"], rec["id"])
        if rec["id"] in self.held_ids:
            why.append("cohort_id")
        if content in self.held_c:
            why.append("cohort_content")
        if record_item_keys(rec["id"]) & self.held_i:
            why.append("cohort_item")
        if self.held_t and self.text_keys(rec) & self.held_t:
            why.append("cohort_text")
        try:
            self.guard.check({k: v for k, v in rec.items() if k != "_probe"})
        except ForbiddenRow:
            why.append("split_guard")
        pr = rec["_probe"]
        item = self.fp.check(pr["state"], pr["questions"])
        why += [f"fp_{k}" for k, v in item.items() if v]
        full = self.fp.check(rec["state"], rec["questions"])
        fh = self.fam[fam].hits(probe_texts(pr))
        if fh:
            why.append("family_overlap")
        return why, {k: v for k, v in full.items() if v}


def _probe_of(fam: str, rec: dict) -> dict:
    """Item-tier probe re-derived from the written record (for re-verification; never trusts the build's probe)."""
    st = rec["state"]
    if fam == "api_bank":
        return {"state": {"dialogue": [t.get("text") or {"api_name": t["api_name"], "param_dict": t["param_dict"]}
                                       for t in st["dialogue"]]}, "questions": {}}
    if fam == "bfcl":
        return {"state": {"request": st["conversation"][0][-1]["content"]}, "questions": {}}
    return cruxeval.item_probe(st["code"], st["input"])


def refuse_frozen(log=print) -> dict:
    from harness import heldout
    out = {}
    for name, (cat, tag) in REQUESTED_FROZEN.items():
        try:
            heldout.check_generator({"name": f"b1fam_{name}", "task_family": tag, "targets": [cat]})
            raise RuntimeError(f"{name} ({cat}) is no longer frozen: re-plan before generating it")
        except heldout.FrozenFamilyError as e:
            out[name] = {"catalog": cat, "status": "refused: heldout-draw-v1 FROZEN", "error": str(e)}
            log(f"  {name}: refused ({e})")
    for fam, (_, cat, tag, _, _) in FAMILIES.items():
        heldout.check_generator({"name": f"b1fam_{fam}", "task_family": tag, "targets": [cat]})
    return out


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(out_dir: Path = OUT_DIR, seed: int = config.SEED, log=print) -> dict:
    from train.cohort import record_content_hash

    t0 = time.time()
    refused = refuse_frozen(log)
    g = Guards(log)
    out_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"seed": seed, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "refused": refused,
               "guards": g.identity(), "families": {}}
    pool_content: set[str] = set()
    for fam, (mod, cat, tag, names, prov) in FAMILIES.items():
        fn = getattr(mod, "load", None) or mod.generate
        cands = fn(seed, int(TARGETS[fam] * OVERSAMPLE.get(fam, 1.0)), log=log)
        drop, ctx = collections.Counter(), collections.Counter()
        examples = []
        kept_lines, man, cmap = [], [], []
        for rec in cands:
            if len(kept_lines) >= TARGETS[fam]:
                drop["not_needed_target_reached"] += 1
                continue
            ch = record_content_hash(rec)
            why, full = g.check(fam, rec, ch)
            if ch in pool_content:
                why.append("pool_duplicate")
            if why:
                for w in why:
                    drop[w] += 1
                drop["dropped"] += 1
                if len(examples) < 5:
                    examples.append({"id": rec["id"], "why": why})
                continue
            for k in full:
                ctx[f"context_{k}"] += 1
            pool_content.add(ch)
            rec.pop("_probe")
            line = canonical_json(rec)
            kept_lines.append(line)
            man.append({"id": rec["id"], "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                        "content_sha256": ch, "source": rec["source"], "family": rec["family"]})
        rate = drop["dropped"] / max(1, len(cands) - drop["not_needed_target_reached"])
        log(f"  {fam}: kept {len(kept_lines)}/{len(cands)}; drops {dict(drop)}; context {dict(ctx)}")
        if prov == "synthetic" and rate > SYNTH_MAX_DROP[fam]:
            raise RuntimeError(f"{fam}: synthetic drop rate {rate:.3f} > {SYNTH_MAX_DROP[fam]}: a template overlaps the "
                               f"Index; examples {examples}")
        path = out_dir / f"{fam}.jsonl.gz"
        tmp = path.with_suffix(".gz.part")
        with gzip.GzipFile(tmp, "wb", mtime=0) as f:
            f.write("".join(l + "\n" for l in kept_lines).encode("utf-8"))
        tmp.replace(path)
        mp = MANIFEST_DIR / f"{fam}.manifest.jsonl.gz"
        with gzip.GzipFile(mp, "wb", mtime=0) as f:
            f.write("".join(canonical_json(m) + "\n" for m in man).encode("utf-8"))
        pos = collections.Counter()
        nopt = collections.Counter()
        for l in kept_lines:
            r = json.loads(l)
            a = r["aug"]
            pos[str(a.get("gold_pos"))] += 1
            nopt[str(a.get("n_options", a.get("n_candidates")))] += 1
        summary["families"][fam] = {
            "index_family": sorted(names)[0], "catalog": cat, "provenance": prov, "source": json.loads(kept_lines[0])["source"],
            "n_candidates": len(cands), "n_records": len(kept_lines), "drop_rate": round(rate, 4), "drops": dict(drop), "context_overlap": dict(ctx),
            "drop_examples": examples, "path": str(path), "sha256_gz": _sha_file(path),
            "sha256_uncompressed": hashlib.sha256("".join(l + "\n" for l in kept_lines).encode("utf-8")).hexdigest(),
            "manifest": str(mp.relative_to(config.REPO_ROOT)).replace("\\", "/"), "manifest_sha256": _sha_file(mp),
            "gold_pos": dict(sorted(pos.items(), key=lambda kv: int(kv[0]))),
            "options_or_candidates": dict(sorted(nopt.items(), key=lambda kv: int(kv[0])))}
        if fam == "api_bank":
            summary["families"][fam]["source_detail"] = {"hf": "liminghao1630/API-Bank", "revision": apibank.REVISION,
                                                        "files": apibank.FILES, "split": "training-data (train)"}
    summary["verify"] = verify(out_dir, g, log)
    summary["seconds"] = round(time.time() - t0, 1)
    (MANIFEST_DIR / "manifest.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def verify(out_dir: Path, g: Guards, log=print) -> dict:
    """Read every output back and re-run all guards with probes re-derived from the written records. Raises on any
    hit, on a manifest mismatch, on >16 options or on a gold label outside its options."""
    from train.cohort import record_content_hash

    res, all_c, all_ids, recs_all = {}, {}, [], []
    for fam in FAMILIES:
        man = {}
        with gzip.open(MANIFEST_DIR / f"{fam}.manifest.jsonl.gz", "rt", encoding="utf-8") as f:
            for l in f:
                m = json.loads(l)
                man[m["id"]] = m
        n = 0
        with gzip.open(out_dir / f"{fam}.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                rec = json.loads(line)
                n += 1
                m = man.get(rec["id"])
                if m is None or m["sha256"] != hashlib.sha256(line.encode("utf-8")).hexdigest():
                    raise AssertionError(f"{rec['id']}: line does not match the manifest")
                ch = record_content_hash(rec)
                if ch != m["content_sha256"] or ch in all_c:
                    raise AssertionError(f"{rec['id']}: content hash mismatch or duplicate")
                all_c[ch] = rec["id"]
                for qid, q in rec["questions"].items():
                    if len(q["criteria"]) > 16 or rec["labels"][qid] not in q["criteria"]:
                        raise AssertionError(f"{rec['id']}/{qid}: options or label invalid")
                rec["_probe"] = _probe_of(fam, rec)
                why, _ = g.check(fam, rec, ch)
                if why:
                    raise AssertionError(f"VERIFY FAILED {rec['id']}: {why}")
                rec.pop("_probe")
                all_ids.append(rec["id"])
                recs_all.append(rec)
        if n != len(man):
            raise AssertionError(f"{fam}: {n} records != {len(man)} manifest rows")
        res[fam] = {"records_rechecked": n, "hits": 0}
    g.co.assert_draw_disjoint(all_ids, {i: c for c, i in all_c.items()}, recs_all)
    res["cohort_v3_assert_draw_disjoint"] = "passed"
    log(f"  verify: {res}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m data.b1_families.build")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args(argv)
    s = build(Path(a.out), a.seed)
    print(json.dumps({k: {kk: v.get(kk) for kk in ("n_records", "sha256_gz", "drops")} for k, v in s["families"].items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
