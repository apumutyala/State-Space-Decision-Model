"""Q9-AF v2 corpus check: rebuild job-w3c2e's distillation mix exactly and scan EVERY window for eval material.

    python -X utf8 -m jobs.q9af_corpus_check.corpus_check run --config configs/q9af_v2_h100.json \
        --dev-rows /data/slices/dev-rows.jsonl.gz --index-dir /data/doc-overlap/218affe13e367d07 \
        --mix /tmp/q9cc/mix --out /output/check [--workers 6] [--expect jobs/q9af_corpus_check/w3c2e_mix_pin.json]
    python -X utf8 -m jobs.q9af_corpus_check.corpus_check compare --rebuilt <check/mix.json> --expected <w3c2e mix.json>
    python -X utf8 -m jobs.q9af_corpus_check.corpus_check resolve --flagged <check/flagged.jsonl.gz> --windows-used N

Why: job-w3c2e (jobs/q9af_v2) built its 280M-token generic mix in-job (convert/mixdata.py, stage D) with a DEV-ONLY
13-gram filter (``dev13-v1``). Operator rule: we can't train on test (TEST, held-out, the frozen catalogs
[9, 20, 22, 31, 32, 33, 44, 58, 64] and the eval attack templates included), so w3c2e's result is provisional until
the corpus it trained on is checked against all of them. w3c2e exports only mix.json (counts + sha256s, inside its
``phases/D.tgz``), no per-document or per-window ids, so the check rebuilds the mix.

Rebuild: ``convert.mixdata.build`` with w3c2e's config (``mix.decontam`` = dev13-v1, seed, sources, pinned HF
revisions in scripts/sf_fetch.py, the same DEV rows by sha256) writes the same train.u32; its ``on_doc`` hook hands
every WRITTEN document (id = stream/source/file/row group/row, its token offset and length) to the scan as it is
written, so nothing is materialised. Gates: the scanned documents tile train.u32 exactly (sum of lengths = train
tokens, per-source kept counts = mix.json), and, once w3c2e's mix.json is pinned (``--expect``) or compared offline
(``compare``), every count and sha256 equals w3c2e's (train_sha256 equal = byte-identical training stream).

Scan (per document, worker processes; the classes are reported separately):
  policy   ``train.doc_overlap_guard`` rule v1 over the whole draw (``scan_hits``: R / E / U, boilerplate and the
           reviewed allowlist) against the PINNED eval-hash index (DEV + TEST + held-out, hashes only, sha256 =
           data/manifests/doc_overlap/index-pin.json). A FAIL is classed by its best eval unit: TEST / HELDOUT / DEV,
           plus FROZEN when that unit's family is a frozen catalog (heldout-draw-v1).
  contact  any DISTINCTIVE eval 13-gram (in a non-allowlisted unit and in < 100 eval rows; the full-eval-v1 filter's
           rule), classed by every eval row it occurs in (TEST / HELDOUT / DEV / FROZEN). Informational: a 13-word
           overlap is not by itself a copied document, but it is exactly what full-eval-v1 now drops.
  attack   ``train.attack_guard`` (every eval attack template fragment, base64 payloads decoded).
  frozen generator check: ``harness.heldout.check_generator`` on the mix's generator spec and ``check_record`` on
           every source file's metadata (when the kit is importable; the frozen-name scan below runs regardless).
Windows: train.u32 is consumed as non-overlapping seq_len windows in one permutation (``convert.joint``:
``WindowStream(seed=cfg.seed + 7)``); every flagged document lists its windows and their permutation ranks, so the
windows w3c2e actually trained on are the ranks < its ``windows_used_after`` (``resolve``).
Output (ids and counts only; no eval text, no TEST text): report.json, flagged.jsonl.gz, mix.json (rebuilt).
Exit: 0 report written (verdict CLEAN or CONTAMINATED); 16 report written but the rebuild != the pinned w3c2e mix;
14 eval index / pin refused; 15 the scanned documents do not tile the rebuilt stream.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np

from convert import mixdata as M
from train import attack_guard as A
from train import doc_overlap_guard as G

ROOT = Path(__file__).resolve().parents[2]
W3C2E = {"job": "job-w3c2e", "config": "configs/q9af_v2_h100.json", "decontam": M.DECONTAM_DEV,
         "export_D_sha256": "efd3be06f64c295f915deac38e344d4b9c4a6296b4a947ac5fb53393aa0c17c9"}
JOINT_SEED_OFFSET = 7  # convert/joint.py: WindowStream(train, L, B, seed=cfg["seed"] + 7)
CLASSES = ("TEST", "HELDOUT", "DEV", "FROZEN")
EXAMPLES = 25
CHUNK_DOCS = 64
EXIT_INDEX, EXIT_TILE, EXIT_NOT_REPRODUCED = 14, 15, 16
GENERATOR = {"name": "convert.mixdata generic distillation mix (q9af_v2)", "task_family": "generic_text_distillation",
             "targets": [], "sources": {"web": "HuggingFaceFW/fineweb-edu", "chat": "HuggingFaceH4/ultrachat_200k",
                                        "math": "HuggingFaceTB/finemath"}}


class CheckError(RuntimeError):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


# ----------------------------------------------------------------------------- frozen catalogs (heldout-draw-v1)
def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@lru_cache(maxsize=1)
def frozen_tokens() -> dict[str, int]:
    """harness.heldout._tokens without the kit import: {normalised family / name / alias: frozen catalog id}, read
    from the frozen manifests after checking their pinned sha256 (read-only)."""
    from harness import config as hc

    raw_d = (hc.MANIFEST_DIR / "heldout-draw-v1.json").read_bytes()
    raw_f = (hc.MANIFEST_DIR / "families-v1.json").read_bytes()
    if hashlib.sha256(raw_d).hexdigest() != hc.HELDOUT_DRAW_V1_SHA256 or \
            hashlib.sha256(raw_f).hexdigest() != hc.FAMILIES_V1_SHA256:
        raise CheckError(EXIT_INDEX, "heldout-draw-v1 / families-v1 sha256 != harness.config pins")
    spec, fam = json.loads(raw_d), json.loads(raw_f)["benchmarks"]
    out = {}
    for n in spec["frozen"]:
        b = fam[str(n)]
        for t in [b["family"], b["name"], *spec["aliases"][str(n)]]:
            if len(_norm(t)) >= 3:
                out[_norm(t)] = int(n)
    return out


@lru_cache(maxsize=None)
def frozen_of(name) -> tuple[int, ...]:
    ns = _norm(name)
    return tuple(sorted({n for tok, n in frozen_tokens().items() if tok in ns}))


def generator_check(files: list[str]) -> dict:
    """The frozen-catalog generator check where it applies: the mix's generator spec and its source-file metadata."""
    recs = [{"id": f, "source": f.split("/", 1)[0], "file": f} for f in files]
    own = {f: frozen_of(f) for f in files if frozen_of(f)} | ({"generator": frozen_of(json.dumps(GENERATOR))}
                                                            if frozen_of(json.dumps(GENERATOR)) else {})
    out = {"generator": GENERATOR, "frozen_name_hits": own}
    try:
        from harness import heldout as H
    except Exception as e:  # noqa: BLE001  the kit is in the job image; a laptop / dry run records why it skipped
        out["harness_heldout"] = f"not importable ({type(e).__name__}); frozen-name scan only"
        return out
    errs = []
    for obj, fn in [(GENERATOR, H.check_generator)] + [(r, H.check_record) for r in recs]:
        try:
            fn(obj)
        except H.FrozenFamilyError as e:
            errs.append(str(e)[:300])
    out["harness_heldout"] = {"checked": 1 + len(recs), "refused": errs}
    return out


# ----------------------------------------------------------------------------- eval index
def load_index(index_dir: str, index_sha: str | None = None):
    """The pinned eval-hash index (hashes only). Without ``index_sha`` the committed pin decides (rule, allowlist and
    array sha256 must all match it); ``index_sha`` is for the CPU dry run's synthetic index."""
    pin = G.read_pin()
    if index_sha is None:
        if not pin:
            raise CheckError(EXIT_INDEX, "no data/manifests/doc_overlap/index-pin.json")
        if pin.get("rule") != G.RULE or pin.get("allowlist_sha256") != G.allowlist_sha256():
            raise CheckError(EXIT_INDEX, "index pin was made under a different rule / allowlist")
        index_sha = pin["index_sha256"]
    try:
        idx = G.load_index_dir(Path(index_dir), index_sha)
    except G.DocOverlapError as e:
        raise CheckError(EXIT_INDEX, str(e)) from e
    return idx, {"index_dir": str(index_dir), "index_sha256": index_sha, "pinned": index_sha == (pin or {}).get(
        "index_sha256"), "rows_by_split": idx.meta.get("rows_by_split"), "n_grams": idx.meta.get("n_grams")}


# ----------------------------------------------------------------------------- per-document scan (worker side)
_W: dict = {}


def _winit(index_dir: str, index_sha: str) -> None:
    _W["idx"] = G.load_index_dir(Path(index_dir), index_sha)
    _W["m"] = A.eval_matcher()


def _wscan(items: list[tuple[int, str]]) -> list[tuple]:
    """[(doc k, n doc grams, eval-index positions, attack template keys)] for docs with any eval gram or attack hit."""
    idx, m = _W["idx"], _W["m"]
    out = []
    for k, text in items:
        g = G.text_grams([text])
        pos = idx.lookup(g) if len(g) else np.empty(0, np.int64)
        keys = sorted(A.record_hits({"state": text}, m))
        if len(pos) or keys:
            out.append((k, int(len(g)), pos.astype(np.int64), keys))
    return out


class Collector:
    """``mixdata.build``'s on_doc hook: records every written doc's identity and span, scans its text in workers."""

    def __init__(self, roots: dict, idx, index_dir: str, index_sha: str, workers: int):
        self.roots, self.workers = roots, workers
        self.docs: list[tuple] = []          # (stream, src, file idx, row group, row, token offset, n tokens)
        self.files: dict[tuple, int] = {}
        self.offsets: dict[str, int] = {}
        self.dropped: dict[str, int] = {}
        self.res: dict[int, tuple] = {}
        self.buf: list[tuple[int, str]] = []
        self.inflight: deque = deque()
        self.pool = None
        if workers > 0:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            # spawn, never fork: the parent's tokenizer thread pool must not be forked into the workers
            self.pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_winit,
                                            initargs=(index_dir, index_sha))
        else:
            _W["idx"], _W["m"] = idx, A.eval_matcher()

    def file_rel(self, src: str, fp: str) -> str:
        return f"{src}/" + os.path.relpath(fp, self.roots[src]).replace(os.sep, "/")

    def __call__(self, stream, src, fp, rg, row, text, ntok, dropped):
        if dropped is not None:
            key = f"{stream}/{src}/{dropped}"
            self.dropped[key] = self.dropped.get(key, 0) + 1
            return
        fi = self.files.setdefault((src, fp), len(self.files))
        off = self.offsets.get(stream, 0)
        self.offsets[stream] = off + ntok
        k = len(self.docs)
        self.docs.append((stream, src, fi, int(rg), int(row), off, int(ntok)))
        self.buf.append((k, text))
        if len(self.buf) >= CHUNK_DOCS:
            self._submit()

    def _take(self, out):
        for r in out:
            self.res[r[0]] = r[1:]

    def _submit(self):
        items, self.buf = self.buf, []
        if not items:
            return
        if self.pool is None:
            self._take(_wscan(items))
            return
        self.inflight.append(self.pool.submit(_wscan, items))
        while len(self.inflight) > 4 * self.workers:
            self._take(self.inflight.popleft().result())

    def finish(self):
        self._submit()
        while self.inflight:
            self._take(self.inflight.popleft().result())
        if self.pool is not None:
            self.pool.shutdown()

    def doc_id(self, k: int) -> str:
        st, src, fi, rg, row, _, _ = self.docs[k]
        return f"{st}/{src}/f{fi}/rg{rg}/r{row}"


# ----------------------------------------------------------------------------- analysis
def window_ranks(n_windows: int, seed: int) -> np.ndarray:
    """rank[w] = position of window w in WindowStream's first permutation (trained iff rank < windows_used)."""
    perm = np.random.default_rng(seed).permutation(n_windows)
    rank = np.empty(n_windows, np.int64)
    rank[perm] = np.arange(n_windows)
    return rank


def _windows(off: int, n: int, L: int, n_windows: int) -> list[int]:
    return [w for w in range(off // L, (off + n - 1) // L + 1) if w < n_windows]


def _agg():
    return {"docs": 0, "tokens": 0, "windows": set(), "per_source": {}, "eval_rows": set(), "examples": []}


def _add(a: dict, src: str, ntok: int, wins: list[int], doc_id: str, ex: dict | None, eval_rows=()):
    a["docs"] += 1
    a["tokens"] += ntok
    a["windows"].update(wins)
    ps = a["per_source"].setdefault(src, {"docs": 0, "tokens": 0})
    ps["docs"] += 1
    ps["tokens"] += ntok
    a["eval_rows"].update(eval_rows)
    if len(a["examples"]) < EXAMPLES:
        a["examples"].append({"id": doc_id, **(ex or {})})


def _final(a: dict, rank: np.ndarray | None, windows_used: int | None) -> dict:
    w = sorted(a["windows"])
    out = {"docs": a["docs"], "tokens": a["tokens"], "windows": len(w), "per_source": dict(sorted(a["per_source"].items())),
           "eval_rows": len(a["eval_rows"]), "examples": a["examples"]}
    if rank is not None and windows_used is not None and w:
        out["windows_trained"] = int((rank[np.asarray(w)] < windows_used).sum())
    return out


def analyse(col: Collector, meta: dict, idx, cfg: dict, windows_used: int | None = None, flagged_path=None) -> dict:
    L = int(cfg["seq_len"])
    T = int(meta["train_tokens"])
    n_windows = (T - 1) // L
    rank = window_ranks(n_windows, int(cfg["seed"]) + JOINT_SEED_OFFSET)
    rows_meta = idx.meta["rows"]
    # ---- tiling gate: the scanned docs ARE the stream
    tr = [k for k, d in enumerate(col.docs) if d[0] == "train"]
    tiles = {"train_tokens": T, "scanned_tokens": int(col.offsets.get("train", 0)), "scanned_docs": len(tr),
             "per_source_docs": {}, "ok": True}
    for src in M.SOURCES:
        got = sum(1 for k in tr if col.docs[k][1] == src)
        tiles["per_source_docs"][src] = {"scanned": got, "mix_json": meta["sources"][src]["docs_kept"]}
        tiles["ok"] &= got == meta["sources"][src]["docs_kept"]
    tiles["ok"] &= tiles["scanned_tokens"] == T
    if not tiles["ok"]:
        raise CheckError(EXIT_TILE, f"scanned docs do not tile train.u32: {json.dumps(tiles)}")
    streams = ["train"] + [f"heldout_{s}" for s in M.SOURCES]
    out = {"tiling": tiles, "streams": {}}
    fh = gzip.open(flagged_path, "wt", encoding="utf-8") if flagged_path else None
    for stream in streams:
        ks = [k for k, d in enumerate(col.docs) if d[0] == stream]
        if not ks:
            continue
        per_source_rows: dict[str, int] = {}
        hits = []
        for k in ks:
            src = col.docs[k][1]
            per_source_rows[src] = per_source_rows.get(src, 0) + 1
            r = col.res.get(k)
            if r is not None and len(r[1]):
                hits.append((col.doc_id(k), src, r[0], r[1]))
        rep = G.scan_hits(hits, len(ks), per_source_rows, idx, keep_ids=True, fails_cap=None)
        fail_of = {f["id"]: f for f in rep["fails"]}
        is_train = stream == "train"
        pol = {c: _agg() for c in CLASSES}
        con = {c: _agg() for c in CLASSES + ("UNATTRIBUTED",)}
        att, full_drop, attack_sets = _agg(), _agg(), {}
        for k in ks:
            r = col.res.get(k)
            if r is None:
                continue
            st, src, fi, rg, row, off, ntok = col.docs[k]
            did = col.doc_id(k)
            wins = _windows(off, ntok, L, n_windows) if is_train else []
            ng, pos, keys = r
            rec = {}
            # policy (doc_overlap_guard rule v1 over the whole draw)
            f = fail_of.get(did)
            if f is not None:
                bu = f["best_unit"] or {}
                fz = frozen_of(bu.get("family") or "")
                cls = [bu.get("split", "?")] + (["FROZEN"] if fz else [])
                ex = {"reasons": f["reasons"], "split": bu.get("split"), "family": bu.get("family"),
                      "eval_run_id": bu.get("run_id"), "field": bu.get("field"), "row_coverage": f["row_coverage"],
                      "unit_coverage": f["best_unit_coverage"], "frozen": list(fz)}
                for c in cls:
                    if c in pol:
                        _add(pol[c], src, ntok, wins, did, ex, [bu.get("run_id")])
                rec["policy"] = ex
            # contact (any distinctive eval 13-gram: the full-eval-v1 filter's rule)
            if len(pos):
                dist = pos[((idx.fl[pos] & 1) > 0) & (idx.df[pos] < G.TEMPLATE_EVAL_DF)]
                if len(dist):
                    urows = np.unique(idx.u_row[idx.units_of(dist)]) if len(dist) else np.empty(0, np.int64)
                    splits = sorted({rows_meta[int(u)][0] for u in urows})
                    fz = sorted({n for u in urows for n in frozen_of(rows_meta[int(u)][2] or "")})
                    runs = {rows_meta[int(u)][1] for u in urows}
                    ex = {"distinct_grams": int(len(dist)), "doc_grams": ng, "splits": splits, "frozen": fz,
                          "eval_rows": len(runs)}
                    for c in splits + (["FROZEN"] if fz else []):
                        if c in con:
                            _add(con[c], src, ntok, wins, did, ex, [rows_meta[int(u)][1] for u in urows
                                                                     if c == "FROZEN" or rows_meta[int(u)][0] == c])
                    if not len(urows):
                        _add(con["UNATTRIBUTED"], src, ntok, wins, did, ex)
                    rec["contact"] = ex
            if keys:
                _add(att, src, ntok, wins, did, {"templates": keys})
                for key in keys:
                    attack_sets[key] = attack_sets.get(key, 0) + 1
                rec["attack"] = keys
            if "contact" in rec or keys:
                _add(full_drop, src, ntok, wins, did, None)
            if rec and fh is not None:
                fh.write(json.dumps({"id": did, "stream": st, "source": src, "file": rg_file(col, fi), "row_group": rg,
                                     "row": row, "token_offset": off, "n_tokens": ntok, "windows": wins,
                                     "window_ranks": [int(rank[w]) for w in wins], **rec}) + "\n")
        s_out = {"docs": len(ks), "tokens": int(col.offsets.get(stream, 0)), "docs_with_any_eval_gram": len(hits),
                 "policy_fail": {c: _final(a, rank if is_train else None, windows_used) for c, a in pol.items()},
                 "contact": {c: _final(a, rank if is_train else None, windows_used) for c, a in con.items()},
                 "attack": _final(att, rank if is_train else None, windows_used) | {"per_template": dict(sorted(
                     attack_sets.items()))},
                 "full_eval_v1_would_drop": _final(full_drop, rank if is_train else None, windows_used),
                 "doc_overlap_rule": {k: rep[k] for k in ("n_records", "n_fail", "fail_ids_sha256", "per_source_fail",
                                                          "per_reason", "per_eval_family_fail",
                                                          "eval_rows_attributed_by_split",
                                                          "eval_rows_union_covered_by_split", "allowlist_saved",
                                                          "n_boilerplate", "n_template_grams")}}
        out["streams"][stream] = s_out
    if fh is not None:
        fh.close()
    out["windows"] = {"seq_len": L, "n_windows": n_windows, "perm_seed": int(cfg["seed"]) + JOINT_SEED_OFFSET,
                      "windows_used": windows_used}
    t = out["streams"]["train"]
    bad = {c: t["policy_fail"][c]["docs"] for c in CLASSES if t["policy_fail"][c]["docs"]}
    if t["attack"]["docs"]:
        bad["ATTACK"] = t["attack"]["docs"]
    out["verdict"] = "CONTAMINATED" if bad else "CLEAN"
    out["verdict_counts"] = bad
    return out


def rg_file(col: Collector, fi: int) -> str:
    for (src, fp), i in col.files.items():
        if i == fi:
            return col.file_rel(src, fp)
    return "?"


# ----------------------------------------------------------------------------- reproduction vs w3c2e
SRC_KEYS = ("tokens", "quota", "row_groups", "docs_seen", "docs_kept", "docs_dropped_dev_ngram", "files")
HELD_KEYS = ("tokens", "row_groups", "docs_seen", "docs_kept", "docs_dropped_dev_ngram", "units")
TOP_KEYS = ("ngram", "dev_rows_sha256", "dev_ngrams", "eos_id", "train_tokens_target", "shares", "train_tokens",
            "train_sha256", "heldout_sha256")


def compare_mix(rebuilt: dict, expected: dict) -> dict:
    diffs = []
    for k in TOP_KEYS:
        if rebuilt.get(k) != expected.get(k):
            diffs.append({"key": k, "rebuilt": rebuilt.get(k), "w3c2e": expected.get(k)})
    for part, keys in (("sources", SRC_KEYS), ("heldout", HELD_KEYS)):
        for s in M.SOURCES:
            a, b = rebuilt.get(part, {}).get(s, {}), expected.get(part, {}).get(s, {})
            for k in keys:
                if a.get(k) != b.get(k):
                    diffs.append({"key": f"{part}.{s}.{k}", "rebuilt": a.get(k), "w3c2e": b.get(k)})
    if expected.get("decontam", M.DECONTAM_DEV) != M.DECONTAM_DEV:
        diffs.append({"key": "decontam", "w3c2e": expected.get("decontam")})
    return {"match": not diffs, "diffs": diffs}


def read_expect(path: str | None) -> dict | None:
    if not path:
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d.get("mix") if isinstance(d, dict) and "mix" in d else d


# ----------------------------------------------------------------------------- run
def tokenizer_ident(cfg: dict) -> dict:
    out = {}
    try:
        from convert.model import resolve_model_dir

        d = Path(resolve_model_dir(os.path.expandvars(cfg["model_dir"])))
        for f in ("tokenizer.json", "tokenizer_config.json"):
            if (d / f).is_file():
                out[f] = hashlib.sha256((d / f).read_bytes()).hexdigest()
        out["snapshot"] = d.name
    except Exception as e:  # noqa: BLE001  identity is reported, the tokens sha256 is the gate
        out["error"] = repr(e)[:200]
    try:
        import importlib.metadata as im

        out["versions"] = {p: im.version(p) for p in ("transformers", "tokenizers", "pyarrow", "numpy")}
    except Exception as e:  # noqa: BLE001
        out["versions_error"] = repr(e)[:200]
    return out


def run(cfg: dict, dev_rows: str, index_dir: str, mix_dir: str, out_dir: str, workers: int, expect: str | None,
        windows_used: int | None = None, index_sha: str | None = None, tok=None, roots: dict | None = None) -> tuple[int, dict]:
    t0 = time.time()
    dc = M.decontam_of(cfg["mix"])
    if dc != W3C2E["decontam"]:
        raise CheckError(EXIT_NOT_REPRODUCED, f"config filter {dc!r} != w3c2e's {W3C2E['decontam']!r}")
    if (Path(mix_dir) / "mix.json").exists():
        raise CheckError(EXIT_TILE, f"{mix_dir}/mix.json exists: the scan needs a fresh rebuild")
    idx, ident = load_index(index_dir, index_sha)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots = roots or {s: os.environ[M.SOURCE_ENV[s]] for s in M.SOURCES}
    col = Collector(roots, idx, index_dir, ident["index_sha256"], workers)
    try:
        meta = M.build(cfg, dev_rows, mix_dir, tok=tok, roots=roots, on_doc=col)
    finally:
        col.finish()
    t_build = time.time() - t0
    (out / "mix.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    rep = analyse(col, meta, idx, cfg, windows_used, out / "flagged.jsonl.gz")
    exp = read_expect(expect)
    repro = compare_mix(meta, exp) if exp else {"match": None, "status": "pending: w3c2e's mix.json is not pinned "
                                                "yet; run `corpus_check compare --rebuilt mix.json --expected <w3c2e "
                                                "phases/D.tgz:out/mix.json>`"}
    rep = {"check": "q9af_corpus_check/v1", "w3c2e": W3C2E, "decontam_rebuilt": dc, "eval_index": ident,
           "attack": {"rule": A.RULE, "templates_sha256": A.eval_templates_sha256(), "n_templates": len(A.eval_templates())},
           "doc_overlap_rule": G.RULE, "frozen_catalogs": sorted(set(frozen_tokens().values())),
           "reproduction": repro, "mix": {k: v for k, v in meta.items() if k != "sources"} | {
               "sources": {s: {k: v for k, v in d.items() if k != "files"} for s, d in meta["sources"].items()}},
           "dropped_seen_by_hook": dict(sorted(col.dropped.items())), "files": sorted(
               col.file_rel(s, f) for (s, f) in col.files), "generator_check": None, "tokenizer": tokenizer_ident(cfg)
           if tok is None else {"fake": True}, **rep,
           "secs": {"build_and_scan": round(t_build, 1), "total": round(time.time() - t0, 1)}, "workers": workers}
    rep["generator_check"] = generator_check(rep["files"])
    if repro["match"] is False:
        rep["verdict"] = "NOT_REPRODUCED/" + rep["verdict"]
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=list), encoding="utf-8")
    return (EXIT_NOT_REPRODUCED if repro["match"] is False else 0), rep


def _dry_run() -> bool:
    return bool(os.environ.get("Q9CC_DRYRUN")) and not Path("/opt/ssdm/m1").is_dir()


def valid_report(rep: dict) -> bool:
    """jobs/common/finalize.py validator: a complete census of a tiled, reproduced-or-pending rebuild."""
    try:
        return (rep["check"] == "q9af_corpus_check/v1" and rep["verdict"] in ("CLEAN", "CONTAMINATED")
                and rep["tiling"]["ok"] is True and rep["reproduction"]["match"] in (True, None)
                and (rep["eval_index"]["pinned"] is True or _dry_run())  # the dry run's index is synthetic
                and "train" in rep["streams"])
    except (KeyError, TypeError):
        return False


def summary(rep: dict) -> dict:
    t = rep["streams"]["train"]
    return {"verdict": rep["verdict"], "reproduction": rep["reproduction"].get("match"),
            "train_docs": t["docs"], "train_tokens": t["tokens"], "n_windows": rep["windows"]["n_windows"],
            "policy_fail_docs": {c: v["docs"] for c, v in t["policy_fail"].items()},
            "contact_docs": {c: v["docs"] for c, v in t["contact"].items()}, "attack_docs": t["attack"]["docs"],
            "full_eval_v1_would_drop_docs": t["full_eval_v1_would_drop"]["docs"], "secs": rep["secs"]}


def resolve(flagged: str, windows_used: int) -> dict:
    """Trained-window counts per class once w3c2e's windows_used_after is known (ranks < windows_used)."""
    out: dict = {}
    with gzip.open(flagged, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["stream"] != "train":
                continue
            cls = []
            if "policy" in d:
                cls += [f"policy:{d['policy']['split']}"] + (["policy:FROZEN"] if d["policy"]["frozen"] else [])
            if "contact" in d:
                cls += [f"contact:{s}" for s in d["contact"]["splits"]] + (["contact:FROZEN"] if d["contact"]["frozen"] else [])
            if "attack" in d:
                cls.append("attack")
            tw = [w for w, r in zip(d["windows"], d["window_ranks"]) if r < windows_used]
            for c in cls:
                a = out.setdefault(c, {"docs": 0, "docs_in_trained_windows": 0, "trained_windows": set()})
                a["docs"] += 1
                a["docs_in_trained_windows"] += bool(tw)
                a["trained_windows"].update(tw)
    return {c: {"docs": a["docs"], "docs_in_trained_windows": a["docs_in_trained_windows"],
                "trained_windows": len(a["trained_windows"])} for c, a in sorted(out.items())}


class FakeTok:
    """CPU dry run only (refused inside the job image): whitespace tokens with a stable md5 vocabulary."""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[int(hashlib.md5(w.encode()).hexdigest()[:6], 16) for w in t.split()] for t in texts]}

    def convert_tokens_to_ids(self, t):
        assert t == "<|endoftext|>"
        return 7


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.q9af_corpus_check.corpus_check")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--dev-rows", required=True)
    r.add_argument("--index-dir", required=True)
    r.add_argument("--index-sha", help="dry run only: a synthetic index's sha256 instead of the committed pin")
    r.add_argument("--mix", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r.add_argument("--expect")
    r.add_argument("--windows-used", type=int)
    r.add_argument("--fake-tokenizer", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--rebuilt", required=True)
    c.add_argument("--expected", required=True)
    v = sub.add_parser("resolve")
    v.add_argument("--flagged", required=True)
    v.add_argument("--windows-used", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        res = compare_mix(json.loads(Path(a.rebuilt).read_text(encoding="utf-8")), read_expect(a.expected))
        print(json.dumps(res, indent=1))
        return 0 if res["match"] else EXIT_NOT_REPRODUCED
    if a.cmd == "resolve":
        print(json.dumps(resolve(a.flagged, a.windows_used), indent=1))
        return 0
    in_image = Path("/opt/ssdm/m1").is_dir()
    if (a.fake_tokenizer or a.index_sha) and in_image:
        print("corpus_check: --fake-tokenizer / --index-sha are CPU dry-run seams, refused in the job image")
        return 2
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    try:
        code, rep = run(cfg, a.dev_rows, a.index_dir, a.mix, a.out, a.workers, a.expect, a.windows_used, a.index_sha,
                        tok=FakeTok() if a.fake_tokenizer else None)
    except CheckError as e:
        print(f"corpus_check: {e}", flush=True)
        return e.code
    print(json.dumps(summary(rep)))
    return code


if __name__ == "__main__":
    sys.exit(main())
