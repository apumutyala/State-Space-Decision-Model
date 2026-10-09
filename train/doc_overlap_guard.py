"""Document-overlap guard: no training row may carry the DOCUMENT text of an Index eval row (DEV, TEST or held-out).

Operator rule: we can't train on test, and that includes eval DOCUMENTS (premises, passages, trial reports, tool
catalogs), not just eval questions. The 6-gram audit (analysis/NGRAM_TEST_AUDIT_2026-10-09.md) found NLI4CT-2024
clinical-trial reports reused across that dataset's train/test splits, so the ``nli4ct`` training rows carry the
premises of 40 DEV + 72 TEST Index rows. This gate refuses such rows. ``train.attack_guard`` is the sibling gate for
safety test material (job exit 33); this one is job exit 34.

Eval side (read-only, hashed; no eval text is ever written): EVERY row of the Index suite the harness loads through the
decision-index kit (``harness.suite.load_suite().rows()``: the 0.2.1 rows + added rows, i.e. DEV, TEST and every
held-out / unsliced row, frozen new-domain benchmarks included) plus the rebuilt 0.3 GSM8K rows. DEV / TEST labels come
from ``harness/manifests/{dev,test}-manifest*.jsonl.gz`` by run_id; TEST rows are read by manifest id from the suite
files, the harness unseal gate (``harness.cli.load_slice``) is never invoked, and only ids/counts are reported.
Every DEV and TEST manifest id must be found, and every source file must exist (fail closed: ``EvalDataMissing``).
Each eval row is cut into document UNITS: one per top-level ``state`` key (``S:<key>``; a non-dict state is ``S``),
all question ``instructions`` (``I``: ANLI / MuSR / ContractNLI / BRIGHT / ToolRet carry their passage or candidate
pool there), and all question ``criteria`` (``O``).

Training side: the DOCUMENT text of a row is every string it shows the model: every leaf of ``state`` (premise,
passage, evidence, context, trial sections, dialogue, tool catalogs: ``available_tools`` / ``functions`` / ``tools``
descriptions and parameters) plus question ``instructions`` and ``criteria`` (mnli / vast carry their premise in the
instructions, nq its candidate passages in the criteria). Field map observed per family (2026-10-09 schemas):
state.premise (anli), state.evidence[].text (hover, boolq), state.primary_trial / secondary_trial .text (nli4ct),
state.prompt / response (ragtruth), state.background_reading (clinc150, when2call, humicroedit_s1), state.tools /
functions / available_tools (when2call, glaive_fc, toolace, b1fam_*), state.participant_opinions (habermas),
state.review (acos), state.code / input (cruxeval), questions.*.instructions (mnli, vast), questions.*.criteria (nq).

Rule (``RULE``): text is normalised exactly as ``train.attack_guard.normalise`` (NFKC, format chars dropped, Cyrillic
homoglyphs folded, casefold, non-word runs -> one space); 13-grams (K) never cross a string leaf; each 13-gram is a
64-bit hash. A training row FAILS when, after the exclusions below,
  (R) >= 50% of its own document 13-grams are eval 13-grams (and at least min(3, its grams) of them), or
  (E) it covers >= 50% of some eval unit's 13-grams (and at least min(3, that unit's grams)), or
  (U) the draw as a whole covers >= 50% of an eval unit and this row alone covers >= 25% of it (min(3, grams) too):
      the NLI4CT shape, where a premise = primary + secondary trial section and each half arrives in another row.
Exclusions:
  (a) BOILERPLATE: 13-grams in > 1% of eval rows AND > 1% of the scanned training rows (computed per draw, listed).
  (b) ALLOWLIST (``data/manifests/doc_overlap/allowlist-v1.json``, reviewed): eval units that are a shared public
      reference corpus used by design (e.g. HoVer's Wikipedia evidence). Their 13-grams do not count for (R) unless
      they also occur in a non-allowlisted unit, and (E)/(U) skip them; every row the allowlist saves is counted per
      entry and per draw.
(E)/(U) use the eval 13-grams that occur in at most POST_CAP (256) eval units; each unit's denominator is its own count
of such grams minus the draw's boilerplate grams (a 13-gram in more units is a template / shared-pool gram).

The eval hash index is built once (``build``), cached outside git under ``ssdm-data/m2/doc-overlap/<key16>/`` and
sha-pinned in ``data/manifests/doc_overlap/index-pin.json`` (key = sha256 of the rule, the allowlist and every eval
source file). A box without the suite rows loads the shipped index (hashes only, no eval text) from
$SSDM_DOC_OVERLAP_INDEX, $SSDM_DATA/m2/doc-overlap/<key16> or /data/m2/doc-overlap/<key16>, only when its array sha256
equals the pin; with neither the suite nor a pinned index the gate refuses (fail closed).

    python -X utf8 -m train.doc_overlap_guard build                                     # build / verify the cached index
    python -X utf8 -m train.doc_overlap_guard scan --records R.jsonl.gz [--ids IDS] [--out rep.json]   # exit 4 on a fail
    python -X utf8 -m train.doc_overlap_guard clean-spec --records R.jsonl.gz --out spec.json [--ids-out ids.gz]
Job shells map exit 4 -> fail 34.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from train.attack_guard import ids_sha256, iter_jsonl, normalise

ROOT = Path(__file__).resolve().parents[1]
K = 13
THRESH = 0.5
UNION_CONTRIB = 0.25
MIN_COVERED = 3
BOILER_FRAC = 0.01
POST_CAP = 256
EXIT_HIT = 4   # job shells map 4 -> fail 34 (attack_guard: 3 -> 33)
JOB_EXIT = 34
RULE = ("doc_overlap_guard/v1: 13-gram (attack_guard.normalise, no leaf crossing) row document vs Index eval units "
        "(DEV+TEST+held-out); FAIL if covered >= 0.5 of the row's grams (R) or of an eval unit's grams (E), or the "
        "draw covers >= 0.5 of a unit and the row >= 0.25 of it (U); min 3 covered grams; boilerplate = eval df > 1% "
        "AND train df > 1%; reviewed allowlist of shared public corpora; units over 256-unit df skip (E)/(U)")
PIN_PATH = ROOT / "data" / "manifests" / "doc_overlap" / "index-pin.json"
ALLOWLIST_PATH = ROOT / "data" / "manifests" / "doc_overlap" / "allowlist-v1.json"
DEFAULT_CACHE = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data")) / "m2" / "doc-overlap"
FIELD_I, FIELD_O = "I", "O"

_P = np.uint64(0x9E3779B97F4A7C15)
_tok_cache: dict[str, int] = {}


class DocOverlapError(ValueError):
    """Base: the doc-overlap gate refuses (a training row carries eval document text, or the gate cannot run)."""


class EvalDataMissing(DocOverlapError):
    """The eval side cannot be read completely (fail closed)."""


class DocOverlapInTraining(DocOverlapError):
    """A training row carries Index eval document text."""


# ------------------------------------------------------------------------------------------------- hashing
def _tok_hash(t: str) -> int:
    h = _tok_cache.get(t)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little")
        if len(_tok_cache) < 400_000:
            _tok_cache[t] = h
    return h


def _mix(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30)); x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27)); x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def leaf_grams(words: list[str], k: int = K) -> np.ndarray:
    """All k-gram hashes (uint64, with repeats) of one normalised leaf."""
    m = len(words) - k + 1
    if m <= 0:
        return np.empty(0, np.uint64)
    th = np.fromiter((_tok_hash(w) for w in words), np.uint64, len(words))
    h = np.zeros(m, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * _P + th[j:j + m]
        return _mix(h)


def text_grams(texts: Iterable[str], k: int = K) -> np.ndarray:
    """Unique k-gram hashes of a set of leaves (no gram crosses a leaf)."""
    parts = [leaf_grams(normalise(s), k) for s in texts if s]
    parts = [p for p in parts if len(p)]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)


def _leaves(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _leaves(v)


def _questions(rec: dict) -> list[dict]:
    qs = rec.get("questions") or {}
    vals = qs.values() if isinstance(qs, dict) else qs if isinstance(qs, list) else []
    return [q for q in vals if isinstance(q, dict)]


def document_leaves(rec: dict) -> list[str]:
    """Every string a TRAINING row shows the model as context: state leaves + question instructions + criteria."""
    out = list(_leaves(rec.get("state")))
    for q in _questions(rec):
        out += list(_leaves(q.get("instructions"))) + list(_leaves(q.get("criteria")))
    return out


def eval_units(row: dict) -> list[tuple[str, list[str]]]:
    """[(field, leaves)] of one EVAL row: one unit per top-level state key, then all instructions, then all criteria."""
    st = row.get("state")
    units: list[tuple[str, list[str]]] = []
    if isinstance(st, dict):
        for key, v in st.items():
            units.append((f"S:{key}", list(_leaves(v))))
    elif st is not None:
        units.append(("S", list(_leaves(st))))
    qs = _questions(row)
    units.append((FIELD_I, [s for q in qs for s in _leaves(q.get("instructions"))]))
    units.append((FIELD_O, [s for q in qs for s in _leaves(q.get("criteria"))]))
    return [(f, ls) for f, ls in units if ls]


# ------------------------------------------------------------------------------------------------- allowlist
def load_allowlist(path: str | Path = ALLOWLIST_PATH) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for e in d["entries"]:
        for k in ("id", "eval_family", "eval_field", "corpus", "justification", "reviewed"):
            if not e.get(k):
                raise DocOverlapError(f"allowlist entry {e.get('id')!r} lacks {k}")
    return d


def allowlist_sha256(path: str | Path = ALLOWLIST_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allow_entry(allow: dict, family: str | None, field: str) -> int:
    """Index of the allowlist entry that covers an eval unit, or -1."""
    for i, e in enumerate(allow["entries"]):
        if e["eval_family"] == family and (e["eval_field"] == field or
                                           (e["eval_field"].endswith("*") and field.startswith(e["eval_field"][:-1]))):
            return i
    return -1


# ------------------------------------------------------------------------------------------------- eval sources
def _harness_paths() -> dict[str, Path]:
    from harness import config as hc

    man = hc.MANIFEST_DIR
    return {"dev_manifest": man / "dev-manifest.jsonl.gz", "dev_manifest_gsm8k03": man / "dev-manifest-0.3-gsm8k.jsonl.gz",
            "test_manifest": man / "test-manifest.jsonl.gz", "test_manifest_gsm8k03": man / "test-manifest-0.3-gsm8k.jsonl.gz",
            "gsm8k03_rows": Path(hc.GSM8K_03_ROWS)}


def _suite():
    try:
        try:
            import decision_index  # noqa: F401
        except ImportError:
            from harness import config as hc

            if (Path(hc.KIT_CLONE) / "decision_index").is_dir():
                sys.path.insert(1, str(hc.KIT_CLONE))
        from harness.suite import load_suite

        return load_suite()
    except (FileNotFoundError, OSError, ImportError) as e:
        raise EvalDataMissing(f"Index suite not readable: {e}") from e


def eval_source_files() -> dict[str, Path]:
    s = _suite()
    out = {f"suite:{p.name}": Path(p) for p in s.row_paths}
    if s.exclusions_path.exists():
        out[f"suite:{s.exclusions_path.name}"] = Path(s.exclusions_path)
    out.update(_harness_paths())
    missing = [f"{k}={v}" for k, v in out.items() if not Path(v).exists()]
    if missing:
        raise EvalDataMissing(f"eval source file(s) missing: {missing}")
    return out


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def index_key(srcs: dict[str, Path] | None = None) -> tuple[str, dict]:
    srcs = srcs or eval_source_files()
    ident = {"rule": RULE, "k": K, "post_cap": POST_CAP, "allowlist_sha256": allowlist_sha256(),
             "sources": {k: _sha_file(v) for k, v in sorted(srcs.items())}}
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest(), ident


def _manifest_ids(p: Path) -> set[str]:
    return {m["run_id"] for m in iter_jsonl(p)}


def iter_eval_rows() -> Iterator[tuple[str, str, str | None, dict]]:
    """(split, run_id, family, row) for every Index eval row; split DEV / TEST / HELDOUT (every other suite row)."""
    hp = _harness_paths()
    for k, p in hp.items():
        if not p.exists():
            raise EvalDataMissing(f"{k} missing: {p}")
    dev = _manifest_ids(hp["dev_manifest"]) | _manifest_ids(hp["dev_manifest_gsm8k03"])
    test = _manifest_ids(hp["test_manifest"]) | _manifest_ids(hp["test_manifest_gsm8k03"])
    seen: set[str] = set()

    def rows():
        yield from _suite().rows(apply_exclusions=False)
        yield from iter_jsonl(hp["gsm8k03_rows"])

    for r in rows():
        rid = str((r.get("_evaluation") or {}).get("run_id"))
        if rid in seen:
            continue
        seen.add(rid)
        split = "TEST" if rid in test else "DEV" if rid in dev else "HELDOUT"
        yield split, rid, r.get("family"), r
    miss_d, miss_t = len(dev - seen), len(test - seen)
    if miss_d or miss_t:
        raise EvalDataMissing(f"eval rows missing from the suite: {miss_d} DEV and {miss_t} TEST manifest ids")


# ------------------------------------------------------------------------------------------------- index
class EvalIndex:
    """Sorted unique eval 13-gram hashes with per-gram eval-row df, flags, allowlist entry and unit postings.

    h (uint64) sorted; df (uint16, clipped) eval rows holding the gram; fl (uint8) bit 1 = in a non-allowlisted unit, bit 2 = in
    an allowlisted unit; ent (uint8) first allowlist entry (255 none); pstart (int32, -1 = more than POST_CAP units)
    and pcnt (uint16) into post (int32 unit ids). Units: u_row (int32), u_field (int16 code), u_den (int32 grams with
    postings), u_allow (int16). meta: rows [[split, run_id, family]], fields [names], allowlist ids, counts."""

    ARRAYS = ("h", "df", "fl", "ent", "pstart", "pcnt", "post", "u_row", "u_field", "u_den", "u_allow")

    def __init__(self, arrays: dict[str, np.ndarray], meta: dict):
        for k in self.ARRAYS:
            setattr(self, k, arrays[k])
        self.meta = meta
        self.n_rows = int(meta["n_rows"])
        self.split = np.array([r[0] for r in meta["rows"]]) if meta.get("rows") else np.array([])

    def lookup(self, g: np.ndarray) -> np.ndarray:
        """Index positions of the grams of g present in the index (others dropped)."""
        if not len(g) or not len(self.h):
            return np.empty(0, np.int64)
        pos = np.searchsorted(self.h, g)
        pos[pos >= len(self.h)] = 0
        return pos[self.h[pos] == g].astype(np.int64)

    def units_of(self, pos: np.ndarray) -> np.ndarray:
        ps = self.pstart[pos]
        keep = ps >= 0
        if not keep.any():
            return np.empty(0, np.int32)
        return self.post[_ranges(ps[keep], self.pcnt[pos[keep]].astype(np.int64))]

    def unit_label(self, u: int) -> dict:
        r = self.meta["rows"][int(self.u_row[u])]
        al = int(self.u_allow[u])
        return {"split": r[0], "run_id": r[1], "family": r[2], "field": self.meta["fields"][int(self.u_field[u])],
                "grams": int(self.u_den[u]), "allowlist": self.meta["allowlist_ids"][al] if al >= 0 else None}


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenated index ranges [s, s + c) for each (s, c), vectorised."""
    counts = counts.astype(np.int64)
    tot = int(counts.sum())
    if not tot:
        return np.empty(0, np.int64)
    ends = np.cumsum(counts)
    return np.repeat(starts.astype(np.int64) - (ends - counts), counts) + np.arange(tot, dtype=np.int64)


def build_index(rows: Iterable[tuple[str, str, str | None, dict]], allow: dict, n_parts: int = 4,
                rows_factory=None) -> EvalIndex:
    """Build from eval rows. ``rows_factory()`` (re-iterable source) lets the build run in ``n_parts`` hash partitions
    so peak RAM is ~1/n_parts of the (gram, unit) pairs; with a plain iterable it runs in one partition."""
    factory = rows_factory if rows_factory is not None else (lambda: rows)
    if rows_factory is None:
        n_parts = 1
    fields: dict[str, int] = {}
    meta_rows: list[list] = []
    u_row: list[int] = []
    u_field: list[int] = []
    u_allow: list[int] = []
    parts: list[dict[str, np.ndarray]] = []
    shift = np.uint64(62)
    for part in range(n_parts):
        hs, us = [], []
        uid = 0
        for ri, (split, rid, fam, row) in enumerate(factory()):
            if part == 0:
                meta_rows.append([split, rid, fam])
            for field, leaves in eval_units(row):
                g = text_grams(leaves)
                if part == 0:
                    fields.setdefault(field, len(fields))
                    u_row.append(ri); u_field.append(fields[field]); u_allow.append(allow_entry(allow, fam, field))
                if n_parts > 1 and len(g):
                    g = g[(g >> shift) % np.uint64(n_parts) == np.uint64(part)] if n_parts <= 4 else \
                        g[g % np.uint64(n_parts) == np.uint64(part)]
                if len(g):
                    hs.append(g); us.append(np.full(len(g), uid, np.int32))
                uid += 1
        H = np.concatenate(hs) if hs else np.empty(0, np.uint64)
        U = np.concatenate(us) if us else np.empty(0, np.int32)
        del hs, us
        order = np.lexsort((U, H))
        H, U = H[order], U[order]
        del order
        parts.append(_finish_part(H, U, np.asarray(u_row, np.int32), np.asarray(u_allow, np.int16)))
        del H, U
    # merge partitions (disjoint gram sets) into one sorted index
    h = np.concatenate([p["h"] for p in parts])
    order = np.argsort(h, kind="stable")
    out: dict[str, np.ndarray] = {"h": h[order]}
    for k in ("df", "fl", "ent", "pcnt"):
        out[k] = np.concatenate([p[k] for p in parts])[order]
    # postings: re-base per partition, then gather in the merged order
    post_parts, ps_parts, base = [], [], 0
    for p in parts:
        ps = p["pstart"].copy()
        ps[ps >= 0] += base
        ps_parts.append(ps); post_parts.append(p["post"]); base += len(p["post"])
    pstart = np.concatenate(ps_parts)[order]
    post_all = np.concatenate(post_parts) if post_parts else np.empty(0, np.int32)
    del parts, ps_parts, post_parts, order
    keep = pstart >= 0
    cnt = out["pcnt"].astype(np.int64)
    new_start = np.full(len(pstart), -1, np.int64)
    c = np.where(keep, cnt, 0)
    offs = np.concatenate([[0], np.cumsum(c)[:-1]]) if len(c) else np.empty(0, np.int64)
    new_start[keep] = offs[keep]
    post = post_all[_ranges(pstart[keep], cnt[keep])] if keep.any() else np.empty(0, np.int32)
    del post_all
    if len(post) >= 2**31 - 1:
        raise DocOverlapError("postings exceed int32 offsets")
    out["pstart"], out["post"] = new_start.astype(np.int32), post
    out["df"] = np.minimum(out["df"], 65535).astype(np.uint16)  # only compared with 1% of the eval rows
    u_row_a, u_allow_a = np.asarray(u_row, np.int32), np.asarray(u_allow, np.int16)
    u_den = np.bincount(post, minlength=len(u_row)).astype(np.int32) if len(post) else np.zeros(len(u_row), np.int32)
    out.update(u_row=u_row_a, u_field=np.asarray(u_field, np.int16), u_den=u_den, u_allow=u_allow_a)
    by_split: dict[str, int] = {}
    for r in meta_rows:
        by_split[r[0]] = by_split.get(r[0], 0) + 1
    meta = {"rule": RULE, "k": K, "post_cap": POST_CAP, "n_rows": len(meta_rows), "rows": meta_rows,
            "rows_by_split": by_split, "fields": sorted(fields, key=fields.get), "n_units": len(u_row),
            "n_grams": int(len(out["h"])), "n_postings": int(len(post)),
            "allowlist_ids": [e["id"] for e in allow["entries"]]}
    return EvalIndex(out, meta)


def _finish_part(H: np.ndarray, U: np.ndarray, u_row: np.ndarray, u_allow: np.ndarray) -> dict[str, np.ndarray]:
    """(gram, unit) pairs sorted by (gram, unit) -> per-gram arrays + postings for grams in <= POST_CAP units."""
    if not len(H):
        z = np.empty(0)
        return {"h": z.astype(np.uint64), "df": z.astype(np.uint32), "fl": z.astype(np.uint8), "ent": z.astype(np.uint8),
                "pcnt": z.astype(np.uint16), "pstart": z.astype(np.int64), "post": z.astype(np.int32)}
    new = np.r_[True, H[1:] != H[:-1]]
    start = np.nonzero(new)[0]
    ucnt = np.diff(np.r_[start, len(H)])
    rows = u_row[U]
    new_row = new | np.r_[True, rows[1:] != rows[:-1]]
    df = np.add.reduceat(new_row.astype(np.uint32), start)
    al = u_allow[U]
    fl = (np.bitwise_or.reduceat(np.where(al < 0, 1, 2).astype(np.uint8), start)).astype(np.uint8)
    entv = np.where(al < 0, 255, al).astype(np.int16)
    ent = np.minimum.reduceat(entv, start).astype(np.uint8)
    small = ucnt <= POST_CAP
    sel = np.repeat(small, ucnt)
    post = U[sel].astype(np.int32)
    c = np.where(small, ucnt, 0)
    offs = np.concatenate([[0], np.cumsum(c)[:-1]])
    pstart = np.where(small, offs, -1).astype(np.int64)
    return {"h": H[start], "df": df.astype(np.uint32), "fl": fl, "ent": ent,
            "pcnt": np.minimum(ucnt, 65535).astype(np.uint16), "pstart": pstart, "post": post}


def _arrays_sha(d: Path) -> str:
    """sha256 over the per-array .npy file sha256s, in ARRAYS order (the pinned identity of an index directory)."""
    return hashlib.sha256("".join(_sha_file(d / f"{k}.npy") for k in EvalIndex.ARRAYS).encode()).hexdigest()


def save_index(idx: EvalIndex, d: Path, ident: dict) -> dict:
    """One .npy per array (loaded memory-mapped: the laptop's RAM is shared) + meta.json; hashes only, no eval text."""
    d.mkdir(parents=True, exist_ok=True)
    for k in EvalIndex.ARRAYS:
        tmp = d / f"{k}.tmp.npy"
        np.save(tmp, np.ascontiguousarray(getattr(idx, k)))
        tmp.replace(d / f"{k}.npy")
    sha = _arrays_sha(d)
    meta = dict(idx.meta) | {"index_sha256": sha, "ident": ident}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def load_index_dir(d: Path, expect_sha: str | None) -> EvalIndex:
    mp = d / "meta.json"
    if not mp.exists() or not all((d / f"{k}.npy").exists() for k in EvalIndex.ARRAYS):
        raise EvalDataMissing(f"no eval hash index in {d}")
    sha = _arrays_sha(d)
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if sha != meta.get("index_sha256") or (expect_sha and sha != expect_sha):
        raise DocOverlapError(f"eval hash index {d} sha256 {sha[:12]} != pinned {str(expect_sha or meta.get('index_sha256'))[:12]}")
    if meta.get("rule") != RULE:
        raise DocOverlapError("eval hash index was built under a different rule")
    return EvalIndex({k: np.load(d / f"{k}.npy", mmap_mode="r") for k in EvalIndex.ARRAYS}, meta)


def read_pin() -> dict | None:
    return json.loads(PIN_PATH.read_text(encoding="utf-8")) if PIN_PATH.exists() else None


_TEST_INDEX: list[EvalIndex] = []


def set_index_for_tests(idx: EvalIndex | None) -> None:
    """Tests only: install a synthetic in-process index (never read from env or disk)."""
    _TEST_INDEX.clear()
    if idx is not None:
        _TEST_INDEX.append(idx)
    get_index.cache_clear()


@lru_cache(maxsize=1)
def get_index() -> EvalIndex:
    """The pinned eval index. With the suite on disk: key recomputed from the sources; the cache dir is built if
    absent. Without it (a job box): the shipped index in $SSDM_DOC_OVERLAP_INDEX must match the committed pin."""
    if _TEST_INDEX:
        return _TEST_INDEX[0]
    pin = read_pin()
    env = os.environ.get("SSDM_DOC_OVERLAP_INDEX")
    try:
        key, ident = index_key()
    except EvalDataMissing:
        if not pin:
            raise
        if pin.get("allowlist_sha256") != allowlist_sha256() or pin.get("rule") != RULE:
            raise DocOverlapError("index pin was made under a different rule / allowlist")
        sub = pin["key"][:16]
        for d in ([Path(env)] if env else []) + [DEFAULT_CACHE / sub, Path("/data/m2/doc-overlap") / sub,
                                                  Path("/data/doc-overlap") / sub]:
            if (d / "meta.json").exists():
                return load_index_dir(d, pin["index_sha256"])
        raise
    if pin and pin.get("key") != key:
        raise DocOverlapError(f"eval sources / rule / allowlist changed since the index pin ({pin.get('key', '')[:12]} "
                              f"!= {key[:12]}): rebuild with `build` and re-pin")
    d = Path(env) if env else DEFAULT_CACHE / key[:16]
    if not (d / "meta.json").exists():
        idx = build_index(None, load_allowlist(), n_parts=4, rows_factory=iter_eval_rows)
        save_index(idx, d, ident)
    return load_index_dir(d, pin["index_sha256"] if pin else None)


def pin_record(idx: EvalIndex, key: str) -> dict:
    return {"rule": RULE, "key": key, "index_sha256": idx.meta.get("index_sha256"),
            "allowlist_sha256": allowlist_sha256(), "n_eval_rows": idx.n_rows, "rows_by_split": idx.meta["rows_by_split"],
            "n_units": idx.meta["n_units"], "n_grams": idx.meta["n_grams"], "n_postings": idx.meta["n_postings"]}


# ------------------------------------------------------------------------------------------------- scan
def scan(records: Iterable[dict], idx: EvalIndex | None = None, keep_ids: bool = True,
         want: set[str] | None = None) -> dict[str, Any]:
    """Scan training rows. One streaming pass keeps only rows with any eval-gram hit (compact arrays); the boilerplate
    set (needs the draw's train df) and the draw-union rule (U) are resolved after the pass."""
    idx = idx or get_index()
    t0 = time.time()
    cand_thr = BOILER_FRAC * idx.n_rows
    n = 0
    found: set[str] = set()
    per_source_rows: dict[str, int] = {}
    hits: list[tuple] = []
    cand_df: dict[int, int] = {}
    for rec in records:
        if not isinstance(rec, dict) or "state" not in rec:
            continue
        rid = str(rec.get("id"))
        if want is not None:
            if rid not in want or rid in found:
                continue
            found.add(rid)
        n += 1
        src = str(rec.get("source") or rec.get("family"))
        per_source_rows[src] = per_source_rows.get(src, 0) + 1
        g = text_grams(document_leaves(rec))
        if not len(g):
            continue
        pos = idx.lookup(g)
        if not len(pos):
            continue
        hi = pos[idx.df[pos] > cand_thr]
        for p in hi.tolist():
            cand_df[p] = cand_df.get(p, 0) + 1
        hits.append((rid, src, len(g), pos.astype(np.int64)))
    if want is not None and found != want:
        raise DocOverlapError(f"{len(want - found)} requested id(s) not in the records, e.g. {sorted(want - found)[:3]}")
    boiler = {p for p, c in cand_df.items() if c > BOILER_FRAC * max(n, 1)}
    boiler_arr = np.array(sorted(boiler), np.int64)
    nunits = len(idx.u_row)
    den = idx.u_den.astype(np.int64)
    if len(boiler_arr):  # boilerplate grams leave every unit's denominator too
        den = den - np.bincount(idx.units_of(boiler_arr), minlength=nunits)
    allowed_u = idx.u_allow >= 0
    # draw-union coverage of each unit (boilerplate excluded)
    union_pos = np.unique(np.concatenate([h[3] for h in hits])) if hits else np.empty(0, np.int64)
    if len(boiler_arr) and len(union_pos):
        union_pos = union_pos[~np.isin(union_pos, boiler_arr)]
    ucov_cnt = np.bincount(idx.units_of(union_pos), minlength=nunits) if len(union_pos) else np.zeros(nunits, np.int64)
    ucov = np.where(den > 0, ucov_cnt / np.maximum(den, 1), 0.0)
    union_units = (ucov >= THRESH) & (ucov_cnt >= np.minimum(MIN_COVERED, den)) & (den > 0)
    fails: list[dict] = []
    saved: dict[str, dict[str, int]] = {}
    for rid, src, ng, pos in hits:
        nb = pos[~np.isin(pos, boiler_arr)] if len(boiler_arr) else pos
        u = idx.units_of(nb)
        if len(u):
            uu, uc = np.unique(u, return_counts=True)
        else:
            uu, uc = np.empty(0, np.int32), np.empty(0, np.int64)
        strict = nb[(idx.fl[nb] & 1) > 0]
        r_cov = len(strict) / ng
        r_fail = r_cov >= THRESH and len(strict) >= min(MIN_COVERED, ng)
        r_cov_all = len(nb) / ng
        r_fail_all = r_cov_all >= THRESH and len(nb) >= min(MIN_COVERED, ng)
        d = den[uu] if len(uu) else np.empty(0, np.int64)
        ok_min = uc >= np.minimum(MIN_COVERED, d)
        cov = np.where(d > 0, uc / np.maximum(d, 1), 0.0)
        e_mask = (cov >= THRESH) & ok_min
        u_mask = union_units[uu] & (cov >= UNION_CONTRIB) & ok_min if len(uu) else np.empty(0, bool)
        al = allowed_u[uu] if len(uu) else np.empty(0, bool)
        e_fail = bool((e_mask & ~al).any())
        u_fail = bool((u_mask & ~al).any())
        reasons = "".join(c for c, f in (("R", r_fail), ("E", e_fail), ("U", u_fail)) if f)
        if reasons:
            cand = np.nonzero((e_mask | u_mask) & ~al)[0]
            best = int(uu[cand[np.argmax(cov[cand])]]) if len(cand) else int(uu[np.argmax(cov)]) if len(uu) else -1
            lab = idx.unit_label(best) if best >= 0 else {}
            fails.append({"id": rid, "source": src, "reasons": reasons, "row_coverage": round(r_cov, 4),
                          "row_grams": ng, "best_unit_coverage": round(float(cov[uu == best][0]), 4) if best >= 0 else 0.0,
                          "best_unit": lab})
        elif r_fail_all or bool((e_mask & al).any()):  # saved only by the allowlist
            ent_ids = idx.meta["allowlist_ids"]
            if (e_mask & al).any():
                j = np.nonzero(e_mask & al)[0]
                e = int(idx.u_allow[int(uu[j[np.argmax(cov[j])]])])
            else:
                al_only = nb[(idx.fl[nb] & 1) == 0]
                vals, cnts = np.unique(idx.ent[al_only], return_counts=True)
                e = int(vals[np.argmax(cnts)])
            name = ent_ids[e] if 0 <= e < len(ent_ids) else "?"
            saved.setdefault(name, {})
            saved[name][src] = saved[name].get(src, 0) + 1
    fail_ids = sorted(f["id"] for f in fails)
    per_source_fail: dict[str, int] = {}
    per_reason: dict[str, int] = {}
    per_eval_family: dict[str, int] = {}
    eval_units_hit: dict[str, set] = {}
    for f in fails:
        per_source_fail[f["source"]] = per_source_fail.get(f["source"], 0) + 1
        for c in f["reasons"]:
            per_reason[c] = per_reason.get(c, 0) + 1
        fam = (f["best_unit"] or {}).get("family")
        per_eval_family[str(fam)] = per_eval_family.get(str(fam), 0) + 1
        bu = f["best_unit"] or {}
        eval_units_hit.setdefault(bu.get("split", "?"), set()).add(bu.get("run_id"))
    union_split: dict[str, int] = {}
    union_rows: dict[str, set] = {}
    for u in np.nonzero(union_units & ~allowed_u)[0].tolist():
        lab = idx.meta["rows"][int(idx.u_row[u])]
        union_rows.setdefault(lab[0], set()).add(lab[1])
    for s, v in union_rows.items():
        union_split[s] = len(v)
    rep = {"rule": RULE, "index_sha256": idx.meta.get("index_sha256"), "allowlist_sha256": _allow_sha_safe(),
           "n_eval_rows": idx.n_rows, "n_records": n, "n_rows_any_eval_gram": len(hits), "n_fail": len(fails),
           "fail_ids_sha256": ids_sha256(fail_ids), "per_source_rows": dict(sorted(per_source_rows.items())),
           "per_source_fail": dict(sorted(per_source_fail.items())), "per_reason": per_reason,
           "per_eval_family_fail": dict(sorted(per_eval_family.items())),
           "eval_rows_attributed_by_split": {k: len(v) for k, v in sorted(eval_units_hit.items())},
           "eval_rows_union_covered_by_split": dict(sorted(union_split.items())),
           "allowlist_saved": saved, "boilerplate": _boiler_report(idx, boiler, cand_df),
           "secs": round(time.time() - t0, 1)}
    if keep_ids:
        rep["fail_ids"] = fail_ids
        rep["fails"] = fails[:5000]
    return rep


def _allow_sha_safe() -> str | None:
    try:
        return allowlist_sha256()
    except OSError:
        return None


def _boiler_report(idx: EvalIndex, boiler: set[int], cand_df: dict[int, int]) -> list[dict]:
    """Boilerplate grams by hash (text is never stored on the eval side), with eval and train df."""
    return sorted(({"gram_hash": f"{int(idx.h[p]):016x}", "eval_df": int(idx.df[p]), "train_df": cand_df[p]}
                   for p in boiler), key=lambda d: -d["train_df"])


def scan_file(path: str | Path, keep_ids: bool = True, ids_file: str | Path | None = None,
              idx: EvalIndex | None = None) -> dict[str, Any]:
    want = None
    if ids_file:
        op = gzip.open if str(ids_file).endswith(".gz") else open
        with op(ids_file, "rt", encoding="utf-8") as fh:
            want = {l.strip() for l in fh if l.strip()}
    rep = scan(iter_jsonl(path), idx=idx, keep_ids=keep_ids, want=want)
    rep["records"] = str(path)
    if ids_file:
        rep["ids_file"] = str(ids_file)
    return rep


def record_fails(recs: Iterable[dict], idx: EvalIndex | None = None) -> list[str]:
    """Ids of the given records that FAIL (boilerplate and rule U computed over exactly these records)."""
    return scan(recs, idx=idx, keep_ids=True)["fail_ids"]


# ------------------------------------------------------------------------------------------------- pinned clean spec
def clean_spec(rep: dict[str, Any]) -> dict[str, Any]:
    """Pinned as ``train_set.doc_clean`` (or a dev cell's top-level ``doc_clean``): the set's failing rows leave the pool."""
    return {"rule": RULE, "filter": "drop every row train.doc_overlap_guard.scan FAILS over the whole set",
            "index_sha256": rep["index_sha256"], "allowlist_sha256": rep["allowlist_sha256"],
            "n_records": rep["n_records"], "n_excluded": rep["n_fail"], "n_kept": rep["n_records"] - rep["n_fail"],
            "excluded_ids_sha256": rep["fail_ids_sha256"], "per_source_fail": rep["per_source_fail"]}


def check_pinned(rep: dict[str, Any], pin: dict[str, Any] | None) -> tuple[bool, str]:
    if not rep["n_fail"]:
        return True, "no fail"
    if not pin:
        return False, f"{rep['n_fail']} row(s) carry Index eval document text and no doc_clean spec is pinned"
    for k in ("index_sha256", "allowlist_sha256", "excluded_ids_sha256", "n_excluded"):
        if k not in pin:
            return False, f"doc_clean pin lacks {k}"
    if pin["index_sha256"] != rep["index_sha256"] or pin["allowlist_sha256"] != rep["allowlist_sha256"]:
        return False, "doc_clean pin was made against a different eval index / allowlist"
    if int(pin["n_excluded"]) != rep["n_fail"] or pin["excluded_ids_sha256"] != rep["fail_ids_sha256"]:
        return False, (f"doc-overlap fails ({rep['n_fail']}, {rep['fail_ids_sha256'][:12]}) != pinned exclusions "
                       f"({pin['n_excluded']}, {str(pin['excluded_ids_sha256'])[:12]})")
    return True, "fails == pinned exclusions"


def summary(rep: dict[str, Any]) -> dict[str, Any]:
    return {k: rep[k] for k in ("n_records", "n_fail", "per_source_fail", "per_reason",
                                "eval_rows_attributed_by_split", "eval_rows_union_covered_by_split", "secs")}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.doc_overlap_guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build"); s.add_argument("--write-pin", action="store_true")
    s = sub.add_parser("scan"); s.add_argument("--records", required=True); s.add_argument("--ids"); s.add_argument("--out")
    s = sub.add_parser("clean-spec"); s.add_argument("--records", required=True); s.add_argument("--out", required=True)
    s.add_argument("--ids-out")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            t0 = time.time()
            idx = get_index()
            key, _ = index_key()
            pin = pin_record(idx, key) | {"build_secs": round(time.time() - t0, 1)}
            if a.write_pin:
                PIN_PATH.parent.mkdir(parents=True, exist_ok=True)
                PIN_PATH.write_text(json.dumps(pin, indent=1) + "\n", encoding="utf-8")
            print(json.dumps(pin))
            return 0
        rep = scan_file(a.records, ids_file=getattr(a, "ids", None))
    except DocOverlapError as e:
        print(f"doc_overlap_guard: {e}", flush=True)
        return EXIT_HIT
    if a.cmd == "clean-spec":
        spec = clean_spec(rep) | {"records": str(a.records)}
        Path(a.out).write_text(json.dumps(spec, indent=1), encoding="utf-8")
        if a.ids_out:
            with gzip.open(a.ids_out, "wt", encoding="utf-8") as fh:
                fh.write("\n".join(rep["fail_ids"]) + "\n")
        print(json.dumps({k: spec[k] for k in ("n_records", "n_excluded", "n_kept", "excluded_ids_sha256")}))
        return 0
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(summary(rep)))
    return EXIT_HIT if rep["n_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
