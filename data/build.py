"""End-to-end sample build: load -> render/augment -> inject -> validate -> held-out guard -> Index dedup -> write.

Outputs (outside git): ``<out>/records.jsonl.gz``. Committed: a manifest (record id + sha256 per line) and the
receipt (dedup log, summary). Quotas are counts of KEPT records, so dropped records are replaced by the next item
of the same source; the dedup log counts every record checked.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

from . import augment as A
from . import config, convert as C, injection as I
from .dedup import DedupLog, Fingerprint
from .text import canonical_json, words

# Kept-record shares of the sample (sum 1.0).
SHARES = {"banking77": 0.07, "clinc150": 0.07, "anli": 0.09, "hellaswag": 0.08, "winogrande": 0.07,
          "gsm8k": 0.07, "mmlu_aux": 0.09, "sst5": 0.05, "boolq": 0.06, "when2call": 0.06, "toolace": 0.08,
          "nq": 0.09, "multi": 0.06, "factchange": 0.06}
MULTI_FROM = ["banking77", "clinc150", "anli", "winogrande", "hellaswag", "mmlu_aux", "gsm8k", "sst5", "boolq"]
P_INJECT = 0.10          # in-place injection on single-question records (label unchanged)
P_INJECT_FACT = 0.15     # one variant of a fact-change pair also carries an injection
P_PAD = 0.08


def _rec_sha(rec: dict) -> str:
    return hashlib.sha256(canonical_json(rec).encode("utf-8")).hexdigest()


def _len_words(rec: dict) -> int:
    return len(words(canonical_json({"state": rec["state"], "questions": rec["questions"]})))


def build_sample(n_target: int = 20000, out: str | None = None, receipt: str | None = None,
                 seed: int = config.SEED, fingerprint: Fingerprint | None = None, sources: dict | None = None,
                 log=print) -> dict:
    t0 = time.time()
    fp = fingerprint or Fingerprint.load(config.FINGERPRINT)
    dlog = DedupLog(fp.meta)
    shares = sources or SHARES
    if abs(sum(shares.values()) - 1.0) > 1e-9:
        raise ValueError(f"shares sum to {sum(shares.values())}, not 1")
    quotas = {k: int(round(v * n_target)) for k, v in shares.items()}
    quotas[max(quotas, key=quotas.get)] += n_target - sum(quotas.values())
    rng_master = random.Random(seed)
    iters = {sid: C.LOADERS[sid](seed) for sid in C.LOADERS if sid in quotas or sid in MULTI_FROM}
    pad_pool = []
    if "nq" in C.LOADERS:
        rows = C.nq_pairs(seed)
        r = random.Random(f"{seed}|pad")
        pad_pool = [rows[r.randrange(len(rows))]["answer"] for _ in range(4000)]
    kept: list[dict] = []
    stats = collections.defaultdict(collections.Counter)
    held_out_checked = 0

    def accept(rec: dict) -> bool:
        nonlocal held_out_checked
        A.validate(rec)
        I.assert_no_heldout(rec["state"], rec["questions"], rec["id"])
        held_out_checked += 1
        item = fp.check(rec["_probe"], {})
        full = fp.check(rec["state"], rec["questions"])
        return dlog.add(rec["source"] if rec["family"] != "multi" else "multi", rec["id"], item, full)

    def note(rec: dict, bucket: str):
        s = stats[bucket]
        s["records"] += 1
        s[f"family:{rec['family']}"] += 1
        a = rec.get("aug", {})
        for k in ("placement", "state_shape", "id_style", "shape"):
            if k in a:
                s[f"{k}:{a[k]}"] += 1
        if "injection" in a:
            s[f"injection:{a['injection']['family']}:{a['injection']['template']}"] += 1
        if a.get("padding_passages"):
            s["padded"] += 1
        for q in rec["questions"].values():
            s[f"qtype:{q['type']}"] += 1
            if q["type"] == "choice":
                n = len(q["criteria"])
                b = "2" if n == 2 else "3-4" if n <= 4 else "5-10" if n <= 10 else "11-64" if n <= 64 else \
                    "65-151" if n <= 151 else "152-255"
                s[f"options:{b}"] += 1
        L = _len_words(rec)
        lb = "<64" if L < 64 else "64-255" if L < 256 else "256-1023" if L < 1024 else "1024-4095" if L < 4096 else ">=4096"
        s[f"words:{lb}"] += 1

    # ---- single-source records
    for sid in [s for s in quotas if s in C.LOADERS]:
        got = 0
        for base in iters[sid]:
            if got >= quotas[sid]:
                break
            rng = random.Random(f"{seed}|render|{base['id']}")
            rec = A.render(base, rng, pad_pool=pad_pool, p_pad=P_PAD)
            if rec["aug"].get("shape") == "single" and rng.random() < P_INJECT:
                inj = I.inject(rec, rng)
                rec = inj or rec
            if accept(rec):
                kept.append(rec); note(rec, sid); got += 1
        if got < quotas[sid]:
            log(f"  {sid}: source exhausted at {got}/{quotas[sid]}")
        log(f"  {sid}: kept {got} ({time.time() - t0:.0f}s)")

    # ---- multi-question records (items drawn from the same iterators: no item reused)
    got, k = 0, 0
    while got < quotas.get("multi", 0):
        rng = random.Random(f"{seed}|multi|{k}"); k += 1
        m = rng.choice([2, 2, 3, 4])
        parts = []
        for _ in range(m):
            sid = rng.choice(MULTI_FROM)
            base = next(iters[sid], None)
            if base is None:
                continue
            parts.append(A.render(base, random.Random(f"{seed}|render|{base['id']}"), placement="state"))
        if len(parts) < 2:
            continue
        rec = A.merge(parts, rng)
        if accept(rec):
            kept.append(rec); note(rec, "multi"); got += 1
    log(f"  multi: kept {got}")

    # ---- fact-change pairs (both variants kept or both dropped)
    got, k = 0, 0
    fams = sorted(I.FACT_FAMILIES)
    while got < quotas.get("factchange", 0):
        fam = fams[k % len(fams)]
        pair = I.fact_change_pair(fam, k, seed)
        rng = random.Random(f"{seed}|fcaug|{k}"); k += 1
        pair = [A.reshuffle_options(p, rng) for p in pair]
        if rng.random() < P_INJECT_FACT:
            v = rng.randrange(2)
            inj = I.inject(pair[v], rng, family="d1a")
            if inj is not None:
                pair[v] = inj
        oks = [accept(p) for p in pair]
        if all(oks):
            for p in pair:
                kept.append(p); note(p, "factchange"); got += 1
    log(f"  factchange: kept {got}")

    dup = [k for k, v in collections.Counter(r["id"] for r in kept).items() if v > 1]
    if dup:
        raise AssertionError(f"duplicate record ids: {dup[:5]}")
    # ---- write
    out_dir = Path(out or config.BUILD_DIR / f"sample-{n_target}-seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl.gz"
    manifest_lines = []
    with gzip.open(rec_path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as f:
        for rec in kept:
            rec.pop("_probe", None)
            line = canonical_json(rec)
            f.write(line + "\n")
            manifest_lines.append(canonical_json({"id": rec["id"], "sha256": hashlib.sha256(line.encode()).hexdigest(),
                                                  "source": rec["source"], "family": rec["family"]}))
    h = hashlib.sha256()
    with gzip.open(rec_path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    total = collections.Counter()
    for c in stats.values():
        total.update(c)
    summary = {
        "n_target": n_target, "n_kept": len(kept), "seed": seed, "quotas": quotas,
        "records_uncompressed_sha256": h.hexdigest(), "records_path": str(rec_path),
        "held_out_template_guard_checked": held_out_checked,
        "dedup_totals": dlog.to_json()["totals"],
        "by_bucket": {k: dict(sorted(v.items())) for k, v in sorted(stats.items())},
        "overall": dict(sorted(total.items())),
        "seconds": round(time.time() - t0, 1),
    }
    res = {"summary": summary, "dedup_log": dlog.to_json(), "manifest_lines": manifest_lines, "records": kept}
    if receipt:
        rd = Path(receipt)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "dedup-log.json").write_text(json.dumps(res["dedup_log"], indent=1) + "\n", encoding="utf-8")
        (rd / "build-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        mp = config.MANIFEST_DIR / f"sample-{n_target}-seed{seed}.manifest.jsonl.gz"
        with gzip.GzipFile(mp, "wb", mtime=0) as f:
            f.write(("\n".join(manifest_lines) + "\n").encode("utf-8"))
    return res
