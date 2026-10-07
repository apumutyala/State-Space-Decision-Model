"""Frozen new-domain proxy (heldout-draw-v1) and the contamination check data builders must call.

The 18 zero-coverage benchmarks (families-v1 tier 'none', edition 0.3) were split by a seeded draw into
9 FROZEN and 9 GENERATABLE (manifests/heldout-draw-v1.json; `draw()` reproduces it). Frozen benchmarks are
the new-domain proxy and must never be targeted by training data or generators.

Data builders:
    from harness.heldout import check_record, check_generator, FrozenFamilyError
    check_generator({"name": "gen_cruxeval", "task_family": "code_execution", "targets": [43]})   # ok
    check_record(record)   # raises FrozenFamilyError if any tag names a frozen benchmark or family

CLI (exit 1 on any hit):
    python -X utf8 -I -m harness.heldout check records.jsonl[.gz] [more files]

A record/generator is refused when any of its tag fields (`source`, `family`, `task_family`, `generator`,
`name`, `targets`/`target_benchmarks`, also inside `meta`/`metadata`/`provenance`/`aug`) contains a frozen
catalog id (as int or "catalog:<id>") or, after lower-casing and dropping non-alphanumerics, contains a frozen
family name or alias as a substring. It is deliberately fail-closed: a false positive is fixed by renaming a tag,
never by editing the frozen list (that needs a new heldout-draw version and the operator's approval).
"""

import functools
import gzip
import hashlib
import json
import re
import sys

from harness import config, families

DRAW = config.MANIFEST_DIR / "heldout-draw-v1.json"
TAG_FIELDS = ("source", "family", "task_family", "generator", "name", "targets", "target_benchmarks", "benchmark", "catalog_id")
NESTED = ("meta", "metadata", "provenance", "aug")


class FrozenFamilyError(ValueError):
    pass


@functools.lru_cache(maxsize=None)
def spec():
    raw = DRAW.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.HELDOUT_DRAW_V1_SHA256:
        raise RuntimeError("heldout-draw-v1.json changed; it is frozen -- a new draw needs a new version file and operator approval")
    return json.loads(raw.decode("utf-8"))


def pool(edition="0.3"):
    from harness.suite import index_ids

    return sorted(n for n in index_ids(edition) if families.tier_of(n) == "none")


def draw(seed, pool_ids, k=None):
    """(frozen, generatable), both sorted. key(n) = sha256('heldout-draw-v1:{seed}:{n}')."""
    k = len(pool_ids) // 2 if k is None else k
    order = sorted(pool_ids, key=lambda n: hashlib.sha256(f"heldout-draw-v1:{seed}:{n}".encode()).hexdigest())
    return sorted(order[:k]), sorted(order[k:])


def frozen():
    return list(spec()["frozen"])


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@functools.lru_cache(maxsize=None)
def _tokens():
    fam = families.load()["benchmarks"]
    out = {}
    for n in frozen():
        b = fam[str(n)]
        for t in [b["family"], b["name"], *spec()["aliases"][str(n)]]:
            if len(_norm(t)) >= 3:
                out[_norm(t)] = n
    return out


def _tags(obj):
    for f in TAG_FIELDS:
        if f in obj and obj[f] is not None:
            v = obj[f]
            yield from (v if isinstance(v, (list, tuple, set)) else [v])
    for f in NESTED:
        if isinstance(obj.get(f), dict):
            yield from _tags(obj[f])


def hits(obj):
    """[(tag, frozen catalog id)] for every tag that names a frozen benchmark/family."""
    fz = set(frozen())
    out = []
    for t in _tags(obj):
        if isinstance(t, bool):
            continue
        if isinstance(t, int):
            if t in fz:
                out.append((t, t))
            continue
        s = str(t)
        m = re.fullmatch(r"(?:catalog:)?(\d+)", s.strip())
        if m and int(m.group(1)) in fz:
            out.append((s, int(m.group(1))))
            continue
        ns = _norm(s)
        for tok, n in _tokens().items():
            if tok in ns:
                out.append((s, n))
    return out


def check_record(rec):
    h = hits(rec)
    if h:
        raise FrozenFamilyError(f"record {rec.get('id')!r} targets frozen held-out benchmark(s) {sorted({n for _, n in h})} via tags {sorted({str(t) for t, _ in h})}")
    return rec


def check_generator(gen_spec):
    """A generator spec must declare `task_family` or `targets`; refused if either names a frozen family."""
    if not any(k in gen_spec for k in ("task_family", "targets", "target_benchmarks")):
        raise FrozenFamilyError("generator spec must declare task_family or targets (fail-closed)")
    h = hits(gen_spec)
    if h:
        raise FrozenFamilyError(f"generator {gen_spec.get('name')!r} targets frozen held-out benchmark(s) {sorted({n for _, n in h})}")
    return gen_spec


def check_files(paths):
    total, bad = 0, []
    for p in paths:
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                total += 1
                h = hits(r)
                if h:
                    bad.append({"id": r.get("id"), "frozen": sorted({n for _, n in h})})
    return {"draw": spec()["id"], "frozen": frozen(), "records": total, "refused": len(bad), "examples": bad[:20]}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] != "check":
        sys.exit(__doc__)
    rep = check_files(argv[1:])
    print(json.dumps(rep, indent=1))
    return 1 if rep["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
