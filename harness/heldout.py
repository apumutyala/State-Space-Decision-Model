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

A record/generator is refused when any non-content leaf (every key except item content such as state/questions/
expected/labels; keys matched case-insensitively, any nesting depth, free-text descriptions included) names a
frozen benchmark: a frozen family name or alias as a substring after lower-casing and dropping non-alphanumerics,
"catalog<sep>NN" anywhere, or any integer token under a key containing target/catalog/benchmark (58, 58.0,
"58, 9", "catalog-58", dict keys, ids inside lists of dicts). A generator must also declare a NON-EMPTY
task_family or targets. It is deliberately fail-closed: a false positive is fixed by renaming a tag,
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
# Keys whose values are item CONTENT (prompts, options, gold); never scanned, so item text cannot false-positive.
CONTENT_KEYS = {"state", "questions", "expected", "labels", "answers", "response", "criteria", "untrusted_fields",
                "choices", "options", "prompt", "completion", "messages", "text"}
# Keys whose values (and everything nested under them, dict keys included) are read as benchmark ids: any key
# containing "catalog" or "benchmark", plus these target spellings. (A bare "target" is NOT one: injection records
# use it for the hijack's target option, e.g. "k_9e2a6267".)
ID_KEY_PARTS = ("catalog", "benchmark")
ID_KEYS = {"targets", "target_benchmark", "target_benchmarks", "target_catalog", "target_catalogs", "target_ids"}
DECLARE_KEYS = ("task_family", "targets", "target_benchmarks")
_ID_TOKEN = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.0+)?(?![A-Za-z0-9])")
_CATALOG_ANYWHERE = re.compile(r"catalog\s*[:#_\-\s]\s*(\d+)", re.I)


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


def _is_id_key(k):
    k = str(k).lower()
    return k in ID_KEYS or any(part in k for part in ID_KEY_PARTS)


def _leaves(obj, id_ctx=False):
    """(value, is_id_context) for every scalar leaf outside content keys; dict keys under an id key count too."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in CONTENT_KEYS:
                continue
            ctx = id_ctx or _is_id_key(k)
            if id_ctx:
                yield k, True
            yield from _leaves(v, ctx)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _leaves(v, id_ctx)
    elif obj is not None and not isinstance(obj, bool):
        yield obj, id_ctx


def _ids_in(v):
    if isinstance(v, (int, float)):
        return [int(v)] if float(v).is_integer() else []
    return [int(float(t)) for t in _ID_TOKEN.findall(str(v))]


def hits(obj):
    """[(tag, frozen catalog id)] for every non-content leaf that names a frozen benchmark/family.

    - names, families and aliases: substring match on any string leaf (lower-cased, non-alphanumerics dropped),
      free-text descriptions included;
    - ids: any standalone integer token (58, 58.0, "58, 9", "catalog-58", dict keys) under an id key (contains
      catalog/benchmark, or targets/target_benchmarks/...), and "catalog<sep>NN" anywhere."""
    fz = set(frozen())
    out = []
    for v, id_ctx in _leaves(obj):
        if id_ctx:
            out += [(v, n) for n in _ids_in(v) if n in fz]
        if isinstance(v, str):
            out += [(v, int(m)) for m in _CATALOG_ANYWHERE.findall(v) if int(m) in fz]
            ns = _norm(v)
            out += [(v, n) for tok, n in _tokens().items() if tok in ns]
    return out


def _declared(v):
    if isinstance(v, bool) or v is None:
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple, set, dict)):
        return any(_declared(x) for x in (v.keys() if isinstance(v, dict) else v)) or (isinstance(v, dict) and any(_declared(x) for x in v.values()))
    return False


def check_record(rec):
    h = hits(rec)
    if h:
        raise FrozenFamilyError(f"record {rec.get('id')!r} targets frozen held-out benchmark(s) {sorted({n for _, n in h})} via tags {sorted({str(t) for t, _ in h})}")
    return rec


def check_generator(gen_spec):
    """A generator spec must declare `task_family` or `targets`; refused if either names a frozen family."""
    if not any(_declared(gen_spec.get(k)) for k in DECLARE_KEYS):
        raise FrozenFamilyError("generator spec must declare a non-empty task_family or targets (fail-closed)")
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
