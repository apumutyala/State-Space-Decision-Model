"""Score-item invariance requests (evidence lane; SEPARATE from jevkit's G1, never a replacement for it).

Why: frozen ``jevkit/perturb.py`` never perturbs SCORE items (option_reverse / option_shuffle are n/a for score,
key_relabel is choice-only), so G1 says nothing about score items. This script emits, for score items only, the two
symmetries of docs/guarantees/REORDER_INVARIANCE.md Theorem 2T (claude/ev-guarantee-proofs):

  (i)  display-order permutations of the score options:   score_reverse (seed 0), score_shuffle (--shuffle-seeds)
  (ii) handle relabels, level value travelling in CONTENT: score_relabel  (fresh opaque handles ``s_<hex>``, display
       order kept canonical), score_permute (the handles "0".."K-1" permuted among the levels, a non-identity
       bijection; display order kept canonical)

Level renaming (2T (iii)) is NOT emitted: it changes the question.

  python -I -X utf8 scripts/ev_score_perturb.py (--public | --items <items.jsonl>) --out <dir> \
      [--shuffle-seeds 0,1,2] [--ops score_reverse,score_shuffle,score_relabel,score_permute]

Rendering (deterministic, identical for base and perturbed requests; ``RENDERING``):
  * every option's CONTENT is ``level v of K: <rubric text>`` with v the jevkit level value 0..K-1 (= the base label)
    and K the number of levels; the rubric text is ``question.criteria[v]`` verbatim;
  * ``question.scale``: the full ordered scale in CANONICAL level order (v = 0..K-1), contents only, no handles. It is
    the same list in every request of an item (the shared prefix of the lead's JEV3-TYPED design);
  * ``question.options``: {handle: content} in DISPLAY order. This dict's key order is what the model sees, so the
    files are written WITHOUT sort_keys and the on-disk order is asserted == the in-memory order after writing (#245);
  * ``labels``: the handles in display order (a list, so it survives a sort_keys dump and exposes one).
  Base and perturbed requests therefore differ ONLY by option display order and handles (state, instructions, scale
  and the content multiset are byte-identical; ev_score_flip.py re-checks this).

Requests: ``base.jsonl`` (handles "0".."K-1", canonical order, op ``base``) and ``perturbed.jsonl`` (id
``<base id>~<op>~<seed>``; ``perturbation.label_map`` = {base label (level) -> perturbed handle}). ``manifest.json``
holds counts per op, file sha256s and the rendering id.

READ THIS before using the result:
  * The BASE readout must also be produced from ``base.jsonl`` (level-as-content rendering). Comparing perturbed
    requests against preds made with jevkit's stock score rendering mixes a rendering change into every pair and
    makes the check meaningless.
  * A head that reads the level from the HANDLE (e.g. int(handle)) is EXPECTED to fail score_relabel /
    score_permute (Corollary 2T.1); that failure is the check working, not a bug in it.
  * Public items are eval-only: this is a format-level use (no training, tuning or selection on them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RENDERING = "ev-score-level-as-content/1: content = 'level v of K: <rubric>' (v = 0..K-1); scale canonical; no handles in scale"
OPS = ("score_reverse", "score_shuffle", "score_relabel", "score_permute")
REORDER_OPS = ("score_reverse", "score_shuffle")
RELABEL_OPS = ("score_relabel", "score_permute")


def render_level(v: int, k: int, rubric) -> str:
    return f"level {v} of {k}: {rubric if isinstance(rubric, str) else json.dumps(rubric, ensure_ascii=False)}"


def _rng(base_id: str, op: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"ev-score-inv\x1f{base_id}\x1f{op}\x1f{seed}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _contents(item) -> list[str]:
    crit = item.question.get("criteria")
    k = item.k
    if isinstance(crit, dict):
        rub = [crit.get(str(v)) for v in range(k)]
    elif isinstance(crit, list) and len(crit) == k:
        rub = list(crit)
    else:
        raise ValueError(f"{item.id}: score criteria must be a list of K rubric texts (or a dict keyed '0'..'K-1')")
    return [render_level(v, k, rub[v]) for v in range(k)]


def _request(item, rid: str, order: list[str], handle: dict[str, str], op: str, seed: int) -> dict:
    """order: base labels (levels) in display order; handle: base label -> handle shown."""
    cont = _contents(item)
    q = {"type": "score", "instructions": item.question.get("instructions"), "scale": cont,
         "options": {handle[lab]: cont[int(lab)] for lab in order}}
    return {"id": rid, "family": item.family, "tier": item.tier, "split": item.split, "state": item.state,
            "question": q, "labels": [handle[lab] for lab in order], "k": item.k, "rendering": RENDERING,
            "perturbation": {"op": op, "seed": seed, "base_id": item.id, "label_map": {lab: handle[lab] for lab in item.labels},
                             "display_order": list(order)}}


def base_request(item) -> dict:
    return _request(item, item.id, list(item.labels), {lab: lab for lab in item.labels}, "base", 0)


def perturb(item, op: str, seed: int) -> dict | None:
    if item.type != "score" or item.k < 2:
        return None
    labs = list(item.labels)
    ident = {lab: lab for lab in labs}
    rid = f"{item.id}~{op}~{seed}"
    if op == "score_reverse":
        return _request(item, rid, labs[::-1], ident, op, seed)
    if op == "score_shuffle":
        o = list(labs)
        _rng(item.id, op, seed).shuffle(o)
        if o == labs:
            o = labs[1:] + labs[:1]  # identity draw: rotate (always a real reorder)
        return _request(item, rid, o, ident, op, seed)
    if op == "score_relabel":
        r = _rng(item.id, op, seed)
        hs: list[str] = []
        while len(hs) < len(labs):
            h = "s_" + "%06x" % r.getrandbits(24)
            if h not in hs:
                hs.append(h)
        return _request(item, rid, labs, dict(zip(labs, hs)), op, seed)
    if op == "score_permute":
        p = list(labs)
        _rng(item.id, op, seed).shuffle(p)
        if p == labs:
            p = labs[1:] + labs[:1]
        return _request(item, rid, labs, dict(zip(labs, p)), op, seed)
    raise ValueError(f"unknown op {op!r}")


def make(items, ops=OPS, shuffle_seeds=(0, 1, 2)) -> tuple[list[dict], list[dict]]:
    sc = [it for it in items if it.type == "score"]
    base = [base_request(it) for it in sc]
    pert = []
    for it in sc:
        for op in ops:
            for sd in (shuffle_seeds if op == "score_shuffle" else (0,)):
                p = perturb(it, op, sd)
                if p is not None:
                    pert.append(p)
    return base, pert


def dump_line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False)  # NO sort_keys: the options dict order is the display order


def write_rows(rows: list[dict], path: Path) -> str:
    """Write JSONL, then re-read and assert every row (and its options key order) equals the in-memory row."""
    data = "".join(dump_line(r) + "\n" for r in rows).encode("utf-8")
    Path(path).write_bytes(data)
    back = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(back) == len(rows), f"{path}: {len(back)} rows on disk != {len(rows)} in memory"
    for a, b in zip(rows, back):
        assert a == b and list(a["question"]["options"]) == list(b["question"]["options"]) \
            and a["labels"] == list(b["question"]["options"]), f"{path}: on-disk order != in-memory order for {a['id']}"
    return hashlib.sha256(data).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -X utf8 scripts/ev_score_perturb.py", description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--public", action="store_true", help="the pinned public items (eval-only; format-level use)")
    g.add_argument("--items", default=None, help="any jevkit item JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--shuffle-seeds", default="0,1,2")
    a = ap.parse_args(argv)
    from jevkit.items import load_items, load_public
    items = load_public() if a.public else load_items(a.items)
    ops = [o for o in a.ops.split(",") if o]
    bad = [o for o in ops if o not in OPS]
    if bad:
        ap.error(f"unknown ops {bad}; known {OPS}")
    seeds = tuple(int(s) for s in a.shuffle_seeds.split(",") if s != "")
    base, pert = make(items, ops, seeds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = {"schema": "ev-score-perturb/1", "rendering": RENDERING, "source": "public" if a.public else str(a.items),
           "n_items": len(items), "n_score_items": len(base), "k_hist": {}, "ops": ops, "shuffle_seeds": list(seeds),
           "per_op": {}, "base_sha256": write_rows(base, out / "base.jsonl"),
           "perturbed_sha256": write_rows(pert, out / "perturbed.jsonl"),
           "note": "base readout must use base.jsonl (level-as-content); a handle-reading head is EXPECTED to fail "
                   "score_relabel / score_permute; separate from jevkit G1"}
    for r in base:
        man["k_hist"][str(r["k"])] = man["k_hist"].get(str(r["k"]), 0) + 1
    for r in pert:
        man["per_op"][r["perturbation"]["op"]] = man["per_op"].get(r["perturbation"]["op"], 0) + 1
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("n_items", "n_score_items", "k_hist", "per_op")}))
    return 0 if base else 2


if __name__ == "__main__":
    raise SystemExit(main())
