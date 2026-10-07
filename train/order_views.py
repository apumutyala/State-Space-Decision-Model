"""Opt-in reversed option descriptions for permutation-consistency training data.

The ``ordered`` layout renders a choice question's options in canonical order (sorted id, ``decide.schema``).
This helper constructs paired examples for a future consistency objective; it does not enable a trainer or cell.

``reversed_record(rec, qid)`` is the second view of one choice question: the SAME option ids in the same canonical
order, with the option TEXTS assigned in reverse (the text shown last is now shown first) and the gold label moved
with its text. Every other field of the record is unchanged, so the shared prefix differs only in the target
question's option block. ``Item.view`` holds that view's row and ``Item.view_perm[k]`` = the base option index of
view option ``k`` (for a reversal, ``K - 1 - k``), so the trainer can compare the two distributions option by option.

Only ``choice`` questions with K >= 2 and explicit descriptions get a view (``noul`` is fixed true/false; ``score``
levels are ordinal and never permuted, ``heads.layout``). Views are deterministic (no RNG: a reversal), and a view
that the cell's K / T filter would drop leaves its base item single-view.

Attachment is explicit: callers must supply only the training list. The ordinary item builder never attaches views.
Choices whose descriptions are None use the ids themselves as content; swapping None cannot preserve their labels.
The caller must also ensure answer semantics follow descriptions: references to particular option ids in the state
or instructions can make reassignment invalid even when descriptions are present.
"""
from __future__ import annotations

import copy
from typing import Any

from decide.schema import canonical_options

#: The only view construction this module implements. Cell configuration / trainer integration are separate work.
VIEW = "reverse"


def reversed_record(rec: dict[str, Any], qid: str) -> tuple[dict[str, Any], list[int]] | None:
    """(view record, perm) for choice question ``qid`` of ``rec``, or None if the question gets no view.
    ``perm[k]`` = the canonical index in ``rec`` of the option shown at canonical index ``k`` in the view."""
    from train.guard import record_request

    req = record_request(rec)
    if req is None or qid not in (req.get("questions") or {}) or qid not in (rec.get("labels") or {}):
        return None
    q = req["questions"][qid]
    if q.get("type") != "choice":
        return None
    opts = canonical_options(q)
    k = len(opts)
    if k < 2:
        return None
    ids = [o for o, _ in opts]
    texts = [t for _, t in opts]
    if any(t is None for t in texts):
        return None
    gold = str(rec["labels"][qid])
    if gold not in ids:
        return None
    perm = list(reversed(range(k)))  # view index j shows base index perm[j]
    view = copy.deepcopy(rec)
    if "request" in view and view["request"] is not None:
        new_q = view["request"]["questions"][qid]
    else:
        new_q = view["questions"][qid]
    new_q["criteria"] = {ids[j]: copy.deepcopy(texts[perm[j]]) for j in range(k)}
    view["labels"][qid] = ids[perm.index(ids.index(gold))]
    view["id"] = f"{rec.get('id', '?')}#rev:{qid}"
    return view, perm


def attach_views(items: list, records: dict[str, dict], enc, layout: str, template: str, guard, keep) -> dict:
    """Attach ``view`` / ``view_perm`` to every eligible item in place; ``keep(row) -> bool`` is the cell's K / T
    filter. Returns counts for cell.json. The view row goes through ``items_from_records`` (same guard check, same
    encoder, same template and layout) as its base."""
    from train.data import items_from_records

    n_view = n_none = n_drop = 0
    for it in items:
        rid, qid = it.id.rsplit("/", 1)
        got = reversed_record(records[rid], qid)
        if got is None:
            n_none += 1
            continue
        view_rec, perm = got
        vit = [x for x in items_from_records([{**view_rec, "labels": {qid: view_rec["labels"][qid]}}], enc,
                                             layout=layout, template=template, guard=guard)]
        if len(vit) != 1 or not keep(vit[0].row):
            n_drop += 1
            continue
        v = vit[0]
        base_ids = it.row.option_ids
        if v.row.option_ids != base_ids or base_ids[perm[v.label]] != base_ids[it.label]:
            raise ValueError(f"{it.id}: reversed view does not map gold onto gold")
        it.view, it.view_perm = v, tuple(perm)
        n_view += 1
    return {"view": VIEW, "with_view": n_view, "no_view": n_none, "view_filtered": n_drop}
