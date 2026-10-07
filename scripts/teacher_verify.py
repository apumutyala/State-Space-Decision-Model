"""CPU receipt: the Clef-flash teacher export (job-tnrwu, art-ggrs8) against the training draw it claims to cover.

    python -X utf8 scripts/teacher_verify.py --teacher-dir <art>/clef-flash-teacher/teacher \
        --train-dir <ssdm-data>/m2/build/sample-150000-seed20261006 --out receipts/<dir>/teacher_verify.json

Checks (any failure raises; nothing is written unless all pass):
  1. ``train.teacher.load_teacher``: pinned identity, file sha256, draw sha256, per-row content hash vs selection.json,
     well-formed distributions, no DEV/TEST id, no cohort v2 / v3 record (id, content, source item).
  2. The q9 feature cache's training draw, recomputed with THIS tree's ``train.train_set.draw`` over the
     pointer-verified 150k set and ``jobs/q9_feature_cache/cache_config.json`` (cohort v3 excluded before the draw), is
     the teacher's selected-ids list in the same order (so stock_train's 20,000 rows are teacher records).
  3. ``verify_records``: every drawn record's content hash = the teacher row's; the production DEV/TEST guard (content
     index pinned by jobs/m2_pilot/guard_pin.json) and ``harness.heldout.check_record`` (frozen families) pass on it.
  4. Option alignment: every training item (``items_from_records``) maps to its teacher distribution by
     record id / question id / option id, or falls back (exhausted record / question absent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _encode(text: str, trusted: bool = True) -> list[int]:
    """Option ids do not depend on tokenisation; a deterministic stand-in keeps this CPU-only and model-free."""
    return [ord(c) % 997 for c in text] or [0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    from scripts.m2_e2e_dryrun import ensure_kit

    ensure_kit()
    from data.build import content_guard
    from harness.heldout import check_record, frozen
    from train.data import items_from_records
    from train.teacher import attach, load_teacher, sha256_file, verify_records
    from train.train_set import draw

    teacher = load_teacher(a.teacher_dir)
    cfg = json.loads((ROOT / "jobs" / "q9_feature_cache" / "cache_config.json").read_text(encoding="utf-8"))
    by_id, order, ident = draw(cfg, a.train_dir)
    if order != teacher.draw_ids:
        raise SystemExit("the q9 cache training draw != the teacher's selected ids (order or membership)")
    guard, gident = content_guard()
    recs = [by_id[i] for i in order]
    vr = verify_records(teacher, recs, guard=guard, frozen_check=check_record)
    # option alignment, per question, through the trainer's own item builder
    items = []
    for r in recs:
        items += items_from_records([r], _encode, layout="ordered", template=cfg["template"])
    att = attach(items, teacher)
    agree = n_lab = 0
    qtypes: dict[str, int] = {}
    for it in items:
        qtypes[it.row.qtype] = qtypes.get(it.row.qtype, 0) + 1
        if it.teacher is not None:
            n_lab += 1
            agree += int(max(range(len(it.teacher)), key=lambda k: it.teacher[k]) == it.label)
    fallback_records = sorted(i for i, r in teacher.rows.items() if r.probs is None)
    out = {
        "what": "Clef-flash teacher export (job-tnrwu, artifact art-ggrs8) verified against the q9-cache / m2 first-cell "
                "seed-0 training draw, CPU, $0",
        "teacher": teacher.summary(),
        "teacher_files_sha256": {p.name: sha256_file(p) for p in sorted(Path(a.teacher_dir).iterdir())},
        "draw": {"recomputed_with": "train.train_set.draw + jobs/q9_feature_cache/cache_config.json (this tree)",
                 "drawn_ids_sha256": ident["drawn_ids_sha256"], "n_drawn": ident["n_drawn"],
                 "order_equals_teacher_selected_ids": True,
                 "files_uncompressed_sha256": ident["files_uncompressed_sha256"],
                 "cohort_v3_excluded": ident.get("cohort_v3_excluded")},
        "records": {**vr, "guard_identity": {k: v for k, v in gident.items() if k != "content_index"},
                    "content_index": {k: v for k, v in (gident.get("content_index") or {}).items() if k != "path"},
                    "frozen_families": sorted(frozen()), "frozen_family_hits": 0, "guard_hits": 0},
        "items": {**att, "qtypes": qtypes, "teacher_argmax_equals_label": agree / max(n_lab, 1),
                  "n_items_with_teacher": n_lab},
        "fallback_records": {"n": len(fallback_records), "ids_sha256":
                             hashlib.sha256("\n".join(fallback_records).encode()).hexdigest()},
        "secs": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("draw", "items", "fallback_records", "secs")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
