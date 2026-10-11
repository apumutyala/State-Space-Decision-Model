"""Attribute full-row Index Fingerprint hits (regate_jev2a fingerprint) to segments: role, length, catalog. Counts only.

    python -X utf8 scripts/regate_fp_attrib.py --draw <dir> --fp-report index-fingerprint.json --out fp-attrib.json
"""
import argparse, collections, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import regate_jev2a as RG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from data import config, text as T
    from data.dedup import Fingerprint
    fp = Fingerprint.load(config.FINGERPRINT)
    by_role_len = collections.Counter(); by_cat = collections.Counter(); by_matcher_role = collections.Counter()
    rows_state_hit = collections.Counter(); rows_only_q = collections.Counter(); rows = 0
    long_state_examples = []
    for r in RG.recs(Path(a.draw) / "records.jsonl.gz"):
        st = r["state"] if isinstance(r["state"], dict) else {"state": r["state"]}
        s_segs = T.segments(st, {})
        q_segs = [s for s in T.segments(None, r["questions"])]
        hit_state = hit_q = False
        for role, segs in (("state", s_segs), ("question", q_segs)):
            for s in segs:
                h = {k: v for k, v in fp.check({"x": s}, {}).items() if v}
                if not h:
                    continue
                n = len(s.split())
                b = "1-2" if n <= 2 else "3-5" if n <= 5 else "6-12" if n <= 12 else "13+"
                by_role_len[f"{role}/{b}w"] += 1
                for k, cats in h.items():
                    by_matcher_role[f"{role}/{k}"] += 1
                    for c in cats:
                        by_cat[str(c)] += 1
                if role == "state":
                    hit_state = True
                    if n >= 6 and len(long_state_examples) < 30:
                        long_state_examples.append([r["id"], n, sorted(h)])
                else:
                    hit_q = True
        if hit_state or hit_q:
            rows += 1
            if hit_state:
                rows_state_hit[RG.fam(r)] += 1
            else:
                rows_only_q[RG.fam(r)] += 1
        if len(getattr(fp, "_seg_cache", ())) > 100_000:
            fp._seg_cache.clear(); fp._txt_cache.clear()
    out = {"rows_with_any_segment_hit": rows, "rows_with_state_segment_hit_by_family": dict(rows_state_hit),
           "rows_only_question_segment_hit_by_family": dict(rows_only_q),
           "segment_hits_by_role_and_wordlen": dict(by_role_len), "segment_hits_by_role_matcher": dict(by_matcher_role),
           "catalog_hits_top": collections.Counter(by_cat).most_common(25),
           "state_segment_hits_ge6_words_examples": long_state_examples}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out)[:3000])


if __name__ == "__main__":
    main()
