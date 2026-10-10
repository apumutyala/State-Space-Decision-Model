"""Train-row vs jevkit-eval-item rendering diff (tokenizer only, CPU; no weights).

r2's jevkit arm answers each of the 231 pinned public items through ``SystemOneAdapter(decider)`` ->
``DecisionModelBackend._rows`` -> ``heads.layout.build_row(item_request(item), "q", enc, template, layout)``. The cell
trains on ``train.data.items_from_records`` -> the same ``build_row`` on the jev1 record. Same function, so any
train/serve gap is in the DATA SHAPE: state type (str / dict / list), question type (noul / choice / score), option
keys + labels, noul descriptions, prompt skeleton, length. For one eval item per JevBench family (and one jev1 row of
the mapped family, same qtype where one exists) this prints both renderings' skeletons and the token-level diff.

    set CUDA_VISIBLE_DEVICES= & python -X utf8 scripts/check_jev1_vs_jevkit_render.py --out report.json
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))

# jev1 sub-family prefix -> JevBench family (data/jev1.py FAMILIES)
SUB2FAM = [("jev1_model_routing", "routing"), ("jev1_ticket_routing", "routing"), ("jev1_hs2_", "adequacy"),
           ("jev1_agent_audit", "adequacy"), ("jev1_aegis", "policy"), ("jev1_access_policy", "policy"),
           ("jev1_banking77", "intent"), ("jev1_clinc150", "intent"), ("jev1_massive_intent", "intent"),
           ("jev1_massive_scenario", "intent"), ("jev1_sev_", "severity"), ("jev1_sst5", "severity"),
           ("jev1_notice_", "enum"), ("jev1_massive_slot", "enum"), ("jev1_lp_", "long_policy"),
           ("jev1_events_", "state_tracking"), ("jev1_woz_", "state_tracking")]


_BR = ROOT / "receipts/2026-10-10-jev1-draw/bschema-draw-jev1/build-report.json"
_EXACT = {}
if _BR.exists():  # the build's own sub-family -> family map wins over the prefix table
    for _f, _v in json.loads(_BR.read_text(encoding="utf-8")).get("families", {}).items():
        for _s in (_v.get("subfamilies") or {}):
            _EXACT[f"jev1_{_s}"] = _f


def fam_of(sub: str) -> str:
    return _EXACT.get(sub) or next((f for p, f in SUB2FAM if sub.startswith(p)), "?")


def skeleton(text: str) -> list[str]:
    """Structure lines: content masked, keeping the line's lead token (``record:``, ``A) ``, ``Answer:``...)."""
    out = []
    for ln in text.split("\n"):
        m = re.match(r"^(<\|[^|]+\|>\w*|[A-Za-z_ ]{1,24}:|[A-Z]{1,2}\) |[-*] |\s+)", ln)
        out.append((m.group(1) if m else (ln[:1] if ln else "")) + ("..." if ln and (not m or len(ln) > len(m.group(1))) else ""))
    return out


def render(row, tok) -> dict:
    ids = list(row.ids)
    text = tok.decode(ids)
    return {"T": len(ids), "qtype": row.qtype, "K": row.n_options, "option_ids": list(row.option_ids)[:12],
            "continuations": [tok.decode(list(c)) for c in row.continuations][:12],
            "head": tok.decode(ids[:40]), "tail": tok.decode(ids[-24:]), "skeleton": skeleton(text), "_ids": ids,
            "_text": text}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(DATA / "m2/build/bschema-draw-jev1/records.jsonl.gz"))
    ap.add_argument("--cell", default=str(ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_jev1_q35_4b.json"))
    ap.add_argument("--max-scan", type=int, default=400000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS
    from heads.layout import build_row, hf_segment_encoder
    from jevkit.items import load_public
    from jevkit.systemone import item_request
    from train.data import items_from_records
    from train.run_cell import to_train_record

    cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    tpl, lay = cfg["template"], cfg.get("layout", "ordered")
    tdir = DEFAULT_TOKENIZERS / BACKBONES["qwen3.5-4b"].tokenizer_dir
    tok = AutoTokenizer.from_pretrained(str(tdir))
    enc = hf_segment_encoder(tok)
    items = load_public()
    by_fam: dict = {}
    for it in items:
        by_fam.setdefault(it.family, []).append(it)
    jev_fams = {f for _, f in SUB2FAM}
    train_pick: dict = {}  # (fam, qtype) -> raw record
    state_types: dict = {}
    with gzip.open(a.records, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n >= a.max_scan:
                break
            r = json.loads(line)
            fam = ((r.get("aug") or {}).get("jev1") or {}).get("family8") or fam_of(r.get("family", "?"))
            qt = r["questions"]["q"].get("type", "choice")
            state_types.setdefault(fam, {}).setdefault(type(r["state"]).__name__, 0)
            state_types[fam][type(r["state"]).__name__] += 1
            if (fam, qt) not in train_pick:
                train_pick[(fam, qt)] = r
    rep = {"template": tpl, "layout": lay, "eval_families": {f: len(v) for f, v in sorted(by_fam.items())},
           "covered": sorted(set(by_fam) & jev_fams), "uncovered": sorted(set(by_fam) - jev_fams),
           "train_state_types": state_types, "families": {}, "findings": []}
    for fam in sorted(by_fam):
        fam_items = by_fam[fam]
        ev_types = sorted({i.question.get("type", "choice") for i in fam_items})
        ev_state = sorted({type(i.state).__name__ for i in fam_items})
        entry = {"n_items": len(fam_items), "eval_qtypes": ev_types, "eval_state_types": ev_state, "pairs": []}
        for qt in ev_types:
            it = next(i for i in fam_items if i.question.get("type", "choice") == qt)
            try:
                erow = build_row(item_request(it), "q", enc, template=tpl, layout=lay)
                ev = render(erow, tok)
            except Exception as e:  # noqa: BLE001
                ev = {"error": f"{type(e).__name__}: {e}"}
            raw = train_pick.get((fam, qt))
            pair = {"qtype": qt, "eval_id": it.id, "eval_noul_keys": sorted(k for k in it.question if k not in
                    ("instructions", "criteria", "type")), "eval": ev}
            if raw is None:
                pair["train"] = None
                if fam in jev_fams:
                    rep["findings"].append(f"{fam}/{qt}: eval has this qtype, jev1 has NO row of it")
            else:
                trow = items_from_records([to_train_record(raw)], enc, layout=lay, template=tpl)[0].row
                tr = render(trow, tok)
                pair["train_id"] = raw["id"]
                pair["train_noul_keys"] = sorted(k for k in raw["questions"]["q"] if k not in
                                                 ("instructions", "criteria", "type"))
                pair["train"] = tr
                if "error" not in ev:
                    e_ids, t_ids = ev["_ids"], tr["_ids"]
                    cp = next((k for k in range(min(len(e_ids), len(t_ids))) if e_ids[k] != t_ids[k]),
                              min(len(e_ids), len(t_ids)))
                    cs = next((k for k in range(min(len(e_ids), len(t_ids))) if e_ids[-1 - k] != t_ids[-1 - k]),
                              min(len(e_ids), len(t_ids)))
                    sk = [d for d in difflib.unified_diff(tr["skeleton"], ev["skeleton"], "train", "eval", n=0,
                                                          lineterm="") if d[:1] in "+-" and d[:3] not in ("+++", "---")]
                    pair["diff"] = {"common_prefix_tokens": cp, "common_prefix_text_end": tok.decode(e_ids[max(0, cp - 8):cp]),
                                    "common_suffix_tokens": cs, "suffix_text": tok.decode(e_ids[len(e_ids) - cs:]),
                                    "skeleton_diff": sk[:40],
                                    "continuation_form_equal": [len(c) for c in erow.continuations][:1] ==
                                    [len(c) for c in trow.continuations][:1]}
                    if cs < 6:
                        rep["findings"].append(f"{fam}/{qt}: answer-slot suffix differs (common suffix {cs} tokens)")
                    if type(it.state).__name__ != type(raw["state"]).__name__:
                        rep["findings"].append(f"{fam}/{qt}: state type eval {type(it.state).__name__} vs train "
                                               f"{type(raw['state']).__name__}")
                    if pair["eval_noul_keys"] != pair["train_noul_keys"]:
                        rep["findings"].append(f"{fam}/{qt}: question keys eval {pair['eval_noul_keys']} vs train "
                                               f"{pair['train_noul_keys']}")
                    if ev["continuations"][:2] != tr["continuations"][:2] and qt != "choice":
                        rep["findings"].append(f"{fam}/{qt}: option continuations eval {ev['continuations'][:3]} vs "
                                               f"train {tr['continuations'][:3]}")
            for side in ("eval", "train"):
                if isinstance(pair.get(side), dict):
                    pair[side].pop("_ids", None)
                    pair[side].pop("_text", None)
            entry["pairs"].append(pair)
        rep["families"][fam] = entry
    s = json.dumps(rep, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes((s + "\n").encode("utf-8"))
    print(json.dumps({"findings": rep["findings"], "covered": rep["covered"], "uncovered": rep["uncovered"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
