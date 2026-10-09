"""Fill the Q9-AF v2 conversion pins of mode q9af_v2_v3w from job-w3c2e's artifact (run at launch, commit alone).

    python -X utf8 scripts/fill_conversion_pins.py --export-json <export.json> --gdn <gdn_converted.pt> \
        --decision <decision.json> --job job-w3c2e [--artifact art-xxxxx] [--dry]

Refuses (exit 10, nothing written) unless:
  * decision.json is a valid q9af-v2-rule-v1 verdict (jobs.q9af_v2.rule.valid_decision) and the verdict is STRONG or
    ADVANCE (the rule's actions for MARGINAL / FOLD / FOLD_KL / NO_MEASUREMENT are not a cell launch);
  * export.json is ok, mode final, attention_free, converted == the rule's ATTENTION_LAYERS [3, 7, ..., 31] and its
    sha256 field == the .pt file's sha256; the .pt holds exactly those layers (state dicts, weights_only load);
  * jobs.q9af_cell.hooks.check_export accepts the filled q9af block (no hybrid).
Then writes, in the cell config jobs/m2_first_cell/m2_first_cell_rwkv7_q9af_v2_v3w.json, q9af.converted_sha256,
q9af.export_json_sha256 and q9af.converted_from (job, artifact, verdict, numbers), and the context data pin
jobs/m2_first_cell/gdn_q9af_v2.sha256 ("<sha>  gdn/gdn_converted.pt", "<sha>  gdn/export.json"). Also reports whether the
doc-overlap gate can run on the box (data/manifests/doc_overlap/index-pin.json committed; the B1 100k-v2 draw needs a
train_set.doc_clean pin if its scan fails) -- a warning, not a refusal (the heldout phase fails closed, exit 34).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CELL = ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_q9af_v2_v3w.json"
DATA_PIN = ROOT / "jobs/m2_first_cell/gdn_q9af_v2.sha256"
LAUNCH_VERDICTS = ("STRONG", "ADVANCE")


def check(export_json: Path, gdn: Path, decision: Path) -> tuple[dict, dict]:
    from jobs.q9af_cell.hooks import sha256
    from jobs.q9af_v2.rule import ATTENTION_LAYERS, valid_decision

    dec = json.loads(decision.read_text(encoding="utf-8"))
    if not valid_decision(dec):
        raise ValueError("decision.json is not a valid q9af-v2-rule-v1 verdict")
    if dec["verdict"] not in LAUNCH_VERDICTS:
        raise ValueError(f"verdict {dec['verdict']}: the rule's action is not a cell launch ({dec.get('action')})")
    meta = json.loads(export_json.read_text(encoding="utf-8"))
    want = {"ok": True, "mode": "final", "attention_free": True, "converted": ATTENTION_LAYERS}
    bad = {k: meta.get(k) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"export.json is not the full attention-free export: {bad}")
    got = sha256(gdn)
    if meta.get("sha256") != got:
        raise ValueError(f"export.json sha256 {meta.get('sha256')} != {gdn.name} sha256 {got}")
    import torch

    sd = torch.load(gdn, map_location="cpu", weights_only=True)
    if sorted(int(k) for k in sd) != ATTENTION_LAYERS:
        raise ValueError(f"{gdn.name} layers {sorted(sd)} != {ATTENTION_LAYERS}")
    return dec, meta


def doc_overlap_status(cfg: dict) -> dict:
    pin = ROOT / "data/manifests/doc_overlap/index-pin.json"
    return {"index_pin_committed": pin.exists(), "doc_clean_pinned": bool((cfg.get("train_set") or {}).get("doc_clean")),
            "note": "the heldout phase runs train.doc_overlap_guard on the whole B1 100k-v2 set: it needs the index pin "
                    "(+ the shipped index or the suite rows) and, if the scan fails, a train_set.doc_clean pin "
                    "(python -X utf8 -m train.doc_overlap_guard clean-spec --records ...)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-json", required=True, type=Path)
    ap.add_argument("--gdn", required=True, type=Path)
    ap.add_argument("--decision", required=True, type=Path)
    ap.add_argument("--job", default="job-w3c2e")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--cell", type=Path, default=CELL)
    ap.add_argument("--data-pin", type=Path, default=DATA_PIN)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    from jobs.q9af_cell.hooks import check_export, sha256

    try:
        dec, meta = check(a.export_json, a.gdn, a.decision)
        cfg = json.loads(a.cell.read_text(encoding="utf-8"))
        pins = {"converted_sha256": sha256(a.gdn), "export_json_sha256": sha256(a.export_json),
                "converted_from": (f"{a.job}{' ' + a.artifact if a.artifact else ''} {meta.get('file')} (jobs/q9af_v2 "
                                   f"export, mode {meta['mode']}, converted {meta['converted']}); q9af-v2-rule-v1 "
                                   f"{dec['verdict']}: proxy600 diff {dec['numbers'].get('proxy_diff_point')} "
                                   f"[lo {dec['numbers'].get('proxy_diff_lo')}]"),
                "hybrid": False}
        cfg["q9af"] = {**cfg["q9af"], **pins}
        label = check_export(cfg, a.export_json, a.gdn)
        if not label["attention_free"]:
            raise ValueError("check_export: not attention-free")
    except (ValueError, OSError, KeyError) as e:
        print(f"fill_conversion_pins: {e}", file=sys.stderr)
        return 10
    out = {"pins": pins, "label": label, "verdict": dec["verdict"], "doc_overlap": doc_overlap_status(cfg),
           "dry": a.dry}
    if not a.dry:
        a.cell.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
        a.data_pin.write_text(f"{pins['converted_sha256']}  gdn/gdn_converted.pt\n"
                              f"{pins['export_json_sha256']}  gdn/export.json\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
