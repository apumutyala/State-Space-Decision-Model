"""Fill the Q9-AF v2 conversion pins of mode q9af_v2_v3w from ANY q9af_v2 conversion job's artifact (run at launch,
commit alone): job-w3c2e's export or a later rerun's (e.g. a full-eval-v1 decontam rerun). Nothing in the mode names a
conversion job; this script records which one, and under which decontamination, in the cell's pins.

    python -X utf8 scripts/fill_conversion_pins.py --export-json <export.json> --gdn <gdn_converted.pt> \
        --decision <decision.json> --job <conversion job id> --conversion-config <the config JSON that job ran> \
        [--artifact art-xxxxx] [--commit <conversion branch sha>] [--dry]

``--conversion-config`` is the jobs/q9af_v2 config the conversion job ran (its job_params ``config``, e.g.
configs/q9af_v2_h100.json at the job's sha): its ``mix.decontam`` setting is recorded verbatim (null when the config has
no such key: the original DEV-only 13-gram filter) with the config's sha256. The pinned block ``q9af.conversion`` =
{job, artifact, commit, mix_decontam, config_sha256, decision_sha256, verdict} goes into the backbone provenance
(heads.backbone.q9af_pin_check), hence the training lineage / backbone identity and the parent-arm identity
(train.m2_gates.expected_parent_identity): two conversions can never be confused. The mode ships
``q9af.conversion: null``, which q9af_pin_check refuses (unpinned conversion).

Refuses (exit 10, nothing written) unless:
  * decision.json is a valid q9af-v2-rule-v1 verdict (jobs.q9af_v2.rule.valid_decision) and the verdict is STRONG or
    ADVANCE (the rule's actions for MARGINAL / FOLD / FOLD_KL / NO_MEASUREMENT are not a cell launch);
  * export.json is ok, mode final, attention_free, converted == the rule's ATTENTION_LAYERS [3, 7, ..., 31] and its
    sha256 field == the .pt file's sha256; the .pt holds exactly those layers (state dicts, weights_only load);
  * jobs.q9af_cell.hooks.check_export accepts the filled q9af block (no hybrid).
Then writes, in the cell config jobs/m2_first_cell/m2_first_cell_rwkv7_q9af_v2_v3w.json, q9af.converted_sha256,
q9af.export_json_sha256, q9af.conversion and q9af.converted_from (job, artifact, decontam, verdict, numbers); the job id
into job_params_q9af_v2_v3w.json's model and submit label (its JOB_SLOT placeholder); and the context data pin
jobs/m2_first_cell/gdn_q9af_v2.sha256 ("<sha>  gdn/gdn_converted.pt", "<sha>  gdn/export.json"). Also reports whether the
doc-overlap gate can run on the box (data/manifests/doc_overlap/index-pin.json committed; the B1 100k-v2 draw needs a
train_set.doc_clean pin if its scan fails) -- a warning, not a refusal (the heldout phase fails closed, exit 34).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CELL = ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_q9af_v2_v3w.json"
PARAMS = ROOT / "jobs/m2_first_cell/job_params_q9af_v2_v3w.json"
JOB_SLOT = "<conversion job: q9af.conversion>"  # the job_params placeholder the job id replaces
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


def conversion_block(job: str, config: Path, decision: Path, dec: dict, artifact: str | None,
                     commit: str | None) -> dict:
    """The conversion's identity: which job, and the decontamination its training mix ran under."""
    from jobs.q9af_cell.hooks import sha256

    c = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(c.get("mix"), dict) or "train_tokens" not in c["mix"]:
        raise ValueError(f"{config.name} is not a jobs/q9af_v2 config (no mix.train_tokens)")
    return {"job": job, "artifact": artifact, "commit": commit, "mix_decontam": c["mix"].get("decontam"),
            "config_sha256": sha256(config), "decision_sha256": sha256(decision), "verdict": dec["verdict"]}


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
    ap.add_argument("--job", required=True, help="the conversion job id (job-w3c2e, or a rerun's)")
    ap.add_argument("--conversion-config", required=True, type=Path,
                    help="the jobs/q9af_v2 config JSON that conversion job ran (its mix.decontam is recorded)")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--commit", default=None, help="the conversion job's branch commit")
    ap.add_argument("--cell", type=Path, default=CELL)
    ap.add_argument("--params", type=Path, default=PARAMS)
    ap.add_argument("--data-pin", type=Path, default=DATA_PIN)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    from jobs.q9af_cell.hooks import check_export, sha256

    try:
        dec, meta = check(a.export_json, a.gdn, a.decision)
        if not re.fullmatch(r"job-[0-9a-z]+", a.job):
            raise ValueError(f"--job {a.job!r} is not a job id (job-xxxxx)")
        conv = conversion_block(a.job, a.conversion_config, a.decision, dec, a.artifact, a.commit)
        cfg = json.loads(a.cell.read_text(encoding="utf-8"))
        pins = {"converted_sha256": sha256(a.gdn), "export_json_sha256": sha256(a.export_json), "conversion": conv,
                "converted_from": (f"{a.job}{' ' + a.artifact if a.artifact else ''} {meta.get('file')} (jobs/q9af_v2 "
                                   f"export, mode {meta['mode']}, converted {meta['converted']}, mix.decontam "
                                   f"{conv['mix_decontam']!r}); q9af-v2-rule-v1 "
                                   f"{dec['verdict']}: proxy600 diff {dec['numbers'].get('proxy_diff_point')} "
                                   f"[lo {dec['numbers'].get('proxy_diff_lo')}]"),
                "hybrid": False}
        cfg["q9af"] = {**cfg["q9af"], **pins}
        label = check_export(cfg, a.export_json, a.gdn)
        if not label["attention_free"]:
            raise ValueError("check_export: not attention-free")
        params_text = a.params.read_text(encoding="utf-8") if a.params.exists() else None
        if params_text is not None and JOB_SLOT not in params_text:
            raise ValueError(f"{a.params.name} has no {JOB_SLOT!r} slot (already filled? the pins are filled once)")
    except (ValueError, OSError, KeyError) as e:
        print(f"fill_conversion_pins: {e}", file=sys.stderr)
        return 10
    out = {"pins": pins, "label": label, "verdict": dec["verdict"], "doc_overlap": doc_overlap_status(cfg),
           "dry": a.dry}
    if not a.dry:
        a.cell.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
        if params_text is not None:
            a.params.write_text(params_text.replace(JOB_SLOT, a.job), encoding="utf-8", newline="")
        a.data_pin.write_text(f"{pins['converted_sha256']}  gdn/gdn_converted.pt\n"
                              f"{pins['export_json_sha256']}  gdn/export.json\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
