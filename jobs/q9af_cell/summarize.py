"""Q9-AF cell report + the pre-registered verdict (jobs/q9af_cell/rule.py).

    python -X utf8 -m jobs.q9af_cell.summarize cell --cell <cell.json> --train-rc <rc> --gates-report <report.json> \
        --gates-rc <rc> --child-units <units-child-g13.json> --parent-arm <dir> --index-rows <dev-rows.jsonl.gz> \
        --index-frozen <frozen.json> --index-frozen-sha256 <sha> --subset jobs/common/dev_subset_m1.json \
        --label <export label.json> --out-dir <dir> [--B 2000]
    python -X utf8 -m jobs.q9af_cell.summarize pair --r64 <dir>/report.json --r256 <dir>/report.json --out <pair.json>

Index numbers, child and its own parent arm (the untrained Q9-AF stock readout, the SAME frozen full-DEV rows):
* slices: the FULL DEV slice (frozen rows, ``jobs.common.dev_full``) and the 2,008-row DEV subset
  (``jobs/common/dev_subset_m1.json``) restricted to the same frozen rows;
* editions: 0.2.1 (BINDING, rule.EDITION_BINDING) and 0.3 public DEV without GSM8K twins (REPORTED; rule.EDITION_REPORTED);
* per arm: index CI, areas; paired child - parent (cluster bootstrap, ``harness.bootstrap.compare``, B, seed 0);
* family columns (``harness.families``): in_domain, new_domain (= new_domain_proxy) and new_domain_frozen, per arm and
  paired;
* references (never re-measured here): Clef-flash 55.78 subset / 56.79 full DEV (0.2.1), 57.35 full DEV at 32k
  (job-ftpnf), stock Q9 38.27 subset (rule constants).
``report.json`` holds everything; ``q9af_cell_result.json`` is the compact (< 31 KiB, LESSONS L-27) validated record
``finalize.py`` stamps into the platform result.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

from jobs.q9af_cell import rule as R

RESULT_SCHEMA = "ssdm.q9af_cell.result.v1"
RESULT_MAX_BYTES = 31 * 1024
COLUMNS = {"in_domain": "in_domain", "new_domain": "new_domain_proxy", "new_domain_frozen": "new_domain_frozen"}


def _ci(d: dict | None) -> dict | None:
    return None if d is None else {k: round(float(d[k]), 3) for k in ("point", "lo", "hi") if k in d}


def index_block(rows: list[dict], child: dict, parent: dict, edition: str, B: int, seed: int = 0) -> dict[str, Any]:
    """Child, parent and paired child - parent on ``rows`` under ``edition`` (+ the family columns)."""
    from harness import bootstrap as HB
    from harness import families as HF
    from harness import score as S

    c, p = S.decompose(rows, child, edition=edition), S.decompose(rows, parent, edition=edition)
    cmp = HB.compare(c, p, B=B, seed=seed)
    crep, prep = HB.replicate_skills([c, p], B=B, seed=seed)  # the same replicates compare() draws (same B, seed)
    fc, fp, fd = HF.summarize(crep, edition), HF.summarize(prep, edition), HF.compare(crep, prep, edition)
    cols = {name: {"child": _ci(fc[g]), "parent": _ci(fp[g]), "diff": _ci(fd[g]),
                   "benchmarks": fc[g].get("benchmarks")} for name, g in COLUMNS.items()}
    return {"edition": edition, "rows": len(rows), "B": B, "seed": seed,
            "child": {"index": _ci(cmp["candidate"]["index"]),
                      "areas": {a: _ci(v) for a, v in cmp["candidate"]["areas"].items()}},
            "parent": {"index": _ci(cmp["baseline"]["index"]),
                       "areas": {a: _ci(v) for a, v in cmp["baseline"]["areas"].items()}},
            "diff": {"index": {**_ci(cmp["diff"]["index"]), "excludes_zero": cmp["diff"]["index"]["excludes_zero"]},
                     "areas": {a: _ci(v) for a, v in cmp["diff"]["areas"].items()}},
            "columns": cols}


def _rows(path, frozen_path, frozen_sha) -> list[dict]:
    from harness.suite import read_rows
    from jobs.common.dev_full import load_frozen, select_rows

    return select_rows(read_rows(path), load_frozen(frozen_path, frozen_sha))


def train_summary(cell: dict) -> dict:
    tr = cell.get("train") or {}
    v = tr.get("verdict") or {}
    return {"steps": tr.get("steps"), "T_max": (cell.get("ladder") or {}).get("T_max"),
            "stopped": tr.get("stopped"), "pass": v.get("pass"), "checks": v.get("checks"),
            "loss_first": v.get("loss_first"), "loss_last": v.get("loss_last"), "peak_gb": v.get("peak_gb"),
            "ladder": {T: {k: x.get(k) for k in ("ok", "peak_gb", "step_s", "row_len", "k")}
                       for T, x in ((cell.get("ladder") or {}).get("points") or {}).items()},
            "rate_check": cell.get("train_rate_check"), "smoke": {k: (cell.get("smoke") or {}).get(k)
                                                                   for k in ("rel_feature_diff", "grad_cos", "pass")},
            "data": {k: (cell.get("data") or {}).get(k) for k in ("n_train_records", "n_train_items", "n_eval_items")}}


def build(a) -> tuple[dict, dict]:
    cell = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    cfg = cell["config"]
    label = json.loads(Path(a.label).read_text(encoding="utf-8"))
    gates = json.loads(Path(a.gates_report).read_text(encoding="utf-8")) if a.gates_report and Path(
        a.gates_report).exists() else None
    report: dict[str, Any] = {"schema": RESULT_SCHEMA, "rule": R.RULE_ID, "variant": cfg.get("variant"),
                              "lora_rank": cfg["lora_rank"], "label": label, "train_rc": a.train_rc,
                              "gates_rc": a.gates_rc, "train": train_summary(cell),
                              "references": {"clef_flash_subset_021": R.CLEF_FLASH_SUBSET_021,
                                             "clef_flash_full_dev_021": R.CLEF_FLASH_FULL_DEV_021,
                                             "clef_flash_full_dev_32k": R.CLEF_FLASH_FULL_DEV_32K,
                                             "stock_q9_subset_021": R.STOCK_Q9_SUBSET_021,
                                             "note": "references from receipts (job-sgihj, job-ftpnf, INDEX_0.3.md); "
                                                     "not re-measured here; 57.35 = full DEV with the 51 over-length "
                                                     "rows re-run at 32k"},
                              "index": {}}
    if gates is not None:
        pv = gates.get("promotion") or {}
        report["gates"] = {"verdict": pv.get("verdict"), "promotable": pv.get("promotable"),
                           "stages": sorted((gates.get("verdicts") or {}).keys()),
                           "failed": {s: v.get("failed") for s, v in (gates.get("verdicts") or {}).items()},
                           "margins_status": (gates.get("margins") or {}).get("status") if isinstance(
                               gates.get("margins"), dict) else None}
    if a.child_units and Path(a.child_units).exists():
        child = json.loads(Path(a.child_units).read_text(encoding="utf-8"))["index_results"]
        parent = json.loads((Path(a.parent_arm) / "units-g13.json").read_text(encoding="utf-8"))["index_results"]
        if a.dry_run:  # synthetic kit-format rows of jobs.q9af_cell.dry_gates: no frozen DEV manifest, no subset ids
            from harness.suite import read_rows

            rows = read_rows(a.index_rows)
            sub_ids = {r["_evaluation"]["run_id"] for r in rows[: max(2, len(rows) // 2)]}
            report["dry_run"] = "synthetic rows; 'subset_2008' = the first half of them (NOT the 2,008-row subset)"
        else:
            rows = _rows(a.index_rows, a.index_frozen, a.index_frozen_sha256)
            sub_ids = set(json.loads(Path(a.subset).read_text(encoding="utf-8"))["run_ids"])
        sub = [r for r in rows if r["_evaluation"]["run_id"] in sub_ids]
        for ed in (R.EDITION_BINDING, R.EDITION_REPORTED):
            report["index"][ed] = {"full_dev": index_block(rows, child, parent, ed, a.B),
                                   "subset_2008": index_block(sub, child, parent, ed, a.B)}
        report["index"]["subset_rows_present"] = {"n": len(sub), "of": len(sub_ids)}
    full = ((report["index"].get(R.EDITION_BINDING) or {}).get("full_dev")) or {}
    m = {"export_pinned": bool(label.get("sha256")), "attention_free": bool(label.get("attention_free")),
         "train_rc": a.train_rc, "train_pass": bool(report["train"].get("pass")), "gates_rc": a.gates_rc,
         "gates_stages": (report.get("gates") or {}).get("stages") or [],
         "full_dev_child": ((full.get("child") or {}).get("index") or {}).get("point"),
         "full_dev_gain": (full.get("diff") or {}).get("index")}
    report["rule_inputs"] = m
    report["verdict"] = R.decide(m)
    return report, compact(report)


def compact(rep: dict) -> dict:
    """The platform result: headline numbers only (L-27: the full report is report.json)."""
    out = {k: rep.get(k) for k in ("schema", "rule", "variant", "lora_rank", "train_rc", "gates_rc", "references")}
    out["label"] = {k: rep["label"].get(k) for k in ("attention_free", "label", "sha256")}
    out["verdict"] = rep["verdict"]["verdict"]
    out["why"] = rep["verdict"].get("why")
    out["gates"] = rep.get("gates")
    out["train"] = {k: rep["train"].get(k) for k in ("steps", "T_max", "pass", "peak_gb", "loss_first", "loss_last")}
    idx = {}
    for ed in (R.EDITION_BINDING, R.EDITION_REPORTED):
        for sl in ("full_dev", "subset_2008"):
            b = ((rep["index"].get(ed) or {}).get(sl)) or {}
            if b:
                idx[f"{ed}/{sl}"] = {"child": b["child"]["index"], "parent": b["parent"]["index"],
                                     "diff": b["diff"]["index"],
                                     "columns": {c: {"child": v["child"], "diff": v["diff"]}
                                                 for c, v in b["columns"].items()}}
    out["index"] = idx
    out["full_report"] = "report.json"
    if rep.get("dry_run"):
        out["dry_run"] = rep["dry_run"]
    if len(json.dumps(out)) > RESULT_MAX_BYTES:  # hard fallback: drop the column detail
        for v in out["index"].values():
            v.pop("columns", None)
    return out


def valid_result(r: Any) -> bool:
    """``jobs/common/finalize.py`` validator: a rule verdict over a schema-complete compact result."""
    try:
        return (r["schema"] == RESULT_SCHEMA and r["rule"] == R.RULE_ID and r["verdict"] in R.VERDICTS
                and isinstance(r["label"]["attention_free"], bool) and len(json.dumps(r)) <= RESULT_MAX_BYTES
                and (r["verdict"] in ("NO_MEASUREMENT", "THROUGHPUT")
                     or all(math.isfinite(r["index"][f"{R.EDITION_BINDING}/full_dev"][k]["point"])
                            for k in ("child", "parent", "diff"))))
    except Exception:  # noqa: BLE001
        return False


def valid_parent(r: Any) -> bool:
    """finalize validator of the parent mode: every pin the cell params need is present."""
    try:
        p = r["pins_for_cell_params"]
        return (r["schema"] == RESULT_SCHEMA and r["variant"] == "parent" and isinstance(r["label"]["attention_free"], bool)
                and len(str(p["parent_arm_sha256"])) == 64
                and all(len(str(v)) == 64 for v in p["parent_arm_files"].values() if v is not None)
                and p["parent_arm_files"].get("units-g13.json") is not None)
    except Exception:  # noqa: BLE001
        return False


def export_trainable(ckpt_dir: str | Path, out_dir: str | Path) -> dict:
    """The trained LoRA + head weights of the latest checkpoint (no optimiser state: r256 fp32 adapters ~2.5 GB fit the
    5 GiB artifact), with the step, config hash and lineage needed to re-attach them; verified by re-reading."""
    import hashlib

    import torch

    from train.trainer import Trainer

    p = Trainer.latest(ckpt_dir)
    if p is None:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
    st = torch.load(p, map_location="cpu", weights_only=False)
    keep = {"trainable": st["trainable"], "step": st["step"], "config_hash": st["config_hash"],
            "lineage": st.get("lineage"), "source": p.name}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "adapters.pt"
    tmp = f.with_suffix(".pt.tmp")
    torch.save(keep, tmp)
    tmp.replace(f)
    back = torch.load(f, map_location="cpu", weights_only=False)
    if sorted(back["trainable"]) != sorted(st["trainable"]) or not all(
            torch.equal(back["trainable"][k], st["trainable"][k]) for k in st["trainable"]):
        raise RuntimeError("re-loaded adapters differ from the checkpoint")
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    (out / "adapters.pt.sha256").write_text(f"{h}  adapters.pt" + chr(10), encoding="utf-8", newline=chr(10))
    return {"file": "adapters.pt", "sha256": h, "bytes": f.stat().st_size, "step": st["step"],
            "n_tensors": len(st["trainable"])}


def pair(r64: dict, r256: dict) -> dict:
    """E9 off-box: R.compare_ranks over the two variants' reports. The paired r256 - r64 difference needs both
    children's per-row results on the same rows; it is computed by ``pair_diff`` from their units files."""
    return {"rule": R.RULE_ID, "r64": r64["verdict"], "r256": r256["verdict"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.q9af_cell.summarize")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--cell", required=True)
    c.add_argument("--train-rc", type=int, required=True)
    c.add_argument("--gates-report", default=None)
    c.add_argument("--gates-rc", type=int, default=None)
    c.add_argument("--child-units", default=None)
    c.add_argument("--parent-arm", default=None)
    c.add_argument("--index-rows", default=None)
    c.add_argument("--index-frozen", default=None)
    c.add_argument("--index-frozen-sha256", default=None)
    c.add_argument("--subset", default="jobs/common/dev_subset_m1.json")
    c.add_argument("--label", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--B", type=int, default=2000)
    c.add_argument("--dry-run", action="store_true", help="CPU dry run: synthetic rows (jobs.q9af_cell.dry_gates)")
    e = sub.add_parser("export")
    e.add_argument("--ckpt-dir", required=True)
    e.add_argument("--out", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--r64-dir", required=True, help="cell r64 output dir (report.json, units-child-g13.json)")
    p.add_argument("--r256-dir", required=True)
    p.add_argument("--index-rows", required=True)
    p.add_argument("--index-frozen", required=True)
    p.add_argument("--index-frozen-sha256", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "cell":
        rep, res = build(a)
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
        (out / "q9af_cell_result.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(json.dumps({"verdict": res["verdict"], "why": res["why"]}))
        return 0 if valid_result(res) else 9
    if a.cmd == "export":
        print(json.dumps(export_trainable(a.ckpt_dir, a.out)))
        return 0
    from harness import bootstrap as HB
    from harness import score as S

    reps = {k: json.loads((Path(d) / "report.json").read_text(encoding="utf-8")) for k, d in
            (("r64", a.r64_dir), ("r256", a.r256_dir))}
    units = {k: json.loads((Path(d) / "units-child-g13.json").read_text(encoding="utf-8"))["index_results"]
             for k, d in (("r64", a.r64_dir), ("r256", a.r256_dir))}
    rows = _rows(a.index_rows, a.index_frozen, a.index_frozen_sha256)
    cmp = HB.compare(S.decompose(rows, units["r256"], edition=R.EDITION_BINDING),
                     S.decompose(rows, units["r64"], edition=R.EDITION_BINDING), B=a.B, seed=0)
    d = _ci(cmp["diff"]["index"])
    out = {**pair(reps["r64"], reps["r256"]), "paired_r256_minus_r64_full_dev_021": d,
           "rank_verdict": R.compare_ranks(reps["r64"]["verdict"], reps["r256"]["verdict"], d)}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["rank_verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
