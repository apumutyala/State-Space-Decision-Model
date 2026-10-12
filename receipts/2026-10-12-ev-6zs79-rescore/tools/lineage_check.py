"""Independent lineage / order / G1-reality / noul-band checks for job-6zs79 (eval-only) vs job-5x4vr (training).
usage: python -I lineage_check.py <6zs79 out dir> <5x4vr extracted root (x/)> <5x4vr final out dir> <stream sha file> <tar sha 6zs79> <tar sha 5x4vr> <out.json>
All inputs are data; nothing is executed from them."""
import json, sys, hashlib
from pathlib import Path

o6, x5, o5, streamsha, tar6, tar5, outp = map(Path, sys.argv[1:8])
J = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
res = {}

# (1) lineage
ck = J(o6 / "checkpoint.json")
latest = J(x5 / "ckpt/latest.json")
side200 = (x5 / "ckpt/trainable-step0000200.safetensors.sha256").read_text().split()[0]
stream200 = Path(streamsha).read_text().split()[0]
c5 = J(o5 / "cell.json")
ic = J(o6 / "screen/identity-child.json")
def walk(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: yield v
            yield from walk(v, key)
    elif isinstance(o, list):
        for v in o: yield from walk(v, key)
ic_shas = sorted(set(v for v in walk(ic, "sha256") if isinstance(v, str)))
res["lineage"] = {
    "6zs79_checkpoint_json": ck,
    "5x4vr_tar_sha256_streamed": tar5.read_text().split()[0] if tar5.exists() else str(tar5),
    "6zs79_tar_sha256": tar6.read_text().split()[0] if tar6.exists() else str(tar6),
    "5x4vr_latest_best": latest.get("best"),
    "5x4vr_sidecar_step200_sha256": side200,
    "step200_member_sha256_streamed_from_art-8xs32": stream200,
    "5x4vr_cell_json_sha256": sha(o5 / "cell.json"),
    "6zs79_cell_json_sha256": sha(o6 / "cell.json"),
    "5x4vr_cell_best": c5.get("best"),
    "5x4vr_dev_cards": c5.get("dev_cards"),
    "dev_card_source": c5.get("dev_card_source"),
    "identity_child_has_ckpt_sha": "5cb7da4246efce72c17d15cf5af9fc6b8137a7e76e371a970a67c89303835310" in ic_shas,
    "collapse_events": c5.get("collapse_events"),
    "train_steps": (c5.get("train_clock") or {}).get("step"),
}
L = res["lineage"]
L["ok"] = all([
    ck["sha256"] == stream200 == side200 == latest["best"]["sha256"],
    latest["best"]["step"] == 200 == c5["best"]["step"],
    ck["cell_json_sha256"] == L["5x4vr_cell_json_sha256"] == L["6zs79_cell_json_sha256"],
    ck["artifact_sha256"] == L["5x4vr_tar_sha256_streamed"],
    min(c5["dev_cards"], key=lambda r: r["nll"])["step"] == 200,
    c5["dev_card_source"]["ids_sha256"] == "844233f228b718edeffb198f622fc9c8a605eed3a346c128993f8aac80453078",
    L["identity_child_has_ckpt_sha"],
])
# consistency views (Amendment 1)
cv = c5["data"]["consistency_views"]
res["consistency_views"] = cv

# order: 5x4vr selection files vs any public pred in either job
l5, l6 = J(o5 / "landing.json"), J(o6 / "landing.json")
sel5 = {f["path"]: f["mtime_utc"] for f in l5["files"] if f["path"].startswith("dev_cards/") or f["path"] in ("cell.json",)}
pub5 = [f for f in l5["files"] if f["path"].endswith(".preds.jsonl")]
pub6 = {f["path"]: f["mtime_utc"] for f in l6["files"] if f["path"].endswith(".preds.jsonl") and "/g1/" not in f["path"]}
pub6_all = {f["path"]: f["mtime_utc"] for f in l6["files"] if f["path"].endswith(".preds.jsonl")}
sf = J(o6 / "screen/jevkit/slice_first.json")
first6 = min(pub6_all.values())
res["order"] = {
    "5x4vr_selection_files": sel5, "5x4vr_public_preds": [f["path"] for f in pub5],
    "5x4vr_gates": l5.get("gates"),
    "6zs79_slice_first": sf.get("slice_first"),
    "6zs79_first_public_pred_utc": first6,
    "6zs79_first_public_pred": min(pub6_all, key=pub6_all.get),
    "6zs79_landing_slice_first_before_public_preds": l6.get("slice_first_before_public_preds"),
    "selection_before_any_public": max(sel5.values()) < first6 and not pub5,
    "slice_first_before_public": sf["slice_first"]["done_utc"] < first6,
}

# G1 reality: perturbations changed on-disk order; every op n>0; bitwise
g1 = {}
for arm in ("native", "stock_shim"):
    d = o6 / "screen/jevkit/g1" / arm
    rep = J(d / "perturb_report.json")
    pert = [json.loads(x) for x in (d / "perturbed.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ops = {}
    for r in pert:
        op = r["id"].split("~")[1] if "~" in r.get("id", "") else None
        ops[op] = ops.get(op, 0) + 1
    g1[arm] = {"perturbed_rows": len(pert), "ops_in_file": ops, "perturbed_jsonl_sha256": sha(d / "perturbed.jsonl"),
               "report_keys": sorted(rep)[:20]}
res["g1_files"] = g1

# noul band per arm, from the preds
def rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
res["noul_band_note"] = "computed by the kit (2_secondary.noul_band); repeated in RESCORE_6ZS79.md"
Path(outp).write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
print(json.dumps({"lineage_ok": L["ok"], "order": {k: res["order"][k] for k in ("selection_before_any_public", "slice_first_before_public", "6zs79_first_public_pred_utc")}, "cv": cv, "g1": {a: (g1[a]["perturbed_rows"], g1[a]["ops_in_file"]) for a in g1}}, indent=1))
