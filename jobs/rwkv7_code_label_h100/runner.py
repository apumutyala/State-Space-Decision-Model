"""Production qualification phases. Never print private requests or exception text."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import tarfile
import time

from jobs.rwkv7_code_label_screen import runner as S
from jobs.rwkv7_code_label_screen import rule as R
from jobs.rwkv7_code_label_h100 import preflight as P

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROWS = Path("/data/index-rows.jsonl.gz")


def params():
    return json.loads((HERE / "job_params.json").read_text())


def admit():
    p = params()
    authorization = p["spend_authorization"]
    if (p["launch_enabled"] is not True or not authorization.get("issue")
            or type(authorization.get("approved_max_usd")) not in (float, int)
            or authorization["approved_max_usd"] < p["hard_cap_usd"]):
        raise ValueError("launch disabled until reviewed merged proof and explicit spend-issue authorization")
    manifest = json.loads(Path("/opt/ssdm/CONTEXT_MANIFEST.json").read_text())
    if (manifest.get("builder") != "jobs.common.launch" or manifest.get("job") != "rwkv7_code_label_h100"
            or len(manifest.get("m1_sha", "")) != 40
            or manifest.get("data_sha256", {}).get("data/index-rows.jsonl.gz") != R.ROWS_SHA256
            or manifest.get("files_sha256", {}).get("jobs/rwkv7_code_label_h100/job_params.json") != S.sha(HERE / "job_params.json")):
        raise ValueError("production context identity failed")
    return manifest


def checked_rows(path=ROWS):
    if S.sha(path) != R.ROWS_SHA256:
        raise ValueError("pinned private DEV input hash mismatch")
    subset = S.checked_cohort()
    rows = S.read_jsonl_strict(path)
    if len(rows) != R.ROWS or {r["_evaluation"]["run_id"] for r in rows} != set(subset["run_ids"]):
        raise ValueError("exact DEV cohort required")
    by_id = {r["_evaluation"]["run_id"]: r for r in rows}
    return [by_id[rid] for rid in subset["run_ids"]]


def verify_snapshot(snapshot):
    import third_party.rwkv7_hf as V
    for line in (ROOT / "jobs/rwkv7/snapshot.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if S.sha(Path(snapshot) / name) != digest:
            raise ValueError("production snapshot hash mismatch")
    V.verify_snapshot_code(snapshot)


def setup(out):
    manifest = admit()
    if any((out / name).exists() for name in ("setup.json", "probe.json", "report.json", "receipt.json",
            "v1.predictions.jsonl", "v2.predictions.jsonl", "RWKV7_CODE_LABEL_QUALIFICATION_COMPLETE")):
        raise ValueError("fresh production output directory required; no implicit resume")
    checked_rows()
    # Imported only after the input/context checks. No arbitrary model identifier.
    from huggingface_hub import snapshot_download
    model, revision, _ = S.pinned_model_identity()
    snapshot = snapshot_download(model, revision=revision)
    verify_snapshot(snapshot)
    S.write_json(out / "setup.json", {"snapshot_path": snapshot, "commit_sha": manifest["m1_sha"],
        "context_manifest_sha256": S.sha("/opt/ssdm/CONTEXT_MANIFEST.json"), "model": params()["model"],
        "rows_sha256": R.ROWS_SHA256, "ids_sha256": R.IDS_SHA256, "rows_denominator": R.ROWS})


def production(out):
    import torch
    import transformers
    from decide import Decider
    from decide.backends.hf_readout import load_rwkv7_hf
    from decide.backends.rwkv_g1_code_label import G1CodeLabelBackend
    from decide.backends.rwkv_g1_code_label_v2 import G1CodeLabelV2Backend
    from jobs.common.readout_engine import _load_code_label_decode

    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "H100" not in torch.cuda.get_device_name(0)
            or torch.__version__.split("+")[0] != "2.13.0" or transformers.__version__ != "5.17.0"):
        raise ValueError("pinned single-H100 stack required")
    torch.manual_seed(R.BOOTSTRAP_SEED)
    snapshot = json.loads((out / "setup.json").read_text())["snapshot_path"]
    verify_snapshot(snapshot)
    decode = _load_code_label_decode(snapshot)
    plugin = load_rwkv7_hf(snapshot, device="cuda", dtype="bfloat16", wkv=None)
    if plugin.kernel_stack["wkv_implementation"] != "chunked":
        raise ValueError("unregistered production kernel")
    backends = {"v1": G1CodeLabelBackend(plugin, tokenizer_decode=decode),
                "v2": G1CodeLabelV2Backend(plugin, tokenizer_decode=decode)}
    deciders = {a: Decider(b, enable_v0_1_deltas=True, backend_max_input_tokens=25600) for a, b in backends.items()}
    for arm, decider in deciders.items():
        S.check_revision(decider.revision(), arm)
    # One shared eval model; both arm identities and kernels are therefore matched.
    return torch, plugin, decode, deciders


def peak_check(torch):
    reserved = torch.cuda.max_memory_reserved()
    free, total = torch.cuda.mem_get_info()
    if reserved > P.PEAK_FRACTION * total or free < P.MIN_FREE_BYTES:
        raise ValueError("real-shape memory stop")
    return {"peak_reserved_bytes": reserved, "min_free_bytes": free}


def timed_capture(torch, row, decider):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    rec = S.capture(row, decider)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    memory = peak_check(torch)
    if rec["status"] not in ("ok", "exhausted"):
        raise ValueError("unexpected non-ok model result")
    return rec, elapsed, memory


def parity(torch, plugin, decode, deciders):
    from decide import Decider
    from decide.backends.rwkv_g1_code_label_v2 import G1CodeLabelV2Backend
    # No DEV text: finite full-vocabulary checks happen inside both backends.
    row = {"state": "Synthetic reference state: amber is available, blue is unavailable.",
        "questions": {f"q{i}": {"type": "choice", "instructions": f"Select an available color for slot {i}.",
                                "criteria": {"a": "amber", "b": "blue"}} for i in range(3)},
        "_evaluation": {"run_id": "synthetic:production-parity"}}
    recompute = Decider(G1CodeLabelV2Backend(plugin, tokenizer_decode=decode, fork=False),
                        enable_v0_1_deltas=True, backend_max_input_tokens=25600)
    left, _, _ = timed_capture(torch, row, deciders["v2"])
    right, _, _ = timed_capture(torch, row, recompute)
    if left["status"] != "ok" or right["status"] != "ok":
        raise ValueError("synthetic production parity reference refused")
    a, b = left["raw_output"]["raw_probabilities"], right["raw_output"]["raw_probabilities"]
    delta = max(abs(a[q][k] - b[q][k]) for q in a for k in a[q])
    if delta > P.PARITY_ATOL:
        raise ValueError("fork/recompute production parity stop")
    return delta


def probe(out):
    rows = {r["_evaluation"]["run_id"]: r for r in checked_rows()}
    torch, plugin, decode, deciders = production(out)
    volume = json.loads(P.VOLUME.read_text())
    report = {"synthetic": False, "gpu": torch.cuda.get_device_name(0),
        "total_memory_bytes": torch.cuda.mem_get_info()[1], "samples": [],
        "volume_sha256": S.sha(P.VOLUME), "revisions": {a: d.revision() for a, d in deciders.items()},
        "parity_max_abs": parity(torch, plugin, decode, deciders)}
    for arm in ("v1", "v2"):
        for rid in P.probe_ids(volume):
            deciders[arm].backend._enc_memo.clear()
            warm, _, _ = timed_capture(torch, rows[rid], deciders[arm])
            if warm["status"] != "ok":
                raise ValueError("real-shape warmup reference refused")
            seconds, memories = [], []
            for _ in range(P.TIMED_REPEATS):
                rec, elapsed, memory = timed_capture(torch, rows[rid], deciders[arm])
                if rec["status"] != "ok":
                    raise ValueError("real-shape probe reference refused")
                seconds.append(elapsed)
                memories.append(memory)
            report["samples"].append({"arm": arm, "run_id": rid, "seconds": seconds,
                "peak_reserved_bytes": max(m["peak_reserved_bytes"] for m in memories),
                "min_free_bytes": min(m["min_free_bytes"] for m in memories)})
            S.write_json(out / "probe.json", report)
            (out / "heartbeat").touch()


def rate_check(elapsed, completed_reference, remaining_reference, wall_elapsed, cap_seconds):
    if wall_elapsed > cap_seconds:
        raise ValueError("scoring wall deadline")
    if min(completed_reference.values()) <= 0:
        return
    projected_remaining = sum(elapsed[a] / completed_reference[a] * remaining_reference[a] for a in elapsed)
    if wall_elapsed + projected_remaining > P.RATE_MARGIN * cap_seconds:
        raise ValueError("live measured workload projection stop")


def screen(out):
    rows = checked_rows()
    load_start = time.monotonic()
    torch, _, _, deciders = production(out)
    load_seconds = time.monotonic() - load_start
    if load_seconds > 300:
        raise ValueError("second model load exceeded its five-minute allowance")
    probe = json.loads((out / "probe.json").read_text())
    volume = json.loads(P.VOLUME.read_text())
    guard = P.evaluate(probe, volume, params()["phase_caps_minutes"]["screen"])
    if json.loads((out / "preflight.json").read_text()) != guard:
        raise ValueError("preflight artifact drift")
    revisions = {a: d.revision() for a, d in deciders.items()}
    if revisions != probe["revisions"] or probe["volume_sha256"] != S.sha(P.VOLUME):
        raise ValueError("probe/production identity drift")
    S.write_json(out / "identities.json", {"rows_sha256": R.ROWS_SHA256, "ids_sha256": R.IDS_SHA256,
        "revisions": revisions, "probe_sha256": S.sha(out / "probe.json"), "seed": R.BOOTSTRAP_SEED})
    elapsed = {a: 0.0 for a in deciders}
    completed = dict(elapsed)
    remaining = {a: P.reference_seconds(volume["totals"]["admitted"][a]) for a in deciders}
    statuses = {a: collections.Counter() for a in deciders}
    start = time.monotonic()  # all loads/self-checks have completed before this clock
    handles = {a: (out / f"{a}.predictions.jsonl").open("x", encoding="utf-8", newline="\n") for a in deciders}
    try:
        for index, row in enumerate(rows):
            rid = row["_evaluation"]["run_id"]
            # Alternate the first arm to avoid always giving one arm a warmer device.
            for arm in (("v1", "v2") if index % 2 == 0 else ("v2", "v1")):
                deciders[arm].backend._enc_memo.clear()
                rec, seconds, memory = timed_capture(torch, row, deciders[arm])
                expected_status = "ok" if volume["per_row"][rid]["admitted"][arm] else "exhausted"
                if rec["status"] != expected_status:
                    raise ValueError("admission/status drift; no row may be silently dropped")
                # Raw predictions include IDs/probabilities/revision fingerprints, never state, instructions or gold.
                handles[arm].write(json.dumps(rec, allow_nan=False) + "\n")
                handles[arm].flush()
                elapsed[arm] += seconds
                statuses[arm][rec["status"]] += 1
                weight = P.reference_seconds(volume["per_row"][rid][arm]) if expected_status == "ok" else 0
                completed[arm] += weight
                remaining[arm] = max(0, remaining[arm] - weight)
            wall = time.monotonic() - start
            if index >= 19 or wall >= 120:
                rate_check(elapsed, completed, remaining, wall, 60 * params()["phase_caps_minutes"]["screen"])
            S.write_json(out / "progress.json", {"paired_rows": index + 1, "denominator": R.ROWS,
                "status_counts": {a: dict(c) for a, c in statuses.items()}, "arm_seconds": elapsed,
                "wall_seconds": wall, "model_load_seconds": load_seconds, "last_peak": memory})
            (out / "heartbeat").touch()
    finally:
        for handle in handles.values():
            handle.close()


def audit(out):
    report = S.audit_files(ROWS, out / "v1.predictions.jsonl", out / "v2.predictions.jsonl", out / "identities.json")
    for arm in ("v1", "v2"):
        if report["arms"][arm]["status_counts"] != {"ok": 2000, "exhausted": 8}:
            raise ValueError("all eight exhausted rows must remain in denominator")
    report["execution_attestation"] = {"setup_sha256": S.sha(out / "setup.json"),
        "probe_sha256": S.sha(out / "probe.json"), "preflight_sha256": S.sha(out / "preflight.json"),
        "progress_sha256": S.sha(out / "progress.json"), "gpu_used": True, "training": False}
    S.write_json(out / "report.json", report)


def finalize(out, rc):
    marker = out / "RWKV7_CODE_LABEL_QUALIFICATION_COMPLETE"
    marker.unlink(missing_ok=True)
    completed = rc == 0
    if completed:
        report = json.loads((out / "report.json").read_text())
        if (report["screen_status"] != "complete" or report["rows_denominator"] != R.ROWS
                or report.get("synthetic") is not False or report.get("promotable") is not False
                or report.get("beats_clef") is not False or report.get("paired_ok_rows") != 2000
                or report.get("input_sha256", {}).get("rows") != R.ROWS_SHA256):
            raise ValueError("strict full-denominator audit required")
        for arm in ("v1", "v2"):
            if (report["arms"][arm]["status_counts"] != {"ok": 2000, "exhausted": 8}
                    or report["input_sha256"][arm + "_predictions"] != S.sha(out / f"{arm}.predictions.jsonl")):
                raise ValueError("audited prediction identity drift")
    # Export on success AND failure; never include /data, requests, source rows or model weights.
    names = ("setup.json", "probe.json", "preflight.json", "identities.json", "v1.predictions.jsonl",
             "v2.predictions.jsonl", "progress.json", "report.json", "failure.json")
    manifest = {name: S.sha(out / name) for name in names if (out / name).is_file()}
    if completed and any(name not in manifest for name in names[:-1]):
        raise ValueError("missing required qualification artifact")
    S.write_json(out / "SHA256SUMS.json", manifest)
    export = out / "qualification.tgz"
    with tarfile.open(export, "w:gz") as tar:
        for name in (*manifest, "SHA256SUMS.json"):
            tar.add(out / name, arcname=name, recursive=False)
    with tarfile.open(export, "r:gz") as tar:
        for member in tar.getmembers():
            import hashlib
            if not member.isfile() or member.name not in {*manifest, "SHA256SUMS.json"}:
                raise ValueError("invalid export member")
            want = manifest.get(member.name, S.sha(out / "SHA256SUMS.json"))
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest() != want:
                raise ValueError("export readback mismatch")
    receipt = {"ok": completed, "exit_code": rc, "status": "complete" if completed else "incomplete",
        "export": {"path": export.name, "sha256": S.sha(export), "bytes": export.stat().st_size, "verified": True},
        "commit_sha": json.loads((out / "setup.json").read_text()).get("commit_sha") if (out / "setup.json").exists() else None,
        "rows_denominator": R.ROWS, "promotable": False, "beats_clef": False, "spend_authorized": False,
        "actual_cost_usd": None, "actual_cost_source": "coordinator must append provider billed receipt after completion",
        "planned_worst_case_usd": params()["hard_cap_usd"], "report": "report.json" if completed else None}
    S.write_json(out / "receipt.json", receipt)
    S.write_json(Path(os.environ.get("GMN_RESULT_PATH", str(out / "result.json"))), receipt)
    if completed:
        marker.write_text(S.sha(out / "receipt.json") + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("admit", "setup", "probe", "screen", "audit", "finalize", "dry-run"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rc", type=int, default=1)
    args = parser.parse_args(argv)
    if args.phase == "admit":
        try:
            admit()
        except Exception:
            print("Launch disabled: requires merged reviewed proof and explicit spend-issue authorization.")
            return 2
        return 0
    if args.out is None:
        parser.error("--out required")
    try:
        if args.phase == "dry-run":
            args.out.mkdir(parents=True, exist_ok=False)
            result = S.dry_run(args.out)
            S.write_json(args.out / "report.json", result)
        elif args.phase == "finalize":
            finalize(args.out, args.rc)
        else:
            # Direct module execution cannot bypass launch admission either.
            admit()
            globals()[args.phase](args.out)
    except Exception as exc:
        if args.out.exists():
            S.write_json(args.out / "failure.json", {"phase": args.phase, "error_type": type(exc).__name__,
                "status": "incomplete", "private_exception_text_withheld": True})
        print(json.dumps({"phase": args.phase, "status": "incomplete", "error_type": type(exc).__name__}))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
