"""CPU-only artifact audit and synthetic production-path dry run; no model loader.

The audit reads only an exact hashed, explicitly supplied DEV subset. It never
loads a slice, opens TEST, downloads weights, or emits a promotion verdict.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

from decide.calibration import OptionCountTemperature
from decide.schema import canonical_json, canonical_options, request_fingerprint, wire_answer
from harness.jsonl import read_jsonl_strict, read_results_strict
from jobs.rwkv7_code_label_screen import rule as R

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, values):
    Path(path).write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in values), encoding="utf-8", newline="\n")


def checked_cohort():
    from jobs.common.dev_subset import ids_sha256, load_subset

    subset = load_subset(check_manifest=True)
    if (len(subset["run_ids"]) != R.ROWS or len(set(subset["run_ids"])) != R.ROWS
            or ids_sha256(subset["run_ids"]) != R.IDS_SHA256
            or subset["dev_manifest_sha256_uncompressed"] != R.MANIFEST_SHA256):
        raise ValueError("pinned DEV cohort changed")
    return subset


def pinned_model_identity():
    params = json.loads((ROOT / "jobs/rwkv7/job_params_readout.json").read_text(encoding="utf-8"))
    model_id, snapshot = params["model"].rsplit("@", 1)
    entries = [line.split("  ", 1) for line in
               (ROOT / "jobs/rwkv7/snapshot.sha256").read_text(encoding="utf-8").splitlines() if line]
    weights = sorted((name, digest) for digest, name in entries if name.endswith(".safetensors"))
    if not weights:
        raise ValueError("missing pinned weight hashes")
    # Exact hf_readout._hash_files formula: one file uses its digest; multiple
    # files hash sorted basename:digest lines, including each trailing newline.
    weights_hash = weights[0][1] if len(weights) == 1 else hashlib.sha256(
        "".join(f"{name}:{digest}\n" for name, digest in weights).encode()).hexdigest()
    return model_id, snapshot, weights_hash


def check_revision(revision, arm, *, synthetic=False):
    required = {"revision_id", "backend", "family", "model_id", "snapshot", "files_sha256",
                "prompt_template", "inference_config", "inference_config_sha256", "adapter_version",
                "max_context", "notes"}
    if not isinstance(revision, dict) or not required <= revision.keys():
        raise ValueError(f"{arm}: missing revision identity fields")
    for key in ("revision_id", "backend", "family", "model_id", "inference_config_sha256", "adapter_version"):
        if not isinstance(revision[key], str) or not revision[key]:
            raise ValueError(f"{arm}: malformed revision {key}")
    if revision["snapshot"] is not None and (not isinstance(revision["snapshot"], str) or not revision["snapshot"]):
        raise ValueError(f"{arm}: malformed revision snapshot")
    if ((revision["max_context"] is not None and
         (type(revision["max_context"]) is not int or revision["max_context"] <= 0))
            or not isinstance(revision["notes"], dict)):
        raise ValueError(f"{arm}: malformed revision max_context/notes")
    files, template, config = (revision[k] for k in ("files_sha256", "prompt_template", "inference_config"))
    for value, keys, label in ((files, ("weights", "tokenizer"), "files_sha256"),
                               (template, ("id", "sha256"), "prompt_template")):
        # Nonempty strings preserve the explicit synthetic weights="not-loaded"
        # sentinel while preventing a shared absence from passing matching.
        if not isinstance(value, dict) or any(not isinstance(value.get(k), str) or not value[k] for k in keys):
            raise ValueError(f"{arm}: missing/malformed revision {label}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", files["weights"]) and not (synthetic and files["weights"] == "not-loaded"):
        raise ValueError(f"{arm}: malformed weights hash; synthetic sentinel forbidden in real audit")
    if not synthetic:
        model_id, snapshot, weights_hash = pinned_model_identity()
        for key, expected in (("model_id", model_id), ("snapshot", snapshot)):
            if revision[key] != expected:
                raise ValueError(f"{arm}: changed pinned model {key}")
        if files["weights"] != weights_hash:
            raise ValueError(f"{arm}: changed pinned weights hash")
    runtime = {"dtype", "device", "torch", "transformers", "kernel_stack"}
    if not isinstance(config, dict):
        raise ValueError(f"{arm}: malformed inference_config")
    if not runtime <= config.keys():
        raise ValueError(f"{arm}: missing runtime identity fields")
    if (any(not isinstance(config[k], str) or not config[k] for k in ("dtype", "device", "torch"))
            or (config["transformers"] is not None and
                (not isinstance(config["transformers"], str) or not config["transformers"]))
            or not isinstance(config["kernel_stack"], dict)):
        raise ValueError(f"{arm}: malformed runtime identity")
    # The merged volume receipt pins the existing renderer, codebook, sources and
    # scoring configuration. Runtime/model identity is separately matched below.
    reference = json.loads((ROOT / "receipts/2026-10-07-D-code-label-v2-fork/dev-volume.json").read_text(encoding="utf-8"))[arm + "_revision"]
    for key in ("backend", "family", "prompt_template", "adapter_version", "max_context", "notes"):
        if revision[key] != reference[key]:
            raise ValueError(f"{arm}: changed revision {key}")
    if {k: v for k, v in config.items() if k not in runtime} != {k: v for k, v in reference["inference_config"].items() if k not in runtime}:
        raise ValueError(f"{arm}: changed scoring/config/source hash")
    if hashlib.sha256(canonical_json(config).encode()).hexdigest() != revision["inference_config_sha256"]:
        raise ValueError(f"{arm}: invalid inference config hash")
    if files["tokenizer"] != reference["files_sha256"]["tokenizer"]:
        raise ValueError(f"{arm}: changed tokenizer hash")
    # Exact HFReadoutBackend._build_revision format: internal consistency only,
    # never execution attestation for externally supplied prediction records.
    expected_id = (
        f"{revision['family']}:{revision['model_id']}@{(revision['snapshot'] or 'local')[:12]}"
        f";w={files['weights'][:12]};tok={files['tokenizer'][:12]}"
        f";prompt={template['id']}.{template['sha256'][:8]}"
        f";cfg={revision['inference_config_sha256'][:8]};adapter={revision['adapter_version']}"
    )
    if revision["revision_id"] != expected_id:
        raise ValueError(f"{arm}: inconsistent revision ID")


def audit_arm(rows, records, revision, minimum_ok):
    import math

    ids = [row["_evaluation"]["run_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(records) != set(ids):
        raise ValueError("exact row coverage required; missing, extra or duplicate rows")
    counts = collections.Counter()
    ok_ids = set()
    for row in rows:
        rid = row["_evaluation"]["run_id"]
        rec = records[rid]
        status = rec.get("status")
        if status not in ("ok", "unsupported", "exhausted"):
            raise ValueError(f"{rid}: error/unknown status is not a completed screen")
        counts[status] += 1
        evidence = rec.get("screen_evidence") or {}
        if evidence.get("revision_id") != revision["revision_id"]:
            raise ValueError(f"{rid}: revision drift")
        request = {"model": "screen", "state": row["state"], "questions": row["questions"]}
        if evidence.get("request_fingerprint") != request_fingerprint(request):
            raise ValueError(f"{rid}: request changed")
        response = rec.get("response")
        raw = (rec.get("raw_output") or {}).get("raw_probabilities")
        if status != "ok":
            if (not isinstance(response, dict) or not isinstance(response.get("answers"), dict)
                    or response["answers"] != {} or raw != {} or evidence.get("served_probabilities") != {}
                    or evidence.get("served_config") is not None or not isinstance(rec.get("error"), dict)
                    or not isinstance(rec["error"].get("code"), str) or not rec["error"]["code"]):
                raise ValueError(f"{rid}: non-ok row needs explicit reason and no partial answers")
            continue
        ok_ids.add(rid)
        if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
            raise ValueError(f"{rid}: malformed answer fields")
        qs = row["questions"]
        served = evidence.get("served_probabilities")
        config = evidence.get("served_config") or {}
        if (config.get("calibration") != OptionCountTemperature().identity()
                or config.get("temperature_per_question") != {qid: 1.0 for qid in qs}
                or config.get("policy_masking") != "none (served covers every supplied option)"):
            raise ValueError(f"{rid}: calibration identity/masking changed")
        if not isinstance(raw, dict) or set(raw) != set(qs) or served != raw:
            raise ValueError(f"{rid}: raw/served full-domain probabilities required")
        if set(response.get("answers", {})) != set(qs):
            raise ValueError(f"{rid}: incomplete answer fields")
        for qid, q in qs.items():
            probs = raw[qid]
            options = {option for option, _ in canonical_options(q)}
            if (not isinstance(probs, dict) or set(probs) != options
                    or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values())
                    or abs(sum(probs.values()) - 1) > R.PROBABILITY_TOLERANCE):
                raise ValueError(f"{rid}/{qid}: incomplete/nonfinite/unnormalized option distribution")
            if response["answers"][qid] != wire_answer(q, probs):
                raise ValueError(f"{rid}/{qid}: wire answer differs from identity served probabilities")
    if len(ok_ids) < minimum_ok:
        raise ValueError(f"inadequate answered count: {len(ok_ids)} < {minimum_ok}; denominator stays {len(rows)}")
    return {"denominator": len(rows), "status_counts": dict(counts), "ok_rows": len(ok_ids)}, ok_ids


def summarize(rows, arms, revisions, *, synthetic=False):
    from harness import bootstrap, calibration, score
    from jobs.clef_baseline.summarize import g1_audit

    ids = [row["_evaluation"]["run_id"] for row in rows]
    if synthetic:
        if len(rows) != 12 or set(ids) != {f"synthetic:{i}" for i in range(12)}:
            raise ValueError("synthetic proof must use the fixed synthetic cohort")
    else:
        from jobs.common.dev_subset import ids_sha256

        if len(rows) != R.ROWS or len(set(ids)) != R.ROWS or ids_sha256(ids) != R.IDS_SHA256:
            raise ValueError("exact pinned full DEV cohort required")
    minimum = len(rows) - 1 if synthetic else R.MIN_OK_ROWS
    reports, ok_sets, benches = {}, {}, {}
    for arm in ("v1", "v2"):
        check_revision(revisions[arm], arm, synthetic=synthetic)
        reports[arm], ok_sets[arm] = audit_arm(rows, arms[arm], revisions[arm], minimum)
        score.require_results(rows, arms[arm])
        fields = calibration.fields(rows, arms[arm])
        # No ECE=0 fallback for empty/insufficient calibration data.
        if len(fields) < 2:
            raise ValueError("inadequate calibration field count")
        benches[arm] = score.decompose(rows, arms[arm], edition=R.EDITION)
        rep = bootstrap.replicate_skills([benches[arm]], B=R.BOOTSTRAP_REPLICATES, seed=R.BOOTSTRAP_SEED)[0]
        reports[arm]["slice_index"] = bootstrap.summarize_one(rep, level=R.CI_LEVEL)
        reports[arm]["raw_audit"] = g1_audit(list(arms[arm].values()))
        reports[arm]["calibration_diagnostic"] = calibration.board_ece(fields, sample=False, base_only=False)
    for key in ("model_id", "snapshot", "files_sha256", "max_context"):
        if revisions["v1"][key] != revisions["v2"][key]:
            raise ValueError(f"unmatched model {key}")
    for key in ("dtype", "device", "torch", "transformers", "kernel_stack"):
        if revisions["v1"]["inference_config"][key] != revisions["v2"]["inference_config"][key]:
            raise ValueError(f"unmatched runtime {key}")
    paired = len(ok_sets["v1"] & ok_sets["v2"])
    if paired < minimum:
        raise ValueError("inadequate paired answered count")
    comparison = bootstrap.compare(benches["v2"], benches["v1"], B=R.BOOTSTRAP_REPLICATES,
                                   seed=R.BOOTSTRAP_SEED, level=R.CI_LEVEL)
    return {"schema": "ssdm.rwkv7-code-label-screen.report.v1", "rule": R.RULE_VERSION,
            "synthetic": synthetic, "rows_denominator": len(rows), "paired_ok_rows": paired,
            "arms": reports, "paired_v2_minus_v1": comparison,
            "scope": "synthetic API proof" if synthetic else "pinned DEV integrity screen",
            **R.verdict(True)}


def capture(row, decider):
    from jobs.common.readout_engine import map_response

    request = {"model": "screen", "state": row["state"], "questions": row["questions"]}
    response = decider.decide(request)
    ss = response["ssdm"]
    status = ss["status"]
    if status == "ok":
        wire, raw = map_response(response)
    else:
        wire, raw = {"answers": {}}, {"raw_probabilities": {}}
    return {"run_id": row["_evaluation"]["run_id"], "status": status, "response": wire,
            "raw_output": raw, "error": ss["error"],
            "screen_evidence": {"revision_id": ss["diagnostics"]["revision_id"],
                "request_fingerprint": request_fingerprint(request),
                "served_probabilities": ss["served_probabilities"], "served_config": ss["served_config"]}}


def audit_files(rows_path, v1_path, v2_path, identities_path):
    # Hash before decompression/parsing: no arbitrary split or content is admitted.
    if sha(rows_path) != R.ROWS_SHA256:
        raise ValueError("pinned DEV row file hash changed")
    subset = checked_cohort()
    rows = read_jsonl_strict(rows_path)
    if len(rows) != R.ROWS or {r["_evaluation"]["run_id"] for r in rows} != set(subset["run_ids"]):
        raise ValueError("exact pinned full DEV cohort required")
    identities = json.loads(Path(identities_path).read_text(encoding="utf-8"))
    if identities.get("rows_sha256") != R.ROWS_SHA256 or identities.get("ids_sha256") != R.IDS_SHA256:
        raise ValueError("prediction cohort identity changed")
    arms = {"v1": read_results_strict(v1_path, subset["run_ids"]), "v2": read_results_strict(v2_path, subset["run_ids"])}
    result = summarize(rows, arms, identities["revisions"])
    result["input_sha256"] = {"rows": sha(rows_path), "v1_predictions": sha(v1_path),
                             "v2_predictions": sha(v2_path), "identities": sha(identities_path)}
    return result


def dry_run(out):
    from jobs.rwkv7_code_label_screen.synthetic import fixture

    rows, deciders = fixture()
    revisions = {arm: decider.revision() for arm, decider in deciders.items()}
    arms = {arm: {row["_evaluation"]["run_id"]: capture(row, decider) for row in rows}
            for arm, decider in deciders.items()}
    for arm in arms:
        write_jsonl(out / f"{arm}.predictions.jsonl", arms[arm].values())
        arms[arm] = read_results_strict(out / f"{arm}.predictions.jsonl", [r["_evaluation"]["run_id"] for r in rows])
    write_jsonl(out / "synthetic.rows.jsonl", rows)
    write_json(out / "synthetic.identities.json", {"revisions": revisions, "synthetic": True})
    result = summarize(rows, arms, revisions, synthetic=True)
    result.update({"gpu_used": False, "model_loaded": False, "cost_usd": 0,
                   "data_opened": "synthetic only; no DEV or TEST items",
                   "raw_predictions_sha256": {arm: sha(out / f"{arm}.predictions.jsonl") for arm in arms}})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    dry = subs.add_parser("dry-run")
    dry.add_argument("--out", type=Path, required=True)
    audit = subs.add_parser("audit")
    for name in ("rows", "v1", "v2", "identities", "out"):
        audit.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    # Fresh destination prevents stale success reports after a failed rerun.
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        result = dry_run(args.out) if args.mode == "dry-run" else audit_files(args.rows, args.v1, args.v2, args.identities)
    except Exception as exc:
        write_json(args.out / "report.json", {**R.verdict(False), "error_type": type(exc).__name__, "error": str(exc)})
        return 3
    write_json(args.out / "report.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
