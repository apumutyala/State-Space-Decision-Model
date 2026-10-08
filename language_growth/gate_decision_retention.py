"""Authorized DEV Index retention through the unchanged official harness.

Only explicit, independently hash-authorized scoring metadata and prediction
exports are opened. No suite loader, original dataset, TEST, retention holdout,
model, or inference path is used. Authorization is trusted host evidence, not
an authenticated signature. It must certify complete source-case/group export
and original request hashes before this offline scorer is invoked.

Authorization schema ``ssdm.language_authorized_dev_index.v1`` contains:
``split=DEV``, ``purpose=language_parent_decision_regression``, ``edition``,
``parent_identity``, ``candidate_identity``, ``scoring_rows_sha256``, both
``*_predictions_sha256``, ``cohort``, ``bootstrap`` and ``items``. Bootstrap
pins ``draws``, ``seed`` and ``confidence=0.95``; passing requires >=2000 draws.
Identities contain exactly
``artifact_sha256`` and ``revision_id``. Cohort contains ``scope`` (full_dev or
dev_subset), ``expected_rows``, ``benchmark_rows`` (catalog ID string to row
count), ``unavailable_benchmarks`` (reviewed absent board slots),
``complete_groups``, ``complete_source_cases``
and ``minimum_cases_per_benchmark`` (at least 20). Each item contains exactly
``run_id``, ``request_sha256``, ``scoring_row_sha256`` and ``source_case_id``.
Scoring rows are approved ordinary kit rows without state/task text: only
_evaluation, questions, expected, scoring, metadata and provenance are allowed.
The host supplies source_case_id computed from the original frozen case graph;
recomputing cases after stripping state would lose identical-state links.

Predictions are JSONL (or an existing run_id-keyed JSON object), retain every
authorized row, and include request_sha256, checkpoint_sha256 and revision_id.
Ordinary kit status/response exports are accepted. Legacy single-question
choice probability exports are supported, never caller-provided accuracy.
Missing, refused, malformed and errored outputs retain the Index denominator.
Older accuracy-only authorization without Index metadata is rejected.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "ssdm.language_authorized_dev_index.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("missing or invalid SHA-256 pin")
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _json(text):
    return json.loads(text, object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _read(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(digest):
        raise ValueError("authorized export file hash mismatch")
    return raw.decode("utf-8")


def _identity(value):
    if not isinstance(value, dict):
        raise ValueError("actual runtime identity required")
    sha = value.get("artifact_sha256", value.get("checkpoint_sha256"))
    revision = value.get("parent_revision", value.get("revision_id"))
    if not isinstance(revision, str) or not revision:
        raise ValueError("actual served revision required")
    return {"artifact_sha256": _sha(sha), "revision_id": revision}


def _records(text):
    # Existing harness exports may use a run_id-keyed object instead of JSONL.
    try:
        value = _json(text)
    except json.JSONDecodeError:
        value = [_json(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict) and not ("run_id" in value or "item_id" in value):
        rows = []
        for key, row in value.items():
            if not isinstance(row, dict) or row.get("run_id", key) != key:
                raise ValueError("prediction map run ID mismatch")
            rows.append({**row, "run_id": key})
    else:
        rows = value if isinstance(value, list) else [value]
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("prediction row must be an object")
        rid = row.get("run_id", row.get("item_id"))
        if not isinstance(rid, str) or not rid or rid in result:
            raise ValueError("missing or duplicate prediction run ID")
        result[rid] = row
    return result


def _result(prediction, row, item, identity):
    if (prediction.get("request_sha256", prediction.get("payload_sha256")) != item["request_sha256"]
            or prediction.get("checkpoint_sha256") != identity["artifact_sha256"]
            or prediction.get("revision_id") != identity["revision_id"]):
        raise ValueError("prediction request/checkpoint/revision identity mismatch")
    result = {"status": "error"}
    if prediction.get("status", "ok" if "probabilities" in prediction else None) != "ok":
        return result
    try:
        response = prediction.get("response")
        if response is None and "probabilities" in prediction:
            if len(row["questions"]) != 1:
                raise ValueError("legacy probabilities require one choice question")
            qid, question = next(iter(row["questions"].items()))
            if question["type"] != "choice":
                raise ValueError("legacy probabilities are choice-only")
            probs = prediction["probabilities"]
            choice = min(probs, key=lambda k: (-probs[k], k))
            response = {"answers": {qid: {"type": "choice", "choice": choice, "probabilities": probs}}}
        from decision_index.engines.base import validate
        validate(row["questions"], response)
        # The kit accepts a loose normalization tolerance; preserve the bridge's
        # strict distribution contract and boolean/numeric distinction here.
        for answer in response["answers"].values():
            probs = answer.get("probabilities")
            if probs is not None and (any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in probs.values())
                                      or abs(sum(probs.values()) - 1) > 1e-9):
                raise ValueError("invalid probability distribution")
            if answer["type"] == "noul" and type(answer["noul"]) not in (int, float):
                raise ValueError("invalid noul probability")
        result = {"status": "ok", "response": response}
    except (ValueError, TypeError, KeyError, AttributeError):
        pass  # malformed answers remain errors in the official denominator
    return result


def decision_index_retention(authorization, authorization_sha256, scoring_rows,
                             parent_predictions, candidate_predictions, *,
                             qualified_parent_identity, candidate_identity,
                             draws=2000, seed=20261008):
    """Paired candidate-parent Index points; pass requires a complete full DEV.

    The lower 95% confidence bound must be >=0, with fixed board weights and
    no regression margin. This is decision retention only, never language or
    safety qualification. A partial diagnostic always returns passed=False.
    """
    if type(draws) is not int or draws < 100 or type(seed) is not int:
        raise ValueError("at least 100 paired bootstrap draws and integer seed required")
    auth = _json(_read(authorization, authorization_sha256))
    if (not isinstance(auth, dict) or auth.get("schema") != SCHEMA or auth.get("split") != "DEV"
            or auth.get("purpose") != "language_parent_decision_regression"):
        raise ValueError("separately authorized DEV Index metadata required; accuracy exports are insufficient")
    parent_id, candidate_id = _identity(qualified_parent_identity), _identity(candidate_identity)
    if auth.get("parent_identity") != parent_id or auth.get("candidate_identity") != candidate_id:
        raise ValueError("authorization differs from actual qualified parent/candidate identity")
    if auth.get("bootstrap") != {"draws": draws, "seed": seed, "confidence": .95}:
        raise ValueError("bootstrap configuration differs from trusted preregistration")
    cohort = auth["cohort"]
    if (cohort.get("scope") not in {"full_dev", "dev_subset"}
            or type(cohort.get("expected_rows")) is not int or cohort["expected_rows"] < 1
            or cohort.get("complete_groups") is not True or cohort.get("complete_source_cases") is not True
            or type(cohort.get("minimum_cases_per_benchmark")) is not int
            or cohort["minimum_cases_per_benchmark"] < 20):
        raise ValueError("complete authorized groups/source cases and minimum cohort required")
    items = auth["items"]
    if not isinstance(items, list) or len(items) != cohort["expected_rows"]:
        raise ValueError("authorized item census mismatch")
    inventory = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"run_id", "request_sha256", "scoring_row_sha256", "source_case_id"}:
            raise ValueError("incomplete authorized item identity")
        rid, case = item["run_id"], item["source_case_id"]
        if not isinstance(rid, str) or not rid or rid in inventory or not isinstance(case, str) or not case:
            raise ValueError("empty or duplicate authorized item/source case")
        _sha(item["request_sha256"]); _sha(item["scoring_row_sha256"])
        inventory[rid] = item
    rows = [_json(line) for line in _read(scoring_rows, auth["scoring_rows_sha256"]).splitlines() if line.strip()]
    by_id, clusters = {}, {}
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"_evaluation", "questions", "expected", "scoring", "metadata", "provenance"}:
            raise ValueError("only approved scoring metadata allowed; original input text is forbidden")
        ev = row["_evaluation"]
        if ev.get("split", "DEV") not in {"DEV", "dev"}:
            raise ValueError("scoring metadata is not DEV")
        rid = ev["run_id"]
        if rid not in inventory or rid in by_id or _digest(row) != inventory[rid]["scoring_row_sha256"]:
            raise ValueError("scoring row identity/census mismatch")
        if ev["payload_sha256"] != inventory[rid]["request_sha256"]:
            raise ValueError("scoring row original request identity mismatch")
        key = (ev["catalog_id"], str(ev["group_id"]))
        case = inventory[rid]["source_case_id"]
        if key in clusters and clusters[key] != case:
            raise ValueError("linked group split across source cases")
        clusters[key] = case
        by_id[rid] = row
    if set(by_id) != set(inventory):
        raise ValueError("scoring metadata must cover every authorized item")
    from harness import score, bootstrap
    from harness.suite import index_ids, in_edition
    edition = auth["edition"]
    census = {}
    for row in rows:
        key = str(row["_evaluation"]["catalog_id"])
        census[key] = census.get(key, 0) + 1
    expected_census = cohort.get("benchmark_rows")
    if (not isinstance(expected_census, dict) or expected_census != census
            or any(type(n) is not int or n < 1 for n in expected_census.values())):
        raise ValueError("authorized DEV benchmark census mismatch")
    unavailable = cohort.get("unavailable_benchmarks")
    if (not isinstance(unavailable, list) or any(type(n) is not int for n in unavailable)
            or len(set(unavailable)) != len(unavailable)
            or set(unavailable) - set(index_ids(edition))
            or set(unavailable) & {int(n) for n in census}):
        raise ValueError("invalid reviewed unavailable benchmark inventory")
    if any(not in_edition(row, edition) for row in rows):
        raise ValueError("scoring row is outside authorized Index edition")
    systems, errors = [], {}
    for name, path, identity in (("parent", parent_predictions, parent_id), ("candidate", candidate_predictions, candidate_id)):
        predictions = _records(_read(path, auth[name + "_predictions_sha256"]))
        if set(predictions) != set(inventory):
            raise ValueError("predictions must cover identical complete authorized items, including errors")
        results = {rid: _result(predictions[rid], row, inventory[rid], identity) for rid, row in by_id.items()}
        errors[name] = sum(result["status"] != "ok" for result in results.values())
        benches = score.decompose(rows, results, clusters="group", edition=edition)
        benches = {n: bench.collapse(lambda group, n=n: clusters[(n, str(group))]) for n, bench in benches.items()}
        systems.append(benches)
    parent, candidate = systems
    cases = {str(n): len(bench.clusters) for n, bench in candidate.items()}
    missing = sorted(set(index_ids(edition)) - set(candidate))
    complete = (cohort["scope"] == "full_dev" and draws >= 2000
                and set(candidate) | set(unavailable) == set(index_ids(edition))
                and all(n >= cohort["minimum_cases_per_benchmark"] for n in cases.values()))
    # The frozen bootstrap's area summary assumes at least one vector-valued
    # contribution per area. Supply explicit zero contributions for missing
    # benchmarks, exactly the board convention, without inventing scored rows
    # or renormalizing a partial panel. These never count toward completeness.
    import numpy as np
    def zero_value(total):
        return np.zeros(total.shape[0]), np.zeros(total.shape[0])
    for benches in systems:
        for n in missing:
            benches[n] = score.Bench(n, ["missing-benchmark"], [[0.0]], zero_value, "missing", edition)
    paired = bootstrap.compare(candidate, parent, B=draws, seed=seed, level=.95, renormalize=False)
    criterion = paired["diff"]["index"]["lo"] >= 0
    return {"schema": "ssdm.language_decision_index_retention.v1", "status": "authorized_dev_index_measured",
            "passed": bool(complete and criterion), "criterion_observed": bool(criterion),
            "complete_cohort": bool(complete), "metric": "decision_index", "edition": edition,
            "renormalize_missing_benchmarks": False, "paired": paired, "n_items": len(rows),
            "source_cases_per_benchmark": cases, "error_rows": errors,
            "missing_benchmarks": missing,
            "authorized_benchmark_rows": expected_census, "reviewed_unavailable_benchmarks": unavailable,
            "parent_identity": parent_id, "candidate_identity": candidate_id,
            "authorization_sha256": authorization_sha256, "scoring_rows_sha256": auth["scoring_rows_sha256"],
            "parent_predictions_sha256": auth["parent_predictions_sha256"],
            "candidate_predictions_sha256": auth["candidate_predictions_sha256"],
            "reason": "complete full-DEV no-regression criterion" if complete else "incomplete full-DEV cohort; diagnostic only"}
