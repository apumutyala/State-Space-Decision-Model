"""Validate fixture packaging/references only; this is not a semantic kernel."""

from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{path.name}: duplicate JSON key {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def validate_pack(root=ROOT):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    require(manifest["schema"] == "ssdm.language_fixture_manifest.v0", "manifest schema")
    for link_key in ("contract_reference", "plan_reference"):
        link = (root / manifest[link_key]).resolve()
        require(link.is_file(), f"broken {link_key}")
    catalog_path = (root / manifest["catalog"]).resolve()
    require(catalog_path.is_relative_to(root), "catalog path escapes fixtures")
    catalog = read_json(catalog_path)
    require(catalog["schema"] == "ssdm.language_fixture_catalog.v0", "catalog schema")
    worlds = {item["world_id"]: item["context_ref"] for item in manifest["worlds"]}
    require(len(worlds) == len(manifest["worlds"]), "duplicate world IDs")
    require(all(ref in catalog["snapshots"] for ref in worlds.values()), "unknown world snapshot")
    records = {r["record_ref"]: r for r in catalog["records"]}
    require(len(records) == len(catalog["records"]), "duplicate catalog record IDs")
    lookup = {
        "record_ref": records,
        "context_ref": catalog["snapshots"],
        "input_context_ref": catalog["snapshots"],
        "policy_ref": catalog["policies"],
        "registry_ref": catalog["registries"],
        "entity_ref": catalog["entities"],
        "source_ref": catalog["sources"],
    }

    def references(value, where):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in lookup:
                    require(isinstance(child, str) and child in lookup[key], f"{where}: unresolved {key}: {child}")
                if key == "state_history_context_refs":
                    require(all(ref in catalog["snapshots"] for ref in child), f"{where}: unknown state history")
                references(child, where)
        elif isinstance(value, list):
            for child in value:
                references(child, where)

    references(catalog["records"], "catalog")
    for snapshot, data in catalog["snapshots"].items():
        require(len(data["record_refs"]) == len(set(data["record_refs"])), f"{snapshot}: duplicate record")
        require(all(ref in records for ref in data["record_refs"]), f"{snapshot}: unknown record")
    for record in records.values():
        require(all(ref in records for ref in record["conflicts"]), "unresolved conflict reference")
        for value in record["fields"].values():
            if value["value_type"] == "integer":
                require(type(value["value"]) is int, "integer field is not an exact integer")
            elif value["value_type"] == "decimal":
                require(isinstance(value["value"], str), "decimal field must be a string")
                try:
                    require(Decimal(value["value"]).is_finite(), "nonfinite decimal")
                except InvalidOperation as exc:
                    raise ValueError("invalid decimal spelling") from exc

    ids, files, question_ids = set(), set(), set()
    for entry in manifest["cases"]:
        path = (root / entry["path"]).resolve()
        require(path.is_relative_to(root / "cases") and path.suffix == ".json", "case path escapes cases")
        case = read_json(path)
        cid = entry["case_id"]
        require(cid not in ids and path not in files, f"duplicate case {cid}")
        require(re.fullmatch(r"LG\d{2}", cid) is not None, f"unstable case ID {cid}")
        ids.add(cid)
        files.add(path)
        require(case["case_id"] == cid and case["tags"] == entry["tags"], f"{cid}: manifest mismatch")
        require(case["world_id"] == entry["world_id"] and worlds[case["world_id"]] == case["host"]["context_ref"], f"{cid}: world mismatch")
        require(isinstance(case["question"], str) and case["question"], f"{cid}: missing question")
        require(case["label_provenance"] == "codex_authored_pilot_design_expectation_not_independent_evaluation", f"{cid}: provenance missing")
        expected = case["expected"]
        require(expected["provider_precondition"] in {"pass", "fail"}, f"{cid}: provider status")
        require(expected["kernel_status"] in {"accepted", "rejected", "not_run"}, f"{cid}: kernel status")
        require(expected["kernel_reason_category"] is None or expected["kernel_reason_category"] in manifest["kernel_reason_categories"], f"{cid}: reason category")
        require(type(expected["useful_complete_answer"]) is bool, f"{cid}: useful flag")
        require(isinstance(expected["why"], str) and expected["why"], f"{cid}: missing explanation")
        references(case, cid)
        host = case["host"]
        decision = case["decision_only"]
        require(decision["question_id"] == entry["decision_question_id"], f"{cid}: decision question ID")
        require(decision["question_id"] == f"{cid}-D0" and decision["question_id"] not in question_ids, f"{cid}: unstable or duplicate decision question ID")
        question_ids.add(decision["question_id"])
        require(decision["question"] == case["question"], f"{cid}: different decision-only task")
        require(decision["evidence_context_ref"] == host["context_ref"] and decision["policy_ref"] == host["policy_ref"], f"{cid}: decision-only evidence mismatch")
        option_ids = [option["id"] for option in decision["options"]]
        require(len(option_ids) >= 2 and len(option_ids) == len(set(option_ids)), f"{cid}: decision choice IDs")
        require(all(option_id.startswith(f"{cid}-") for option_id in option_ids), f"{cid}: unstable option ID")
        require(all(isinstance(option["description"], str) and option["description"] for option in decision["options"]), f"{cid}: empty option")
        require(decision["target_option_id"] in option_ids, f"{cid}: missing decision target")
        obligations = {item["obligation_id"]: item for item in host["unmet_obligations"]}
        for obligation in obligations.values():
            require(all(ref in host["query_constants"] or ref in records for ref in obligation["allowed_refs"]), f"{cid}: unresolved obligation reference")
        proposals = [case["proposal"]]
        repaired = case.get("repair", {}).get("repaired_proposal")
        if repaired is not None:
            proposals.append(repaired)
        for proposal in proposals:
            require(proposal["schema"] == manifest["semantic_program_schema"], f"{cid}: proposal schema")
            require(isinstance(proposal["nodes"], list) and proposal["nodes"], f"{cid}: missing nodes")
            node_ids = [node["id"] for node in proposal["nodes"]]
            require(len(node_ids) == len(set(node_ids)), f"{cid}: duplicate node ID")
            require(all(output in node_ids for output in proposal["outputs"]), f"{cid}: unresolved output node")
            for node in proposal["nodes"]:
                require(isinstance(node["op"], str), f"{cid}: operation name missing")
                if node["op"] == "Ask":
                    require(node["obligation_ref"] in obligations, f"{cid}: unknown ASK obligation")
                    require(all(ref in host["query_constants"] or ref in records for ref in node["references"]), f"{cid}: unresolved ASK reference")
                if node["op"] == "Compare":
                    for side in ("left", "right"):
                        ref = node[side]
                        require(isinstance(ref, dict) and ref.get("project") == "object" and ref.get("node") in node_ids, f"{cid}: missing declared projection reference")
                # Deliberately do not interpret operations, assert truth, check graph
                # semantics, enforce policy, or reject intentionally invalid proposals.

    require(len(ids) == manifest["case_count"], "case count mismatch")
    require(files == {p.resolve() for p in (root / "cases").glob("*.json")}, "unlisted or missing case file")
    return len(ids), len(records)


if __name__ == "__main__":
    try:
        case_count, record_count = validate_pack()
        print(f"PASS: {case_count} fixture files, {record_count} records; JSON shape and declared references only.")
        print("No kernel execution, semantic validation, model run, or independent label validation was performed.")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
