"""Historical electricity lifecycle knowledge; no model-authored factual values.

The trusted host imports a typed table. Model proposals contain row references
only. ReadTechnology and LowestMedian compile to existing ReadFact, Compare and
Assert operations; there is no second arithmetic or evidence-admission path.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..kernel import check_program, validate_release
from ..types import CheckResult, Fact

MODULE_SCHEMA = "ssdm.energy_knowledge.v0.1"
PROGRAM_SCHEMA = "ssdm.energy_program.v0.1"
MODULE_REF = "ipcc-ar5-2014-lifecycle-subset-v1"
SCOPE = "IPCC-AR5-2014/Table-A.III.2"
UNIT = "gCO2eq/kWh"
BOUNDARY = "lifecycle_including_albedo_effect"
UNCERTAINTY = "reported_minimum_median_maximum_not_confidence_interval"
SOURCE_URL = "https://archive.ipcc.ch/pdf/assessment-report/ar5/wg3/ipcc_wg3_ar5_annex-iii.pdf"
FIELDS = ("lifecycle_minimum", "lifecycle_median", "lifecycle_maximum")
LOWEST_TASK = ("Among the technologies in this module, which has the lowest median "
               "lifecycle emissions in IPCC AR5 (2014)?")
DATA_PATH = Path(__file__).with_name("data") / "ipcc_ar5_lifecycle.json"


def _module(document: Any) -> dict[str, Any]:
    """Validate host-owned table topology and fixed historical interpretation."""
    keys = {"schema", "module_ref", "assessment_year", "temporal_scope", "unit",
            "boundary", "uncertainty", "source", "rows"}
    if not isinstance(document, dict) or set(document) != keys:
        raise ValueError("invalid_module")
    expected = {"schema": MODULE_SCHEMA, "module_ref": MODULE_REF,
                "assessment_year": 2014, "temporal_scope": SCOPE, "unit": UNIT,
                "boundary": BOUNDARY, "uncertainty": UNCERTAINTY}
    if any(document[key] != value for key, value in expected.items()):
        raise ValueError("unsupported_module_scope")
    source = document["source"]
    if source != {"source_ref": "ipcc-ar5-wgiii-annex-iii",
                  "title": "IPCC AR5 WGIII (2014), Annex III",
                  "url": SOURCE_URL, "table": "Table A.III.2"}:
        raise ValueError("unsupported_source")
    rows = document["rows"]
    if not isinstance(rows, list) or not 2 <= len(rows) <= 16:
        raise ValueError("invalid_table")
    seen_refs, seen_names = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"record_ref", "technology", *FIELDS}:
            raise ValueError("invalid_row")
        for key in ("record_ref", "technology"):
            value = row[key]
            if (not isinstance(value, str) or not value or len(value) > 128
                    or any(ord(c) < 32 for c in value)):
                raise ValueError("invalid_row")
        if row["record_ref"] in seen_refs or row["technology"].casefold() in seen_names:
            raise ValueError("ambiguous_row")
        seen_refs.add(row["record_ref"])
        seen_names.add(row["technology"].casefold())
        # Keep literals exact and modestly bounded. No floats, NaN, exponents,
        # bool-as-integer values or context-dependent Decimal arithmetic.
        values = []
        for field in FIELDS:
            literal = row[field]
            if (not isinstance(literal, str) or len(literal) > 24
                    or not literal.isascii() or not literal.replace(".", "", 1).isdigit()
                    or literal.startswith(".") or literal.endswith(".")):
                raise ValueError("invalid_numeric_literal")
            values.append(Decimal(literal))
        if not 0 <= values[0] <= values[1] <= values[2]:
            raise ValueError("invalid_reported_range")
    return document


def load_energy_module(path: str | Path = DATA_PATH) -> dict[str, Any]:
    """Load a host-owned extraction, not a model proposal or live-world feed."""
    return _module(json.loads(Path(path).read_text(encoding="utf-8")))


def energy_environment(document: dict[str, Any]) -> tuple[dict, dict]:
    """Host import: give the table ordinary kernel evidence and authority.

    The caller owns this catalog and its current policy. Import is a trust
    decision: schema validation does not authenticate a source transcription.
    """
    module = _module(document)
    source_ref = module["source"]["source_ref"]
    records = []
    for row in module["rows"]:
        records.append({
            "record_ref": row["record_ref"], "version": 1,
            "entity_ref": row["technology"], "source_ref": source_ref,
            "audience": "public", "epistemic_class": "authoritative_structured_premise",
            "freshness": "valid_in_declared_snapshots", "conflicts": [],
            "fields": {field: {
                "value": row[field], "value_type": "decimal", "unit": UNIT,
                "temporal_scope": SCOPE,
                "quantity_kind": f"{BOUNDARY}_{field}",
            } for field in FIELDS},
        })
    host = {"context_ref": MODULE_REF, "policy_ref": "energy-public-v1",
            "registry_ref": "energy-grammar-v1", "audience": "public",
            "query_constants": {}, "unmet_obligations": [],
            "limits": {"max_program_nodes": 64}, "provider": {"state_handle": None}}
    catalog = {
        "sources": {source_ref: {**module["source"], "premise_admission": "accepted",
                                 "assessment_year": 2014, "boundary": BOUNDARY,
                                 "uncertainty": UNCERTAINTY}},
        "records": records,
        "snapshots": {MODULE_REF: {"audience": "public",
                                   "record_refs": [row["record_ref"] for row in module["rows"]]}},
        "policies": {"energy-public-v1": {"status": "current", "audience": "public",
                                          "forbidden_audiences": ["private"]}},
        "registries": {"energy-grammar-v1": {
            "allowed_ops": ["ReadFact", "Compare", "Assert"],
            "registered_fields": list(FIELDS),
            "field_predicates": {field: "Has" + "".join(p.title() for p in field.split("_"))
                                 for field in FIELDS},
            "comparators": ["less_than"], "ask_categories": [], "style_ids": ["plain"],
        }},
    }
    return host, catalog


def _compile(proposal: dict, module: dict, host: dict, catalog: dict) -> dict:
    if (not isinstance(proposal, dict)
            or set(proposal) != {"schema", "module_ref", "op", "record_ref"}
            or proposal["schema"] != PROGRAM_SCHEMA or proposal["module_ref"] != MODULE_REF
            or proposal["op"] not in {"ReadTechnology", "LowestMedian"}):
        raise ValueError("invalid_proposal")
    rows = {row["record_ref"]: row for row in module["rows"]}
    selected = proposal["record_ref"]
    if not isinstance(selected, str) or selected not in rows or host.get("audience") != "public":
        raise ValueError("unavailable")
    # The module declares the complete comparison domain. Require every row
    # in the active public snapshot; no silently omitted competitors.
    if set(catalog["snapshots"][host["context_ref"]]["record_refs"]) != set(rows):
        raise ValueError("incomplete_or_expanded_domain")
    expected_host, expected_catalog = energy_environment(module)
    # Bind source interpretation and all numeric data to this exact extraction.
    # Policies/freshness/conflicts remain live host inputs, checked by the kernel.
    by_ref = {record["record_ref"]: record for record in catalog["records"]}
    source_ref = module["source"]["source_ref"]
    if catalog["sources"][source_ref] != expected_catalog["sources"][source_ref]:
        raise ValueError("source_changed")
    for expected in expected_catalog["records"]:
        current = by_ref[expected["record_ref"]]
        if any(current[key] != expected[key] for key in
               ("entity_ref", "source_ref", "fields")):
            raise ValueError("module_evidence_mismatch")
    if (host["registry_ref"] != expected_host["registry_ref"]
            or catalog["registries"][host["registry_ref"]]
            != expected_catalog["registries"][expected_host["registry_ref"]]):
        raise ValueError("unsupported_registry")
    nodes, outputs = [], []

    def add(op: str, **arguments: Any) -> str:
        node_id = f"n{len(nodes) + 1}"
        nodes.append({"id": node_id, "op": op, **arguments})
        return node_id

    # All three reported statistics reach the final answer as checked Facts.
    median_nodes = {}
    for field in FIELDS:
        fact = add("ReadFact", record_ref=selected, field=field)
        outputs.append(add("Assert", proposition=fact))
        if field == "lifecycle_median":
            median_nodes[selected] = fact
    if proposal["op"] == "LowestMedian":
        for ref in rows:
            if ref == selected:
                continue
            other = add("ReadFact", record_ref=ref, field="lifecycle_median")
            comparison = add("Compare", left={"node": median_nodes[selected], "project": "object"},
                             right={"node": other, "project": "object"}, relation="less_than")
            outputs.append(add("Assert", proposition=comparison))
    return {"schema": "ssdm.semantic_program.v0.1",
            **{key: host[key] for key in ("context_ref", "policy_ref", "registry_ref")},
            "nodes": nodes, "outputs": outputs, "style": "plain"}


def check_energy_program(proposal: dict, document: dict, host: dict, catalog: dict) -> CheckResult:
    """Check the proposed row, table completeness and every existing semantic op.

    LowestMedian establishes a unique minimum over this module, not an optimum
    over unlisted technologies. A tie or missing competitor cannot pass.
    """
    try:
        return check_program(_compile(proposal, _module(document), host, catalog), host, catalog)
    except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
        return CheckResult("rejected", "unsupported_assertion")


def public_energy_answer(task_input: str, proposal: dict, document: dict,
                         host: dict, catalog: dict) -> dict[str, str]:
    """Release only either exact historical table question or generic abstention.

    The model supplies the operation/row proposal. It supplies no answer values,
    prose, comparison relation, competitor list or authority metadata.
    """
    try:
        module = _module(document)
        checked = check_energy_program(proposal, module, host, catalog)
        if checked.status != "accepted":
            return {"status": "unavailable"}
        row = next(row for row in module["rows"] if row["record_ref"] == proposal["record_ref"])
        expected_task = (LOWEST_TASK if proposal["op"] == "LowestMedian" else
                         f"In IPCC AR5 (2014), what lifecycle emissions are reported for {row['technology']}?")
        if task_input != expected_task:
            return {"status": "unavailable"}
        released = validate_release(checked, host, catalog)
        # Rebuild the domain projection immediately before realization as well.
        current = check_energy_program(proposal, module, host, catalog)
        if current.status != "accepted" or current.to_dict() != released.to_dict():
            return {"status": "unavailable"}
        facts = [act.content for act in released.outputs[:3]]
        if any(not isinstance(fact, Fact) for fact in facts):
            return {"status": "unavailable"}
        minimum, median, maximum = (str(fact.object) for fact in facts)
        technology = json.dumps(facts[0].subject, ensure_ascii=True)
        domain = ", ".join(json.dumps(item["technology"], ensure_ascii=True) for item in module["rows"])
        finding = (f"{technology} has the unique lowest reported median among the listed technologies "
                   f"({domain}): " if proposal["op"] == "LowestMedian"
                   else f"{technology}: reported median ")
        text = (f"IPCC AR5 (2014), Table A.III.2; lifecycle including albedo effect. "
                f"{finding}{median} {UNIT}. Reported minimum {minimum}, maximum {maximum} {UNIT}; "
                "these are literature summary statistics, not a confidence interval or a current plant estimate. "
                f"Source: {SOURCE_URL}")
        return {"status": "ok", "text": text}
    except (ValueError, TypeError, KeyError, AttributeError, StopIteration, ArithmeticError):
        return {"status": "unavailable"}


def main() -> None:
    """Run the single authored example; no inference provider or service is used."""
    module = load_energy_module()
    host, catalog = energy_environment(module)
    example = json.loads(DATA_PATH.with_name("example.json").read_text(encoding="utf-8"))
    print(json.dumps(public_energy_answer(example["task"], example["proposal"], module, host, catalog),
                     indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
