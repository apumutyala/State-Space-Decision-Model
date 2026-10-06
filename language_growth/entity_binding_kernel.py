"""Small v0.1 checked-release adapter for resolved read/explain targets.

The canonical host audience view supplies binding records. Only the existing
kernel and release-validating renderer may produce text. Compare is unsupported.
Binding checks the requested operation/field within its supported grammar.
Query date versus evidence temporal scope is not validated by this adapter.
"""
from dataclasses import dataclass

from . import kernel, realize
from .entity_binding import overlay_request
from .parent_bridge import BridgeError, audience_state


@dataclass(frozen=True, slots=True)
class AdapterResult:
    status: str
    text: str | None = None
    program: dict | None = None
    reason: str | None = None


def _program(host, nodes, output):
    return {"schema": "ssdm.semantic_program.v0.1", "context_ref": host["context_ref"],
            "policy_ref": host["policy_ref"], "registry_ref": host["registry_ref"],
            "nodes": nodes, "outputs": [output], "style": "plain"}


def _check_render(program, host, catalog):
    checked = kernel.check_program(program, host, catalog)
    if checked.status != "accepted":
        return AdapterResult("kernel_rejected", program=program, reason=checked.reason)
    try:
        text = realize.render_checked(checked, host, catalog)
    except (TypeError, ValueError):
        return AdapterResult("unreleased", program=program, reason="release_failed")
    return AdapterResult("accepted", text, program)


def release_bound_answer(task_input, predicted, host, catalog):
    """Return renderer-produced text, or no text. No direct model/text fallback.

    Unresolved input releases nothing, even when host Ask obligations exist.
    This adapter has no independently trusted router and cannot authorize Ask.
    """
    try:
        request = {"state": audience_state(task_input, host, catalog)}
        bound = overlay_request(request, predicted)
    except (BridgeError, KeyError, TypeError, ValueError):
        return AdapterResult("unreleased", reason="invalid_input")
    if bound["disposition"] != "resolved":
        return AdapterResult("unreleased", reason="unresolved_binding")
    selection = bound["predicted"]
    if selection["kind"] not in ("read", "explain"):
        return AdapterResult("unreleased", reason="unsupported_operation")
    nodes = [{"id": "n1", "op": "ReadFact", "record_ref": selection["left"], "field": selection["field"]},
             {"id": "n2", "op": "Assert", "proposition": "n1"}]
    output = "n2"
    if selection["kind"] == "explain":
        nodes.append({"id": "n3", "op": "ExplainDerivation", "speech_act": "n2"})
        output = "n3"
    return _check_render(_program(host, nodes, output), host, catalog)
