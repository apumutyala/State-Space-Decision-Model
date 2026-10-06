"""Optional per-field provenance (``ssdm.provenance``): planned 0.2 extension, EXPERIMENTAL.

Contract 0.1 (PR #4 @ 787d64f) lists provenance as a planned optional 0.2 extension; D carries
it only behind ``Decider(enable_experimental_provenance=True)``. Without the flag a request
carrying ``ssdm.provenance`` is ``unsupported/provenance_not_enabled``.

Shape::

    "ssdm": {"provenance": {
      "schema": "ssdm.provenance.v0",
      "serializer": "<serialization_ref>",            # populated by the L bridge
      "default_trust": "data",                       # optional; only "data" is accepted
      "segments": {
        "/state": [{"start": 0, "end": 143, "trust": "instruction", "origin": "host_template",
                    "source_ref": "...", "audience": "public", "epistemic_class": "..."}, ...],
        "/questions/<qid>/instructions": [...],
        "/questions/<qid>/criteria/<option_id>": [...]  # choice id, score level index, or noul true/false
      }}}

Field paths (D's pinned experimental encoding, ``PATH_ENCODING``): RFC 6901 JSON Pointer into
the legacy request, with ``~`` escaped as ``~0`` and ``/`` as ``~1`` inside ids. This is
unambiguous for any question/option id (the earlier dotted form was not for ids containing
``.``). To be ratified with 0.2.

Rules (enforced here; any violation is ``error/invalid_provenance``, never a
silent fallback):

* Offsets are Unicode code-point indices into the exact string D scores.
  ``state`` must be a string when provenance is supplied. A field named in
  ``segments`` must exist and be a string (an absent instruction, or a ``null``
  description, cannot carry segments).
* Per field, segments are half-open ``[start, end)`` with ``start < end``.
  They are sorted, do not overlap, and tile ``[0, len)`` with no gaps. Empty
  strings take an empty list.
* ``trust`` is ``instruction`` or ``data``. ``origin`` is one of
  ``host_template``, ``record``, ``task_input``, ``program_prefix`` or
  ``other``. ``source_ref``, ``audience`` and ``epistemic_class`` are optional
  trace pass-through strings. Unknown keys are rejected.
* A field without an entry is one ``data`` segment covering the whole string.
  ``default_trust`` may only be ``data``: contract 0.1 says absent provenance
  defaults to data, and an instruction default needs a verified host-template
  binding, never the field's name. Instruction authority therefore comes only
  from explicitly declared spans. Question ids, option ids and the adapter
  scaffolding are not request fields; they are labelled ``data`` and
  ``template`` respectively.

Tokenisation (``decide.prompt``): each segment is encoded separately, so no BPE
merge crosses a trust boundary. Both instruction and data segments go through
the canonical untrusted encoder (``decide.untrusted``). Data segments must not
yield control tokens, and neither may instruction segments, because D adds its
own. ``trust`` only labels the tokens. A backend that consumes a per-token trust
mask (ISE/ASIDE) reports ``applied = true``; the stock readout does not use the
mask and reports ``applied = false``.
"""

from __future__ import annotations

from typing import Any

from .schema import RequestRejected, canonical_options

PROVENANCE_SCHEMA = "ssdm.provenance.v0"
TRUST = ("instruction", "data")
ORIGINS = ("host_template", "record", "task_input", "program_prefix", "other")
PATH_ENCODING = "rfc6901-json-pointer"
TOP_KEYS = {"schema", "serializer", "default_trust", "segments"}
SEG_REQUIRED = {"start", "end", "trust", "origin"}
SEG_OPTIONAL = {"source_ref", "audience", "epistemic_class"}
DEFAULT_TRUST = "data"


def _esc(token: str) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def state_path() -> str:
    return "/state"


def instructions_path(qid: str) -> str:
    return f"/questions/{_esc(qid)}/instructions"


def criteria_path(qid: str, option_id: str) -> str:
    return f"/questions/{_esc(qid)}/criteria/{_esc(option_id)}"


def _bad(msg: str) -> RequestRejected:
    return RequestRejected("error", "invalid_provenance", msg)


def field_strings(request: dict[str, Any]) -> dict[str, tuple[str, Any]]:
    """field name -> (kind, value) for every request field provenance may describe."""
    out: dict[str, tuple[str, Any]] = {state_path(): ("state", request["state"])}
    for qid, q in request["questions"].items():
        out[instructions_path(qid)] = ("instructions", q.get("instructions"))
        if q["type"] == "noul":
            crit = q.get("criteria") or {}
            for k in ("true", "false"):
                out[criteria_path(qid, k)] = ("criteria", crit.get(k))
        else:
            for oid, desc in canonical_options(q):
                out[criteria_path(qid, oid)] = ("criteria", desc)
    return out


def validate_provenance(request: dict[str, Any], prov: Any) -> dict[str, Any]:
    """Validate and return the normalised provenance (segments for every string field)."""
    if not isinstance(prov, dict):
        raise _bad("provenance must be an object")
    unknown = set(prov) - TOP_KEYS
    if unknown:
        raise _bad(f"unknown provenance keys {sorted(unknown)}")
    if prov.get("schema") != PROVENANCE_SCHEMA:
        raise _bad(f"provenance.schema must be {PROVENANCE_SCHEMA}")
    if "serializer" in prov and not isinstance(prov["serializer"], str):
        raise _bad("provenance.serializer must be a string")
    default_trust = prov.get("default_trust", DEFAULT_TRUST)
    if default_trust != DEFAULT_TRUST:
        raise _bad("provenance.default_trust must be data (instruction authority only via declared spans)")
    if not isinstance(request.get("state"), str):
        raise _bad("state must be a string when provenance is supplied")
    segments = prov.get("segments", {})
    if not isinstance(segments, dict):
        raise _bad("provenance.segments must be an object")
    fields = field_strings(request)
    for name in segments:
        if name not in fields:
            raise _bad(f"provenance names unknown field {name!r}")
        if not isinstance(fields[name][1], str):
            raise _bad(f"provenance field {name!r} is absent or not a string")

    normalised: dict[str, list[dict[str, Any]]] = {}
    for name, (kind, value) in fields.items():
        if not isinstance(value, str):
            continue
        n = len(value)
        if name not in segments:
            normalised[name] = (
                [{"start": 0, "end": n, "trust": DEFAULT_TRUST, "origin": "other", "defaulted": True}] if n else []
            )
            continue
        segs = segments[name]
        if not isinstance(segs, list):
            raise _bad(f"{name}: segments must be a list")
        pos = 0
        clean = []
        for s in segs:
            if not isinstance(s, dict):
                raise _bad(f"{name}: segment must be an object")
            keys = set(s)
            if not SEG_REQUIRED <= keys or keys - SEG_REQUIRED - SEG_OPTIONAL:
                raise _bad(f"{name}: segment keys must be {sorted(SEG_REQUIRED)} plus optional {sorted(SEG_OPTIONAL)}")
            st, en = s["start"], s["end"]
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in (st, en)):
                raise _bad(f"{name}: start/end must be integers")
            if st != pos:
                raise _bad(f"{name}: segments must tile the string (gap/overlap at {pos}, got start {st})")
            if not st < en <= n:
                raise _bad(f"{name}: need start < end <= len ({n})")
            if s["trust"] not in TRUST:
                raise _bad(f"{name}: trust must be instruction or data")
            if s["origin"] not in ORIGINS:
                raise _bad(f"{name}: origin must be one of {ORIGINS}")
            for k in SEG_OPTIONAL & keys:
                if not isinstance(s[k], str):
                    raise _bad(f"{name}: {k} must be a string")
            clean.append(dict(s))
            pos = en
        if pos != n:
            raise _bad(f"{name}: segments end at {pos}, string length is {n}")
        normalised[name] = clean
    return {
        "schema": PROVENANCE_SCHEMA,
        "path_encoding": PATH_ENCODING,
        "serializer": prov.get("serializer"),
        "default_trust": default_trust,
        "segments": normalised,
    }


def split(value: str, segs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """[(substring, trust)] in order; tiles ``value``."""
    return [(value[s["start"] : s["end"]], s["trust"]) for s in segs]
