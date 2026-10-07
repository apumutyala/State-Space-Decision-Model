"""Detached q_end cache and protocol-only composition. No checkpoint loader."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def initial_operation(request):
    """Routing uses protocol shape only, never task text, criteria text or gold."""
    questions = request.get("questions")
    if not isinstance(questions, dict) or set(questions) != {"slot-001-operation"}:
        return False
    question = questions["slot-001-operation"]
    if not isinstance(question, dict) or question.get("type") != "choice":
        return False
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        return False
    try:
        state = json.loads(request["state"])
    except (KeyError, TypeError, ValueError):
        return False
    from language_growth.parent_bridge import SERIALIZATION_REF
    return (isinstance(state, dict) and state.get("schema") == SERIALIZATION_REF
            and state.get("accepted_partial_program") == []
            and state.get("proposed_slot_choices") == [])


def operation_records(data):
    """Verify permitted evidence, but discard every teacher-forced later slot."""
    from language_growth.training_records import export_training_records
    from language_growth.parent_bridge import audience_state
    result = {}
    for split, count in (("development_train", 24), ("development_validation", 12)):
        examples = data[split]
        if len(examples) != count:
            raise ValueError("fixed world count required")
        records = [r for r in export_training_records(examples, split=split)
                   if set(r["labels"]) == {"slot-001-operation"}]
        if len(records) != count or len({r["id"] for r in records}) != count:
            raise ValueError("exactly one initial operation per world required")
        by_id = {e["task_id"]: e for e in examples}
        for record in records:
            request = record["request"]
            if not initial_operation(request):
                raise ValueError("operation prefix/shape mismatch")
            state = json.loads(request["state"])
            training = by_id[record["id"]]["training_only"]
            permitted = json.loads(audience_state(state["task_input"], training["host"], training["catalog"]))
            permitted["proposed_slot_choices"] = []
            if state != permitted or set(request) != {"model", "state", "questions", "ssdm"}:
                raise ValueError("gold/prefix/evidence leakage")
            if any(k in request["questions"]["slot-001-operation"] for k in
                   ("label", "gold", "target", "axis_target", "training_only")):
                raise ValueError("label in question")
        result[split] = records
    return result


def row_identity(row_id, row, *, parent, encoder, template):
    """Allowlisted identity: Item.label/axis_target and arbitrary meta excluded."""
    from heads.layout import LAYOUT_VERSION
    fields = ("qid", "qtype", "option_ids", "ids", "state_span", "field_span",
              "option_spans", "option_seqs", "continuations", "layout")
    return {"row_id": row_id, "row": {k: deepcopy(getattr(row, k)) for k in fields},
            "layout_version": LAYOUT_VERSION, "template": deepcopy(template),
            "parent": deepcopy(parent), "encoder": deepcopy(encoder)}


def capture(model, item, *, parent, encoder, template):
    import torch
    from heads.features import build_head_input
    if item.row.qid != "slot-001-operation" or item.row.qtype != "choice":
        raise ValueError("cache only initial operation rows")
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("cache requires frozen backbone")
    model.eval()
    with torch.no_grad():
        x = build_head_input(model.backbone, [item.row], need_fork=False, pad_id=model.pad_id)
    identity = row_identity(item.id, item.row, parent=parent, encoder=encoder, template=template)
    entry = {"key": digest(identity), "identity": identity,
             "q_end": x.q_end.detach().cpu().float().clone(),
             "e": x.e.detach().cpu().float().clone(),
             "option_mask": x.option_mask.detach().cpu().bool().clone(),
             "ks": list(x.ks), "qtypes": [item.row.qtype]}
    validate_entry(entry)
    return entry


def validate_entry(entry):
    import torch
    if set(entry) != {"key", "identity", "q_end", "e", "option_mask", "ks", "qtypes"}:
        raise ValueError("cache fields are allowlisted")
    if digest(entry["identity"]) != entry["key"]:
        raise ValueError("cache identity drift")
    q, e, mask = entry["q_end"], entry["e"], entry["option_mask"]
    if (q.ndim != 2 or q.shape[0] != 1 or e.ndim != 3 or e.shape[0] != 1
            or e.shape[2] != q.shape[1] or mask.shape != e.shape[:2]
            or entry["ks"] != [e.shape[1]] or entry["qtypes"] != ["choice"]
            or mask.dtype != torch.bool or not mask.all()):
        raise ValueError("cache shape/type drift")
    for tensor in (q, e):
        if (tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                or tensor.requires_grad or not torch.isfinite(tensor).all()):
            raise ValueError("cache must be detached finite CPU FP32")
    if mask.device.type != "cpu":
        raise ValueError("CPU mask required")


def cached_forward(model, entries):
    import torch
    from heads.features import HeadInput
    for entry in entries:
        validate_entry(entry)
    q = torch.cat([e["q_end"] for e in entries])
    ks = [e["ks"][0] for e in entries]
    kmax = max(ks)
    opts = torch.cat([torch.nn.functional.pad(e["e"], (0, 0, 0, kmax-e["ks"][0])) for e in entries])
    mask = torch.cat([torch.nn.functional.pad(e["option_mask"], (0, kmax-e["ks"][0])) for e in entries])
    # The reused q_end head reads only q_end/e/option_mask; dummy unused fields
    # are constructed here, never cached or derived from labels.
    h = q[:, None, :]
    empty = torch.zeros((len(entries), 1), dtype=torch.bool)
    x = HeadInput(h=h, token_mask=empty, evidence_mask=empty, q_pre=q*0,
                  q_end=q, e=opts, option_mask=mask, ks=ks)
    out = model.head(x)
    return model.typed(out.logits, [e["qtypes"][0] for e in entries], ks, opts)


class OperationOnlyBackend:
    """One outer Decider keeps a single composite revision for every call."""
    name = "operation_only_stock_composition"

    def __init__(self, learned, stock):
        self.learned, self.stock = learned, stock

    def revision(self):
        identity = {"backend": self.name, "route": "initial-operation-empty-prefix-v1",
                    "learned": self.learned.revision(), "stock": self.stock.revision()}
        return {**identity, "model_id": identity["stock"]["model_id"],
                "revision_id": self.name + "@" + digest(identity)}

    def route(self, request):
        return self.learned if initial_operation(request) else self.stock

    def plan(self, request):
        return self.route(request).plan(request)

    def score(self, request, check_deadline):
        return self.route(request).score(request, check_deadline)
