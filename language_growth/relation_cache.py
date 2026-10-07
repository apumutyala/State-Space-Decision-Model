"""L1r1 detached relation features. No pretrained loader or experiment runner.

Previous operation/operand choices are explicit diagnostic inputs. The current
relation target, Item labels and arbitrary metadata never enter feature identity.
Caller provenance is consistency-checked, not authenticated. A session hashes
the frozen parent once and checks ordinary tensor mutation before each use;
``verify_parent`` repeats the full byte check at the experiment/checkpoint boundary.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .relation_router import RELATION_QID, RelationOnlyBackend, relation_request


SCHEMA = "ssdm.relation-feature-cache.v1"
ROW_FIELDS = ("qid", "qtype", "option_ids", "ids", "state_span", "field_span",
              "option_spans", "option_seqs", "continuations", "layout")
ENTRY_FIELDS = {"schema", "key", "identity", "q_end", "e", "option_mask", "ks",
                "qtypes", "features_sha256"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _row(row):
    return {name: deepcopy(getattr(row, name)) for name in ROW_FIELDS}


def _tensor_digest(tensors):
    h = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        h.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _model_check(model):
    import torch
    from heads.heads import QEndSemanticAxes

    if type(model.head) is not QEndSemanticAxes:
        raise ValueError("relation cache requires the q_end semantic head")
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("relation cache requires a frozen parent")
    for value in list(model.parameters()) + list(model.buffers()):
        if (value.device.type != "cpu"
                or (value.is_floating_point() and value.dtype != torch.float32)):
            raise ValueError("relation cache requires CPU FP32 model tensors")


def fit_items(data, encode, *, guard, template="jrt-rwkv-g1-v1", layout="ordered"):
    """Use the pinned full-world guard BEFORE selecting 576 relation Items.

    This delegates to the reviewed consumer, which converts all 3,072 fit rows
    and checks all runtime token lengths. Calibration/validation never optimize.
    """
    from .l1_relation_data import L1RelationData

    if type(data) is not L1RelationData or layout != "ordered":
        raise ValueError("pinned L1 relation data and ordered layout required")
    items, worlds = data.fit_items(encode, guard=guard, template=template, layout=layout)
    if (len(items) != 576 or len(worlds) != 576
            or set(worlds) != {item.id for item in items}
            or set(worlds.values()) != data.world_data.fit_world_ids):
        raise ValueError("complete 576-row fit relation partition required")
    return items, worlds


class RelationFeatureCache:
    """One fixed parent/encoder/template context; head parameters may train.

    The merged ``RelationOnlyBackend`` remains the live serving seam. Cached
    features train the same native head; they are never a task-answer router.
    """
    def __init__(self, model, encode, *, parent, template="jrt-rwkv-g1-v1", layout="ordered"):
        from decide.prompt import TEMPLATES
        from heads.layout import LAYOUT_VERSION
        from train.lineage import backbone_identity

        _model_check(model)
        if layout != "ordered" or not callable(getattr(encode, "identity", None)):
            raise ValueError("identified encoder and ordered layout required")
        encoder = encode.identity()
        if not isinstance(encoder, dict) or not encoder or not isinstance(parent, dict) or not parent:
            raise ValueError("nonempty parent and encoder identities required")
        native = getattr(model.backbone, "provenance", {}).get("track_l_parent")
        if native is not None and native != parent:
            raise ValueError("parent provenance does not match native backbone")
        self.model, self.encode = model, encode
        self.template = TEMPLATES[template] if isinstance(template, str) else template
        self._identity = deepcopy({"schema": SCHEMA, "parent": parent,
            "backbone": backbone_identity(model.backbone, full=True), "encoder": encoder,
            # The shared lineage digest excludes LoRA tensors and buffers. Bind
            # every resident state tensor too, including frozen adapters.
            "parent_state_sha256": _tensor_digest(model.backbone.state_dict()),
            "template": self.template.identity(), "layout": layout,
            "layout_version": LAYOUT_VERSION, "pad_id": model.pad_id})
        digest(self._identity)
        self._versions = self._tensor_versions()
        self._metadata = self._parent_metadata()

    def _parent_metadata(self):
        from train.lineage import _config_dict

        bb = self.model.backbone
        return deepcopy({"class": type(bb.model).__name__, "spec": bb.spec.name,
            "capture": bb.capture, "layer": bb.layer, "config": _config_dict(bb.model),
            "provenance": getattr(bb, "provenance", {})})

    def _tensor_versions(self):
        return tuple((name, id(tensor), tensor._version) for name, tensor in
                     list(self.model.backbone.named_parameters()) + list(self.model.backbone.named_buffers()))

    def _check(self):
        _model_check(self.model)
        if (self._tensor_versions() != self._versions
                or self._parent_metadata() != self._metadata
                or self.encode.identity() != self._identity["encoder"]
                or self.template.identity() != self._identity["template"]
                or self.model.pad_id != self._identity["pad_id"]):
            raise ValueError("relation feature context drift")

    def identity(self):
        self._check()
        return deepcopy(self._identity)

    def verify_parent(self):
        from train.lineage import backbone_identity

        self._check()
        if (backbone_identity(self.model.backbone, full=True) != self._identity["backbone"]
                or _tensor_digest(self.model.backbone.state_dict()) != self._identity["parent_state_sha256"]):
            raise ValueError("frozen parent bytes or provenance changed")

    def capture(self, item, request):
        import torch
        from heads.features import build_head_input
        from heads.layout import build_row

        self._check()
        if not relation_request(request):
            raise ValueError("exact relation request and prior choices required")
        if not isinstance(item.id, str) or not item.id:
            raise ValueError("nonempty cache row identity required")
        expected = build_row(request, RELATION_QID, self.encode,
                             template=self.template, layout="ordered")
        if _row(expected) != _row(item.row):
            raise ValueError("request/LayoutRow mismatch")
        lengths = [len(expected.ids), *(len(expected.ids) + len(c) for c in expected.continuations)]
        if max(lengths) > 2048:
            raise ValueError("relation context exceeds 2048 tokens; no truncation")
        identity = {"context": self.identity(), "row_id": item.id,
                    "request": deepcopy(request), "row": _row(expected),
                    "prior_choices": json.loads(request["state"])["proposed_slot_choices"]}
        self.model.eval()
        with torch.no_grad():
            x = build_head_input(self.model.backbone, [expected], need_fork=False,
                                 pad_id=self.model.pad_id)
        entry = {"schema": SCHEMA, "key": digest(identity), "identity": identity,
                 "q_end": x.q_end.detach().cpu().float().clone(),
                 "e": x.e.detach().cpu().float().clone(),
                 "option_mask": x.option_mask.detach().cpu().bool().clone(),
                 "ks": list(x.ks), "qtypes": ["choice"]}
        entry["features_sha256"] = _tensor_digest({k: entry[k] for k in ("q_end", "e", "option_mask")})
        validate_entry(entry, context=self.identity())
        self._check()
        return entry

    def forward(self, entries):
        """Differentiable head-only log probabilities; variable K is padded."""
        from .operation_cache import cached_forward

        self._check()
        if not entries:
            raise ValueError("nonempty relation cache batch required")
        context = self.identity()
        for entry in entries:
            validate_entry(entry, context=context)
        # Reuse the existing q_end/e tensor math without changing its operation
        # capture or runner. Its generic validator accepts this narrowed view.
        keys = {"key", "identity", "q_end", "e", "option_mask", "ks", "qtypes"}
        return cached_forward(self.model, [{k: entry[k] for k in keys} for entry in entries])


def validate_entry(entry, *, context):
    import torch
    from .operation_cache import validate_entry as validate_tensors

    if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS or entry["schema"] != SCHEMA:
        raise ValueError("relation cache fields are allowlisted")
    identity = entry["identity"]
    if (not isinstance(identity, dict)
            or set(identity) != {"context", "row_id", "request", "row", "prior_choices"}
            or not isinstance(identity["row_id"], str) or not identity["row_id"]
            or identity["context"] != context or not relation_request(identity["request"])
            or identity["prior_choices"] != json.loads(identity["request"]["state"])["proposed_slot_choices"]
            or not isinstance(identity["row"], dict)
            or set(identity["row"]) != set(ROW_FIELDS)
            or identity["row"]["qid"] != RELATION_QID
            or identity["row"]["qtype"] != "choice" or identity["row"]["layout"] != "ordered"
            or not isinstance(entry["ks"], list) or len(entry["ks"]) != 1
            or type(entry["ks"][0]) is not int
            or identity["row"]["option_ids"] != sorted(identity["request"]["questions"][RELATION_QID]["criteria"])
            or len(identity["row"]["option_ids"]) != entry["ks"][0]
            or any(not isinstance(entry[k], torch.Tensor) for k in ("q_end", "e", "option_mask"))):
        raise ValueError("relation cache identity/context mismatch")
    keys = {"key", "identity", "q_end", "e", "option_mask", "ks", "qtypes"}
    validate_tensors({k: entry[k] for k in keys})
    if _tensor_digest({k: entry[k] for k in ("q_end", "e", "option_mask")}) != entry["features_sha256"]:
        raise ValueError("relation feature bytes changed")


def training_lineage(entries, targets, *, context):
    """Supervision is bound only here, separately from feature/cache identity."""
    if (not entries or len(entries) != len(targets)
            or len({e["key"] for e in entries}) != len(entries)
            or len({e["identity"]["row_id"] for e in entries}) != len(entries)):
        raise ValueError("one target per unique relation cache row required")
    for entry, target in zip(entries, targets):
        validate_entry(entry, context=context)
        if type(target) is not int or not 0 <= target < entry["ks"][0]:
            raise ValueError("target outside relation options")
    return {"schema": "ssdm.relation-cache-training.v1", "context": deepcopy(context),
            "cache_keys": [e["key"] for e in entries],
            "features_sha256": [e["features_sha256"] for e in entries],
            "targets_sha256": digest([{"row_id": e["identity"]["row_id"], "target": target}
                                      for e, target in zip(entries, targets)])}


def _train_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(("head.", "typed."))}


def _optimizer_check(optimizer, model):
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    expected = list(model.head_parameters())
    if [id(p) for p in actual] != [id(p) for p in expected]:
        raise ValueError("optimizer must own exactly the ordered head parameters")


def save_checkpoint(path, cache, optimizer, *, lineage, step):
    import torch

    cache.verify_parent()
    _optimizer_check(optimizer, cache.model)
    if lineage.get("context") != cache.identity() or type(step) is not int or step < 0:
        raise ValueError("checkpoint lineage/step mismatch")
    torch.save({"schema": "ssdm.relation-cache-head.v1", "lineage": deepcopy(lineage),
                "step": step, "trainable": _train_state(cache.model),
                "optimizer_class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                "optimizer": optimizer.state_dict()}, path)


def restore_checkpoint(path, cache, optimizer, *, lineage, step):
    import torch

    cache.verify_parent()
    _optimizer_check(optimizer, cache.model)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (set(state) != {"schema", "lineage", "step", "trainable", "optimizer_class", "optimizer"}
            or state["schema"] != "ssdm.relation-cache-head.v1" or state["lineage"] != lineage
            or lineage.get("context") != cache.identity() or type(state["step"]) is not int
            or state["step"] != step or type(step) is not int or step < 0
            or state["optimizer_class"] != type(optimizer).__module__ + "." + type(optimizer).__qualname__):
        raise ValueError("checkpoint lineage/step/optimizer mismatch")
    expected = _train_state(cache.model)
    if (set(state["trainable"]) != set(expected) or any(
            not isinstance(state["trainable"][k], torch.Tensor)
            or state["trainable"][k].shape != value.shape
            or state["trainable"][k].dtype != value.dtype
            or not torch.isfinite(state["trainable"][k]).all() for k, value in expected.items())):
        raise ValueError("checkpoint head tensor mismatch")
    # Validate optimizer loading on an isolated copy before mutating the model.
    probe = deepcopy(optimizer)
    probe.load_state_dict(deepcopy(state["optimizer"]))
    for parameter, values in probe.state.items():
        for name, value in values.items():
            if isinstance(value, torch.Tensor) and (
                    not torch.isfinite(value).all()
                    or (name != "step" and value.shape != parameter.shape)
                    or (name == "step" and value.numel() != 1)):
                raise ValueError("checkpoint optimizer tensor mismatch")
    cache.model.load_state_dict(state["trainable"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    cache.verify_parent()
