"""L1r1 CPU mechanism proof: random native RWKV only, never pretrained weights."""
from copy import deepcopy
from dataclasses import replace
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from language_growth import relation_cache as cache
from language_growth.l1_relation_data import load_l1_relation_data
from language_growth.parent_bridge import SERIALIZATION_REF
from language_growth.relation_router import RELATION_QID, RelationOnlyBackend


@pytest.fixture(autouse=True)
def cpu_threads():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert not torch.cuda.is_available()
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


PARENT = {"model_id": "random-native-rwkv", "snapshot": "tiny-seed-31"}


class Encoder:
    """Short synthetic sequences, not the campaign tokenizer or token census."""
    def __call__(self, text, trusted=True):
        return [4 + b for b in text.encode()[::16]]

    def identity(self):
        return {**PARENT, "kind": "test-strided-bytes"}


def request(k=3, prefix="first"):
    state = {"schema": SERIALIZATION_REF, "task_input": "Opaque task", "audience": "public",
        "context_ref": "ctx", "policy_ref": "pol", "registry_ref": "reg", "records": [],
        "registry": {"registered_fields": [], "comparators": [], "ask_categories": []},
        "query_constants": {}, "unmet_obligations": [], "accepted_partial_program": [],
        "proposed_slot_choices": [{"slot": slot, "option_id": str(i), "description": prefix + slot}
                                  for i, slot in enumerate(("operation", "left", "right"))]}
    return {"model": PARENT["model_id"], "state": json.dumps(state, sort_keys=True),
            "questions": {RELATION_QID: {"type": "choice", "instructions": "Opaque relation",
                "criteria": {str(i): "Opaque " + str(i) for i in reversed(range(k))}}}}


def item(req, encode, row_id="tiny/slot-004-relation", template="jrt-rwkv-g1-v1"):
    from heads.layout import build_row
    return SimpleNamespace(id=row_id, row=build_row(req, RELATION_QID, encode, template=template),
                           label=0, axis_target=None, metadata={})


def native():
    from heads.backbone import tiny_hf_backbone
    from language_growth.rwkv_decision_head import rwkv_decision_backend

    backend = rwkv_decision_backend(tiny_hf_backbone("rwkv7", seed=31, hidden=16).model,
        Encoder(), source_identity=PARENT, head="qend_axes", seed=71)
    features = cache.RelationFeatureCache(backend.model, backend.encode, parent=PARENT)
    return backend, features


def test_native_cache_live_parity_with_variable_options_and_prefixes():
    backend, features = native()
    entries, rows = [], []
    for i, (k, prefix) in enumerate(((1, "first"), (2, "changed"), (4, "other-long-prefix"))):
        req = request(k, prefix)
        it = item(req, backend.encode, f"tiny-{i}/{RELATION_QID}")
        entry = features.capture(it, req)
        entries.append(entry)
        rows.append(it.row)
        with torch.no_grad():
            torch.testing.assert_close(features.forward([entry]), backend.model([it.row]).log_probs,
                                       rtol=1e-5, atol=1e-6)
        assert entry["identity"]["row"]["option_ids"] == [str(j) for j in range(k)]
        assert entry["identity"]["prior_choices"] == json.loads(req["state"])["proposed_slot_choices"]
    with torch.no_grad():
        torch.testing.assert_close(features.forward(entries), backend.model(rows).log_probs,
                                   rtol=1e-5, atol=1e-6)
    assert torch.isneginf(features.forward(entries)[0, 1:]).all()
    features.verify_parent()


def test_cache_identity_excludes_current_labels_and_arbitrary_metadata():
    backend, features = native()
    req = request()
    it = item(req, backend.encode)
    first = features.capture(it, req)
    it.label = 2
    it.axis_target = (99.,) * 6
    it.metadata = {"world_id": "SECRET", "label": "SECRET"}
    it.row.meta.update({"world_id": "SECRET", "label": "SECRET", "gold": "SECRET"})
    second = features.capture(it, req)
    assert first["key"] == second["key"]
    assert first["features_sha256"] == second["features_sha256"]
    assert "SECRET" not in json.dumps(second["identity"])
    assert set(second) == cache.ENTRY_FIELDS
    for name in ("q_end", "e"):
        assert second[name].dtype == torch.float32 and second[name].device.type == "cpu"
        assert not second[name].requires_grad and second[name].grad_fn is None
    assert cache.training_lineage([first], [0], context=features.identity()) != cache.training_lineage(
        [first], [2], context=features.identity())
    assert first["key"] == second["key"]
    with pytest.raises(ValueError, match="allowlisted"):
        cache.validate_entry({**second, "label": 2}, context=features.identity())


@pytest.mark.parametrize("change", ["prefix", "options", "tokens", "order", "qid", "layout"])
def test_request_row_mismatch_refuses_before_feature_extraction(change, monkeypatch):
    backend, features = native()
    req = request()
    it = item(req, backend.encode)
    if change == "prefix":
        state = json.loads(req["state"])
        state["proposed_slot_choices"][1]["description"] = "different" * 20
        req["state"] = json.dumps(state)
    elif change == "options":
        req["questions"][RELATION_QID]["criteria"]["0"] = "different" * 20
    elif change == "tokens": it.row.ids[0] += 1
    elif change == "order": it.row.option_ids.reverse()
    elif change == "qid": it.row.qid = "slot-001-operation"
    else: it.row.layout = "symmetric"
    monkeypatch.setattr("heads.features.build_head_input", lambda *a, **k: pytest.fail("must reject before forward"))
    with pytest.raises(ValueError, match="mismatch"):
        features.capture(it, req)


def test_context_feature_drift_and_invalid_head_refused():
    backend, features = native()
    req = request()
    entry = features.capture(item(req, backend.encode), req)
    changed = deepcopy(entry)
    changed["e"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="bytes changed"):
        features.forward([changed])
    changed = deepcopy(entry)
    changed["identity"]["context"]["encoder"]["kind"] = "different"
    changed["key"] = cache.digest(changed["identity"])
    with pytest.raises(ValueError, match="context"):
        features.forward([changed])
    with pytest.raises(ValueError, match="provenance"):
        cache.RelationFeatureCache(backend.model, backend.encode, parent={"different": True})
    with torch.no_grad():
        next(backend.model.backbone.parameters()).add_(1)
    with pytest.raises(ValueError, match="drift"):
        features.forward([entry])
    backend.model.head = torch.nn.Linear(16, 3)
    with pytest.raises(ValueError, match="q_end"):
        cache.RelationFeatureCache(backend.model, backend.encode, parent=PARENT)


def test_context_identity_defensive_copy_and_parent_metadata_guard():
    backend, features = native()
    pinned = features.identity()
    copy = features.identity()
    copy["parent"]["snapshot"] = "mutated"
    assert features.identity() == pinned
    backend.model.backbone.layer = 0
    with pytest.raises(ValueError, match="drift"):
        features.identity()


def test_boundary_hash_includes_buffers_and_mutation_bypassing_versions():
    backend, _ = native()
    backend.model.backbone.register_buffer("fixture_buffer", torch.ones(2))
    features = cache.RelationFeatureCache(backend.model, backend.encode, parent=PARENT)
    pinned = features.identity()
    backend.model.backbone.fixture_buffer.data.add_(1)
    # Raw .data mutation bypasses normal tensor versions, so the experiment's
    # full terminal/checkpoint verification must still reject changed bytes.
    assert features.identity() == pinned
    with pytest.raises(ValueError, match="bytes"):
        features.verify_parent()


def test_2048_limit_includes_continuation_and_never_truncates(monkeypatch):
    backend, _ = native()
    class BoundaryEncoder(Encoder):
        padding = 1
        def __call__(self, text, trusted=True):
            return [4] * self.padding if text.startswith('{"accepted_partial_program"') else super().__call__(text, trusted)
        def identity(self): return {**super().identity(), "padding": self.padding}
    encode = BoundaryEncoder()
    req = request()
    base = item(req, encode)
    encode.padding += 2048 - len(base.row.ids) - max(map(len, base.row.continuations))
    features = cache.RelationFeatureCache(backend.model, encode, parent=PARENT)
    exact = item(req, encode)
    assert len(exact.row.ids) + max(map(len, exact.row.continuations)) == 2048
    class ReachedExtraction(Exception): pass
    def reached(*args, **kwargs): raise ReachedExtraction
    monkeypatch.setattr("heads.features.build_head_input", reached)
    with pytest.raises(ReachedExtraction): features.capture(exact, req)
    encode.padding += 1
    features = cache.RelationFeatureCache(backend.model, encode, parent=PARENT)
    with pytest.raises(ValueError, match="2048"):
        features.capture(item(req, encode), req)


def test_cache_import_has_no_runtime_or_pretrained_access():
    code = r'''
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise AssertionError('runtime import')
sys.meta_path.insert(0, Reject())
from language_growth import relation_cache
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_native_one_step_checkpoint_reload_is_strict_and_parent_bitwise_frozen(tmp_path):
    backend, features = native()
    req = request()
    entry = features.capture(item(req, backend.encode), req)
    lineage = cache.training_lineage([entry], [0], context=features.identity())
    optimizer = torch.optim.AdamW(backend.model.head_parameters(), lr=.01)
    parent_before = {k: v.clone() for k, v in backend.model.backbone.state_dict().items()}
    initial = cache._train_state(backend.model)
    cache.save_checkpoint(tmp_path/"initial.pt", features, optimizer, lineage=lineage, step=0)
    loss = -features.forward([entry])[0, 0]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    trained = cache._train_state(backend.model)
    assert any(not torch.equal(initial[k], trained[k]) for k in trained)
    cache.save_checkpoint(tmp_path/"trained.pt", features, optimizer, lineage=lineage, step=1)
    with torch.no_grad(): expected = features.forward([entry]).clone()
    cache.restore_checkpoint(tmp_path/"initial.pt", features, optimizer, lineage=lineage, step=0)
    assert all(torch.equal(cache._train_state(backend.model)[k], v) for k, v in initial.items())
    cache.restore_checkpoint(tmp_path/"trained.pt", features, optimizer, lineage=lineage, step=1)
    with torch.no_grad(): torch.testing.assert_close(features.forward([entry]), expected, rtol=0, atol=0)
    assert all(torch.equal(backend.model.backbone.state_dict()[k], v) for k, v in parent_before.items())
    assert all(p.grad is None for p in backend.model.backbone.parameters())
    fresh_backend, fresh_features = native()
    fresh_optimizer = torch.optim.AdamW(fresh_backend.model.head_parameters(), lr=.01)
    cache.restore_checkpoint(tmp_path/"trained.pt", fresh_features, fresh_optimizer, lineage=lineage, step=1)
    with torch.no_grad(): torch.testing.assert_close(fresh_features.forward([entry]), expected, rtol=0, atol=0)
    for changed_lineage, step in (({**lineage, "targets_sha256": "different"}, 1), (lineage, 0)):
        with pytest.raises(ValueError, match="lineage"):
            cache.restore_checkpoint(tmp_path/"trained.pt", features, optimizer, lineage=changed_lineage, step=step)
    original = torch.load(tmp_path/"trained.pt", weights_only=True)
    for change in ("shape", "dtype", "missing", "nonfinite", "optimizer"):
        broken = deepcopy(original)
        key = next(iter(broken["trainable"]))
        if change == "shape": broken["trainable"][key] = torch.zeros(1)
        elif change == "dtype": broken["trainable"][key] = broken["trainable"][key].double()
        elif change == "missing": del broken["trainable"][key]
        elif change == "nonfinite": broken["trainable"][key].fill_(float("nan"))
        else:
            state = next(iter(broken["optimizer"]["state"].values()))
            state["exp_avg"] = torch.zeros(1)
        torch.save(broken, tmp_path/"broken.pt")
        with pytest.raises(ValueError, match="mismatch"):
            cache.restore_checkpoint(tmp_path/"broken.pt", features, optimizer, lineage=lineage, step=1)
        assert all(torch.equal(cache._train_state(backend.model)[k], v) for k, v in trained.items())
    wrong_optimizer = torch.optim.AdamW(backend.model.parameters(), lr=.01)
    with pytest.raises(ValueError, match="ordered head"):
        cache.restore_checkpoint(tmp_path/"trained.pt", features, wrong_optimizer, lineage=lineage, step=1)


@pytest.mark.parametrize("target", [-1, 3, True, 0.5])
def test_training_targets_are_separate_and_validated(target):
    backend, features = native()
    req = request()
    entry = features.capture(item(req, backend.encode), req)
    with pytest.raises(ValueError, match="target outside"):
        cache.training_lineage([entry], [target], context=features.identity())
    with pytest.raises(ValueError, match="unique"):
        cache.training_lineage([entry, entry], [0, 0], context=features.identity())


def test_real_full_fit_item_conversion_guards_3072_rows_before_576_selection(monkeypatch):
    from train.guard import SplitGuard
    from language_growth.l1_world_training import L1WorldTrainingData
    data = load_l1_relation_data()
    original = L1WorldTrainingData.fit_items
    seen = []
    def watched(worlds, *args, **kwargs):
        seen.append((len(worlds.fit_records), worlds.fit_world_ids))
        result = original(worlds, *args, **kwargs)
        assert len(result[0]) == 3072
        return result
    monkeypatch.setattr(L1WorldTrainingData, "fit_items", watched)
    class Guard:
        def __init__(self): self.calls = []
        def check(self, record):
            self.calls.append(record["id"])
            SplitGuard.synthetic_only("relation CPU fixture").check(record)
    guard = Guard()
    items, worlds = cache.fit_items(data, Encoder(), guard=guard)
    assert seen == [(3072, data.world_data.fit_world_ids)]
    assert len(guard.calls) == 3072 and len(items) == len(worlds) == 576
    assert set(worlds.values()) == data.world_data.fit_world_ids
    assert not set(worlds.values()) & data.world_data.calibration_world_ids
    assert len(data.calibration_records) == 192
    assert all(it.row.qid == RELATION_QID for it in items)
    assert all("world_id" not in it.row.meta and "labels" not in it.row.meta for it in items)
    seen.clear()
    for rows in (data.world_data.fit_records[:-1], data.fit_records,
                 data.world_data.fit_records + data.world_data.calibration_records):
        bad = replace(data, world_data=replace(data.world_data, fit_records=rows))
        with pytest.raises(ValueError): cache.fit_items(bad, Encoder(), guard=guard)
    assert seen == []  # pinned verification fails before Item conversion


def test_live_native_composition_and_malformed_fallback_preserve_requests():
    backend, features = native()
    class Stock:
        def __init__(self): self.calls = []
        def revision(self): return {**PARENT, "revision_id": "stock"}
        def plan(self, req): self.calls.append(req); return "stock-plan"
        def score(self, req, check): self.calls.append(req); check(); return "stock-score"
    stock = Stock()
    router = RelationOnlyBackend(backend, stock)
    req = request()
    before = deepcopy(req)
    assert router.route(req) is backend
    result = router.score(req, lambda: None)
    entry = features.capture(item(req, backend.encode), req)
    with torch.no_grad():
        lp = features.forward([entry])[0]
    for i, oid in enumerate(entry["identity"]["row"]["option_ids"]):
        assert result.logits[RELATION_QID][oid] == pytest.approx(float(lp[i]), abs=1e-6)
    assert req == before
    bad_requests = []
    for change in ("missing", "reordered", "extra", "nonrelation", "history", "state"):
        bad = deepcopy(req)
        state = json.loads(bad["state"])
        if change == "missing": state["proposed_slot_choices"].pop()
        elif change == "reordered": state["proposed_slot_choices"].reverse()
        elif change == "extra": state["gold"] = "forbidden"
        elif change == "nonrelation": bad["questions"] = {"slot-001-operation": bad["questions"][RELATION_QID]}
        elif change == "history": bad["history"] = []
        bad["state"] = "not-json" if change == "state" else json.dumps(state)
        bad_requests.append(bad)
    for bad in bad_requests:
        saved = deepcopy(bad)
        assert router.plan(bad) == "stock-plan"
        assert router.score(bad, lambda: None) == "stock-score"
        assert stock.calls[-1] is bad and bad == saved
        with pytest.raises(ValueError, match="exact relation"):
            features.capture(item(req, backend.encode), bad)
