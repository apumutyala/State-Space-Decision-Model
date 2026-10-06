"""PR #14 review items: device correctness, cache-copy fork, LoRA targeting, split guard, loss arms.

CPU only. Device correctness is checked on the ``meta`` device: every weight and input lives on
meta, so any tensor built on the default CPU device without ``device=`` makes the forward or
backward fail with a device mismatch (meta and CPU tensors do not mix)."""

import gzip
import json
import re

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from heads import HEADS, ByteEncoder, DecisionModel, build_row  # noqa: E402
from heads.backbone import BACKBONES, tiny_backbone, tiny_hf_backbone  # noqa: E402
from heads.features import build_head_input  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.guard import ForbiddenRow, SplitGuard  # noqa: E402
from train.lora import apply_lora, lora_param_count  # noqa: E402
from train.losses import LossConfig, decision_loss, nll  # noqa: E402

ENC = ByteEncoder()
SMALL = {"clef_control": {"width": 16, "heads": 2, "feedforward": 32, "layers": 1, "routing_layers": 1}}


def req():
    return {
        "model": "t",
        "state": "ticket: checkout down since 9am; vendor Acme",
        "questions": {
            "dept": {"type": "choice", "instructions": "team?",
                     "criteria": {"billing": "payments", "tech": "outages", "sales": None}},
            "down": {"type": "noul", "instructions": "is a service down?"},
            "urg": {"type": "score", "criteria": ["later", "week", "today"]},
        },
    }


def rows(layout="ordered"):
    return [build_row(req(), q, ENC, template=TOY_TEMPLATE, layout=layout) for q in ("dept", "down", "urg")]


# ------------------------------------------------------------------------- devices
@pytest.mark.parametrize("name", sorted(HEADS))
@pytest.mark.parametrize("layout", ["ordered", "symmetric"])
def test_forward_and_backward_stay_on_the_backbone_device(name, layout):
    fork_mode = "cache" if name == "fork" and layout == "ordered" else "recompute"
    m = DecisionModel(tiny_backbone(seed=0, hidden=16, fork_mode=fork_mode), name, SMALL.get(name)).to("meta")
    m.train()
    out = m(rows(layout))
    assert out.log_probs.device.type == "meta" and out.inputs.e.device.type == "meta"
    y = torch.tensor([0, 1, 2], device="meta")
    loss = decision_loss(out.log_probs, y, ["choice", "noul", "score"], LossConfig(arm="clef"))
    loss.backward()
    assert all(p.grad is None or p.grad.device.type == "meta" for p in m.parameters())


def test_meta_check_catches_a_cpu_tensor():
    """The meta check above is not vacuous: mixing a CPU-built tensor into meta math fails."""
    x = torch.zeros(3, 4, device="meta")
    with pytest.raises(Exception):
        x + torch.zeros(3, 4)  # a CPU tensor built without device=


# ------------------------------------------------------------------------- cache-copy fork
@pytest.mark.parametrize("family", ["tiny", "q9af", "qwen3.5", "mamba"])
def test_cache_fork_equals_recompute_fork(family):
    def bb(mode):
        return tiny_backbone(0, 16, fork_mode=mode) if family == "tiny" else tiny_hf_backbone(family, fork_mode=mode)

    with torch.no_grad():
        a = build_head_input(bb("cache"), rows(), need_fork=True)
        b = build_head_input(bb("recompute"), rows(), need_fork=True)
    assert torch.allclose(a.fork, b.fork, atol=1e-5)
    assert torch.allclose(a.h * a.token_mask.unsqueeze(-1), b.h * b.token_mask.unsqueeze(-1), atol=1e-5)


@pytest.mark.parametrize("family", ["q9af", "mamba"])
def test_cache_fork_gradients_equal_recompute(family):
    grads = []
    for mode in ("cache", "recompute"):
        b = tiny_hf_backbone(family, fork_mode=mode)
        torch.manual_seed(5)
        apply_lora(b.model, b.spec.lora_groups, rank=2)
        for n, p in b.model.named_parameters():
            if n.endswith("lora_B"):
                nn.init.normal_(p, std=0.1)
        x = build_head_input(b, rows(), need_fork=True)
        x.fork.pow(2).sum().backward()
        grads.append({n: p.grad.clone() for n, p in b.model.named_parameters() if p.requires_grad})
    for k in grads[0]:
        assert torch.allclose(grads[0][k], grads[1][k], atol=1e-4), k


def test_cache_fork_refuses_hf_gradient_checkpointing():
    b = tiny_hf_backbone("q9af", fork_mode="cache")
    b.enable_gradient_checkpointing()
    b.model.train()
    with pytest.raises(RuntimeError, match="gradient checkpointing"):
        build_head_input(b, rows(), need_fork=True)


# ------------------------------------------------------------------------- LoRA targeting
class _Lin(nn.Module):
    def __init__(self, names):
        super().__init__()
        for n in names:
            setattr(self, n, nn.Linear(4, 4))


def _tree(layer_children: dict[str, list[str]], n_layers: int, top: str) -> nn.Module:
    """Module tree with real-checkpoint names: ``<top>.layers.<i>.<child>.<linear>``."""
    root = nn.Module()
    body = nn.Module()
    layers = nn.ModuleList()
    for _ in range(n_layers):
        blk = nn.Module()
        for child, lins in layer_children.items():
            setattr(blk, child, _Lin(lins))
        layers.append(blk)
    body.layers = layers
    setattr(root, top, body)
    return root


RWKV7_HF = {"att": ["receptance", "key", "value", "output", "w1", "w2"], "ffn": ["key", "value"]}
RWKV7_FLA = {"attn": ["r_proj", "k_proj", "v_proj", "o_proj"], "ffn": ["key", "value"]}
MAMBA3 = {"mixer": ["in_proj", "out_proj"], "mlp": ["fc1", "fc2"]}


def _spec_scaled(spec, n_layers):
    """Same regexes, expected counts rescaled to a small mock depth."""
    return [(g, rx, cnt * n_layers // spec.n_layers) for g, rx, cnt in spec.lora_groups]


def test_rwkv7_hf_names_match_every_group():
    spec = BACKBONES["rwkv7-13.3b"]
    model = _tree(RWKV7_HF, 3, "model")
    found = apply_lora(model, _spec_scaled(spec, 3), rank=2)
    assert len(found["att"]) == 12 and len(found["ffn"]) == 6


def test_rwkv7_fla_names_fail_loudly_instead_of_training_ffn_only():
    spec = BACKBONES["rwkv7-13.3b"]
    with pytest.raises(ValueError, match="matched no nn.Linear"):
        apply_lora(_tree(RWKV7_FLA, 3, "model"), _spec_scaled(spec, 3), rank=2)


def test_mamba3_targets_mixer_and_gated_mlp():
    spec = BACKBONES["mamba3-1.5b"]
    assert spec.n_layers == 24
    found = apply_lora(_tree(MAMBA3, 24, "backbone"), spec.lora_groups, rank=2)
    assert len(found["mixer"]) == 48 and len(found["mlp"]) == 48
    assert all(re.search(r"mlp\.fc[12]$", n) for n in found["mlp"])


def test_count_mismatch_fails():
    spec = BACKBONES["mamba3-1.5b"]
    with pytest.raises(ValueError, match="expected"):
        apply_lora(_tree(MAMBA3, 12, "backbone"), spec.lora_groups, rank=2)  # 48-layer assumption would fail too


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_spec_lora_shapes_agree_with_group_counts(name):
    spec = BACKBONES[name]
    assert sum(c for _, _, c in spec.lora_shapes) == sum(c for _, _, c in spec.lora_groups)
    assert lora_param_count(spec.lora_shapes, 1) > 0


@pytest.mark.parametrize("family", ["qwen3.5", "q9af", "mamba"])
def test_tiny_hf_specs_match_real_module_names(family):
    b = tiny_hf_backbone(family)
    found = apply_lora(b.model, b.spec.lora_groups, rank=2)  # raises on zero or wrong counts
    assert all(found.values())
    if family == "q9af":
        assert not any("self_attn" in n for ns in found.values() for n in ns)


def test_single_regex_with_no_match_raises():
    with pytest.raises(ValueError):
        apply_lora(tiny_backbone().model, r"no_such_module$", rank=2)


# ------------------------------------------------------------------------- loss arms
def test_ce_arm_is_pure_ce_on_score_rows():
    lp = torch.log_softmax(torch.tensor([[0.3, 1.0, -0.5]]), -1)
    y = torch.tensor([2])
    ce = decision_loss(lp, y, ["score"], LossConfig(arm="ce"))
    assert torch.allclose(ce, nll(lp, y).mean())
    assert LossConfig(arm="clef").effective_rps_weight == 1.0
    assert decision_loss(lp, y, ["score"], LossConfig(arm="ce", rps_weight=1.0)) > ce  # explicit opt-in only


# ------------------------------------------------------------------------- split guard
def _manifest(tmp_path, entries):
    p = tmp_path / "test-manifest.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return p


def test_guard_refuses_manifest_rows_and_index_metadata(tmp_path):
    rec = toy_records(2, seed=9)
    from decide.schema import canonical_json, sha256_text

    p = _manifest(tmp_path, [{"run_id": "11:X:test:1", "group_id": "X:test:1", "row_sha256": "ab" * 32,
                              "payload_sha256": sha256_text(canonical_json(rec[1]["request"]))}])
    g = SplitGuard.from_manifests([p])
    assert len(g.manifests[str(p)]) == 64
    g.check(rec[0])  # unrelated synthetic record passes
    with pytest.raises(ForbiddenRow, match="request hash"):
        g.check(rec[1])
    with pytest.raises(ForbiddenRow, match="run_id"):
        g.check({**rec[0], "run_id": "11:X:test:1"})
    with pytest.raises(ForbiddenRow, match="Index metadata"):
        SplitGuard.synthetic_only("t").check({**rec[0], "_evaluation": {"run_id": "z"}})
    with pytest.raises(ForbiddenRow):
        items_from_records([rec[1]], ENC, template=TOY_TEMPLATE, guard=g)


def test_trainer_requires_a_guard():
    from train.trainer import Trainer, TrainConfig

    items = items_from_records(toy_records(4), ENC, template=TOY_TEMPLATE)
    m = DecisionModel(tiny_backbone(hidden=16), "integrator")
    with pytest.raises(ValueError, match="SplitGuard"):
        Trainer(m, TrainConfig(lora_rank=2), items)


def test_temperature_fitted_on_eval_items_is_refused(tmp_path):
    from train.calibration import FeatureTemperature
    from train.evalcard import run_card

    items = items_from_records(toy_records(4), ENC, template=TOY_TEMPLATE)
    m = DecisionModel(tiny_backbone(hidden=16), "integrator")
    lp = torch.log_softmax(torch.randn(len(items), 4), -1)
    t = FeatureTemperature.fit(lp, torch.zeros(len(items), dtype=torch.long),
                               torch.ones(len(items), 3, dtype=torch.float64), [i.id for i in items])
    with pytest.raises(ValueError, match="eval items"):
        run_card(m, items, step=0, temperature=t)
