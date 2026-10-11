"""Consistency-view training (docs/CONSISTENCY.md): the view generator (stdlib), the symmetric-KL term, the trainer arm
and its bit-identical OFF path (also checked against the parent tree when SSDM_BASE_TREE points at it)."""
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import consistency_views as CV  # noqa: E402


def _records(n=40):
    """Synthetic train-like records: choice / noul / score, string (markdown, lists) and dict states."""
    out = []
    for i in range(n):
        kind = ("choice", "noul", "score")[i % 3]
        if i % 4 == 0:
            state = {"customer": f"user {i}", "status": "open", "note": "Needs a refund for the order"}
        elif i % 4 == 1:
            state = f"# Ticket {i}\n- item one\n- item two\n**Priority:** normal\n\nCustomer  wrote: hello"
        elif i % 4 == 2:
            state = f"Status: pending\n1. first step\n2. second step\nThe client asked about e-mail delivery {i}."
        else:
            state = {"events": [f"Event {i} started", "Event closed"], "meta": {"Owner": "team A"}}
        if kind == "choice":
            q = {"type": "choice", "instructions": f"Choose the most appropriate route for case {i}.",
                 "criteria": {"billing": "Billing issue & refund request", "tech": "Technical problem with the app",
                              "sales": "Customer wants to buy a large plan"}}
            lab = ["billing", "tech", "sales"][i % 3]
        elif kind == "noul":
            q = {"type": "noul", "instructions": f"Determine whether the request in case {i} is resolved.",
                 "criteria": {"true": "Required information is present", "false": "Details are missing"}}
            lab = "true" if i % 2 else "false"
        else:
            q = {"type": "score", "instructions": f"Classify the urgency of case {i}.",
                 "criteria": ["No action needed", "Quick reply required", "Begin escalation"]}
            lab = str(i % 3)
        out.append({"id": f"rec{i}", "request": {"model": "toy", "state": state, "questions": {"q": q}},
                    "labels": {"q": lab}})
    return out


# ------------------------------------------------------------------------------------------ view generator (CPU)
def test_views_are_deterministic_and_seeded():
    recs = _records()
    a = [CV.view_record(r, "q", seed=3) for r in recs]
    b = [CV.view_record(copy.deepcopy(r), "q", seed=3) for r in recs]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = [CV.view_record(r, "q", seed=4) for r in recs]
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)
    assert all(r == r0 for r, r0 in zip(recs, _records()))  # inputs never mutated


def test_views_preserve_labels_and_option_domain_and_change_the_surface():
    fams = {"format": 0, "instruction": 0, "option": 0}
    for r in _records(120):
        got = CV.view_record(r, "q", seed=0)
        assert got is not None
        v, fam, ops = got
        fams[fam] += 1
        q0, q1 = r["request"]["questions"]["q"], v["request"]["questions"]["q"]
        assert v["labels"] == r["labels"] and q1["type"] == q0["type"]
        c0, c1 = q0["criteria"], q1["criteria"]
        assert (list(c0) == list(c1)) if isinstance(c0, dict) else len(c0) == len(c1)  # keys AND order
        assert v != r and v["id"] == f"{r['id']}#cv:q" and ops
        if fam != "format":
            assert v["request"]["state"] == r["request"]["state"]
        if fam != "instruction":
            assert q1["instructions"] == q0["instructions"]
        if fam != "option":
            assert c1 == c0
    assert min(fams.values()) >= 15  # every family is exercised


def test_sampling_rate_and_family_restriction():
    ids = [f"m2:x:{i}/q" for i in range(4000)]
    rate = sum(CV.sampled(0, i, 0.25) for i in ids) / len(ids)
    assert abs(rate - 0.25) < 0.02
    for r in _records(30):
        got = CV.view_record(r, "q", seed=1, families=("option",))
        assert got is None or got[1] == "option"
    with pytest.raises(ValueError):
        CV.view_record(_records(1)[0], "q", families=("reverse",))


def test_label_check_fails_closed():
    r = _records(1)[0]
    v = copy.deepcopy(r)
    v["request"]["questions"]["q"]["criteria"] = dict(reversed(list(r["request"]["questions"]["q"]["criteria"].items())))
    with pytest.raises(ValueError):
        CV.check_label_preserved(r, v, "q")
    v = copy.deepcopy(r)
    v["labels"]["q"] = "tech" if r["labels"]["q"] != "tech" else "sales"
    with pytest.raises(ValueError):
        CV.check_label_preserved(r, v, "q")


def test_independent_of_the_jevkit_test_operators():
    src = (ROOT / "train/consistency_views.py").read_text(encoding="utf-8")
    import re

    assert not re.search(r"^\s*(import|from)\s+jevkit", src, re.M)
    from jevkit import perturb as P

    ours = {w.lower() for pair in CV.INSTRUCTION_SYNONYMS + CV.OPTION_SYNONYMS for w in pair}
    theirs = {w.lower() for pair in P._PARA for w in pair}
    assert not ours & theirs  # no shared rewrite vocabulary (no contractions, e.g./i.e., approximately, ...)
    assert not set(CV.INSTRUCTION_FRAMES) & set(P._FRAMES)
    for r in _records(120):
        got = CV.view_record(r, "q", seed=0)
        v, fam, _ = got
        s0, s1 = r["request"]["state"], v["request"]["state"]
        q0, q1 = r["request"]["questions"]["q"], v["request"]["questions"]["q"]
        if fam == "format" and isinstance(s0, str):
            assert not s1.startswith("\n") and not s1.endswith("\n")  # jevkit whitespace adds both
            assert s1.count("  ") <= s0.count("  ")  # jevkit doubles space runs; we only collapse them
            assert not any(ln.startswith("> ") for ln in s1.split("\n"))  # jevkit markup quotes the state
        if fam == "format" and isinstance(s0, dict) and isinstance(s1, dict):
            assert list(s1) == list(s0)  # jevkit markup reverses dict keys; we never reorder
        if fam == "option":  # jevkit toggles the final period; we never touch final punctuation
            descs = (lambda c: list(c.values()) if isinstance(c, dict) else c)
            for a, b in zip(descs(q0["criteria"]), descs(q1["criteria"])):
                assert a.rstrip().endswith(".") == b.rstrip().endswith(".")
        if fam == "instruction":
            assert not any(q1["instructions"] == f.format(q0["instructions"]) for f in P._FRAMES)


# ------------------------------------------------------------------------------------------ losses and trainer (torch)
def _torch():
    return pytest.importorskip("torch")


def test_symmetric_kl_properties_and_gradient():
    torch = _torch()
    from train.losses import symmetric_kl

    p = torch.log_softmax(torch.tensor([[2.0, 0.5, -1.0], [0.3, 0.1, float("-inf")]], dtype=torch.float64), -1)
    q = torch.log_softmax(torch.tensor([[0.0, 1.0, 0.0], [1.3, -0.2, float("-inf")]], dtype=torch.float64), -1)
    kl = symmetric_kl(p, q)
    pe, qe = p.exp(), q.exp()
    m = torch.isfinite(p)
    want = (torch.where(m, pe * (p - q), 0.0) + torch.where(m, qe * (q - p), 0.0)).sum(-1)
    assert torch.allclose(kl, want) and (kl > 0).all()
    assert torch.allclose(symmetric_kl(q, p), kl)
    assert torch.equal(symmetric_kl(p, p), torch.zeros(2, dtype=torch.float64))
    # gradient sanity: autograd == central finite differences on the logits, and a descent step lowers the KL
    z = torch.tensor([[0.4, -0.3, 1.1]], dtype=torch.float64, requires_grad=True)
    t = torch.log_softmax(torch.tensor([[1.0, 0.0, -0.5]], dtype=torch.float64), -1)
    f = lambda zz: symmetric_kl(torch.log_softmax(zz, -1), t).sum()  # noqa: E731
    f(z).backward()
    fd = torch.zeros_like(z)
    for j in range(3):
        e = torch.zeros_like(z)
        e[0, j] = 1e-6
        fd[0, j] = (f(z.detach() + e) - f(z.detach() - e)) / 2e-6
    assert torch.allclose(z.grad, fd, atol=1e-7)
    assert float(f(z.detach() - 0.1 * z.grad)) < float(f(z.detach()))
    with pytest.raises(ValueError):
        symmetric_kl(p, torch.log_softmax(torch.zeros(2, 3, dtype=torch.float64), -1))  # different valid options


def _guard():
    from train.guard import SplitGuard

    return SplitGuard.synthetic_only("unit tests: consistency views")


def _toy_items(n=8, seed=0, view_frac=1.0):
    from heads import ByteEncoder
    from train.consistency_views import attach_views
    from train.data import TOY_TEMPLATE, items_from_records

    recs = [r for r in _records(n)]
    enc = ByteEncoder()
    items = items_from_records(recs, enc, template=TOY_TEMPLATE)
    cov = attach_views(items, {r["id"]: r for r in recs}, enc, layout="ordered", template=TOY_TEMPLATE,
                       guard=_guard(), keep=lambda row: True, seed=seed, view_frac=view_frac)
    return items, cov, recs


def _model(seed=0):
    torch = _torch()
    from heads import DecisionModel
    from heads.backbone import tiny_backbone

    torch.manual_seed(seed)
    return DecisionModel(tiny_backbone(seed=seed, hidden=16), "integrator")


def _cfg(**kw):
    from train.losses import LossConfig
    from train.trainer import TrainConfig

    base = dict(head="integrator", lora_rank=4, batch_size=2, grad_accum=2, max_steps=3, loss=LossConfig(arm="clef"))
    base.update(kw)
    return TrainConfig(**base)


def test_attach_views_training_only_and_coverage():
    _torch()
    items, cov, _ = _toy_items(12)
    assert cov["with_view"] == 12 and sum(cov["by_family"].values()) == 12
    for it in items:
        v = it.cview
        assert v.row.option_ids == it.row.option_ids and v.label == it.label and v.row.ids != it.row.ids
    from train.trainer import Trainer

    plain, _, _ = _toy_items(4, view_frac=1e-9)
    assert all(it.cview is None for it in plain)
    with pytest.raises(ValueError, match="eval_items"):
        Trainer(_model(), _cfg(consistency={"lambda": 0.5}), items, eval_items=items, guard=_guard())
    with pytest.raises(ValueError, match="no training item"):
        Trainer(_model(), _cfg(consistency={"lambda": 0.5}), plain, guard=_guard())
    with pytest.raises(ValueError, match="is off"):
        Trainer(_model(), _cfg(), items, guard=_guard())


def test_consistency_spec_validation_and_run_cell_block():
    _torch()
    from train.run_cell import consistency_fields
    from train.trainer import consistency_spec

    assert consistency_spec({}) == {} and consistency_fields({}) == {} and consistency_fields({"consistency": {}}) == {}
    s = consistency_fields({"consistency": {"view_frac": 0.25, "lambda": 0.5}})
    assert s["view_frac"] == 0.25 and s["view_grad"] is True and s["controller"] is None
    for bad in ({"lambda": -1}, {"view_frac": 0}, {"families": ["reverse"]}, {"weight": 1},
                {"controller": {"epsilon": 0.01}}, {"controller": {"epsilon": 0, "kappa_p": 1, "kappa_i": 0.1,
                                                                   "nu": 0, "lambda_init": 0, "lambda_max": 1}}):
        with pytest.raises(ValueError):
            consistency_spec(bad)
    with pytest.raises(ValueError):
        consistency_fields({"consistency": {"lambda": 0.5}, "order_consistency": {"view": "reverse", "weight": 0.25}})


def test_kl_term_gradient_on_tiny_model():
    torch = _torch()
    from train.trainer import Trainer

    items, _, _ = _toy_items(4)
    model = _model()
    tr = Trainer(model, _cfg(consistency={"lambda": 1.0}), items, guard=_guard())
    model.train()
    batch = items[:2]
    fw = model([i.row for i in batch])
    term = tr._consistency_term(fw.log_probs.float(), batch)
    assert term.ndim == 0 and float(term) > 0 and math.isfinite(float(term))
    term.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads) and all(torch.isfinite(g).all() for g in grads)
    # lambda scales the term linearly; view_grad False keeps the value, but no gradient flows through the view call
    tr.cons_lambda = 0.5
    model.zero_grad(set_to_none=True)
    t2 = tr._consistency_term(model([i.row for i in batch]).log_probs.float(), batch)
    assert torch.allclose(t2, term.detach() * 0.5, rtol=1e-5, atol=1e-7)
    tr.cons["view_grad"] = False
    t3 = tr._consistency_term(model([i.row for i in batch]).log_probs.float(), batch)
    assert torch.allclose(t3, t2.detach(), rtol=1e-5, atol=1e-7)


def test_training_with_views_logs_kl_and_the_term_lowers_it():
    """Same seed, data and steps: lambda_c 4 ends with a lower orig/view KL than lambda_c 0 (task loss alone)."""
    torch = _torch()
    from train.losses import symmetric_kl
    from train.trainer import Trainer

    def run(lam):
        items, _, _ = _toy_items(8)
        model = _model()
        tr = Trainer(model, _cfg(max_steps=25, lr_head=3e-3, consistency={"lambda": lam}), items, guard=_guard())
        hist = tr.train()
        assert len(hist) == 25 and all(math.isfinite(h["loss"]) and h["cons_paired"] == 4 for h in hist)
        assert all(h["cons_lambda"] == lam and h["cons_kl"] >= 0 for h in hist)
        model.eval()
        with torch.no_grad():
            a = model([i.row for i in items]).log_probs.float()
            b = model([i.cview.row for i in items]).log_probs.float()
        k = min(a.shape[1], b.shape[1])
        return float(symmetric_kl(a[:, :k], b[:, :k]).mean())

    assert run(4.0) < run(0.0)


def test_nupi_controller_moves_lambda_and_checkpoints(tmp_path):
    _torch()
    from train.trainer import Trainer

    items, _, _ = _toy_items(8)
    ctl = {"epsilon": 1e-9, "kappa_p": 0.0, "kappa_i": 10.0, "nu": 0.0, "lambda_init": 0.0, "lambda_max": 5.0}
    tr = Trainer(_model(), _cfg(max_steps=3, ckpt_dir=str(tmp_path), consistency={"controller": ctl}), items,
                 guard=_guard())
    h = tr.train()
    assert h[0]["cons_lambda"] == 0.0 and tr.cons_lambda > 0.0  # KL > epsilon: the multiplier rises
    tr.save()
    tr2 = Trainer(_model(), _cfg(max_steps=3, ckpt_dir=str(tmp_path), consistency={"controller": ctl}), items,
                  guard=_guard())
    assert tr2.resume() and tr2.cons_lambda == tr.cons_lambda
    assert tr2.cons_ctrl.state_dict() == tr.cons_ctrl.state_dict()
    loose = {**ctl, "epsilon": 1e9, "lambda_init": 1.0}
    tr3 = Trainer(_model(), _cfg(max_steps=2, consistency={"controller": loose}), items, guard=_guard())
    h3 = tr3.train()
    assert h3[0]["cons_lambda"] == 1.0 and tr3.cons_lambda == 0.0  # KL far below epsilon: the warm start decays to 0


# ------------------------------------------------------------------------------------------ the OFF path
GOLDEN = r'''
import hashlib, json, os, sys
import torch
from heads import ByteEncoder, DecisionModel
from heads.backbone import tiny_backbone
from train.data import TOY_TEMPLATE, items_from_records
from train.guard import SplitGuard
from train.losses import LossConfig
from train.trainer import Trainer, TrainConfig
recs = json.loads(sys.argv[1])
out = {}
for name, kw in (("micro", {}), ("rows", {"row_batch": 2})):
    items = items_from_records(recs, ByteEncoder(), template=TOY_TEMPLATE)
    torch.manual_seed(0)
    model = DecisionModel(tiny_backbone(seed=0, hidden=16), "integrator")
    extra = {"consistency": {}} if os.environ.get("CONS_EXPLICIT") == "1" else {}
    cfg = TrainConfig(head="integrator", lora_rank=4, batch_size=2, grad_accum=2, max_steps=4,
                      loss=LossConfig(arm="clef", label_smoothing=0.05), **kw, **extra)
    tr = Trainer(model, cfg, items, guard=SplitGuard.synthetic_only("golden"))
    h = tr.train()
    sha = hashlib.sha256()
    for n, p in sorted(tr.trainable_state().items()):
        sha.update(n.encode()); sha.update(p.float().numpy().tobytes())
    out[name] = {"losses": [repr(x["loss"]) for x in h], "history_keys": sorted({k for x in h for k in x}),
                 "params": sha.hexdigest(), "hash": cfg.hash(), "to_json": json.dumps(cfg.to_json(), sort_keys=True),
                 "lineage": json.dumps(tr.lineage, sort_keys=True, default=str),
                 "state_keys": sorted(tr.state())}
print(json.dumps(out, sort_keys=True))
'''


def _golden(tree: Path, explicit: bool) -> dict:
    recs = _records(10)
    env = {**os.environ, "PYTHONPATH": str(tree), "CONS_EXPLICIT": "1" if explicit else "0", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", GOLDEN, json.dumps(recs)], cwd=str(tree), env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_off_path_identity_in_this_tree():
    _torch()
    from train.trainer import TrainConfig

    assert TrainConfig(consistency={}).hash() == TrainConfig().hash()
    assert "consistency" not in TrainConfig().to_json() and TrainConfig().to_json() == TrainConfig(consistency={}).to_json()
    a, b = _golden(ROOT, False), _golden(ROOT, True)
    assert a == b
    assert all("cons_kl" not in v["history_keys"] and "consistency" not in v["state_keys"] for v in a.values())


def test_off_path_bit_identical_to_the_parent_tree():
    """SSDM_BASE_TREE = a checkout of the parent sha (stack/qwen35/jev2-4b @a5b4cb33; the cpu_tests job sets it): the
    same toy run (micro-batch and row_batch paths) gives the same losses, trainable weights, config hash, to_json,
    lineage and checkpoint keys, bit for bit."""
    base = os.environ.get("SSDM_BASE_TREE")
    if not base:
        pytest.skip("SSDM_BASE_TREE not set (the cpu_tests job sets it to the parent tree)")
    _torch()
    assert _golden(Path(base), False) == _golden(ROOT, False) == _golden(ROOT, True)
