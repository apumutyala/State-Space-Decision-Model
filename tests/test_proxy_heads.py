"""Proxy heads + loop on SYNTHETIC cached features (train/proxy_heads.py, train/proxy_loop.py). CPU, $0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from train import proxy_loop as PL  # noqa: E402
from train.proxy_heads import AFMixer, AttnRef, ForkProxy, ProxyQuestion, ProxyRequest, make_proxy_head  # noqa: E402
from train import proxy_rule as R  # noqa: E402
from train.proxy_rule import CANDIDATE_FAMILIES, RECIPES  # noqa: E402

D = 16


def _q(g, qid, qtype, k, label=-1, bank="state"):
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return ProxyQuestion(qid, qtype, [str(i) for i in range(k)] if qtype != "noul" else ["true", "false"],
                         r(k, D).bfloat16(), r(k, D).bfloat16(), r(k, D).bfloat16(), r(D).bfloat16(), r(D).bfloat16(),
                         r(k) - 3, torch.full((k,), 2.0), "digits" if qtype == "score" else "option_n", bank, label)


def synthetic(n: int, seed: int, learnable: bool = True) -> list[ProxyRequest]:
    """The gold option's fork feature carries a fixed direction, so a head can learn it."""
    g = torch.Generator().manual_seed(seed)
    sig = torch.randn(D, generator=torch.Generator().manual_seed(99))
    out = []
    for i in range(n):
        qs = []
        for j, (qt, k) in enumerate((("choice", 4), ("noul", 2), ("score", 3))[: 1 + i % 3]):
            y = int(torch.randint(k, (1,), generator=g))
            q = _q(g, f"q{j}", qt, k, y)
            if learnable:
                q.fork = (q.fork.float() + torch.zeros(k, D).index_fill_(0, torch.tensor([y]), 1) * sig).bfloat16()
            qs.append(q)
        out.append(ProxyRequest(f"r{i}", "train", qs, {"state": torch.randn(5, D, generator=g).bfloat16()}))
    return out


@pytest.mark.parametrize("rid", [r["id"] for r in RECIPES if r["arm"] == "stock" and not r.get("kd")])
def test_every_recipe_builds_trains_and_is_deterministic(rid):
    rec = dict(next(r for r in RECIPES if r["id"] == rid))
    if rec.get("train_rows"):
        rec["train_rows"] = 12  # matched-data controls (R14 / R15) on the synthetic set's first 12 rows
    data = synthetic(24, 0)
    budget = {**PL.TRAIN, "steps": 6, "grad_accum": 4, "warmup_steps": 2}
    h1, i1 = PL.train_head(rec, data, seed=3, budget=budget)
    h2, i2 = PL.train_head(rec, data, seed=3, budget=budget)
    assert i1["steps"] == 6 and i1["loss_last"] == i2["loss_last"]
    for (n, a), (_, b) in zip(h1.state_dict().items(), h2.state_dict().items()):
        assert torch.equal(a, b), n
    assert h1.attention_free == (rec["family"] in CANDIDATE_FAMILIES)
    for r in data[:3]:
        for lp, q in zip(h1(r), r.questions):
            assert lp.shape == (q.k,) and abs(float(lp.exp().sum()) - 1) < 1e-4


def test_mixer_learns_the_signal_and_beats_chance():
    data, dev = synthetic(120, 0), synthetic(60, 1)
    head, _ = PL.train_head(RECIPES[1], data, seed=0, budget={**PL.TRAIN, "steps": 90, "grad_accum": 4})
    assert PL.nll(head, dev)["acc"] > 0.8


@pytest.mark.parametrize("cls", [AFMixer, AttnRef, ForkProxy])
def test_option_permutation_equivariance(cls):
    torch.manual_seed(0)
    head = cls(D).eval()
    r = synthetic(1, 5, learnable=False)[0]
    q = r.questions[0]  # choice, 4 options (score levels are ordinal and never permuted)
    perm = torch.tensor([2, 0, 3, 1])
    q2 = ProxyQuestion(q.qid, q.qtype, [q.option_ids[i] for i in perm], q.fork[perm], q.e[perm], q.lex[perm], q.q_pre,
                       q.q_end, q.lm_lp[perm], q.lm_ntok[perm], q.key_class, q.bank_key)
    a = head(ProxyRequest("a", "x", [q], r.banks))[0]
    b = head(ProxyRequest("b", "x", [q2], r.banks))[0]
    assert torch.allclose(a[perm], b, atol=1e-5)


def test_mixer_field_order_invariance_and_field_mixing_matters():
    torch.manual_seed(0)
    head = AFMixer(D).eval()
    r = synthetic(3, 7, learnable=False)[2]  # three questions
    a = {q.qid: lp for q, lp in zip(r.questions, head(r))}
    rr = ProxyRequest("x", "x", list(reversed(r.questions)), r.banks)
    b = {q.qid: lp for q, lp in zip(rr.questions, head(rr))}
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5)
    alone = head(ProxyRequest("y", "x", r.questions[:1], r.banks))[0]
    assert not torch.allclose(alone, a[r.questions[0].qid], atol=1e-4)  # other fields change the decision


def test_routing_is_a_recurrence_not_attention():
    """No normalisation over the evidence: doubling every chunk vector changes the routed state (an attention-weighted
    average of v would only change through v itself); and the mixer's ablation flag removes the bank's influence."""
    torch.manual_seed(0)
    head = AFMixer(D).eval()
    r = synthetic(1, 3, learnable=False)[0]
    base = head(r)[0]
    r2 = ProxyRequest("z", "x", r.questions, {"state": torch.randn(9, D).bfloat16()})
    assert not torch.allclose(base, head(r2)[0])
    off = make_proxy_head({"family": "mixer", "no_route": True}, D, 0).eval()
    assert torch.allclose(off(r)[0], off(r2)[0])


def test_fork_proxy_is_the_m2_fork_head():
    from heads.features import HeadInput
    from heads.heads import FinalStateFork

    head = make_proxy_head({"family": "fork"}, D, 4).eval()
    assert isinstance(head.head, FinalStateFork)
    q = synthetic(1, 2, learnable=False)[0].questions[0]
    f = q.fork.float()[None]
    x = HeadInput(h=f.new_zeros((1, 1, D)), token_mask=torch.ones(1, 1, dtype=torch.bool),
                  evidence_mask=torch.ones(1, 1, dtype=torch.bool), q_pre=q.q_pre.float()[None],
                  q_end=q.q_end.float()[None], e=q.e.float()[None], option_mask=torch.ones(1, q.k, dtype=torch.bool),
                  fork=f, ks=[q.k])
    assert torch.allclose(head.logits(ProxyRequest("a", "x", [q]))[0], head.head(x).logits[0])


def test_write_results_kit_rows_and_unanswerable_rows_are_errors(tmp_path):
    reqs = synthetic(2, 0, learnable=False)
    rows = {}
    for r in reqs:
        qs = {}
        for q in r.questions:
            if q.qtype == "noul":
                qs[q.qid] = {"type": "noul", "instructions": "?"}
            elif q.qtype == "score":
                qs[q.qid] = {"type": "score", "instructions": "?", "criteria": ["a", "b", "c"]}
            else:
                qs[q.qid] = {"type": "choice", "instructions": "?", "criteria": {o: None for o in q.option_ids}}
        rows[r.rid] = {"questions": qs, "_evaluation": {"run_id": r.rid, "catalog_id": 1}}
    rows["missing"] = {"questions": {}, "_evaluation": {"run_id": "missing", "catalog_id": 1}}
    reqs[1].skipped = {"q9": "T > dev_max_t"}
    preds = PL.stock_predictions(reqs)
    c = PL.write_results(tmp_path / "r.jsonl", rows, reqs, preds, "t")
    assert c == {"ok": 1, "error": 2}
    lines = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = lines[0]
    assert ok["status"] == "ok" and ok["response"]["answers"]["q0"]["type"] == "choice"
    p = ok["response"]["answers"]["q0"]["probabilities"]
    assert abs(sum(p.values()) - 1) < 1e-3
    assert {x["status"] for x in lines[1:]} == {"error"}


def test_train_budget_is_fixed_steps_and_first_rows_is_exact():
    data = synthetic(30, 0)  # 1-3 questions per request
    rows = PL.first_rows(data, 17)
    assert sum(len(r.questions) for r in rows) == 17
    assert [q.qid for r in rows for q in r.questions] == [q.qid for r in data for q in r.questions][:17]
    with pytest.raises(ValueError):
        PL.first_rows(data, 10**6)
    _, info = PL.train_head({**RECIPES[11], "train_rows": 17}, data, 0, {**PL.TRAIN, "steps": 7, "grad_accum": 3})
    assert info["steps"] == 7 and info["train_questions"] == 17
    assert PL.TRAIN["steps"] == 2000 and PL.TRAIN["grad_accum"] == 8  # A1(b)


# ------------------------------------------------------------------ single-shot HOLDOUT (review B2)
def _fake_receipts(out, ids, budget=PL.TRAIN, cache_ident=None):
    for rid in ids:
        rec = next(r for r in RECIPES if r["id"] == rid)
        d = out / f"{rid}_s0_select"
        d.mkdir(parents=True)
        (d / "receipt.json").write_text(json.dumps({
            "recipe": rec, "budget_sha256": PL.budget_sha(budget), "cache_identity": cache_ident or {},
            "amendments": list(R.AMENDMENTS),
            "teacher": {"teacher_sha256": R.TEACHER["sha256"],
                        "selection_drawn_ids_sha256": R.TEACHER["drawn_ids_sha256"]} if rec.get("kd") else None,
            "score": {"all": {"point": 40.0 + int(rid[1:]), "lo": 0, "hi": 1}}, "results": "x",
            "results_sha256": "0", "head_sha256": "0"}), encoding="utf-8")


def test_select_summary_requires_every_candidate_and_is_written_once(tmp_path, monkeypatch):
    monkeypatch.setattr(PL, "cache_identity", lambda cache, arms: {})
    monkeypatch.setattr(PL, "compare", lambda *a, **k: {"diff": {"point": 0.0}})
    _fake_receipts(tmp_path, ["R00"] + [f"R{i:02d}" for i in range(1, 15)])  # R15-R17 missing
    with pytest.raises(PL.OutOfOrder, match="R15"):
        PL.select_summary(tmp_path, "cache")
    _fake_receipts(tmp_path, ["R15", "R16"])
    with pytest.raises(PL.OutOfOrder, match="R17"):  # A2: the KD candidates are candidates like any other
        PL.select_summary(tmp_path, "cache")
    _fake_receipts(tmp_path, ["R17"])
    doc = PL.select_summary(tmp_path, "cache")
    assert doc["chosen"] == "R17" and doc["candidates"] == list(R.candidate_ids())
    with pytest.raises(PL.OutOfOrder, match="once"):
        PL.select_summary(tmp_path, "cache")


def test_select_summary_refuses_mixed_budgets_and_a_changed_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(PL, "compare", lambda *a, **k: {"diff": {"point": 0.0}})
    ids = [f"R{i:02d}" for i in range(18)]
    _fake_receipts(tmp_path, ids, budget={**PL.TRAIN, "steps": 5})
    monkeypatch.setattr(PL, "cache_identity", lambda cache, arms: {})
    with pytest.raises(PL.OutOfOrder, match="budget"):
        PL.select_summary(tmp_path, "cache")
    t2 = tmp_path / "b"
    _fake_receipts(t2, ids, cache_ident={"stock": {"train": "a"}, "conv1": {}, "e5": {}})
    monkeypatch.setattr(PL, "cache_identity", lambda cache, arms: {"stock": {"train": "CHANGED"}, "conv1": {}, "e5": {}})
    with pytest.raises(PL.OutOfOrder, match="cache identity"):
        PL.select_summary(t2, "cache")


# ------------------------------------------------------------------ HOLDOUT hardening (re-review N1 / N2 / N3)
def _select_world(out, monkeypatch, chosen_bias="R03"):
    """A complete, authentic SELECT stage on disk: one receipt dir per recipe (results.jsonl + head.pt + receipt with
    their sha256s), select_summary written by PL.select_summary itself, a cache with stock holdout / train manifests,
    dev rows and Clef-flash results covering every HOLDOUT id."""
    import gzip

    monkeypatch.setattr(PL, "cache_identity", lambda cache, arms: {a: {"m": "1"} for a in arms})
    monkeypatch.setattr(PL, "compare", lambda *a, **k: {"diff": {"point": 0.0}})
    for r in RECIPES:
        d = out / f"{r['id']}_s0_select"
        d.mkdir(parents=True)
        (d / "results.jsonl").write_text(r["id"], encoding="utf-8")
        (d / "head.pt").write_bytes(r["id"].encode())
        pt = 40.0 + (5.0 if r["id"] == chosen_bias else 0.0) + int(r["id"][1:]) / 100
        (d / "receipt.json").write_text(json.dumps({
            "recipe": r, "budget_sha256": PL.budget_sha(PL.TRAIN), "cache_identity": {r["arm"]: {"m": "1"}},
            "amendments": list(R.AMENDMENTS),
            "teacher": {"teacher_sha256": R.TEACHER["sha256"],
                        "selection_drawn_ids_sha256": R.TEACHER["drawn_ids_sha256"]} if r.get("kd") else None,
            "score": {"all": {"point": pt, "lo": 0, "hi": 1}}, "results": str(d / "results.jsonl"),
            "results_sha256": PL._sha(d / "results.jsonl"), "head_sha256": PL._sha(d / "head.pt")}), encoding="utf-8")
    PL.select_summary(out, "cache")
    cache = out.parent / "cache"
    for sp in ("holdout", "train"):
        (cache / "stock" / sp).mkdir(parents=True, exist_ok=True)
        (cache / "stock" / sp / "MANIFEST.jsonl").write_text("{}\n", encoding="utf-8")
    ids = PL.subset_ids(R.HOLDOUT_SUBSET)
    dev = out.parent / "dev.jsonl.gz"
    with gzip.open(dev, "wt", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"_evaluation": {"run_id": i, "catalog_id": 1}, "questions": {}}) + "\n")
    clef = out.parent / "clef.jsonl"
    clef.write_text("".join(json.dumps({"run_id": i, "status": "ok"}) + "\n" for i in ids), encoding="utf-8")
    return str(cache), str(dev), str(clef)


def _fake_run(calls, counts, point, fail_seed=None):
    def run(recipe, cache, dev_rows, out, split, subset, seed, *a, **k):
        calls.append((recipe["id"], split, seed))
        if seed == fail_seed:
            raise RuntimeError("simulated crash mid-HOLDOUT")
        rec = {"seed": seed, "counts": counts, "score": {"all": {"point": point}, "0.3": {"all": {"point": point}},
                                                         R.VERDICT_EDITION: {"vs_clef_flash": None,
                                                                             **{c: {} for c in R.DOMAIN_COLUMNS}}}}
        d = out / f"{recipe['id']}_s{seed}_{split}"
        d.mkdir()
        (d / "receipt.json").write_text(json.dumps(rec), encoding="utf-8")
        return rec
    return run


FULL = {"ids": 2014, "rows": 2014, "cached": 2014}


def test_holdout_runs_once_for_the_rederived_choice(tmp_path, monkeypatch):
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    calls = []
    monkeypatch.setattr(PL, "run_recipe", _fake_run(calls, FULL, 51.0))
    with pytest.raises(PL.OutOfOrder, match="only the chosen recipe"):
        PL.holdout(out, cache, dev, clef, 10, recipe="R11")
    assert not list(out.glob("*_holdout")) and not (out / "holdout_attempt.json").exists()
    v = PL.holdout(out, cache, dev, clef, 10, recipe="R03")
    assert v["verdict"] == "go" and v["chosen"] == "R03" and [c[0] for c in calls] == ["R03"] * 3
    assert v["scope"] == R.VERDICT_SCOPE and v["holdout_attempt_sha256"] == PL._sha(out / "holdout_attempt.json")
    with pytest.raises(PL.OutOfOrder, match="once"):
        PL.holdout(out, cache, dev, clef, 10)


def test_holdout_rederives_the_choice_and_refuses_a_hand_edited_summary(tmp_path, monkeypatch):
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    sp = out / "select_summary.json"
    doc = json.loads(sp.read_text(encoding="utf-8"))
    sp.write_text(json.dumps({**doc, "chosen": "R11"}), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="own rows choose 'R03'"):
        PL.holdout(out, cache, dev, clef, 10)
    rows = [{**r, "select": 99.0} if r["id"] == "R11" else r for r in doc["rows"]]  # inflate a score too
    sp.write_text(json.dumps({**doc, "chosen": "R11", "rows": rows}), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="R11: select_summary SELECT score"):  # consistent but forged
        PL.holdout(out, cache, dev, clef, 10)
    assert not list(out.glob("*_holdout"))


def test_holdout_checks_receipts_and_files_against_the_summary(tmp_path, monkeypatch):
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    (out / "R07_s0_select" / "head.pt").write_bytes(b"swapped")
    with pytest.raises(PL.OutOfOrder, match="R07: the SELECT results / head files"):
        PL.holdout(out, cache, dev, clef, 10)
    t2 = tmp_path / "p"
    cache, dev, clef = _select_world(t2, monkeypatch)
    rec = t2 / "R05_s0_select" / "receipt.json"
    rec.write_text(json.dumps({**json.loads(rec.read_text(encoding="utf-8")), "head_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="R05: select_summary head_sha256"):
        PL.holdout(t2, cache, dev, clef, 10)


def test_holdout_refuses_changed_code(tmp_path, monkeypatch):
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    real = PL.code_identity()
    monkeypatch.setattr(PL, "code_identity", lambda: {**real, "train/proxy_heads.py": "edited"})
    with pytest.raises(PL.OutOfOrder, match="code_identity"):
        PL.holdout(out, cache, dev, clef, 10)


def test_holdout_refuses_partial_coverage_and_a_budget_mismatch(tmp_path, monkeypatch):
    """N3: the holdout-side refusals (select_summary's own are tested above)."""
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    sp = out / "select_summary.json"
    doc = json.loads(sp.read_text(encoding="utf-8"))
    sp.write_text(json.dumps({**doc, "rows": [r for r in doc["rows"] if r["id"] != "R12"]}), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match=r"no SELECT score for \['R12'\]"):
        PL.holdout(out, cache, dev, clef, 10)
    sp.write_text(json.dumps({**doc, "candidates": doc["candidates"][:-1]}), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="every candidate"):
        PL.holdout(out, cache, dev, clef, 10)
    sp.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="budget"):
        PL.holdout(out, cache, dev, clef, 10, budget={**PL.TRAIN, "steps": 1999})
    assert not list(out.glob("*_holdout")) and not (out / "holdout_attempt.json").exists()


def test_holdout_validates_every_input_before_any_training_or_directory(tmp_path, monkeypatch):
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    calls = []
    monkeypatch.setattr(PL, "run_recipe", _fake_run(calls, FULL, 51.0))
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(PL.OutOfOrder, match="--clef"):
        PL.holdout(out, cache, dev, str(tmp_path / "missing.jsonl"), 10)
    short = tmp_path / "clef_short.jsonl"
    short.write_text("".join(Path(clef).read_text(encoding="utf-8").splitlines(keepends=True)[:-3]), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="miss 3 HOLDOUT ids"):
        PL.holdout(out, cache, dev, str(short), 10)
    with pytest.raises(PL.OutOfOrder, match="--dev-rows"):
        PL.holdout(out, cache, str(tmp_path / "nope.gz"), clef, 10)
    with pytest.raises(PL.OutOfOrder, match="no stock holdout"):
        PL.holdout(out, str(tmp_path / "empty_cache"), dev, clef, 10)
    assert calls == [] and sorted(p.name for p in out.iterdir()) == before  # nothing created, nothing trained


def test_a_crash_after_validation_resumes_the_same_attempt(tmp_path, monkeypatch):
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    calls = []
    monkeypatch.setattr(PL, "run_recipe", _fake_run(calls, FULL, 49.0, fail_seed=1))
    with pytest.raises(RuntimeError, match="simulated crash"):
        PL.holdout(out, cache, dev, clef, 10)
    assert (out / "holdout_attempt.json").exists() and not (out / "verdict.json").exists()
    assert (out / "R03_s0_holdout").exists() and not (out / "R03_s1_holdout").exists()
    # changed input after the attempt was bound: refused
    Path(clef).write_text(Path(clef).read_text(encoding="utf-8") + json.dumps({"run_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="holdout_attempt.json binds"):
        PL.holdout(out, cache, dev, clef, 10)
    Path(clef).write_text("".join(Path(clef).read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    calls.clear()
    monkeypatch.setattr(PL, "run_recipe", _fake_run(calls, FULL, 49.0))
    v = PL.holdout(out, cache, dev, clef, 10)
    assert [c[2] for c in calls] == [1, 2]  # seed 0 kept from the first attempt; no re-choice
    assert v["verdict"] == "escalate" and v["escalate_to"] == "user" and v["chosen"] == "R03"


def test_holdout_dirs_without_an_attempt_are_refused(tmp_path, monkeypatch):
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    (out / "R03_s0_holdout").mkdir()
    with pytest.raises(PL.OutOfOrder, match="without a holdout_attempt.json"):
        PL.holdout(out, cache, dev, clef, 10)


def test_holdout_incomplete_split_is_no_measurement(tmp_path, monkeypatch):
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    monkeypatch.setattr(PL, "run_recipe", _fake_run([], {"ids": 2014, "rows": 2010, "cached": 2010}, 60.0))
    assert PL.holdout(out, cache, dev, clef, 10)["verdict"] == "no-measurement"


def test_run_recipe_receipt_dir_appears_only_when_complete(tmp_path, monkeypatch):
    """N2: a crash inside run_recipe leaves only a *.partial-* directory, never a *_holdout receipt."""
    monkeypatch.setattr(PL, "load_split", lambda *a, **k: synthetic(4, 0))
    monkeypatch.setattr(PL, "subset_ids", lambda subset: ["a"])
    monkeypatch.setattr(PL, "_dev_rows", lambda dev, ids: {})
    monkeypatch.setattr(PL, "cache_identity", lambda cache, arms: {})

    def boom(*a, **k):
        raise RuntimeError("scoring crashed")

    monkeypatch.setattr(PL, "report", boom)
    with pytest.raises(RuntimeError, match="scoring crashed"):
        PL.run_recipe({**RECIPES[11]}, "c", "d", tmp_path, "holdout", R.HOLDOUT_SUBSET, 0, "clef", None, 10,
                      {**PL.TRAIN, "steps": 1, "grad_accum": 2})
    assert not list(tmp_path.glob("*_holdout")) and len(list(tmp_path.glob("*_holdout.partial-*"))) == 1
    monkeypatch.setattr(PL, "report", lambda *a, **k: {"all": {"point": 1.0}})
    rec = PL.run_recipe({**RECIPES[11]}, "c", "d", tmp_path, "holdout", R.HOLDOUT_SUBSET, 0, "clef", None, 10,
                        {**PL.TRAIN, "steps": 1, "grad_accum": 2})
    d = tmp_path / "R11_s0_holdout"
    assert (d / "receipt.json").exists() and rec["results"] == str(d / "results.jsonl")
    assert rec["head_sha256"] == PL._sha(d / "head.pt") and rec["code_identity"] == PL.code_identity()


def test_holdout_cli_requires_clef_and_dev_rows_and_an_existing_out(tmp_path):
    with pytest.raises(SystemExit):
        PL.main(["holdout", "--cache", "c", "--out", str(tmp_path)])
    assert PL.main(["holdout", "--cache", "c", "--dev-rows", "d", "--clef", "x", "--out", str(tmp_path / "new")]) == 4
    assert not (tmp_path / "new").exists()


def test_recipes_only_never_writes_select_summary(tmp_path, monkeypatch):
    (tmp_path / "stock_check.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    monkeypatch.setattr(PL, "run_recipe", lambda *a, **k: {"score": {"all": {"point": 1.0}}})
    assert PL.main(["recipes", "--cache", "c", "--dev-rows", "d", "--out", str(tmp_path), "--only", "R01,R11"]) == 0
    assert not (tmp_path / "select_summary.json").exists()


def test_recipes_and_holdout_need_a_passing_stock_check(tmp_path):
    assert PL.main(["recipes", "--cache", "c", "--out", str(tmp_path)]) == 4
    (tmp_path / "stock_check.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
    assert PL.main(["holdout", "--cache", "c", "--dev-rows", "d", "--clef", "x", "--out", str(tmp_path)]) == 4


def test_holdout_refuses_a_summary_made_under_other_rule_amendments(tmp_path, monkeypatch):
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch)
    sp = out / "select_summary.json"
    doc = json.loads(sp.read_text(encoding="utf-8"))
    assert doc["amendments"] == list(R.AMENDMENTS)
    sp.write_text(json.dumps({**doc, "amendments": []}), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="amendments"):
        PL.holdout(out, cache, dev, clef, 10)
    sp.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(R, "AMENDMENTS", tuple(R.AMENDMENTS) + ("A2",))  # the rule gained an amendment since SELECT
    with pytest.raises(PL.OutOfOrder, match="amendments"):
        PL.holdout(out, cache, dev, clef, 10)
    assert not list(out.glob("*_holdout")) and not (out / "holdout_attempt.json").exists()


def test_a3_stock_check_wiring_masks_reference_failures(tmp_path, monkeypatch):
    """A3: stock-check needs the pinned --stock, scores the cached readout with the reference's failed rows masked,
    reports both deltas and the masked ids, and fails closed on a wrong reference file."""
    ids = [f"r{i}" for i in range(40)]
    ref = tmp_path / "ref.jsonl"
    ref.write_text("".join(json.dumps({"run_id": i, "status": "error" if i == "r3" else "ok"}) + "\n" for i in ids),
                   encoding="utf-8")
    monkeypatch.setattr(PL, "subset_ids", lambda subset: list(ids))
    monkeypatch.setattr(PL, "load_split", lambda *a, **k: [])
    monkeypatch.setattr(PL, "stock_predictions", lambda reqs: {})
    monkeypatch.setattr(PL, "_dev_rows", lambda path, want: {})

    cached_status = {"r3": "ok"}

    def fake_write(path, dev_rows, reqs, preds, model):
        Path(path).write_text("".join(json.dumps({"run_id": i, "status": cached_status.get(i, "ok"), "response": {
            "answers": {"q": {"type": "noul", "noul": 0.5}}}}) + "\n" for i in ids), encoding="utf-8")

    def fake_score(path, subset, B):  # index = 38.27 + 0.4 * (cached-ok rows the reference did not answer)
        rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]
        return {"slice_index": {"index": {"point": 38.27 + 0.4 * (rows[3]["status"] == "ok")}}}

    monkeypatch.setattr(PL, "write_results", fake_write)
    monkeypatch.setattr(PL, "score", fake_score)
    monkeypatch.setattr(PL, "compare", lambda c, b, s, B: {"candidate": c})
    out = tmp_path / "o"
    out.mkdir()
    with pytest.raises(PL.OutOfOrder, match="--stock"):
        PL.stock_check(out, "c", "d", None, 10)
    with pytest.raises(PL.OutOfOrder, match="pinned reference"):
        PL.stock_check(out, "c", "d", str(ref), 10)
    monkeypatch.setattr(R, "STOCK_REFERENCE_SHA256", PL._sha(ref))
    monkeypatch.setattr(R, "STOCK_CHECK_MASKED_N", 1)
    monkeypatch.setattr(R, "STOCK_CHECK_MASKED_IDS_SHA256", R.masked_ids_sha256(["r3"]))
    rep = PL.stock_check(out, "c", "d", str(ref), 10)
    assert rep["ok"] and abs(rep["delta"]) < 1e-9 and abs(rep["unmasked_delta"] - 0.4) < 1e-9
    assert rep["masked"] == {"n": 1, "n_rows": 40, "frac": 1 / 40, "max_frac": 0.05,
                             "ids_sha256": R.masked_ids_sha256(["r3"]), "ids": ["r3"]}
    assert rep["masked_ids_pinned_ok"] and rep["masked_rows_valid"] and rep["masked_problems"] == []
    assert rep["by_reference_status"] == {"error": {"n": 1, "point_only_this_masked": 38.27,
                                                    "contribution": rep["unmasked_point"] - 38.27}}
    assert rep["vs_job_5cmat"]["candidate"].endswith("stock_from_cache_select_masked.jsonl")
    assert rep["vs_job_5cmat_unmasked"]["candidate"].endswith("stock_from_cache_select.jsonl")
    assert json.loads((out / "stock_check.json").read_text(encoding="utf-8")) == rep
    # three failed reference rows of 40 = 7.5% > 5%: fail closed although the masked delta is 0
    ref.write_text("".join(json.dumps({"run_id": i, "status": "error" if i in ("r3", "r4", "r5") else "ok"}) + "\n"
                           for i in ids), encoding="utf-8")
    monkeypatch.setattr(R, "STOCK_REFERENCE_SHA256", PL._sha(ref))
    rep = PL.stock_check(out, "c", "d", str(ref), 10)
    assert not rep["ok"] and rep["within_tol"] and not rep["mask_cap_ok"]
    # A3(d): a masked row whose CACHED answer is broken fails closed, although the masked delta is 0
    ref.write_text("".join(json.dumps({"run_id": i, "status": "error" if i == "r3" else "ok"}) + "\n" for i in ids),
                   encoding="utf-8")
    monkeypatch.setattr(R, "STOCK_REFERENCE_SHA256", PL._sha(ref))
    cached_status["r3"] = "error"
    rep = PL.stock_check(out, "c", "d", str(ref), 10)
    assert not rep["ok"] and not rep["masked_rows_valid"] and rep["masked_problems"]
    with pytest.raises(PL.OutOfOrder, match="stock-check failed"):
        PL.require_stock_check(out)


def test_from_record_keeps_the_longest_model_input():
    q = {"qid": "q", "qtype": "choice", "option_ids": ["A", "B"], "fork": None, "e": None, "lex": None, "q_pre": None,
         "q_end": None, "lm_lp": PL.torch.zeros(2), "lm_ntok": PL.torch.ones(2), "key_class": "letter", "T": 17000}
    from train.proxy_heads import from_record

    r = from_record({"rid": "x", "split": "holdout", "questions": [q, {**q, "qid": "q2", "T": 900}]})
    assert r.max_tokens == 17000
    q.pop("T")
    assert from_record({"rid": "x", "split": "holdout", "questions": [q]}).max_tokens is None


def test_a3_long_rows_report(tmp_path, monkeypatch):
    """Report-only: how many HOLDOUT rows exceed 16,384 model tokens and what each comparator scored on them."""
    ids = [f"r{i}" for i in range(6)]
    long_ids = ["r1", "r4"]
    files = {}
    for name, bad in (("candidate", set()), ("stock_from_cache", set()), ("clef_flash", {"r4"})):
        f = tmp_path / f"{name}.jsonl"
        f.write_text("".join(json.dumps({"run_id": i, "status": "unsupported" if i in bad else "ok"}) + "\n"
                             for i in ids), encoding="utf-8")
        files[name] = str(f)
    sub = tmp_path / "sub.json"
    sub.write_text(json.dumps({"run_ids": ids}), encoding="utf-8")
    monkeypatch.setattr(PL, "subset_file", lambda base, keep, out: json.dumps([i for i in ids if keep(i, 1)]))
    worth = {"candidate": 1.0, "stock_from_cache": 0.5, "clef_flash": 0.25}

    def fake_score(path, subset, B):
        name = Path(path).stem
        return {"slice_index": {"index": {"point": 40.0 + (worth[name] if subset == str(sub) else 0.0)}}}

    monkeypatch.setattr(PL, "score", fake_score)
    monkeypatch.setattr(PL, "compare", lambda c, b, s, B: {"diff": {"pair": [Path(c).stem, Path(b).stem], "subset": s}})
    rep = PL.long_rows_report(files, long_ids, str(sub), tmp_path, 10)
    assert rep["n"] == 2 and rep["ids"] == long_ids and rep["max_input_tokens"] == 16384
    c = rep["comparators"]
    assert c["candidate"]["long_rows_worth"] == 1.0 and c["stock_from_cache"]["long_rows_worth"] == 0.5
    assert c["clef_flash"]["status_on_long_rows"] == {"ok": 1, "unsupported": 1, "missing": 0}
    d = rep["candidate_vs_clef_flash_without_long_rows"]
    assert d["pair"] == ["candidate", "clef_flash"] and json.loads(d["subset"]) == ["r0", "r2", "r3", "r5"]
    assert PL.long_rows_report(files, [], str(sub), tmp_path, 10) == {"max_input_tokens": 16384, "n": 0, "ids": []}
