"""Score-item invariance check (scripts/ev_score_perturb.py + scripts/ev_score_flip.py), synthetic items only."""
from __future__ import annotations

import hashlib
import json
import math

import pytest

from jevkit.items import Item, write_items
from scripts import ev_score_flip as SF
from scripts import ev_score_perturb as SP

RUB = ["no impact", "minor impact, workaround exists", "major impact on a core function", "total outage, data loss"]


def _items() -> list[Item]:
    out = []
    for i in range(6):
        k = 4 if i < 5 else 5
        crit = RUB + ["catastrophic"] if k == 5 else list(RUB)
        out.append(Item(id=f"sc{i}", family="severity", state=f"incident report {i}: users affected {i * 7 % 11}",
                        question={"type": "score", "instructions": "Rate the severity.", "criteria": crit},
                        labels=[str(v) for v in range(k)], expected=i % k, split="public").validate())
    out.append(Item(id="ch0", family="routing", state="s", question={"type": "choice", "instructions": "pick",
                    "criteria": {"a": "x", "b": "y"}}, labels=["a", "b"], expected="a", split="public").validate())
    return out


def _softmax_by_content(z: dict[str, float], content: dict[str, str]) -> dict[str, float]:
    """Normaliser that sums in content-sorted order (C3 + section 4: bitwise under any display order / handles)."""
    m = max(z.values())
    e = {h: math.exp(z[h] - m) for h in z}
    s = 0.0
    for h in sorted(z, key=lambda h: content[h]):
        s += e[h]
    return {h: e[h] / s for h in z}


def _target(state: str) -> float:
    return int(hashlib.sha256(state.encode()).hexdigest()[:4], 16) % 400 / 100.0


def content_head(row: dict) -> dict:
    """Reads the level from CONTENT only ('level v of K: ...')."""
    opts = row["question"]["options"]
    z = {}
    for h, c in opts.items():
        v = int(c.split(" ", 2)[1])
        z[h] = -abs(v - _target(row["state"])) + 0.01 * (len(c) % 7)
    return {"type": "score", "probabilities": _softmax_by_content(z, opts)}


def handle_head(row: dict) -> dict:
    """Reads the level from the HANDLE (int(handle); fresh handles: their display slot)."""
    opts = row["question"]["options"]
    z = {}
    for i, h in enumerate(opts):
        v = int(h) if h.isdigit() else i
        z[h] = -abs(v - _target(row["state"]))
    return {"type": "score", "probabilities": _softmax_by_content(z, opts)}


def _preds(rows, head) -> list[dict]:
    return [{"id": r["id"], "answer": head(r), "status": "ok"} for r in rows]


def _run(tmp_path, head, ops=SP.OPS, sort_keys=False, check_ops=SP.OPS):
    items = _items()
    base, pert = SP.make(items, ops)
    d = tmp_path / "req"
    d.mkdir()
    SP.write_rows(base, d / "base.jsonl")
    if sort_keys:
        (d / "perturbed.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pert), encoding="utf-8")
    else:
        SP.write_rows(pert, d / "perturbed.jsonl")
    disk_b, disk_p = SF.rj(d / "base.jsonl"), SF.rj(d / "perturbed.jsonl")
    # the head answers the requests AS READ FROM DISK (what a job would serve)
    for name, rows in (("base.preds.jsonl", disk_b), ("perturbed.preds.jsonl", disk_p)):
        (d / name).write_text("".join(json.dumps(p) + "\n" for p in _preds(rows, head)), encoding="utf-8")
    return SF.run_dir(items, d, ops=check_ops)


def test_counts_and_rendering():
    base, pert = SP.make(_items())
    assert len(base) == 6  # the choice item is ignored
    ops = [r["perturbation"]["op"] for r in pert]
    assert {o: ops.count(o) for o in SP.OPS} == {"score_reverse": 6, "score_shuffle": 18, "score_relabel": 6,
                                                 "score_permute": 6}
    b = base[0]
    assert b["question"]["scale"][2] == "level 2 of 4: major impact on a core function"
    assert list(b["question"]["options"]) == ["0", "1", "2", "3"]
    for r in pert:  # scale (shared prefix) canonical and identical to base; only order / handles differ
        bb = next(x for x in base if x["id"] == r["perturbation"]["base_id"])
        assert r["question"]["scale"] == bb["question"]["scale"]
        assert sorted(r["question"]["options"].values()) == sorted(bb["question"]["options"].values())
        assert r["labels"] == list(r["question"]["options"])
    assert SP.make(_items()) == (base, pert)  # deterministic


def test_content_head_passes_bitwise(tmp_path):
    r = _run(tmp_path, content_head)
    assert r["score_inv_pass"], r["failures"]
    assert r["bitwise"] and r["max_abs_dp_observed"] == 0.0
    for op in SP.OPS:
        v = r["per_operator"][op]
        assert v["n"] > 0 and v["flips"] == 0 and v["max_abs_dE"] == 0.0
        assert v["changed_vs_base"] == v["n_rows"] == v["matches_regenerated"] == v["content_identical"]


def test_handle_head_fails_relabel(tmp_path):
    r = _run(tmp_path, handle_head)
    assert not r["score_inv_pass"]
    pm = r["per_operator"]["score_permute"]
    assert pm["flips"] > 0 and pm["max_abs_dp"] > 0.3 and pm["max_abs_dE"] > 0.5
    assert any(f.startswith("score_permute") for f in r["failures"])
    # the handle head reads base handles "0".."K-1" -> display order alone does not move it (2T: (i) passes, (ii) fails)
    assert r["per_operator"]["score_reverse"]["flips"] == 0 and r["per_operator"]["score_reverse"]["max_abs_dp"] == 0.0


def test_sort_keys_file_fails_as_vacuous(tmp_path):
    r = _run(tmp_path, content_head, sort_keys=True)
    assert not r["score_inv_pass"]
    rev = r["per_operator"]["score_reverse"]
    assert rev["changed_vs_base"] == 0 and rev["matches_regenerated"] == 0
    assert any("score_reverse" in f and "vacuous" in f for f in r["failures"])
    assert any("in-memory order" in f for f in r["failures"])


def test_missing_op_fails_closed(tmp_path):
    r = _run(tmp_path, content_head, ops=("score_reverse", "score_shuffle", "score_relabel"))
    assert not r["score_inv_pass"]
    assert r["per_operator"]["score_permute"]["n"] == 0
    assert any(f.startswith("score_permute: no valid pairs") for f in r["failures"])


def test_invalid_answer_fails(tmp_path):
    items = _items()
    base, pert = SP.make(items)
    bp = _preds(base, content_head)
    pp = _preds(pert, content_head)
    pp[0]["answer"] = {"type": "score", "probabilities": {"0": 1.0}}  # wrong label set
    r = SF.score_inv_check(items, base, pert, bp, pp)
    assert not r["score_inv_pass"] and any("invalid" in f for f in r["failures"])


def test_label_map_round_trip():
    items = _items()
    base, pert = SP.make(items)
    bmap = {r["id"]: r for r in base}
    for r in pert:
        b = bmap[r["perturbation"]["base_id"]]
        lm = r["perturbation"]["label_map"]
        assert sorted(lm) == b["labels"] and sorted(lm.values()) == sorted(r["labels"])  # a bijection onto the handles
        for lab, h in lm.items():  # the handle shows exactly the base level's content
            assert r["question"]["options"][h] == b["question"]["options"][lab]
            assert r["question"]["options"][h].startswith(f"level {lab} of {b['k']}:")
        # a point mass on any perturbed handle maps back to the level it carries
        inv = {h: lab for lab, h in lm.items()}
        for h in r["labels"]:
            d = SF.dist({"type": "score", "choice": h}, r["labels"])
            back = {lab: d[lm[lab]] for lab in b["labels"]}
            assert max(back, key=back.get) == inv[h]
        op = r["perturbation"]["op"]
        if op in SP.RELABEL_OPS:
            assert r["perturbation"]["display_order"] == b["labels"] and any(k != v for k, v in lm.items())
        else:
            assert lm == {lab: lab for lab in b["labels"]} and r["perturbation"]["display_order"] != b["labels"]


def test_write_rows_asserts_order(tmp_path, monkeypatch):
    base, _ = SP.make(_items())
    monkeypatch.setattr(SP, "dump_line", lambda row: json.dumps(row, sort_keys=True))
    bad = [dict(base[0], question=dict(base[0]["question"], options=dict(reversed(list(base[0]["question"]["options"].items())))),
                labels=list(reversed(base[0]["labels"])))]
    with pytest.raises(AssertionError):
        SP.write_rows(bad, tmp_path / "x.jsonl")


def test_cli_end_to_end(tmp_path):
    ip = tmp_path / "items.jsonl"
    write_items(_items(), ip)
    d = tmp_path / "req"
    assert SP.main(["--items", str(ip), "--out", str(d)]) == 0
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert man["n_score_items"] == 6 and man["per_op"]["score_shuffle"] == 18
    for name, src in (("base.preds.jsonl", "base.jsonl"), ("perturbed.preds.jsonl", "perturbed.jsonl")):
        (d / name).write_text("".join(json.dumps(p) + "\n" for p in _preds(SF.rj(d / src), content_head)), encoding="utf-8")
    assert SF.main(["--items", str(ip), "--dir", str(d), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "SCORE_INV.md").read_text(encoding="utf-8").startswith("# Score-item invariance (separate from G1): PASS")
    (d / "perturbed.preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in _preds(SF.rj(d / "perturbed.jsonl"), handle_head)),
                                             encoding="utf-8")
    assert SF.main(["--items", str(ip), "--dir", str(d), "--out", str(tmp_path / "o2")]) == 3


def test_cell_rescore_section_separate_from_g1(tmp_path):
    """--score-inv adds section 6b; the G1 section, verdict, flags and exit code are unchanged by it."""
    from jevkit.items import load_public
    from scripts import ev_cell_rescore as K
    from tests import test_ev_cell_rescore as T

    pub = load_public()
    art = tmp_path / "art"
    jd = art / "out" / "screen" / "jevkit"
    jd.mkdir(parents=True)
    for arm, model in (("stock_shim", T.first_listed_model), ("native_shimcal", T.invariant_model)):
        (jd / f"{arm}.preds.jsonl").write_text("".join(json.dumps(T._rec(it, model(it))) + "\n" for it in pub),
                                               encoding="utf-8")
    (art / "out" / "screen.json").write_text(json.dumps({"jevkit": {}}), encoding="utf-8")
    T.make_g1_dir(jd / "g1", T.invariant_model)
    runs = {}
    for name, head in (("none", None), ("content", content_head), ("handle", handle_head)):
        argv = ["--art", str(art), "--out", str(tmp_path / f"out_{name}"), "--price", T.PRICE, "--B", "20"]
        if head is not None:
            d = tmp_path / f"si_{name}"
            base, pert = SP.make(pub)
            d.mkdir()
            SP.write_rows(base, d / "base.jsonl")
            SP.write_rows(pert, d / "perturbed.jsonl")
            for fn, rows in (("base.preds.jsonl", base), ("perturbed.preds.jsonl", pert)):
                (d / fn).write_text("".join(json.dumps(p) + "\n" for p in _preds(rows, head)), encoding="utf-8")
            argv += ["--score-inv", str(d)]
        rc = K.main(argv)
        runs[name] = (rc, json.loads((tmp_path / f"out_{name}" / "cell_rescore.json").read_text(encoding="utf-8")))
    g1 = {k: json.dumps(v[1]["6_g1"], sort_keys=True) for k, v in runs.items()}
    assert g1["none"] == g1["content"] == g1["handle"]
    assert runs["none"][0] == runs["content"][0] == runs["handle"][0]
    assert runs["none"][1]["flags"] == runs["content"][1]["flags"] == runs["handle"][1]["flags"]
    assert runs["none"][1]["6b_score_inv"] is None
    si = runs["content"][1]["6b_score_inv"]
    assert si["score_inv_pass"] and si["bitwise"] and si["n_score_items"] == 18
    assert {op: si["per_operator"][op]["n"] for op in SP.OPS} == {"score_reverse": 18, "score_shuffle": 54,
                                                                  "score_relabel": 18, "score_permute": 18}
    assert not runs["handle"][1]["6b_score_inv"]["score_inv_pass"]
    assert any(w.startswith("score-inv FAIL") for w in runs["handle"][1]["warnings"])
    md = (tmp_path / "out_content" / "CELL_RESCORE.md").read_text(encoding="utf-8")
    assert "(6b) Score-item invariance (separate from G1" in md and "score-inv PASS" in md and "(6) G1" in md
