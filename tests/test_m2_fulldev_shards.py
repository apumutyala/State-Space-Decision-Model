"""Sharded full DEV (modes fulldev_eval_d13_v3w_s1of2 / _s2of2): the pre-declared, case-closed, token-balanced shard
plan; the CPU merge into the unsharded child.results.jsonl (and its refusals); the partial-row export + --seed-partial
resume. CPU only: no model (load_child / index_predict stubbed); the kit (decision_index) is needed only where the frozen
harness.cases clusters are built."""
import hashlib
import io
import json
import random
import sys
import tarfile
import types
from pathlib import Path

import pytest

from jobs.m2_first_cell import fulldev_eval as F

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs/m2_first_cell"
SHARD_MODES = ("fulldev_eval_d13_v3w_s1of2", "fulldev_eval_d13_v3w_s2of2")
PINNED = {1: "5aa8ef1531b4c171f34b6b8174fc04c3ec3e8c28eb4ba396663e605432bb86fe",
          2: "f523934e7b85a32317b4d59e55bfdc39dbd9de338ac90c9c440234ab2c51202c"}
DEV_IDS_SHA = "4f310d28996ca91e932c497a00c54abe4ced1277e3dd63c21a4ff4366fd7d030"
DEV_ROWS = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data\slices\dev-rows.jsonl.gz")


def _kit():
    pytest.importorskip("decision_index", reason="needs the decision-index kit (harness.cases)")
    if not sys.flags.utf8_mode:
        pytest.skip("needs python -X utf8")


def _rows(n=240, seed=0):
    """Synthetic DEV rows: several benchmarks, kit groups, and a source key (catalog 22 song_id) that links groups
    into multi-group cases, so clusters are coarser than groups."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        cat = (22, 7, 31)[i % 3]
        g = rng.randrange(60)
        rows.append({"_evaluation": {"run_id": f"r{i:04d}", "catalog_id": cat, "group_id": g,
                                     "payload_sha256": f"p{i}"},
                     "metadata": {"song_id": g % 9} if cat == 22 else {}, "state": f"s{i}", "questions": []})
    lengths = {r["_evaluation"]["run_id"]: rng.randrange(200, 9000) for r in rows}
    return rows, lengths


# ------------------------------------------------------------------------------------------------ the plan
def test_shard_plan_is_deterministic_case_closed_and_balanced():
    _kit()
    from harness.cases import case_ids

    rows, lengths = _rows()
    a = F.shard_plan(rows, lengths, 2)
    sh = rows[:]
    random.Random(7).shuffle(sh)
    b = F.shard_plan(sh, lengths, 2)  # row order does not matter
    assert [s["ids_sha256"] for s in a["shards"]] == [s["ids_sha256"] for s in b["shards"]]
    assert F.check_shard_plan(a, 2) is a
    cid = case_ids(rows)
    where = {}
    for s in a["shards"]:
        for rid in s["run_ids"]:
            e = next(r["_evaluation"] for r in rows if r["_evaluation"]["run_id"] == rid)
            c = cid[(e["catalog_id"], str(e["group_id"]))]
            assert where.setdefault(c, s["k"]) == s["k"], f"case {c} straddles shards"
    assert len(set(cid.values())) < len({(r["_evaluation"]["catalog_id"], r["_evaluation"]["group_id"]) for r in rows})
    tok = {}
    for r in rows:
        e = r["_evaluation"]
        tok[cid[(e["catalog_id"], str(e["group_id"]))]] = tok.get(cid[(e["catalog_id"], str(e["group_id"]))], 0) \
            + lengths[e["run_id"]]
    loads = [s["tokens"] for s in a["shards"]]
    assert sum(loads) == sum(lengths.values()) and max(loads) - min(loads) <= max(tok.values())
    assert F.shard_plan(rows, lengths, 2, seed="other")["shards"][0]["ids_sha256"] != a["shards"][0]["ids_sha256"]
    three = F.shard_plan(rows, lengths, 3)
    assert len(three["shards"]) == 3 and F.check_shard_plan(three, 3)


def test_parse_shard():
    assert F.parse_shard("1/2") == (1, 2) == F.parse_shard("shard1of2")
    for bad in ("0/2", "3/2", "1/1", "full", "smoke", None, "shard1of"):
        assert F.parse_shard(bad) is None


def test_committed_plan_reproduces_the_pinned_hashes_and_covers_dev():
    plan = F.load_shard_plan(2)
    assert {s["k"]: s["ids_sha256"] for s in plan["shards"]} == PINNED
    assert plan["dev_ids_sha256"] == DEV_IDS_SHA and plan["n_rows"] == F.N_DEV
    a, b = (set(s["run_ids"]) for s in plan["shards"])
    assert not a & b and F.ids_sha256(a | b) == DEV_IDS_SHA
    # balance: each shard within 0.2 % of half the DEV tokens (= its expected minutes; one rate for all rows)
    lengths = json.loads(F.LENGTHS.read_text(encoding="utf-8"))["lengths"]
    for s in plan["shards"]:
        assert sum(lengths[i] for i in s["run_ids"]) == s["tokens"]
        assert abs(s["tokens"] / plan["total_tokens"] - 0.5) < F.SHARD_TOL
    assert F.selected_ids("shard1of2") == a and F.selected_ids("shard2of2") == b
    assert F.selected_ids("full") is None


def test_committed_plan_rebuilds_from_the_real_dev_rows():
    _kit()
    if not DEV_ROWS.is_file():
        pytest.skip("no local DEV rows")
    from harness.suite import read_rows

    rows = read_rows(str(DEV_ROWS))
    lengths = json.loads(F.LENGTHS.read_text(encoding="utf-8"))["lengths"]
    for k in (1, 2):
        e = F.verify_shard_rows(rows, lengths, f"shard{k}of2")
        assert e["ids_sha256"] == PINNED[k]


def test_tampered_plan_is_refused(tmp_path, monkeypatch):
    plan = json.loads(F.shard_file(2).read_text(encoding="utf-8"))
    moved = plan["shards"][0]["run_ids"].pop()
    plan["shards"][1]["run_ids"].append(moved)
    p = tmp_path / "fulldev_shards_2.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(F, "shard_file", lambda n: p)
    with pytest.raises(ValueError, match="ids_sha256"):
        F.selected_ids("shard1of2")


# ------------------------------------------------------------------------------------------------ score + merge
@pytest.fixture
def env(monkeypatch, tmp_path):
    """score() over synthetic rows with a synthetic committed plan; model + predictions stubbed."""
    _kit()
    from train import m2_gates as G

    rows, lengths = _rows(90, seed=3)
    plan = F.shard_plan(rows, lengths, 2)
    pf = tmp_path / "fulldev_shards_2.json"
    pf.write_text(json.dumps(plan), encoding="utf-8")
    lf = tmp_path / "lengths.json"
    lf.write_text(json.dumps({"lengths": lengths}), encoding="utf-8")
    monkeypatch.setattr(F, "shard_file", lambda n: pf)
    monkeypatch.setattr(F, "LENGTHS", lf)
    monkeypatch.setattr(F, "N_DEV", len(rows))
    monkeypatch.setitem(sys.modules, "harness.suite", types.SimpleNamespace(read_rows=lambda p: rows))
    ck = {"sha256": "c" * 64}

    def load_child(a, **kw):
        return types.SimpleNamespace(decide=lambda req: {}), False, {"arm": "child", "checkpoint": dict(ck)}

    monkeypatch.setattr(F, "load_child", load_child)
    monkeypatch.setattr(F, "plan_rows", lambda d, r, m: {"n_rows": len(r), "n_refused": 0, "n_over_16384": 0,
                                                         "max_row_tokens": 1})
    monkeypatch.setattr(F, "disk_watch", lambda paths, **k: (lambda: None))
    monkeypatch.setattr(F, "_meta_post", lambda door: True)

    class _Prog:
        def __init__(self, p):
            pass

        def plan(self, w):
            pass

        def wrap(self, fn):
            return fn

    scored = []

    def predict(decide, rs, untempered_from_raw):
        rid = rs[0]["_evaluation"]["run_id"]
        scored.append(rid)
        return {rid: {"run_id": rid, "status": "ok", "response": {"answers": [int(rid[1:]) % 3]},
                      "total_wall_ms": 5 + int(rid[1:])}}

    monkeypatch.setattr(G, "Progress", _Prog)
    monkeypatch.setattr(G, "beat_wrap", lambda fn: fn)
    monkeypatch.setattr(G, "unit_weight", lambda r: 1)
    monkeypatch.setattr(G, "index_predict", predict)
    cell = tmp_path / "cell.json"
    cfg = json.loads((M / "m2_first_cell_rwkv7_fulldev_eval_d13_v3w.json").read_text(encoding="utf-8"))
    cell.write_text(json.dumps({"config": cfg}), encoding="utf-8")

    def args(name, rows_select="full", code_sha="a" * 40, seed=None):
        return types.SimpleNamespace(rows="rows", rows_select=rows_select, arm="child", readout="v3w",
                                     max_input_tokens=65536, cell=str(cell), out_dir=str(tmp_path / name / "fulldev"),
                                     resume_dir=str(tmp_path / name / "ck"), progress=str(tmp_path / name / "p.jsonl"),
                                     code_sha=code_sha, seed_partial=seed)

    return types.SimpleNamespace(rows=rows, plan=plan, args=args, scored=scored, ck=ck, tmp=tmp_path)


def _ids(rows):
    return [r["_evaluation"]["run_id"] for r in rows]


def test_merged_shards_are_byte_identical_to_the_unsharded_run(env):
    assert F.score(env.args("full")) == 0
    assert not (env.tmp / "full/fulldev/shard-child.json").exists()  # unsharded: no shard record
    for k in (1, 2):
        assert F.score(env.args(f"s{k}", f"shard{k}of2")) == 0
        m = json.loads((env.tmp / f"s{k}/fulldev/shard-child.json").read_text())
        assert m["ids_sha256"] == env.plan["shards"][k - 1]["ids_sha256"] and m["rows_mode"] == f"shard{k}of2"
        got = [json.loads(x)["run_id"] for x in (env.tmp / f"s{k}/fulldev/child.results.jsonl").read_text().splitlines()]
        assert set(got) == set(env.plan["shards"][k - 1]["run_ids"])
    out = env.tmp / "merged/fulldev/child.results.jsonl"
    rec = F.merge([env.tmp / "s2", env.tmp / "s1/fulldev"], Path("rows"), out)  # any order, dir or artifact root
    assert out.read_bytes() == (env.tmp / "full/fulldev/child.results.jsonl").read_bytes()
    # agent-a: the merged ids hash == the unsharded mode's ids hash (fulldev_eval.ids_sha256 over every DEV run_id)
    assert rec["ids_sha256"] == F.ids_sha256(_ids(env.rows)) == env.plan["dev_ids_sha256"]
    assert rec["shard_ids_sha256"] == [s["ids_sha256"] for s in env.plan["shards"]]
    assert json.loads((out.parent / "child.results.merge.json").read_text())["results_sha256"] == F._sha(out)


def test_merge_cli_exit_codes(env, capsys):
    for k in (1, 2):
        assert F.score(env.args(f"s{k}", f"shard{k}of2")) == 0
    out = env.tmp / "m/child.results.jsonl"
    assert F.main(["merge", str(env.tmp / "s1"), "--rows", "rows", "--out", str(out)]) == 11
    assert not out.exists()
    assert F.main(["merge", str(env.tmp / "s1"), str(env.tmp / "s2"), "--rows", "rows", "--out", str(out)]) == 0


def _two_shards(env):
    for k in (1, 2):
        assert F.score(env.args(f"s{k}", f"shard{k}of2")) == 0
    return [env.tmp / "s1/fulldev", env.tmp / "s2/fulldev"]


def _edit(d, **kw):
    p = d / "shard-child.json"
    m = json.loads(p.read_text())
    m.update(kw)
    p.write_text(json.dumps(m))


@pytest.mark.parametrize("case,match", [
    ("missing", "one shard per k"),
    ("twice", "more than one shard|one shard per k"),
    ("checkpoint", "checkpoint_sha256"),
    ("readout", "readout_fingerprint"),
    ("code", "code_sha"),
    ("no_code", "code_sha"),
    ("tampered", "results_sha256"),
    ("dropped_row", "pre-declared shard ids hash"),
    ("other_plan", "seed"),
])
def test_merge_refusals(env, case, match):
    s1, s2 = _two_shards(env)
    shards = [s1, s2]
    if case == "missing":
        shards = [s1]
    elif case == "twice":
        shards = [s1, s1]
    elif case == "checkpoint":
        _edit(s2, checkpoint_sha256="d" * 64)
    elif case == "readout":
        _edit(s2, readout_fingerprint="e" * 64)
    elif case == "code":
        _edit(s2, code_sha="b" * 40)
    elif case == "no_code":
        _edit(s1, code_sha=None)
        _edit(s2, code_sha=None)
    elif case == "tampered":
        p = s1 / "child.results.jsonl"
        p.write_text(p.read_text().replace('"ok"', '"OK"', 1))
    elif case == "dropped_row":  # a row removed and every recorded hash "fixed up" to match: the plan still refuses
        p = s1 / "child.results.jsonl"
        ls = p.read_text().splitlines(keepends=True)[1:]
        p.write_text("".join(ls))
        ids = [json.loads(x)["run_id"] for x in ls]
        _edit(s1, results_sha256=F._sha(p), ids_sha256=F.ids_sha256(ids), rows_ids_sha256=F.ids_sha256(ids))
    elif case == "other_plan":
        _edit(s1, seed="x")
        _edit(s2, seed="x")
    out = env.tmp / "m/child.results.jsonl"
    with pytest.raises(F.MergeError, match=match):
        F.merge(shards, Path("rows"), out)
    assert not out.exists()


def test_merge_refuses_overlapping_shards(env):
    s1, s2 = _two_shards(env)
    extra = (s1 / "child.results.jsonl").read_text().splitlines(keepends=True)[0]
    p = s2 / "child.results.jsonl"
    p.write_text(p.read_text() + extra)
    _edit(s2, results_sha256=F._sha(p))
    with pytest.raises(F.MergeError, match="not disjoint"):
        F.merge([s1, s2], Path("rows"), env.tmp / "m.jsonl")


def test_score_refuses_a_plan_that_does_not_rebuild(env, monkeypatch):
    plan = json.loads(F.shard_file(2).read_text())
    a, b = plan["shards"]
    # swap one whole cluster's worth of bookkeeping: a valid-looking (disjoint, covering, self-hashed) but NOT rebuilt plan
    rid = a["run_ids"].pop()
    b["run_ids"] = sorted(b["run_ids"] + [rid])
    for s in (a, b):
        s["n_rows"], s["ids_sha256"] = len(s["run_ids"]), F.ids_sha256(s["run_ids"])
    F.shard_file(2).write_text(json.dumps(plan))
    assert F.score(env.args("s1", "shard1of2")) == 10


# ------------------------------------------------------------------------------------------------ partial + seed
def _stop_after(monkeypatch, n):
    calls = [0]

    def watch(paths, **k):
        def low():
            calls[0] += 1
            return {"ck": 1} if calls[0] > n else None
        return low

    monkeypatch.setattr(F, "disk_watch", watch)


def test_seed_partial_resumes_only_the_unscored_tail(env, monkeypatch):
    _stop_after(monkeypatch, 5)
    assert F.score(env.args("a", "shard1of2")) == F.DISK_LOW_RC
    part = env.tmp / "a/ck/child.partial.jsonl"
    idf = env.tmp / "a/ck/child.partial.identity.json"
    assert len(part.read_text().splitlines()) == 5 and json.loads(idf.read_text())["rows_mode"] == "shard1of2"
    seed = env.tmp / "export/fulldev"  # what the runner's fd_export_partial copies into the phase export
    seed.mkdir(parents=True)
    for f in (part, idf):
        (seed / f.name).write_bytes(f.read_bytes())
    monkeypatch.setattr(F, "disk_watch", lambda paths, **k: (lambda: None))
    env.scored.clear()
    assert F.score(env.args("b", "shard1of2", seed=str(seed / "child.partial.jsonl"))) == 0
    n1 = env.plan["shards"][0]["n_rows"]
    assert len(env.scored) == n1 - 5
    t = json.loads((env.tmp / "b/fulldev/timing-child.json").read_text())
    assert t["rows_resumed"] == 5 and t["rows_this_attempt"] == n1 - 5
    assert F.score(env.args("s2", "shard2of2")) == 0
    out = env.tmp / "m/child.results.jsonl"
    F.merge([env.tmp / "b/fulldev", env.tmp / "s2/fulldev"], Path("rows"), out)
    assert F.score(env.args("full")) == 0
    assert out.read_bytes() == (env.tmp / "full/fulldev/child.results.jsonl").read_bytes()


@pytest.mark.parametrize("what", ["checkpoint", "readout", "code", "rows", "stray", "no_code", "no_identity"])
def test_seed_partial_identity_checks(env, monkeypatch, what):
    _stop_after(monkeypatch, 4)
    assert F.score(env.args("a", "shard1of2")) == F.DISK_LOW_RC
    seed = env.tmp / "a/ck/child.partial.jsonl"
    idf = env.tmp / "a/ck/child.partial.identity.json"
    ident = json.loads(idf.read_text())
    monkeypatch.setattr(F, "disk_watch", lambda paths, **k: (lambda: None))
    kw = {}
    if what == "checkpoint":
        env.ck["sha256"] = "d" * 64
    elif what == "readout":
        ident["readout_fingerprint"] = "0" * 64
    elif what == "code":
        kw["code_sha"] = "b" * 40
    elif what == "no_code":
        kw["code_sha"] = None
    elif what == "rows":
        kw["rows_select"] = "shard2of2"
    elif what == "stray":  # a row of the OTHER shard smuggled into the seed
        other = env.plan["shards"][1]["run_ids"][0]
        with open(seed, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": other, "status": "ok", "response": {}}) + "\n")
    elif what == "no_identity":
        idf.unlink()
    if what != "no_identity":
        idf.write_text(json.dumps(ident))
    a = env.args("b", kw.get("rows_select", "shard1of2"), code_sha=kw.get("code_sha", "a" * 40), seed=str(seed))
    env.scored.clear()
    assert F.score(a) == 10
    rp = env.tmp / "b/ck/child.partial.jsonl"
    assert env.scored == [] and (not rp.exists() or not rp.read_text().strip())  # nothing written


def test_resume_slot_of_another_run_is_refused(env, monkeypatch):
    _stop_after(monkeypatch, 3)
    assert F.score(env.args("a", "shard1of2")) == F.DISK_LOW_RC
    monkeypatch.setattr(F, "disk_watch", lambda paths, **k: (lambda: None))
    env.ck["sha256"] = "f" * 64  # same slot, another checkpoint: never mix rows
    assert F.score(env.args("a", "shard1of2")) == 10


def test_seed_fetch_extracts_only_the_partial_and_checks_the_sha(tmp_path):
    part, ident = b'{"run_id":"r1"}\n', b'{"arm":"child"}'
    tgz = tmp_path / "fulldev.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        for name, b in (("out/fulldev/child.partial.jsonl", part), ("out/fulldev/child.partial.identity.json", ident),
                        ("out/fulldev/evil.py", b"x"), ("logs/run.log", b"log")):
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    sha = hashlib.sha256(tgz.read_bytes()).hexdigest()
    p = F.seed_fetch(tgz.as_uri(), sha, tmp_path / "seed")
    assert p.read_bytes() == part and (p.parent / "child.partial.identity.json").read_bytes() == ident
    assert sorted(x.name for x in p.parent.iterdir()) == ["child.partial.identity.json", "child.partial.jsonl",
                                                          "seed.tgz"]
    with pytest.raises(ValueError, match="sha256"):
        F.seed_fetch(tgz.as_uri(), "0" * 64, tmp_path / "seed2")


# ------------------------------------------------------------------------------------------------ params + runner
@pytest.mark.parametrize("mode", SHARD_MODES)
def test_shard_params_price_with_headroom_and_fit_the_hold(mode):
    from jobs.common.costs import quoted_max_cost_usd, schedule_minutes, worst_case_usd
    from jobs.m2_first_cell import checkpoint_loader as L

    k = int(mode[-4])
    p = json.loads((M / f"job_params_{mode}.json").read_text(encoding="utf-8"))
    base = json.loads((M / "job_params_fulldev_eval_d13_v3w.json").read_text(encoding="utf-8"))
    assert p["mode"] == mode and p["rows"] == f"shard{k}of2" and p["readout"] == "v3w"
    assert L.checkpoint_block(p) == L.checkpoint_block(base)  # the same checkpoint bytes as the unsharded mode
    cfg = json.loads((ROOT / p["cell_config"]).read_text(encoding="utf-8"))
    assert cfg == json.loads((M / "m2_first_cell_rwkv7_fulldev_eval_d13_v3w.json").read_text(encoding="utf-8"))
    e = F.estimate(p)
    assert e["ok"] and e["headroom_pct"]["fulldev"] >= 25
    assert p["phase_caps_minutes"]["fulldev"] >= 1.25 * e["minutes"]["fulldev"]
    assert schedule_minutes(p) == p["max_duration_minutes"] and quoted_max_cost_usd(p) <= 25
    assert worst_case_usd(p) <= p["hard_cap_usd"]
    assert p["submit"]["command"].endswith(f"first_cell_job.sh {mode}")
    assert "fulldev_shards_2.json" in " ".join(p["context"]["manifest_files"])
    assert p["context"]["data"][-1]["pin_json"][0] == f"jobs/m2_first_cell/job_params_{mode}.json"
    keys = {json.loads((M / f"job_params_{m}.json").read_text(encoding="utf-8"))["submit"]["key"]
            for m in SHARD_MODES + ("fulldev_eval_d13_v3w", "fulldev_eval_d13")}
    assert len(keys) == 4


def test_unsharded_pricing_unchanged():
    p = json.loads((M / "job_params_fulldev_eval_d13_v3w.json").read_text(encoding="utf-8"))
    e = F.estimate(p)
    assert e["minutes"] == {"fulldev": 500.0} and e["rows"] == "full" and "plan_minutes_full" not in p


def test_runner_exports_the_partial_on_any_exit_and_passes_seed_and_code_sha():
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    fin = sh[sh.index("finish() {"):sh.index("fail() {")]
    assert fin.index("fd_export_partial") < fin.index("export_phase final")
    loop = sh[sh.index('for ARMNAME in "${FD_ARMS[@]}"'):sh.index('say "phase report"')]
    assert loop.index("|| fd_export_partial") < loop.index('export_phase "$PH"')
    assert '"${SEED_ARGS[@]}"' in loop and "--code-sha" in loop
    assert "SEED_PARTIAL_SHA256" in sh and "fulldev_eval seed-fetch" in sh
