"""train.doc_overlap_guard: no training row may carry Index eval DOCUMENT text (synthetic eval index, no real data)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import doc_overlap_guard as DG  # noqa: E402

RNG = random.Random(7)
VOCAB = [f"w{i}" for i in range(5000)]


def _words(n, rng=RNG):
    return " ".join(rng.choice(VOCAB) for _ in range(n))


TRIAL = _words(160)          # an eval NLI4CT premise (CTR section)
TRIAL_B = _words(160)        # its secondary trial
WIKI = _words(150)           # a Wikipedia paragraph used as HoVer evidence
CLAIM = "the actor was born in a small town in ohio and later starred in a famous film released in 1990"  # 20 words
TEMPLATE = "Read the passage below carefully and decide which of the listed options best answers the question asked"


def _eval_rows(n_filler=200):
    rows = [("TEST", "NLI4CT-2024:test:aaaa", "NLI4CT-2024",
             {"state": {"primary_trial": {"text": [TRIAL]}, "secondary_trial": {"text": [TRIAL_B]}},
              "questions": {"q": {"instructions": f"{TEMPLATE} statement: the drug helps", "criteria": {"e": "Entailment"}}}}),
            ("DEV", "HoVer:dev:1", "HoVer claim verification",
             {"state": {"claim": CLAIM,
                        "evidence": [{"text": WIKI}]}, "questions": {"q": {"instructions": TEMPLATE}}})]
    for i in range(n_filler):  # unrelated eval rows sharing the instruction template (boilerplate on both sides)
        rows.append(("HELDOUT", f"X:{i}", "Filler", {"state": {"doc": _words(40)}, "questions": {"q": {"instructions": TEMPLATE}}}))
    return rows


@pytest.fixture()
def idx():
    allow = DG.load_allowlist()
    rows = _eval_rows()
    index = DG.build_index(None, allow, rows_factory=lambda: iter(rows), n_parts=2)
    index.meta["index_sha256"] = "synthetic"
    DG.set_index_for_tests(index)
    yield index
    DG.set_index_for_tests(None)


def _rec(i, state, source="anli", instr=TEMPLATE):
    return {"id": f"m2:t:{i}", "source": source, "family": "nli", "state": state,
            "questions": {"a": {"type": "choice", "instructions": instr, "criteria": {"x": "x", "y": "y"}}}}


def _clean(n, start=1000):
    return [_rec(start + i, {"premise": _words(60)}) for i in range(n)]


def _paraphrase(text, p_replace, rng):
    """Replace each word with probability p (a 13-gram survives with probability (1 - p) ** 13)."""
    return " ".join(rng.choice(VOCAB) if rng.random() < p_replace else x for x in text.split())


def test_copied_eval_premise_fails(idx):
    recs = _clean(300) + [_rec(1, {"primary_trial": {"text": [TRIAL]}, "secondary_trial": {"text": [TRIAL_B]}}, "nli4ct")]
    rep = DG.scan(recs, idx)
    assert rep["fail_ids"] == ["m2:t:1"]
    f = rep["fails"][0]
    assert "R" in f["reasons"] and "E" in f["reasons"]
    assert f["best_unit"]["family"] == "NLI4CT-2024" and f["best_unit"]["split"] == "TEST"
    assert rep["per_source_fail"] == {"nli4ct": 1}


def test_half_premise_rows_fail_by_draw_union(idx):
    # each row holds one of the two trial sections plus its own text: alone it covers < 50% of the eval unit set
    # in (E) only through its own section (S:primary_trial is its own unit, so E fires); the union rule is exercised
    # on a premise split INSIDE one unit
    half = TRIAL.split()
    a, b = " ".join(half[:85]), " ".join(half[75:])
    recs = _clean(300) + [_rec(1, {"t": [a, _words(200)]}, "nli4ct"), _rec(2, {"t": [b, _words(200)]}, "nli4ct")]
    rep = DG.scan(recs, idx)
    assert set(rep["fail_ids"]) == {"m2:t:1", "m2:t:2"}
    assert all("U" in f["reasons"] for f in rep["fails"])
    # either half alone (without the other in the draw) passes: < 50% of the unit, < 50% of its own grams
    assert DG.scan(_clean(300) + [_rec(1, {"t": [a, _words(200)]}, "nli4ct")], idx)["n_fail"] == 0


def test_wikipedia_allowlisted_rows_pass_and_are_counted(idx):
    recs = _clean(300) + [_rec(i, {"passage": WIKI}, "nq") for i in range(3)]
    rep = DG.scan(recs, idx)
    assert rep["n_fail"] == 0
    assert rep["allowlist_saved"] == {"hover-wikipedia-evidence": {"nq": 3}}


def test_allowlist_never_exempts_the_item_field(idx):
    rep = DG.scan(_clean(300) + [_rec(1, {"passage": WIKI, "claim": CLAIM}, "hover")], idx)
    assert rep["fail_ids"] == ["m2:t:1"]
    assert rep["fails"][0]["best_unit"]["field"] == "S:claim" and rep["fails"][0]["best_unit"]["allowlist"] is None


def test_paraphrased_doc_at_30pct_coverage_passes(idx):
    rng = random.Random(3)
    para = _paraphrase(TRIAL, 0.09, rng)  # ~9% of words replaced -> ~30% of 13-grams survive
    g_eval = DG.text_grams([TRIAL])
    g_para = DG.text_grams([para])
    cov = len(set(g_eval.tolist()) & set(g_para.tolist())) / len(g_eval)
    assert 0.2 <= cov <= 0.45, cov
    rep = DG.scan(_clean(300) + [_rec(1, {"t": [para]}, "nli4ct")], idx)
    assert rep["n_fail"] == 0, rep["fails"]
    # a light edit (~3% of words) keeps > 50% of the grams and fails
    para2 = _paraphrase(TRIAL, 0.03, random.Random(4))
    assert DG.scan(_clean(300) + [_rec(1, {"t": [para2]}, "nli4ct")], idx)["n_fail"] == 1


def test_boilerplate_is_excluded_and_listed(idx):
    # every training row repeats the eval instruction template (> 1% of eval rows and of training rows)
    rep = DG.scan(_clean(300), idx)
    assert rep["n_fail"] == 0
    assert rep["boilerplate"] and all(b["eval_df"] > 0.01 * idx.n_rows for b in rep["boilerplate"])
    # a short row that is ONLY the template is not failed: boilerplate never counts as coverage
    short = [_rec(i, {"x": "ok"}, "vast") for i in range(10)]
    assert DG.scan(_clean(300) + short, idx)["n_fail"] == 0


def test_normalisation_defeats_case_and_punctuation(idx):
    shouty = TRIAL.upper().replace(" ", " , ")
    assert DG.scan(_clean(300) + [_rec(1, {"t": [shouty]}, "nli4ct")], idx)["fail_ids"] == ["m2:t:1"]


def test_eval_text_in_instructions_or_criteria_counts(idx):
    r = _rec(1, {"x": "short"}, "mnli", instr=TRIAL)
    assert DG.scan(_clean(300) + [r], idx)["fail_ids"] == ["m2:t:1"]
    r = _rec(2, {"x": "short"}, "nq")
    r["questions"]["a"]["criteria"] = {"p1": TRIAL_B}
    assert DG.scan(_clean(300) + [r], idx)["fail_ids"] == ["m2:t:2"]


def test_clean_spec_and_pin_check(idx):
    recs = _clean(300) + [_rec(1, {"t": [TRIAL]}, "nli4ct")]
    rep = DG.scan(recs, idx)
    spec = DG.clean_spec(rep)
    assert spec["n_excluded"] == 1 and spec["n_kept"] == 300
    assert DG.check_pinned(rep, spec)[0]
    assert not DG.check_pinned(rep, None)[0]
    assert not DG.check_pinned(rep, spec | {"n_excluded": 2})[0]
    assert DG.check_pinned(DG.scan(_clean(300), idx), None)[0]


def test_missing_eval_data_fails_closed(monkeypatch, tmp_path):
    DG.set_index_for_tests(None)
    monkeypatch.setenv("SSDM_DI_SUITE", str(tmp_path / "nope"))
    from harness import config as hc

    monkeypatch.setattr(hc, "SUITE_DIR", tmp_path / "nope")
    monkeypatch.setattr(DG, "PIN_PATH", tmp_path / "no-pin.json")
    monkeypatch.delenv("SSDM_DOC_OVERLAP_INDEX", raising=False)
    with pytest.raises(DG.EvalDataMissing):
        DG.get_index()
    DG.get_index.cache_clear()


def test_shipped_index_must_match_pin(idx, tmp_path, monkeypatch):
    d = tmp_path / "ix"
    meta = DG.save_index(idx, d, {"synthetic": True})
    loaded = DG.load_index_dir(d, meta["index_sha256"])
    assert len(loaded.h) == len(idx.h)
    with pytest.raises(DG.DocOverlapError):
        DG.load_index_dir(d, "0" * 64)


def test_partitioned_build_equals_single(idx):
    rows = _eval_rows()
    one = DG.build_index(None, DG.load_allowlist(), rows_factory=lambda: iter(rows), n_parts=1)
    four = DG.build_index(None, DG.load_allowlist(), rows_factory=lambda: iter(rows), n_parts=4)
    for k in ("h", "df", "fl", "ent", "u_den"):
        assert (getattr(one, k) == getattr(four, k)).all(), k
    for p in range(0, len(one.h), max(1, len(one.h) // 50)):
        a = one.units_of(__import__("numpy").array([p]))
        b = four.units_of(__import__("numpy").array([p]))
        assert sorted(a.tolist()) == sorted(b.tolist())


def test_cli_scan_exit_codes(idx, tmp_path):
    import gzip

    p = tmp_path / "r.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in _clean(300) + [_rec(1, {"t": [TRIAL]}, "nli4ct")]:
            fh.write(json.dumps(r) + "\n")
    assert DG.main(["scan", "--records", str(p)]) == DG.EXIT_HIT
    q = tmp_path / "c.jsonl.gz"
    with gzip.open(q, "wt", encoding="utf-8") as fh:
        for r in _clean(300):
            fh.write(json.dumps(r) + "\n")
    assert DG.main(["scan", "--records", str(q)]) == 0


# ----------------------------------------------------------------------------- wiring: draw / heldout gate / load_cell_data
def _full_rec(i, state):
    return {"id": f"m2:t:{i}", "schema": "ssdm-train-v0", "source": "nli4ct" if "t" in state else "anli",
            "family": "nli", "source_split": "train", "state": state, "untrusted_fields": [], "labels": {"a": "x"},
            "questions": {"a": {"type": "choice", "instructions": "Pick one.", "criteria": {"x": "x", "y": "y"}}}}


def _set_with(tmp_path, recs):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_m2_first_cell import _train_set

    return _train_set(tmp_path, recs)


def _wiring_recs(n=8):
    return [_full_rec(i, {"premise": _words(40, random.Random(10_000 + i))}) for i in range(n)] + [_full_rec(50, {"t": [TRIAL]})]


def test_draw_refuses_an_unpinned_doc_fail_and_accepts_the_clean_spec(idx, tmp_path):
    from train.train_set import DocOverlapRefusal, draw

    d, cfg = _set_with(tmp_path, _wiring_recs())
    with pytest.raises(DocOverlapRefusal):
        draw(cfg, d)
    spec = DG.clean_spec(DG.scan_file(d / "records.jsonl.gz"))
    assert spec["n_excluded"] == 1 and spec["per_source_fail"] == {"nli4ct": 1}
    pinned = {**cfg, "n_train_records": 8, "train_set": {**cfg["train_set"], "doc_clean": spec}}
    got, tr, ident = draw(pinned, d)
    assert "m2:t:50" not in tr and len(tr) == 8
    assert ident["doc_overlap_guard"]["n_excluded"] == 1 and ident["doc_overlap_guard"]["drawn_fails"] == 0
    wrong = {**pinned, "train_set": {**cfg["train_set"], "doc_clean": {**spec, "excluded_ids_sha256": "0" * 64}}}
    with pytest.raises(DocOverlapRefusal):
        draw(wrong, d)


def test_heldout_gate_exit_4_on_doc_fail(idx, tmp_path):
    from train.train_set import heldout_gate

    d, cfg = _set_with(tmp_path, _wiring_recs(4))
    assert heldout_gate(cfg, d) == 4  # the job shells map 4 -> exit 34
    spec = DG.clean_spec(DG.scan_file(d / "records.jsonl.gz"))
    pinned = {**cfg, "train_set": {**cfg["train_set"], "doc_clean": spec}}
    assert heldout_gate(pinned, d, tmp_path / "h.json") == 0
    assert json.loads((tmp_path / "h.json").read_text())["doc_overlap_guard"]["n_excluded"] == 1


def test_gate_fails_closed_when_the_index_cannot_load(tmp_path, monkeypatch):
    from train.train_set import DocOverlapRefusal, draw

    d, cfg = _set_with(tmp_path, _wiring_recs(4))

    def boom():
        raise DG.EvalDataMissing("no suite")

    monkeypatch.setattr(DG, "get_index", boom)
    with pytest.raises(DocOverlapRefusal):
        draw(cfg, d)


def test_load_cell_data_without_train_set_cannot_bypass(idx, tmp_path):
    """A cell config with no train_set trains straight from --records: the gate runs inside load_cell_data."""
    import gzip

    from train.run_cell import EXIT, load_cell_data
    from train.train_set import DocOverlapRefusal

    p = tmp_path / "records.jsonl.gz"
    recs = _wiring_recs(30)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    cfg = {"n_train_records": 31, "n_eval_records": 0, "seed": 0}
    with pytest.raises(DocOverlapRefusal) as e:
        load_cell_data(cfg, str(p))
    assert isinstance(e.value, DG.DocOverlapError) and EXIT["doc_overlap"] == 34
    rep = DG.scan(recs)
    pinned = cfg | {"doc_clean": DG.clean_spec(rep)}
    _, _, tr, _, ident = load_cell_data(pinned, str(p))
    assert "m2:t:50" not in tr and len(tr) >= 29 and ident["doc_overlap_guard"]["n_excluded"] == 1


def test_template_with_a_tool_name_slot_is_not_a_document():
    """BFCL-style eval instructions = a long shared template + 'Candidate tool: <name>'; synthetic training rows that
    reuse the template with the same tool name share only 1-3 distinctive grams of the unit: not a document copy."""
    frame = ("Given the complete user conversation and the published function schemas in state mark each candidate "
             "tool yes if it should be invoked to answer the request or no otherwise candidate tool")
    rows = [("DEV", f"BFCL:{i}", "BFCL-tool-selection",
             {"state": {"functions": [{"description": _words(30, random.Random(500 + i))}]},
              "questions": {"q": {"instructions": f"{frame} {t}"}}})
            for i, t in enumerate(["send email", "lock door", "calculate loan payment"] * 12)]
    index = DG.build_index(None, DG.load_allowlist(), rows_factory=lambda: iter(rows), n_parts=1)
    index.meta["index_sha256"] = "synthetic"
    train = [_rec(i, {"functions": [{"description": _words(30, random.Random(900 + i))}]}, "b1fam_bfcl_synth",
                  instr=f"{frame} {t}") for i, t in enumerate(["send email", "lock door", "calculate loan payment"] * 10)]
    assert DG.scan(_clean(100) + train, index)["n_fail"] == 0
    # copying an eval row's function catalog (the document) still fails
    cat = rows[0][3]["state"]["functions"][0]["description"]
    assert DG.scan(_clean(100) + [_rec(77, {"functions": [{"description": cat}]}, "b1fam_bfcl_synth")], index)["fail_ids"] == ["m2:t:77"]


def test_eval_side_frame_is_template_even_in_a_small_draw(idx):
    # TEMPLATE sits in 202 eval rows (>= 100): a row carrying it fails neither alone nor in a tiny draw
    r = _rec(1, {"x": TEMPLATE + " " + TEMPLATE}, "hellaswag", instr="ok")
    assert DG.scan([r], idx)["n_fail"] == 0


def test_repeated_document_in_state_stays_gated():
    """A state (document) unit that repeats across >= 100 eval rows and >= 10 training rows is NOT a template."""
    doc = _words(60, random.Random(4242))
    rows = [("TEST", f"N:{i}", "NLI4CT-2024", {"state": {"primary_trial": {"text": [doc]}}, "questions": {}})
            for i in range(120)]
    index = DG.build_index(None, DG.load_allowlist(), rows_factory=lambda: iter(rows), n_parts=1)
    index.meta["index_sha256"] = "synthetic"
    train = [_rec(i, {"t": [doc]}, "nli4ct") for i in range(20)]
    assert len(DG.scan(_clean(100) + train, index)["fail_ids"]) == 20


def test_allowlist_never_saves_the_eval_family_own_source():
    prem = _words(80, random.Random(77))
    rows = _eval_rows() + [("DEV", "ANLI:1", "ANLI", {"state": {}, "questions": {"q": {"instructions": prem}}})]
    index = DG.build_index(None, DG.load_allowlist(), rows_factory=lambda: iter(rows), n_parts=1)
    index.meta["index_sha256"] = "synthetic"
    assert DG.scan(_clean(150) + [_rec(1, {"passage": prem}, "nq")], index)["n_fail"] == 0  # shared text: allowlisted
    assert DG.scan(_clean(150) + [_rec(2, {"premise": prem}, "anli")], index)["fail_ids"] == ["m2:t:2"]  # own source


def test_stateless_record_fails_closed(idx):
    with pytest.raises(DG.DocOverlapError):
        DG.scan([{"id": "m2:t:1", "source": "x", "questions": {}}], idx)


def test_meta_counts_must_match_the_pin(idx):
    DG.check_meta_vs_pin(idx, {"n_eval_rows": idx.n_rows, "n_units": idx.meta["n_units"], "n_grams": idx.meta["n_grams"]})
    with pytest.raises(DG.DocOverlapError):
        DG.check_meta_vs_pin(idx, {"n_eval_rows": idx.n_rows + 1})


def test_every_job_params_ships_the_pinned_index():
    """Every job_params that ships the doc-overlap eval-hash index ships exactly the committed index-pin's directory
    (all 12 files, each with a pin_sha256; with the local cache present the pins equal the files and their .npy
    aggregate equals the pin's index_sha256), so a stale index can't drift into a launch again."""
    import glob
    import hashlib

    pin = json.loads((ROOT / "data/manifests/doc_overlap/index-pin.json").read_text(encoding="utf-8"))
    key16 = pin["key"][:16]
    names = set(DG.EvalIndex.ARRAYS) | {"meta"}
    cache = DG.DEFAULT_CACHE / key16
    n_files = 0
    for f in glob.glob(str(ROOT / "jobs/**/job_params*.json"), recursive=True):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        items = [x for x in (d.get("context") or {}).get("data") or [] if "doc-overlap/" in str(x.get("dest", ""))]
        if not items:
            continue
        n_files += 1
        dirs = {x["dest"].rsplit("/", 1)[0] for x in items}
        assert dirs == {f"m2/doc-overlap/{key16}"}, (f, dirs)
        got = {x["dest"].rsplit("/", 1)[1].split(".")[0] for x in items}
        assert got == names and all(x["src"] == x["dest"] and len(x.get("pin_sha256", "")) == 64 for x in items), f
        if (cache / "meta.json").exists():
            for x in items:
                assert hashlib.sha256((DG.DEFAULT_CACHE.parent.parent / x["src"]).read_bytes()).hexdigest() == x["pin_sha256"], (f, x)
            pins = {x["dest"].rsplit("/", 1)[1]: x["pin_sha256"] for x in items}
            agg = hashlib.sha256("".join(pins[f"{k}.npy"] for k in DG.EvalIndex.ARRAYS).encode()).hexdigest()
            assert agg == pin["index_sha256"], f
    assert n_files >= 9


def test_every_cell_on_the_150k_set_pins_the_same_clean_specs():
    """Every m2_first_cell cell config that draws from the 150k set pins the SAME attack_clean and doc_clean specs as
    the base cell. A cell without them refuses at the heldout phase (exit 33 / 34) after setup spend; the PR #232
    anti-collapse template (d1_fla_safe) was written before PR #233 / #235 pinned them and merged without either."""
    import glob

    base = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7.json").read_text(encoding="utf-8"))["train_set"]
    n = 0
    for f in sorted(glob.glob(str(ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7*.json"))):
        ts = json.loads(Path(f).read_text(encoding="utf-8")).get("train_set") or {}
        if ts.get("pointer") != base["pointer"]:
            continue
        n += 1
        for k in ("attack_clean", "doc_clean"):
            assert ts.get(k) == base[k], (Path(f).name, k)
    assert n >= 10 and (ROOT / base["doc_clean"]["spec"]).exists() and (ROOT / base["attack_clean"]["spec"]).exists()
