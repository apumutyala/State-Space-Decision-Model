"""R3 jrt-rwkv-g1-v3 family (decide.prompt_g1v3, decide.backends.rwkv_g1_v3, jobs/rwkv7 JOB=readout_g1v3).

Pure checks run everywhere; the tokenizer checks need the pinned G1k tokenizer files (SSDM_RWKV7_TOKDIR or
$SSDM_DATA/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b...), the scoring checks also torch (a tiny random RWKV7
with the real 65,536 vocabulary, CPU fp32).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from decide.prompt import CANARY_REQUEST, render_g1v2
from decide.prompt_g1v3 import (CATALOG_MIN_K, WORDLIKE, content_free_request, load_v3_codes, presentation_order,
                                question_codes, render_g1v3, reread_m, reread_suffix)
from decide.schema import canonical_options, canonical_question_ids

ROOT = Path(__file__).resolve().parents[1]
TOKDIR = Path(os.environ.get("SSDM_RWKV7_TOKDIR") or Path(os.environ.get("SSDM_DATA", ROOT.parent.parent / "ssdm-data"))
              / "tokenizers" / "RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba")


def v2_table():
    from decide.backends.rwkv_g1_filtered_codes import load_frozen_table

    return [c.code for c in load_frozen_table()]


def v3_table():
    return [e["code"] for e in load_v3_codes()[0]]


def catalog_req(k=30, qid="tool"):
    return {"model": "m", "state": "Dialogue: the user wants to add an agenda item for Friday.\n\nCatalog in state.",
            "questions": {qid: {"type": "choice", "instructions": "Select the single API tool to invoke.",
                                "criteria": {f"option_{i}": f"Api{i}Name" for i in range(k)}},
                          "flag": {"type": "noul", "instructions": "Is a tool needed?"}}}


def user_turn(s):
    return s[s.index("User: "):s.rindex("\n\nAssistant:")]


def test_v3_without_block_is_v2_byte_for_byte():
    t2 = v2_table()
    for req in (CANARY_REQUEST, catalog_req(30), catalog_req(5), dict_instr_req()):
        for qid in canonical_question_ids(req["questions"]):
            assert render_g1v3(req, qid, t2, None, fix_instr=False) == render_g1v2(req, qid, t2)


def dict_instr_req():
    return {"model": "m", "state": "",
            "questions": {"doc_0": {"type": "choice", "instructions": {"candidate": "def add(a, b):\n\n    return a+b",
                                                                       "task": "Is this tool relevant to: sum two ints?"},
                                    "criteria": {"no": None, "yes": None}}}}


def test_v2f_fix_renders_dict_instructions_and_changes_nothing_else():
    from decide.schema import render

    t2 = v2_table()
    req = dict_instr_req()
    buggy, fixed = render_g1v2(req, "doc_0", t2), render_g1v3(req, "doc_0", t2, None)
    assert "def add(a, b)" not in buggy and "INSTRUCTION: doc_0" in buggy  # the v2 bug (job-zhyvg)
    # v1's serializer: canonical JSON (newlines inside the document are JSON-escaped, so no blank line can form)
    assert "INSTRUCTION: " + render(req["questions"]["doc_0"]["instructions"]) in fixed
    assert '"candidate":"def add(a, b):\\n\\n    return a+b"' in fixed and "\n\n" not in user_turn(fixed)
    for req in (CANARY_REQUEST, catalog_req(30)):  # string instructions: v2f == v2
        for qid in canonical_question_ids(req["questions"]):
            assert render_g1v3(req, qid, t2, None) == render_g1v2(req, qid, t2)


def test_code_key_only_for_long_catalogs_and_no_blank_lines():
    t3 = v3_table()
    for k in (CATALOG_MIN_K - 1, CATALOG_MIN_K, 53, 151):
        s = render_g1v3(catalog_req(k), "tool", t3)
        assert ("CODE KEY (code = option):" in s) == (k >= CATALOG_MIN_K)
        assert "\n\n" not in user_turn(s)
        assert s.endswith("Reply with exactly one option code.\n\nAssistant: <think></think>\nANSWER:")
    s = render_g1v3(catalog_req(53), "tool", t3)
    key = s[s.index("CODE KEY"):]
    codes = question_codes(catalog_req(53)["questions"]["tool"], list(range(53)), t3)
    opts = canonical_options(catalog_req(53)["questions"]["tool"])
    for c, (oid, desc) in zip(codes, opts):  # opaque ids -> the description is the label
        assert f"\n{c} = {desc}\n" in key
    assert s.rindex("OPTIONS:") < s.index("QUESTION AGAIN: Select the single API tool") < s.index("CODE KEY")


def test_v3_code_table_rule():
    entries, _ = load_v3_codes()
    codes = [e["code"] for e in entries]
    assert codes[:26] == [chr(65 + i) for i in range(26)] and len(codes) == 255 == len(set(codes))
    assert not WORDLIKE.intersection(codes[26:])
    assert {"AN", "AS", "AT", "BE", "DO", "ER", "ES", "EU"} <= WORDLIKE
    assert len({e["token_id"] for e in entries}) == 255


def test_presentation_order_is_a_seeded_permutation_mapped_back():
    a = catalog_req(12)["questions"]["tool"]
    b = {**a, "criteria": {f"option_{i}": "x" for i in range(13)}}
    oa, ob = presentation_order(a, "tool"), presentation_order(b, "tool")
    assert sorted(oa) == list(range(12)) and oa == presentation_order(dict(a), "tool")
    assert oa != list(range(12)) and ob[:12] != oa  # depends on option ids, not on descriptions or data values
    assert presentation_order({**a, "criteria": {k: "other" for k in a["criteria"]}}, "tool") == oa
    t3 = v3_table()
    codes = question_codes(a, oa, t3)
    assert codes[oa[0]] == "A" and sorted(codes) == sorted(t3[:12])
    s = render_g1v3({"model": "m", "state": "s", "questions": {"tool": a}}, "tool", t3, permute=True)
    first = canonical_options(a)[oa[0]][0]
    assert f"\nA. {first}: " in s


def test_content_free_request_keeps_label_space():
    r = catalog_req(10)
    cf = content_free_request(r, "N/A")
    assert cf["state"] == "N/A" and all(q["instructions"] == "N/A" for q in cf["questions"].values())
    assert {k: q["criteria"] for k, q in cf["questions"].items() if "criteria" in q} == \
        {k: q["criteria"] for k, q in r["questions"].items() if "criteria" in q}


def test_calibration_and_reread_math():
    pytest.importorskip("torch")
    from decide.backends.rwkv_g1_v3 import calibrate_logits, combine_reread, shortlist

    lp = {"a": -0.1, "b": -2.5, "c": -4.0}
    same, _ = calibrate_logits(lp, [{"a": 0.0, "b": 0.0, "c": 0.0}])
    z = math.log(sum(math.exp(v) for v in lp.values()))
    assert all(abs(same[o] - (lp[o] - z)) < 1e-12 for o in lp)
    cal, pcf = calibrate_logits(lp, [{"a": -0.05, "b": -3.0, "c": -6.0}, {"a": -0.2, "b": -2.0, "c": -5.0}])
    assert abs(sum(math.exp(v) for v in cal.values()) - 1) < 1e-12 and abs(sum(pcf.values()) - 1) < 1e-12
    fp = {"a": -3.0, "b": -0.5, "c": -1.0, "d": -0.9, "e": -4.0}
    s = shortlist(fp, list(fp), 3)
    assert s == [1, 2, 3]
    out = combine_reread(fp, {"b": -2.0, "c": -0.1, "d": -1.0})
    p = {o: math.exp(v) for o, v in out.items()}
    p1 = {o: math.exp(v) / sum(math.exp(x) for x in fp.values()) for o, v in fp.items()}
    assert abs(sum(p.values()) - 1) < 1e-9
    assert abs(p["a"] - p1["a"]) < 1e-12 and abs(p["e"] - p1["e"]) < 1e-12
    assert max(p, key=p.get) == "c"
    assert reread_m(2) == 0 and reread_m(3) == 2 and reread_m(53) == 3


# ------------------------------------------------------------------------------------------- real tokenizer checks
needs_tok = pytest.mark.skipif(not (TOKDIR / "tokenizer.json").exists(), reason="SSDM_RWKV7_TOKDIR: pinned G1k tokenizer absent")


def _enc():
    pytest.importorskip("transformers")
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

    tok = Rwkv7Tokenizer.from_pretrained(str(TOKDIR))
    return lambda text, trusted=True: tok(text, add_special_tokens=False).input_ids


@needs_tok
def test_v3_codes_are_single_tokens_in_context():
    from decide.prompt_g1v3 import build_code_table

    enc = _enc()
    entries, _ = load_v3_codes()
    assert build_code_table(enc) == entries
    t3 = v3_table()
    for k in (2, 30, 151):
        req = catalog_req(k)
        for qid in canonical_question_ids(req["questions"]):
            s = render_g1v3(req, qid, t3)
            ids = enc(s)
            n = len(canonical_options(req["questions"][qid]))
            for j in sorted({0, 1, n - 1}):
                assert enc(s + " " + t3[j]) == ids + [entries[j]["token_id"]]
            suf = s + reread_suffix(req["questions"][qid], [0, n - 1], t3[:n])
            assert enc(suf + " " + t3[0]) == enc(suf) + [entries[0]["token_id"]]


def _tiny_plugin(enc):
    torch = pytest.importorskip("torch")
    from decide.backends.hf_readout import ModelPlugin
    from decide.backends.state_fork import RWKV7StateOps
    from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

    m = Rwkv7ForCausalLM(Rwkv7Config(vocab_size=65536, hidden_size=32, num_heads=4, head_dim=8, num_hidden_layers=2,
                                     intermediate_size=64, a_low_rank_dim=8, decay_low_rank_dim=8, gate_low_rank_dim=8,
                                     v_low_rank_dim=8))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    m = m.float().eval()
    fwd = lambda ids, n: m(input_ids=ids, logits_to_keep=n, use_cache=False).logits  # noqa: E731
    return ModelPlugin("rwkv7", "tiny", None, enc, fwd, device="cpu", dtype="float32", special_ids=frozenset({0}),
                       state_ops=RWKV7StateOps(m), max_context=25600), m


@needs_tok
def test_variants_score_end_to_end_with_proper_distributions():
    torch = pytest.importorskip("torch")
    from decide import Decider
    from decide.backends.rwkv_g1_v3 import G1V3Backend

    plug, model = _tiny_plugin(_enc())
    req = catalog_req(12)
    req["questions"]["pick"] = {"type": "choice", "instructions": "Pick one.", "criteria": {"x": "ex", "y": "why", "z": None, "w": "dub"}}
    out = {}
    for mode in ("catalog", "permute", "reread", "calibrate", "v2f", "words"):
        be = G1V3Backend(plug, mode=mode)
        res = be.score(req, lambda: None)
        out[mode] = (be, res)
        r = Decider(be, enable_v0_1_deltas=True).decide(req)
        assert r["ssdm"]["status"] == "ok", r["ssdm"]
        for qid, d in r["ssdm"]["raw_probabilities"].items():
            assert abs(sum(d.values()) - 1) < 1e-6 and set(d) == {o for o, _ in canonical_options(req["questions"][qid])}
    base = out["catalog"][1].logits
    rr = out["reread"][1]
    for qid, fp in rr.extras["first_pass_logits"].items():  # v3r's first pass IS v3
        assert all(abs(fp[o] - base[qid][o]) < 1e-5 for o in fp)
        top = sorted(fp, key=lambda o: -fp[o])[:3]
        assert sorted(rr.extras["shortlist"][qid]) == sorted(top)
    assert set(rr.extras["shortlist"]) == {"pick", "tool"}  # k >= 3 only; the noul question is not re-read
    # second pass from the shared prefix state == a full forward over the second-pass prompt
    be = out["reread"][0]
    q = next(x for x in be._questions(req) if x["qid"] == "pick")
    s_idx = [q["options"].index(o) for o in rr.extras["shortlist"]["pick"]]
    ids2 = be._full(q["text"] + reread_suffix(req["questions"]["pick"], s_idx, q["codes"]))
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=torch.tensor([ids2]), use_cache=False).logits[0, -1].float(), -1)
    p2 = {q["options"][j]: lp[q["tokens"][j]].item() for j in s_idx}
    p1 = {o: math.exp(v) for o, v in rr.extras["first_pass_logits"]["pick"].items()}
    z1 = sum(p1.values())
    mass = sum(p1[o] for o in p2) / z1
    z2 = sum(math.exp(v) for v in p2.values())
    for o in p2:
        assert abs(math.exp(rr.logits["pick"][o]) - mass * math.exp(p2[o]) / z2) < 1e-4
    # v3c: the prior is exactly the v3 readout of the content-free requests
    bc, rc = out["calibrate"]
    assert set(rc.extras["calibration_prior"]) == {"flag", "tool", "pick"}
    cat = out["catalog"][0]
    from decide.prompt_g1v3 import CONTENT_FREE_INPUTS
    from decide.backends.rwkv_g1_v3 import calibrate_logits

    priors = [cat.score(content_free_request(req, f), lambda: None).logits for f in CONTENT_FREE_INPUTS]
    for qid in ("pick", "tool"):
        exp, _ = calibrate_logits(base[qid], [p[qid] for p in priors])
        assert all(abs(exp[o] - rc.logits[qid][o]) < 1e-5 for o in exp)
    # v3p: same options, permuted codes; the A code is on the permuted first option
    bp = out["permute"][0]
    qp = next(x for x in bp._questions(req) if x["qid"] == "tool")
    order = presentation_order(req["questions"]["tool"], "tool")
    assert qp["codes"][order[0]] == "A"
    # v3w: word ids (pick: w/x/y/z, flag: true/false) scored by their own first token; option_N ids keep letters
    bw = out["words"][0]
    qw = {x["qid"]: x for x in bw._questions(req)}
    assert qw["pick"]["words"] and qw["flag"]["words"] and not qw["tool"]["words"]
    assert qw["pick"]["tokens"] == [bw.first_token(o) for o in qw["pick"]["options"]]
    for x in qw.values():  # the proxy first token is the real next token after the actual prompt
        if x["words"]:
            for o, t in zip(x["options"], x["tokens"]):
                assert bw.plugin.encode(x["text"] + " " + o, True)[len(x["prompt"]) - 1] == t
    # v2f: the v2 prompt (v2 code table, no code key) when instructions are strings
    b2 = out["v2f"][0]
    assert [x["text"] for x in b2._questions(req)] == [render_g1v2(req, q, v2_table()) for q in canonical_question_ids(req["questions"])]


def test_runner_and_driver_know_readout_g1v3():
    from jobs.rwkv7.readout_variants import DEFAULT_ORDER, VARIANTS, parse_variants

    assert parse_variants(DEFAULT_ORDER) == ["v2", "v2f", "v3", "v3w", "v3r", "v3p", "v1", "v3c"] and VARIANTS["v1"] == "rwkv7-hf" and set(VARIANTS) == set(parse_variants(DEFAULT_ORDER))
    with pytest.raises(SystemExit):
        parse_variants("v3,v3")
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    runner = ROOT / "jobs/rwkv7/rwkv7_job.sh"
    assert subprocess.run(["bash", "-n", str(runner)]).returncode == 0
    r = subprocess.run(["bash", str(runner)], env={"JOB": "nope", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert r.returncode == 2 and "readout_g1v3" in r.stdout


def test_rate_guard_counts_several_results_files(tmp_path):
    from jobs.common.rate_guard import _rows

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("1\n2\n")
    b.write_text("3\n")
    assert _rows(a) == 2 and _rows(f"{a},{b}") == 3 and _rows(f"{a},{tmp_path / 'none'}") == 2
    assert _rows(str(tmp_path / "*.jsonl")) == 3 and _rows(str(tmp_path / "v*" / "results.jsonl")) == 0


# ------------------------------------------------------------------------------------------------- v3w word ids
def test_word_ids_rule_and_rendering():
    from decide.prompt_g1v3 import word_ids_ok

    ft = lambda o: o.lower()[:3]  # noqa: E731  (stand-in first-token function)
    sent = {"type": "choice", "instructions": "Sentiment?", "criteria": {"Negative": None, "Neutral": None, "Positive": None}}
    assert word_ids_ok(sent, ft)
    assert not word_ids_ok({"type": "choice", "criteria": {"option_0": "a", "option_1": "b"}}, ft)  # opaque
    assert not word_ids_ok({"type": "choice", "criteria": {"Negation": None, "Negative": None}}, ft)  # same first token
    wino = {"type": "choice", "instructions": "Fill the blank.", "criteria": {"A": "Emily", "B": "Samantha"}}
    assert word_ids_ok(wino, ft)
    req = {"model": "m", "state": "s", "questions": {"q": wino, "s": sent}}
    s = render_g1v3(req, "q", v3_table(), word_qids=frozenset({"q", "s"}))
    assert "\n- A: Emily" in s and "A. A" not in s and "\n- Negative" in s and "exactly one option id." in s
    assert "CODE KEY" not in render_g1v3(catalog_req(12), "tool", v3_table(), word_qids=frozenset({"tool"}))


@needs_tok
def test_word_first_tokens_on_common_label_sets():
    enc = _enc()
    from decide.prompt_g1v3 import word_ids_ok

    ctx = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"
    base = enc(ctx)

    def ft(o):
        ids = enc(ctx + " " + o)
        return ids[len(base)] if ids[: len(base)] == base and len(ids) > len(base) else None

    for ids in (["yes", "no"], ["true", "false"], ["SUPPORTED", "NOT_SUPPORTED"], ["Entailment", "Contradiction", "NotMentioned"],
                ["Negative", "Neutral", "Positive"], ["A", "B", "C", "D"]):
        assert word_ids_ok({"type": "choice", "criteria": {i: None for i in ids}}, ft), ids


# ------------------------------------------------------------------------------- field coverage over the DEV rows
DEV_ROWS = Path(os.environ.get("SSDM_DATA", ROOT.parent.parent / "ssdm-data")) / "slices" / "dev-rows.jsonl.gz"


def field_texts(req):
    """(field name, text that must appear in the prompt) for every request field: state, every instruction, every
    option id and description. Non-strings are rendered as canonical JSON (v1's serializer); blank lines collapsed."""
    from decide.prompt import g1_clean
    from decide.schema import render

    def txt(v):
        return g1_clean(v if isinstance(v, str) else render(v)).strip()

    out = []
    if req["state"] not in (None, ""):
        out.append(("state", txt(req["state"])))
    for k in canonical_question_ids(req["questions"]):
        q = req["questions"][k]
        if q.get("instructions") not in (None, ""):
            out.append((f"{k}.instructions", txt(q["instructions"])))
        for oid, desc in canonical_options(q):
            out.append((f"{k}.option_id", txt(oid)))
            if desc is not None:
                out.append((f"{k}.option_desc", txt(desc)))
    return [(n, t) for n, t in out if t]


def coverage(rows, renderers):
    """{renderer: {family: [missing field names]}} over ``rows`` (one prompt per row: the schema lists every field)."""
    import collections

    miss = {r: collections.defaultdict(list) for r in renderers}
    for row in rows:
        req = {"model": "m", "state": row["state"], "questions": row["questions"]}
        qid = canonical_question_ids(req["questions"])[0]
        for name, fn in renderers.items():
            s = fn(req, qid)
            for field, t in field_texts(req):
                if t not in s:
                    miss[name][row["family"]].append(field)
    return {r: dict(m) for r, m in miss.items()}


def dev_renderers():
    from decide.prompt_g1v3 import word_ids_ok

    t2, t3 = v2_table(), v3_table()
    words = lambda req: frozenset(k for k in req["questions"] if word_ids_ok(req["questions"][k], lambda o: o))  # noqa: E731
    return {"v2": lambda r, q: render_g1v2(r, q, t2), "v2f": lambda r, q: render_g1v3(r, q, t2, None),
            "v3": lambda r, q: render_g1v3(r, q, t3), "v3p": lambda r, q: render_g1v3(r, q, t3, permute=True),
            "v3w": lambda r, q: render_g1v3(r, q, t3, word_qids=words(r))}


@pytest.mark.skipif(not DEV_ROWS.exists(), reason="DEV rows absent (SSDM_DATA/slices/dev-rows.jsonl.gz)")
def test_every_request_field_reaches_every_fixed_prompt():
    """A sample of rows from EVERY family (all rows with SSDM_FULL_COVERAGE=1): no field of the request may be missing
    from the v2f / v3 / v3p / v3w prompt. The v2 control is known to drop dict-valued instructions (reported only)."""
    import collections
    import gzip
    import json

    full = os.environ.get("SSDM_FULL_COVERAGE") == "1"
    per, rows = collections.Counter(), []
    with gzip.open(DEV_ROWS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if full or per[r["family"]] < 3:
                per[r["family"]] += 1
                rows.append(r)
    res = coverage(rows, dev_renderers())
    report = os.environ.get("SSDM_COVERAGE_REPORT")
    if report:
        Path(report).write_text(json.dumps({"rows": len(rows), "families": len(per), "missing": {
            r: {fam: dict(collections.Counter(v)) for fam, v in m.items()} for r, m in res.items()}}, indent=1))
    assert len(per) > 20
    for name in ("v2f", "v3", "v3p", "v3w"):
        assert not res[name], (name, {f: sorted(set(v)) for f, v in res[name].items()})


def test_g1v3_decision_validator_needs_every_variant_index_not_a_top_level_one(tmp_path):
    """job-fgu2c exited 9: valid_readout_g1v3 delegated to valid_readout (top-level index.point), which the per-variant
    g1v3 decision never has."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("decide_rwkv7", Path(__file__).resolve().parents[1] / "scripts/decide_rwkv7.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for v in m.G1V3_VARIANTS:
        (tmp_path / "index" / v).mkdir(parents=True)
        (tmp_path / "index" / v / "score.json").write_text(json.dumps(
            {"rows": 2008, "slice_index": {"index": {"point": 30.0, "lo": 28.0, "hi": 32.0}}}))
        (tmp_path / "index" / v / "ece.json").write_text("{}")
    d = m.decide_readout_g1v3(str(tmp_path))
    assert d["verdict"] == "readout-measured" and m.valid_readout_g1v3(d) and not m.valid_readout(d)
    d["variants"]["v3w"]["index"] = {"point": None}
    assert not m.valid_readout_g1v3(d)
    assert m.main("readout_g1v3", str(tmp_path), str(tmp_path / "decision.json")) == 0
