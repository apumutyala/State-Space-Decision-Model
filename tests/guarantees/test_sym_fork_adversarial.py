"""Adversarial checks of the sym-fork reorder / relabel invariance claim (evidence lane, SYM_CHECK.md).

Goes beyond ``test_reorder_invariance.py``: K = 2..10, several backbone seeds, near-identical / unicode / whitespace
contents, id styles, ids inside descriptions, empty descriptions, exact duplicates (tie_set), multi-question batching,
and the row itself (prefix excludes the options). Whole served path: build_row -> tiny random Qwen3.5 hybrid (CPU,
fp32, cache fork) -> fork head -> decide.Decider. Tolerance is 0 everywhere (bitwise), as the lead claims.

    CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/guarantees/test_sym_fork_adversarial.py
    SSDM_SYM_SEEDS=0,1,2,3,7  (backbone seeds; default 0,1,2)
"""
from __future__ import annotations

import os
import random
import unicodedata
from functools import lru_cache
from typing import Any, Sequence

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

SEEDS = [int(s) for s in os.environ.get("SSDM_SYM_SEEDS", "0,1,2").split(",")]
LAYOUT = os.environ.get("SSDM_SYM_LAYOUT", "sym-fork")


@lru_cache(maxsize=8)
def decider(seed: int, layout: str = LAYOUT):
    from decide import Decider
    from heads.backbone import tiny_hf_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder
    from heads.model import DecisionModel
    from train.data import TOY_TEMPLATE

    torch.manual_seed(seed)
    model = DecisionModel(tiny_hf_backbone("qwen3.5", seed=seed, fork_mode="cache"), "fork", seed=seed).float().eval()
    return Decider(DecisionModelBackend(model, ByteEncoder(), template=TOY_TEMPLATE, layout=layout))


def req(state: Any, options: Sequence[tuple[str, Any]], extra_q: dict | None = None) -> dict:
    qs = {"q": {"type": "choice", "instructions": "Which option fits the state?",
                "criteria": {str(i): c for i, c in options}}}
    if extra_q:
        qs.update(extra_q)
    return {"model": "sym-adv", "state": state, "questions": qs}


def run(seed: int, state: Any, options, extra_q=None, layout: str = LAYOUT) -> dict:
    resp = decider(seed, layout).decide(req(state, options, extra_q))
    ss = resp.get("ssdm") or {}
    assert ss.get("status", "ok") == "ok", ss.get("error")
    return resp


def probs(resp, qid="q") -> dict[str, float]:
    return dict(resp["ssdm"]["served_probabilities"][qid])


def perms(k: int, n: int, rng: random.Random) -> list[list[int]]:
    out = [list(range(k))[::-1]]
    for _ in range(n):
        p = list(range(k))
        rng.shuffle(p)
        out.append(p)
    return out


def check(seed: int, state: Any, options: list[tuple[str, Any]], n_perm: int = 4, n_rel: int = 3,
          relabel: bool = True, id_style: str = "x") -> dict:
    """Bitwise per-content comparison over reorders and (optionally) relabels composed with a shuffle.
    Returns diffs; asserts nothing (callers decide what is claimed)."""
    rng = random.Random(f"{seed}|{len(options)}|{state}")
    base = run(seed, state, options)
    bp, ba = probs(base), base["answers"]["q"]
    out = {"reorder_dp": 0.0, "reorder_choice_diff": 0, "relabel_dp": 0.0, "relabel_choice_diff": 0,
           "tie_set_diff": 0}
    content = dict(options)
    for p in perms(len(options), n_perm, rng):
        o2 = [options[i] for i in p]
        r = run(seed, state, o2)
        pp = probs(r)
        out["reorder_dp"] = max(out["reorder_dp"], max(abs(pp[k] - bp[k]) for k in bp))
        out["reorder_choice_diff"] += r["answers"]["q"]["choice"] != ba["choice"]
        out["tie_set_diff"] += sorted(r["answers"]["q"].get("tie_set", [])) != sorted(ba.get("tie_set", []))
    if relabel:
        for j in range(n_rel):
            fresh = _fresh_ids(len(options), id_style, rng)
            sigma = {o[0]: fresh[i] for i, o in enumerate(options)}
            inv = {v: k for k, v in sigma.items()}
            o2 = [(sigma[i], c) for i, c in options]
            rng.shuffle(o2)
            r = run(seed, state, o2)
            pp = {inv[k]: v for k, v in probs(r).items()}
            out["relabel_dp"] = max(out["relabel_dp"], max(abs(pp[k] - bp[k]) for k in bp))
            a = r["answers"]["q"]
            # decision compared by CONTENT (ids differ by construction)
            out["relabel_choice_diff"] += content[inv[a["choice"]]] != content[ba["choice"]]
            out["tie_set_diff"] += (sorted(inv[x] for x in a.get("tie_set", [])) != sorted(ba.get("tie_set", [])))
    return out


def _fresh_ids(k: int, style: str, rng: random.Random) -> list[str]:
    if style == "num":
        ids = [str(rng.randrange(10 ** 6)) + str(i) for i in range(k)]
    elif style == "alpha":
        ids = ["".join(rng.choice("abcdefghij") for _ in range(3)) + chr(65 + i) for i in range(k)]
    elif style == "long":
        ids = [("opt-" * 20) + f"{rng.randrange(10 ** 9)}_{i}" for i in range(k)]
    else:
        ids = [f"x{rng.randrange(10 ** 6):06d}_{i}" for i in range(k)]
    rng.shuffle(ids)
    return ids


def _contents(k: int, rng: random.Random) -> list[str]:
    return [f"candidate {rng.randrange(10 ** 6)} {'abc'[i % 3] * (i + 1)}" for i in range(k)]


# --------------------------------------------------------------------------------------------- K x seeds
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("k", list(range(2, 11)))
def test_k_and_seed_sweep_bitwise(seed, k):
    rng = random.Random(1000 * seed + k)
    opts = [(f"opt_{j}", c) for j, c in enumerate(_contents(k, rng))]
    d = check(seed, f"state seed={seed} k={k}", opts, n_perm=3, n_rel=2)
    assert d["reorder_dp"] == 0.0 and d["relabel_dp"] == 0.0, d
    assert d["reorder_choice_diff"] == 0 and d["relabel_choice_diff"] == 0, d


# --------------------------------------------------------------------------------------------- adversarial contents
LONG = "The applicant has a stable income history and the requested amount is within the policy limit, " * 3
CONTENT_CASES = {
    "shared_long_prefix": [LONG + "approve", LONG + "approve.", LONG + "decline", LONG + "refer"],
    "one_char_diff": ["option a", "option b", "option c", "Option a"],
    "unicode_nfc_vs_nfd": [unicodedata.normalize("NFC", "café"), unicodedata.normalize("NFD", "café"),
                           "cafe", "café​", "рай"],
    "emoji_cjk_rtl": ["✅ yes", "❌ no", "決定する", "כן", "🤷"],
    "whitespace_inner": ["a b", "a  b", "a\tb", "a\nb", "a b"],
}


@pytest.mark.parametrize("seed", SEEDS[:2])
@pytest.mark.parametrize("case", sorted(CONTENT_CASES))
def test_adversarial_contents_bitwise(seed, case):
    opts = [(f"o{j}", c) for j, c in enumerate(CONTENT_CASES[case])]
    d = check(seed, "state for " + case, opts, n_perm=4, n_rel=3)
    assert d["reorder_dp"] == 0.0 and d["relabel_dp"] == 0.0, d
    assert d["reorder_choice_diff"] == 0 and d["relabel_choice_diff"] == 0 and d["tie_set_diff"] == 0, d


@pytest.mark.parametrize("style", ["num", "alpha", "long", "x"])
def test_id_styles_bitwise(style):
    rng = random.Random(style)
    opts = [(i, c) for i, c in zip(_fresh_ids(5, style, rng), _contents(5, rng))]
    d = check(SEEDS[0], "state ids " + style, opts, n_perm=3, n_rel=3, id_style=style)
    assert d["reorder_dp"] == 0.0 and d["relabel_dp"] == 0.0, d


def test_ids_inside_descriptions():
    # each description mentions OTHER options' ids; a relabel leaves descriptions unchanged (they are content)
    opts = [("A", "pick B if unsure"), ("B", "A is wrong; choose B"), ("C", "neither A nor B"), ("10", "option 9"),
            ("9", "option 10")]
    d = check(SEEDS[0], "ids in text", opts, n_perm=4, n_rel=3)
    assert d["reorder_dp"] == 0.0 and d["relabel_dp"] == 0.0, d


# --------------------------------------------------------------------------------------------- whitespace-only diffs
def test_leading_trailing_whitespace_variants_collapse_to_duplicates():
    """Descriptions are .strip()ped (heads.symmetric.option_text): " yes", "yes ", "yes" are the SAME content.
    Records the behaviour: they must form one tie_set and get bitwise-equal probability."""
    opts = [("a", "yes"), ("b", " yes"), ("c", "yes \n"), ("d", "no")]
    r = run(SEEDS[0], "ws", opts)
    p = probs(r)
    assert p["a"] == p["b"] == p["c"], p
    d = check(SEEDS[0], "ws", opts, n_perm=5, n_rel=3)
    assert d["reorder_dp"] == 0.0 and d["relabel_dp"] == 0.0 and d["tie_set_diff"] == 0, d


# --------------------------------------------------------------------------------------------- empty descriptions
def test_empty_descriptions_reorder_holds():
    opts = [("alpha", ""), ("beta", None), ("gamma", "   "), ("delta", "real text")]
    d = check(SEEDS[0], "empty", opts, n_perm=5, relabel=False)
    assert d["reorder_dp"] == 0.0 and d["reorder_choice_diff"] == 0, d


def test_empty_descriptions_relabel_is_not_claimed_and_does_break():
    """Declared exception (id is the content when the description is empty): relabel is expected to MOVE mass.
    This test documents the boundary: if it ever stops breaking, the id is no longer read for these options."""
    opts = [("alpha", ""), ("beta", ""), ("gamma", ""), ("delta", "")]
    d = check(SEEDS[0], "all-empty", opts, n_perm=2, n_rel=4)
    assert d["reorder_dp"] == 0.0
    assert d["relabel_dp"] > 0.0, d


def test_empty_description_marked_id_as_content():
    from heads.layout import ByteEncoder, build_row
    from train.data import TOY_TEMPLATE

    row = build_row(req("s", [("alpha", ""), ("beta", "text")]), "q", ByteEncoder(), template=TOY_TEMPLATE,
                    layout=LAYOUT)
    assert row.meta["id_as_content"] == ["alpha"]


# --------------------------------------------------------------------------------------------- exact duplicates
@pytest.mark.parametrize("seed", SEEDS)
def test_duplicates_tie_set_invariant_and_position_free(seed):
    opts = [("p", "same text"), ("q", "other"), ("r", "same text"), ("s", "third"), ("t", "same text")]
    base = run(seed, "dups", opts)
    p = probs(base)
    assert p["p"] == p["r"] == p["t"], p  # equal content -> bitwise-equal probability
    a = base["answers"]["q"]
    if a["choice"] in ("p", "r", "t"):
        assert sorted(a["tie_set"]) == ["p", "r", "t"], a
    d = check(seed, "dups", opts, n_perm=6, n_rel=4)
    assert d["reorder_dp"] == 0.0 and d["relabel_dp"] == 0.0, d
    assert d["tie_set_diff"] == 0 and d["relabel_choice_diff"] == 0, d


def test_duplicates_force_winning_tie_set():
    """Search seeds/states until a duplicated content WINS, then check the tie_set is the full duplicate group under
    every reorder / relabel, and the served 'choice' is always inside it (decision by content, never by position)."""
    found = 0
    for seed in SEEDS:
        for st in range(12):
            opts = [("m", "dup X"), ("n", "solo Y"), ("k", "dup X"), ("z", "solo Z")]
            base = run(seed, f"force {st}", opts)
            a = base["answers"]["q"]
            if a["choice"] not in ("m", "k"):
                continue
            found += 1
            assert sorted(a["tie_set"]) == ["k", "m"], a
            rng = random.Random(st)
            for _ in range(4):
                o2 = opts[:]
                rng.shuffle(o2)
                fresh = _fresh_ids(4, "alpha", rng)
                sigma = {o[0]: fresh[i] for i, o in enumerate(opts)}
                inv = {v: k for k, v in sigma.items()}
                r = run(seed, f"force {st}", [(sigma[i], c) for i, c in o2])["answers"]["q"]
                assert sorted(inv[x] for x in r["tie_set"]) == ["k", "m"], r
                assert inv[r["choice"]] in ("k", "m"), r
            if found >= 2:
                return
    pytest.skip("no seed/state made the duplicate win; tie_set path not exercised")


# --------------------------------------------------------------------------------------------- batching
@pytest.mark.parametrize("seed", SEEDS[:2])
def test_batch_vs_single_bitwise(seed):
    """All questions of a request go through ONE forward (shared prefix, padded batch). The probabilities of question
    q must not depend on which other questions share the batch (pad width / batch size)."""
    opts = [("a", "short"), ("b", "a somewhat longer option text"), ("c", "mid length")]
    single = probs(run(seed, "batch state", opts))
    extra = {"q2": {"type": "choice", "instructions": "Another question?",
                    "criteria": {"x": "a much, much longer option than any in q " * 4, "y": "n"}},
             "q3": {"type": "noul", "instructions": "Is it true?"}}
    batched = probs(run(seed, "batch state", opts, extra_q=extra))
    diff = max(abs(single[k] - batched[k]) for k in single)
    assert diff == 0.0, {"single": single, "batched": batched, "max_abs_dp": diff}


# --------------------------------------------------------------------------------------------- the row itself
def test_prefix_excludes_option_list():
    from heads.layout import ByteEncoder, build_row
    from train.data import TOY_TEMPLATE

    enc = ByteEncoder()
    r1 = build_row(req("S", [("a", "AAAA unique1"), ("b", "BBBB unique2")]), "q", enc, template=TOY_TEMPLATE,
                   layout=LAYOUT)
    r2 = build_row(req("S", [("zz", "totally different"), ("yy", "qqq")]), "q", enc, template=TOY_TEMPLATE,
                   layout=LAYOUT)
    assert r1.ids == r2.ids  # prefix = f(state, question, K) only
    text = bytes(t - 4 for t in r1.ids).decode("utf-8")
    for s in ("AAAA", "BBBB", "unique"):  # no option text in the shared prefix
        assert s not in text
    # ids never in continuations under sym-fork
    for c in r1.continuations:
        ct = bytes(t - 4 for t in c).decode("utf-8")
        assert not ct.lstrip().startswith(("a:", "b:"))
    # K is in the prefix (declared): different K -> different prefix
    r3 = build_row(req("S", [("a", "x"), ("b", "y"), ("c", "z")]), "q", enc, template=TOY_TEMPLATE, layout=LAYOUT)
    assert r3.ids != r1.ids


def test_row_bit_identical_under_reorder_and_relabel():
    from heads.layout import ByteEncoder, build_row
    from train.data import TOY_TEMPLATE

    enc = ByteEncoder()
    opts = [("a", "one"), ("b", "two"), ("c", "three"), ("d", "two")]
    r0 = build_row(req("S", opts), "q", enc, template=TOY_TEMPLATE, layout=LAYOUT)
    rng = random.Random(0)
    for _ in range(10):
        o2 = [(f"new{rng.randrange(999)}_{i}", c) for i, (_, c) in enumerate(opts)]
        rng.shuffle(o2)
        r = build_row(req("S", o2), "q", enc, template=TOY_TEMPLATE, layout=LAYOUT)
        assert r.ids == r0.ids and r.continuations == r0.continuations and r.option_seqs == r0.option_seqs
