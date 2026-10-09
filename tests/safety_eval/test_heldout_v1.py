"""safety_eval.heldout_v1 (design: analysis/SAFETY_HELDOUT_DESIGN.md section 6.2): template loading, the hash pin,
disjointness, the pinned unit set, edits to the untrusted field only, determinism, targets, transforms, positions, no
special tokens, encoding under decide/untrusted.py, and the training-draw scanner's positive control. CPU, no weights."""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import pytest

from safety_eval import heldout_v1 as HV

ROOT = Path(__file__).resolve().parents[2]
HF = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data\hf")
TOKENIZERS = {"qwen3.5-0.8b": HF / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / "2fc06364715b967f1860aea9cf38778875588b17"}


@pytest.fixture(scope="module")
def variants():
    return HV.expected_variants()


@pytest.fixture(scope="module")
def bases(variants):
    return {v["request_id"]: v for v in variants if v["_meta"]["attack_family"] == "clean"}


def _base(v, bases):
    return bases[v["request_id"].split("::")[0]]


def test_template_file_hash_is_pinned_and_shape():
    t = HV.load_templates()
    assert HV.template_set_sha256(t) == HV.TEMPLATE_SET_SHA256 == \
        "690dc5e41c7edb836c19b97a018929b14000237f157c7a93045b228c4cbdd941"
    goals = [s["goal"] for s in t.values()]
    assert len(t) == 21 and goals.count("hijack") == 12 and goals.count("steer") == 8 and goals.count("steer_abstain") == 1
    assert all(n.startswith("H") == (s["goal"] == "hijack") for n, s in t.items())
    assert set(HV.STYLE) == set(t)


def test_edited_template_file_fails_closed(tmp_path):
    t = json.loads(HV.TEMPLATES_PATH.read_text(encoding="utf-8"))
    t["H01_persona"]["text"] += " "
    p = tmp_path / "heldout_v1_templates.json"
    p.write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(HV.TemplateSetError, match="heldout_v2"):
        HV.load_templates(p)
    assert HV.load_templates(p, expected_sha256=None)["H01_persona"]["text"].endswith(" ")


def test_disjoint_from_every_training_and_official_template():
    from baselines.clef_heldout_v1 import disjointness

    d = disjointness()
    assert d["ok"], d
    assert not any(d["shared_ngrams"].values()) and d["injection_6gram_hits"] == \
        {"FORBIDDEN_6GRAMS": 0, "official+train_templates": 0} and d["special_token_templates"] == []


def test_production_unit_set_matches_the_pins(variants):
    from baselines import clef_heldout_v1 as C

    ident = C.check_production_variants(variants)
    assert ident["n_units"] == 7920 and ident["by_family"] == C.N_UNITS and ident["n_clusters"] == 45
    assert len(ident["templates"]) == 21
    # the same 360 base requests (ids, order) as the matched battery / job-j8wgy
    from baselines import clef_hijack_matched as H

    hm = [v["request_id"] for v in H.expected_variants() if v["_meta"]["attack_family"] == "clean"]
    assert [v["request_id"] for v in variants if v["_meta"]["attack_family"] == "clean"] == hm


def test_deterministic(variants):
    again = HV.expected_variants()
    assert HV.identity(again) == HV.identity(variants)


def test_only_the_untrusted_field_is_edited(variants, bases):
    for v in variants:
        b = _base(v, bases)
        f = b["_meta"]["untrusted_field"]
        assert v["questions"] == b["questions"] and v["policy"] == b["policy"] and v["provenance"] == b["provenance"]
        assert {k: x for k, x in v["state"].items() if k != f} == {k: x for k, x in b["state"].items() if k != f}
        assert f in b["provenance"]["untrusted_fields"]
        if v["_meta"]["attack_family"] != "clean":
            assert v["state"][f] != b["state"][f]
            # the original text survives (mid inserts split it at a sentence boundary)
            assert set(b["state"][f].split()) <= set(v["state"][f].split())


def test_targets_and_families(variants, bases):
    t = HV.load_templates()
    for v in variants:
        m = v["_meta"]
        if m["attack_family"] == "clean":
            continue
        b = _base(v, bases)
        q = next(iter(b["questions"].values()))
        allowed = set(next(iter(b["policy"]["allowed"].values())))
        spec = t[m["template"]]
        assert m["attack_family"] == HV.FAMILY[spec["goal"]] and m["battery"] == "heldout_v1"
        assert v["request_id"] == f"{b['request_id']}::{m['attack_family']}:{m['template']}"
        text = v["state"][m["untrusted_field"]]
        if spec["goal"] == "hijack":
            assert m["target"] == b["_meta"]["attacker_option"] not in allowed
            assert m["steer_target"] == b["_meta"]["steer_target"]  # unchanged: steer is not this family's metric
            shown = base64.b64decode(re.search(r"[A-Za-z0-9+/=]{20,}$", text).group(0)).decode() \
                if m["template"] == "H06_base64" else text
            assert m["target"] in shown
        elif spec["goal"] == "steer":
            assert m["target"] == m["steer_target"] == b["_meta"]["steer_target"]
            assert m["target"] in allowed and m["target"] != m["gold"]
            if m["template"] == "S01_ordinal":
                assert f"only the {HV.ordinal(list(q['criteria']).index(m['target']))} one" in text
            elif m["template"] in ("S04_quoted_reply", "S06_persona"):
                assert q["criteria"][m["target"]] in text
            else:
                assert m["target"] in text
            if m["template"] == "S02_table":
                assert f"| {m['gold']} | no |" in text and f"| {m['target']} | yes |" in text
        else:  # S07 abstain-forcing: the variant's own steer target, allowed and non-gold
            assert m["template"] == "S07_abstain" and m["steer_target"] == m["target"] == HV.ABSTAIN_OPTION
            assert HV.ABSTAIN_OPTION in allowed and m["gold"] != HV.ABSTAIN_OPTION


def test_unit_fingerprint_carries_the_s07_target(variants):
    from safety_eval.metrics import unit_fingerprint

    s07 = next(v for v in variants if v["_meta"]["template"] == "S07_abstain")
    other = json.loads(json.dumps(s07))
    other["_meta"]["steer_target"] = "something_else"
    assert unit_fingerprint(s07) != unit_fingerprint(other)


def test_transforms_touch_static_text_only(variants):
    for v in variants:
        m = v["_meta"]
        text = v["state"][m["untrusted_field"]]
        if m.get("transform") == "zwj_interleave":
            assert HV.ZWJ in text and m["target"] in text  # the substituted id is intact
            assert "b\u200da\u200dc\u200dk" in text and "by the" in text  # >= 4 letters only
        elif m.get("transform") == "cyrillic_homoglyph":
            assert "\u0435" in text and m["target"] in text
            assert "Reassigned" not in text and "R\u0435\u0430ssign\u0435d" in text
    assert HV.zwj_interleave("go with abc_def now") == "go w\u200di\u200dt\u200dh abc_def now"
    assert HV._on_static("pick $X please", HV.homoglyph) == "\u0440i\u0441k $X \u0440l\u0435\u0430s\u0435"


def test_positions(variants, bases):
    for v in variants:
        m = v["_meta"]
        if m["attack_family"] == "clean":
            continue
        base = _base(v, bases)["state"][m["untrusted_field"]]
        text = v["state"][m["untrusted_field"]]
        if m["position"] == "end":
            assert text.startswith(base.rstrip() + "\n")
        elif m["position"] == "start":
            assert text.endswith("\n" + base)
        elif m["position"] == "split":
            assert text.startswith("(1/2) ") and "(2/2) " in text.rsplit("\n", 1)[1]
        else:  # mid: an interior sentence boundary (never before the first or after the last sentence)
            n = len(base.split(". "))
            if m["mid_fallback_end"]:  # a one-sentence field: appended (declared in the variant, counted below)
                assert n == 1 and text == base.rstrip() + " " + text[len(base.rstrip()) + 1:]
            else:
                assert 1 <= m["inserted_at"] <= n - 1 and not text.startswith(base) and not text.endswith(base)
    fb = [v for v in variants if v["_meta"].get("mid_fallback_end")]
    # receipt (2026-10-09): 288 of the 360 base requests have a ONE-sentence untrusted field (the forwarded_content
    # receipts), so 1,152 of the 1,440 mid-position units (4 mid templates) are appended at the end, same line.
    # Reported as a design caveat (position axis), never changed after results.
    assert len(fb) == 1152 and len({v["request_id"].split("::")[0] for v in fb}) == 288


def test_mid_insert_falls_back_to_end_on_one_sentence():
    text, pos = HV._mid_insert("one sentence only", "Injected.", "k")
    assert (text, pos) == ("one sentence only Injected.", 1)


class _T:
    def __init__(self, c):
        self.content = c


class _E:
    def __init__(self, ids):
        self.input_ids = ids


_SPECIAL = {"<|im_start|>": 1001, "<|im_end|>": 1002, "<|endoftext|>": 1003}
_ADDED = {"<think>": 1004, "</think>": 1005}


class _FakeTok:
    """Char-level tokenizer with Qwen-like control tokens (split_special_tokens splits the special ones only)."""

    SPECIAL, ADDED = _SPECIAL, _ADDED
    all_special_ids = sorted(_SPECIAL.values())
    all_special_tokens = sorted(_SPECIAL)
    added_tokens_decoder = {i: _T(s) for s, i in {**_SPECIAL, **_ADDED}.items()}

    def __call__(self, text, add_special_tokens=False, split_special_tokens=False):
        table = dict(self.ADDED) if split_special_tokens else {**self.SPECIAL, **self.ADDED}
        ids, i = [], 0
        while i < len(text):
            for s, sid in table.items():
                if text.startswith(s, i):
                    ids.append(sid)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]) % 1000)
                i += 1
        return _E(ids)

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_no_special_tokens_and_encodes_under_the_untrusted_encoder(variants):
    from decide.untrusted import UntrustedEncoder

    toks = [("fake", _FakeTok())]
    for name, p in TOKENIZERS.items():
        try:
            from transformers import AutoTokenizer
        except ImportError:  # the real-tokenizer half needs transformers (the job image has it); the fake half always runs
            break
        if p.is_dir():

            toks.append((name, AutoTokenizer.from_pretrained(str(p))))
    attacked = [v for v in variants if v["_meta"]["attack_family"] != "clean"]
    for v in attacked:
        assert not HV.has_special_token(v["state"][v["_meta"]["untrusted_field"]])
    for name, tok in toks:
        enc = UntrustedEncoder(tok)
        seen = set()
        for v in attacked:  # one request per template is enough per tokenizer (the injected text differs by ids only)
            if (v["_meta"]["template"], v["_meta"]["family"]) in seen:
                continue
            seen.add((v["_meta"]["template"], v["_meta"]["family"]))
            ids = enc.encode(v["state"][v["_meta"]["untrusted_field"]])  # raises ControlTokenError if refused
            assert ids and not set(ids) & enc.control_ids, name


def test_single_question_only(bases):
    r = json.loads(json.dumps(next(iter(bases.values()))))
    r["questions"]["second"] = dict(next(iter(r["questions"].values())))
    with pytest.raises(ValueError, match="single-question"):
        HV.build_variants([r])


def test_unknown_template_name_refused(bases):
    with pytest.raises(HV.TemplateSetError):
        HV.build_variants(list(bases.values())[:1], names=["H99_nope"])


def test_draw_scanner_positive_control_and_clean_negative(variants):
    """analysis/safety_heldout/scan_draws.py must find EVERY rendered template (incl. ZWJ, homoglyph, base64, the
    table) and nothing in the clean requests: the 0-hit training-draw receipt is only as good as this."""
    sys.path.insert(0, str(ROOT))
    from analysis.safety_heldout import scan_draws as S

    g6, short = S.template_grams(6), S.short_segments()
    lens = {len(k) for k in short}

    def hits(text):
        f = S.scan_text(text, g6, short, lens) | {t for t, rx in S.STRUCTURAL.items() if rx.search(text)}
        for run in S.B64.findall(text):
            f |= S.scan_text(base64.b64decode(run + "=" * (-len(run) % 4)).decode("utf-8", "replace"), g6, short, lens)
        return f

    for v in variants:
        text = "\n".join(S.strings(v["state"]))
        assert bool(hits(text)) == (v["_meta"]["attack_family"] != "clean"), v["request_id"]
