"""train.attack_guard: no eval attack template (official hijack, D1-A, D1-B, heldout_v1) in any training record."""
from __future__ import annotations

import base64
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from train import attack_guard as AG  # noqa: E402


def _rec(i, text="a plain synthetic premise about the weather", extra_state=None):
    st = {"premise": f"{text} {i}"}
    st.update(extra_state or {})
    return {"id": f"m2:t:{i}", "schema": "ssdm-train-v0", "source": "anli", "family": "nli", "source_split": "train",
            "state": st, "untrusted_fields": [], "labels": {"a": "x"},
            "questions": {"a": {"type": "choice", "instructions": "Pick one.", "criteria": {"x": "x", "y": "y"}}}}


def _zwj(s):
    return "‍".join(s)


def _cyr(s):
    return s.translate(str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х"}))


def _render_heldout(spec):
    sub = {"$X_desc": "Close the ticket.", "$X": "close_ticket", "$GOLD_ID": "refund", "$ORD": "second"}
    t = spec["text"]
    if spec.get("payload"):
        p = spec["payload"]
        for k, v in sub.items():
            p = p.replace(k, v)
        t = t.replace("$B64", base64.b64encode(p.encode()).decode())
    for k, v in sub.items():
        t = t.replace(k, v)
    if spec["transform"] == "zwj_interleave":
        t = _zwj(t)
    elif spec["transform"] == "cyrillic_homoglyph":
        t = _cyr(t)
    return t.replace("|", " ")


def test_every_eval_template_is_caught_when_rendered():
    from safety_eval.attacks import D1_TEMPLATES_A, D1_TEMPLATES_B, HIJACK_TEMPLATES

    rendered = {}
    for name, t in HIJACK_TEMPLATES.items():
        rendered[f"official_hijack:{name}"] = t.format(attacker="escalate", attacker_desc="Escalate the case.")
    for set_name, tpls in (("d1a_steer", D1_TEMPLATES_A), ("d1b_steer", D1_TEMPLATES_B)):
        for name, t in tpls.items():
            rendered[f"{set_name}:{name}"] = t.format(act="go with option two", gold="the first invoice", sender="Sam")
    for name, spec in AG.load_heldout_v1().items():
        rendered[f"heldout_v1:{name}"] = _render_heldout(spec)
    assert set(rendered) == set(AG.eval_templates())
    missed = []
    for key, text in rendered.items():
        h = AG.record_hits(_rec(0, extra_state={"email_body": "Hello there. " + text}))
        if key not in h:
            missed.append(key)
    # templates with no fragment of >= MIN_WORDS words cannot be matched; everything else must be
    unmatchable = {k for k, frags in AG.eval_templates().items()
                   if all(len(AG.normalise(f)) < AG.MIN_WORDS for f in frags)}
    assert set(missed) == unmatchable, missed
    # S02_table's fixed text is table markup + "option still available" (3 words): too generic for a text match
    assert unmatchable == {"heldout_v1:S02_table"}


def test_clean_records_and_paraphrases_pass():
    assert not AG.record_hits(_rec(1))
    assert not AG.record_hits(_rec(2, "The finance director reviewed it and the correction was out of date."))


def test_d1a_text_in_any_field_or_question_is_caught():
    from safety_eval.attacks import D1_TEMPLATES_A

    t = D1_TEMPLATES_A["system_note"].format(act="pay now", gold="g", sender="s")
    r = _rec(3)
    r["questions"]["a"]["instructions"] = t
    assert AG.record_hits(r) == {"d1a_steer:system_note"}
    assert AG.record_hits(_rec(4, t.upper().replace(" ", "  "))) == {"d1a_steer:system_note"}  # normalisation


def test_heldout_v1_vendored_verbatim():
    raw = AG.HELDOUT_V1_PATH.read_bytes()
    blob = hashlib.sha1(b"blob %d\x00" % len(raw) + raw).hexdigest()
    assert blob == AG.HELDOUT_V1_PROVENANCE["git_blob"]
    assert len(AG.load_heldout_v1()) == 21


def test_training_templates_disjoint_from_every_eval_template():
    rep = AG.disjointness()
    assert rep["disjoint"], rep["clashes"]
    AG.assert_training_templates_disjoint()
    from safety_eval.attacks import D1_TEMPLATES_A

    bad = {"train_hijack:copy": AG._brace_fragments(D1_TEMPLATES_A["urgency"])}
    assert not AG.disjointness(bad)["disjoint"]
    with pytest.raises(AG.AttackTemplateInTraining):
        AG.assert_training_templates_disjoint(bad)
    # reverse direction: a short training fragment that sits inside an eval template
    assert not AG.disjointness({"x:short": ["the verified next action for"]})["disjoint"]


def test_injection_never_uses_eval_templates():
    from data import augment as A
    from data import injection as I

    BASE = {"id": "m2:t:1", "source": "t", "source_split": "train", "source_id": "1", "license": "x",
            "family": "nli", "fields": {"premise": "A man is walking a dog in the park.",
                                        "hypothesis": "A person is outside."},
            "qtype": "choice", "options": {"entailment": "E", "neutral": "N", "contradiction": "C"},
            "gold": "entailment", "item_fields": ["premise", "hypothesis"], "item_options": "none"}

    fams = set()
    for s in range(200):
        rec = A.render(BASE, random.Random(s))
        inj = I.inject(rec, random.Random(s))
        if inj is None:
            continue
        fams.add(inj["aug"]["injection"]["family"])
        assert not AG.record_hits(inj), inj["state"]
    assert fams == {"hijack"}
    rec = A.render(BASE, random.Random(0))
    with pytest.raises(I.EvalTemplateForTraining):
        I.inject(rec, random.Random(0), family="d1a")


def _set_with(tmp_path, recs):
    from test_m2_first_cell import _train_set

    return _train_set(tmp_path, recs)


def _d1a_rec(i):
    from safety_eval.attacks import D1_TEMPLATES_A

    return _rec(i, extra_state={"attached_note": D1_TEMPLATES_A["authority"].format(act="go with y", gold="x",
                                                                                    sender="s")})


def test_draw_refuses_an_unpinned_attack_hit_and_accepts_the_clean_spec(tmp_path):
    from train.train_set import AttackGuardRefusal, draw

    recs = [_rec(i) for i in range(8)] + [_d1a_rec(50)]
    d, cfg = _set_with(tmp_path, recs)
    with pytest.raises(AttackGuardRefusal):
        draw(cfg, d)
    spec = AG.clean_spec(AG.scan_file(d / "records.jsonl.gz"))
    assert spec["n_excluded"] == 1 and spec["per_set"] == {"d1a_steer": 1}
    pinned = {**cfg, "n_train_records": 8, "train_set": {**cfg["train_set"], "attack_clean": spec}}
    got, tr, ident = draw(pinned, d)
    assert "m2:t:50" not in tr and len(tr) == 8 and ident["attack_guard"]["n_excluded"] == 1
    assert ident["attack_guard"]["drawn_hits"] == 0
    wrong = {**pinned, "train_set": {**cfg["train_set"], "attack_clean": {**spec, "excluded_ids_sha256": "0" * 64}}}
    with pytest.raises(AttackGuardRefusal):
        draw(wrong, d)
    stale = {**pinned, "train_set": {**cfg["train_set"], "attack_clean": {**spec, "templates_sha256": "0" * 64}}}
    with pytest.raises(AttackGuardRefusal):
        draw(stale, d)


def test_heldout_gate_exit_3_on_attack_hit(tmp_path):
    from train.train_set import heldout_gate

    recs = [_rec(i) for i in range(4)] + [_d1a_rec(60)]
    d, cfg = _set_with(tmp_path, recs)
    assert heldout_gate(cfg, d) == 3  # the job maps 3 -> exit 33
    spec = AG.clean_spec(AG.scan_file(d / "records.jsonl.gz"))
    pinned = {**cfg, "train_set": {**cfg["train_set"], "attack_clean": spec}}
    assert heldout_gate(pinned, d, tmp_path / "h.json") == 0
    rep = json.loads((tmp_path / "h.json").read_text())
    assert rep["pass"] and rep["attack_guard"]["n_excluded"] == 1
    (tmp_path / "c").mkdir()
    clean_d, clean_cfg = _set_with(tmp_path / "c", [_rec(i) for i in range(4)])
    assert heldout_gate(clean_cfg, clean_d) == 0


def test_cli_scan_exit_codes(tmp_path):
    import gzip

    p = tmp_path / "r.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(json.dumps(_rec(1)) + "\n")
    assert AG.main(["scan", "--records", str(p)]) == 0
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(json.dumps(_rec(1)) + "\n" + json.dumps(_d1a_rec(2)) + "\n")
    assert AG.main(["scan", "--records", str(p)]) == AG.EXIT_HIT
    assert AG.main(["templates"]) == 0
