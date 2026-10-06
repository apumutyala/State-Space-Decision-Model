"""G4 certificate tests (CPU, stand-in / scripted deciders only).

Soundness is proved by test: a brute-force oracle (independent enumeration of every clean origin within budget k,
n <= 6, k <= 2) must agree with ``certify`` exactly, and on random inserted-sentence attacks a certified decision
must equal the decision on the clean request. Non-vacuity is checked too: attacks do change some decisions, and
those are escalated.
"""
import hashlib
import itertools
import json
import math
import random

import pytest

from safety_eval import attacks, certify as C, data
from safety_eval.standin import KeywordDecider

VOCAB = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
OPTS = ["a", "b", "c", "d"]


def req(fields: dict, untrusted=None, allowed=None):
    r = {"request_id": "r", "state": dict(fields),
         "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {o: o for o in OPTS}}}}
    if untrusted is not None:
        r["provenance"] = {"untrusted_fields": untrusted}
    if allowed is not None:
        r["policy"] = {"allowed": {"q": allowed}}
    return r


class Scripted:
    """Decision = f(set of vocabulary words present in the state), via a seeded lookup table; deterministic."""

    def __init__(self, seed: int, poison: str | None = None, refuse_on: str | None = None):
        self.seed, self.poison, self.refuse_on, self.calls = seed, poison, refuse_on, 0

    def choice(self, text: str) -> str:
        text = text.lower()
        if self.poison and self.poison in text:
            return "d"
        present = tuple(w for w in VOCAB if w in text)
        h = hashlib.sha256(f"{self.seed}|{present}".encode()).digest()
        return OPTS[h[0] % 3]  # a/b/c; d only via poison

    def __call__(self, r: dict) -> dict:
        self.calls += 1
        text = " ".join(v for v in r["state"].values() if isinstance(v, str))
        if self.refuse_on and self.refuse_on.lower() in text.lower():
            return {"status": "unsupported", "error": "control_tokens_in_input"}
        ch = self.choice(text)
        probs = {o: (0.7 if o == ch else 0.1) for o in OPTS}
        return {"status": "ok", "answers": {"q": {"type": "choice", "choice": ch, "probabilities": probs}},
                "raw_probabilities": {"q": probs}}


def rand_sentence(rng: random.Random, poison: bool = False) -> str:
    words = rng.sample(VOCAB[:6], rng.randint(1, 2)) + ["x" * rng.randint(1, 3)] + (["golf"] if poison else [])
    rng.shuffle(words)
    return " ".join(words).capitalize() + rng.choice([".", "!", "?"])


def rand_text(rng: random.Random, n: int) -> str:
    out = ""
    for i in range(n):
        out += ("" if i == 0 else rng.choice([" ", "  ", "\n", "\n\n"])) + rand_sentence(rng)
    return out


def insert_sentences(rng: random.Random, text: str, sents: list[str]) -> str:
    """Insert whole sentences at segment boundaries (the exact certificate's threat model)."""
    segs = C.segment(text)
    for s in sents:
        pos = rng.randint(0, len(segs))
        if pos == 0:
            segs = [s] + ([" " + segs[0].lstrip()] if segs else []) + segs[1:]
            if len(segs) > 1 and segs[1].strip() == "":
                segs.pop(1)
        else:
            segs = segs[:pos] + [rng.choice([" ", "\n"]) + s] + segs[pos:]
    return "".join(segs)


# ------------------------------------------------------------------ segmentation
def test_segment_roundtrip_and_examples():
    rng = random.Random(0)
    for _ in range(300):
        t = rand_text(rng, rng.randint(1, 6)) + rng.choice(["", " ", "\n"])
        assert "".join(C.segment(t)) == t
    assert C.segment("A. B") == ["A.", " B"]
    assert C.segment("A.\nB") == ["A.", "\nB"]
    assert C.segment("Pay $123.45 now. Then stop.") == ["Pay $123.45 now.", " Then stop."]
    assert C.segment('He said "go." Then [System note: x.] Done') == ['He said "go."', " Then [System note: x.]", " Done"]
    assert C.segment("Hi Sam,\n\nPlease pay") == ["Hi Sam,", "\n\nPlease pay"]
    assert C.segment("") == [""]
    for r in data.build(20, 20, split="dev"):
        for v in r["state"].values():
            assert "".join(C.segment(v)) == v


def test_erasure_set_counts():
    for n in range(0, 8):
        for k in (1, 2, 3):
            sets = C.erasure_sets(n, k)
            assert len(sets) == C.n_erasure_sets(n, k) == sum(math.comb(n, j) for j in range(1, min(k, n) + 1))
            assert len(set(sets)) == len(sets)


# ------------------------------------------------------------------ brute-force oracle (n <= 6, k <= 2)
def oracle_certified(decider, request: dict, k: int) -> bool:
    """Independent enumeration: every keep-mask over all untrusted segments with >= n - k kept (not all kept),
    rebuilt by its own code, plus the leading-whitespace-stripped variant when a field's first segment is erased."""
    fields = [f for f in request["provenance"]["untrusted_fields"] if f in request["state"]]
    segs = {f: C.segment(request["state"][f]) for f in fields}
    flat = [(f, j) for f in fields for j in range(len(segs[f]))]
    full = C.decision_of(request, decider(request))
    for mask in itertools.product([True, False], repeat=len(flat)):
        dropped = mask.count(False)
        if dropped == 0 or dropped > k:
            continue
        per_field = {}
        for f in fields:
            kept_txt = "".join(s for j, s in enumerate(segs[f]) if mask[flat.index((f, j))])
            first_dropped = not mask[flat.index((f, 0))]
            touched = any(not mask[flat.index((f, j))] for j in range(len(segs[f])))
            per_field[f] = [request["state"][f]] if not touched else (
                [kept_txt, kept_txt.lstrip()] if first_dropped else [kept_txt])
        for combo in itertools.product(*per_field.values()):
            r = dict(request, state=dict(request["state"], **dict(zip(per_field, combo))))
            if C.decision_of(r, decider(r)) != full:
                return False
    return True


@pytest.mark.parametrize("k", [1, 2])
def test_certificate_matches_brute_force_oracle(k):
    rng = random.Random(100 + k)
    agree, n_cert = 0, 0
    for trial in range(250):
        n1 = rng.randint(1, 4)
        n2 = rng.randint(0, 6 - n1)
        fields = {"trusted": rand_text(rng, 2), "u1": rand_text(rng, n1)}
        untrusted = ["u1"]
        if n2:
            fields["u2"] = rand_text(rng, n2)
            untrusted.append("u2")
        r = req(fields, untrusted)
        dec = Scripted(trial, poison=rng.choice([None, "golf"]))
        rec = C.certify(dec, r, k, max_decide_calls=10_000, escalate_on_alarm=False)
        want = oracle_certified(dec, r, k)
        assert rec["certified"] == want, (trial, fields)
        assert rec["n_segments"] == sum(len(C.segment(fields[f])) for f in untrusted)
        agree += 1
        n_cert += want
    assert agree == 250 and 0 < n_cert < 250  # both outcomes exercised


def test_certify_on_real_decide_path_matches_oracle():
    """Through decide.Decider (OracleBackend, hash-random logits) via the canonical bridge."""
    try:
        from decide import Decider
        from decide.backends import OracleBackend
        from safety_eval.decide_bridge import DecideBridge
    except ImportError:
        pytest.skip("decide/ not importable")
    br = DecideBridge(Decider(OracleBackend(margin=0.0), enable_v0_1_deltas=True))
    for r0 in data.build(6, 6, split="dev"):
        r = data.strip_meta(r0)
        rec = C.certify(br, r, 1, escalate_on_alarm=False)
        assert rec["certified"] == oracle_certified(br, r, 1)
        json.dumps(rec)


# ------------------------------------------------------------------ soundness on inserted-sentence attacks
@pytest.mark.parametrize("k", [1, 2])
def test_soundness_on_random_insertions(k):
    rng = random.Random(7 + k)
    checked = changed_and_escalated = 0
    for trial in range(800):
        clean = req({"trusted": "Customer asks to pay.", "u": rand_text(rng, rng.randint(1, 4))}, ["u"])
        n_ins = rng.randint(1, k)
        att_text = insert_sentences(rng, clean["state"]["u"], [rand_sentence(rng, rng.random() < .5)
                                                             for _ in range(n_ins)])
        attacked = dict(clean, state=dict(clean["state"], u=att_text))
        assert C.is_covered(attacked, clean, k), (clean["state"]["u"], att_text)
        dec = Scripted(trial, poison="golf")
        rec = C.certify(dec, attacked, k, escalate_on_alarm=False)
        d_clean = C.decision_of(clean, dec(clean))
        d_att = C.decision_of(attacked, dec(attacked))
        if rec["certified"]:
            checked += 1
            assert d_att == d_clean, "certified decision differs from the clean decision"
        if d_att != d_clean:
            assert not rec["certified"] and rec["action"] == "escalate"
            changed_and_escalated += 1
    assert checked > (20 if k == 1 else 5) and changed_and_escalated > 20  # non-vacuous in both directions


def test_two_field_budget_is_joint():
    clean = req({"u1": "Alpha x. Bravo y.", "u2": "Charlie z."}, ["u1", "u2"])
    att = dict(clean, state={"u1": "Alpha x. Golf p. Bravo y.", "u2": "Golf q. Charlie z."})
    assert not C.is_covered(att, clean, 1) and C.is_covered(att, clean, 2)
    dec = Scripted(3, poison="golf")
    # Two poison insertions exceed k = 1: erasing either one leaves the other, so k = 1 certifies the steered
    # decision. This is the documented limit (the guarantee is per budget), not a soundness failure.
    assert C.certify(dec, att, 1)["certified"] is True and C.certify(dec, att, 1)["decision"] == {"q": "d"}
    assert C.certify(dec, att, 2)["certified"] is False  # erasing both golf sentences changes the decision


def test_trusted_fields_are_never_erased():
    dec = Scripted(1, poison="golf")
    r = req({"trusted": "Golf is the instruction.", "u": "Alpha x. Bravo y."}, ["u"])
    rec = C.certify(dec, r, 1)
    assert rec["certified"] and rec["n_segments"] == 2  # poison in a trusted field is not an erasure target
    r2 = req({"trusted": "Golf is the instruction.", "u": "Alpha x. Bravo y."})  # no provenance -> all untrusted
    assert C.certify(dec, r2, 1)["n_segments"] == 3


def test_cost_guard_escalates_without_partial_certification():
    dec = Scripted(0)
    r = req({"u": " ".join(f"Alpha {i}." for i in range(30))}, ["u"])
    rec = C.certify(dec, r, 2, max_decide_calls=256)
    assert rec["reason"] == "certificate_budget_exceeded" and rec["action"] == "escalate"
    assert rec["n_decide_calls"] == 1 and not rec["certified"]
    assert C.n_erasure_sets(30, 2) == 465
    assert C.certify(Scripted(0), r, 2, max_decide_calls=1000)["n_decide_calls"] > 256


def test_fail_closed_cases():
    dec = Scripted(0, refuse_on="hotel")
    assert C.certify(dec, req({"u": "Hotel x. Alpha."}, ["u"]))["reason"] == "decide_status:unsupported"
    # an erasure whose origin is refused counts as a changed decision
    rec = C.certify(Scripted(0, refuse_on="Alpha x. Alpha z"), req({"u": "Alpha x. Alpha y. Alpha z."}, ["u"]))
    assert rec["certified"] is False and rec["first_changed"]["decision"]["status"] == "unsupported"
    rec = C.certify(Scripted(0, refuse_on="zzz"), req({"u": {"nested": "x"}}, ["u"]))
    assert rec["reason"] == "untrusted_field_not_text" and rec["action"] == "escalate"
    with pytest.raises(ValueError):
        C.certify(dec, req({"u": "Alpha."}, ["u"]), 0)


def test_alarm_escalates_but_stays_certified():
    dec = Scripted(5)
    r = req({"u": "Alpha x. Bravo y."}, ["u"], allowed=["b", "c", "d"])
    rec = C.certify(dec, r, 1)
    full_choice = dec.choice("Alpha x. Bravo y.")
    if full_choice == "a":  # raw argmax outside the allowed set -> alarm
        assert rec["alarm"] and rec["action"] == "escalate" and rec["reason"] == "alarm"


def test_certified_decider_wrapper_is_json_ready():
    dec = KeywordDecider()
    w = C.CertifiedDecider(dec, k=1)
    r = data.strip_meta(data.build(4, 2, split="dev")[0])
    out = w(r)
    assert set(out["abstention"]) == {"abstained", "reason", "mechanism"}
    assert out["abstention"]["abstained"] == (out["certificate"]["action"] != "execute")
    json.dumps(out)


# ------------------------------------------------------------------ stand-in end to end (D1-A, cost metrics)
@pytest.fixture(scope="module")
def d1_rows():
    reqs = data.build(30, 20, split="dev")
    vs = attacks.build_variants(reqs, hijack=[], d1=None)  # clean + D1-A (B is held out)
    return C.evaluate(KeywordDecider(), vs, k=1)


def test_standin_d1_guarantee_and_cost_metrics(d1_rows):
    m = C.g4_metrics(d1_rows, n_boot=50)
    ins, cl = m["inserted"], m["clean"]
    assert m["guarantee_checked_units"] > 0
    assert ins["covered_rate"]["rate"] == 1.0  # the ported D1 inserter is inside the segment threat model
    assert ins["certified_steer_rate"]["num"] == 0  # by construction
    assert ins["steer_rate_uncertified"]["num"] > 0  # the stand-in IS steered without the certificate
    for key in ("escalation_rate", "auto_execute_rate", "certified_but_wrong_rate", "decide_calls"):
        assert key in cl and key in ins
    assert cl["escalation_rate"]["rate"] + cl["auto_execute_rate"]["rate"] == pytest.approx(1.0)
    assert cl["decide_calls"]["max"] <= 1 + C.n_erasure_sets(10, 1)
    json.dumps(m)


# ------------------------------------------------------------------ randomized ablation (smoothing)
def test_smoothing_bound_arithmetic():
    assert C.ablation_delta(0.2, 1) == pytest.approx(0.2)
    assert C.ablation_delta(0.2, 2) == pytest.approx(0.36)
    assert C.insertion_radius(0.61, 0.2) == 1  # 0.61 > (1 + .2)/2, 0.61 < (1 + .36)/2
    assert C.insertion_radius(0.59, 0.2) == 0
    assert C.edit_radius(0.71, 0.2) == 1  # 2*.71 - 1 = .42 > .4
    assert C.edit_radius(0.69, 0.2) == 0


def test_smoothed_certificate_sound_on_poison_insertions():
    """Smoothed decision on the attacked request equals the smoothed decision on the clean request whenever
    certified (reference estimated with many samples). Probabilistic: allowed failures <= delta-scale."""
    rng = random.Random(11)
    certified = bad = 0
    for trial in range(60):
        clean = req({"u": rand_text(rng, rng.randint(2, 4))}, ["u"])
        att = dict(clean, state={"u": insert_sentences(rng, clean["state"]["u"], ["Golf now."])})
        dec = Scripted(trial, poison="golf")
        rec = C.smoothed_certify(dec, att, 1, keep_prob=0.15, n_select=16, n_estimate=200, delta=0.05, seed=trial)
        if rec["certified"]:
            certified += 1
            dist, how = C.smoothed_distribution(dec, clean, 0.15)  # exact g(clean)
            assert how == "exact" and sum(dist.values()) == pytest.approx(1.0)
            top = max(dist, key=dist.get)
            bad += C._jsonable_decision(top) != rec["decision"]
    assert certified > 20 and bad <= 2  # probabilistic certificate: rare failures allowed (prob <= delta each)


def test_smoothed_distribution_exact_matches_monte_carlo():
    dec = Scripted(4, poison="golf")
    r = req({"u": "Alpha x. Golf now. Bravo y. Charlie z."}, ["u"])
    ex, how = C.smoothed_distribution(dec, r, 0.3)
    mc, how2 = C.smoothed_distribution(dec, r, 0.3, exact_max_segments=0, n_mc=4000, seed=1)
    assert how == "exact" and how2.startswith("monte_carlo")
    for d, p in ex.items():
        assert mc.get(d, 0.0) == pytest.approx(p, abs=0.03)


def test_smoothed_decider_os_entropy_and_rewrite():
    dec = Scripted(2, poison="golf")
    r = req({"u": "Alpha x. Golf now. Bravo y."}, ["u"])
    rec = C.smoothed_certify(dec, r, 1, keep_prob=0.1, n_estimate=200)  # seed=None -> SystemRandom
    assert rec["seed"] == "os-entropy"
    out = C.SmoothedDecider(dec, k=1, keep_prob=0.1, n_estimate=200, seed=3)(r)
    json.dumps(out)
    if out["certificate"]["certified"] and not out["certificate"]["smoothed_matches_base"]:
        assert out["answers"]["q"]["choice"] == out["certificate"]["decision"]["q"]
        assert out["base_answers"]["q"]["choice"] == "d"  # base decider followed the poison


# ------------------------------------------------------------------ PR #20 review follow-ups
class Evidence:
    """Decisive evidence 'bravo' -> a; 'golf' (attack) or no evidence -> d (the default is also the target)."""

    def __call__(self, r):
        t = " ".join(v for v in r["state"].values() if isinstance(v, str)).lower()
        ch = "a" if ("bravo" in t and "golf" not in t) else "d"
        probs = {o: (0.7 if o == ch else 0.1) for o in OPTS}
        return {"status": "ok", "answers": {"q": {"type": "choice", "choice": ch, "probabilities": probs}},
                "raw_probabilities": {"q": probs}}


@pytest.mark.parametrize("tail", ["\n", "\n\n", " ", "\n "])
@pytest.mark.parametrize("sep", ["", " "])
def test_trailing_whitespace_insertion_is_covered_and_caught(tail, sep):
    clean = req({"u": "Alpha x. Bravo y." + tail}, ["u"])
    att = dict(clean, state={"u": clean["state"]["u"] + sep + "Golf now."})
    assert C.is_covered(att, clean, 1)  # sent-leadws-v2 trailing-whitespace origins
    rec = C.certify(Evidence(), att, 1)
    assert rec["decision"] == {"q": "d"} and not rec["certified"]


@pytest.mark.parametrize("text", ["Alpha x. Golf now Bravo y.",  # merge, no terminal punctuation
                                  "Alpha x. Bravo golf y.",  # mid-sentence
                                  "Alpha x. Bravo y.golf now"])  # no space
def test_uncovered_insertions_sentence_unit_vs_span_unit(text):
    clean = req({"u": "Alpha x. Bravo y."}, ["u"])
    att = dict(clean, state={"u": text})
    dec = Evidence()
    assert C.decision_of(clean, dec(clean)) != C.decision_of(att, dec(att))
    # sentence unit: outside the threat model, and certified-and-changed (the documented limit)
    assert not C.is_covered(att, clean, 1)
    assert C.certify(dec, att, 1)["certified"] is True
    # span unit (hardened): covered and not certified
    assert C.is_covered(att, clean, 1, unit="span")
    rec = C.certify(dec, att, 1, unit="span", max_decide_calls=10_000)
    assert rec["certified"] is False and rec["unit"] == "span"


def test_span_unit_matches_brute_force_and_requires_k1():
    rng = random.Random(5)
    for trial in range(60):
        text = rand_text(rng, rng.randint(1, 3))
        r = req({"u": text}, ["u"])
        dec = Scripted(trial, poison="golf")
        toks = C.tokens(text)
        assert "".join(toks) == text
        full = C.decision_of(r, dec(r))
        want = all(C.decision_of(o, dec(o)) == full for o in
                   (dict(r, state={"u": "".join(toks[:i] + toks[j:])})
                    for i in range(len(toks)) for j in range(i + 1, len(toks) + 1)))
        assert C.certify(dec, r, 1, unit="span", max_decide_calls=10_000)["certified"] == want
    assert C.certify(Scripted(0), req({"u": "Alpha."}, ["u"]), 2, unit="span")["reason"] == "span_unit_requires_k1"


def test_is_covered_rejects_trusted_field_edits():
    clean = req({"trusted": "Pay the due invoice.", "u": "Alpha x."}, ["u"])
    att = dict(clean, state={"trusted": "Pay the other invoice.", "u": "Alpha x. Golf now."})
    assert not C.is_covered(att, clean, 1)
    att2 = dict(clean, state={"trusted": "Pay the due invoice.", "u": "Alpha x. Golf now."})
    assert C.is_covered(att2, clean, 1)
    att3 = dict(att2, policy={"allowed": {"q": ["a"]}})
    assert not C.is_covered(att3, clean, 1)


def test_certified_but_wrong_denominator_is_certified():
    base = {"cluster": "c", "template": "t", "kind": "clean", "covered": True, "steered": False,
            "changed_vs_clean": False, "n_decide_calls": 1, "suffix_cost": 0.0, "arm": "exact"}
    rows = [dict(base, certified=True, auto=True, correct=True),
            dict(base, certified=True, auto=False, correct=False),  # certified but alarmed -> escalated
            dict(base, certified=False, auto=False, correct=False)]
    m = C.g4_metrics(rows)["clean"]
    assert m["certified_but_wrong_rate"]["num"] == 1 and m["certified_but_wrong_rate"]["den"] == 2
    assert m["wrong_given_auto"]["num"] == 0 and m["wrong_given_auto"]["den"] == 1


def test_uncovered_stress_arm_reports_certified_and_changed():
    reqs = data.build(6, 4, split="dev")
    rows = C.uncovered_stress(KeywordDecider(), reqs, 1)
    summ = C.uncovered_summary(rows)  # asserts certified-and-changed is 0 on covered rows
    assert set(summ) == set(C.UNCOVERED_MODES)
    assert summ["trailing_ws"]["covered_rate"] == 1.0
    assert summ["mid_sentence"]["covered_rate"] < 1.0
    json.dumps(summ)
