"""heldout-draw-v1: the seeded 9/9 split of the zero-coverage benchmarks reproduces, and the contamination check
refuses anything tagged with a frozen family while passing the existing 150k mix."""

import pytest

from harness import config, families, heldout

# Literal pins (deliberately duplicated from config so editing both is required to move them)
FAMILIES_V1_SHA256 = "e0e4bd5f76ca928f2a05c3f52150592aed462b4018350346eb723e9b40096f46"
HELDOUT_DRAW_V1_SHA256 = "c3172003e65e933450337476e6122b2fa90d187cf8d09c56852e7a038915a4cb"
M150K = config.DATA_ROOT / "m2" / "build" / "sample-150000-seed20261006" / "sample-150000-seed20261006.manifest.jsonl.gz"

# (source, family) pairs present in the committed 150k build (manifest sha256 0be33669...)
MIX_150K = [("anli", "multi"), ("anli", "nli"), ("banking77", "intent_banking"), ("banking77", "multi"), ("boolq", "multi"), ("boolq", "yesno_passage"),
            ("clinc150", "intent_general"), ("clinc150", "multi"), ("glaive_fc", "tool_select"), ("gsm8k", "math_mcq"), ("gsm8k", "multi"),
            ("hellaswag", "continuation"), ("hellaswag", "multi"), ("humicroedit_s1", "humor_score"), ("humicroedit_s2", "humor_pair"), ("mixed", "multi"),
            ("mmlu_aux", "mcq"), ("mmlu_aux", "multi"), ("nq", "retrieval"), ("sst5", "multi"), ("sst5", "sentiment_score"),
            ("synthetic_factchange", "factchange_delivery"), ("synthetic_factchange", "factchange_rating"), ("synthetic_factchange", "factchange_renewal"),
            ("synthetic_factchange", "factchange_schedule"), ("synthetic_factchange", "factchange_spend_limit"), ("synthetic_factchange", "factchange_stock"),
            ("toolace", "tool_multi"), ("toolace", "tool_select"), ("when2call", "response_select"), ("winogrande", "fill_blank"), ("winogrande", "multi")]


def test_draw_reproduces_from_seed():
    s = heldout.spec()
    assert s["seed"] == 20261007
    pool = heldout.pool("0.3")
    assert pool == s["pool"] and len(pool) == 18 and 48 not in pool
    fz, gen = heldout.draw(s["seed"], pool)
    assert fz == s["frozen"] == [9, 20, 22, 31, 32, 33, 44, 58, 64]
    assert gen == s["generatable"] == [23, 38, 40, 41, 43, 50, 56, 59, 61]
    assert not set(fz) & set(gen) and set(fz) | set(gen) == set(pool)
    # the draw is the draw: the three E6 targets sit where it put them, nothing forced
    assert 43 in gen and 44 in fz and 9 in fz
    # a different seed gives a different split (the seed matters)
    assert heldout.draw(s["seed"] + 1, pool)[0] != fz


def test_family_groups_follow_the_draw():
    assert sorted(families.members("new_domain_frozen", "0.3")) == heldout.frozen()
    assert set(families.members("new_domain_generatable", "0.3")) | set(families.members("new_domain_frozen", "0.3")) == set(families.members("new_domain_proxy", "0.3"))


@pytest.mark.parametrize("bad", [
    {"name": "gen_x", "task_family": "causal_query"},
    {"name": "gen_y", "targets": [44]},
    {"name": "gen_cladder_v1", "task_family": "whatever"},
    {"name": "gen_z", "targets": ["catalog:58"]},
    {"name": "BBH-style", "task_family": "mixed"},
    {"name": "device records", "task_family": "device_state_record"},
    {"name": "chess positions", "targets": []},
    {"name": "no declaration"},
    # re-review B1: empty declarations, dict-keyed ids, free-text descriptions
    {"name": "x", "targets": []},
    {"name": "x", "task_family": None},
    {"name": "x", "task_family": ""},
    {"name": "x", "targets": {"58": 1}},
    {"name": "x", "task_family": "logic", "description": "BIG-Bench Hard style"},
    # re-review S3: more fields, case-insensitive keys, id spellings, list-of-dict targets
    {"task_family": "x", "dataset": "bbh"},
    {"task_family": "x", "task": "MuSR-like murder mysteries"},
    {"task_family": "x", "benchmark_name": "New Yorker"},
    {"task_family": "x", "tags": ["chess"]},
    {"task_family": "x", "Meta": {"Source": "cladder-gen"}},
    {"targets": ["catalog: 58"]},
    {"targets": ["catalog-58"]},
    {"targets": [58.0]},
    {"targets": "58, 9"},
    {"targets": [{"catalog_id": 58}]},
    {"targets": [{"id": 44}]},
])
def test_generator_check_refuses_frozen(bad):
    with pytest.raises(heldout.FrozenFamilyError):
        heldout.check_generator(bad)


@pytest.mark.parametrize("ok", [
    {"name": "gen_cruxeval_exec", "task_family": "code_execution", "targets": [43]},
    {"name": "gen_hover_claims", "task_family": "claim_verification", "targets": [61]},
    {"name": "gen_ragtruth", "targets": ["catalog:59"]},
    {"name": "gen_hover", "task_family": "claim_verification", "targets": [{"catalog_id": 61}], "description": "HoVer-style multi-hop claims, 3 hops"},
])
def test_generator_check_passes_generatable(ok):
    assert heldout.check_generator(ok) is ok


def test_record_check():
    with pytest.raises(heldout.FrozenFamilyError):
        heldout.check_record({"id": "r1", "source": "gen_musr", "family": "mcq"})
    with pytest.raises(heldout.FrozenFamilyError):
        heldout.check_record({"id": "r2", "source": "gen", "family": "x", "meta": {"targets": [22]}})
    for src, fam in MIX_150K:
        heldout.check_record({"id": "m", "source": src, "family": fam})


def test_training_source_families_are_not_frozen():
    d = families.load()
    frozen_fams = {d["benchmarks"][str(n)]["family"] for n in heldout.frozen()}
    assert not set(d["training_sources"].values()) & frozen_fams


@pytest.mark.skipif(not M150K.exists(), reason="150k build not present locally")
def test_existing_150k_manifest_passes():
    rep = heldout.check_files([M150K])
    assert rep["records"] == 150000 and rep["refused"] == 0


def test_manifest_hashes_pinned():
    import hashlib

    # families-v1 and heldout-draw-v1 are frozen contracts: any edit must be a new version file
    assert hashlib.sha256(families.FAMILIES.read_bytes()).hexdigest() == FAMILIES_V1_SHA256
    assert hashlib.sha256(heldout.DRAW.read_bytes()).hexdigest() == HELDOUT_DRAW_V1_SHA256
    assert (config.FAMILIES_V1_SHA256, config.HELDOUT_DRAW_V1_SHA256) == (FAMILIES_V1_SHA256, HELDOUT_DRAW_V1_SHA256)
    # the frozen ids themselves, so a re-pinned file with a different draw also fails
    ids_sha = hashlib.sha256(",".join(map(str, heldout.frozen())).encode()).hexdigest()
    assert ids_sha == hashlib.sha256(b"9,20,22,31,32,33,44,58,64").hexdigest()


def test_record_check_ignores_item_content_and_option_targets():
    # injection records name a hijack target option ("target": "k_9e2a6267"); content keys are never scanned
    rec = {"id": "m2:x~inj", "source": "mmlu_aux", "family": "mcq", "aug": {"injection": {"target": "k_9e2a6267"}},
           "state": "The New Yorker cartoon about chess and BBH", "questions": {"q": {"criteria": {"A": "CLadder"}}}, "labels": {"result": 9}}
    assert heldout.check_record(rec) is rec
