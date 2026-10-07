"""heldout-draw-v1: the seeded 9/9 split of the zero-coverage benchmarks reproduces, and the contamination check
refuses anything tagged with a frozen family while passing the existing 150k mix."""

import pytest

from harness import config, families, heldout

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
])
def test_generator_check_refuses_frozen(bad):
    with pytest.raises(heldout.FrozenFamilyError):
        heldout.check_generator(bad)


@pytest.mark.parametrize("ok", [
    {"name": "gen_cruxeval_exec", "task_family": "code_execution", "targets": [43]},
    {"name": "gen_hover_claims", "task_family": "claim_verification", "targets": [61]},
    {"name": "gen_ragtruth", "targets": ["catalog:59"]},
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
