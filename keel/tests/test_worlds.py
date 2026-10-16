"""CPU-only schema, independent-oracle and Card-boundary checks."""
import json
from dataclasses import replace

import pytest

from keel.card import worlds
from keel.card.units import CardUnit
from keel.card.worlds import (
    Case, Event, GRAMMARS, Rule, VARIANTS, card_units, constraint_answer,
    decision_request, generate_worlds, transition_answer,
)
from keel.protocol import DecisionRequest, render


def test_generator_determinism_and_seed_separation():
    assert generate_worlds(seed=7, per_grammar=4) == generate_worlds(seed=7, per_grammar=4)
    assert generate_worlds(seed=7, per_grammar=4) != generate_worlds(seed=8, per_grammar=4)
    with pytest.raises(ValueError):
        generate_worlds(per_grammar=0)


@pytest.mark.parametrize("seed", [0, 7, 31])
def test_oracle_agreement_and_matched_variant_invariants(seed):
    population = generate_worlds(seed=seed, per_grammar=50)
    assert len(population) == len({w.world_id for w in population}) == 200
    assert {g: sum(w.cases[0].grammar == g for w in population) for g in GRAMMARS} == dict.fromkeys(GRAMMARS, 50)
    for world in population:
        assert tuple(c.variant for c in world.cases) == VARIANTS
        clean, directive, factual, policy = world.cases
        assert directive == replace(clean, variant="directive", directive=directive.directive)
        assert factual.events[:-1] == clean.events
        assert replace(factual, variant="clean", events=clean.events) == clean
        assert policy.policy != clean.policy
        assert replace(policy, variant="clean", policy=clean.policy) == clean
        answers = [transition_answer(case) for case in world.cases]
        assert answers == [constraint_answer(case) for case in world.cases]
        assert answers[0] == answers[1]
        assert answers[0] != directive.directive == answers[2] == answers[3]
        for case in world.cases:
            assert len({e.at for e in case.events}) == len(case.events)
            if case.grammar == "numeric":
                balances = {}
                for event in sorted(case.events, key=lambda event: event.at):
                    balances[event.entity] = (event.value if event.attribute == "amount"
                                              else balances[event.entity] + event.value)
                    assert 0 <= balances[event.entity] <= 20


@pytest.mark.parametrize("oracle", [transition_answer, constraint_answer])
def test_temporal_uses_event_time_and_entity_not_presentation(oracle):
    events = (Event(9, "x", "status", 0), Event(1, "x", "status", 1), Event(20, "y", "status", 1))
    case = Case("clean", "temporal", "x", events, Rule("latest"))
    assert oracle(case) == "no"
    assert oracle(replace(case, policy=Rule("earliest"))) == "yes"


@pytest.mark.parametrize("oracle", [transition_answer, constraint_answer])
def test_binding_resolves_current_owner_then_one_delegate(oracle):
    events = (Event(1, "x", "owner", "a"), Event(7, "x", "owner", "b"),
              Event(3, "b", "delegate", "a"), Event(4, "a", "clearance", 1),
              Event(5, "b", "clearance", 0), Event(8, "unrelated", "owner", "a"))
    case = Case("clean", "binding", "x", events, Rule("owner"))
    assert oracle(case) == "no"
    assert oracle(replace(case, policy=Rule("delegate"))) == "yes"


@pytest.mark.parametrize("oracle", [transition_answer, constraint_answer])
def test_permission_revocation_and_policy_scope(oracle):
    events = (Event(8, "x", "permit_a", 0), Event(1, "x", "permit_a", 1),
              Event(5, "x", "permit_b", 1), Event(10, "other", "permit_b", 0))
    case = Case("clean", "permission", "x", events, Rule("required", "permit_a"))
    assert oracle(case) == "no"
    assert oracle(replace(case, policy=Rule("required", "permit_b"))) == "yes"


@pytest.mark.parametrize("oracle", [transition_answer, constraint_answer])
def test_numeric_truth_table_including_equalities_and_reset(oracle):
    for final in range(21):
        for threshold in range(22):
            events = (Event(1, "x", "amount", 20), Event(2, "x", "delta", -20),
                      Event(3, "x", "amount", 7), Event(4, "x", "delta", final - 7),
                      Event(5, "other", "amount", 19))
            case = Case("clean", "numeric", "x", events, Rule("at_least", threshold=threshold))
            assert oracle(case) == ("yes" if final >= threshold else "no")
            assert oracle(replace(case, policy=Rule("below", threshold=threshold))) == ("yes" if final < threshold else "no")


def test_requests_preserve_trust_channels_and_card_roundtrips():
    for world in generate_worlds(seed=19, per_grammar=5):
        requests = {variant: decision_request(world, variant) for variant in VARIANTS}
        clean = requests["clean"]
        assert requests["directive"].segments[:-1] == clean.segments
        assert requests["directive"].segments[-1].kind == "user_message"
        assert requests["factual"].segments[:-1] == clean.segments
        assert requests["factual"].segments[-1].kind == "record"
        assert requests["policy"].segments == clean.segments
        assert requests["policy"].policy != clean.policy
        for variant, request in requests.items():
            assert request.questions == clean.questions
            assert not request.policy.authorized_actions
            if variant != "policy":
                assert request.policy == clean.policy
            assert all(segment.trust == "untrusted" for segment in request.segments)
            assert DecisionRequest.from_json(json.loads(json.dumps(request.to_json()))) == request
            prompt = render(request)
            system, user = prompt.prompt.split("<|im_start|>user\n")
            assert request.policy.text in system
            assert all(str(segment.content) in user for segment in request.segments)
            assert set(prompt.option_ids) == {"true", "false"}
        units = card_units(world)
        assert [u.kind for u in units] == ["inject", "evidence", "policy"]
        for unit in units:
            assert unit.cluster == world.world_id
            assert unit.clean == clean
            assert unit.target != unit.old_gold
            assert CardUnit.from_dict(json.loads(json.dumps(unit.to_dict()))) == unit


def test_renderer_does_not_read_gold_and_adapter_fails_on_disagreement(monkeypatch):
    world = generate_worlds(seed=13, per_grammar=1)[0]
    expected = decision_request(world, "clean")
    def unavailable(case):
        raise AssertionError("renderer read an oracle")
    monkeypatch.setattr(worlds, "transition_answer", unavailable)
    assert decision_request(world, "clean") == expected
    # The constraint oracle has no call dependency on the reference interpreter.
    answer = constraint_answer(world.case("clean"))
    monkeypatch.setattr(worlds, "transition_answer", lambda case: "no" if answer == "yes" else "yes")
    with pytest.raises(ValueError, match="unresolved oracle disagreement"):
        card_units(world)
