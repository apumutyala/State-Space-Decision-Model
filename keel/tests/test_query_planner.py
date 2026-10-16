"""Synthetic CPU tests: planner inputs, literal clauses and real host enforcement."""
import inspect

import pytest

from keel.host import ArgumentClause
from keel.workflows.query_planner import QueryGrant, ToolSpec, extract_literals, plan_query


def spec(name, **fields):
    return ToolSpec(name, "trusted tool schema", {key: {"type": kind} for key, kind in fields.items()})


PAY = spec("send_money", recipient="string", amount="number", subject="string", date="string")
READ = spec("get_balance")


def allowed(grant, tool, args, specs):
    authority = grant.authority(specs)
    return tool in authority.authorized_actions("agent") and authority.arguments_allowed(tool, args)


def test_interface_cannot_receive_environment_or_ground_truth():
    assert tuple(inspect.signature(plan_query).parameters) == ("prompt", "specs", "mode")
    with pytest.raises(TypeError):
        plan_query("Pay the bill", (PAY,), environment={"recipient": "attacker"})


def test_grants_depend_on_query_not_previous_calls():
    clean = plan_query("Show my balance", (PAY, READ))
    plan_query("Pay EUR 50 to DE02120300000000202051", (PAY, READ))
    assert clean.manifest() == plan_query("Show my balance", (PAY, READ)).manifest()
    assert clean.tools == ("get_balance",)


def test_unknown_write_tool_is_not_automatically_granted():
    assert plan_query("Do everything", (spec("execute_python"), READ)).tools == ("get_balance",)


def test_fixed_payment_fields_match_in_host():
    tools = (PAY, READ)
    grant = plan_query("Pay EUR 50 to DE02120300000000202051 on 2027-01-01", tools)
    args = {"recipient": "DE02120300000000202051", "amount": 50, "date": "2027-01-01"}
    assert allowed(grant, "send_money", args, tools)
    for change in ({"recipient": "GB82WEST12345698765432"}, {"amount": 51}, {"amount": True}, {"amount": "50"}):
        assert not allowed(grant, "send_money", args | change, tools)
    assert all(isinstance(c, ArgumentClause) for c in grant.clauses)


def test_one_of_keeps_literal_destinations_not_arbitrary_values():
    tool = spec("send_channel_message", channel="string", body="string")
    grant = plan_query("Post to #alpha and #beta channels", (tool,))
    assert grant.clauses[0].one_of["channel"] == ("alpha", "beta")
    assert allowed(grant, tool.name, {"channel": "beta", "body": "arbitrary"}, (tool,))
    assert not allowed(grant, tool.name, {"channel": "gamma"}, (tool,))


def test_upper_bound_is_used_only_for_explicit_limit():
    grant = plan_query("Pay up to EUR 50 to DE02120300000000202051", (PAY,))
    assert grant.clauses[0].at_most == {"amount": 50}
    assert allowed(grant, PAY.name, {"recipient": "DE02120300000000202051", "amount": 49}, (PAY,))
    assert not allowed(grant, PAY.name, {"recipient": "DE02120300000000202051", "amount": 51}, (PAY,))
    fixed = plan_query("Pay EUR 50 to DE02120300000000202051", (PAY,))
    assert fixed.clauses[0].fixed["amount"] == 50


def test_missing_values_free_or_withheld_according_to_mode():
    prompt = "Pay the invoice using the bank details in the file"
    assert plan_query(prompt, (PAY,), mode="literals").tools == ("send_money",)
    assert plan_query(prompt, (PAY,), mode="literal_required").tools == ()
    assert plan_query("Pay EUR 50 to DE02120300000000202051", (PAY,), mode="literal_required").tools == ("send_money",)


def test_tools_arm_does_not_constrain_arguments():
    grant = plan_query("Pay EUR 50", (PAY,), mode="tools")
    assert grant.clauses == ()
    assert allowed(grant, PAY.name, {"recipient": "any", "amount": 1000}, (PAY,))


def test_email_lists_are_fixed_as_whole_arrays():
    tool = spec("send_email", recipients="array", body="string")
    grant = plan_query("Send email to a@example.org and b@example.org", (tool,))
    assert allowed(grant, tool.name, {"recipients": ["a@example.org", "b@example.org"]}, (tool,))
    assert not allowed(grant, tool.name, {"recipients": ["a@example.org", "evil@example.org"]}, (tool,))
    assert not allowed(grant, tool.name, {"recipients": ["a@example.org"]}, (tool,))


def test_names_channels_files_dates_and_urls_are_prompt_literals():
    values = extract_literals('Tell Alice Smith to read report.csv in channel "ops" on 2027-01-02 at https://example.org/x.')
    assert "Alice Smith" in values["name"]
    assert values["channel"] == ["ops"]
    assert values["file"] == ["report.csv"]
    assert values["date"] == ["2027-01-02"]
    assert values["url"] == ["https://example.org/x"]


def test_host_defaults_used_without_changing_clause_semantics():
    tool = ToolSpec("get_recent", "trusted", {"n": {"type": "integer", "default": 100}})
    grant = QueryGrant((tool.name,), (ArgumentClause(tool.name, fixed={"n": 100}),))
    assert allowed(grant, tool.name, {}, (tool,))
    assert not allowed(grant, tool.name, {"n": 1}, (tool,))


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="unknown planner mode"):
        plan_query("Pay the bill", (PAY,), mode="guess")
