"""CPU reproduction of PR #38's integer carry defect and its exact fix."""
from decimal import Decimal, Inexact, Rounded, localcontext
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from language_growth.kernel import _exact_sum, check_program
from language_growth.realize import render_checked
from language_growth.tests.test_exact_sum import opted_world, program
from language_growth.types import Fact

BASE = "6d8fa4b655c2a7242473ad4681b73a61527f86be"
prior = types.ModuleType("language_growth._prior_kernel")
sys.modules[prior.__name__] = prior
source = subprocess.check_output(["git", "show", BASE + ":language_growth/kernel.py"], cwd=ROOT)
exec(compile(source, "<prior-kernel>", "exec"), prior.__dict__)
prior_render = types.ModuleType("language_growth._prior_realize")
source = subprocess.check_output(["git", "show", BASE + ":language_growth/realize.py"], cwd=ROOT)
source = source.replace(b"from .kernel import", b"from ._prior_kernel import")
exec(compile(source, "<prior-realize>", "exec"), prior_render.__dict__)

# Scope the default limit to this verification process; never disable it.
sys.set_int_max_str_digits(4300)
host, catalog = opted_world()
for record in catalog["records"]:
    if record["record_ref"] in ("atlas-v1", "boreal-v1"):
        record["fields"]["price"].update(value=9 * 10**4299, value_type="integer")
old = prior.check_program(program(host), host, catalog)
assert old.status == "accepted"
failures = []
for name, action in (("render", lambda: prior_render.render_checked(old, host, catalog)),
                     ("json", lambda: json.dumps(old.to_dict()))):
    try:
        action()
    except ValueError as error:
        assert "4300 digits" in str(error)
        failures.append(name)
assert failures == ["render", "json"]
fixed = check_program(program(host), host, catalog)
assert fixed.status == "accepted"
wanted = "18" + "0" * 4299
text = render_checked(fixed, host, catalog)
assert text == render_checked(fixed, host, catalog)
assert wanted + ' "USD"' in text
assert json.loads(json.dumps(fixed.to_dict()))["outputs"][0]["content"]["value"] == wanted

# An independent rational oracle checks value, signs and Decimal exponent
# handling, including positive exponents not currently exposed by source syntax.
values = (0, 7, -8, 10**80, -(10**80), Decimal("0.125"), Decimal("-0.125"),
          Decimal("-0.00"), Decimal("1E+80"), Decimal("-1E+80"), Decimal("1E-80"))
def fact(value):
    return Fact("synthetic", "Price", value, "USD", "2026-10", "source", "record", 1, "price", "price")

count = 0
with localcontext() as context:
    context.prec = 1
    context.Emin, context.Emax = -1, 1
    context.traps[Inexact] = context.traps[Rounded] = True
    for left in values:
        for right in values:
            result = _exact_sum(fact(left), fact(right))
            assert isinstance(result.value, Decimal)
            assert Fraction(result.value) == Fraction(left) + Fraction(right)
            assert json.loads(json.dumps(result.to_dict()))["value"] == str(result.value)
            count += 1
print(json.dumps({"prior_head": BASE, "prior_status": old.status,
                  "prior_failures": failures, "fixed_status": fixed.status,
                  "result_digits": len(wanted), "deterministic_render": True,
                  "json_roundtrip": True, "rational_oracle_pairs": count}))
