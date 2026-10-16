"""Synthetic fixtures and exact mutation labels; deliberately no verifier imports."""
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction as F
from functools import lru_cache
import json
import math
import random
import re

ENDPOINTS = ("basic_arithmetic", "decimal_arithmetic", "gcd", "lcm",
             "prime_factorization", "number_sorting", "polynomial_equations",
             "syllogism", "math_verify", "gsm8k")
DEV = ("boundary", "sign", "partial", "candidates", "quantifier", "whitespace", "rational")
HELD = ("duplicates", "trailing", "units", "latex", "tolerance", "delimiters", "scientific")


def decimal_text(value):
    value = F(value)
    with localcontext() as ctx:
        ctx.prec = 60
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def numeric(text):
    """Parse only our closed numeric grammar, without eval or SymPy."""
    text = text.strip()
    if text.startswith(r"\boxed{") and text.endswith("}"):
        text = text[7:-1]
    match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if match:
        return numeric(match[1]) / numeric(match[2])
    if text.startswith("(") and text.endswith(")"):
        return numeric(text[1:-1])
    if text.endswith("%"):
        return numeric(text[:-1]) / 100
    return F(text)


def rounded(value, kind):
    with localcontext() as ctx:
        ctx.prec = 60
        v = Decimal(value.numerator) / Decimal(value.denominator)
        if kind == "roots":
            return v.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        ctx.prec = 12
        ctx.rounding = ROUND_HALF_UP
        return +v


def factors(n):
    # Independent trial division, not Reasoning Gym's factorizer or answer field.
    out = []
    for divisor in range(2, math.isqrt(n) + 1):
        while n % divisor == 0:
            out.append(F(divisor))
            n //= divisor
    if n > 1:
        out.append(F(n))
    return out


def statement(formula):
    q, a, b = formula
    return {"all": f"All {a} are {b}", "no": f"No {a} are {b}",
            "some": f"Some {a} are {b}", "some_not": f"Some {a} are not {b}"}[q]


def holds(formula, model):
    q, a, b = formula
    a, b = ord(a) - ord("A"), ord(b) - ord("A")
    members = [t for t in range(8) if model & (1 << t) and t & (1 << a)]
    in_b = [bool(t & (1 << b)) for t in members]
    return {"all": all(in_b), "no": not any(in_b),
            "some": any(in_b), "some_not": any(not x for x in in_b)}[q]


@lru_cache(maxsize=None)
def models(premises):
    # One representative of each occupied membership type suffices for categorical
    # all/no/some/some-not formulas, so these 255 models are complete, not a bound.
    return tuple(m for m in range(1, 256) if all(holds(p, m) for p in premises))


def entails(premises, conclusion):
    possible = models(premises)
    assert possible, "Unsatisfiable fixtures are not used"
    return all(holds(conclusion, m) for m in possible)


@dataclass
class Problem:
    endpoint: str
    index: int
    kind: str
    values: list
    question: str
    metadata: dict = field(default_factory=dict)
    premises: tuple = ()
    conclusion: tuple = ()

    def render(self, tokens):
        if self.kind == "ordered":
            return json.dumps(tokens)
        if self.kind == "factors":
            return " × ".join(tokens)
        if self.kind == "roots":
            return ", ".join(tokens)
        return tokens[0] if tokens else ""

    def correct(self, tokens):
        if self.kind == "logic":
            return len(tokens) == 1 and tokens[0].strip().lower() == ("yes" if self.values[0] else "no")
        values = [numeric(t) for t in tokens]
        if self.kind == "roots":
            return set(rounded(v, "roots") for v in values) == set(rounded(v, "roots") for v in self.values)
        if self.kind == "factors":
            return sorted(values) == sorted(self.values)
        if self.kind == "decimal":
            return len(values) == 1 and rounded(values[0], "decimal") == rounded(self.values[0], "decimal")
        return values == self.values

    @property
    def tokens(self):
        if self.kind == "logic":
            return ["Yes" if self.values[0] else "No"]
        return [decimal_text(v) for v in self.values]

    @property
    def entry(self):
        answer = self.render(self.tokens)
        if self.endpoint == "gsm8k":
            answer = "#### " + answer
        return {"question": self.question, "answer": answer,
                "metadata": {"source_dataset": self.endpoint, **self.metadata}}


def problems(endpoint, n=64):
    for i in range(n):
        rng = random.Random(901_016 + i)
        a, b = rng.randint(2, 100), rng.randint(2, 100)
        kind, metadata, premises, conclusion = "scalar", {}, (), ()
        if endpoint in ("basic_arithmetic", "math_verify", "gsm8k"):
            c = rng.randint(-20, 20)
            values, question = [F(a * b + c)], f"Compute {a} * {b} + ({c})."
        elif endpoint == "decimal_arithmetic":
            x, y = F(rng.randint(10, 9999), 1000), F(rng.randint(10, 9999), 1000)
            kind, values = "decimal", [x + y]
            question = f"Compute {decimal_text(x)} + {decimal_text(y)} to at most 12 significant digits, half up."
        elif endpoint in ("gcd", "lcm"):
            result = math.gcd(a, b) if endpoint == "gcd" else a * b // math.gcd(a, b)
            values, question = [F(result)], f"Find the {endpoint} of {a} and {b}."
            metadata = {"numbers": [a, b]}
        elif endpoint == "prime_factorization":
            n_value = rng.randint(4, 1000)
            kind, values = "factors", factors(n_value)
            question = f"Give the prime factorization of {n_value}, separated by ×."
            metadata = {"number": n_value}
        elif endpoint == "number_sorting":
            original = [F(rng.randint(-10000, 10000), 100) for _ in range(4)]
            kind, values = "ordered", sorted(original)
            question = f"Sort {', '.join(map(decimal_text, original))} ascending, as a list of numeric strings."
            metadata = {"direction": "ascending", "original_numbers": list(map(decimal_text, original))}
        elif endpoint == "polynomial_equations":
            kind = "roots"
            values = [] if i % 8 == 0 else list(map(F, sorted(rng.sample(range(-9, 10), 2 + i % 2))))
            expression = "x*x+1" if not values else "*".join(f"(x-({v}))" for v in values)
            question = f"Find all real roots of {expression}=0, comma separated, to four decimal places; empty if none."
        elif endpoint == "syllogism":
            kind = "logic"
            forms = [(q, x, y) for q in ("all", "no", "some", "some_not")
                     for x in "ABC" for y in "ABC" if x != y]
            while True:
                premises = tuple(rng.sample(forms, 2))
                if models(premises):
                    break
            conclusion = rng.choice(forms)
            values = [entails(premises, conclusion)]
            question = "; ".join(map(statement, premises)) + ". Does " + statement(conclusion) + " follow? Answer Yes or No."
        else:
            raise ValueError(endpoint)
        yield Problem(endpoint, i, kind, values, question, metadata, premises, conclusion)


def mutations(p, split):
    selected = DEV if split == "dev" else HELD
    tokens, kind = p.tokens, p.kind

    def item(family, variant, ts, text=None, presentation="native", complete=True, claims=True):
        valid = bool(complete and claims and p.correct(ts))
        text = p.render(ts) if text is None else text
        if p.endpoint == "gsm8k":
            text = "#### " + text
        return {"endpoint": p.endpoint, "problem": p.index, "family": family,
                "variant": variant, "answer": text, "valid": valid,
                "presentation": presentation, "complete": complete,
                "claims_valid": bool(claims), "reference_tokens": ts,
                "reference_values": [str(v) for v in p.values], "question": p.question}

    yield item("canonical", "original", tokens)
    for family in selected:
        if family == "whitespace":
            for before, after in ((" ", " "), ("\n", "\n"), ("\t", "")):
                yield item(family, repr((before, after)), tokens, before + p.render(tokens) + after)
        elif family == "quantifier" and kind == "logic":
            q, a, b = p.conclusion
            for flipped in ("all", "no", "some", "some_not"):
                if flipped != q:
                    assertion = (flipped, a, b)
                    yield item(family, flipped, tokens, p.render(tokens) + "; " + statement(assertion),
                               "extra_content", claims=entails(p.premises, assertion))
        elif family == "sign" and kind == "logic":
            yield item(family, "polarity", ["No" if tokens[0] == "Yes" else "Yes"])
        elif family == "partial":
            if kind in ("ordered", "roots", "factors") and tokens:
                yield item(family, "omit_last", tokens[:-1])
            elif kind in ("scalar", "decimal"):
                shortened = tokens[0][:-1]
                yield item(family, "truncate", [shortened] if shortened not in ("", "-") else [], complete=bool(shortened))
        elif family == "candidates":
            alternative = "No" if kind == "logic" and tokens[0] == "Yes" else "Yes"
            if kind != "logic":
                alternative = decimal_text((p.values[0] if p.values else F(0)) + 1)
            for reverse in (False, True):
                parts = [p.render(tokens), alternative]
                if reverse:
                    parts.reverse()
                yield item(family, str(reverse), tokens, " or ".join(parts), "extra_content", complete=False)
        elif family == "duplicates" and kind in ("ordered", "roots", "factors") and tokens:
            yield item(family, "repeat_first", [tokens[0]] + tokens)
        elif family == "trailing":
            yield item(family, "contradiction", tokens, p.render(tokens) + "; also 0 = 1", "extra_content", claims=F(0) == F(1))
            yield item(family, "neutral", tokens, p.render(tokens) + "; exact.", "extra_content")
        elif family == "delimiters":
            if kind == "logic":
                yield item(family, "lowercase", [tokens[0].lower()], presentation="alternate")
            elif kind == "ordered":
                yield item(family, "unquoted", tokens, "[" + ", ".join(tokens) + "]", "alternate")
            elif kind == "factors":
                yield item(family, "asterisk", tokens, " * ".join(tokens), "alternate")
            elif kind == "roots":
                yield item(family, "semicolon", tokens, "; ".join(tokens), "alternate")
            else:
                yield item(family, "parentheses", [f"({tokens[0]})"], presentation="alternate")
        elif kind != "logic" and tokens:
            vals = list(map(numeric, tokens))
            if family == "boundary":
                for delta in (-1, 1):
                    yield item(family, str(delta), [decimal_text(v + delta) for v in vals])
            elif family == "sign":
                yield item(family, "negate", [decimal_text(-v) for v in vals])
            elif family == "rational":
                yield item(family, "divide_two", [f"{v.numerator * 2}/{v.denominator * 2}" for v in vals], presentation="alternate")
            elif family == "units" and kind in ("scalar", "decimal"):
                yield item(family, "percent_changed", [tokens[0] + "%"], presentation="alternate")
                yield item(family, "percent_equivalent", [decimal_text(vals[0] * 100) + "%"], presentation="alternate")
            elif family == "latex" and kind in ("scalar", "decimal"):
                for multiplier in (1, 2):
                    v = vals[0]
                    text = rf"\boxed{{\frac{{{v.numerator * multiplier}}}{{{v.denominator * 2}}}}}"
                    yield item(family, str(multiplier), [text], presentation="alternate")
            elif family == "tolerance":
                for delta in map(F, ("0.000000000001", "0.00000001", "0.0001", "0.01", "0.5", "1", "2")):
                    yield item(family, str(delta), [decimal_text(v + delta) for v in vals])
            elif family == "scientific":
                yield item(family, "e_zero", [t + "e0" for t in tokens], presentation="alternate")
                yield item(family, "scaled", [decimal_text(v / 10) + "e1" for v in vals], presentation="alternate")
