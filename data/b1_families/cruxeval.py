"""CRUXEval-output-choice (catalog 43, heldout-draw-v1 GENERATABLE): SYNTHETIC generator, outputs by execution.

CRUXEval publishes only its 800-problem benchmark (no train split), so programs are generated here: a typed
step grammar composes 2-5 statements over the arguments (str / list[int] / list[str] / dict[str,int] / int), the
program is executed on a generated input, and the gold option is ``repr`` of the returned value. Every distractor is
(a) the output of a one-constant mutation of the same program on the same input, or (b) a type-preserving
perturbation of the gold value; each is checked distinct from the gold by repr AND by value (``==``) and the gold
is re-derived by a second, independent execution before the record is kept.

Eval presentation copied from the Index rows: ``state = {code, input, task: "CRUXEval output selection"}``, one
choice question ``answer``, ids ``option_0..``, descriptions are Python literals, the eval instruction verbatim.
Option count follows the eval mix (2-10); gold position uniform.

Execution: generated code only (never downloaded code), in a namespace with a small builtins whitelist; the
grammar has no unbounded loop (every loop iterates a finite sequence or a bounded counter) and outputs over 160
characters are discarded.
"""
from __future__ import annotations

import ast
import copy
import random
import re
import string

from .common import make_record, shuffled_options

FAMILY = "CRUXEval-output-choice"
SOURCE = "b1fam_cruxeval_synth"
LICENSE = "generated (repo code; no third-party text)"
TASK = "CRUXEval output selection"
INSTR = "Choose the correct output of f called with the supplied input arguments. Candidates are Python literal values."
# eval option counts: 2:256 3:139 4:84 5:36 6:31 7:10 8:8 9:1 10:5
N_WEIGHTS = {2: 256, 3: 139, 4: 84, 5: 36, 6: 31, 7: 10, 8: 8, 9: 1, 10: 5}
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ("len", "sorted", "set", "list", "dict", "sum", "max", "min", "abs", "range", "enumerate",
                           "reversed", "str", "int", "tuple", "zip", "any", "all", "isinstance", "bool")}
MAX_REPR = 160

WORDS = ["apple", "river", "stone", "cloud", "mint", "lamp", "zebra", "ocean", "pixel", "tiger", "maple", "quartz",
         "ember", "delta", "orbit", "cedar", "fjord", "nylon", "basil", "comet", "gamma", "haze", "ivory", "jolt"]
ALPH = string.ascii_lowercase + string.ascii_uppercase + string.digits + " -_.,!?#"


# ------------------------------------------------------------------------------------------- inputs
def rand_text(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.4:
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 4)))
    if r < 0.8:
        return "".join(rng.choice(ALPH) for _ in range(rng.randint(0, 14)))
    w = rng.choice(WORDS)
    return "".join(c.upper() if rng.random() < 0.3 else c for c in w) + rng.choice(["", "123", "!!", "  ", "-x"])


def rand_nums(rng):
    return [rng.randint(-9, 20) for _ in range(rng.randint(0, 8))]


def rand_strs(rng):
    return [rng.choice(WORDS + ["a", "bb", "", "xy", "Q", "ok"]) for _ in range(rng.randint(0, 6))]


def rand_dict(rng):
    ks = rng.sample(WORDS + list("abcxyz"), 0 if rng.random() < 0.08 else rng.randint(1, 5))
    return {k: rng.randint(-5, 15) for k in ks}


SIGS = [  # (params with types, input maker)
    ([("text", "str")], lambda r: [rand_text(r)]),
    ([("text", "str"), ("ch", "chr")], lambda r: [rand_text(r), r.choice(list("aeiost -x1"))]),
    ([("text", "str"), ("n", "int")], lambda r: [rand_text(r), r.randint(-2, 6)]),
    ([("nums", "list_int")], lambda r: [rand_nums(r)]),
    ([("nums", "list_int"), ("k", "int")], lambda r: [rand_nums(r), r.randint(-3, 7)]),
    ([("strings", "list_str")], lambda r: [rand_strs(r)]),
    ([("d", "dict")], lambda r: [rand_dict(r)]),
    ([("d", "dict"), ("key", "chr")], lambda r: [rand_dict(r), r.choice(WORDS[:8] + list("abx"))]),
]


# ------------------------------------------------------------------------------------------- grammar
class Prog:
    def __init__(self, rng):
        self.rng, self.lines, self.vars, self.k = rng, [], {}, 0

    def fresh(self, base):
        self.k += 1
        return f"{base}{self.k}" if self.k > 1 and base in self.vars else base

    def of(self, *types):
        vs = [v for v, t in self.vars.items() if t in types]
        return self.rng.choice(vs) if vs else None


def _c(rng):  # a short str constant
    return rng.choice(["a", "e", "o", "x", " ", "-", "ab", "an", "1", "!", "st", "_"])


def step(p: Prog) -> bool:
    r = p.rng
    cands = []
    s = p.of("str")
    if s:
        cands += [
            lambda: p.lines.append(f"{s} = {s}.replace({_c(r)!r}, {_c(r)!r})"),
            lambda: p.lines.append(f"{s} = {s}.{r.choice(['upper', 'lower', 'swapcase', 'title', 'capitalize', 'strip'])}()"),
            lambda: p.lines.append(f"{s} = {s}[::-1]"),
            lambda: p.lines.append(f"{s} = {s}[{r.randint(1, 4)}:] + {s}[:{r.randint(1, 4)}]"),
            lambda: p.lines.append(f"{s} = {s}.{r.choice(['center', 'ljust', 'rjust'])}({r.randint(4, 14)}, {r.choice('*.#-')!r})"),
            lambda: p.lines.append(f"{s} = {r.choice(['-', '', ' ', '_', ', '])!r}.join({s}.split())"),
            lambda: p.lines.append(f"{s} = {s}.{r.choice(['lstrip', 'rstrip', 'strip'])}({r.choice(['a', 'x ', '!', '0123456789', '-_'])!r})"),
            lambda: _loop_chars(p, s),
        ]
        cands += [lambda: _new(p, "count", "int", f"{s}.count({_c(r)!r})"),
                  lambda: _new(p, "idx", "int", f"{s}.find({_c(r)!r})"),
                  lambda: _new(p, "parts", "list_str", f"{s}.split({r.choice([' ', 'a', '-', 'e'])!r})"),
                  lambda: _new(p, "chars", "list_str", f"sorted(set({s}))"),
                  lambda: _freq(p, s),
                  lambda: _new(p, "flag", "bool", f"{s}.{r.choice(['startswith', 'endswith'])}({_c(r)!r})"),
                  lambda: _new(p, "flag", "bool", f"{s}.{r.choice(['isalpha', 'isdigit', 'islower', 'isupper', 'isspace'])}()")]
    ch = p.of("chr")
    if ch and s:
        cands += [lambda: p.lines.append(f"{s} = {s}.replace({ch}, {r.choice(['', '*', ch + ' * 2', ch + '.upper()'])})"),
                  lambda: _new(p, "pos", "int", f"{s}.rfind({ch})"),
                  lambda: _new(p, "pieces", "list_str", f"{s}.split({ch})")]
    li = p.of("list_int")
    if li:
        x = r.randint(-5, 12)
        cands += [
            lambda: p.lines.append(f"{li}.append({x})"),
            lambda: p.lines.append(f"{li}.insert({r.randint(-2, 3)}, {x})"),
            lambda: p.lines.append(f"{li} = [n {r.choice(['*', '+', '-', '%'])} {r.randint(2, 4)} for n in {li}]"),
            lambda: p.lines.append(f"{li} = [n for n in {li} if n % {r.randint(2, 3)} == {r.randint(0, 1)}]"),
            lambda: p.lines.append(f"{li} = [n for n in {li} if n {r.choice(['>', '<', '>=', '!='])} {r.randint(-1, 8)}]"),
            lambda: p.lines.append(f"{li}.sort({r.choice(['', 'reverse=True'])})"),
            lambda: p.lines.append(f"{li}.reverse()"),
            lambda: p.lines.append(f"{li} = sorted(set({li}))"),
            lambda: p.lines.append(f"{li} = {li}[{r.randint(0, 2)}:{r.randint(3, 6)}]"),
            lambda: p.lines.append(f"{li} = {li}[::{r.choice([2, -1, -2])}]"),
            lambda: p.lines.extend([f"if {x} in {li}:", f"    {li}.remove({x})"]),
            lambda: p.lines.extend([f"while len({li}) > {r.randint(1, 5)}:", f"    {li}.pop({r.choice(['', '0'])})"]),
            lambda: p.lines.extend([f"for i in range(len({li})):", f"    if i % 2 == {r.randint(0, 1)}:",
                                    f"        {li}[i] = {r.choice(['-' + li + '[i]', li + '[i] * 2', li + '[i] + i'])}"]),
            lambda: _new(p, "total", "int", f"sum({li})"),
            lambda: _new(p, "top", "int", f"max({li}) if {li} else {r.randint(-1, 0)}"),
            lambda: _new(p, "size", "int", f"len({li})"),
            lambda: _new(p, "pairs", "list_tuple", f"list(enumerate({li}))[{r.randint(0, 2)}:]"),
        ]
    lk = p.of("int")
    if lk and li:
        cands += [lambda: p.lines.append(f"{li} = [n * {lk} for n in {li}]"),
                  lambda: p.lines.append(f"{li} = {li}[{lk}:] + {li}[:{lk}]"),
                  lambda: p.lines.append(f"{li}.append({lk})")]
    if lk and s:
        cands += [lambda: p.lines.append(f"{s} = {s}[:{lk}] + {s}[{lk}:].upper()"),
                  lambda: p.lines.append(f"{s} = {s} * abs({lk}) if abs({lk}) < 4 else {s}")]
    if lk:
        cands += [lambda: p.lines.append(f"{lk} = {lk} * {r.randint(2, 5)} {r.choice(['+', '-'])} {r.randint(0, 9)}"),
                  lambda: p.lines.append(f"{lk} = {lk} % {r.randint(2, 7)}"),
                  lambda: p.lines.append(f"{lk} = {lk} // {r.randint(2, 4)}"),
                  lambda: p.lines.append(f"{lk} = abs({lk}) + {r.randint(0, 3)}")]
    ls = p.of("list_str")
    if ls:
        cands += [lambda: p.lines.append(f"{ls} = [w.{r.choice(['upper', 'title', 'strip'])}() for w in {ls}]"),
                  lambda: p.lines.append(f"{ls} = [w for w in {ls} if len(w) {r.choice(['>', '<=', '=='])} {r.randint(1, 4)}]"),
                  lambda: p.lines.append(f"{ls}.sort({r.choice(['', 'reverse=True', 'key=len'])})"),
                  lambda: p.lines.append(f"{ls} = [w[::-1] for w in {ls}]"),
                  lambda: _new(p, "joined", "str", f"{r.choice(['-', '', ' ', '|'])!r}.join({ls})"),
                  lambda: _new(p, "lengths", "list_int", f"[len(w) for w in {ls}]"),
                  lambda: _new(p, "n", "int", f"len({ls})")]
    dv = p.of("dict")
    if dv:
        key = p.of("chr") if p.vars.get("key") == "chr" else None
        kk = key or repr(r.choice(WORDS[:6] + list("abx")))
        cands += [lambda: p.lines.append(f"{dv}[{kk}] = {r.randint(-3, 9)}"),
                  lambda: p.lines.append(f"{dv}.pop({kk}, None)"),
                  lambda: p.lines.append(f"{dv} = dict(sorted({dv}.items()))"),
                  lambda: p.lines.append(f"{dv} = {{k: v {r.choice(['*', '+', '-'])} {r.randint(1, 3)} for k, v in {dv}.items()}}"),
                  lambda: p.lines.append(f"{dv} = {{k: v for k, v in {dv}.items() if v {r.choice(['>', '<', '!='])} {r.randint(0, 6)}}}"),
                  lambda: _new(p, "items", "list_tuple", f"sorted({dv}.items(), key=lambda kv: {r.choice(['kv[1]', 'kv[0]', '-kv[1]'])})"),
                  lambda: _new(p, "keys", "list_str", f"list({dv})"),
                  lambda: _new(p, "total", "int", f"sum({dv}.values())"),
                  lambda: _new(p, "found", "bool", f"{kk} in {dv}")]
    if not cands:
        return False
    r.choice(cands)()
    return True


def _new(p: Prog, base, typ, expr):
    v = base if base not in p.vars else f"{base}{len(p.vars)}"
    p.lines.append(f"{v} = {expr}")
    p.vars[v] = typ


def _loop_chars(p: Prog, s):
    r = p.rng
    pred = r.choice(["isalpha", "isdigit", "isupper", "islower", "isspace"])
    op = r.choice(["upper()", "lower()", "swapcase()", "* 2", ""])
    other = r.choice(["''", "'*'", "ch", "'_'"])
    body = f"ch.{op}" if op.endswith(")") else (f"ch {op}" if op else "ch")
    p.lines += ["result = ''", f"for ch in {s}:", f"    if ch.{pred}():", f"        result += {body}",
                "    else:", f"        result += {other}", f"{s} = result"]


def _freq(p: Prog, s):
    v = "freq" if "freq" not in p.vars else f"freq{len(p.vars)}"
    p.lines += [f"{v} = {{}}", f"for ch in {s}:", f"    {v}[ch] = {v}.get(ch, 0) + 1"]
    p.vars[v] = "dict"


def make_program(rng: random.Random) -> tuple[str, list]:
    params, mk = rng.choice(SIGS)
    p = Prog(rng)
    for name, typ in params:
        p.vars[name] = typ
    for _ in range(rng.randint(2, 5)):
        step(p)
    vs = [v for v, t in p.vars.items() if t != "chr"]
    r = rng.random()
    last = re.match(r"^(\w+)", next(l for l in reversed(p.lines) if not l.startswith(" "))).group(1)
    main = last if last in p.vars and p.vars[last] != "chr" else rng.choice(vs)
    if r < 0.72:
        ret = [f"return {main}"]
    elif r < 0.9 and len(vs) > 1:
        other = rng.choice([v for v in vs if v != main])
        ret = [f"return ({main}, {other})" if rng.random() < 0.6 else f"return [{main}, {other}]"]
    else:
        ints = [v for v, t in p.vars.items() if t in ("int", "list_int", "str", "list_str")]
        cv = rng.choice(ints) if ints else main
        cond = f"len({cv}) > {rng.randint(1, 6)}" if p.vars.get(cv) != "int" else f"{cv} > {rng.randint(-1, 6)}"
        alt = rng.choice([v for v in vs if v != main] or [main])
        ret = [f"if {cond}:", f"    return {main}", f"return {alt}"]
    sig = ", ".join(n for n, _ in params)
    code = "\n".join([f"def f({sig}):"] + ["    " + l for l in p.lines + ret])
    return code, mk(rng)


def run(code: str, args: list):
    ns: dict = {"__builtins__": SAFE_BUILTINS}
    exec(compile(code, "<gen>", "exec"), ns)  # generated by make_program above, never external code
    return ns["f"](*copy.deepcopy(args))


def literal_ok(v) -> bool:
    s = repr(v)
    if len(s) > MAX_REPR:
        return False
    try:
        return ast.literal_eval(s) == v
    except Exception:
        return False


_NUM = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def mutants(code: str, rng: random.Random, k: int = 12) -> list[str]:
    """One-constant mutations of the program body (an int literal +-1/+2, or a string literal swapped)."""
    head, body = code.split("\n", 1)
    out = []
    nums = list(_NUM.finditer(body))
    strs = list(re.finditer(r"'([^'\\]{0,3})'", body))
    for _ in range(k):
        if nums and (not strs or rng.random() < 0.6):
            m = rng.choice(nums)
            new = str(max(0, int(m.group(1)) + rng.choice([-2, -1, 1, 2])))
            out.append(head + "\n" + body[:m.start()] + new + body[m.end():])
        elif strs:
            m = rng.choice(strs)
            new = repr(rng.choice(["a", "e", "x", "", " ", "-", "b", "1"]))
            out.append(head + "\n" + body[:m.start()] + new + body[m.end():])
    return out


def perturb(v, rng: random.Random):
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, int):
        return [v + 1, v - 1, v * 2, -v, v + rng.randint(2, 5)]
    if isinstance(v, str):
        o = [v[::-1], v.upper(), v.lower(), v[:-1], v + v[-1:] if v else "a", v.swapcase(), v.strip(), v[1:]]
        return o + ([v.replace(v[0], "")] if v else [" "])
    if isinstance(v, list):
        o = [v[::-1], v[:-1], v[1:], sorted(v, key=repr), v + v[-1:]]
        if v and all(isinstance(x, int) for x in v):
            o += [[x + 1 for x in v], [abs(x) for x in v]]
        return o + ([[]] if v else [[0]])
    if isinstance(v, tuple):
        return [tuple(v[::-1])] + [tuple([p] + list(v[1:])) for p in perturb(v[0], rng)[:3]] + [list(v)]
    if isinstance(v, dict):
        ks = list(v)
        o = [dict(list(v.items())[:-1]), {k: x + 1 if isinstance(x, int) else x for k, x in v.items()}]
        if ks:
            o.append({**v, ks[0]: (v[ks[0]] + 1) if isinstance(v[ks[0]], int) else v[ks[0]]})
            o.append(dict(sorted(v.items(), key=lambda kv: repr(kv), reverse=True)))
        return o + [{}] if v else o + [{"a": 1}]
    return []


def item_probe(code: str, inp: str) -> dict:
    """Item tier for the contamination guard: the WHOLE function as one whitespace-collapsed segment (plus the input),
    so exact / minhash / 13-gram compare functions, not generic lines (``for ch in text:``) that every program shares;
    the edit/contain matchers still see it and drop a record containing any 6-30-token Index code segment."""
    return {"state": {"code": " ".join(code.split()), "input": inp}, "questions": {}}


def fmt_input(args: list) -> str:
    return ", ".join(repr(a) for a in args)


def generate(seed: int, target: int = 2400, log=print) -> list[dict]:
    ns, ws = zip(*N_WEIGHTS.items())
    out, seen_code = [], set()
    k, stats = 0, {"exec_error": 0, "bad_literal": 0, "too_few_distractors": 0, "dup": 0, "trivial": 0}
    while len(out) < target:
        rng = random.Random(f"{seed}|crux|{k}")
        k += 1
        if k > 100 * target:
            raise RuntimeError(f"cruxeval: only {len(out)} records after {k} draws")
        code, args = make_program(rng)
        if code in seen_code:
            stats["dup"] += 1
            continue
        try:
            gold = run(code, args)
            again = run(code, args)
        except Exception:
            stats["exec_error"] += 1
            continue
        if gold != again or not literal_ok(gold):
            stats["bad_literal"] += 1
            continue
        if gold == args[0] and len(args) == 1:
            stats["trivial"] += 1
            if rng.random() < 0.8:
                continue
        n_opts = rng.choices(ns, ws)[0]
        pool = []
        for m in mutants(code, rng):
            try:
                pool.append(run(m, args))
            except Exception:
                pass
        rng.shuffle(pool)
        pool += perturb(gold, rng)
        distract, reprs = [], {repr(gold)}
        for v in pool:
            try:
                same = (v == gold) or (type(v) is not type(gold) and not isinstance(gold, (tuple, list)))
            except Exception:
                same = True
            if same or not literal_ok(v) or repr(v) in reprs:
                continue
            reprs.add(repr(v))
            distract.append(repr(v))
            if len(distract) >= n_opts - 1:
                break
        if len(distract) < n_opts - 1:
            if len(distract) < 1:
                stats["too_few_distractors"] += 1
                continue
            n_opts = len(distract) + 1
        seen_code.add(code)
        opts, pos = shuffled_options(repr(gold), distract, rng, n_opts)
        if ast.literal_eval(opts[pos]) != run(code, args):  # independent re-derivation of the gold
            raise AssertionError(f"cruxeval {k}: gold re-derivation failed")
        criteria = {f"option_{i}": o for i, o in enumerate(opts)}
        state = {"code": code, "input": fmt_input(args), "task": TASK}
        q = {"answer": {"type": "choice", "instructions": INSTR, "criteria": criteria}}
        rec = make_record(family=FAMILY, source=SOURCE, source_id=f"{seed}-{k - 1}", source_split="synthetic",
                          provenance="synthetic", license=LICENSE, state=state, questions=q,
                          labels={"answer": f"option_{pos}"},
                          aug={"n_options": n_opts, "gold_pos": pos, "generator": "data.b1_families.cruxeval",
                               "verified": "executed twice + gold re-derived after shuffling"},
                          probe=item_probe(code, state["input"]))
        out.append(rec)
    log(f"  cruxeval: {len(out)} records from {k} draws; rejected {stats}")
    return out
