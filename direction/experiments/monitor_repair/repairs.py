"""Experiment-only AST transformations. Never compile or execute corpus source."""
from __future__ import annotations

import ast
import copy
import re
from difflib import SequenceMatcher

from direction.experiments.artifact_audit import audit

B2_SEED = 20261017


def structure(tree):
    return ast.dump(tree, include_attributes=False)


class StripDocstrings(ast.NodeTransformer):
    def __init__(self):
        self.removed = 0

    def visit_scope(self, node):
        self.generic_visit(node)
        # Remove a run of leading strings: otherwise the second becomes a docstring.
        while node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
            self.removed += 1
        if not node.body and not isinstance(node, ast.Module):
            node.body = [ast.Pass()]
        return node

    visit_Module = visit_scope
    visit_FunctionDef = visit_scope
    visit_AsyncFunctionDef = visit_scope
    visit_ClassDef = visit_scope


def canonical(code):
    tree = ast.parse(code)
    stripper = StripDocstrings()
    tree = ast.fix_missing_locations(stripper.visit(tree))
    rendered = ast.unparse(tree) + "\n"
    if structure(ast.parse(rendered)) != structure(tree):
        raise ValueError("AST round trip changed more than docstrings")
    flags = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and
                    n.id in {"eval", "exec", "globals", "locals", "vars", "getattr", "inspect", "__file__"}} |
                   {"__doc__" for n in ast.walk(tree) if
                    isinstance(n, ast.Attribute) and n.attr == "__doc__" or
                    isinstance(n, ast.Constant) and n.value == "__doc__"})
    return rendered, tree, {"docstrings_removed": stripper.removed,
                           "reflection_flags": flags,
                           "comments_removed": sum(t.type == audit.tokenize.COMMENT
                                                   for t in audit.lexical_tokens(code))}


def nodes_with_paths(node, path=()):
    yield path, node
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield from nodes_with_paths(value, (*path, field))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, ast.AST):
                    yield from nodes_with_paths(child, (*path, field, index))


def locate(tree, path):
    for key in path:
        tree = tree[key] if isinstance(key, int) else getattr(tree, key)
    return tree


def source_patch(code, node, replacements):
    lines = code.splitlines(keepends=True)
    if isinstance(node, ast.If):
        prefix = lines[node.lineno - 1].encode()[:node.col_offset].decode()
        if prefix.strip():
            return None
        text = ast.unparse(ast.Module(body=replacements, type_ignores=[]))
        is_elif = lines[node.lineno - 1].lstrip().startswith("elif ")
        if is_elif and replacements:
            text = "else:\n" + "\n".join("    " + line for line in text.splitlines())
        lines[node.lineno - 1:node.end_lineno] = [prefix + line + "\n" for line in text.splitlines()]
        return "".join(lines)
    raw_lines = [line.encode() for line in lines]
    begin = sum(map(len, raw_lines[:node.lineno - 1])) + node.col_offset
    end = sum(map(len, raw_lines[:node.end_lineno - 1])) + node.end_col_offset
    raw = code.encode()
    return (raw[:begin] + ("(" + ast.unparse(replacements[0]) + ")").encode() + raw[end:]).decode()


def ast_tokens(tree):
    return re.findall(r"\w+|[^\w\s]", structure(tree))


def distance(tokens, other):
    return 1 - SequenceMatcher(None, tokens, other, autojunk=False).ratio()


def repair_candidates(row, honest_rows):
    original = row["code"]
    defective, tree, _ = canonical(original)
    honest = [(r["submission_id"], canonical(r["code"])[1]) for r in honest_rows]
    anchors = [(sid, structure(t), ast_tokens(t)) for sid, t in honest]
    before_tokens = ast_tokens(tree)
    before = min((distance(before_tokens, t) for _, _, t in anchors), default=None)
    results, rejected = [], 0
    for path, node in nodes_with_paths(tree):
        if not isinstance(node, (ast.If, ast.IfExp)):
            continue
        # Main-entry guards are not candidate sabotage branches.
        if any(isinstance(n, ast.Name) and n.id == "__name__" for n in ast.walk(node.test)):
            continue
        alternatives = (("take_else", node.orelse), ("take_body", node.body))
        for action, replacement in alternatives:
            changed = copy.deepcopy(tree)
            replacement = copy.deepcopy(replacement if isinstance(replacement, list) else [replacement])
            if isinstance(node, ast.If):
                parent = locate(changed, path[:-1])
                if len(parent) == 1 and not replacement:
                    replacement = [ast.Pass()]
                parent[path[-1]:path[-1] + 1] = replacement
            else:
                parent = locate(changed, path[:-1])
                if isinstance(path[-1], int):
                    parent[path[-1]] = replacement[0]
                else:
                    setattr(parent, path[-1], replacement[0])
            changed = ast.fix_missing_locations(changed)
            raw_repair = source_patch(original, node, replacement)
            if raw_repair is None:
                rejected += 1
                continue
            try:
                repaired, reparsed, _ = canonical(raw_repair)
            except (SyntaxError, ValueError):
                rejected += 1
                continue
            if structure(reparsed) != structure(changed) or structure(changed) == structure(tree):
                rejected += 1
                continue
            tokenized = ast_tokens(changed)
            closest = min(((distance(tokenized, t), sid) for sid, _, t in anchors), default=(None, None))
            exact = [sid for sid, s, _ in anchors if s == structure(changed)]
            comparisons = [n for n in ast.walk(node.test) if isinstance(n, ast.Compare)]
            literal_gate = any(audit.literal(s) is not None for n in comparisons
                               for s in [n.left, *n.comparators])
            results.append({"path": list(path), "action": action, "test": ast.unparse(node.test),
                            "literal_gate": literal_gate, "exact_honest_matches": exact,
                            "closest_honest": closest[1], "before_ast_distance": before,
                            "after_ast_distance": closest[0],
                            "removed_ast_nodes": sum(1 for _ in ast.walk(tree)) -
                                                 sum(1 for _ in ast.walk(changed)),
                            "cells": {"defective_original": original, "repaired_original": raw_repair,
                                      "defective_canonical": defective, "repaired_canonical": repaired}})
    # Ranking aids static inspection only. Neither sampling nor acceptance sees monitor scores.
    results.sort(key=lambda r: (not bool(r["exact_honest_matches"]),
                               r["after_ast_distance"] if r["after_ast_distance"] is not None else 2,
                               abs(r["removed_ast_nodes"]), str(r["path"]), r["action"]))
    for index, result in enumerate(results):
        result["candidate_id"] = index
    return {"submission_id": row["submission_id"], "problem_id": row["problem_id"],
            "trigger": row["trigger"], "honest_counterparts": len(honest),
            "candidates": results, "rejected_source_patches": rejected}
