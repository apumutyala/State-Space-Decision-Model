"""Fail-closed v0/capability union and group-disjoint synthetic selection data."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random

from keel.distill.sources import ContaminationError, digest, guard_units, read_rows, write_rows
from keel.distill.teacher import argmax
from keel.protocol import DecisionRequest, render

CANDIDATES = [
    {"name": "r16-e1-balanced", "rank": 16, "epochs": 1, "capability_fraction": 0.5},
    {"name": "r32-e1-balanced", "rank": 32, "epochs": 1, "capability_fraction": 0.5},
    {"name": "r16-e2-capability", "rank": 16, "epochs": 2, "capability_fraction": 2 / 3},
]


def validate_source(row: dict) -> None:
    source = row.get("source", {})
    if source.get("source_split") not in {"synthetic", "train"}:
        raise ContaminationError("training row lacks an allowed source split")
    if row.get("eval_only") or source.get("eval_only") or row.get("verification", {}).get("eval_only"):
        raise ContaminationError("evaluation-only row in training mix")
    if not all(source.get(k) for k in ("source", "source_id", "content_sha256")):
        raise ContaminationError("training provenance incomplete")
    if any(x in source["source"].casefold() for x in ("jevbench", "public-231", "eval-only")):
        raise ContaminationError("public benchmark provenance in training mix")
    if not row.get("group") or not row.get("id"):
        raise ValueError("missing item/group identity")
    req = DecisionRequest.from_json(row["request"])
    if set(row["teacher_probs"]) != set(render(req).option_ids) or row["gold"] not in row["teacher_probs"]:
        raise ValueError("label options disagree with rendered request")
    argmax(row["teacher_probs"])
    if row["kind"] == "capability":
        v = row.get("verification", {})
        if v.get("qwen_samples") != 2 or v.get("cross_family") != "openai/gpt-oss-20b" or not v.get("verdict_sha256"):
            raise ValueError("capability row lacks independent verified labels")


def split_capability(rows: list[dict], dev_groups: int = 160) -> tuple[list[dict], list[dict]]:
    groups = sorted({r["group"] for r in rows}, key=lambda g: digest(["capability-dev-v1", g]))
    if not 0 < dev_groups < len(groups):
        raise ValueError("synthetic dev must leave nonempty training groups")
    heldout = set(groups[:dev_groups])
    return [r for r in rows if r["group"] not in heldout], [r for r in rows if r["group"] in heldout]


def assert_disjoint(train: list[dict], dev: list[dict]) -> None:
    for key in (lambda r: r["id"], lambda r: r["group"], lambda r: r["source"]["content_sha256"]):
        if {key(r) for r in train} & {key(r) for r in dev}:
            raise ValueError("train/dev item, group or content overlap")


def assemble(v0: list[dict], capability: list[dict], v0_dev: list[dict], *, dev_groups=160) -> tuple[list[dict], list[dict], dict]:
    for r in v0 + capability + v0_dev:
        validate_source(r)
    if any(r["kind"] != "capability" for r in capability):
        raise ValueError("new labels must be capability rows")
    cap_train, cap_dev = split_capability(capability, dev_groups)
    train, dev = v0 + cap_train, v0_dev + cap_dev
    if len({r["id"] for r in train + dev}) != len(train + dev):
        raise ValueError("duplicate row identities")
    assert_disjoint(train, dev)
    # A pair cannot be separated by filtering or by the source-group split.
    for pool in (cap_train, cap_dev):
        pairs = collections.defaultdict(list)
        for r in pool:
            if r.get("pair"):
                pairs[r["group"]].append(r)
        if any(len(p) != 2 or {r["gold"] for r in p} != {"true", "false"} for p in pairs.values()):
            raise ValueError("capability noul balance/pair invariant broken")
    report = {"v0_rows": len(v0), "capability_train": len(cap_train), "capability_dev": len(cap_dev),
              "v0_dev": len(v0_dev), "train": len(train), "dev": len(dev),
              "train_sha256": digest(train), "dev_sha256": digest(dev), "public_rows": 0,
              "train_kinds": dict(collections.Counter(r["kind"] for r in train)),
              "dev_selection": "label-blind source-group SHA256 before student scoring",
              "candidates": CANDIDATES}
    return train, dev, report


def mix(rows: list[dict], capability_fraction: float, seed=20261015, limit: int | None = None) -> list[dict]:
    """Keep the entire union; oversample the underweight source, never discard v0."""
    if not 0 < capability_fraction < 1:
        raise ValueError("mix fraction must be interior")
    cap = [r for r in rows if r["kind"] == "capability"]
    old = [r for r in rows if r["kind"] != "capability"]
    if not cap or not old:
        raise ValueError("both training sources are required")
    rng = random.Random(seed)
    if limit:
        nc = round(limit * capability_fraction)
        no = limit - nc
    else:
        total = max(len(cap) / capability_fraction, len(old) / (1 - capability_fraction))
        nc, no = round(total * capability_fraction), round(total * (1 - capability_fraction))
    def draw(pool, count):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return [shuffled[i % len(shuffled)] for i in range(count)]
    result = draw(cap, nc) + draw(old, no)
    rng.shuffle(result)
    return result


def select_candidate(results: list[dict]) -> dict:
    """Balanced-source dev loss is primary; accuracy then declared order break ties."""
    if {r["name"] for r in results} != {c["name"] for c in CANDIDATES}:
        raise ValueError("all registered candidates must complete")
    import math
    if any(not math.isfinite(r["dev"]["balanced_loss"]) for r in results):
        raise ValueError("nonfinite synthetic dev objective")
    order = {c["name"]: i for i, c in enumerate(CANDIDATES)}
    return min(results, key=lambda r: (r["dev"]["balanced_loss"], -r["dev"]["balanced_accuracy"], order[r["name"]]))


def main():
    from keel.paths import private_root
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(private_root() / "distill_v1"))
    p.add_argument("--dev-groups", type=int, default=160)
    a = p.parse_args()
    out = Path(a.out)
    v0_root = private_root() / "distill_v0"
    v0 = read_rows(v0_root / "teacher-train-screened/labels.jsonl.gz")
    from keel.distill.evidence_screen import require_screened_evidence
    require_screened_evidence(v0)
    if len(v0) != 8105:
        raise ValueError("expected the frozen 8,105-row v0 union")
    v0_dev = read_rows(v0_root / "teacher-pilot-screened/labels.jsonl.gz")
    train, dev, report = assemble(v0, read_rows(out / "capability-labels.jsonl.gz"), v0_dev, dev_groups=a.dev_groups)
    report["train_guards"] = guard_units(train, out / "train-guard-input.jsonl.gz")
    report["dev_guards"] = guard_units(dev, out / "dev-guard-input.jsonl.gz")
    write_rows(out / "train.jsonl.gz", train)
    write_rows(out / "dev.jsonl.gz", dev)
    (out / "assembly.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("guards")}, indent=2))


if __name__ == "__main__":
    main()
