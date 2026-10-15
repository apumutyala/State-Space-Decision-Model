"""Rebuild a small train-only draw with existing jev1 synthetic family builders."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path

from train import attack_guard as AG
from train import jevbench_guard as JG


class ContaminationError(ValueError):
    """A training candidate overlaps the pinned evaluation corpus."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_rows(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: str | Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_public_disjoint(row: dict, fp=None) -> None:
    """All five production fingerprint matchers and the short exact-segment gate."""
    if fp is None:
        pair = JG.load()
        if not pair:
            raise ContaminationError("JevBench guard is disabled")
        _, fp = pair
    hits = {k: v for k, v in JG.fingerprint_item_hits(fp, row["state"], row["questions"]).items() if v}
    if hits or JG.exact_segment_hits(fp, row["state"], row["questions"]):
        raise ContaminationError(f"{row['id']}: JevBench overlap: {sorted(hits)}")


def guard_draw(path: str | Path) -> dict:
    """Also apply the draw-level document rule; an unavailable guard is an error."""
    rep = JG.rescan([path], full_fingerprint=True, min_free_mb=0, log=lambda *_: None)
    draw = rep["draws"][str(Path(path))]
    if draw["jevbench_training_gate_fail"] or draw["jevbench_fingerprint_rows_hit"]:
        raise ContaminationError("JevBench document/fingerprint overlap in completed draw")
    attacks = AG.scan(read_rows(path), keep_ids=False)
    if attacks["n_hit"]:
        raise AG.AttackTemplateInTraining("evaluation attack text in source draw")
    return {"jevbench": rep, "attacks": attacks,
            "decision_index_document_guard": {"run": False, "reason": "index unavailable on this workstation"}}


def guard_units(units: list[dict], path: str | Path) -> dict:
    """Scan all final training-visible text, including authored facts and trusted policy."""
    from keel.protocol import DecisionRequest

    records = []
    for unit in units:
        req = DecisionRequest.from_json(unit["request"])
        records.append({"id": unit["id"], "source": unit["source"]["source"],
                        "state": [s.content for s in req.segments] + [req.policy.text],
                        "questions": {q.qid: q.as_wire() for q in req.questions}})
    write_rows(path, records)
    return guard_draw(path)


def build(n: int = 3300, pilot_n: int = 192) -> tuple[list[dict], list[dict], dict]:
    """Use three repository generator families; no public benchmark item is a source.

    Severity is isliced before gen_severity's unrelated SST loader, so no source
    files from the lab workstation are needed. All requested rows are validated.
    """
    from data import augment, jev1

    if n < 3 or not 0 < pilot_n < n:
        raise ValueError("need at least three items and a nonempty train/pilot split")
    count = (n + 2) // 3
    pools = [list(jev1._access_policy(i + 100000) for i in range(count)),
             list(itertools.islice(jev1.gen_severity(count, 0), count)),
             list(jev1._event_state(i + 100000) for i in range(count))]
    candidates = [r for triple in itertools.zip_longest(*pools) for r in triple if r is not None][:n]
    seen, rows = set(), []
    pair = JG.load()
    if not pair:
        raise ContaminationError("JevBench guard disabled")
    _, fp = pair
    for r in candidates:
        augment.validate(r)
        assert_public_disjoint(r, fp)
        AG.assert_clean_record(r)
        key = digest({"state": r["state"], "questions": r["questions"]})
        if key in seen:
            continue
        seen.add(key)
        parent = r["aug"]["b1_view"]["parent_id"]
        r["distill"] = {"group": parent, "content_sha256": key, "builder": "data.jev1",
                        "builder_seed": jev1.SEED, "source_split": "synthetic", "eval_only": False}
        rows.append(r)
    groups = sorted({r["distill"]["group"] for r in rows}, key=lambda g: digest(["distill-pilot-v0", g]))
    heldout = set(groups[:pilot_n])
    pilot = [r for r in rows if r["distill"]["group"] in heldout]
    train = [r for r in rows if r["distill"]["group"] not in heldout]
    assert_split(train, pilot)
    report = {"requested": n, "built": len(rows), "duplicates_removed": n - len(rows),
              "train": len(train), "pilot": len(pilot), "seed": jev1.SEED,
              "type_mix": dict(collections.Counter(r["questions"]["q"]["type"] for r in rows)),
              "family_mix": dict(collections.Counter(r["aug"]["jev1"]["family8"] for r in rows)),
              "train_sha256": digest(train), "pilot_sha256": digest(pilot),
              "builder_attempt": "jev3 policy generator unavailable: its Windows-only prior jev1 pool is absent; "
                                 "jev1 synthetic access_policy, severity and event_state builders run directly"}
    return train, pilot, report


def assert_split(train: list[dict], pilot: list[dict]) -> None:
    for name, key in (("item", lambda r: r["id"]), ("group", lambda r: r["distill"]["group"]),
                      ("content", lambda r: r["distill"]["content_sha256"])):
        if {key(r) for r in train} & {key(r) for r in pilot}:
            raise ValueError(f"train/pilot {name} overlap")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=3300)
    p.add_argument("--pilot-n", type=int, default=192)
    a = p.parse_args()
    out = Path(a.out)
    train, pilot, report = build(a.n, a.pilot_n)
    for name, rows in (("train", train), ("pilot", pilot)):
        path = out / f"{name}.jsonl.gz"
        write_rows(path, rows)
        report[f"{name}_guards"] = guard_draw(path)
    (out / "sources-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("guards")}, indent=2))


if __name__ == "__main__":
    main()
