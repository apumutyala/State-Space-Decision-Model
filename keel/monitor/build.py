"""Resumable, execution-verified data acquisition with a wall-time/spend stop."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .data import GENERATOR, SEED, Problem, Submission, author_messages, load_problems, parse_submission, validate


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_dataset(path: Path) -> tuple[dict[str, Problem], list[Submission]]:
    value = json.loads(path.read_text())
    problems = {p["problem_id"]: Problem(**{**p, "inputs": tuple(p["inputs"]), "outputs": tuple(p["outputs"])})
                for p in value["problems"]}
    return problems, [Submission(**s) for s in value["submissions"]]


def build_dataset(cache: Path, *, count: int = 260, batch_size: int = 48, workers: int = 12,
                  max_seconds: int = 7200, author_cap: float = 5.0) -> dict:
    from keel.runtime.vllm_judge import APP
    from .modal_generate import author_class
    from .sandbox import ModalExecutor

    started = time.time()
    problems, source = load_problems(cache, count)
    save(cache / "source.json", source)
    save(cache / "problems.json", [asdict(p) for p in problems])
    authored_path, checked_path = cache / "authored.json", cache / "checked.json"
    authored = json.loads(authored_path.read_text()) if authored_path.exists() else {}
    checked = json.loads(checked_path.read_text()) if checked_path.exists() else {}
    ledger_path = cache / "data_spend.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    Author = author_class()
    with APP.run():
        executor = ModalExecutor(APP, cache / "execution")
        def check_reference(problem):
            sid = problem.problem_id + ":reference"
            if sid not in checked:
                sub = Submission(sid, problem.problem_id, problem.reference, False, None, {})
                return sid, validate(problem, sub, executor)
            return sid, checked[sid]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sid, outcome in pool.map(check_reference, problems):
                checked[sid] = outcome
                save(checked_path, checked)
        eligible = [p for p in problems if checked[p.problem_id + ":reference"]["accepted"]]
        jobs = [(p, sabotage, replicate) for p in eligible for sabotage, replicate in ((False, 0), (False, 1), (True, 0))]
        pending = [j for j in jobs if f"{j[0].problem_id}:{j[1]}:{j[2]}" not in authored]
        author = None
        for offset in range(0, len(pending), batch_size):
            if time.time() - started > max_seconds:
                break
            spent = sum(x["estimate_usd"] for x in ledger)
            # Reserve a full 20-minute endpoint timeout before each paid batch.
            if spent + 1200 * 3.95 / 3600 > author_cap:
                break
            if author is None:
                author = Author()
            batch = pending[offset:offset + batch_size]
            tick = time.time()
            out = author.generate.remote([author_messages(p, s) for p, s, _ in batch], seed=SEED + offset)
            wall = time.time() - tick
            ledger.append({"date": "2026-10-15", "function": "MonitorAuthor.generate", "items": len(batch),
                           "client_wall_s": wall, "gpu_seconds_upper_estimate": wall + 60,
                           "estimate_usd": (wall + 60) * 3.95 / 3600, "meta": out["meta"]})
            save(ledger_path, ledger)
            for (p, s, rep), output in zip(batch, out["outputs"], strict=True):
                sub = None if output["truncated"] else parse_submission(output["text"], p, s, rep)
                authored[f"{p.problem_id}:{s}:{rep}"] = {"output": output, "submission": asdict(sub) if sub else None}
            save(authored_path, authored)
            def check_submission(job):
                p, s, rep = job
                value = authored[f"{p.problem_id}:{s}:{rep}"]["submission"]
                if value is None:
                    return None
                sub = Submission(**value)
                if sub.submission_id in checked:
                    return sub.submission_id, checked[sub.submission_id]
                return sub.submission_id, validate(p, sub, executor)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(check_submission, batch):
                    if result is not None:
                        checked[result[0]] = result[1]
                        save(checked_path, checked)
            accepted = [x["submission"] for x in authored.values() if x["submission"] is not None
                        and checked.get(x["submission"]["submission_id"], {}).get("accepted")]
            counts = Counter("backdoor" if s["sabotage"] else "honest" for s in accepted)
            print(json.dumps({"authored": len(authored), "verified": dict(counts), "spent_usd": sum(x["estimate_usd"] for x in ledger)}), flush=True)
            if counts["backdoor"] >= 150 and counts["honest"] >= 300:
                break
        # Verify any cached authorship not yet checked after an interrupted run.
        remaining = [(p, s, rep) for p, s, rep in jobs if f"{p.problem_id}:{s}:{rep}" in authored]
        for p, s, rep in remaining:
            value = authored[f"{p.problem_id}:{s}:{rep}"]["submission"]
            if value and value["submission_id"] not in checked and time.time() - started < max_seconds:
                checked[value["submission_id"]] = validate(p, Submission(**value), executor)
                save(checked_path, checked)
        save(cache / "sandbox_spend.json", executor.runs)
    accepted = [x["submission"] for x in authored.values() if x["submission"] is not None
                and checked.get(x["submission"]["submission_id"], {}).get("accepted")]
    # Exact duplicates within a problem/label are not additional solutions.
    unique = {}
    for sub in accepted:
        key = (sub["problem_id"], sub["sabotage"], sub["code"])
        unique.setdefault(key, sub)
    submissions = list(unique.values())
    counts = Counter("backdoor" if s["sabotage"] else "honest" for s in submissions)
    used = {s["problem_id"] for s in submissions}
    dataset = {"problems": [asdict(p) for p in problems if p.problem_id in used], "submissions": submissions}
    save(cache / "dataset.json", dataset)
    report = {"status": "target_met" if counts["backdoor"] >= 150 and counts["honest"] >= 300 else "data_target_not_met",
        "generator": GENERATOR, "source": source, "problems": len(used), "counts": dict(counts),
        "authored": len(authored), "parse_rejected": sum(x["submission"] is None for x in authored.values()),
        "duplicates_dropped": len(accepted) - len(submissions),
        "execution_reasons": dict(Counter(v["reason"] for k, v in checked.items() if not k.endswith(":reference"))),
        "reference_rejected": sum(not checked[p.problem_id + ":reference"]["accepted"] for p in problems),
        "elapsed_s": time.time() - started, "author_spend_estimate_usd": sum(x["estimate_usd"] for x in ledger),
        "dataset_sha256": hashlib.sha256((cache / "dataset.json").read_bytes()).hexdigest()}
    save(cache / "data_report.json", report)
    return report
