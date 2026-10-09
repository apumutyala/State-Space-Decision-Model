"""Reproduce D13 joint continuation-cap bypass and produce a bounded patch; no Torch."""
from __future__ import annotations
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT.parent / "d13"
OUT = ROOT / "receipts/2026-10-09-C-multirecord-quota/joint-option-cap"
COMMIT = "8d92f04f6bbc46ec40cf19b2ab5786435e139266"


def method(source):
    klass = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    fn = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_next_record_rows")
    env = {"Item": object, "item_record": lambda item: item.record}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train/trainer.py", "exec"), env)
    return env[fn.name]


def fake(records, cap):
    items = [SimpleNamespace(record=f"r{rid}", row=SimpleNamespace(n_options=k))
             for rid, ks in enumerate(records) for k in ks]
    return SimpleNamespace(train_items=items, cfg=SimpleNamespace(row_options=cap), cursor=0, epoch=0,
                           _order=lambda epoch: list(range(len(items))))


def shape(items):
    return {"rows": len(items), "continuations": sum(i.row.n_options for i in items),
            "record_ids": [i.record for i in items]}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TRAINER, text=True).strip() == COMMIT
    path = TRAINER / "train/trainer.py"
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    old = """            end = min(end, self.cursor + n - len(rows))
            rows += [self.train_items[i] for i in order[self.cursor:end]]
            self.cursor = end"""
    new = """            end = min(end, self.cursor + n - len(rows))
            pending = [self.train_items[i] for i in order[self.cursor:end]]
            # The joint head uses one continuation call across ALL prefix groups.
            # Bound its total without splitting a pending record across steps.
            if self.cfg.row_options:
                pending_k = sum(it.row.n_options for it in pending)
                if pending_k > self.cfg.row_options:
                    raise ValueError("joint record exceeds row_options; raise the cap after a memory gate")
                if sum(it.row.n_options for it in rows) + pending_k > self.cfg.row_options:
                    break
            rows += pending
            self.cursor = end"""
    assert source.count(old) == 1
    patched = source.replace(old, new)
    before, after = method(source), method(patched)
    # A complete K77 C record: top-profile77 + score5 + five binary questions2 each.
    records = [[77, 5, 2, 2, 2, 2, 2], [77]]
    original = fake(records, 160)
    baseline = shape(before(original, 8))
    assert baseline["continuations"] == 169
    fixed = fake(records, 160)
    a, b = shape(after(fixed, 8)), shape(after(fixed, 8))
    assert (a["rows"], a["continuations"], b["rows"], b["continuations"]) == (7, 92, 1, 77)
    assert fixed.cursor == 8 and fixed.epoch == 0
    # No skipped/duplicated rows at a checkpoint boundary or next epoch.
    resumed = fake(records, 160)
    resumed.cursor = 7
    assert shape(after(resumed, 8)) == b
    assert shape(after(fixed, 8)) == a and fixed.epoch == 1
    # Zero cap and exact-fit cap preserve existing behavior and cursor.
    for cap in (0, 169, 1000):
        left, right = fake(records, cap), fake(records, cap)
        assert shape(before(left, 8)) == shape(after(right, 8)) == baseline
        assert (left.cursor, left.epoch) == (right.cursor, right.epoch)
    # Oversized logical records fail before a forward or cursor advance.
    oversized = fake(records, 91)
    try:
        after(oversized, 8)
        raise AssertionError("oversized logical record unexpectedly admitted")
    except ValueError:
        assert oversized.cursor == 0 and oversized.epoch == 0
    assert path.read_bytes() == raw
    OUT.mkdir(parents=True)
    patch = "".join(difflib.unified_diff(source.splitlines(keepends=True), patched.splitlines(keepends=True),
                                      fromfile="a/train/trainer.py", tofile="b/train/trainer.py"))
    (OUT / "joint-option-cap.patch").write_text(patch, encoding="utf-8")
    report = {"schema": "ssdm.c_joint_option_cap.v1", "source_commit": COMMIT,
              "source_sha256": hashlib.sha256(raw).hexdigest(),
              "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
              "row_budget": 8, "row_options": 160, "before": baseline, "after_steps": [a, b],
              "resume_and_epoch_checks": "pass", "zero_and_exact_cap_checks": "pass",
              "oversized_record_refuses_without_cursor_advance": True,
              "source_worktree_unchanged": True, "torch_imported": False,
              "scope": "actual D13 selection method; continuation count proof, not GPU memory/throughput measurement",
              "integration": "apply alongside C-consuming joint recipe; no loss, model, eval or data changes; A owns adoption",
              "gpu_spend_usd": 0}
    (OUT / "verification.json").write_text(json.dumps(report, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
