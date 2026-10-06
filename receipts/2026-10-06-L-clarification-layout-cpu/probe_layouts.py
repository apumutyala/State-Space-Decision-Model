"""Record synthetic missing-field layout coverage without model or audit data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from language_growth.development_data import generate_development_tasks


def main() -> None:
    counts = {kind: Counter() for kind in ("ask_missing", "read_with_unrelated_obligation")}
    for seed in range(8):
        for task in generate_development_tasks(seed, 25, clarification_mode=True):
            kind = task["training_only"]["kind"]
            if kind in counts:
                fields = tuple(tuple(sorted(record["fields"]))
                               for record in task["training_only"]["catalog"]["records"])
                counts[kind][json.dumps(fields, separators=(",", ":"))] += 1
    if set(counts["ask_missing"]) != set(counts["read_with_unrelated_obligation"]):
        raise AssertionError("layout support differs between Ask and answerable arms")
    if len(counts["ask_missing"]) != 6:
        raise AssertionError("not all two-missing-field layouts are represented")
    result = {kind: dict(sorted(counter.items())) for kind, counter in counts.items()}
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8", newline="\n")
    print(json.dumps({kind: {"tasks": sum(counter.values()), "layouts": len(counter)}
                      for kind, counter in counts.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
