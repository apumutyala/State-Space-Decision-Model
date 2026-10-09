"""CPU dry-run seam of jobs/q9af_corpus_check/q9af_corpus_check_job.sh (Q9CC_DRYRUN=1) + the synthetic fixtures the
tests share. No real data: random-word sources with planted eval material, a synthetic eval-hash index.

    prepare(root)                                   # data/, index, config.json, data.sha256, CONTEXT_MANIFEST.json
    python -X utf8 -m tests.q9af_corpus_check_stub fetch <SSDM_DATA_ROOT>   # the runner's fetch step
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

KEY16 = "218affe13e367d07"


def words(prefix: str, n: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(f"{prefix}{rng.randrange(4000)}" for _ in range(n))


TEST_DOC = words("t", 120, 1)        # a TEST row's premise: copied whole into a web doc -> policy TEST
FROZEN_DOC = words("b", 90, 2)       # a held-out BBH (frozen catalog 58) row's input: copied into a math doc
DEV_DOC = words("d", 80, 3)          # a DEV row's question: 13 of its words inside a long web doc -> contact DEV only
DEV_FILTERED = words("x", 50, 4)     # a DEV row's state that a web doc equals exactly -> dropped by dev13-v1
TEMPLATE = "read the passage below and choose the single best option from the list that follows it"


def eval_rows():
    rows = [("TEST", "T:1", "ANLI", {"state": {"premise": TEST_DOC}, "questions": {"q": {"instructions": TEMPLATE}}}),
            ("HELDOUT", "B:1", "BBH fixed-option tasks", {"state": {"input": FROZEN_DOC},
                                                          "questions": {"q": {"instructions": TEMPLATE}}}),
            ("DEV", "D:1", "MMLU", {"state": {"q": DEV_DOC}, "questions": {"q": {"instructions": TEMPLATE}}})]
    for i in range(150):
        rows.append(("HELDOUT", f"X:{i}", "Filler", {"state": {"doc": words("f", 30, 100 + i)},
                                                     "questions": {"q": {"instructions": TEMPLATE}}}))
    return rows


def build_index(d: Path) -> str:
    from train import doc_overlap_guard as G

    rows = eval_rows()
    idx = G.build_index(None, G.load_allowlist(), rows_factory=lambda: iter(rows), n_parts=2)
    return G.save_index(idx, d, {"synthetic": True})["index_sha256"]


def attack_fragment() -> str:
    from train import attack_guard as A

    for key, frags in sorted(A.eval_templates().items()):
        for f in frags:
            if len(A.normalise(f)) >= 8:
                return f
    raise RuntimeError("no attack fragment of >= 8 words")


def tiny_config() -> dict:
    return {"seed": 0, "seq_len": 64, "model_dir": "unused",
            "mix": {"train_tokens": 6000, "shares": {"web": 0.5, "chat": 0.3, "math": 0.2},
                    "heldout_tokens": {"web": 150, "chat": 150, "math": 150}, "decontam": "dev13-v1"}}


def write_dev_rows(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"state": DEV_FILTERED, "questions": {"q": {"type": "choice"}}}) + "\n"
    with open(p, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:  # byte-stable across builds
        f.write(line.encode("utf-8"))


def write_sources(root: Path) -> dict:
    """{src: dir} with the mixdata globs; planted docs sit in the first row group (inside every quota)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from convert.mixdata import HELDOUT_GLOBS, SOURCES, TRAIN_GLOBS

    atk = attack_fragment()
    plant = {"web": {1: TEST_DOC, 2: words("w", 40, 7) + " " + " ".join(DEV_DOC.split()[10:23]) + " " + words("w", 40, 8),
                     3: DEV_FILTERED},
             "math": {1: words("m", 20, 9) + " " + FROZEN_DOC},
             "chat": {2: [{"role": "user", "content": words("u", 20, 10) + " " + atk},
                          {"role": "assistant", "content": words("a", 30, 11)}]}}
    roots = {}
    for s in SOURCES:
        r = root / s
        roots[s] = str(r)
        for gi, g in enumerate(TRAIN_GLOBS[s] + HELDOUT_GLOBS[s]):
            p = r / g.replace("*", "x")
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            seed = 1000 * (gi + 1) + len(s)
            if s == "chat":
                rows = [{"messages": plant["chat"].get(k) if gi == 0 and k in plant["chat"] else
                         [{"role": "user", "content": words("q", 20, seed + k)},
                          {"role": "assistant", "content": words("r", 40, seed + 500 + k)}]} for k in range(80)]
            else:
                rows = [{"text": plant[s].get(k) if gi == 0 and k in plant[s] else words(s[0] + "z", 60, seed + k)}
                        for k in range(120)]
            pq.write_table(pa.Table.from_pylist(rows), p, row_group_size=10)
    return roots


def prepare(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    write_dev_rows(data / "slices" / "dev-rows.jsonl.gz")
    sha = build_index(data / "doc-overlap" / KEY16)
    (root / "index.sha").write_text(sha, encoding="utf-8", newline="\n")
    lines = []
    for p in sorted([data / "slices" / "dev-rows.jsonl.gz", *(data / "doc-overlap" / KEY16).iterdir()]):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(data).as_posix()}")
    (root / "data.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (root / "config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    (root / "CONTEXT_MANIFEST.json").write_text(json.dumps({"dry_run": True}), encoding="utf-8")
    return root


def fetch(data_root: str) -> int:
    dr = Path(data_root)
    roots = write_sources(dr / "sources")
    env = {"web": "FINEWEB_EDU_DIR", "chat": "ULTRACHAT_DIR", "math": "FINEMATH_DIR"}
    with open(dr / "paths.env", "a", encoding="utf-8", newline="\n") as f:
        for s, name in env.items():
            f.write(f"export {name}={Path(roots[s]).as_posix()}\n")
            (dr / f"DONE_{name}").write_text(s, encoding="utf-8")
    print("stub fetch ok")
    return 0


def main(argv) -> int:
    cmd, *args = argv
    if cmd == "fetch":
        return fetch(args[0])
    raise SystemExit(f"unknown stub command {cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
