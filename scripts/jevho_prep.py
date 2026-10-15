"""jev-heldout raw prep: turn the few parquet / large sources into small compact JSONL caches under ssdm-data/raw/jevho.

    python -X utf8 -I scripts/jevho_prep.py boolq        (pyarrow venv)  boolq/validation parquet -> boolq/validation.jsonl.gz
    python -X utf8 -I scripts/jevho_prep.py glaive-used                  ids of every Glaive row any m2 build ever used
    python -X utf8 -I scripts/jevho_prep.py glaive       (pyarrow venv)  unused Glaive rows -> glaive/candidates.jsonl.gz

Downloaded bytes are untrusted DATA (parsed, never executed). Nothing here prints item text; only counts and shas.

glaive-used: every ``glaive_fc:<row>`` id in every ``*.manifest.jsonl.gz`` and every jev ``records.jsonl.gz`` under
ssdm-data/m2/build (the 150k / 300k / b1 / jev draws), so tool_selection only uses Glaive TRAIN rows no build ever
touched ("train-unused"). glaive: one streamed pass over the on-disk Glaive parquet (row index = the m2 id scheme);
rows in the jev-heldout hash partition (sha256("jevho-part|glaive:<i>") % 8 == 0) that are unused, single-turn,
<= 240-char user turn and whose first assistant turn is a function call to one of the listed tools.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")
RAW = DATA / "raw" / "jevho"
BUILD = DATA / "m2" / "build"
GLAIVE_PARQUET = DATA / "m2" / "sources" / "glaive_fc" / "0000.parquet"
GLAIVE_FN_HEAD = "Use them if required -"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def boolq() -> None:
    import pyarrow.parquet as pq
    src = RAW / "boolq" / "validation-00000-of-00001.parquet"
    out = RAW / "boolq" / "validation.jsonl.gz"
    n = 0
    with gzip.open(out, "wt", encoding="utf-8") as fo:
        for b in pq.ParquetFile(src).iter_batches(batch_size=500, columns=["question", "answer", "passage"]):
            for q, a, p in zip(*(b.column(i).to_pylist() for i in range(3))):
                fo.write(json.dumps({"row": n, "question": q, "answer": bool(a), "passage": p}, ensure_ascii=False) + "\n")
                n += 1
    print(json.dumps({"boolq_validation_rows": n, "parquet_sha256": sha_file(src)}))


def glaive_used() -> None:
    pat = re.compile(r"glaive_fc:(\d+)")
    used: set[int] = set()
    files = sorted(BUILD.rglob("*.manifest.jsonl.gz")) + sorted(BUILD.glob("bschema-draw-jev*/records.jsonl.gz"))
    per = {}
    for f in files:
        k = 0
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "glaive_fc:" in line:
                    for m in pat.finditer(line):
                        used.add(int(m.group(1)))
                        k += 1
        if k:
            per[str(f.relative_to(BUILD))] = k
    (RAW / "glaive").mkdir(parents=True, exist_ok=True)
    (RAW / "glaive" / "used-ids.json").write_text(json.dumps({"n_used": len(used), "files_with_hits": per,
                                                              "n_files_scanned": len(files),
                                                              "ids": sorted(used)}) + "\n", encoding="utf-8")
    print(json.dumps({"n_used": len(used), "files_scanned": len(files), "files_with_hits": len(per)}))


def _tools(sysmsg: str):
    at = sysmsg.find(GLAIVE_FN_HEAD)
    if at < 0:
        return None
    s, pos, tools, dec = sysmsg[at + len(GLAIVE_FN_HEAD):], 0, [], json.JSONDecoder()
    try:
        while True:
            while pos < len(s) and s[pos].isspace():
                pos += 1
            if pos >= len(s):
                break
            obj, pos = dec.raw_decode(s, pos)
            tools.append(obj)
    except json.JSONDecodeError:
        return None
    ok = [t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"]
          and isinstance(t.get("description"), str) and t["description"].strip()]
    return ok if len(ok) == len(tools) else None


def glaive() -> None:
    import pyarrow.parquet as pq
    used = set(json.loads((RAW / "glaive" / "used-ids.json").read_text(encoding="utf-8"))["ids"])
    pool: dict[str, str] = {}
    out = RAW / "glaive" / "candidates.jsonl.gz"
    st = {"rows": 0, "used_skipped": 0, "out_of_partition": 0, "ineligible": 0, "kept": 0}
    i = 0
    with gzip.open(out, "wt", encoding="utf-8") as fo:
        for b in pq.ParquetFile(GLAIVE_PARQUET).iter_batches(batch_size=500, columns=["system", "chat"]):
            for sysmsg, chat in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
                sysmsg, chat = sysmsg or "", chat or ""
                st["rows"] += 1
                tools = _tools(sysmsg)
                for t in tools or []:
                    if len(pool) < 30000 and t["name"] not in pool and len(t["name"]) <= 60:
                        pool[t["name"]] = t["description"].strip()[:160]
                if int(hashlib.sha256(f"jevho-part|glaive:{i}".encode()).hexdigest()[:8], 16) % 8:
                    st["out_of_partition"] += 1
                elif i in used:
                    st["used_skipped"] += 1
                else:
                    ok = bool(tools) and chat.startswith("USER:") and "\nASSISTANT:" in chat
                    if ok:
                        user, rest = chat[len("USER:"):].split("\nASSISTANT:", 1)
                        user, first = user.strip(), rest.split("<|endoftext|>", 1)[0].strip()
                        m = re.search(r'"name"\s*:\s*"([^"]+)"', first) if first.startswith("<functioncall>") else None
                        own = {t["name"]: t["description"].strip()[:160] for t in tools}
                        ok = bool(user) and len(user) <= 240 and "USER:" not in user and m is not None and \
                            m.group(1) in own
                    if ok:
                        rec = {"i": i, "user": user, "gold": m.group(1), "tools": own,
                               "row_sha256": hashlib.sha256(json.dumps([sysmsg, chat], ensure_ascii=False).encode()).hexdigest()}
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        st["kept"] += 1
                    else:
                        st["ineligible"] += 1
                i += 1
    (RAW / "glaive" / "name-pool.json").write_text(json.dumps(pool, ensure_ascii=False) + "\n", encoding="utf-8")
    st["parquet_sha256"] = sha_file(GLAIVE_PARQUET)
    st["pool_names"] = len(pool)
    (RAW / "glaive" / "prep-report.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(st))


if __name__ == "__main__":
    {"boolq": boolq, "glaive-used": glaive_used, "glaive": glaive}[sys.argv[1]]()
