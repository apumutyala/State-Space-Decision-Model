"""B1 v2 add-on views: low-positive-rate yes/no detection and minority-support claim checks (BOARD B1, b1-draw100k-v2).

Why: the D1 cell over-calls the majority-positive label on families missing from its draw (ACOS "yes" 25% vs gold
0.6%, RAGTruth true 91% vs 42%, NLI4CT Entailment 64% vs 36%). These views re-present ALREADY-ALLOWED, sha-locked
train-split sources (``data/manifests/sources.lock.json``; no new download) in the eval layouts of those families,
with labels at the data's natural base rate (no rebalancing toward yes):

- ``member`` (CLINC150 train + oos_train, BANKING77 train): ACOS-style presence checks. One record = one user
  message + 12 yes/no questions "is intent X expressed?", the 12 candidates being the intents most lexically similar
  to the message (token Jaccard, seeded sha256 tie-break; CLINC's out-of-scope pseudo-intent is never a candidate).
  The gold intent is a candidate only when it ranks in that shortlist, and out-of-scope messages have none, so the
  yes rate is whatever the data gives (measured in the build report), and many records are all-no.
- ``claim`` (ANLI train r1-r3): HoVer-style claim verification. Evidence = premise, claim = hypothesis;
  SUPPORTED iff the ANLI label is entailment (neutral and contradiction are NOT_SUPPORTED): natural ~32% supported.
- ``verify`` (MMLU auxiliary_train): per-candidate answer verification, one yes/no question per option, exactly
  the gold option is yes: natural 1/K (25% for 4 options).

Every view keeps its native item identity (``<native id>~view:<kind>``, ``aug.b1_view.parent_id``), so a view and
the native row of the same source item never co-occur in a draw, and every row passes ``data.b1.Filter`` (cohort
v2/v3, SplitGuard, ``harness.heldout.check_record``, Index fingerprint, Index content / text / item-text, K/T cell
filter) and then ``data.b1_check``. The instruction and option wording is ours except the two HoVer label
descriptions and its one-line question (Index boilerplate shared by every HoVer row: a layout, not an item).

Raw rows: CLINC150 / BANKING77 are read from their locked JSON / CSV directly; ANLI / MMLU-aux are parquet, which the
draw interpreter cannot read (no pyarrow), so ``dump`` (run once with any interpreter that has pyarrow) writes their
(id, fields, gold) to ``<BUILD_DIR>/b1-lowrate-src/`` and records the parquet sha256s (checked against the lock); the
draw verifies the dump's own sha256 against ``dump-manifest.json``.

NOT built here, by rule: Home appliances (catalog 9, family ``device_state_record``) is FROZEN in
``harness/manifests/heldout-draw-v1.json``; ``check_generator`` refuses any device-handling or device-membership
generator (receipt ``refused-home-appliances.json``).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from . import config

SRC_DIR_NAME = "b1-lowrate-src"
N_MEMBER = 12
YES_NO_MEMBER = {"yes": "The message expresses this intent.", "no": "The message does not express this intent."}
YES_NO_VERIFY = {"yes": "This candidate is the correct answer.", "no": "This candidate is not the correct answer."}
HOVER_Q = "Is the claim supported by the evidence?"
HOVER_CRIT = {"SUPPORTED": "The evidence supports the claim.", "NOT_SUPPORTED": "The evidence does not support the claim."}
GEN_SPECS = {  # harness.heldout.check_generator specs (none targets a frozen family)
    "member": {"name": "b1v2_intent_membership", "task_family": "intent_membership", "targets": [4, 5]},
    "claim": {"name": "b1v2_claim_verification", "task_family": "claim_verification", "targets": [61, 42]},
    "verify": {"name": "b1v2_answer_verification", "task_family": "answer_verification", "targets": [57]},
}
REFUSED_SPECS = {
    "home_appliance_handling": {"name": "b1v2_home_appliance_handling", "task_family": "device_handling", "targets": [9]},
    "device_membership": {"name": "b1v2_device_membership", "task_family": "device_state_record"},
}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _toks(text: str) -> set[str]:
    return set(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold().replace("_", " ")))


def _key(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ raw sources


def dump(out: Path | None = None) -> dict:
    """ANLI train r1-r3 and MMLU auxiliary_train parquet -> jsonl.gz (needs pyarrow; data only)."""
    import pyarrow.parquet as pq
    from . import sources as S

    out = Path(out or config.BUILD_DIR / SRC_DIR_NAME)
    out.mkdir(parents=True, exist_ok=True)
    man = {"inputs": {}, "outputs": {}}
    labels = ["entailment", "neutral", "contradiction"]
    lines = []
    for rnd in (1, 2, 3):
        f = f"plain_text/train_r{rnd}-00000-of-00001.parquet"
        p = S.verified_path("anli", f)
        man["inputs"][f"anli/{f}"] = _sha(p)
        for r in pq.read_table(p).to_pylist():
            lines.append(json.dumps({"id": f"m2:anli:{r['uid']}", "premise": r["premise"], "hypothesis": r["hypothesis"],
                                     "gold": labels[int(r["label"])]}, ensure_ascii=False, sort_keys=True))
    f = "auxiliary_train/train-00000-of-00001.parquet"
    p = S.verified_path("mmlu_aux", f)
    man["inputs"][f"mmlu_aux/{f}"] = _sha(p)
    mm = []
    for i, row in enumerate(pq.read_table(p).to_pylist()):
        r = row["train"]
        if r["choices"] and len(r["choices"]) >= 2:
            mm.append(json.dumps({"id": f"m2:mmlu_aux:{i}", "question": r["question"], "choices": list(r["choices"]),
                                  "gold": int(r["answer"])}, ensure_ascii=False, sort_keys=True))
    for name, ls in (("anli.jsonl.gz", lines), ("mmlu_aux.jsonl.gz", mm)):
        with gzip.GzipFile(out / name, "wb", mtime=0) as fh:
            fh.write(("\n".join(ls) + "\n").encode("utf-8"))
        man["outputs"][name] = {"sha256": _sha(out / name), "rows": len(ls)}
    (out / "dump-manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    return man


class Raw:
    """id -> raw source row for CLINC150, BANKING77 (locked files) and the ANLI / MMLU-aux dump (sha-verified)."""

    def __init__(self, build_dir: Path | None = None, want: set[str] | None = None):
        """``want``: keep only these ANLI / MMLU-aux ids in memory (the laptop's RAM is shared)."""
        from . import sources as S

        self.ident = {}
        d = json.load(open(S.verified_path("clinc150", "data_full.json"), encoding="utf-8"))
        rows = [(t, l) for t, l in d["train"]] + [(t, "oos") for t, _ in d["oos_train"]]  # train + oos_train only
        self.clinc = {f"m2:clinc150:{i}": r for i, r in enumerate(rows)}
        self.clinc_intents = sorted({l for _, l in rows} - {"oos"})
        with open(S.verified_path("banking77", "train.csv"), encoding="utf-8", newline="") as f:
            brows = list(csv.DictReader(f))
        self.banking = {f"m2:banking77:{i}": (r["text"], r["category"]) for i, r in enumerate(brows)}
        self.banking_intents = sorted({r["category"] for r in brows})
        src = Path(build_dir or config.BUILD_DIR) / SRC_DIR_NAME
        man = json.loads((src / "dump-manifest.json").read_text(encoding="utf-8"))
        lock = json.loads(S.LOCK.read_text(encoding="utf-8"))
        for k, sha in man["inputs"].items():
            if lock[k]["sha256"] != sha:
                raise RuntimeError(f"{k}: dump input does not match sources.lock.json")
        self.anli, self.mmlu = {}, {}
        for name, tgt in (("anli.jsonl.gz", self.anli), ("mmlu_aux.jsonl.gz", self.mmlu)):
            if _sha(src / name) != man["outputs"][name]["sha256"]:
                raise RuntimeError(f"{name}: dump drifted from dump-manifest.json")
            with gzip.open(src / name, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        if want is None or r["id"] in want:
                            tgt[r["id"]] = r
        self.ident = {"dump_manifest_sha256": _sha(src / "dump-manifest.json"), "dump": man}


# ------------------------------------------------------------------ views


def _view(rid: str, source: str, family: str, kind: str, state, questions: dict, labels: dict, lic: str,
          extra: dict | None = None) -> dict:
    return {"schema": "ssdm-train-v0", "id": f"{rid}~view:{kind}", "family": family, "source": source,
            "source_id": rid.split(":", 2)[2], "source_split": "train", "license": lic, "state": state,
            "questions": questions, "labels": labels, "untrusted_fields": [],
            "aug": {"b1_view": {"kind": kind, "parent_id": rid, **(extra or {})}}}


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")


def member_view(raw: Raw, rid: str, seed: int) -> dict | None:
    if rid in raw.clinc:
        text, gold = raw.clinc[rid]
        intents, source, lic = raw.clinc_intents, "clinc150", "CC-BY-3.0"
    elif rid in raw.banking:
        text, gold = raw.banking[rid]
        intents, source, lic = raw.banking_intents, "banking77", "CC-BY-4.0"
    else:
        return None
    q = _toks(text)

    def sim(c):
        lab = _toks(c.replace("_", " "))
        return len(q & lab) / max(1, len(q | lab))
    cands = sorted(intents, key=lambda c: (-sim(c), _key("b1v2-member", seed, rid, c)))[:N_MEMBER]
    questions, labels = {}, {}
    for c in sorted(cands):
        name = c.replace("_", " ")
        qid = f"{_slug(c)}__present"
        questions[qid] = {"type": "choice", "criteria": dict(YES_NO_MEMBER),
                          "instructions": f"Given the message below, decide whether its sender is asking for the "
                                          f"intent ({name}). Answer yes only if this intent is expressed."}
        labels[qid] = "yes" if c == gold else "no"
    return _view(rid, source, "intent_membership", "member", {"message": text, "task": "intent presence check"},
                 questions, labels, lic, {"n_candidates": N_MEMBER, "gold_in_shortlist": gold in cands,
                                          "out_of_scope": gold == "oos"})


def claim_view(raw: Raw, rid: str, seed: int) -> dict | None:
    r = raw.anli.get(rid)
    if r is None:
        return None
    lab = "SUPPORTED" if r["gold"] == "entailment" else "NOT_SUPPORTED"
    return _view(rid, "anli", "claim_verification", "claim",
                 {"claim": r["hypothesis"], "evidence": [{"text": r["premise"]}]},
                 {"q": {"type": "choice", "instructions": HOVER_Q, "criteria": dict(HOVER_CRIT)}}, {"q": lab},
                 "CC-BY-NC-4.0", {"anli_label": r["gold"]})


def verify_view(raw: Raw, rid: str, seed: int) -> dict | None:
    r = raw.mmlu.get(rid)
    if r is None or len(r["choices"]) > 8 or len(set(r["choices"])) != len(r["choices"]):
        return None
    order = sorted(range(len(r["choices"])), key=lambda j: _key("b1v2-verify", seed, rid, j))
    questions, labels = {}, {}
    for n, j in enumerate(order):
        qid = f"candidate_{n + 1}"
        questions[qid] = {"type": "choice", "criteria": dict(YES_NO_VERIFY),
                          "instructions": f"Is this candidate the correct answer to the question? Candidate: "
                                          f"{r['choices'][j]}"}
        labels[qid] = "yes" if j == r["gold"] else "no"
    return _view(rid, "mmlu_aux", "answer_verification", "verify", {"question": r["question"]}, questions, labels,
                 "MIT (constituents: ARC, OBQA, RACE, MCTest; see SOURCES.md)", {"k": len(order)})


VIEWS = {"member": member_view, "claim": claim_view, "verify": verify_view}


def check_specs() -> dict:
    """Generator specs of the added views pass check_generator; the Home appliances asks are refused (fail-closed)."""
    from train.cohort_v3 import _ensure_kit
    _ensure_kit()
    from harness import heldout

    out = {"allowed": {}, "refused": {}}
    for k, s in GEN_SPECS.items():
        heldout.check_generator(s)
        out["allowed"][k] = s
    for k, s in REFUSED_SPECS.items():
        try:
            heldout.check_generator(s)
            raise RuntimeError(f"{k} is no longer refused: re-plan before generating it")
        except heldout.FrozenFamilyError as e:
            out["refused"][k] = {"spec": s, "error": str(e)}
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="data.b1_lowrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump")
    c = sub.add_parser("specs")
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = dump() if a.cmd == "dump" else check_specs()
    if a.cmd == "specs" and a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
