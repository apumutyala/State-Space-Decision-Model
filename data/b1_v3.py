"""B1 v3 families (BOARD B1, b1-draw100k-v3): official TRAIN splits of seven newly fetched sources, rendered in the
Index eval presentation of the matching family, at their natural label base rates.

Sources (``data/sources.py`` REGISTRY, sha256 in ``data/manifests/sources.lock.json``, files under
``SOURCES_DIR/dl-2026-10-08/<name>/``; data only, read with ``python -I``, nothing inside is executed):

=========  =====================================================  ===========================================
family     source (train split only)                              presentation (Index family)
=========  =====================================================  ===========================================
acos       NUSTM/ACOS Laptop + Restaurant ``*_quad_train.tsv``    ACOS-category-sentiment (38): one review, one
                                                                  inventory group of yes/no category/sentiment
                                                                  questions (Laptop: 6 groups of <=64 over 121
                                                                  categories x 3; Restaurant: 13 x 3 = 39)
nli4ct     Task-2-SemEval-2024 ``training_data.zip:train.json``   NLI4CT-2024 (42): trial section(s) +
                                                                  Entailment / Contradiction
vast       zero-shot-stance ``vast_train.csv``                    VAST (41): Topic / Post, A against B favor C
                                                                  neutral (one row per post)
ragtruth   RAGTruth ``response.jsonl`` split=train                RAGTruth (59): prompt + response, noul
                                                                  "contains unsupported content"
hover      HoVer ``hover_train_release_v1.1.json`` + the          HoVer (61): claim + supporting-fact documents,
           official ``wiki_wo_links.db`` for evidence documents   SUPPORTED / NOT_SUPPORTED
habermas   habermas_machine candidate comparisons,                Habermas-consensus (50): question + panel
           TRAINING_DATA_V1-V4, question.split TRAIN              opinions, pick the minimum-summed-rank
                                                                  statement (initial iteration, complete human
                                                                  panel; tied rounds skipped)
mnli       nyu-mll/multi_nli train                                generic binary NLI (entailment vs not:
                                                                  entailment ~1/3, the minority)
=========  =====================================================  ===========================================

Stages: ``convert`` (needs pyarrow for the two parquet files; any interpreter) writes compact per-family rows to
``<BUILD_DIR>/b1-v3-src/<family>.jsonl.gz`` + ``convert-manifest.json`` (input sha256s checked against the lock);
``overlap`` (the draw interpreter) checks every converted row against that family's Index eval rows: source-id
identity and Index fingerprint v3 (exact / ngram13 / minhash, plus edit / contain), writes ``overlap.json`` with the
rows to drop; ``Raw`` (used by ``data.b1``) loads the converted rows (sha-verified), removes the overlap drops and
builds records via ``VIEWS``. Every drawn row then passes ``data.b1.Filter`` with the fingerprint on (any catalog)
and ``data.b1_check``.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

from . import config

SRC_DIR_NAME = "b1-v3-src"
SEED = 20261008
FAMS = ("acos", "nli4ct", "vast", "ragtruth", "hover", "habermas", "mnli")
#: Index family name (suite rows ``family``) and catalog id whose eval rows each family is checked against
INDEX_FAMILY = {"acos": ("ACOS-category-sentiment", 38), "nli4ct": ("NLI4CT-2024", 42), "vast": ("VAST", 41),
                "ragtruth": ("RAGTruth response-level hallucination", 59), "hover": ("HoVer claim verification", 61),
                "habermas": ("Habermas-consensus", 50), "mnli": (None, None)}
LICENSE = {"acos": "none stated (NUSTM/ACOS research release)", "nli4ct": "none stated (SemEval-2024 T2 research release)",
           "vast": "none stated (zero-shot-stance research release)", "ragtruth": "MIT", "hover": "CC-BY-SA-4.0",
           "habermas": "CC-BY-4.0", "mnli": "MultiNLI (OANC / CC-BY-SA-3.0 / MIT mix per HF card)"}
#: MNLI is large: only a seeded subset of this many train pairs is converted (the draw needs a few thousand)
MNLI_KEEP = 40000

# ---- Index presentation strings (layout / label-set boilerplate shared by every row of the family)
ACOS_TASK = "ACOS fixed category/sentiment presence"
ACOS_INSTR = ("Given the review below, decide whether it expresses at least one aspect-category/sentiment pair ({cat}, "
              "{sent}) anywhere in the review. Answer yes only when that exact category and sentiment pair is present; "
              "answer no otherwise. Multiple opposite sentiments are separate valid questions.")
ACOS_CRIT = {"yes": "The exact category/sentiment pair is present.", "no": "The exact category/sentiment pair is absent."}
ACOS_SENT = {"0": "negative", "1": "neutral", "2": "positive"}
#: The Index splits a Laptop review's 363 questions into groups of 64 (Restaurant: one group of 39); a 64-question row
#: renders to ~9,700 tokens in the cell's ordered layout (measured: 40 rows, all > 2,048), so a training row carries a
#: seeded slice of 10 consecutive questions of the same ordered inventory (<= ~1.8k tokens; 12 put 14% over), labels at the natural rate.
ACOS_GROUP = 10
NLI4CT_INSTR = "Classify the statement against the supplied clinical trial evidence:\n"
NLI4CT_CRIT = {"Entailment": "The clinical trial evidence entails the statement.",
               "Contradiction": "The clinical trial evidence contradicts the statement."}
VAST_INSTR = "Topic: {topic}\nPost: {post}\nDetermine the stance of the post toward the topic."
VAST_CRIT = {"A": "against", "B": "favor", "C": "neutral"}
VAST_LABEL = {"0": "A", "1": "B", "2": "C"}  # VAST: 0 con, 1 pro, 2 neutral (Index source_label "1" -> B favor)
RAG_Q = "The response contains content that is not supported by the context in the prompt."
RAG_CRIT = {"true": RAG_Q, "false": "All content of the response is supported by the context in the prompt."}
HOVER_Q = "Is the claim supported by the evidence?"
HOVER_CRIT = {"SUPPORTED": "The evidence supports the claim.", "NOT_SUPPORTED": "The evidence does not support the claim."}
HAB_INSTR = ("Choose the consensus statement you predict this group would rank highest on average, given their expressed "
             "opinions. Each participant has equal weight. Assess the group's preferences, rather than your own policy "
             "preference.")
MNLI_INSTR = "Does the premise entail the hypothesis?\nPremise: {p}\nHypothesis: {h}"
MNLI_CRIT = {"entailment": "The premise entails the hypothesis.",
             "not_entailment": "The premise does not entail the hypothesis (it is neutral or contradicts it)."}
GEN_SPECS = {
    "acos": {"name": "b1v3_acos_presence", "task_family": "acos_category_sentiment", "targets": [38]},
    "nli4ct": {"name": "b1v3_nli4ct", "task_family": "clinical_trial_nli", "targets": [42]},
    "vast": {"name": "b1v3_vast", "task_family": "stance", "targets": [41]},
    "ragtruth": {"name": "b1v3_ragtruth", "task_family": "hallucination_detection", "targets": [59]},
    "hover": {"name": "b1v3_hover", "task_family": "claim_verification", "targets": [61]},
    "habermas": {"name": "b1v3_habermas", "task_family": "consensus_selection", "targets": [50]},
    "mnli": {"name": "b1v3_mnli", "task_family": "nli", "targets": []},
}
#: ACOS category inventories: the Index label set (121 Laptop / 13 Restaurant categories, in Index order)
ACOS_INVENTORY_FILE = Path(__file__).with_name("b1_v3_acos_inventory.json")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _key(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()


def _inventory() -> dict[str, list[str]]:
    inv = json.loads(ACOS_INVENTORY_FILE.read_text(encoding="utf-8"))
    assert len(inv["Laptop-ACOS"]) == 121 and len(inv["Restaurant-ACOS"]) == 13
    return inv


# ------------------------------------------------------------------ convert (raw train files -> compact rows)


def _acos(S) -> list[dict]:
    out = []
    for f, ds, tag in (("Laptop-ACOS/laptop_quad_train.tsv", "Laptop-ACOS", "laptop"),
                       ("Restaurant-ACOS/rest16_quad_train.tsv", "Restaurant-ACOS", "rest16")):
        text = S.verified_path("acos", f).read_text(encoding="utf-8")
        for n, line in enumerate(text.split("\n"), 1):
            if not line.strip():
                continue
            parts = line.split("\t")
            pairs = sorted({(q.split(" ")[1], q.split(" ")[2]) for q in parts[1:]})
            out.append({"id": f"m2:acos:{tag}:train:line-{n}", "ds": ds, "review": parts[0], "pairs": pairs})
    return out


def _nli4ct(S) -> list[dict]:
    z = zipfile.ZipFile(S.verified_path("nli4ct", "training_data.zip"))
    train = json.loads(z.read("train.json"))  # train.json only: dev.json is never read
    cts: dict[str, dict] = {}

    def ct(i):
        if i not in cts:
            cts[i] = json.loads(z.read(f"CT json/{i}.json"))
        return cts[i]
    out = []
    for uid, v in sorted(train.items()):
        sec = v["Section_id"]
        st = {"primary_trial": {"id": v["Primary_id"], "section": sec, "text": ct(v["Primary_id"])[sec]}}
        if v.get("Secondary_id"):
            st["secondary_trial"] = {"id": v["Secondary_id"], "section": sec, "text": ct(v["Secondary_id"])[sec]}
        out.append({"id": f"m2:nli4ct:{uid}", "state": st, "statement": v["Statement"].strip(), "gold": v["Label"],
                    "type": v["Type"]})
    return out


def _vast(S) -> list[dict]:
    csv.field_size_limit(1 << 30)
    out = []
    with open(S.verified_path("vast", "vast_train.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append({"id": f"m2:vast:{r['new_id']}", "post_key": _key("vast-post", r["post"])[:24],
                        "topic": r["topic_str"], "post": r["post"], "gold": VAST_LABEL[r["label"]]})
    return out


def _ragtruth(S) -> list[dict]:
    src = {}
    with open(S.verified_path("ragtruth", "source_info.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                src[str(s["source_id"])] = (s["prompt"], s["task_type"])
    out = []
    with open(S.verified_path("ragtruth", "response.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["split"] != "train" or r["quality"] == "truncated":  # train only; Index keeps good + refusals
                continue
            prompt, task = src[str(r["source_id"])]
            out.append({"id": f"m2:ragtruth:{r['id']}", "source_id": str(r["source_id"]), "task": task,
                        "prompt": prompt, "response": r["response"], "gold": "true" if r["labels"] else "false",
                        "model": r["model"]})
    return out


def _hover(S) -> list[dict]:
    import sqlite3
    claims = json.loads(S.verified_path("hover", "hover_train_release_v1.1.json").read_text(encoding="utf-8"))
    db = S.verified_path("hover", "wiki_wo_links.db")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro&immutable=1", uri=True)
    out = []
    for c in claims:
        titles = []
        for t, _ in c["supporting_facts"]:
            if t not in titles:
                titles.append(t)
        ev = []
        for t in titles:
            row = con.execute("select text from documents where id = ?", (t,)).fetchone()
            if row is None:
                ev = None
                break
            ev.append({"title": t, "text": row[0]})
        if ev is None:
            continue
        out.append({"id": f"m2:hover:{c['uid']}", "claim": c["claim"], "evidence": ev, "gold": c["label"],
                    "hops": c["num_hops"], "hpqa_id": c["hpqa_id"]})
    con.close()
    return out


def _habermas(S) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq

    cols = ["round_id", "iteration_index", "metadata.version", "metadata.provenance", "metadata.status",
            "question.id", "question.text", "question.split", "own_opinion.text", "candidates.text",
            "candidates.metadata.id", "rankings.candidate_ids", "rankings.numerical_ranks"]
    t = pq.ParquetFile(S.verified_path("habermas", "hm_all_candidate_comparisons.parquet")).read(columns=cols)
    groups: dict[str, list[dict]] = {}
    for r in t.to_pylist():
        if not str(r["metadata.version"]).startswith("TRAINING_DATA_") or r["question.split"] != "TRAIN":
            continue
        groups.setdefault(r["round_id"], []).append(r)
    del t
    stats = {"rounds_train": len(groups), "kept": 0, "no_initial_complete_panel": 0, "tie": 0, "inconsistent": 0}
    out = []
    for rid in sorted(groups):
        rows = [r for r in groups[rid] if r["iteration_index"] == 0]
        if not rows or any(r["metadata.provenance"] != "HUMAN_CITIZEN" or r["metadata.status"] != "COMPLETED"
                           for r in rows) or len(rows) < 2:
            stats["no_initial_complete_panel"] += 1
            continue
        cand_ids, cand_txt = rows[0]["candidates.metadata.id"], rows[0]["candidates.text"]
        if not cand_ids or len(cand_ids) < 2 or any(r["candidates.metadata.id"] != cand_ids for r in rows):
            stats["inconsistent"] += 1
            continue
        summed = {c: 0 for c in cand_ids}
        ok = True
        for r in rows:
            if sorted(r["rankings.candidate_ids"] or []) != sorted(cand_ids):
                ok = False
                break
            for c, k in zip(r["rankings.candidate_ids"], r["rankings.numerical_ranks"]):
                summed[c] += int(k)
        if not ok:
            stats["inconsistent"] += 1
            continue
        best = min(summed.values())
        winners = [c for c, v in summed.items() if v == best]
        if len(winners) != 1:
            stats["tie"] += 1
            continue
        text = dict(zip(cand_ids, cand_txt))
        order = sorted(cand_ids, key=lambda c: _key("b1v3-hab", SEED, rid, c))  # seeded option order
        out.append({"id": f"m2:habermas:{rid}", "question_id": rows[0]["question.id"],
                    "question": rows[0]["question.text"], "opinions": [r["own_opinion.text"] for r in rows],
                    "options": [text[c] for c in order], "gold": order.index(winners[0]),
                    "version": rows[0]["metadata.version"], "panel": len(rows)})
        stats["kept"] += 1
    return out, stats


def _mnli(S) -> list[dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(S.verified_path("mnli", "data/train-00000-of-00001.parquet"))
    keyed = []
    for b in pf.iter_batches(batch_size=20000, columns=["pairID", "premise", "hypothesis", "genre", "label"]):
        for r in b.to_pylist():
            if r["label"] in (0, 1, 2):
                keyed.append((_key("b1v3-mnli", SEED, r["pairID"]), r))
        keyed.sort(key=lambda x: x[0])
        del keyed[MNLI_KEEP:]
    return [{"id": f"m2:mnli:{r['pairID']}", "premise": r["premise"], "hypothesis": r["hypothesis"],
             "genre": r["genre"], "gold": "entailment" if r["label"] == 0 else "not_entailment",
             "mnli_label": ["entailment", "neutral", "contradiction"][r["label"]]} for _, r in keyed]


def convert(out: Path | None = None) -> dict:
    from . import sources as S

    out = Path(out or config.BUILD_DIR / SRC_DIR_NAME)
    out.mkdir(parents=True, exist_ok=True)
    lock = json.loads(S.LOCK.read_text(encoding="utf-8"))
    man = {"inputs": {}, "outputs": {}, "notes": {}}
    for fam in FAMS:
        for f in S.files(S.REGISTRY[fam]):
            man["inputs"][f"{fam}/{f}"] = lock[f"{fam}/{f}"]["sha256"]  # verified_path checks the bytes
        res = {"acos": _acos, "nli4ct": _nli4ct, "vast": _vast, "ragtruth": _ragtruth, "hover": _hover,
               "habermas": _habermas, "mnli": _mnli}[fam](S)
        if isinstance(res, tuple):
            res, man["notes"][fam] = res
        res.sort(key=lambda r: r["id"])
        name = f"{fam}.jsonl.gz"
        with gzip.GzipFile(out / name, "wb", mtime=0) as fh:
            fh.write(("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in res) + "\n").encode("utf-8"))
        man["outputs"][name] = {"sha256": _sha(out / name), "rows": len(res)}
        print(fam, len(res), flush=True)
    man["notes"]["mnli"] = f"seeded subset of {MNLI_KEEP} train pairs (sha256 order, seed {SEED})"
    (out / "convert-manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    return man


# ------------------------------------------------------------------ records (Index presentation)


def _rec(r: dict, fam: str, source_family: str, state, questions: dict, labels: dict, extra: dict | None = None):
    return {"schema": "ssdm-train-v0", "id": f"{r['id']}~view:v3", "family": source_family, "source": fam,
            "source_id": r["id"].split(":", 2)[2], "source_split": "train", "license": LICENSE[fam], "state": state,
            "questions": questions, "labels": labels, "untrusted_fields": [],
            "aug": {"b1_view": {"kind": "v3", "parent_id": r["id"], **(extra or {})}}}


def acos_view(r: dict, seed: int) -> dict:
    inv = _inventory()[r["ds"]]
    full = [(c, s) for c in inv for s in ("0", "1", "2")]
    groups = [full[i:i + ACOS_GROUP] for i in range(0, len(full), ACOS_GROUP)]
    g = int(_key("b1v3-acos-group", seed, r["id"])[:8], 16) % len(groups)
    present = {tuple(p) for p in r["pairs"]}
    qs, labs = {}, {}
    for c, s in groups[g]:
        qid = f"{c}__sentiment_{ACOS_SENT[s]}"
        qs[qid] = {"type": "choice", "instructions": ACOS_INSTR.format(cat=c, sent=ACOS_SENT[s]), "criteria": dict(ACOS_CRIT)}
        labs[qid] = "yes" if (c, s) in present else "no"
    # the Index state also carries task="ACOS fixed category/sentiment presence"; that label alone trips the
    # fingerprint's minhash / edit matchers against catalog 38, so the training row carries the review only
    return _rec(r, "acos", "acos_category_sentiment", {"review": r["review"]}, qs, labs,
                {"inventory_group": g + 1, "inventory_group_count": len(groups)})


def nli4ct_view(r: dict, seed: int) -> dict:
    return _rec(r, "nli4ct", "clinical_trial_nli", r["state"],
                {"answer": {"type": "choice", "instructions": NLI4CT_INSTR + r["statement"], "criteria": dict(NLI4CT_CRIT)}},
                {"answer": r["gold"]}, {"type": r["type"]})


def vast_view(r: dict, seed: int) -> dict:
    return _rec(r, "vast", "stance", {}, {"q1": {"type": "choice", "instructions": VAST_INSTR.format(topic=r["topic"],
                post=r["post"]), "criteria": dict(VAST_CRIT)}}, {"q1": r["gold"]}, {"post_key": r["post_key"]})


def ragtruth_view(r: dict, seed: int) -> dict:
    return _rec(r, "ragtruth", "hallucination_detection", {"prompt": r["prompt"], "response": r["response"]},
                {"q": {"type": "noul", "instructions": RAG_Q, "criteria": dict(RAG_CRIT)}}, {"q": r["gold"]},
                {"task": r["task"], "rag_source_id": r["source_id"]})


def hover_view(r: dict, seed: int) -> dict:
    return _rec(r, "hover", "claim_verification", {"claim": r["claim"], "evidence": r["evidence"]},
                {"q": {"type": "choice", "instructions": HOVER_Q, "criteria": dict(HOVER_CRIT)}}, {"q": r["gold"]},
                {"num_hops": r["hops"]})


def habermas_view(r: dict, seed: int) -> dict:
    crit = {f"option_{i}": t for i, t in enumerate(r["options"])}
    return _rec(r, "habermas", "consensus_selection",
                {"policy_question": r["question"], "participant_opinions": r["opinions"]},
                {"consensus": {"type": "choice", "instructions": HAB_INSTR, "criteria": crit}},
                {"consensus": f"option_{r['gold']}"}, {"panel": r["panel"], "cohort": r["version"]})


def mnli_view(r: dict, seed: int) -> dict:
    return _rec(r, "mnli", "nli", {}, {"decision": {"type": "choice", "instructions": MNLI_INSTR.format(
                p=r["premise"], h=r["hypothesis"]), "criteria": dict(MNLI_CRIT)}}, {"decision": r["gold"]},
                {"mnli_label": r["mnli_label"], "genre": r["genre"]})


VIEWS = {"acos": acos_view, "nli4ct": nli4ct_view, "vast": vast_view, "ragtruth": ragtruth_view, "hover": hover_view,
         "habermas": habermas_view, "mnli": mnli_view}


def gold_rate(rec: dict) -> dict:
    """Positive-label counts of one record (for the base-rate report)."""
    out = {}
    for qid, lab in rec["labels"].items():
        out[str(lab)] = out.get(str(lab), 0) + 1
    return out


def _read_rows(src: Path, fam: str, man: dict):
    name = f"{fam}.jsonl.gz"
    if _sha(src / name) != man["outputs"][name]["sha256"]:
        raise RuntimeError(f"{name}: drifted from convert-manifest.json")
    with gzip.open(src / name, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ------------------------------------------------------------------ overlap vs the family's Index eval rows


def _index_rows(fam_name: str):
    for p in (config.SUITE_DIR / "selected-rows.jsonl.gz", config.SUITE_DIR / "added-rows.jsonl.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                if fam_name in line[:400]:
                    r = json.loads(line)
                    if r.get("family") == fam_name:
                        yield r


def _norm(s: str) -> str:
    import re
    import unicodedata
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))


def _id_keys(fam: str, rows) -> dict[str, set]:
    """Identity keys of the family's Index eval rows (source ids and item texts)."""
    k: dict[str, set] = {"source_id": set(), "item_text": set(), "group": set()}
    for r in rows:
        md, pv = r.get("metadata") or {}, r.get("provenance") or {}
        sid = md.get("source_id") or pv.get("source_id")
        if sid:
            k["source_id"].add(str(sid))
        st = r["state"]
        if fam == "acos":
            k["item_text"].add(_norm(st["review"]))
        elif fam == "nli4ct":
            k["item_text"].add(_norm(r["questions"]["answer"]["instructions"][len(NLI4CT_INSTR):]))
        elif fam == "vast":
            ins = r["questions"]["q1"]["instructions"]
            k["item_text"].add(_norm(ins.split("\nPost: ", 1)[1].rsplit("\nDetermine", 1)[0]))
        elif fam == "ragtruth":
            k["item_text"].add(_norm(st["response"]))
            k["group"].add(str(md.get("source_task_id")))
        elif fam == "hover":
            k["item_text"].add(_norm(st["claim"]))
            k["group"].add(str(md.get("hpqa_id")))
        elif fam == "habermas":
            k["source_id"].add(str(md.get("group_id")))
            k["group"].add(_norm(st["policy_question"]))
            k["item_text"].add(_norm(st["policy_question"]))
    return k


def _row_keys(fam: str, r: dict) -> dict[str, str]:
    sid = r["id"].split(":", 2)[2]
    if fam == "acos":
        return {"item_text": _norm(r["review"])}
    if fam == "nli4ct":
        return {"source_id": sid, "item_text": _norm(r["statement"])}
    if fam == "vast":
        return {"source_id": sid, "item_text": _norm(r["post"])}
    if fam == "ragtruth":
        return {"item_text": _norm(r["response"]), "group": r["source_id"]}
    if fam == "hover":
        return {"source_id": sid, "item_text": _norm(r["claim"]), "group": r["hpqa_id"]}
    if fam == "habermas":
        return {"source_id": sid, "item_text": _norm(r["question"]), "group": _norm(r["question"])}
    return {}


def item_payload(fam: str, r: dict) -> dict:
    """The ITEM-tier text of a converted row (what identifies the item; trial sections, evidence documents, prompts
    and the shared instruction / label strings are context or presentation)."""
    if fam == "acos":
        return {"review": r["review"]}
    if fam == "nli4ct":
        return {"statement": r["statement"]}
    if fam == "vast":
        return {"topic": r["topic"], "post": r["post"]}
    if fam == "ragtruth":
        return {"response": r["response"]}
    if fam == "hover":
        return {"claim": r["claim"]}
    if fam == "habermas":
        return {"question": r["question"], "opinions": r["opinions"], "options": r["options"]}
    return {"premise": r["premise"], "hypothesis": r["hypothesis"]}


def overlap(src: Path | None = None, out: Path | None = None) -> dict:
    """Every converted row vs its family's Index eval rows: identity (source id, normalized item text; shared groups
    such as a HotpotQA id or a RAGTruth source task are reported) and the Index fingerprint v3 (exact / ngram13 /
    minhash / edit / contain) on the row's ITEM-tier text, per matcher for that family's catalog and any other
    catalog. Rows with any item hit are dropped (``drop``); a family whose item-overlap rate with its own catalog
    exceeds SYSTEMIC is excluded whole (the eval would have been drawn from the train pool). Full-record hits
    (context and presentation text) are counted here and dropped later by the draw's own Filter."""
    from .dedup import Fingerprint

    SYSTEMIC = 0.5
    src = Path(src or config.BUILD_DIR / SRC_DIR_NAME)
    man = json.loads((src / "convert-manifest.json").read_text(encoding="utf-8"))
    fp = Fingerprint.load(config.FINGERPRINT)
    fp.assert_production()
    rep = {"convert_manifest_sha256": _sha(src / "convert-manifest.json"),
           "fingerprint": {k: fp.meta.get(k) for k in ("version", "npz_sha256", "arrays_sha256", "rows")},
           "systemic_threshold": SYSTEMIC, "families": {}}
    for fam in FAMS:
        name, cat = INDEX_FAMILY[fam]
        keys = _id_keys(fam, list(_index_rows(name))) if name else {}
        n_eval = None
        if name:
            n_eval = sum(1 for _ in _index_rows(name))
        c = {"rows": 0, "id_source_id": 0, "id_item_text": 0, "shared_group_reported": 0,
             "fp_family_catalog": {m: 0 for m in ("exact", "ngram13", "minhash", "edit", "contain")},
             "fp_family_catalog_any": 0, "fp_other_catalog_any": 0, "full_record_fp_any_reported": 0}
        drop, examples = [], []
        for r in _read_rows(src, fam, man):
            c["rows"] += 1
            rec = VIEWS[fam](r, SEED)
            hit = False
            rk = _row_keys(fam, r)
            for k in ("source_id", "item_text"):
                if k in rk and rk[k] in keys.get(k, ()):
                    c[f"id_{k}"] += 1
                    hit = True
            if "group" in rk and rk["group"] in keys.get("group", ()):
                c["shared_group_reported"] += 1
            res = fp.check(item_payload(fam, r), {})
            full = fp.check(rec["state"], rec["questions"])
            if any(full.values()):
                c["full_record_fp_any_reported"] += 1  # context / presentation hits: the draw Filter drops these too
            fam_hit = False
            for m, cats in res.items():
                if cat is not None and cat in cats:
                    c["fp_family_catalog"][m] += 1
                    fam_hit = True
            other = any(x != cat for cats in res.values() for x in cats)
            c["fp_family_catalog_any"] += fam_hit
            c["fp_other_catalog_any"] += other
            if hit or fam_hit or other:
                drop.append(r["id"])
                if len(examples) < 5:
                    examples.append({"id": r["id"], "fp": {m: v for m, v in res.items() if v}, "id_hit": hit})
        c["dropped"] = len(drop)
        c["drop_rate"] = round(len(drop) / max(1, c["rows"]), 4)
        fam_rate = (c["id_source_id"] + c["id_item_text"] + c["fp_family_catalog_any"]) / max(1, c["rows"])
        c["family_overlap_rate_upper"] = round(fam_rate, 4)
        c["systemic"] = fam_rate > SYSTEMIC
        c["index_eval_rows"] = n_eval
        c["examples"] = examples
        rep["families"][fam] = {**c, "drop": sorted(drop)}
        print(fam, {k: v for k, v in c.items() if k != "examples"}, flush=True)
    out = Path(out or src / "overlap.json")
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rep


# ------------------------------------------------------------------ draw-time loader


class Raw:
    """Converted rows of the v3 families (sha-verified), overlap drops removed; ``order(fam, seed)`` gives the seeded
    candidate order (VAST: one row per post), ``record(fam, id, seed)`` builds the record."""

    def __init__(self, build_dir: Path | None = None, fams: tuple[str, ...] = FAMS):
        src = Path(build_dir or config.BUILD_DIR) / SRC_DIR_NAME
        man = json.loads((src / "convert-manifest.json").read_text(encoding="utf-8"))
        from . import sources as S
        lock = json.loads(S.LOCK.read_text(encoding="utf-8"))
        for k, sha in man["inputs"].items():
            if lock[k]["sha256"] != sha:
                raise RuntimeError(f"{k}: convert input does not match sources.lock.json")
        ov = json.loads((src / "overlap.json").read_text(encoding="utf-8"))
        if ov["convert_manifest_sha256"] != _sha(src / "convert-manifest.json"):
            raise RuntimeError("overlap.json was computed on another conversion")
        self.rows: dict[str, dict[str, dict]] = {}
        self.excluded = {}
        for fam in fams:
            f = ov["families"][fam]
            if f["systemic"]:
                self.excluded[fam] = "systemic overlap with the Index eval rows"
                self.rows[fam] = {}
                continue
            drop = set(f["drop"])
            self.rows[fam] = {r["id"]: r for r in _read_rows(src, fam, man) if r["id"] not in drop}
        self.ident = {"convert_manifest_sha256": _sha(src / "convert-manifest.json"),
                      "overlap_sha256": _sha(src / "overlap.json"), "convert": man, "excluded": self.excluded}

    def order(self, fam: str, seed: int) -> list[str]:
        ids = sorted(self.rows.get(fam, {}), key=lambda i: _key("b1", seed, fam, i))
        if fam == "vast":  # one row per post (posts carry several topics)
            seen, out = set(), []
            for i in ids:
                pk = self.rows[fam][i]["post_key"]
                if pk not in seen:
                    seen.add(pk)
                    out.append(i)
            return out
        return ids

    def record(self, fam: str, rid: str, seed: int) -> dict:
        return VIEWS[fam](self.rows[fam][rid], seed)


def check_specs() -> dict:
    from train.cohort_v3 import _ensure_kit
    _ensure_kit()
    from harness import heldout

    out = {}
    for k, s in GEN_SPECS.items():
        heldout.check_generator(s)
        out[k] = s
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="data.b1_v3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("convert")
    sub.add_parser("overlap")
    c = sub.add_parser("specs")
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        res = convert()
    elif a.cmd == "overlap":
        res = {f: {k: v for k, v in x.items() if k != "drop"} for f, x in overlap()["families"].items()}
    else:
        res = check_specs()
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False)[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
