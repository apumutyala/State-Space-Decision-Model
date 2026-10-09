"""bschema-draw-v4 family rows: Index families the gap attribution marks as missing from the trained draws
(claude/b-gap-funding @5fc2b76 analysis/gap_funding/attribution.json, class "data"; DATA_LEVERS_2026-10-09.md on
claude/b-gap-breakdown @47222a0) that bschema-draw-v3b still lacks, built from public TRAIN splits (or public non-eval
corpora of the same task shape) and rendered the way the Index presents the family. Merged into bschema-draw-v3b by
``scripts/b_schema_swap_pool.py`` (P3 checks) after the token check and ``scripts/b_schema_levers_finalize.py``.

Families (each row carries ``aug.lever``):
  esci_relevance   Amazon ESCI (Index 37) official ``train`` split (tasksource/esci @8113b17 shard 0, US locale),
                   pairs whose product_id / query appear in no Index ESCI row (``scripts/b_schema_v4_export.py``).
                   Index presentation: state {search_query, product{title, description, bullet_point, brand, color}},
                   question ``answer``: E / S / C / I with the Index criteria. Natural train label rates.
  nli4ct_trial     NLI4CT-2024 (Index 42) ``training_data.zip:train.json`` statements (the b1-v3-src conversion,
                   overlap.json drops applied) whose primary AND secondary trial appear in NO Index NLI4CT row (the
                   coordinator's 2026-10-09 "eval documents are test" rule: no Index trial document is trained on).
                   Index presentation (data/b1_v3.py nli4ct_view). Natural train label rates.
  phish_email      PhishNChips (Index 56) has no train split; the same task shape from a public non-eval corpus:
                   Phishing Email Curated Datasets (Zenodo 8339691, CC-BY-4.0; CEAS-08, Nazario, SpamAssassin files),
                   emails whose body carries an http(s) link. Index presentation: state {sender, from, subject, body,
                   link_display_text, link_url}; questions verdict (phishing / legitimate), is_phishing (noul),
                   verdict_alt_click (do_not_click / click), all from the corpus label (1 = phishing). The Index's
                   unlabelled sig_* signals and the null-criteria verdict_alt_minimal are not built. Natural corpus
                   rate over the eligible union.
Per-family row target T = the median row count of the parent's Index-presentation train-split families (acos, vast,
habermas, ragtruth, hover in bschema-draw-v3b): a TRAINING-draw quantity, never an eval rate. Each family is capped
by its train availability. Half the rows keep the Index presentation (Index option ids and question names), half get
a schema view (option ids re-styled so the gold lands at a uniform canonical position, instruction frame, question
names), as bschema-draw-v3b's BoolQ lever (review N4).

    python -X utf8 -I -m data.b_schema_levers_v4 --parent <v3b records.jsonl.gz> --src-dir <bschema-levers-v4-src>
        --out <work dir>
"""
from __future__ import annotations

import argparse
import collections
import csv
import email.utils
import gzip
import hashlib
import html
import json
import math
import random
import re
import statistics
from pathlib import Path

from . import config
from .b_schema import canonical, paraphrase, relabel, rename_questions
from .b_schema_levers import BLOCKED, _jsonl_gz, _sha_file, _take, parent_stats

SEED = 20261010
TARGET_FAMILIES = ("acos/acos_category_sentiment", "vast/stance", "habermas/consensus_selection",
                   "ragtruth/hallucination_detection", "hover/claim_verification")
VIEW_STYLES = ("letters", "letters_lower", "digits", "digits1", "option_k")
GEN_MARGIN = 1.6  # generated / target: room for the item fingerprint, token and gate drops; trimmed in finalize
GEN_SPECS = {
    "esci_relevance": {"name": "b4_esci", "task_family": "product_relevance", "targets": [37]},
    "nli4ct_trial": {"name": "b4_nli4ct", "task_family": "clinical_trial_nli", "targets": [42]},
    "phish_email": {"name": "b4_phish_email", "task_family": "phishing_decision", "targets": [56]},
}

# --------------------------------------------------------------------------- Index presentation strings (verbatim)
ESCI_INSTR = "Classify the relevance of this product to the search query using the ESCI categories."
ESCI_CRIT = {"E": "Exact: the product satisfies the search query.",
             "S": "Substitute: a product that could substitute for the requested product.",
             "C": "Complement: a product that complements the requested product.",
             "I": "Irrelevant: the product does not address the requested product need."}
PHISH_Q = {
    "verdict": {"type": "choice", "instructions": "This email arrived in the user's inbox and contains a link (`link_url`). "
                "Is it a phishing attempt, or a legitimate message whose link is safe to click?",
                "criteria": {"phishing": "The email or its link is a phishing attempt or otherwise malicious. The user "
                             "should not click the link.",
                             "legitimate": "A normal, legitimate message. The link is safe to click."}},
    "is_phishing": {"type": "noul", "instructions": "This email is a phishing attempt and the user should not click the "
                    "link in `link_url`.",
                    "criteria": {"true": "The email or its link is a phishing attempt or otherwise malicious.",
                                 "false": "The email is a normal, legitimate message and the link is safe to click."}},
    "verdict_alt_click": {"type": "choice", "instructions": "Should the user click the link in this email?",
                          "criteria": {"do_not_click": "No. The link may be phishing or malicious.",
                                       "click": "Yes. The email and its link look legitimate."}},
}
PHISH_FILES = ("CEAS_08.csv", "Nazario.csv", "SpamAssasin.csv")
PHISH_LIC = "CC-BY-4.0 (Phishing Email Curated Datasets, Champa, Rabbi & Zibran 2024, Zenodo 10.5281/zenodo.8339691)"
ESCI_LIC = "Apache-2.0 (amazon-science/esci-data; tasksource/esci mirror)"
NLI4CT_LIC = "none stated (SemEval-2024 T2 research release)"


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def view_multi(rec: dict, rng: random.Random) -> dict:
    """Schema view of a (multi-question) record: every choice question's option ids re-styled (gold at a uniform
    canonical position), instruction frames, question names; noul questions keep their type."""
    nq, nl, ops = {}, {}, []
    for qid, q in rec["questions"].items():
        g = rec["labels"][qid]
        if q["type"] == "choice":
            style = rng.choice(VIEW_STYLES)
            q, g = relabel(q, g, rng, style)
            ops.append(f"{qid}:ids:{style}")
        if rng.random() < 0.75:
            ins, op = paraphrase(q.get("instructions"), q["type"], rng)
            if op:
                q = {**q, "instructions": ins}
                ops.append(f"{qid}:{op}")
        nq[qid], nl[qid] = q, g
    if rng.random() < 0.6:
        ren = rename_questions(list(nq), rng)
        nq = {ren[k]: v for k, v in nq.items()}
        nl = {ren[k]: v for k, v in nl.items()}
        ops.append("qid_rename")
    rec["questions"], rec["labels"] = nq, nl
    rec["aug"]["schema_synth"] = {"ops": ops, "seed": SEED, "lever_view": True}
    return rec


def _rec(src: str, src_id: str, tag: str, family: str, lic: str, state: dict, questions: dict, labels: dict,
         lever: dict, split: str = "train") -> dict:
    base = f"m2:{src}:{src_id}"
    return {"schema": "ssdm-train-v0", "id": f"{base}~view:{tag}", "source": src, "source_id": src_id,
            "source_split": split, "family": family, "license": lic,
            "provenance": "train-split" if split == "train" else "public-corpus", "state": state,
            "questions": questions, "labels": labels, "untrusted_fields": [],
            "aug": {"b1_view": {"kind": tag, "parent_id": base, "base_id": base}, "lever": lever}}


def _present(rec: dict, rng: random.Random, view: bool) -> dict:
    if view:
        return view_multi(rec, rng)
    rec["aug"]["schema_synth"] = {"ops": ["index_presentation"], "seed": SEED, "lever_view": False}
    return rec


# ------------------------------------------------------------------------------------------------------ ESCI
def esci(src_dir: Path, reuse, prov, n_gen: int, target: int, log=print) -> list[dict]:
    man = json.loads((src_dir / "export-manifest.json").read_text(encoding="utf-8"))
    p = src_dir / "esci.jsonl.gz"
    if _sha_file(p) != man["esci.jsonl.gz_sha256"]:
        raise SystemExit("esci export sha drift")
    rows = sorted(_jsonl_gz(p), key=lambda r: _h(SEED, "esci", r["example_id"]))
    out, stats = [], collections.Counter()
    for r in rows:
        if len(out) >= n_gen:
            break
        sid = f"train-{r['example_id']}"
        if not _take(reuse, "esci", sid):
            stats["held_out_or_reuse_cap"] += 1
            continue
        rng = random.Random(_h(SEED, "esci", r["example_id"]))
        view = rng.random() < 0.5
        state = {"search_query": r["query"], "product": dict(r["product"])}
        q = {"type": "choice", "instructions": ESCI_INSTR, "criteria": dict(ESCI_CRIT)}
        rec = _rec("esci", sid, "v4s" if view else "v4", "product_relevance", ESCI_LIC, state, {"answer": q},
                   {"answer": r["gold"]}, {"lever": "esci_relevance", "label": r["gold"], "query_id": r["query_id"],
                                           "product_id": r["product_id"], "view": view})
        out.append(_present(rec, rng, view))
        stats[f"label_{r['gold']}"] += 1
    stats["records"] = len(out)
    n_lab = man["stats"]
    tr = {k: n_lab.get(f"eligible_{k}", 0) for k in "ESCI"}
    tot = sum(tr.values())
    prov["esci_relevance"] = {
        "file": man["input"]["file"], "sha256": man["input"]["sha256"], "repo": man["input"]["repo"],
        "revision": man["input"]["revision"], "split": "train", "export": man["rule"],
        "export_stats": man["stats"], "train_eligible_label_rate": {k: round(v / tot, 4) for k, v in tr.items()},
        "n_target": target, "n_generated_rule": f"ceil({GEN_MARGIN} * target), trimmed to the target after the checks",
        "rule": "uniform over eligible train pairs (sha256 order); natural train label rates; half Index presentation "
                "(ids E/S/C/I, question 'answer'), half schema view",
        "built": dict(stats)}
    log("esci_relevance", dict(stats))
    return out


# ---------------------------------------------------------------------------------------------------- NLI4CT
def nli4ct(reuse, prov, target: int, log=print) -> list[dict]:
    from .b1_v3 import NLI4CT_CRIT, NLI4CT_INSTR, Raw

    raw = Raw(fams=("nli4ct",))
    index_trials = set()
    for f in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        with gzip.open(config.SUITE_DIR / f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if "NLI4CT" not in line[:300]:
                    continue
                r = json.loads(line)
                if r.get("family") == "NLI4CT-2024":
                    index_trials |= set(r["metadata"]["trial_ids"])
    stats = collections.Counter()
    items = []
    for rid in sorted(raw.rows["nli4ct"]):
        r = raw.rows["nli4ct"][rid]
        st = r["state"]
        trials = {st["primary_trial"]["id"]} | ({st["secondary_trial"]["id"]} if "secondary_trial" in st else set())
        if trials & index_trials:
            stats["index_trial"] += 1
            continue
        items.append(r)
    stats["eligible_items"] = len(items)
    views = 2 if 2 * len(items) <= target else 1
    out = []
    for r in sorted(items, key=lambda r: _h(SEED, "nli4ct", r["id"])):
        sid = r["id"].split(":", 2)[2]
        for v in range(views):
            if len(out) >= target:
                break
            if not _take(reuse, "nli4ct", sid):
                stats["held_out_or_reuse_cap"] += 1
                continue
            rng = random.Random(_h(SEED, "nli4ct", r["id"], v))
            view = v == 1 if views == 2 else rng.random() < 0.5
            state = json.loads(json.dumps(r["state"]))
            if view:
                # schema view: the statement moves into state (key-free state+options text is per statement; several
                # train statements share one trial section, and the P3 checker refuses state+options text repeated
                # across items), the instruction keeps the Index wording without the statement
                stmt = {"statement": r["statement"]}
                state = {**stmt, **state} if rng.random() < 0.5 else {**state, **stmt}
                q = {"type": "choice", "instructions": NLI4CT_INSTR.rstrip(":\n") + ".","criteria": dict(NLI4CT_CRIT)}
            else:
                q = {"type": "choice", "instructions": NLI4CT_INSTR + r["statement"], "criteria": dict(NLI4CT_CRIT)}
            rec = _rec("nli4ct", sid, "v4s" if view else "v4", "clinical_trial_nli", NLI4CT_LIC,
                       state, {"answer": q}, {"answer": r["gold"]},
                       {"lever": "nli4ct_trial", "label": r["gold"], "type": r["type"], "view": view,
                        "item": {"statement": r["statement"]}})
            out.append(_present(rec, rng, view))
            stats[f"label_{r['gold']}"] += 1
    stats["records"] = len(out)
    tr = collections.Counter(r["gold"] for r in raw.rows["nli4ct"].values())
    prov["nli4ct_trial"] = {
        "file": "dl-2026-10-08/nli4ct/training_data.zip:train.json (via ssdm-data/m2/build/b1-v3-src/nli4ct.jsonl.gz)",
        "convert_manifest_sha256": raw.ident["convert_manifest_sha256"], "overlap_sha256": raw.ident["overlap_sha256"],
        "split": "train", "n_train_after_overlap": len(raw.rows["nli4ct"]),
        "train_label_rate": {k: round(v / sum(tr.values()), 4) for k, v in tr.items()},
        "index_trial_ids": len(index_trials),
        "rule": "train statements whose primary and secondary trial are in no Index NLI4CT row (all 5,500 test "
                "statements' trial_ids); natural train label rate; views per item = 2 when 2 x eligible <= target "
                "(view 0 Index presentation, view 1 schema view with the statement in state), else 1; the P3 checker keeps "
                "one Index-presentation row per trial section (state + options text repeats across statements)",
        "views_per_item": views, "n_target": target,
        "n_target_rule": "min(T, views x eligible items): train availability caps the family", "built": dict(stats)}
    log("nli4ct_trial", dict(stats))
    return out


# ----------------------------------------------------------------------------------------------- phishing email
_URL = re.compile(r"https?://[^\s<>\"'()\[\]{}]+", re.I)
_HREF = re.compile(r"<a\b[^>]*?href\s*=\s*[\"']?(https?://[^\"'\s>]+)[^>]*>(.*?)</a\s*>", re.I | re.S)
_HTMLISH = re.compile(r"<\s*(html|body|div|p|br|table|td|a|font|span)\b", re.I)


def _clean_body(b: str) -> tuple[str, str | None, str | None]:
    """(text, link_url, link_display_text). HTML is reduced to text; the first anchor (or the first bare URL) is the
    link; plain-text emails show the URL itself."""
    b = b or ""
    url = disp = None
    if _HTMLISH.search(b):
        m = _HREF.search(b)
        if m:
            url = m.group(1)
            disp = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", m.group(2))).split()) or None
        b = re.sub(r"(?is)<(script|style)\b.*?</\1\s*>", " ", b)
        b = re.sub(r"(?i)<\s*(br|/p|/div|/tr|/li)\b[^>]*>", "\n", b)
        b = re.sub(r"(?i)<a\b[^>]*?href\s*=\s*[\"']?(https?://[^\"'\s>]+)[^>]*>", r" \1 ", b)
        b = html.unescape(re.sub(r"<[^>]+>", " ", b))
    b = b.replace("\r\n", "\n").replace("\r", "\n")
    b = "\n".join(" ".join(line.split()) for line in b.split("\n"))
    b = re.sub(r"\n{3,}", "\n\n", b).strip()
    if url is None:
        m = _URL.search(b)
        if m:
            url = m.group(0).rstrip(".,;:!?")
    if url is not None and url not in b:
        url = None  # the link must be visible in the cleaned body (as in the Index rows)
    return b, url, (disp or url)


def phish(src_dir: Path, lock: dict, reuse, prov, n_gen: int, target: int, log=print) -> list[dict]:
    csv.field_size_limit(1 << 30)
    stats = collections.Counter()
    cand = []
    seen = set()
    shas = {}
    for f in PHISH_FILES:
        p = src_dir / f
        shas[f] = _sha_file(p)
        if shas[f] != lock[f"phish_curated/{f}"]["sha256"]:
            raise SystemExit(f"{f}: sha drift vs sources.lock.json")
        stem = f.rsplit(".", 1)[0].lower()
        with open(p, encoding="utf-8", errors="replace", newline="") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                stats[f"{stem}_rows"] += 1
                lab = str(r.get("label", "")).strip()
                if lab not in ("0", "1"):
                    stats["bad_label"] += 1
                    continue
                name, addr = email.utils.parseaddr(r.get("sender") or "")
                if "@" not in addr:
                    stats["no_sender_address"] += 1
                    continue
                body, url, disp = _clean_body(r.get("body") or "")
                if url is None:
                    stats["no_link"] += 1
                    continue
                if not (40 <= len(body) <= 3000):
                    stats["body_len"] += 1
                    continue
                key = _h(" ".join(re.findall(r"\w+", body.casefold())))
                if key in seen:
                    stats["dup_body"] += 1
                    continue
                seen.add(key)
                subj = " ".join(str(r.get("subject") or "").split())
                cand.append({"sid": f"{stem}-{i}", "label": "phishing" if lab == "1" else "legitimate",
                             "state": {"sender": name.strip() or addr.split("@")[0], "from": addr, "subject": subj,
                                       "body": body, "link_display_text": disp, "link_url": url}, "file": f})
                stats[f"eligible_{stem}_{lab}"] += 1
    n_ph = sum(1 for c in cand if c["label"] == "phishing")
    out = []
    for c in sorted(cand, key=lambda c: _h(SEED, "phish", c["sid"])):
        if len(out) >= n_gen:
            break
        if not _take(reuse, "phish_curated", c["sid"]):
            stats["held_out_or_reuse_cap"] += 1
            continue
        rng = random.Random(_h(SEED, "phish", c["sid"]))
        view = rng.random() < 0.5
        ph = c["label"] == "phishing"
        qs = json.loads(json.dumps(PHISH_Q))
        labs = {"verdict": c["label"], "is_phishing": "true" if ph else "false",
                "verdict_alt_click": "do_not_click" if ph else "click"}
        rec = _rec("phish_curated", c["sid"], "v4s" if view else "v4", "phishing_decision", PHISH_LIC, c["state"], qs,
                   labs, {"lever": "phish_email", "label": c["label"], "file": c["file"], "view": view},
                   split="corpus")
        out.append(_present(rec, rng, view))
        stats[f"built_{c['label']}"] += 1
    stats["records"] = len(out)
    prov["phish_email"] = {
        "files": {f"dl-2026-10-09/phish_curated/{f}": shas[f] for f in PHISH_FILES},
        "repo": "kudzaiprichard/aura-phishing-email-corpus", "revision": "539567cee21d03c83bd2fb1b142c18a80904eb72",
        "origin": "Zenodo 10.5281/zenodo.8339691 (byte-identical; MD5s match the Zenodo API)", "license": PHISH_LIC,
        "split": "corpus (no train/test split; not an Index source: PhishNChips Index rows are generated emails with "
                 "links from a live phishing-URL database)",
        "eligible_rule": "sender has an address; body (HTML reduced to text) 40-3,000 chars and shows an http(s) link; "
                         "exact normalised-body duplicates dropped",
        "n_eligible": len(cand), "eligible_phishing_rate": round(n_ph / max(1, len(cand)), 4),
        "n_target": target, "n_generated_rule": f"ceil({GEN_MARGIN} * target), trimmed to the target after the checks",
        "rule": "uniform over the eligible union (sha256 order): natural corpus rate; questions verdict / is_phishing / "
                "verdict_alt_click from the corpus label; half Index presentation, half schema view",
        "built": dict(stats)}
    log("phish_email", dict(stats))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="data.b_schema_levers_v4")
    ap.add_argument("--parent", required=True)
    ap.add_argument("--src-dir", required=True, help="bschema-levers-v4-src (esci.jsonl.gz + export-manifest.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from .b1 import exclusion_sets
    BLOCKED.update(exclusion_sets()["items"])
    ps = parent_stats(Path(a.parent))
    counts = {f: ps["fam"][f] for f in TARGET_FAMILIES}
    target = int(statistics.median(counts.values()))
    lock = json.loads((config.MANIFEST_DIR / "sources.lock.json").read_text(encoding="utf-8"))
    n_gen = math.ceil(GEN_MARGIN * target)
    prov: dict = {"target_rule": {
        "T": target, "rule": "median row count of the parent's Index-presentation train-split families "
                             "(a training-draw quantity; no eval rate or shape)",
        "parent": str(a.parent).replace("\\", "/").split("ssdm-data/")[-1], "parent_rows": ps["n"],
        "family_rows": counts}}
    rows = []
    rows += esci(Path(a.src_dir), ps["reuse"], prov, n_gen=n_gen, target=target)
    rows += nli4ct(ps["reuse"], prov, target=target)
    rows += phish(config.SOURCES_DIR / "dl-2026-10-09" / "phish_curated", lock, ps["reuse"], prov, n_gen=n_gen,
                  target=target)
    with gzip.open(out / "levers.jsonl.gz", "wt", encoding="utf-8") as fo:
        for r in rows:
            fo.write(canonical(r) + "\n")
    (out / "target_provenance.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(rows), "T": target,
                      "by_lever": collections.Counter(r["aug"]["lever"]["lever"] for r in rows)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
