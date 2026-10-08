"""BOARD B1: weighted training draws (45k and 100k) for the RWKV7-G1k-13.3B decision cells.

Extends the PR #44 150k builder (``data.build.build_sample``) instead of replacing it. Stages, all CPU:

1. ``supplement``: the same builder, same seed, same renderer, same guards (cohort-v2 exclusion, Index fingerprint,
   SplitGuard content index, D1 held-out template guard) run with larger quotas on the sources whose 150k quota left
   capacity unused (glaive_fc, banking77, clinc150, gsm8k). Records whose id is already in the 150k set are byte
   identical there (per-item render seeds) and are taken from the 150k set; only new ids are added.
2. ``draw``: per-family quotas (``WEIGHTS_45K`` / ``WEIGHTS_100K``) over
     - native rows of the guarded 150k set + supplement (cohort v3 removed by id / content / source item / text),
     - derived views of native items (BANKING77/CLINC150 16-option shortlists; Glaive tool-catalog expansions),
     - optional EXTERNAL families (``--families-dir``: the B1-families build of ssdm-train-v0 records, e.g. API-Bank /
       BFCL / CRUXEval train-split or generated rows) with their own quotas (``--families-quotas``); every such row
       passes the same ``Filter`` as the native rows. The shipped 45k v1 uses no external family.
   Every source item is used at most once per draw (a view and its native row never co-occur), near-duplicates by
   key-free normalized text are dropped, and option order is reshuffled with a seeded RNG so the gold's canonical
   position is uniform.
3. ``data.b1_check``: an independent exclusion check over every output row (fails loudly on any overlap), then the
   trainer's own ``train.train_set.draw`` + ``heldout_gate`` with the consuming cell config.

Outputs (outside git) use the 150k set's format so ``train.train_set`` reads them unchanged:
``records.jsonl.gz`` + ``<build>.manifest.jsonl.gz`` + ``<build>.content.jsonl.gz``; committed: the pointer
``data/manifests/<build>.pointer.json`` and ``receipts/2026-10-08-B1-draws/``.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator

from . import config

BUILD_SEED = 20261008
PARENT = "sample-150000-seed20261006"
PARENT_POINTER = "data/manifests/sample-150000-seed20261006.pointer.json"
SUPPLEMENT = "b1-supplement-seed20261006"
#: kept-record counts asked of the supplement build (the raw source sizes: each source is drawn until exhausted)
SUPPLEMENT_COUNTS = {"glaive_fc": 45000, "banking77": 10003, "clinc150": 15100, "gsm8k": 7473}
#: the three train.heldout D1-B soft_framing hits of the 150k set (jobs/m2_first_cell config); dropped at build
HELDOUT_150K = ("m2:mmlu_aux:21118", "m2:mmlu_aux:39640", "m2:mmlu_aux:39643")
VIEW_SEP = "~view:"

# Family -> (area group, kind, source). kind: native = an existing guarded record (option order re-drawn so the gold's
# canonical position is uniform; injected records keep their ids because the injected text names an option id);
# k16 = 16-option intent shortlist view; catalog / catalog_long = Glaive tool-catalog expansion with opaque
# (API-Bank-style) ids to K uniform in 4..16 / 17..32 (long: capped by a char budget so the row fits T<=2048); *_full = the native full-catalog intent row (K 77 / 151, as the
# Index presents it); external = a row of the --families-dir build. Source items are split between the families of
# one source by a fixed id hash (``_role``), so a view and a native row of the same item never co-occur.
FAMILIES: dict[str, tuple[str, str, str]] = {
    "glaive_native": ("tools", "native", "glaive_fc"),
    "glaive_catalog16": ("tools", "catalog", "glaive_fc"),
    "glaive_catalog_long": ("tools", "catalog_long", "glaive_fc"),
    "toolace": ("tools", "native", "toolace"),
    "when2call": ("tools", "native", "when2call"),
    "factchange": ("tools", "native", "synthetic_factchange"),
    "gsm8k": ("knowledge_procedural", "native", "gsm8k"),
    "banking77_k16": ("intent_views", "k16", "banking77"),
    "clinc150_k16": ("intent_views", "k16", "clinc150"),
    "banking77_full": ("intent_views", "native", "banking77"),
    "anli": ("language", "native", "anli"),
    "hellaswag": ("language", "native", "hellaswag"),
    "winogrande": ("language", "native", "winogrande"),
    "sst5": ("language", "native", "sst5"),
    "nq": ("retrieval_other", "native", "nq"),
    "boolq": ("retrieval_other", "native", "boolq"),
    "mmlu_aux": ("knowledge_facts", "native", "mmlu_aux"),
    "humicroedit_s1": ("arts", "native", "humicroedit_s1"),
    "humicroedit_s2": ("arts", "native", "humicroedit_s2"),
    # v2 (data/b1_lowrate.py): low-positive-rate yes/no and minority-support views of allowed train sources
    "clinc150_member": ("intent_views", "member", "clinc150"),
    "banking77_member": ("intent_views", "member", "banking77"),
    "anli_claim": ("language", "claim", "anli"),
    "mmlu_verify": ("knowledge_facts", "verify", "mmlu_aux"),
    # v3 (data/b1_v3.py): official TRAIN splits of seven new sources in the eval presentation of the matching family
    "acos": ("language", "v3", "acos"),
    "nli4ct": ("language", "v3", "nli4ct"),
    "vast": ("language", "v3", "vast"),
    "ragtruth": ("language", "v3", "ragtruth"),
    "mnli": ("language", "v3", "mnli"),
    "hover": ("retrieval_other", "v3", "hover"),
    "habermas": ("arts", "v3", "habermas"),
}
#: view kinds built from raw source rows by data.b1_lowrate (no native line needed; Index fingerprint runs per row)
LOWRATE_KINDS = ("member", "claim", "verify")
#: rows built from the v3 conversion of newly fetched train splits (data.b1_v3; Index fingerprint runs per row)
V3_KINDS = ("v3",)
#: Index area of each family group (for the per-area report)
GROUP_AREA = {"tools": "tools", "knowledge_procedural": "knowledge", "knowledge_facts": "knowledge",
              "intent_views": "retrieval", "retrieval_other": "retrieval", "language": "language", "arts": "arts"}

# Re-aimed weights (coordinator 2026-10-08, research-notes/STOCK_RWKV7_LOSS_BREAKDOWN.md): tools first, then
# procedural knowledge, then BANKING77/CLINC150 views, then retrieval; arts and fact recall de-emphasized. v1 uses
# only the existing guarded pool, so procedural = GSM8K train (exhausted at ~4.5k after cohort v3); the missing
# families come from the separate B1-families build via the --families-dir hook. Only rows that the D4 cell's own
# K<=16 / T<=2048 filter keeps are drawn: the D1 trainer (11fb140) drops K>16 rows, so long catalogs would be dead
# weight. SUPERSEDED 2026-10-08: D4 rides on D8 with max_k raised, so >16-option rows (full intent catalogs, long
# tool catalogs, long NQ candidate lists) are drawn with real weight; the T<=2048 filter still applies. NOTE harness/manifests/heldout-draw-v1.json: CLadder (44), MuSR (32), SATA (33), BBH (58), ChessBench (31),
# Home appliances (9), BPoMP (20), POP909 (22), New Yorker (64) are FROZEN held-out families; no training row may
# target them (harness.heldout.check_record runs on every row here).
# Full CLINC150 catalogs (151 options) tokenize to 3,139-5,342 tokens with the cell template (measured on 30 rows of
# the 150k set) and never fit the T<=2048 rung, so they are not drawn (no clinc150_full family) until a cell trains at
# T>=4096-8192. Full BANKING77 catalogs (77 options) fit at ~1,750 tokens with letters/digits ids, so *_full rows use
# those two id styles.
#: external families (the B1-families build, stack/rwkv7/b1-families @a4bc718: API-Bank official train split, BFCL-
#: and CRUXEval-shaped synthetic rows; MuSR / SATA / CLadder refused there as FROZEN). Keys are ``ext:<file stem>``.
EXT_GROUP = {"ext:api_bank": "tools", "ext:bfcl": "tools", "ext:cruxeval": "knowledge_procedural"}
FAMILIES_DIR = "b1-families"
WEIGHTS_45K = {
    "ext:api_bank": 2284, "ext:bfcl": 2400,                                                  # tools (external) 4,684
    "glaive_native": 3500, "glaive_catalog16": 2000, "glaive_catalog_long": 3000, "toolace": 2000,
    "when2call": 2000, "factchange": 500,                                                    # tools 17,684
    "ext:cruxeval": 2400, "gsm8k": 4500,                                                     # procedural 6,900
    "banking77_k16": 3000, "clinc150_k16": 4500, "banking77_full": 2000,                     # intent 9,500
    "anli": 1500, "hellaswag": 1100, "winogrande": 1200, "sst5": 700,                        # language 4,500
    "nq": 1716, "boolq": 1700,                                                               # retrieval 3,416
    "mmlu_aux": 2000,                                                                        # facts 2,000
    "humicroedit_s1": 500, "humicroedit_s2": 500,                                            # arts 1,000
}
WEIGHTS_100K = {
    "ext:api_bank": 2284, "ext:bfcl": 2400,                                                  # tools (external) 4,684
    "glaive_native": 9500, "glaive_catalog16": 8000, "glaive_catalog_long": 8000, "toolace": 4200,
    "when2call": 4500, "factchange": 2800,                                                   # tools 41,684
    "ext:cruxeval": 2400, "gsm8k": 4500,                                                     # procedural 6,900
    "banking77_k16": 4000, "clinc150_k16": 7000, "banking77_full": 2500,                     # intent 13,500
    "anli": 5500, "hellaswag": 4000, "winogrande": 3500, "sst5": 2000,                       # language 15,000
    "nq": 4500, "boolq": 4000,                                                               # retrieval 8,500
    "mmlu_aux": 11416,                                                                       # facts 11,416
    "humicroedit_s1": 1500, "humicroedit_s2": 1500,                                          # arts 3,000
}
#: families that absorb any shortfall (in order): tools first, then fact knowledge
# v2 (coordinator 2026-10-08, after D1 over-called majority-positive labels on ACOS / RAGTruth / NLI4CT and on
# yes/no membership): the same 100k with 10,500 rows moved from glaive_native (-5,000), mmlu_aux (-3,000) and anli
# (-2,500) to low-positive-rate yes/no and minority-support views at their natural base rates (data/b1_lowrate.py).
WEIGHTS_100K_V2 = {**WEIGHTS_100K, "glaive_native": 4500, "mmlu_aux": 8416, "anli": 3000,
                   "clinc150_member": 3000, "banking77_member": 1500, "anli_claim": 3000, "mmlu_verify": 3000}
assert sum(WEIGHTS_100K_V2.values()) == sum(WEIGHTS_100K.values())
# v3 (coordinator 2026-10-08): 100k v2 + ACOS / NLI4CT / VAST / RAGTruth / HoVer / Habermas (eval presentation) and
# MNLI (generic binary NLI, entailment the minority) at natural base rates, plus +1,500 WinoGrande (volume). The 9,500
# rows come from the backfill families, not from tool families: mmlu_aux -5,500, anli -2,000, and 2,000 fewer
# glaive_native BACKFILL rows (nq's quota 4,500 -> 2,500: its pool yields 817 either way, so the deficit the backfill
# covers shrinks by 2,000). Tools stay >= 40% (v2 42.2% -> ~40.2%).
V3_NEW = {"acos": 1000, "nli4ct": 1000, "vast": 1300, "ragtruth": 1400, "hover": 1400, "habermas": 600, "mnli": 1300}
WEIGHTS_100K_V3 = {**WEIGHTS_100K_V2, "mmlu_aux": WEIGHTS_100K_V2["mmlu_aux"] - 5500,
                   "anli": WEIGHTS_100K_V2["anli"] - 2000, "nq": WEIGHTS_100K_V2["nq"] - 2000,
                   "winogrande": WEIGHTS_100K_V2["winogrande"] + 1500, **V3_NEW}
assert sum(WEIGHTS_100K_V3.values()) == sum(WEIGHTS_100K.values())
BACKFILL = ("glaive_native", "mmlu_aux", "anli")
DRAWS = {"b1-d4-draw45k": WEIGHTS_45K, "b1-draw100k": WEIGHTS_100K, "b1-draw100k-v2": WEIGHTS_100K_V2,
         "b1-draw100k-v3": WEIGHTS_100K_V3}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ------------------------------------------------------------------ stage 1: supplement


def supplement(out: Path | None = None) -> dict:
    """Run the 150k builder with the supplement quotas (same seed/guards). Refuses an existing output."""
    from .build import Exclusion, build_sample, content_guard

    out = Path(out or config.BUILD_DIR / SUPPLEMENT)
    if (out / "records.jsonl.gz").exists():
        raise ValueError(f"refusing to overwrite {out}")
    n = sum(SUPPLEMENT_COUNTS.values())
    shares = {k: v / n for k, v in SUPPLEMENT_COUNTS.items()}
    guard, gid = content_guard()
    res = build_sample(n_target=n, out=str(out), receipt=str(out / "receipt"), seed=config.SEED, sources=shares,
                       exclusion=Exclusion.from_cohort(), guard=guard, keep_records=False, manifest_dir=str(out),
                       log=lambda *x: print(*x, flush=True))
    res["summary"]["content_index_guard_identity"] = gid
    (out / "receipt" / "build-summary.json").write_text(json.dumps(res["summary"], indent=1) + "\n", encoding="utf-8")
    return res["summary"]


# ------------------------------------------------------------------ helpers over records


def item_key(rec_id: str) -> str:
    """Source item of a native or derived id (``~view:`` and ``~inj:`` suffixes stripped; multi/factchange keep
    ``data.build.item_keys`` semantics via the caller)."""
    return rec_id.split(VIEW_SEP, 1)[0].split("~inj:", 1)[0]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold().replace("_", " ")))


def _strings(o: Any) -> Iterator[str]:
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)


def single_choice(rec: dict) -> tuple[str, dict] | None:
    if len(rec["questions"]) != 1:
        return None
    qid, q = next(iter(rec["questions"].items()))
    if q.get("type") != "choice" or not isinstance(q.get("criteria"), dict):
        return None
    return qid, q


ID_STYLES = ("option_k", "letters", "digits", "opaque")


def relabel_uniform(rec: dict, qid: str, options: list[tuple[str, Any]], gold_old: str, rng: random.Random,
                    style: str | None = None) -> dict:
    """Rewrite one choice question: options shuffled, gold placed at a uniformly drawn canonical (sorted-id)
    position, ids relabelled in ``style`` (seeded). Returns the new record (deep-copied question/labels)."""
    from decide.schema import canonical_options

    k = len(options)
    style = style or rng.choice(ID_STYLES)
    others = [o for o in options if o[0] != gold_old]
    gold = next(o for o in options if o[0] == gold_old)
    rng.shuffle(others)
    if style == "option_k":
        new_ids = [f"option_{i}" for i in range(k)]
    elif style == "letters":
        new_ids = [_letters(i) for i in range(k)]
    elif style == "digits":
        new_ids = [str(i) for i in range(k)]
    else:
        new_ids = []
        while len(set(new_ids)) < k:
            new_ids = [f"k_{rng.getrandbits(32):08x}" for _ in range(k)]
    probe = {"type": "choice", "criteria": {i: None for i in new_ids}}
    canon = [oid for oid, _ in canonical_options(probe)]  # the renderer's order (sorted ids)
    pos = rng.randrange(k)
    seq = others[:pos] + [gold] + others[pos:]
    crit = {canon[j]: seq[j][1] for j in range(k)}
    out = json.loads(json.dumps(rec))
    q = out["questions"][qid]
    q["criteria"] = crit
    out["labels"][qid] = canon[pos]
    out.setdefault("aug", {})["id_style"] = style
    inj = out["aug"].get("injection")
    if inj and inj.get("question") == qid:
        target_text = dict(options)[inj["target"]]
        inj["target"] = next(i for i, d in crit.items() if d == target_text)
    return out


def _letters(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


# ------------------------------------------------------------------ views


OOS_TEXT = "out of scope: none of the supported intents"


def intent_k16_view(rec: dict, seed: int) -> dict | None:
    """16-option shortlist of a single-question BANKING77/CLINC150 record: gold, CLINC's OOS option, the injection
    target (if any) and 7 hard negatives by token-set Jaccard similarity to the utterance (ties by seeded sha256),
    filled to 16 by a seeded sha256 order; displayed order and ids reshuffled so the gold position is uniform.
    Selection rule ported from ``train.intent16`` (stack/rwkv7/intent16-views)."""
    sc = single_choice(rec)
    if sc is None or rec["family"] not in ("intent_banking", "intent_general"):
        return None
    qid, q = sc
    opts = q["criteria"]
    gold = rec["labels"][qid]
    required = {gold}
    inj = rec.get("aug", {}).get("injection")
    if inj:
        if inj.get("question") != qid or inj.get("target") not in opts:
            return None
        required.add(inj["target"])
    if rec["source"] == "clinc150":
        oos = [o for o, d in opts.items() if d == OOS_TEXT]
        if len(oos) != 1:
            return None
        required.add(oos[0])
    text = " ".join(_strings(rec["state"])) + " " + (q.get("instructions") or "")
    query = _tokens(text)
    key = lambda oid: digest(f"b1-k16:{seed}:{rec['id']}:{opts[oid]}")
    def sim(oid):
        lab = _tokens(opts[oid])
        return len(query & lab) / max(1, len(query | lab))
    rest = set(opts) - required
    hard = sorted(rest, key=lambda o: (-sim(o), key(o)))[:7]
    sel = required | set(hard)
    sel |= set(sorted(set(opts) - sel, key=key)[:16 - len(sel)])
    rng = random.Random(f"b1-k16:{seed}:{rec['id']}")
    options = [(o, opts[o]) for o in sorted(sel, key=key)]
    out = relabel_uniform(rec, qid, options, gold, rng)
    out["id"] = rec["id"] + VIEW_SEP + "k16"
    out["aug"]["b1_view"] = {"kind": "k16", "parent_id": rec["id"], "k": 16}
    return out


NO_TOOL = "do not call a tool now (answer, ask for missing details, or decline)"


def tool_field(state: Any) -> str | None:
    """The state key holding the published tool list (the renderer renames ``tools`` to a synonym)."""
    if not isinstance(state, dict):
        return None
    for key, v in state.items():
        if isinstance(v, list) and v and all(isinstance(t, dict) and isinstance(t.get("name"), str) for t in v):
            return key
    return None


#: char budget of a long-catalog view (measured: 6,400 chars put half the views over T=2,048; the K/T filter decides)
LONG_VIEW_CHARS = 5600


def glaive_catalog_view(rec: dict, donors: list[dict], k: int, seed: int, tag: str,
                        char_budget: int | None = None, min_k: int | None = None) -> dict | None:
    """Glaive tool choice with the published catalog expanded to ``k`` options by tools of OTHER dialogues (names
    and descriptions distinct from every tool of this record), opaque ids (API-Bank-style), gold uniform."""
    sc = single_choice(rec)
    if sc is None or rec["source"] != "glaive_fc" or not isinstance(rec["state"], dict):
        return None
    qid, q = sc
    field = tool_field(rec["state"])
    if field is None:
        return None
    tools = rec["state"][field]
    size = len(canonical({"s": rec["state"], "q": rec["questions"]}))
    names = {t.get("name") for t in tools}
    descs = {t.get("description") for t in tools}
    rng = random.Random(f"b1-cat:{seed}:{tag}:{rec['id']}")
    extra, seen = [], set(names)
    order = list(range(len(donors)))
    rng.shuffle(order)
    need = k - len(q["criteria"])
    if need <= 0:
        return None
    for j in order:
        t = donors[j]
        if t["name"] in seen or t.get("description") in descs:
            continue
        if char_budget is not None and size + len(canonical(t)) + 40 > char_budget:
            if min_k is not None and len(q["criteria"]) + len(extra) >= min_k:
                break
            continue
        seen.add(t["name"])
        extra.append(t)
        size += len(canonical(t)) + 40
        if len(extra) >= need:
            break
    if len(extra) < need and (min_k is None or len(q["criteria"]) + len(extra) < min_k):
        return None
    out = json.loads(json.dumps(rec))
    new_tools = tools + extra
    rng.shuffle(new_tools)
    out["state"][field] = new_tools
    options = list(q["criteria"].items()) + [(f"x{i}", t["name"]) for i, t in enumerate(extra)]
    out2 = relabel_uniform(out, qid, options, rec["labels"][qid], rng, style="opaque")
    out2["id"] = rec["id"] + VIEW_SEP + tag
    out2["aug"]["b1_view"] = {"kind": tag, "parent_id": rec["id"], "k": len(options)}
    return out2


def rebalance(rec: dict, seed: int, styles: tuple[str, ...] | None = None) -> dict:
    """Native single-choice rows: reshuffle option order + ids so the gold's canonical position is uniform (score
    questions are ordinal and untouched; multi-question rows untouched)."""
    sc = single_choice(rec)
    if sc is None or rec["family"] in ("multi",) or rec["source"] == "synthetic_factchange":
        return rec
    qid, q = sc
    rng = random.Random(f"b1-rebalance:{seed}:{rec['id']}")
    style = rec.get("aug", {}).get("id_style")
    if style not in ID_STYLES:
        style = None
    if styles is not None:
        style = rng.choice(styles)
    return relabel_uniform(rec, qid, list(q["criteria"].items()), rec["labels"][qid], rng, style=style)


# ------------------------------------------------------------------ stage 2: draw


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def native_index(build_dir: Path) -> tuple[dict[str, dict], dict]:
    """id -> {src, source, family, sha256, content} over the 150k set (pointer-verified) and the supplement
    (verified against its build summary). Ids present in both must have byte-identical lines."""
    from train.train_set import read_pointer, uncompressed_sha256, verify_files

    parent_dir = build_dir / PARENT
    ptr = read_pointer(PARENT_POINTER)
    psha = verify_files(parent_dir, ptr)
    sup_dir = build_dir / SUPPLEMENT
    summ = json.loads((sup_dir / "receipt" / "build-summary.json").read_text(encoding="utf-8"))
    ssha = uncompressed_sha256(sup_dir / "records.jsonl.gz")
    if ssha != summ["records_uncompressed_sha256"]:
        raise ValueError("supplement records drifted from its build summary")
    n_sup = sum(SUPPLEMENT_COUNTS.values())
    idx: dict[str, dict] = {}
    shared = 0
    for d, man, con in ((parent_dir, ptr["manifest"], ptr["content_map"]),
                        (sup_dir, f"sample-{n_sup}-seed{config.SEED}.manifest.jsonl.gz",
                         f"sample-{n_sup}-seed{config.SEED}.content.jsonl.gz")):
        content = {e["id"]: e["content_sha256"] for e in _read_jsonl_gz(d / con)}
        for e in _read_jsonl_gz(d / man):
            prev = idx.get(e["id"])
            if prev is not None:
                if prev["sha256"] != e["sha256"]:
                    raise ValueError(f"{e['id']}: supplement line differs from the 150k line")
                shared += 1
                continue
            idx[e["id"]] = {"src": str(d / "records.jsonl.gz"), "source": e["source"], "family": e["family"],
                            "sha256": e["sha256"], "content": content[e["id"]]}
    return idx, {"parent": {"pointer": PARENT_POINTER, "pointer_sha256": ptr["_pointer_sha256"],
                            "files_uncompressed_sha256": psha},
                 "supplement": {"dir": SUPPLEMENT, "records_uncompressed_sha256": ssha, "counts_asked": SUPPLEMENT_COUNTS,
                                "n_kept": summ["n_kept"], "ids_shared_with_150k_byte_identical": shared},
                 "n_native_ids": len(idx)}


def exclusion_sets() -> dict[str, Any]:
    """Held-out identities: cohort v2 (all held-out roles) and v3 (G13 retention, sha 498efa61...), by id, normalized
    content, source item and key-free text; plus the three 150k D1-B hits."""
    from train.cohort import Cohort
    from train.cohort_v3 import cohort_v3
    from .build import item_keys

    v2, v3 = Cohort(), cohort_v3()
    v2_items = set().union(*(item_keys(i) for i in v2.held_out()))
    return {"ids": v2.held_out() | v3.held_out() | set(HELDOUT_150K),
            "content": v2.held_out_content() | v3.held_out_content(),
            "items": v2_items | v3.held_out_items() | set(HELDOUT_150K),
            "text": set(v3.text_keys or ()) | set(getattr(v2, "text_keys", None) or ()),
            "identity": {"v2": v2.identity(), "v3": v3.identity(), "heldout_150k_dropped": list(HELDOUT_150K)}}


def dedup_keys(rec: dict) -> set[str]:
    """Within-draw near-duplicate keys: ``record_text_keys`` minus the state+options ``s:`` key of an EMPTY-state row
    (``placement: instructions`` rows), whose ``s:`` text is only the option list and would collide across items."""
    from train.cohort import record_text_keys
    from heads.layout import is_empty_state

    keys = record_text_keys(rec)
    if is_empty_state(rec["state"]):
        keys = {k for k in keys if not k.startswith("s:")}
    return keys


class Filter:
    """Per-row acceptance: structure, D1 templates, cohort exclusions, global dedup (source item, normalized content,
    key-free text), SplitGuard, frozen held-out families, the Index fingerprint (derived/external rows) and the D4
    cell's own K/T filter (``train.run_cell.keep_item`` on the RWKV7 tokenizer, template jrt-rwkv-g1-v1)."""

    def __init__(self, tokdir: str, ex: dict, max_k: int = 255, max_t: int = 2048, template: str = "jrt-rwkv-g1-v1"):
        from train.cohort_v3 import _ensure_kit
        from train.run_cell import parent_encoder
        from .build import content_guard
        from .dedup import Fingerprint

        _ensure_kit()
        from harness.heldout import check_record
        self.check_record = check_record
        self.enc = parent_encoder("rwkv7", tokdir)
        self.guard, self.guard_identity = content_guard()
        self.fp = Fingerprint.load(config.FINGERPRINT)
        self.fp.assert_production()
        self.ex, self.max_k, self.max_t, self.template = ex, max_k, max_t, template
        from . import b1_check
        self.b1c = b1_check
        self.index = b1_check.build_index(config.DATA_ROOT)  # every DEV / TEST Index row, for the item-text check
        self.templates = b1_check.instruction_templates()
        self.seen_items: set[str] = set()
        self.seen_content: set[str] = set()
        self.seen_text: set[str] = set()
        self.reasons: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        #: id -> longest rendered item (tokens, cell template + RWKV7 tokenizer) of every row that passed the K/T filter
        self.row_tokens: dict[str, int] = {}
        self.hit_examples: dict[str, list] = collections.defaultdict(list)
        self.last_hit = None

    @staticmethod
    def item_keys(rec: dict) -> set[str]:
        from .build import item_keys
        return item_keys(rec.get("aug", {}).get("b1_view", {}).get("parent_id", rec["id"]))

    def why_not(self, rec: dict, fingerprint: bool = False) -> str | None:
        from train.cohort import record_content_hash, record_text_keys
        from train.data import items_from_records
        from train.guard import ForbiddenRow
        from train.run_cell import keep_item, to_train_record
        from . import augment as A, injection as I

        try:
            A.validate(rec)
            I.assert_no_heldout(rec["state"], rec["questions"], rec["id"])
        except AssertionError:
            return "structure_or_d1_template"
        items, ch, tk = self.item_keys(rec), record_content_hash(rec), record_text_keys(rec)
        if rec["id"] in self.ex["ids"] or items & self.ex["items"]:
            return "cohort_id_or_item"
        if ch in self.ex["content"]:
            return "cohort_content"
        if tk & self.ex["text"]:
            return "cohort_text"
        if items & self.seen_items:
            return "dup_item"
        if ch in self.seen_content or dedup_keys(rec) & self.seen_text:
            return "dup_text"
        try:
            self.guard.check(rec)
        except ForbiddenRow:
            return "splitguard"
        try:
            self.check_record(rec)
        except Exception:  # harness.heldout.FrozenFamilyError
            return "frozen_family"
        if fingerprint and any(self.fp.check(rec["state"], rec["questions"]).values()):
            return "fingerprint"
        if ch in self.index["content"]:
            return "index_content"
        if tk & self.index["text"]:
            return "index_text"
        hits = [x for x in self.b1c.item_hits(rec, self.index, self.templates) if not x[3]]
        if hits:
            self.last_hit = (rec["id"], hits[0][1], hits[0][2][:120])
            return "index_item_text"
        if len(canonical({"s": rec["state"], "q": rec["questions"]})) > 6 * self.max_t:  # >6 chars/token never seen
            return "len_prefilter"
        try:
            its = items_from_records([to_train_record(rec)], self.enc, layout="ordered", template=self.template)
        except ValueError:
            return "layout"
        if not its or not all(keep_item(x.row, self.max_k, self.max_t) for x in its):
            return "k_or_t_filter"
        self.row_tokens[rec["id"]] = max(len(x.row.ids) for x in its)
        return None

    def claim(self, rec: dict) -> None:
        from train.cohort import record_content_hash, record_text_keys
        self.seen_items |= self.item_keys(rec)
        self.seen_content.add(record_content_hash(rec))
        self.seen_text |= dedup_keys(rec)

    def accept(self, rec: dict, fam: str, fingerprint: bool = False) -> bool:
        why = self.why_not(rec, fingerprint)
        self.reasons[fam][why or "kept"] += 1
        if why == "index_item_text" and len(self.hit_examples[fam]) < 5:
            self.hit_examples[fam].append(self.last_hit)
        if why is None:
            self.claim(rec)
        return why is None


def _order_key(seed: int, fam: str, rid: str) -> str:
    return digest(f"b1:{seed}:{fam}:{rid}")


def _role(source: str, rid: str, seed: int) -> str | None:
    """Fixed split of a source's items between its families (disjoint source items): Glaive 7:4:3 native / catalog16
    / catalog_long; BANKING77 and CLINC150 5:3 k16 view / full catalog."""
    b = int(digest(f"b1-role:{seed}:{rid}")[:8], 16)
    if source == "glaive_fc":
        m = b % 14
        return "glaive_native" if m < 7 else "glaive_catalog16" if m < 11 else "glaive_catalog_long"
    if source == "banking77":
        return "banking77_k16" if b % 8 < 5 else "banking77_full"
    if source == "clinc150":
        return "clinc150_k16"
    return None


def _catalog_k(rid: str, seed: int, long: bool = False) -> int:
    h = int(digest(f"b1-k:{seed}:{rid}")[:8], 16)
    return 17 + h % 16 if long else 4 + h % 13  # target uniform in 17..32 (capped by the char budget) / 4..16


def candidates(idx: dict, ex: dict, seed: int) -> dict[str, list[str]]:
    from .build import item_keys
    by_src = {s: f for f, (_, kind, s) in FAMILIES.items() if kind == "native"
              and s not in ("glaive_fc", "banking77", "clinc150")}
    out: dict[str, list[str]] = collections.defaultdict(list)
    low = {s: [f for f, (_, kind, s2) in FAMILIES.items() if kind in LOWRATE_KINDS and s2 == s]
           for s in {s2 for _, kind, s2 in FAMILIES.values() if kind in LOWRATE_KINDS}}
    for rid, e in idx.items():
        if e["family"] == "multi" or rid in ex["ids"] or item_keys(rid) & ex["items"] or e["content"] in ex["content"]:
            continue
        fam = _role(e["source"], rid, seed) or by_src.get(e["source"])
        if fam:
            out[fam].append(rid)
        for f in low.get(e["source"], ()):  # v2 views: every native item of the source, in their own seeded order;
            out[f].append(rid)              # they draw after the native families, so those are unchanged
    for f in out:
        out[f].sort(key=lambda r: _order_key(seed, f, r))
    return out


def v3_candidates(v3raw, seed: int) -> dict[str, list[str]]:
    """Seeded candidate order of the v3 families (rows of data.b1_v3.Raw, overlap drops already removed)."""
    return {f: v3raw.order(f, seed) for f, (_, kind, _) in FAMILIES.items() if kind in V3_KINDS}


def _load_lines(idx: dict, want: set[str]) -> dict[str, str]:
    by_src: dict[str, set[str]] = collections.defaultdict(set)
    for rid in want:
        by_src[idx[rid]["src"]].add(rid)
    lines: dict[str, str] = {}
    for src, ids in by_src.items():
        with gzip.open(src, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                m = re.search(r'"id":"((?:[^"\\]|\\.)*)"', line)
                rid = json.loads(line)["id"] if m is None else json.loads('"' + m.group(1) + '"')
                if rid in ids:
                    if hashlib.sha256(line.encode("utf-8")).hexdigest() != idx[rid]["sha256"]:
                        raise ValueError(f"{rid}: line does not match its manifest")
                    lines[rid] = line
    return lines


def intent_k16_view_keep_ids(rec: dict, seed: int) -> dict | None:
    """Injected intent records: the same 16-option subset, original ids kept (the injected text names an id)."""
    plain = {**rec, "aug": {k: x for k, x in rec.get("aug", {}).items() if k != "injection"}}
    v = intent_k16_view(plain, seed)
    if v is None:
        return None
    qid, q = single_choice(rec)
    texts = set(v["questions"][qid]["criteria"].values())
    keep = {o: d for o, d in q["criteria"].items() if d in texts}
    target = rec["aug"]["injection"].get("target")
    if target not in q["criteria"] or rec["labels"][qid] not in keep:
        return None
    if target not in keep:  # swap the target in for the last hard-negative-free slot
        drop = next(o for o in sorted(keep, key=lambda o: digest(f"b1-swap:{seed}:{o}"))
                    if o != rec["labels"][qid] and q["criteria"][o] != OOS_TEXT)
        keep.pop(drop)
        keep[target] = q["criteria"][target]
    out = json.loads(json.dumps(rec))
    out["questions"][qid]["criteria"] = {o: keep[o] for o in sorted(keep)}
    out["id"] = rec["id"] + VIEW_SEP + "k16"
    out["aug"]["b1_view"] = {"kind": "k16", "parent_id": rec["id"], "k": len(keep), "ids_kept": True}
    return out


def _external(families_dir: Path | None) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Merge hook: every ``<families_dir>/<stem>.jsonl.gz`` (ssdm-train-v0) becomes family ``ext:<stem>``; returns
    (rows by family, raw-gz sha256 by file)."""
    if families_dir is None:
        return {}, {}
    out: dict[str, list[dict]] = {}
    shas = {}
    for p in sorted(Path(families_dir).glob("*.jsonl.gz")):
        out["ext:" + p.name[: -len(".jsonl.gz")]] = _read_jsonl_gz(p)
        shas[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out, shas


def select(tokdir: str, quotas: dict[str, int], seed: int = BUILD_SEED, build_dir: Path | None = None,
           families_dir: Path | None = None, external_quotas: dict[str, int] | None = None, log=print) -> dict:
    """Per-family selection in seeded order; returns each family's accepted rows IN ORDER, so any draw whose quotas
    are <= ``quotas`` is the per-family prefix. A family whose pool runs out is topped up from ``BACKFILL``."""
    build_dir = Path(build_dir or config.BUILD_DIR)
    idx, nat_ident = native_index(build_dir)
    ex = exclusion_sets()
    cands = candidates(idx, ex, seed)
    want: set[str] = set()
    for f, ids in cands.items():
        if FAMILIES[f][1] not in LOWRATE_KINDS:
            want.update(ids[: int(quotas.get(f, 0) * 4) + 3000])
    raw = None
    if any(quotas.get(f, 0) for f, (_, kind, _) in FAMILIES.items() if kind in LOWRATE_KINDS):
        from .b1_lowrate import Raw
        raw_want = {r for f, (_, kind, _) in FAMILIES.items() if kind in ("claim", "verify")
                    for r in cands.get(f, [])[: int(quotas.get(f, 0) * 4) + 3000]}
        raw = Raw(build_dir, want=raw_want)
    v3raw = None
    if any(quotas.get(f, 0) for f, (_, kind, _) in FAMILIES.items() if kind in V3_KINDS):
        from .b1_v3 import Raw as V3Raw
        v3raw = V3Raw(build_dir, fams=tuple(f for f, (_, kind, _) in FAMILIES.items() if kind in V3_KINDS))
        cands.update(v3_candidates(v3raw, seed))
    log(f"native ids {len(idx)}; loading {len(want)} candidate lines", flush=True)
    lines = _load_lines(idx, want)
    flt = Filter(tokdir, ex)
    donors, names = [], set()
    for rid in sorted(set().union(*(cands.get(f, []) for f in ("glaive_native", "glaive_catalog16",
                                                                   "glaive_catalog_long")))):
        if rid in lines:
            st0 = json.loads(lines[rid])["state"]
            for t in (st0.get(tool_field(st0)) or [] if tool_field(st0) else []):
                if isinstance(t, dict) and t.get("name") and t["name"] not in names:
                    names.add(t["name"])
                    donors.append(t)
    donors.sort(key=lambda t: digest(f"b1-donor:{seed}:{t['name']}"))
    # long catalogs must fit the T<=2048 rung: draw their donors from the shorter half of the tool schemas
    short = sorted(donors, key=lambda t: (len(canonical(t)), digest(f"b1-donor:{seed}:{t['name']}")))
    short_donors = sorted(short[: max(1, len(short) // 4)], key=lambda t: digest(f"b1-donor:{seed}:{t['name']}"))
    chosen: dict[str, list[dict]] = {}
    order = ["factchange", "glaive_native", "glaive_catalog16", "glaive_catalog_long", "toolace", "when2call",
             "gsm8k", "banking77_k16", "clinc150_k16", "banking77_full", "anli", "hellaswag", "winogrande", "sst5", "nq", "boolq", "mmlu_aux", "humicroedit_s1",
             "humicroedit_s2", "clinc150_member", "banking77_member", "anli_claim", "mmlu_verify",
             "acos", "nli4ct", "vast", "ragtruth", "hover", "habermas", "mnli"]
    assert sorted(order) == sorted(FAMILIES)
    ext, ext_shas = _external(families_dir)
    missing = sorted(set(external_quotas or {}) - set(ext))
    if missing:
        raise ValueError(f"external families {missing} not found in {families_dir}")
    for fam, rows in sorted(ext.items()):  # external families first: they are the scarce ones
        q = (external_quotas or {}).get(fam, 0)
        got = []
        for r in sorted(rows, key=lambda r: _order_key(seed, fam, r["id"])):
            if len(got) >= q:
                break
            # the B1-families build ran the item/context-tiered fingerprint itself; everything else runs here
            if flt.accept(r, fam):
                got.append(r)
        chosen[fam] = got
        log(f"  {fam}: {len(got)}/{q}  {dict(flt.reasons[fam])}", flush=True)
    cursor: dict[str, int] = collections.defaultdict(int)
    labels: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    def fill(fam: str, quota: int, got: list[dict]) -> list[dict]:
        """Accept rows of ``fam`` in seeded order from its cursor until ``quota`` (resumable for backfill)."""
        kind = FAMILIES[fam][1]
        ids = cands.get(fam, [])
        while len(got) < quota and cursor[fam] < len(ids):
            rid = ids[cursor[fam]]
            cursor[fam] += 1
            if kind in V3_KINDS:
                # the Index fingerprint already ran per row on the ITEM tier (data.b1_v3.overlap: every catalog; hits
                # dropped in v3raw); here the full-record fingerprint is off because it fires on shared CONTEXT /
                # template text by design (RAGTruth prompt templates "within N words", NLI4CT trial sections, HoVer
                # Wikipedia documents), the two-tier rule of data/dedup.py. Every other check runs, including the
                # Index item-text line / segment check that data.b1_check repeats.
                rec = v3raw.record(fam, rid, seed)
                if flt.accept(rec, fam, fingerprint=False):
                    got.append(rec)
                continue
            if kind in LOWRATE_KINDS:
                from .b1_lowrate import VIEWS
                rec = VIEWS[kind](raw, rid, seed)
                if rec is None:
                    flt.reasons[fam]["no_view"] += 1
                elif flt.accept(rec, fam, fingerprint=True):
                    got.append(rec)
                continue
            if rid not in lines:
                flt.reasons[fam]["not_loaded"] += 1
                continue
            base = json.loads(lines[rid])
            inj = bool(base.get("aug", {}).get("injection"))
            if kind == "k16":
                rec = intent_k16_view_keep_ids(base, seed) if inj else intent_k16_view(base, seed)
            elif kind == "catalog":
                rec = None if inj else glaive_catalog_view(base, donors, _catalog_k(rid, seed), seed, "cat16")
            elif kind == "catalog_long":
                rec = None if inj else glaive_catalog_view(base, short_donors, _catalog_k(rid, seed, True), seed,
                                                           "catlong", char_budget=LONG_VIEW_CHARS, min_k=17)
            elif fam.endswith("_full"):
                rec = base if inj else rebalance(base, seed, styles=("letters", "digits"))
            else:
                rec = base if inj else rebalance(base, seed)
            if rec is None:
                flt.reasons[fam]["no_view"] += 1
                continue
            if fam == "boolq":  # noul has a fixed true/false axis: balance the gold label itself
                lab = str(next(iter(rec["labels"].values()))).lower()
                if labels[fam][lab] >= (quota + 1) // 2:
                    flt.reasons[fam]["label_balance"] += 1
                    continue
            if flt.accept(rec, fam):
                got.append(rec)
                if fam == "boolq":
                    labels[fam][str(next(iter(rec["labels"].values()))).lower()] += 1
        return got

    for fam in order:
        quota, kind = quotas.get(fam, 0), FAMILIES[fam][1]
        got: list[dict] = []
        if fam == "factchange":
            pairs: dict[str, list[str]] = collections.defaultdict(list)
            for rid in cands[fam]:
                if rid in lines:
                    pairs[rid.split("~inj:", 1)[0].rsplit(":", 1)[0]].append(rid)
            for key in sorted(pairs, key=lambda k: _order_key(seed, fam, k)):
                if len(got) + 2 > quota:
                    break
                recs = [json.loads(lines[r]) for r in sorted(pairs[key])]
                if len(recs) != 2:
                    flt.reasons[fam]["pair_incomplete"] += 1
                    continue
                whys = [flt.why_not(r) for r in recs]
                if whys[0] is None:  # the partner shares the source item, so check it with the item unclaimed
                    flt.claim(recs[0])
                    whys[1] = flt.why_not(recs[1]) if whys[1] is None else whys[1]
                    whys[1] = None if whys[1] == "dup_item" else whys[1]
                if all(w is None for w in whys):
                    flt.claim(recs[1])
                    got.extend(recs)
                    flt.reasons[fam]["kept"] += 2
                else:
                    flt.reasons[fam][next(w for w in whys if w)] += 2
                    if whys[0] is None:  # roll back the first variant's claim
                        from train.cohort import record_content_hash, record_text_keys
                        flt.seen_content.discard(record_content_hash(recs[0]))
                        flt.seen_text -= dedup_keys(recs[0])
                        flt.seen_items -= flt.item_keys(recs[0])
        else:
            got = fill(fam, quota, got)
        chosen[fam] = got
        log(f"  {fam}: {len(got)}/{quota}  {dict(flt.reasons[fam])}", flush=True)
    # backfill: a family whose pool runs out (e.g. GSM8K after cohort v3) is topped up from the tools / facts pools
    deficit = sum(quotas.values()) - sum(len(chosen[f]) for f in order)
    deficit += sum((external_quotas or {}).values()) - sum(len(chosen.get(f, [])) for f in (external_quotas or {}))
    backfilled = {}
    for fam in BACKFILL:
        if deficit <= 0:
            break
        before = len(chosen[fam])
        chosen[fam] = fill(fam, before + deficit, chosen[fam])
        backfilled[fam] = len(chosen[fam]) - before
        deficit -= backfilled[fam]
        log(f"  backfill {fam}: +{backfilled[fam]}", flush=True)
    return {"chosen": chosen, "native": nat_ident, "exclusion": ex["identity"], "backfilled": backfilled,
            "filter_reasons": {f: dict(v) for f, v in flt.reasons.items()}, "guard": flt.guard_identity,
            "fingerprint": {k: flt.fp.meta.get(k) for k in ("version", "npz_sha256", "arrays_sha256", "rows")},
            "encoder": flt.enc.identity() if hasattr(flt.enc, "identity") else None,
            "cell_filter": {"max_k": flt.max_k, "max_t": flt.max_t, "template": flt.template, "layout": "ordered"},
            "seed": seed, "families_dir": str(families_dir) if families_dir else None,
            "families_files_gz_sha256": ext_shas, "row_tokens": flt.row_tokens,
            "index_item_text_examples": dict(flt.hit_examples), "index_boilerplate": flt.index["boilerplate"],
            "lowrate_raw": raw.ident if raw is not None else None,
            "v3_raw": v3raw.ident if v3raw is not None else None}


# ------------------------------------------------------------------ write


def gold_position(rec: dict) -> list[tuple[int, int]]:
    """(canonical gold position, K) per choice question, in the renderer's order."""
    from decide.schema import canonical_options
    out = []
    for qid, q in rec["questions"].items():
        if q["type"] == "choice":
            ids = [o for o, _ in canonical_options(q)]
            out.append((ids.index(rec["labels"][qid]), len(ids)))
    return out


def write_draw(name: str, rows: list[dict], sel: dict, quotas: dict[str, int], out_root: Path,
               builder_sha: str, receipt_dir: Path) -> dict:
    """Write records / manifest / content map in the 150k format (records sorted by id) and the pointer."""
    from train.cohort import record_content_hash

    out = out_root / name
    if (out / "records.jsonl.gz").exists():
        raise ValueError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate ids")
    lines = [canonical(r) for r in rows]
    man = [canonical({"family": r["family"], "id": r["id"], "sha256": digest(line), "source": r["source"]})
           for r, line in zip(rows, lines)]
    con = [json.dumps({"content_sha256": record_content_hash(r), "id": r["id"]}, sort_keys=True,
                      separators=(",", ":")) for r in rows]
    files = {"records": ("records.jsonl.gz", lines), "manifest": (f"{name}.manifest.jsonl.gz", man),
             "content_map": (f"{name}.content.jsonl.gz", con)}
    usha, gsha = {}, {}
    for k, (fn, ls) in files.items():
        blob = ("\n".join(ls) + "\n").encode("utf-8")
        usha[k] = hashlib.sha256(blob).hexdigest()
        with gzip.GzipFile(out / fn, "wb", mtime=0) as f:
            f.write(blob)
        gsha[fn] = hashlib.sha256((out / fn).read_bytes()).hexdigest()
    pointer = {"build": f"{name} (python -X utf8 -I -m data.b1 draw; BOARD B1)",
               "location": f"ssdm-data/m2/build/{name}/ (outside git; deterministic, a rebuild reproduces it)",
               "records_uncompressed_sha256": usha["records"], "manifest": files["manifest"][0],
               "manifest_uncompressed_sha256": usha["manifest"], "content_map": files["content_map"][0],
               "content_map_uncompressed_sha256": usha["content_map"], "n_records": len(rows),
               "excludes_cohort": "m2-dev-cohort-v2 be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2 and "
                                  "m2-dev-cohort-v3 498efa617c931a5ea073bc5c62ce94158270888f64b9bbdb027982cde43876da "
                                  "(id, content, source item, key-free text)",
               "builder": {"module": "data/b1.py", "commit": builder_sha}, "receipt": receipt_dir.as_posix() + "/"}
    # per-family / area / source counts and gold-position balance
    fam_of = {}
    for f, rs in sel["chosen"].items():
        for r in rs:
            fam_of[r["id"]] = f
    by_fam = collections.Counter(fam_of[i] for i in ids)
    by_src = collections.Counter(r["source"] for r in rows)
    def group_of(f):
        return EXT_GROUP.get(f) or FAMILIES[f][0]
    by_area = collections.Counter(GROUP_AREA[group_of(fam_of[i])] for i in ids)
    by_group = collections.Counter(group_of(fam_of[i]) for i in ids)
    pos = collections.defaultdict(collections.Counter)
    for r in rows:
        for p, k in gold_position(r):
            pos[k][p] += 1
    chi = {}
    for k, c in pos.items():
        n = sum(c.values())
        e = n / k
        chi[k] = {"n": n, "max_dev_from_uniform_pct": round(100 * max(abs(c[p] - e) for p in range(k)) / n, 2)}
    def kmax(r):
        return max(len(q["criteria"]) if q["type"] != "noul" else 2 for q in r["questions"].values())
    k16 = collections.Counter("k<=16" if kmax(r) <= 16 else "k>16" for r in rows)
    k16_fam = collections.defaultdict(collections.Counter)
    for r in rows:
        k16_fam[fam_of[r["id"]]]["k<=16" if kmax(r) <= 16 else "k>16"] += 1
    toks = sorted(sel["row_tokens"][i] for i in ids)
    def pct(q):
        return toks[min(len(toks) - 1, int(q * len(toks)))]
    token_hist = {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99), "max": toks[-1], "n_over_2048": sum(t > 2048 for t in toks),
                  "bins": dict(collections.Counter("<=512" if t <= 512 else "513-1024" if t <= 1024 else
                                                   "1025-1536" if t <= 1536 else "1537-2048" if t <= 2048 else ">2048"
                                                   for t in toks))}
    if token_hist["n_over_2048"]:
        raise AssertionError("a row over 2,048 tokens reached the draw")
    lowrate = collections.defaultdict(collections.Counter)  # v2 views: gold label rates at the natural base rate
    for r in rows:
        if r.get("aug", {}).get("b1_view", {}).get("kind") in LOWRATE_KINDS + V3_KINDS:
            for qid in r["questions"]:
                lowrate[fam_of[r["id"]]][str(r["labels"][qid])] += 1
            if not any(str(v) in ("yes", "SUPPORTED", "true", "Entailment", "entailment") for v in r["labels"].values()):
                lowrate[fam_of[r["id"]]]["_records_with_no_positive"] += 1
            lowrate[fam_of[r["id"]]]["_records"] += 1
    noul = collections.Counter()
    for r in rows:
        for qid, q in r["questions"].items():
            if q["type"] == "noul":
                noul[str(r["labels"][qid]).lower()] += 1
    summary = {"build": name, "n_records": len(rows), "quotas": quotas, "by_family": dict(sorted(by_fam.items())),
               "shortfall": {f: q - by_fam.get(f, 0) for f, q in quotas.items() if by_fam.get(f, 0) < q},
               "by_group_share": {g: round(v / len(rows), 4) for g, v in sorted(by_group.items())},
               "by_group": dict(sorted(by_group.items())), "by_index_area": dict(sorted(by_area.items())),
               "by_source": dict(sorted(by_src.items())), "rows_by_max_k": dict(k16),
               "rendered_tokens": token_hist,
               "rows_by_max_k_by_family": {f: dict(v) for f, v in sorted(k16_fam.items())},
               "gold_position_uniformity_by_k": dict(sorted(chi.items())), "noul_gold_labels": dict(noul),
               "lowrate_gold_labels_by_family": {f: dict(v) for f, v in sorted(lowrate.items())},
               "injected_rows": sum(bool(r.get("aug", {}).get("injection")) for r in rows),
               "views": dict(collections.Counter(r["aug"]["b1_view"]["kind"] for r in rows if "b1_view" in r.get("aug", {}))),
               "uncompressed_sha256": usha, "gz_sha256": gsha, "pointer": pointer}
    return summary


def main(argv=None) -> int:
    import argparse
    import subprocess
    import sys

    if not sys.flags.utf8_mode:
        raise SystemExit("run with python -X utf8")
    ap = argparse.ArgumentParser(prog="data.b1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("supplement")
    d = sub.add_parser("draw")
    d.add_argument("--tokdir", required=True, help="pinned RWKV7-G1k tokenizer dir (the D4 cell's encoder)")
    d.add_argument("--draws", nargs="+", default=list(DRAWS), choices=list(DRAWS))
    d.add_argument("--families-dir", default=None, help="merge hook: B1-families build dir (records.jsonl.gz)")
    d.add_argument("--quotas", default=None,
                   help="merge hook: JSON file {draw name: {family: n}} replacing WEIGHTS_* for that draw; families "
                        "named 'ext:<family>' are drawn from --families-dir; each draw must keep its total")
    d.add_argument("--suffix", default="", help="output name suffix, e.g. -v2 for a families-merged rebuild")
    d.add_argument("--receipt", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "supplement":
        print(json.dumps(supplement(), indent=1))
        return 0
    root = config.REPO_ROOT
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "data/b1.py"], capture_output=True,
                           text=True).stdout.strip()
    builder = sha + ("+dirty" if dirty else "")
    builder_sha256 = hashlib.sha256((root / "data" / "b1.py").read_bytes()).hexdigest()
    override = json.loads(Path(a.quotas).read_text(encoding="utf-8")) if a.quotas else {}
    draws = {}
    for n in a.draws:
        q = dict(override.get(n) or DRAWS[n])
        if sum(q.values()) != sum(DRAWS[n].values()):
            raise SystemExit(f"{n}: quotas sum to {sum(q.values())}, not {sum(DRAWS[n].values())}")
        draws[n + a.suffix] = q
    fams = sorted({f for q in draws.values() for f in q})
    big = {f: max(q.get(f, 0) for q in draws.values()) for f in fams}
    fdir = Path(a.families_dir) if a.families_dir else (config.BUILD_DIR / FAMILIES_DIR
                                                         if any(f.startswith("ext:") for f in big) else None)
    sel = select(a.tokdir, {f: v for f, v in big.items() if not f.startswith("ext:")},
                 families_dir=fdir,
                 external_quotas={f: v for f, v in big.items() if f.startswith("ext:")})
    rd = Path(a.receipt)
    rd.mkdir(parents=True, exist_ok=True)
    report = {"builder": {"commit": builder, "data/b1.py sha256": builder_sha256}, "seed": sel["seed"],
              "native": sel["native"], "exclusion": sel["exclusion"], "guard": sel["guard"],
              "fingerprint": sel["fingerprint"], "encoder": sel["encoder"], "cell_filter": sel["cell_filter"],
              "families_dir": sel["families_dir"], "families_files_gz_sha256": sel["families_files_gz_sha256"],
              "backfilled": sel["backfilled"], "index_item_text_examples": sel["index_item_text_examples"],
              "index_boilerplate": sel["index_boilerplate"], "lowrate_raw": sel["lowrate_raw"], "v3_raw": sel["v3_raw"],
              "filter_reasons_over_selection": sel["filter_reasons"], "draws": {}}
    for name, q in draws.items():
        rows = [r for f in q for r in sel["chosen"].get(f, [])[: q[f] + sel["backfilled"].get(f, 0)]]
        s = write_draw(name, rows, sel, q, config.BUILD_DIR, builder, Path(a.receipt.replace("\\", "/")))
        s["builder_sha256"] = builder_sha256
        report["draws"][name] = s
        # LF bytes: the committed pointer is LF (.gitattributes) and its sha256 is part of the training identity
        (root / "data" / "manifests" / f"{name}.pointer.json").write_bytes(
            (json.dumps(s["pointer"], indent=1) + "\n").encode("utf-8"))
        print(json.dumps({k: s[k] for k in ("build", "n_records", "by_group", "shortfall", "rows_by_max_k")}), flush=True)
    (rd / "build-report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
