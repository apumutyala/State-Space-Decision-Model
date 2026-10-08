"""Teacher distributions for knowledge distillation (train/losses.py KD; proxy-rule-v1 amendment A2).

The teacher is Cloudflare Clef-flash (Apache-2.0 open weights; PLAN.md allows open-weight teachers only), exported by
``baselines.clef_teacher`` on the m2 first cell's seed-0 TRAINING draw (job-tnrwu, artifact art-ggrs8). One JSON line
per training record: ``id``, ``content_sha256`` (``train.cohort.record_content_hash``), ``status`` (ok | exhausted),
``raw_probabilities`` {question id: {option id: p}} for ok rows.

``load_teacher(dir)`` reads ``teacher-<release>.jsonl`` + ``.identity.json`` + ``selected-ids.txt`` + ``selection.json``
and refuses (``TeacherIdentityError``):
  * an identity file that differs from the pinned identity (release, repo, revision, backend revision, row count,
    status counts, drawn-ids sha256, file sha256, standin);
  * a teacher file whose sha256 is not the pinned one, or a malformed / duplicated / truncated line;
  * a selected-ids list whose sha256 is not the pinned draw (2bbbf180...), or a teacher id set != the draw;
  * a row whose content hash differs from selection.json's (the record the teacher saw is not the drawn one);
  * an ok row without probabilities, an exhausted row with them, a probability that is not finite / >= 0, or a
    question whose probabilities do not sum to 1 (1e-6).
and refuses (``TeacherContamination``) any teacher id that is a DEV / TEST row id (harness manifests) or a frozen-cohort
record (cohort v2 by id and content; cohort v3 by id, content and source item: ``assert_draw_disjoint``).

Mapping (``TeacherSet.probs``) is by record id -> question id -> option id, with the record's content hash checked when
the caller has it. A training row whose record or question has no teacher distribution gets ``None`` (the loss falls
back to the clef arm); a row whose OPTION SET differs from the teacher's is refused (it is not the question the teacher
answered). ``verify_records`` re-checks the drawn records themselves (content hash, the DEV/TEST ``SplitGuard`` and
``harness.heldout.check_record``: no record may target a frozen held-out family).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
#: tolerance on sum(p) per question (the export writes float64 softmax outputs; measured max |sum - 1| ~ 2e-16)
SUM_TOL = 1e-6


class TeacherIdentityError(ValueError):
    """The teacher files are not the pinned export, or a row cannot be mapped to the training row it claims."""


class TeacherContamination(TeacherIdentityError):
    """A teacher row is a DEV / TEST / frozen-cohort record."""


def pinned() -> dict[str, Any]:
    """The pre-registered teacher (train/proxy_rule.py TEACHER, amendment A2)."""
    from train.proxy_rule import TEACHER

    return dict(TEACHER)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def ids_sha256(ids: list[str]) -> str:
    """As ``baselines.clef_teacher`` / ``train.train_set.draw``: sha256 of the ids joined by newlines, draw order."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


@dataclass(frozen=True)
class TeacherRow:
    id: str
    content_sha256: str
    status: str
    probs: dict[str, dict[str, float]] | None


@dataclass
class TeacherSet:
    rows: dict[str, TeacherRow]
    draw_ids: list[str]
    identity: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)

    def probs(self, rid: str, qid: str, option_ids: Iterable[str],
              content_sha256: str | None = None) -> tuple[float, ...] | None:
        """Teacher probabilities in ``option_ids`` order, or None (no row, exhausted row, or question not answered)."""
        r = self.rows.get(rid)
        if r is None:
            return None
        if content_sha256 is not None and content_sha256 != r.content_sha256:
            raise TeacherIdentityError(f"{rid}: record content {content_sha256[:12]} != teacher's {r.content_sha256[:12]}")
        if r.probs is None or qid not in r.probs:
            return None
        q = r.probs[qid]
        opts = list(option_ids)
        if set(opts) != set(q) or len(opts) != len(q):
            raise TeacherIdentityError(f"{rid}/{qid}: options {sorted(opts)[:6]} != teacher options {sorted(q)[:6]}")
        return tuple(float(q[o]) for o in opts)

    def summary(self) -> dict[str, Any]:
        return {"identity": self.identity, **self.stats}


def _read_lines_strict(path: Path) -> list[dict[str, Any]]:
    """Newline-only split (LESSONS L-15: never ``splitlines``); a malformed line or a missing final newline raises."""
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise TeacherIdentityError(f"{path.name}: truncated last line (no final newline)")
    out = []
    for n, line in enumerate(raw.decode("utf-8").split("\n")[:-1] if raw else [], 1):
        if not line.strip():
            raise TeacherIdentityError(f"{path.name}:{n}: empty line")
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise TeacherIdentityError(f"{path.name}:{n}: malformed JSON ({e})") from e
    return out


IDENTITY_KEYS = ("release", "repo", "revision", "backend_revision", "rows", "status_counts")


def check_identity(ident: dict[str, Any], expected: dict[str, Any]) -> None:
    for k in IDENTITY_KEYS:
        if ident.get(k) != expected[k]:
            raise TeacherIdentityError(f"teacher identity {k}={ident.get(k)!r} != pinned {expected[k]!r}")
    if ident.get("selection_drawn_ids_sha256") != expected["drawn_ids_sha256"]:
        raise TeacherIdentityError("teacher identity selection_drawn_ids_sha256 != the pinned draw "
                                   f"{expected['drawn_ids_sha256'][:12]}")
    if ident.get("teacher_sha256") != expected["sha256"]:
        raise TeacherIdentityError("teacher identity teacher_sha256 != the pinned teacher file sha256")
    if ident.get("standin"):
        raise TeacherIdentityError("teacher identity says standin: not a real teacher export")


def load_teacher(teacher_dir: str | Path, expected: dict[str, Any] | None = None,
                 contamination: bool = True) -> TeacherSet:
    """See the module docstring. ``expected`` defaults to the pinned A2 teacher; ``contamination=False`` is for unit
    tests on synthetic ids only (the real loader always checks)."""
    exp = expected if expected is not None else pinned()
    d = Path(teacher_dir)
    rel = exp["release"]
    f_rows, f_ident = d / f"teacher-{rel}.jsonl", d / f"teacher-{rel}.identity.json"
    f_ids, f_sel = d / "selected-ids.txt", d / "selection.json"
    for p in (f_rows, f_ident, f_ids, f_sel):
        if not p.exists():
            raise TeacherIdentityError(f"teacher file missing: {p}")
    ident = json.loads(f_ident.read_text(encoding="utf-8"))
    check_identity(ident, exp)
    got = sha256_file(f_rows)
    if got != exp["sha256"]:
        raise TeacherIdentityError(f"{f_rows.name} sha256 {got[:12]} != pinned {exp['sha256'][:12]}")
    ids = [x for x in f_ids.read_text(encoding="utf-8").split("\n") if x]
    if ids_sha256(ids) != exp["drawn_ids_sha256"] or len(ids) != exp["rows"] or len(set(ids)) != len(ids):
        raise TeacherIdentityError("selected-ids.txt is not the pinned draw")
    sel = json.loads(f_sel.read_text(encoding="utf-8"))
    if not sel.get("ok") or sel.get("drawn_ids_sha256") != exp["drawn_ids_sha256"]:
        raise TeacherIdentityError("selection.json is not the pinned, ok selection")
    content = sel.get("content_sha256") or {}
    if set(content) != set(ids):
        raise TeacherIdentityError("selection.json content map ids != selected ids")
    rows: dict[str, TeacherRow] = {}
    counts: dict[str, int] = {}
    n_q = 0
    for r in _read_lines_strict(f_rows):
        rid = r.get("id")
        if rid in rows:
            raise TeacherIdentityError(f"duplicate teacher id {rid}")
        if rid not in content:
            raise TeacherIdentityError(f"teacher id {rid} is not in the selected draw")
        if r.get("content_sha256") != content[rid]:
            raise TeacherIdentityError(f"{rid}: teacher content hash != selection.json")
        st = r.get("status")
        counts[st] = counts.get(st, 0) + 1
        probs = r.get("raw_probabilities")
        if st == "ok":
            if not isinstance(probs, dict) or not probs:
                raise TeacherIdentityError(f"{rid}: ok row without probabilities")
            clean = {}
            for qid, q in probs.items():
                if not isinstance(q, dict) or not q:
                    raise TeacherIdentityError(f"{rid}/{qid}: empty teacher distribution")
                vals = [float(v) for v in q.values()]
                if any(not math.isfinite(v) or v < 0 for v in vals):
                    raise TeacherIdentityError(f"{rid}/{qid}: probability not finite / negative")
                if abs(sum(vals) - 1.0) > SUM_TOL:
                    raise TeacherIdentityError(f"{rid}/{qid}: probabilities sum to {sum(vals)!r}")
                clean[str(qid)] = {str(k): float(v) for k, v in q.items()}
                n_q += 1
            rows[rid] = TeacherRow(rid, r["content_sha256"], st, clean)
        else:
            if probs is not None:
                raise TeacherIdentityError(f"{rid}: {st} row carries probabilities")
            rows[rid] = TeacherRow(rid, r["content_sha256"], st, None)
    if set(rows) != set(ids):
        raise TeacherIdentityError(f"teacher rows {len(rows)} != selected ids {len(ids)}")
    if counts != exp["status_counts"]:
        raise TeacherIdentityError(f"status counts {counts} != pinned {exp['status_counts']}")
    ts = TeacherSet(rows, ids, {k: ident.get(k) for k in IDENTITY_KEYS + ("selection_drawn_ids_sha256",
                                                                         "teacher_sha256")},
                    {"rows": len(rows), "status_counts": counts, "questions_with_probs": n_q})
    if contamination:
        ts.stats["contamination"] = assert_clean(ids, content)
    return ts


def assert_clean(ids: list[str], content: dict[str, str]) -> dict[str, Any]:
    """No teacher id is a DEV / TEST row id; none is in the frozen cohort (v2 by id + content; v3 by id, content and
    source item). Returns the identities checked (for receipts)."""
    from train.cohort import Cohort, CohortLeak
    from train.cohort_v3 import cohort_v3
    from train.guard import SplitGuard

    man = ROOT / "harness" / "manifests"
    g = SplitGuard.from_manifests([man / "dev-manifest.jsonl.gz", man / "test-manifest.jsonl.gz"])
    dev_test = sorted(set(ids) & g.ids)
    if dev_test:
        raise TeacherContamination(f"{len(dev_test)} teacher id(s) are DEV/TEST rows, e.g. {dev_test[:3]}")
    v2 = Cohort(expected_sha256="be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2")
    v3 = cohort_v3()
    try:
        v2.assert_draw_disjoint(ids, content)
        v3.assert_draw_disjoint(ids, content)
    except CohortLeak as e:
        raise TeacherContamination(str(e)) from e
    rel = {Path(k).relative_to(ROOT).as_posix() if Path(k).is_relative_to(ROOT) else k: v for k, v in g.manifests.items()}
    return {"dev_test_manifests": rel, "dev_test_id_hits": 0, "cohort_v2": v2.identity()["sha256"],
            "cohort_v3": v3.identity()["sha256"], "cohort_hits": 0}


def verify_records(teacher: TeacherSet, records: Iterable[dict], guard=None, frozen_check=None) -> dict[str, Any]:
    """Re-check the drawn records the teacher saw: every teacher id present once, content hash = the teacher row's,
    ``guard.check`` (DEV/TEST SplitGuard, with the content index in production) and ``frozen_check``
    (``harness.heldout.check_record``) on each. Raises on any failure."""
    from train.cohort import record_content_hash

    seen = set()
    for rec in records:
        rid = rec.get("id")
        if rid not in teacher.rows:
            continue
        if rid in seen:
            raise TeacherIdentityError(f"record {rid} given twice")
        seen.add(rid)
        if record_content_hash(rec) != teacher.rows[rid].content_sha256:
            raise TeacherIdentityError(f"{rid}: record content != the teacher row's")
        if guard is not None:
            guard.check(rec)
        if frozen_check is not None:
            frozen_check(rec)
    missing = sorted(set(teacher.rows) - seen)
    if missing:
        raise TeacherIdentityError(f"{len(missing)} teacher record(s) not among the given records, e.g. {missing[:3]}")
    return {"records_checked": len(seen), "guard": guard is not None, "frozen_check": frozen_check is not None}


INJECTION_ID_MARKER = "~inj"  # PR #44 injection-augmented record ids: "<base id>~inj:<family>:<template>"


def injected_record_ids(records: Iterable[dict] | None = None, ids: Iterable[str] = ()) -> set[str]:
    """Record ids of injection-augmented training rows: ``aug.injection`` set on the record, OR the id carries the
    ``~inj`` marker (the 150k set carries both on the same 12,115 rows; the union is the robust choice)."""
    out = {str(i) for i in ids if INJECTION_ID_MARKER in str(i)}
    if isinstance(records, dict):  # train_set.draw returns {record id: record}
        records = records.values()
    for rec in records or ():
        rid = str(rec.get("id"))
        if (rec.get("aug") or {}).get("injection") or INJECTION_ID_MARKER in rid:
            out.add(rid)
    return out


def attach(items: list, teacher: TeacherSet, mask_injected: bool = False,
           injected_ids: set[str] | None = None) -> dict[str, Any]:
    """Set ``Item.teacher`` (row.option_ids order) on every training item that has a teacher distribution; the rest
    keep ``None`` (clef-arm fallback). Item ids are ``<record id>/<question id>``; the record's content hash
    (``row.meta['content_sha256']``, set by ``items_from_records``) must equal the teacher row's.

    ``mask_injected`` (cell config ``kd_mask_injected``): items of injection-augmented records (``injected_ids``, plus
    any record id with the ``~inj`` marker) get ``teacher = None`` and so the gold-only arm loss (the all-NaN path of
    ``train.losses.decision_loss``). The Clef-flash teacher partly follows the injections (d1a top answer = attacker
    target 23.2%, hijack 13.7%), so distilling it on those rows would train the student toward following them."""
    inj = set(injected_ids or ())
    n = hit = masked = masked_had = 0
    for it in items:
        rid, qid = it.id.rsplit("/", 1)
        tp = teacher.probs(rid, qid, it.row.option_ids, it.row.meta.get("content_sha256"))
        if mask_injected and (rid in inj or INJECTION_ID_MARKER in rid):
            masked += 1
            masked_had += tp is not None
            tp = None
        it.teacher = tp
        n += 1
        hit += tp is not None
    cov = {"items": n, "with_teacher": hit, "fallback": n - hit}
    if mask_injected:
        cov.update({"mask_injected": True, "masked_injected": masked, "masked_injected_had_teacher": masked_had,
                    "fallback_unmasked": n - hit - masked})
    return cov
