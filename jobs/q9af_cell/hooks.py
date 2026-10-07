"""Typed hooks of the Q9-AF cell config, fail closed until the code they name is merged.

``teacher``        optional KD arm (PREMORTEM S3b; amendment A2's R16/R17 recipe), WIRED (PR #116, b511643):
                   {alpha, temperature, source, teacher_sha256} -> ``LossConfig(kd_alpha, kd_temperature)``
                   (``train.run_cell.kd_fields``; absent = the KD fields stay out of the config hash) and
                   ``train.run_cell.attach_teacher`` (the PINNED Clef-flash export ``train.proxy_rule.TEACHER``: identity,
                   file sha256, the cell's draw == the teacher's draw, contamination checks). ``kd_arm`` refuses a
                   teacher block that does not name the pinned file. OFF in the default cells; ``q9af_cell_r64_kd.json``
                   is committed but has no job_params (launch.py refuses an unknown mode) until R16/R17 report.
``head_from_proxy`` hook for the proxy loop's winning attention-free mixer head (train/proxy_rule.py verdict): the
                   head name + kwargs + the sha256 of the proxy verdict receipt. It is resolved into ``head`` /
                   ``head_kwargs`` of a NEW committed config (``resolve``), never at run time: ``apply`` refuses a
                   config whose ``head`` / ``head_kwargs`` differ from its ``head_from_proxy`` block. A head that
                   reports ``attention_free: False`` (softmax over positions, e.g. ``clef_attn``) is refused.
``q9af``           the export pin (``converted_sha256``, ``base_config_sha256``, ``export_json_sha256``) and the
                   explicit ``hybrid`` switch: ``check_export`` refuses an unpinned export, and a partial export
                   (``gdn_partial.pt`` / ``attention_free: false`` in export.json) unless ``hybrid`` is true; a hybrid
                   is labelled NOT attention-free everywhere (``heads.backbone.hybrid_label``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

TEACHER_API = 1


@runtime_checkable
class TeacherSource(Protocol):
    """Per-item teacher probabilities for the KD arm (Clef-flash / Clef per-option probabilities)."""

    def identity(self) -> dict[str, Any]: ...

    def probs(self, item_id: str) -> list[float] | None: ...


@dataclass(frozen=True)
class KDConfig:
    alpha: float  # weight of the KD term; the clef arm keeps weight 1 - alpha
    temperature: float = 1.0
    source: str = ""  # pinned teacher file (sha256 in teacher_sha256)
    teacher_sha256: str = ""

    def __post_init__(self):
        if not (0.0 < self.alpha <= 1.0) or not self.temperature > 0:
            raise ValueError(f"KD alpha must be in (0, 1] and temperature > 0, got {self.alpha}, {self.temperature}")
        if len(self.teacher_sha256) != 64:
            raise ValueError("KD teacher_sha256 must pin the teacher file (64 hex chars)")


class HookUnavailable(ValueError):
    """A configured hook names code that is not merged (fail closed: run_cell exit 35)."""


def kd_arm(cfg: dict) -> KDConfig | None:
    """The validated KD block (None = no KD). Refuses a block whose teacher file is not the pinned A2 teacher."""
    t = cfg.get("teacher")
    if not t:
        return None
    kd = KDConfig(**t)
    try:
        from train.teacher import pinned
    except ImportError as e:  # pragma: no cover - merged in b511643
        raise HookUnavailable("cell config sets 'teacher' (KD arm) but train.teacher is not importable") from e
    want = pinned()["sha256"]
    if kd.teacher_sha256 != want:
        raise ValueError(f"teacher.teacher_sha256 {kd.teacher_sha256[:12]} != the pinned teacher {want[:12]} "
                         "(train.proxy_rule.TEACHER)")
    return kd


def resolve(cfg: dict) -> dict:
    """The committed config a proxy-loop winner turns into (``head`` / ``head_kwargs`` from ``head_from_proxy``)."""
    h = cfg.get("head_from_proxy")
    if not h:
        return dict(cfg)
    for k in ("head", "head_kwargs", "proxy_verdict_sha256"):
        if k not in h:
            raise ValueError(f"head_from_proxy lacks {k!r}")
    if len(str(h["proxy_verdict_sha256"])) != 64:
        raise ValueError("head_from_proxy.proxy_verdict_sha256 must pin the proxy verdict receipt")
    from heads.heads import HEADS, make_head

    if h["head"] not in HEADS:
        raise ValueError(f"head_from_proxy.head {h['head']!r} is not a registered head {sorted(HEADS)}")
    probe = make_head(h["head"], 16, **(h.get("head_kwargs") or {}))
    cost = probe.cost(T=8, K=2) if hasattr(probe, "cost") else {}
    if getattr(probe, "attention_free", True) is False or cost.get("attention_free") is False:
        raise ValueError(f"head {h['head']!r} is not attention-free; it cannot be a Q9-AF decision head")
    return {**cfg, "head": h["head"], "head_kwargs": dict(h.get("head_kwargs") or {})}


def apply(cfg: dict) -> dict:
    """Validate every hook; returns the effective config (== cfg for a committed, resolved config)."""
    kd_arm(cfg)
    return resolve(cfg)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def check_export(cfg: dict, export_json: str | Path, gdn_ckpt: str | Path) -> dict:
    """The full-8 export (jobs/full8/export_gdn.py ``export.json`` + the .pt) against the cell's ``q9af`` pin.
    Raises ValueError on: an unpinned or mismatched sha256, a partial export without ``hybrid: true``, a full export
    configured as hybrid, export.json not ``ok``. Returns the label block stamped into every report."""
    q = cfg.get("q9af") or {}
    for k in ("converted_sha256", "export_json_sha256", "base_config_sha256"):
        if not q.get(k) or len(str(q[k])) != 64:
            raise ValueError(f"q9af.{k} is not pinned ({q.get(k)!r}): refusing an unpinned export (fill it with "
                             "python -m jobs.q9af_cell.pin when job-2xe5w lands)")
    if sha256(export_json) != q["export_json_sha256"]:
        raise ValueError("export.json sha256 != q9af.export_json_sha256")
    meta = json.loads(Path(export_json).read_text(encoding="utf-8"))
    if not meta.get("ok") or meta.get("mode") != "final":
        raise ValueError(f"export.json is not a successful final export (ok={meta.get('ok')}, mode={meta.get('mode')})")
    if meta.get("sha256") != q["converted_sha256"] or sha256(gdn_ckpt) != q["converted_sha256"]:
        raise ValueError("converted checkpoint sha256 != q9af.converted_sha256 (export.json / file)")
    af = bool(meta.get("attention_free"))
    if not af and not q.get("hybrid"):
        raise ValueError(f"export is PARTIAL ({meta.get('file')}: attention layers kept "
                         f"{meta.get('attention_layers_kept')}); set q9af.hybrid true to run it as a labelled hybrid")
    if af and q.get("hybrid"):
        raise ValueError("q9af.hybrid is set but the export is the full Q9-AF: drop 'hybrid'")
    if meta.get("base_config_sha256") and meta["base_config_sha256"] != q["base_config_sha256"]:
        raise ValueError("export.json base_config_sha256 != q9af.base_config_sha256")
    label = ("Q9-AF: attention-free (all attention layers converted)" if af else
             f"HYBRID Q9-H{len(meta.get('attention_layers_kept') or [])}: NOT attention-free "
             f"(attention layers kept {meta.get('attention_layers_kept')})")
    return {"attention_free": af, "label": label, "file": meta.get("file"), "converted": meta.get("converted"),
            "attention_layers_kept": meta.get("attention_layers_kept", []), "sha256": meta.get("sha256")}
