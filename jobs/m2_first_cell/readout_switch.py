"""Job-params knob ``"readout"`` for ``train.m2_gates --stage hijack --readout <r>`` (mode hijack_eval_champ): score a
SAVED checkpoint's child arm under a named eval readout instead of its training template, EXACTLY as
``jobs.m2_first_cell.readout_ab`` / ``readout_temperature.readout_decider`` build it (the same model object, encoder,
checkpoint record and ``shared_prefix``; template + layout from ``readout_ab.READOUTS`` / ``READOUT_LAYOUT``). The
checkpoint is restored and lineage-checked on the cell's own template first (``Trainer.load``); only the prompt changes.

The hijack stage scores both arms UNTEMPERED (argmax-based decisions), so no temperature is involved here. Full-DEV
scoring (``fulldev_eval --readout <r> --refit-temperature``) uses ``readout_temperature`` (the per-readout refit).
"""
from __future__ import annotations

from typing import Any

from jobs.m2_first_cell.readout_ab import READOUT_LAYOUT, READOUTS

TRAINING = "training"


def check(readout: str | None) -> str | None:
    if readout in (None, "", TRAINING):
        return None
    if readout not in READOUTS:
        raise ValueError(f"readout {readout!r}: one of {sorted(READOUTS)} or {TRAINING!r}")
    return readout


def is_native(cfg: dict, readout: str | None) -> bool:
    r = check(readout)
    return r is None or (READOUTS[r] == cfg.get("template") and READOUT_LAYOUT[r] == cfg.get("layout"))


def switch_backend(backend, readout: str | None):
    """The training-template ``DecisionModelBackend`` -> the same model under ``readout`` (readout_ab's swap)."""
    from heads.decide_backend import DecisionModelBackend

    r = check(readout)
    if r is None:
        return backend
    return DecisionModelBackend(backend.model, backend.encode, template=READOUTS[r], layout=READOUT_LAYOUT[r],
                                checkpoint=backend.checkpoint, shared_prefix=backend.shared_prefix)


def identity(cfg: dict, readout: str | None, tempered: bool) -> dict[str, Any]:
    r = check(readout)
    if r is None or is_native(cfg, r):
        return {"requested": readout or TRAINING, "template": cfg.get("template"), "layout": cfg.get("layout"),
                "native": True, "temperature_scope": "training-template fit" if tempered else "untempered"}
    return {"requested": r, "template": READOUTS[r], "layout": READOUT_LAYOUT[r], "native": False,
            "cell_template": cfg.get("template"), "cell_layout": cfg.get("layout"),
            "swap": "jobs.m2_first_cell.readout_ab.child_deciders (same model / encoder / checkpoint / shared_prefix)",
            "temperature_scope": (f"training-template fit (raw-template temperature, {cfg.get('template')}), applied "
                                  f"to readout {r}; no per-readout refit" if tempered else
                                  "untempered (identity calibration)")}
