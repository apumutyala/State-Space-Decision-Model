"""Job-params knob ``"readout"``: score a SAVED checkpoint under a named eval readout instead of its training template,
EXACTLY as ``jobs.m2_first_cell.readout_ab`` scores it (the champion = D1 job-c7pq2 art-bm3e6 + ``v3w``: 38.67
[35.50, 41.73] on the 2,008 subset, job-39gnn, independently reproduced).

Used by ``fulldev_eval score --arm child --readout v3w`` (full DEV) and ``train.m2_gates --stage hijack --readout v3w``
(mode hijack_eval). The swap is the one ``readout_ab.child_deciders`` makes: the SAME model object, encoder, checkpoint
record and ``shared_prefix`` as the training-template backend, another template + layout
(``readout_ab.READOUTS`` / ``READOUT_LAYOUT``). The checkpoint is restored and lineage-checked on the cell's own
template first (``Trainer.load``); only the prompt changes.

Temperature (served calibration). No per-readout refit is available yet: the served decider keeps the ONE temperature
fitted on the TRAINING template's calibration items (``fit_child_temperature``), recorded as
``temperature_scope = "training-template fit (raw-template temperature), applied to readout <r>"``. The Index is
computed on the untempered answers (T = 1) either way, so the refit can only move ECE, never the Index.
TODO(per-readout refit): when the refit lands on stack/rwkv7/ckpt-readout, implement ``refit_temperature`` below
(fit on the calibration items rendered under the readout) and set ``REFIT_AVAILABLE = True``; callers already route
through it.
"""
from __future__ import annotations

from typing import Any

from jobs.m2_first_cell.readout_ab import READOUT_LAYOUT, READOUTS

#: flipped when ``refit_temperature`` is implemented (per-readout temperature refit; see the module docstring)
REFIT_AVAILABLE = False
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


def refit_temperature(model, readout: str, calibration_items: Any):  # pragma: no cover - TODO hook
    """TODO(per-readout refit): fit the served temperature on the calibration items rendered under ``readout``."""
    raise NotImplementedError("per-readout temperature refit is not implemented on this branch")


def identity(cfg: dict, readout: str | None, tempered: bool) -> dict[str, Any]:
    r = check(readout)
    if r is None or is_native(cfg, r):
        return {"requested": readout or TRAINING, "template": cfg.get("template"), "layout": cfg.get("layout"),
                "native": True, "temperature_scope": "training-template fit" if tempered else "untempered"}
    return {"requested": r, "template": READOUTS[r], "layout": READOUT_LAYOUT[r], "native": False,
            "cell_template": cfg.get("template"), "cell_layout": cfg.get("layout"),
            "swap": "jobs.m2_first_cell.readout_ab.child_deciders (same model / encoder / checkpoint / shared_prefix)",
            "per_readout_refit": REFIT_AVAILABLE,
            "temperature_scope": (f"training-template fit (raw-template temperature, {cfg.get('template')}), applied "
                                  f"to readout {r}; no per-readout refit" if tempered else
                                  "untempered (identity calibration)")}
