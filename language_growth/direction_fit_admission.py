"""Fail-closed parent selection for native direction fitting.

This small boundary is kept independent of model runtimes so its CLI routing
can be tested without importing Torch or reconstructing a checkpoint.
"""
from __future__ import annotations


def resolve_direction_fit_parent(*, candidate_checkpoint, parent_report, parent_report_sha256,
                                 parent_qualification, parent_qualification_sha256,
                                 parent_snapshot, stage1_parent_checkpoint=None,
                                 fit_receipt=None, fit_receipt_sha256=None, device="cpu",
                                 dtype="bfloat16", load_qualified_parent, load_language_child):
    """Route a qualified parent or exact receipt-pinned child to the fitter."""
    child_inputs = (stage1_parent_checkpoint, fit_receipt, fit_receipt_sha256)
    if any(child_inputs) and not all(child_inputs):
        raise ValueError("child direction fitting requires the Stage 1 checkpoint and pinned fit receipt")
    if all(child_inputs):
        if parent_snapshot is None:
            raise ValueError("child direction fitting requires strict local snapshot reconstruction")
        return load_language_child(
            candidate_checkpoint, stage1_checkpoint=stage1_parent_checkpoint,
            receipt=fit_receipt, receipt_sha256=fit_receipt_sha256,
            report=parent_report, report_sha256=parent_report_sha256,
            qualification=parent_qualification,
            qualification_sha256=parent_qualification_sha256,
            snapshot=parent_snapshot, device=device, dtype=dtype)
    return load_qualified_parent(
        candidate_checkpoint, report=parent_report,
        report_sha256=parent_report_sha256, qualification=parent_qualification,
        qualification_sha256=parent_qualification_sha256,
        snapshot=parent_snapshot, device=device, dtype=dtype)
