"""Bounded refinement plus independent regressions against five broken kernels."""
import pytest

from keel.host import HostInputs, HostKernel
from keel.host.model_check import InvariantViolation, check_bounds


class UnauthorizedKernel(HostKernel):
    def _decide(self, request, proposal, result):
        status, reason = super()._decide(request, proposal, result)
        if result.act and reason == "action_not_authorized":
            return "executed", "broken_model_grants_authority"
        return status, reason


class SideEffectOnDeferKernel(HostKernel):
    def process(self, request, proposal, decision_result, **kwargs):
        outcome = super().process(request, proposal, decision_result, **kwargs)
        if outcome.status == "deferred":
            self.executor(proposal)
        return outcome


class OverflowApprovalKernel(HostKernel):
    def _decide(self, request, proposal, result):
        status, reason = super()._decide(request, proposal, result)
        return ("executed", "broken_overflow_approval") if status == "held" else (status, reason)


class LaunderingInputs(HostInputs):
    def derive(self, segment_id, content, parents, *, kind="retrieved", source="derived"):
        # Deliberate bug: trust is inferred from requested kind, not parent lineage.
        return self.metadata(segment_id, content, source=source)


class LaunderingKernel(HostKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = LaunderingInputs()


class DuplicateReceiptKernel(HostKernel):
    def process(self, request, proposal, decision_result, **kwargs):
        outcome = super().process(request, proposal, decision_result, **kwargs)
        self.receipt_log.append(self.receipt_log.receipts()[-1])
        return outcome


def test_exhaustive_abstract_model_and_concrete_refinement():
    report = check_bounds()
    assert report.configurations == 48
    assert report.max_depth == 4
    assert report.states == 632
    assert report.transitions == report.concrete_transitions == 16920


@pytest.mark.parametrize("broken,invariant", [(UnauthorizedKernel, "I1"), (SideEffectOnDeferKernel, "I2"),
                                               (OverflowApprovalKernel, "I3"), (LaunderingKernel, "I4"),
                                               (DuplicateReceiptKernel, "I5")])
def test_checker_catches_each_deliberately_broken_invariant(broken, invariant):
    with pytest.raises(InvariantViolation) as failure:
        check_bounds(kernel_type=broken, max_depth=2)
    assert failure.value.invariant == invariant


@pytest.mark.parametrize("depth", [-1, 5, True, 1.5])
def test_checker_rejects_unclaimed_bounds(depth):
    with pytest.raises(ValueError):
        check_bounds(max_depth=depth)
