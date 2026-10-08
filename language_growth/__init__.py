"""CPU-only, evidence-relative L0 semantic language boundary."""

from .kernel import check_program, validate_release
from .types import AskContent, CheckResult, Comparison, ConstantValue, ExactDifference, Fact, SpeechAct

__all__ = ["AskContent", "CheckResult", "Comparison", "ConstantValue", "ExactDifference", "Fact", "SpeechAct", "check_program", "validate_release"]
