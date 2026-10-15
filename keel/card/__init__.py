"""Decision Alignment Card: eval-only paired units, bounded attacks and statistics."""
from .units import CardUnit, evidence_filter, make_static_units
from .metrics import compare_arms, summarize
from .search import Candidate, SearchResult, search

__all__ = ["CardUnit", "Candidate", "SearchResult", "compare_arms", "evidence_filter", "make_static_units", "search", "summarize"]
