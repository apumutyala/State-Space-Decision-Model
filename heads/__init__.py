"""Track D m2 decision heads over a backbone-agnostic hidden-state interface.

``layout``   options-first prompt rows with token spans (ordered = m0 prompt; symmetric = equivariant)
``backbone`` ``HiddenStateBackbone`` wrapper, candidate backbone specs, tiny CPU test backbones
``features`` rows -> ``HeadInput`` (shared pass, option embeddings, optional recompute fork)
``heads``    integrator / fork / snapshot / head_b / clef_control (priced control)
``typed``    noul / choice (<=255) / score (cumulative link)
``model``    ``DecisionModel``
"""

from .heads import HEADS, make_head
from .layout import LAYOUTS, ByteEncoder, LayoutRow, build_row
from .model import DecisionModel

__all__ = ["HEADS", "LAYOUTS", "ByteEncoder", "DecisionModel", "LayoutRow", "build_row", "make_head"]
