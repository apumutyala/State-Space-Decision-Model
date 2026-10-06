"""Adaptive (defence-aware) attack interface — STUB for m0; implementations are due before the m2 gate.

The m2 defences put provenance into the model as channel embeddings: ISE (a learned per-source offset added to
token embeddings, arXiv 2410.09102) and/or ASIDE (a fixed orthogonal rotation of untrusted-token embeddings,
arXiv 2503.10566). An adaptive attacker (Nasr, Carlini et al., 2510.09023: "the attacker moves second") is
given the defence specification, not just the model, and optimises against the defended forward pass:

  * white-box GCG through the defended embedding path: gradients w.r.t. one-hot tokens of the untrusted span are
    taken AFTER the channel transform (rotation/offset applied), so the search knows which tokens survive it;
  * channel-mimicry: search for untrusted-span tokens whose transformed embeddings land near trusted-channel
    instruction embeddings (minimise distance to the trusted-channel centroid of an instruction phrase);
  * attention/recurrence-guided variant (2507.07417 analogue for recurrent state): maximise the decision head's
    accounting attribution to the injected span;
  * D1-style content attacks carry no instruction and are unaffected by channel separation by construction; they
    remain in the adaptive suite unchanged as the floor.

A concrete attack implements ``AdaptiveAttack.craft``. The harness calls it per request, appends/inserts the
returned untrusted text via ``attacks.suffix_variant`` (or a full field replacement), and scores the result
through the same ``decide`` path, so adaptive results share the metric table with every other family
(``attack_family="adaptive"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ChannelDefenceSpec:
    """What the attacker is told about the defence (everything except trained weights it cannot read)."""
    kind: Literal["ise", "aside", "ise+aside", "datamark", "none"]
    channels: tuple[str, ...] = ("trusted", "untrusted")
    # ASIDE: the rotation is public (fixed, parameter-free); ISE: offsets are model weights (white-box only)
    rotation: Any | None = None          # e.g. a [H, H] orthogonal matrix or a seed that regenerates it
    offsets: Any | None = None           # e.g. {channel: [H] tensor} for white-box evaluation
    sanitiser: str = "zwsp+token-check"  # safety_eval.sanitize policy in force
    notes: dict = field(default_factory=dict)


@dataclass
class AttackResult:
    untrusted_text: str                  # full replacement text for the untrusted field
    objective: Literal["steer", "hijack"]
    target_option: str
    queries: int                         # decider calls / forward passes spent (budget accounting)
    white_box: bool
    trace: dict = field(default_factory=dict)


class DefendedModel(Protocol):
    """Handle the adaptive attacker receives: the decide() callable plus (white-box) the defended forward."""

    def decide(self, request: dict) -> dict: ...

    def defended_embeddings(self, token_ids: Any, channel_ids: Any) -> Any: ...


class AdaptiveAttack:
    name = "adaptive-stub"

    def __init__(self, spec: ChannelDefenceSpec, budget: dict):
        self.spec, self.budget = spec, budget

    def craft(self, request: dict, model: DefendedModel, objective: str, target_option: str) -> AttackResult:
        raise NotImplementedError(
            "adaptive attacks are pre-registered for m2 (safety_eval/PREREG.md section 2.4); "
            "m0 ships only this interface")
