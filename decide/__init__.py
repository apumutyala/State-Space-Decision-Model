"""Track D decision adapter: ``decide(request, state_handle=None) -> response``.

See ``decide/adapter.py`` for the wire/envelope contract and
``interfaces/DECISION_LANGUAGE_V0.md`` for the canonical interface.
"""

from .adapter import Decider, decide, set_default_backend
from .calibration import OptionCountTemperature
from .versions import ADAPTER_VERSION, ENVELOPE_SCHEMA, INTERFACE_VERSION

__all__ = [
    "ADAPTER_VERSION",
    "ENVELOPE_SCHEMA",
    "INTERFACE_VERSION",
    "Decider",
    "OptionCountTemperature",
    "decide",
    "set_default_backend",
]
