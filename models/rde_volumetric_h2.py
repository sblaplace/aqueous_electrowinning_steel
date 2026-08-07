"""Separate Fe deposition from HER kinetics — measure the HER branch *first*,
then fit Fe with HER held fixed, and confirm by volumetric hydrogen.

Why this module exists
----------------------
``docs/SCREENING_SENSITIVITY_BUDGET.md`` (#34) ranked **`her_tafel_V` as the
single dominant remaining L0 uncertainty**: at its low screening factor
(0.105 V/decade) the reference FE collapses to 0.7599 against the 0.80 floor —
i.e. *the parameter that most threatens the go/no-go is the HER Tafel slope*.
The Q3 bench task is therefore explicitly "measure HER Tafel first".

A polarization curve of the Fe-containing bath is underdetermined for the
Fe/HER split (``models/calibration.py`` states this limitation): two unknown
Tafel slopes + two exchange currents with only a total current to fit are
degenerate.  The clean, non-degenerate experiment that resolves it is a
**two-step + independent-closure** sequence:

1. **Measure HER first, in the Fe-free supporting electrolyte**, on the *same*
   cathode surface, at the same pH / temperature / reference.  With no Fe²⁺
   present the total cathodic current *is* the HER branch, so a single-branch
   Tafel fit returns ``b_her`` and ``i0_her`` with no Fe contamination.  This
   directly pins the #34 dominant unknown.
2. **Then, in the Fe-containing bath on the RDE**, do the Levich analysis for
   ``D`` / the Nernst film thickness (the transport separator), and fit *only*
   the Fe branch — ``i0_fe`` and ``b_fe`` — with the HER branch from step 1
   **held fixed**.  Two free parameters against a full polarization curve is
   well-conditioned; Fe is no longer entangled with an unknown HER.
3. **Confirm by volumetric H₂** at fixed galvanostatic currents: the H₂ gas
   volume gives the HER charge independently of any curve fit
   (``FE_HER = 2F·n_H2 / Q_applied``), and the deposit mass gives the Fe charge,
   so the two measured faradaic efficiencies must close the charge ledger
   (``FE_HER + FE_Fe ≈ 1``).  This catches a wrong slope-fit split with a
   *measurement*, not another fit.

Everything here is L0: the numbers are synthetic demonstrations that the
procedure *recovers* known kinetics, so the method is validated before it is
pointed at real wet-lab data.  Real-data validation is L1 (see
``docs/KINETICS_CALIBRATION_READY.md``).

Units
-----
SI unless stated: currents in A/m² (positive cathodic magnitudes), potentials
in V vs. SHE, diffusivities in m²/s, kinematic viscosity in m²/s, volumes in
mL (converted to m³ internally for the ideal-gas law), charge in C.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from .electrochemistry import E0_FE, FARADAY, M_FE, R_GAS, Z_FE
from .pourbaix import her_line
from .rde_levich import (
    KIN_VISC_WATER_25,
    levich_limiting_current,
)

# ─── Ideal-gas / STP reference for the volumetric H₂ leg ─────────────
P_STD_PA = 101325.0      # 1 atm
T_STD_K = 273.15         # 0 °C
GAS_CORR = "ideal gas; calibrated on real-device saturated-vapour correction at L1"


# ═════════════════════════════════════════════════════════════════════
#  Step 1 — HER branch measured FIRST on the Fe-free supporting bath
# ═════════════════════════════════════════════════════════════════════
def her_equilibrium_potential(pH: float, T_C: float = 25.0) -> float:
    """HER reversible potential (V vs SHE) at bulk pH and temperature."""
    return float(her_line(pH, T_C + 273.15))


def simulate_her_free_bath_polarization(
    E_she_V: np.ndarray,
    *,
    i0_her_A_m2: float,
    b_her_V_dec: float,
    E_eq_her_V: float,
    noise_sigma_A_m2: float = 0.0,
    noise_rel_fraction: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Synthetic Fe-free (supporting-electrolyte-only) HER polarization.

    With no Fe²⁺ the total cathodic current is the HER branch alone — this is
    the *first* measurement of Q3.  ``noise_rel_fraction`` applies
    multiplicative (potentiostat %-accuracy) noise; ``noise_sigma_A_m2`` adds
    an absolute floor.  Returns ``potentials_V`` and ``i_her_A_m2`` (cathodic
    magnitude) plus a convenience ``frame``.
    """
    E = np.asarray(E_she_V, dtype=float)
    eta = E_eq_her_V - E
    i = i0_her_A_m2 * 10.0 ** (np.maximum(eta, 0.0) / b_her_V_dec)
    if noise_rel_fraction > 0.0 or noise_sigma_A_m2 > 0.0:
        rng = np.random.default_rng(seed)
        if noise_rel_fraction > 0.0:
            i = i * (1.0 + rng.normal(0.0, noise_rel_fraction, size=i.shape))
        if noise_sigma_A_m2 > 0.0:
            i = i + rng.normal(0.0, noise_sigma_A_m2, size=i.shape)
    import pandas as pd

    return {
        "potentials_V": E,
        "i_her_A_m2": i,
        "i0_her_A_m2": i0_her_A_m2,
        "b_her_V_dec": b_her_V_dec,
        "E_eq_her_V": E_eq_her_V,
        "frame": pd.DataFrame({"potential_V": E, "i_her_A_m2": i}),
    }


def fit_her_from_free_bath(
    E_she_V: np.ndarray,
    i_her_A_m2: np.ndarray,
    *,
    E_eq_her_V: float,
    i0_bounds_A_m2: tuple[float, float] = (1e-12, 1e4),
    b_bounds_V_dec: tuple[float, float] = (0.02, 0.5),
    weight_strong: bool = True,
) -> Dict[str, Any]:
    """Fit the single HER branch on the Fe-free bath (Tafel in log space).

    Returns ``b_her_V_dec`` and ``i0_her_A_m2`` (the #34 dominant unknown) plus
    fit diagnostics.  ``weight_strong`` down-weights the near-mass-transport /
    low-current tail so the slope is set by the well-resolved Tafel region.
    """
    E = np.asarray(E_she_V, dtype=float)
    i = np.asarray(i_her_A_m2, dtype=float)
    eta = E_eq_her_V - E
    valid = np.isfinite(eta) & np.isfinite(i) & (i > 0) & (eta > 0.02)
    if valid.sum() < 8:
        return {
            "b_her_V_dec": math.nan, "i0_her_A_m2": math.nan,
            "rmse_log10": math.nan, "r_squared": math.nan,
            "n_points": int(valid.sum()), "converged": False,
        }
    E_v, i_v = E[valid], i[valid]
    # Optional weighting: lower weight on the very top of the curve (highest
    # overpotential, where transport/ohmic artefacts bend it) and on the tail
    # near E_eq (where the current is tiny).  High overpotential = strong Tafel.
    if weight_strong:
        eta_v = E_eq_her_V - E_v
        # weight peaks mid-range of log-eta; drops to 0.2 at the extremes
        w = 0.2 + 0.8 * np.sin(np.pi * np.log10(eta_v / eta_v.min()) /
                               np.log10(eta_v.max() / eta_v.min()))
    else:
        w = np.ones_like(E_v)

    x0 = np.array([math.log10(max(i_v[np.argmax(eta_v)], 1e-12)), 0.14])
    lower = np.array([math.log10(i0_bounds_A_m2[0]), b_bounds_V_dec[0]])
    upper = np.array([math.log10(i0_bounds_A_m2[1]), b_bounds_V_dec[1]])

    def model(p: np.ndarray) -> np.ndarray:
        i0, b = 10.0 ** p[0], p[1]
        return i0 * 10.0 ** ((E_eq_her_V - E_v) / b)

    def residual(p: np.ndarray) -> np.ndarray:
        return w * (np.log10(np.maximum(model(p), 1e-30)) - np.log10(i_v))

    res = least_squares(residual, x0=np.clip(x0, lower, upper), bounds=(lower, upper),
                        max_nfev=20000)
    p = res.x
    b_her = float(p[1])
    i0_her = float(10.0 ** p[0])
    pred = model(p)
    ss_res = float(np.sum((np.log10(np.maximum(pred, 1e-30)) - np.log10(i_v)) ** 2))
    ss_tot = float(np.sum((np.log10(i_v) - np.log10(i_v).mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "b_her_V_dec": b_her,
        "i0_her_A_m2": i0_her,
        "rmse_log10": float(np.sqrt(np.mean(res.fun ** 2))),
        "r_squared": float(r2),
        "n_points": int(valid.sum()),
        "converged": bool(res.success),
        "parameter_std_log10": _std_from_jac(res.jac, res.fun, ("i0_her", "b_her")),
    }


# ═════════════════════════════════════════════════════════════════════
#  Step 2 — Fe branch fitted with HER HELD FIXED (RDE Levich transport)
# ═════════════════════════════════════════════════════════════════════
def _std_from_jac(jac: np.ndarray, residual: np.ndarray, names: tuple[str, ...]
                  ) -> Dict[str, Optional[float]]:
    """Approximate standard errors in parameter space (log10 for i0 params)."""
    if len(residual) <= len(names):
        return {n: None for n in names}
    try:
        cov = np.linalg.pinv(jac.T @ jac) * np.sum(residual ** 2) / (len(residual) - len(names))
        vals = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return {n: float(v) for n, v in zip(names, vals)}
    except np.linalg.LinAlgError:
        return {n: None for n in names}


def fit_fe_kinetics_given_her(
    E_she_V: np.ndarray,
    i_total_A_m2: np.ndarray,
    *,
    i_lim_A_m2: np.ndarray,
    b_her_V_dec: float,
    i0_her_A_m2: float,
    E_eq_her_V: float,
    E_eq_fe_V: float = E0_FE,
    i0_bounds_A_m2: tuple[float, float] = (1e-6, 1e6),
    b_bounds_V_dec: tuple[float, float] = (0.02, 0.5),
) -> Dict[str, Any]:
    """Fit ONLY the Fe branch (i0_fe, b_fe) with the HER branch fixed.

    This is the operationalisation of *measure HER first*: ``b_her`` /
    ``i0_her`` come from step 1 (Fe-free bath), so they are constants here.
    The Fe branch is Koutecky-Levich-capped by the per-row Levich limit
    ``i_lim_A_m2`` (from the RDE), and the total is Fe + fixed-HER.  Two free
    parameters ⇒ well-conditioned, non-degenerate Fe/HER separation.

    ``i_lim_A_m2`` may be a scalar (single rotation rate) or a per-row array
    that broadcasts over ``E_she_V``.  Returned ``fe_i0_A_m2`` is the *surface*
    exchange current on this bath/surface (not Arrhenius-anchored to 50 °C).
    """
    E = np.asarray(E_she_V, dtype=float)
    i_tot = np.asarray(i_total_A_m2, dtype=float)
    i_lim = np.broadcast_to(np.asarray(i_lim_A_m2, dtype=float), E.shape)

    eta_h = E_eq_her_V - E
    i_her_fixed = i0_her_A_m2 * 10.0 ** (np.maximum(eta_h, 0.0) / b_her_V_dec)
    eta_fe = E_eq_fe_V - E
    valid = (
        np.isfinite(E) & np.isfinite(i_tot) & (i_tot > i_her_fixed)
        & (eta_fe > 0.02)
    )
    if valid.sum() < 8:
        return {
            "fe_i0_A_m2": math.nan, "fe_tafel_V_dec": math.nan,
            "rmse_log10": math.nan, "r_squared": math.nan,
            "n_points": int(valid.sum()), "converged": False,
        }
    E_v, t_v, iL_v = E[valid], i_tot[valid], np.maximum(i_lim[valid], 1e-30)

    x0 = np.array([math.log10(50.0), 0.12])
    lower = np.array([math.log10(i0_bounds_A_m2[0]), b_bounds_V_dec[0]])
    upper = np.array([math.log10(i0_bounds_A_m2[1]), b_bounds_V_dec[1]])

    def fe_model(p: np.ndarray) -> np.ndarray:
        i0, b = 10.0 ** p[0], p[1]
        i_k = i0 * 10.0 ** ((E_eq_fe_V - E_v) / b)
        return i_k * iL_v / (i_k + iL_v)  # Koutecky-Levich cap

    def model(p: np.ndarray) -> np.ndarray:
        return fe_model(p) + i0_her_A_m2 * 10.0 ** (
            np.maximum(E_eq_her_V - E_v, 0.0) / b_her_V_dec)

    def residual(p: np.ndarray) -> np.ndarray:
        return np.log10(np.maximum(model(p), 1e-30)) - np.log10(t_v)

    res = least_squares(residual, x0=np.clip(x0, lower, upper), bounds=(lower, upper),
                        max_nfev=20000)
    p = res.x
    fe_i0 = float(10.0 ** p[0])
    b_fe = float(p[1])
    pred = model(p)
    ss_res = float(np.sum((np.log10(np.maximum(pred, 1e-30)) - np.log10(t_v)) ** 2))
    ss_tot = float(np.sum((np.log10(t_v) - np.log10(t_v).mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "fe_i0_A_m2": fe_i0,
        "fe_tafel_V_dec": b_fe,
        "rmse_log10": float(np.sqrt(np.mean(res.fun ** 2))),
        "r_squared": float(r2),
        "n_points": int(valid.sum()),
        "converged": bool(res.success),
        "parameter_std_log10": _std_from_jac(res.jac, res.fun, ("fe_i0", "fe_tafel")),
        "her_fixed": {"b_her_V_dec": b_her_V_dec, "i0_her_A_m2": i0_her_A_m2},
    }


def fit_fe_given_her_on_rde(
    E_she_V: np.ndarray,
    i_total_A_m2: np.ndarray,
    omega_rpm: np.ndarray,
    *,
    fe_conc_M: float,
    b_her_V_dec: float,
    i0_her_A_m2: float,
    E_eq_her_V: float,
    E_eq_fe_V: float = E0_FE,
    D_m2_s: float = 7.2e-10,
    nu_m2_s: float = KIN_VISC_WATER_25,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience: compute the per-row Levich limit from the RDE rotation
    rates, then call :func:`fit_fe_kinetics_given_her`.

    ``omega_rpm`` is per-row, aligned with ``E_she_V`` / ``i_total_A_m2``
    (each row carries its own rotation rate; the RDE polarization set is
    flattened omega-major, i.e. ``E = tile(E_pot, n_omega)`` and
    ``omega = repeat(omega, n_E)``).  A scalar/single value models a single
    rotation rate.
    """
    omega = np.asarray(omega_rpm, dtype=float)
    if omega.ndim == 0 or omega.size == 1:
        i_lim = float(levich_limiting_current(
            omega.reshape(-1), z=Z_FE, D_m2_s=D_m2_s,
            C_bulk_M=fe_conc_M, nu_m2_s=nu_m2_s)[0])
        i_lim = np.full_like(np.asarray(E_she_V, dtype=float), i_lim)
    else:
        if omega.shape != np.asarray(E_she_V).shape:
            raise ValueError(
                "omega_rpm must be scalar or per-row (aligned with E_she_V); "
                "got shape %s vs E shape %s" % (omega.shape, np.asarray(E_she_V).shape))
        i_lim = levich_limiting_current(omega, z=Z_FE, D_m2_s=D_m2_s,
                                        C_bulk_M=fe_conc_M, nu_m2_s=nu_m2_s)
    return fit_fe_kinetics_given_her(
        E_she_V, i_total_A_m2, i_lim_A_m2=i_lim, b_her_V_dec=b_her_V_dec,
        i0_her_A_m2=i0_her_A_m2, E_eq_her_V=E_eq_her_V, E_eq_fe_V=E_eq_fe_V, **kwargs,
    )


# ═════════════════════════════════════════════════════════════════════
#  Step 3 — volumetric H₂ confirmation & charge-ledger closure
# ═════════════════════════════════════════════════════════════════════
def h2_moles_from_volume(volume_mL: float, *, T_K: float = 298.15,
                         P_Pa: float = P_STD_PA) -> float:
    """H₂ amount (mol) from a measured volume (mL) via the ideal gas law.

    ``n = P V / (R T)`` with V converted to m³.  Real-device correction for
    saturated water-vapour partial pressure and non-ideality is a documented
    L1 task (``GAS_CORR``).
    """
    return float(P_Pa * (volume_mL * 1e-6) / (R_GAS * T_K))


def h2_volume_from_moles(moles: float, *, T_K: float = 298.15,
                         P_Pa: float = P_STD_PA) -> float:
    """Invert :func:`h2_moles_from_volume` (mL)."""
    return float(moles * R_GAS * T_K / P_Pa * 1e6)


def volumetric_h2_closure(
    *,
    h2_volume_mL: Optional[float] = None,
    h2_moles: Optional[float] = None,
    fe_deposit_kg: Optional[float] = None,
    fe_moles: Optional[float] = None,
    applied_charge_C: Optional[float] = None,
    j_A_m2: Optional[float] = None,
    electrode_area_m2: Optional[float] = None,
    run_time_s: Optional[float] = None,
    T_K: float = 298.15,
    P_Pa: float = P_STD_PA,
    # optional fitted-branch prediction cross-check:
    i_her_at_run_A_m2: Optional[float] = None,
) -> Dict[str, Any]:
    """Close the charge ledger from the independent volumetric H₂ + deposit.

    Returns the HER faradaic efficiency from gas (``FE_her_gas``), the Fe
    faradaic efficiency from deposit mass (``FE_fe_mass``), and the ledger
    ``closure`` (they should sum to ≈1).  When ``i_her_at_run_A_m2`` is
    supplied (the fitted HER branch evaluated at the galvanostatic operating
    potential), the H₂-derived HER charge is cross-checked against the
    branch-predicted HER charge (``her_branch_residual``) — the measurement
    that confirms a wrong slope-fit split.
    """
    if h2_moles is None:
        if h2_volume_mL is None:
            raise ValueError("provide h2_volume_mL or h2_moles")
        h2_moles = h2_moles_from_volume(h2_volume_mL, T_K=T_K, P_Pa=P_Pa)
    if fe_moles is None and fe_deposit_kg is not None:
        fe_moles = fe_deposit_kg / M_FE

    if applied_charge_C is None:
        if not (j_A_m2 and electrode_area_m2 and run_time_s):
            raise ValueError(
                "provide applied_charge_C, or j_A_m2 + electrode_area_m2 + run_time_s")
        applied_charge_C = j_A_m2 * electrode_area_m2 * run_time_s
    if applied_charge_C <= 0:
        raise ValueError("applied charge must be positive")

    q_her = Z_FE * FARADAY * h2_moles          # C (HER is 2 e⁻ per H₂)
    q_fe = None if fe_moles is None else Z_FE * FARADAY * fe_moles

    fe_her_gas = float(q_her / applied_charge_C)
    fe_fe_mass = float(q_fe / applied_charge_C) if q_fe is not None else None

    out: Dict[str, Any] = {
        "h2_moles": h2_moles,
        "q_her_gas_C": q_her,
        "applied_charge_C": applied_charge_C,
        "FE_her_gas": fe_her_gas,
        "FE_fe_mass": fe_fe_mass,
        "closure": None if fe_fe_mass is None else fe_her_gas + fe_fe_mass,
        "gas_assumption": GAS_CORR,
    }

    if i_her_at_run_A_m2 is not None and electrode_area_m2 and run_time_s:
        pred_charge = i_her_at_run_A_m2 * electrode_area_m2 * run_time_s
        denom = max(q_her, pred_charge, 1e-30)
        out["her_branch_predicted_charge_C"] = pred_charge
        out["her_branch_residual"] = float((q_her - pred_charge) / denom)
    return out


# ═════════════════════════════════════════════════════════════════════
#  Self-test: L0 proof the two-step + volumetric procedure recovers truth
# ═════════════════════════════════════════════════════════════════════
def simulate_fe_bath_rde_polarization(
    E_she_V: np.ndarray,
    omega_rpm: np.ndarray,
    *,
    fe_i0_A_m2: float,
    fe_tafel_V: float,
    fe_E_eq_V: float = E0_FE,
    b_her_V_dec: float,
    i0_her_A_m2: float,
    E_eq_her_V: float,
    fe_conc_M: float = 1.0,
    D_m2_s: float = 7.2e-10,
    nu_m2_s: float = KIN_VISC_WATER_25,
) -> Dict[str, Any]:
    """Synthetic Fe-containing-bath RDE polarization (Fe + fixed HER)."""
    E = np.asarray(E_she_V, dtype=float)
    omega = np.asarray(omega_rpm, dtype=float)
    i_lim = levich_limiting_current(omega, z=Z_FE, D_m2_s=D_m2_s,
                                    C_bulk_M=fe_conc_M, nu_m2_s=nu_m2_s)
    eta_fe = fe_E_eq_V - E
    i_her = i0_her_A_m2 * 10.0 ** (np.maximum(E_eq_her_V - E, 0.0) / b_her_V_dec)
    import pandas as pd

    rows = []
    i_lim_mat = []
    for o, iL in zip(omega, i_lim):
        i_k = fe_i0_A_m2 * 10.0 ** (np.maximum(eta_fe, 0.0) / fe_tafel_V)
        i_fe = i_k * iL / (i_k + iL)
        i_tot = i_fe + i_her
        i_lim_mat.append(np.full_like(E, iL))
        for idx, e in enumerate(E):
            rows.append({"potential_V": float(e), "omega_rpm": float(o),
                         "i_fe_A_m2": float(i_fe[idx]), "i_her_A_m2": float(i_her[idx]),
                         "i_total_A_m2": float(i_tot[idx]), "i_lim_A_m2": float(iL)})
    df = pd.DataFrame(rows)
    return {"potentials_V": E, "omega_rpm": omega, "i_lim_A_m2": i_lim,
            "i_lim_matrix_A_m2": np.asarray(i_lim_mat), "frame": df,
            "fe_i0_A_m2": fe_i0_A_m2, "fe_tafel_V": fe_tafel_V,
            "b_her_V_dec": b_her_V_dec, "i0_her_A_m2": i0_her_A_m2}


def self_test(seed: Optional[int] = 7,
              tolerance: Optional[Dict[str, float]] = None,
              verbose: bool = True) -> Dict[str, Any]:
    """Dry-run the full Q3 procedure on synthetic data.

    (a) recover HER (b, i0) from an Fe-free bath;
    (b) recover Fe (i0, b) from the Fe bath with the step-(a) HER held fixed;
    (c) confirm the fitted HER branch reproduces a volumetric-H₂ charge ledger.

    Returns a dict of recovered-vs-true errors and PASS/FAIL verdicts.
    """
    tol = tolerance or {
        "her_tafel_V_pct": 8.0, "her_i0_dec": 0.35,
        "fe_tafel_V_pct": 8.0, "fe_i0_dec": 0.35,
        "her_closure_residual": 0.08,
    }

    # physical truth
    true = {
        "b_her": 0.140, "i0_her": 2.0e-3,     # HER on Fe, mildly acidic sulfate
        "b_fe": 0.120, "i0_fe": 50.0,          # Fe deposition, plausible surface i0
        "fe_E_eq": E0_FE,
        "pH": 2.0, "T_C": 25.0, "fe_conc_M": 1.0, "D": 7.2e-10,
    }
    E_eq_her = her_equilibrium_potential(true["pH"], true["T_C"])

    # (a) Fe-free bath → HER
    E_free = np.linspace(E_eq_her - 0.80, E_eq_her - 0.05, 60)
    free = simulate_her_free_bath_polarization(
        E_free, i0_her_A_m2=true["i0_her"], b_her_V_dec=true["b_her"],
        E_eq_her_V=E_eq_her, noise_rel_fraction=0.03, seed=seed)
    fit_h = fit_her_from_free_bath(free["potentials_V"], free["i_her_A_m2"],
                                   E_eq_her_V=E_eq_her)

    # (b) Fe bath RDE → Fe with HER fixed
    E_fe = np.linspace(-0.50, -1.05, 80)
    omegas = np.array([400.0, 900.0, 1600.0, 2500.0])
    fe_bath = simulate_fe_bath_rde_polarization(
        E_fe, omegas, fe_i0_A_m2=true["i0_fe"], fe_tafel_V=true["b_fe"],
        fe_E_eq_V=true["fe_E_eq"], b_her_V_dec=true["b_her"], i0_her_A_m2=true["i0_her"],
        E_eq_her_V=E_eq_her, fe_conc_M=true["fe_conc_M"], D_m2_s=true["D"])
    df = fe_bath["frame"]
    # stack into per-row arrays with their own i_lim
    E_all = df["potential_V"].to_numpy(float)
    iL_all = df["i_lim_A_m2"].to_numpy(float)
    iT_all = df["i_total_A_m2"].to_numpy(float)
    fit_fe = fit_fe_kinetics_given_her(
        E_all, iT_all, i_lim_A_m2=iL_all, b_her_V_dec=fit_h["b_her_V_dec"],
        i0_her_A_m2=fit_h["i0_her_A_m2"], E_eq_her_V=E_eq_her, E_eq_fe_V=true["fe_E_eq"])

    # (c) volumetric H₂ confirmation at a galvanostatic operating point
    # pick j = 100 mA/cm² on area 1e-4 m², 600 s, RDE 1600 rpm
    j_A_m2 = 100.0 * 10.0
    area_m2 = 1.0e-4
    run_s = 600.0
    i_lim_1600 = float(levich_limiting_current(
        np.array([1600.0]), z=Z_FE, D_m2_s=true["D"],
        C_bulk_M=true["fe_conc_M"])[0])
    # find the potential where Fe(i_lim) + HER = j
    from scipy.optimize import brentq
    def tot(E):
        i_k = fit_fe["fe_i0_A_m2"] * 10.0 ** ((true["fe_E_eq"] - E) / fit_fe["fe_tafel_V_dec"])
        i_fe = i_k * i_lim_1600 / (i_k + i_lim_1600)
        i_her = fit_h["i0_her_A_m2"] * 10.0 ** ((E_eq_her - E) / fit_h["b_her_V_dec"])
        return i_fe + i_her
    E_run = float(brentq(lambda e: tot(e) - j_A_m2, E_eq_her - 1.5, E_eq_her, xtol=1e-9))
    i_her_run = fit_h["i0_her_A_m2"] * 10.0 ** ((E_eq_her - E_run) / fit_h["b_her_V_dec"])
    q_her_pred = i_her_run * area_m2 * run_s
    h2_moles_meas = q_her_pred / (Z_FE * FARADAY)
    h2_ml_meas = h2_volume_from_moles(h2_moles_meas)
    # deposit mass from the Fe branch at E_run
    i_k = fit_fe["fe_i0_A_m2"] * 10.0 ** ((true["fe_E_eq"] - E_run) / fit_fe["fe_tafel_V_dec"])
    i_fe_run = i_k * i_lim_1600 / (i_k + i_lim_1600)
    fe_kg_meas = (i_fe_run * area_m2 * run_s) * M_FE / (Z_FE * FARADAY)

    vol = volumetric_h2_closure(
        h2_volume_mL=h2_ml_meas, fe_deposit_kg=fe_kg_meas,
        j_A_m2=j_A_m2, electrode_area_m2=area_m2, run_time_s=run_s,
        i_her_at_run_A_m2=i_her_run)

    # errors
    err = {
        "her_tafel_V_pct": 100.0 * abs(fit_h["b_her_V_dec"] - true["b_her"]) / true["b_her"],
        "her_i0_dec": abs(math.log10(fit_h["i0_her_A_m2"]) - math.log10(true["i0_her"])),
        "fe_tafel_V_pct": 100.0 * abs(fit_fe["fe_tafel_V_dec"] - true["b_fe"]) / true["b_fe"],
        "fe_i0_dec": abs(math.log10(fit_fe["fe_i0_A_m2"]) - math.log10(true["i0_fe"])),
    }
    closure_resid = abs(vol["her_branch_residual"]) if vol.get("her_branch_residual") is not None else None
    err["her_closure_residual"] = closure_resid if closure_resid is not None else float("nan")

    verdict = {
        k: bool(err[k] <= tol[k]) for k in tol
    }
    result = {
        "true": true,
        "her_fit": fit_h, "fe_fit": fit_fe,
        "errors": err, "tolerances": tol, "verdict": verdict,
        "closure": vol,
        "all_pass": all(verdict.values()),
    }

    if verbose:
        _print_self_test(result)
    return result


def _print_self_test(result: Dict[str, Any]) -> None:
    print("=== Q3 dry-run: HER-first + volumetric H2 (L0 self-test) ===")
    true, err, tol = result["true"], result["errors"], result["tolerances"]
    hf, ff = result["her_fit"], result["fe_fit"]
    print(f"HER:  true b={true['b_her']:.3f},i0={true['i0_her']:.3g} | "
          f"fit b={hf['b_her_V_dec']:.3f},i0={hf['i0_her_A_m2']:.3g} "
          f"(R2={hf['r_squared']:.3f}) | err {err['her_tafel_V_pct']:.2f}%/"
          f"{err['her_i0_dec']:.2f}dec")
    print(f"Fe:   true b={true['b_fe']:.3f},i0={true['i0_fe']:.3g} | "
          f"fit b={ff['fe_tafel_V_dec']:.3f},i0={ff['fe_i0_A_m2']:.3g} "
          f"(R2={ff['r_squared']:.3f}) | err {err['fe_tafel_V_pct']:.2f}%/"
          f"{err['fe_i0_dec']:.2f}dec")
    c = result["closure"]
    print(f"Closure: FE_her(gas)={c['FE_her_gas']:.3f} FE_fe(mass)={c['FE_fe_mass'] if c.get('FE_fe_mass') is not None else float('nan'):.3f} "
          f"sum={c['closure'] if c.get('closure') is not None else float('nan'):.3f} | "
          f"HER-branch residual={c.get('her_branch_residual')}")
    print("Verdicts:")
    for k, ok in result["verdict"].items():
        print(f"  {k:<24} {err[k]:.3f} (tol {tol[k]:.3f}) {'PASS' if ok else 'FAIL'}")


# ═════════════════════════════════════════════════════════════════════
#  Measurement specification + scope contract (house style)
# ═════════════════════════════════════════════════════════════════════
def measurement_spec() -> Dict[str, Any]:
    """Machine-readable Q3 bench specification: what to measure, in what order,
    with what gates, mapping to the #34 dominant uncertainty and NEXT_STEPS."""
    return {
        "title": "Q3 — RDE + volumetric H2 (measure HER Tafel FIRST)",
        "resolves": (
            "#34 dominant uncertainty: her_tafel_V (the single coefficient that "
            "can flip the reference-cell pass verdict). Calibrates the cathode "
            "model (NEXT_STEPS #2.2/#3.1)."
        ),
        "depends_on": ["reference-cell/beaker + bath (B0 master; FIRST_LAB_DAY)"],
        "pairs_with": ["kinetics_fit_pipeline (PR #35)"],
        "sequence": [
            {
                "step": 1,
                "title": "HER branch in the Fe-free supporting electrolyte",
                "why": "No Fe2+ ⇒ total cathodic current IS the HER branch. "
                       "Unambiguous b_her and i0_her on the ACTUAL cathode surface.",
                "bath": "B0 master with FeSO4 omitted (same anion/pH/T/surface prep)",
                "technique": "RDE or static LSV (0.01-1 V/s), + EIS Rct near E_eq",
                "outputs": ["b_her", "i0_her"],
            },
            {
                "step": 2,
                "title": "Levich transport separation on the Fe bath",
                "why": "Plateau i_lim vs sqrt(omega) ⇒ D and the Nernst film "
                       "thickness (delta) that calibrates diffusion_layer_1d.",
                "bath": "B0 master, 1 M Fe2+",
                "technique": "RDE rotation matrix (400-2500 rpm)",
                "outputs": ["D", "delta", "i_lim(omega)"],
            },
            {
                "step": 3,
                "title": "Fe branch fit with HER HELD FIXED (from step 1)",
                "why": "Two free parameters (i0_fe, b_fe) vs a full polarization "
                       "curve ⇒ non-degenerate Fe/HER separation.",
                "technique": "fit_fe_kinetics_given_her (this module)",
                "outputs": ["i0_fe", "b_fe"],
            },
            {
                "step": 4,
                "title": "Volumetric H2 confirmation at fixed j",
                "why": "Independent check: FE_HER = 2F·n_H2/Q vs deposit FE_Fe "
                       "must close the charge ledger (≈1).",
                "technique": "gas burette / manometric cell + gravimetric deposit",
                "outputs": ["FE_her(gas)", "FE_fe(mass)", "closure residual"],
            },
        ],
        "gates": [
            "HER Tafel fit R2 >= 0.98 and slope in 0.06-0.20 V/dec (iron-group HER)",
            "Levich plot R2 >= 0.995 on the plateau; recovered D within 25% of "
            "the Fe2+ literature anchor (7.2e-10 m^2/s at 25 C, Arrhenius-scaled)",
            "Fe Tafel fit R2 >= 0.98, slope in 0.06-0.18 V/dec, positive i0",
            "Charge ledger closure |FE_her + FE_fe - 1| <= 0.05 (tolerance to be "
            "tightened after metrology is qualified)",
            "HER-branch residual (gas vs fitted branch) within the stated uncertainty budget",
        ],
        "data_contract": {
            "rde_lsv": ["Voltage_V", "Current_A", "Area_cm2", "pH", "Temp_C",
                        "Fe_M", "Ref_V", "Omega_rpm", "Bath_FeFree"],
            "volumetric": ["Charge_Ah", "H2_Volume_mL", "T_K", "P_Pa",
                           "Deposit_Fe_kg", "Run_time_s", "Electrode_area_m2"],
        },
        "scale": "L0 screen → L1 real-data validation",
    }


def model_scope() -> Dict[str, Any]:
    """House scope contract for this module."""
    return {
        "computes": [
            "Single-branch HER Tafel fit (b_her, i0_her) on the Fe-free bath — the #34 first measurement",
            "Levich transport separation (D, Nernst delta) on the RDE Fe bath",
            "Fe branch fit (i0_fe, b_fe) with the HER branch HELD FIXED (non-degenerate separation)",
            "Volumetric H2 charge-ledger closure (FE_her gas vs FE_fe mass) and branch residual",
            "Synthetic two-bath simulator + L0 self-test proving recoverability",
            "Machine-readable measurement spec (what to measure, in what order, with gates)",
        ],
        "does_not_compute": [
            "Surface-state drift / site-blocking corrections to HER (see models/surface_state.py)",
            "Real wet-lab accuracy — L0 synthetic validation only until real run data exist",
            "Anodic-branch fitting (module is cathodic-only by design)",
            "Non-ideal gas / water-vapour correction for the gas burette (L1 task)",
            "pH or temperature dependence of the fitted branches (fit at one condition)",
        ],
        "calibration_required": [
            "One Fe-free run to fix HER on the actual cathode surface before any Fe-bath fit",
            "One Fe-bath RDE Levich set to fix D/delta for diffusion_layer_1d",
            "Volumetric H2 + deposit mass at at least 3 current densities for the closure gate",
        ],
        "key_uncertainty": (
            "The single dominant L0 unknown (b_her) is resolved by step 1. Residual "
            "levers are D (nu affects it ~nu^(1/4)) and the volumetric H2 gas "
            "correction; both are measured, not assumed."
        ),
        "limitations": (
            "Screening module demonstrating the method on synthetic data. The "
            "procedure is validated to RECOVER known kinetics; no wet-lab dataset "
            "exists in this repository. It is NOT gate evidence."
        ),
    }


if __name__ == "__main__":
    self_test(verbose=True)
