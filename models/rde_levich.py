"""Rotating disk electrode (RDE) kinetics/transport separation — Levich and
Koutecky-Levich analysis.

Closes the gap named in ``docs/RESEARCH_PROGRAM.md`` (Tier 1, the only
remaining unbuilt desk model):

    "RDE + Levich for kinetics/transport separation -- not started; the
    measurement that makes ``diffusion_layer_1d`` calibratable. Highest-value
    remaining Tier 1 model+experiment pair."

Why this matters
----------------
The rest of the codebase treats the cathode boundary layer as a *stagnant film*
of thickness ``delta`` (``transport.py``, ``diffusion_layer_1d.py``) with a
Levich-style limiting current ``i_lim = z F D C / delta``.  That delta is an
engineering guess.  An RDE is the instrument that replaces the guess with a
number: the rotating disk gives a hydrodynamically well-defined boundary layer
whose thickness is known analytically (Cochran flow) and whose transport-limited
current obeys the **Levich equation**

    i_lim = 0.62 z F D^(2/3) omega^(1/2) nu^(-1/6) C_bulk

so a plot of ``i_lim`` vs ``omega^(1/2)`` yields the **diffusivity D** directly,
and the corresponding **Nernst film thickness delta** at a chosen rotation rate
(``delta = 1.61 D^(1/3) nu^(1/6) omega^(-1/2)``) is the number to feed
``diffusion_layer_1d``.  The RDE therefore converts a free model parameter
(``delta``) into a *measured* one.

Then, because the RDE boundary layer is under our control, we can separate the
two things that a static beaker conflates:

* **Kinetics** — the Fe^{2+}/Fe and HER Tafel slopes and exchange-current
  densities, via a **Koutecky-Levich** correction
  (``1/i = 1/i_k + 1/(B sqrt(omega))``) at each potential followed by a Tafel
  fit on the transport-free kinetic current.
* **Transport** — the Fe^{2+} diffusion-limited current ``i_lim(omega)``.

This is the "single measurement that gives Tafel slopes for Fe deposition AND
HER on the same surface" that the roadmap names.

Units
-----
SI unless stated: currents in A/m^2 (positive cathodic magnitudes), potentials
in V vs. SHE, diffusivities in m^2/s, kinematic viscosity in m^2/s, rotation in
rpm (converted internally to rad/s).  ``omega`` always means angular frequency
(rad/s) in formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .electrochemistry import E0_FE, FARADAY, R_GAS, T_REF, Z_FE
from .pourbaix import her_line

# ─── Reference transport properties (iron sulfate baths) ─────────────
# Fe2+ diffusivity at infinite dilution, 25 C (m^2/s); same value used across
# transport.py / diffusion_layer_1d.py so the RDE measurement cross-checks them.
D_FE2_25 = 7.2e-10
# Activation energy for the Arrhenius diffusivity correction (J/mol), matching
# diffusion_layer_1d.DIFF_EA_J_MOL so the RDE result can be extrapolated.
DIFF_EA_J_MOL = 18.0e3

# Water properties at 25 C (kg/m^3 and m^2/s) — the bath's minor components move
# the kinematic viscosity by a few percent; treat these as screening values.
RHO_WATER_25 = 997.0          # kg/m^3
KIN_VISC_WATER_25 = 8.9e-7    # m^2/s

# Levich / Cochran constant (dimensionless) for the RDE boundary layer.
LEVICH_0_62 = 0.62
COCHRAN_DELTA_PREFACTOR = 1.61


# ═════════════════════════════════════════════════════════════════════
#  Thermodynamic bath properties (temperature correlations)
# ═════════════════════════════════════════════════════════════════════
def water_density_kg_m3(T_C: float) -> float:
    """IAPWS-adjacent polynomial for liquid water density, valid 0-100 C."""
    T = float(T_C)
    return (
        999.842594
        + 6.793952e-2 * T
        - 9.095290e-3 * T ** 2
        + 1.001685e-4 * T ** 3
        - 1.120083e-6 * T ** 4
        + 6.536332e-9 * T ** 5
    )


def dynamic_viscosity_water_Pa_s(T_C: float) -> float:
    """Dynamic viscosity of liquid water (Pa s), Vogel form, 0-100 C."""
    T_K = float(T_C) + 273.15
    return 2.414e-5 * 10.0 ** (247.8 / (T_K - 140.0))


def kinematic_viscosity_water_m2_s(T_C: float) -> float:
    """Kinematic viscosity of liquid water (m^2/s)."""
    return dynamic_viscosity_water_Pa_s(T_C) / water_density_kg_m3(T_C)


def diffusivity_Arrhenius(
    D_25C_m2_s: float, T_C: float, Ea_J_mol: float = DIFF_EA_J_MOL
) -> float:
    """Arrhenius temperature correction for a diffusivity.

    D(T) = D(25 C) * exp[ Ea/R * (1/298.15 - 1/T) ].  Identical form to
    ``diffusion_layer_1d._diffusivity_T`` so an RDE value at one temperature can
    be carried to the FE engine's operating temperature.
    """
    T = float(T_C) + 273.15
    return float(D_25C_m2_s * np.exp(Ea_J_mol / R_GAS * (1.0 / T_REF - 1.0 / T)))


# ═════════════════════════════════════════════════════════════════════
#  Levich transport-limited current on the rotating disk
# ═════════════════════════════════════════════════════════════════════
def rpm_to_rad_per_s(rpm) -> np.ndarray:
    return np.asarray(rpm, dtype=float) * 2.0 * np.pi / 60.0


def levich_constant_B(
    z: int = Z_FE,
    D_m2_s: float = D_FE2_25,
    C_bulk_M: float = 1.0,
    nu_m2_s: float = KIN_VISC_WATER_25,
) -> float:
    """Levich proportionality constant B (A/m^2 per (rad/s)^1/2).

    i_lim = B * omega^(1/2),  with
    B = 0.62 z F D^(2/3) nu^(-1/6) C_bulk.
    """
    C_bulk_mol_m3 = C_bulk_M * 1e3
    return LEVICH_0_62 * z * FARADAY * D_m2_s ** (2.0 / 3.0) * nu_m2_s ** (-1.0 / 6.0) * C_bulk_mol_m3


def levich_limiting_current(
    omega_rpm,
    z: int = Z_FE,
    D_m2_s: float = D_FE2_25,
    C_bulk_M: float = 1.0,
    nu_m2_s: float = KIN_VISC_WATER_25,
) -> np.ndarray:
    """Levich limiting current (A/m^2) at rotation rate(s) omega_rpm."""
    omega = rpm_to_rad_per_s(omega_rpm)
    B = levich_constant_B(z, D_m2_s, C_bulk_M, nu_m2_s)
    return B * np.sqrt(omega)


def diffusivity_from_levich_B(
    B: float,
    z: int = Z_FE,
    C_bulk_M: float = 1.0,
    nu_m2_s: float = KIN_VISC_WATER_25,
) -> float:
    """Invert the Levich constant for the diffusivity D (m^2/s)."""
    base = B / (LEVICH_0_62 * z * FARADAY * nu_m2_s ** (-1.0 / 6.0) * (C_bulk_M * 1e3))
    return float(base ** 1.5)


def nernst_layer_thickness_m(
    omega_rpm, D_m2_s: float = D_FE2_25, nu_m2_s: float = KIN_VISC_WATER_25
) -> np.ndarray:
    """Effective Nernst diffusion-film thickness (m) on the RDE.

    delta = 1.61 D^(1/3) nu^(1/6) omega^(-1/2).  This is the number to hand to
    ``diffusion_layer_1d`` / ``transport.py`` as their ``boundary_layer``.
    """
    omega = rpm_to_rad_per_s(omega_rpm)
    return COCHRAN_DELTA_PREFACTOR * D_m2_s ** (1.0 / 3.0) * nu_m2_s ** (1.0 / 6.0) / np.sqrt(omega)


# ═════════════════════════════════════════════════════════════════════
#  Koutecky-Levich mixed control
# ═════════════════════════════════════════════════════════════════════
def koutecky_levich_kinetic(i_total: np.ndarray, i_lim: np.ndarray) -> np.ndarray:
    """Transport-free (kinetic) current from a measured total at a known i_lim.

    1/i_k = 1/i_total - 1/i_lim.  Returns NaN where i_total >= i_lim (no valid
    kinetic current because the reaction is transport-limited there).
    """
    i_total = np.asarray(i_total, dtype=float)
    i_lim = np.asarray(i_lim, dtype=float)
    i_lim = np.maximum(i_lim, 1e-30)
    i_total = np.maximum(i_total, 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 / (1.0 / i_total - 1.0 / i_lim)
    out = np.where(i_total >= i_lim * 0.999, np.nan, out)
    return out


def _fit_i_vs_sqrt_omega(i_at_plateau: np.ndarray, omega_rpm: np.ndarray) -> Dict[str, Any]:
    """Fit i = B * sqrt(omega) (through the origin) to plateau currents.

    Returns B, the per-omega residuals, and R^2 against the no-intercept model.
    """
    omega = rpm_to_rad_per_s(omega_rpm)
    x = np.sqrt(omega)
    y = np.asarray(i_at_plateau, dtype=float)
    B = float(np.sum(x * y) / np.sum(x * x))
    yhat = B * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"B_A_m2_per_sqrt_rps": B, "r_squared": r2, "n_points": int(len(y))}


def _fit_tafel(E: np.ndarray, i: np.ndarray, E_eq: float) -> Dict[str, Any]:
    """Fit log10(cathodic kinetic current) vs cathodic overpotential.

    eta = E_eq - E (>0 cathodic).  Returns the Tafel slope (V/decade, positive),
    the exchange current i0 (A/m^2), and fit diagnostics.
    """
    E = np.asarray(E, dtype=float)
    i = np.asarray(i, dtype=float)
    eta = E_eq - E
    valid = np.isfinite(eta) & np.isfinite(i) & (i > 0) & (eta > 0)
    if valid.sum() < 3:
        return {"tafel_slope_V_decade": math.nan, "i0_A_m2": math.nan,
                "r_squared": math.nan, "n_points": int(valid.sum())}
    x = eta[valid]
    y = np.log10(i[valid])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "tafel_slope_V_decade": float(1.0 / slope),
        "i0_A_m2": float(10.0 ** intercept),
        "r_squared": float(r2),
        "n_points": int(valid.sum()),
        "overpotential_min_V": float(x.min()),
        "overpotential_max_V": float(x.max()),
    }


# ═════════════════════════════════════════════════════════════════════
#  Synthetic RDE polarization simulator
# ═════════════════════════════════════════════════════════════════════
@dataclass
class RDEBranch:
    """One cathodic branch (Fe^{2+} -> Fe or HER) on the RDE.

    ``i_lim_A_m2`` = an imposed transport limit (None for a reaction whose
    transport limit is negligible, e.g. HER in a well-buffered bath).
    """

    i0_A_m2: float
    tafel_V_decade: float
    E_eq_V: float
    i_lim_A_m2: Optional[float] = None

    def current(self, E: np.ndarray, i_lim: Optional[float]) -> np.ndarray:
        """Cathodic current magnitude (A/m^2); Koutecky-Levich if limited."""
        E = np.asarray(E, dtype=float)
        eta = self.E_eq_V - E
        i_k = self.i0_A_m2 * 10.0 ** (np.maximum(eta, 0.0) / self.tafel_V_decade)
        if i_lim is None:
            return i_k
        return i_k * i_lim / (i_k + i_lim)


def simulate_rde_polarization(
    E_grid_V: np.ndarray,
    omega_rpm: np.ndarray,
    fe: Optional[RDEBranch] = None,
    her: Optional[RDEBranch] = None,
    z: int = Z_FE,
    D_m2_s: float = D_FE2_25,
    C_fe_M: float = 1.0,
    nu_m2_s: float = KIN_VISC_WATER_25,
) -> "dict":
    """Generate a synthetic RDE polarization dataset (Fe + HER).

    Returns a dict with keys ``potentials_V``, ``omega_rpm``, ``i_lim_A_m2``,
    ``i_fe_A_m2``, ``i_her_A_m2``, ``i_total_A_m2`` and a convenience
    ``frame`` pandas.DataFrame of the long-form rows.
    """
    import pandas as pd

    # i0 = 500 A/m^2 (~50 mA/cm^2) is a realistic Fe-deposition exchange current;
    # it is large enough that Fe reaches a clean transport-limited plateau well
    # before HER onset, which is exactly the experimental condition the method
    # needs.
    fe = fe or RDEBranch(i0_A_m2=500.0, tafel_V_decade=0.120, E_eq_V=E0_FE)
    her = her or RDEBranch(i0_A_m2=0.02, tafel_V_decade=0.150, E_eq_V=-0.1184)

    E = np.asarray(E_grid_V, dtype=float)
    omega = np.asarray(omega_rpm, dtype=float)
    i_lim = levich_limiting_current(omega, z=z, D_m2_s=D_m2_s, C_bulk_M=C_fe_M, nu_m2_s=nu_m2_s)

    rows = []
    for o, iL in zip(omega, i_lim):
        i_fe = fe.current(E, iL)
        i_her = her.current(E, her.i_lim_A_m2)
        for idx, e in enumerate(E):
            rows.append(
                {
                    "potential_V": float(e),
                    "omega_rpm": float(o),
                    "i_lim_A_m2": float(iL),
                    "i_fe_A_m2": float(i_fe[idx]),
                    "i_her_A_m2": float(i_her[idx]),
                    "i_total_A_m2": float(i_fe[idx] + i_her[idx]),
                }
            )
    return {
        "potentials_V": E,
        "omega_rpm": omega,
        "i_lim_A_m2": i_lim,
        "frame": pd.DataFrame(rows),
    }


# ═════════════════════════════════════════════════════════════════════
#  Automatic window recommendation
# ═════════════════════════════════════════════════════════════════════
def recommend_windows_from_polarization(
    df: Any, *, potential_col: str = "potential_V", omega_col: str = "omega_rpm",
    i_col: str = "i_total_A_m2",
) -> Dict[str, Any]:
    """Locate the Fe transport-limited plateau and the analysis windows.

    Heuristics (all data-driven, no prior on D):
      * The **plateau** is the potential where the through-origin Levich fit
        (``i = B sqrt(omega)`` across the rotation matrix) is most linear
        (R^2 maximised).  In the kinetic regime current is ~rotation-independent
        (poor through-origin fit), on the transport plateau it scales exactly as
        ``sqrt(omega)`` (R^2 -> 1), and once HER dominates it becomes
        rotation-independent again (poor fit) — so R^2 has a clean interior
        maximum at the Fe transport-limited plateau.
      * The **kinetic window** is potentials above the plateau (Fe not yet
        transport-limited at the fastest rotation).
      * The **HER window** is potentials below the plateau where total current
        exceeds the fastest-rotation limiting current (Fe saturated, HER adds).

    Returns ``{plateau_E_V, kinetic_window_V, her_window_V}``.
    """
    pots = np.array(sorted(set(df[potential_col].tolist())))
    omegas = np.array(sorted(set(df[omega_col].tolist())))
    x = np.sqrt(rpm_to_rad_per_s(omegas))

    best_r2, best_idx = -1.0, None
    for i, p in enumerate(pots):
        i_at = np.array(
            [df[(df[potential_col] == p) & (df[omega_col] == o)][i_col].mean() for o in omegas]
        )
        if np.any(i_at <= 0) or not np.all(np.isfinite(i_at)):
            continue
        B = float(np.sum(x * i_at) / np.sum(x * x))
        yhat = B * x
        ss_res = float(np.sum((i_at - yhat) ** 2))
        ss_tot = float(np.sum((i_at - i_at.mean()) ** 2))
        r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2, best_idx = r2, i

    if best_idx is None:
        return {"plateau_E_V": None, "kinetic_window_V": None, "her_window_V": None}
    plateau_E = float(pots[best_idx])

    def _current_at(p, col_omega):
        return float(df[(df[potential_col] == p) & (df[omega_col] == col_omega)][i_col].mean())

    omega_fast = float(df[omega_col].max())
    # Reference the fastest-rotation plateau current, which is ~ i_lim(fastest
    # omega): the strictest test for "Fe fully saturated".
    i_lim_max = _current_at(plateau_E, omega_fast)
    kinetic_pots = [p for p in pots if p > plateau_E and _current_at(p, omega_fast) < 0.8 * i_lim_max]
    her_pots = [p for p in pots if p < plateau_E and _current_at(p, omega_fast) > 1.02 * i_lim_max]

    return {
        "plateau_E_V": plateau_E,
        "kinetic_window_V": (float(min(kinetic_pots)), float(max(kinetic_pots))) if kinetic_pots else None,
        "her_window_V": (float(min(her_pots)), float(max(her_pots))) if her_pots else None,
    }


# ═════════════════════════════════════════════════════════════════════
#  Full RDE analysis
# ═════════════════════════════════════════════════════════════════════
def analyze_rde_polarization(  # noqa: C901 - many fit branches
    df: Any,
    *,
    z: int = Z_FE,
    C_fe_M: float = 1.0,
    nu_m2_s: float = KIN_VISC_WATER_25,
    T_C: float = 25.0,
    pH: float = 2.0,
    E_eq_fe_V: Optional[float] = None,
    E_eq_her_V: Optional[float] = None,
    D_ref_m2_s: Optional[float] = None,
    potential_col: str = "potential_V",
    omega_col: str = "omega_rpm",
    i_col: str = "i_total_A_m2",
) -> Dict[str, Any]:
    """Full RDE kinetics/transport separation on a measured polarization set.

    Expects a table with ``potential_V``, ``omega_rpm`` and a total cathodic
    current column (default ``i_total_A_m2``).  Performs:

      1. **Levich analysis** at the transport-limited plateau -> D, B, delta.
      2. **Fe kinetic Tafel** via Koutecky-Levich correction in the kinetic
         window -> b_fe, i0_fe.
      3. **HER Tafel** by subtracting i_lim in the deep-cathodic window ->
         b_her, i0_her.

    If ``D_ref_m2_s`` is given, a ``recovery_checks`` block reports how close
    the measurement-recovered D is to the reference (e.g. the value used to
    generate synthetic data, or a literature value).
    """
    E_eq_fe = E0_FE if E_eq_fe_V is None else E_eq_fe_V
    E_eq_her = her_line(pH, T_C + 273.15) if E_eq_her_V is None else E_eq_her_V

    # ── windows / plateau ───────────────────────────────────────────
    rec = recommend_windows_from_polarization(
        df, potential_col=potential_col, omega_col=omega_col, i_col=i_col
    )
    plateau_E = rec["plateau_E_V"]

    # ── Levich D extraction at the plateau ───────────────────────────
    omegas = np.array(sorted(set(df[omega_col].tolist())))
    plateau_i = np.array(
        [df[(df[potential_col] == plateau_E) & (df[omega_col] == o)][i_col].mean() for o in omegas]
    )
    lev = _fit_i_vs_sqrt_omega(plateau_i, omegas)
    B = lev["B_A_m2_per_sqrt_rps"]
    D_meas = diffusivity_from_levich_B(B, z=z, C_bulk_M=C_fe_M, nu_m2_s=nu_m2_s)
    i_lim_by_omega = B * np.sqrt(rpm_to_rad_per_s(omegas))
    omega_ref = 1600.0
    delta_ref = float(nernst_layer_thickness_m(omega_ref, D_m2_s=D_meas, nu_m2_s=nu_m2_s))

    # ── Fe kinetic Tafel (Koutecky-Levich corrected) ─────────────────
    kin_win = rec["kinetic_window_V"]
    fe_tafel: Dict[str, Any] = {"tafel_slope_V_decade": math.nan, "i0_A_m2": math.nan,
                                "r_squared": math.nan, "n_points": 0}
    if kin_win is not None:
        kpots = np.array(
            [p for p in sorted(set(df[potential_col].tolist())) if kin_win[0] <= p <= kin_win[1]]
        )
        ik_means, ik_pts = [], []
        for p in kpots:
            vals = []
            for o, iL in zip(omegas, i_lim_by_omega):
                itot = float(df[(df[potential_col] == p) & (df[omega_col] == o)][i_col].mean())
                if itot < 0.9 * iL:
                    vals.append(koutecky_levich_kinetic(np.array([itot]), np.array([iL]))[0])
            if vals:
                ik_means.append(float(np.nanmean(vals)))
                ik_pts.append(p)
        if len(ik_means) >= 3:
            fe_tafel = _fit_tafel(np.array(ik_pts), np.array(ik_means), E_eq_fe)

    # ── HER Tafel (i_total - i_lim in the deep-cathodic window) ──────
    her_win = rec["her_window_V"]
    her_tafel: Dict[str, Any] = {"tafel_slope_V_decade": math.nan, "i0_A_m2": math.nan,
                                 "r_squared": math.nan, "n_points": 0}
    if her_win is not None:
        hpots = np.array(
            [p for p in sorted(set(df[potential_col].tolist())) if her_win[0] <= p <= her_win[1]]
        )
        ih_means, ih_pts = [], []
        for p in hpots:
            vals = []
            for o, iL in zip(omegas, i_lim_by_omega):
                itot = float(df[(df[potential_col] == p) & (df[omega_col] == o)][i_col].mean())
                v = itot - iL
                if v > 1.0e-3:
                    vals.append(v)
            if vals:
                ih_means.append(float(np.nanmean(vals)))
                ih_pts.append(p)
        if len(ih_means) >= 3:
            her_tafel = _fit_tafel(np.array(ih_pts), np.array(ih_means), E_eq_her)

    checks = None
    if D_ref_m2_s is not None:
        checks = {
            "D_ref_m2_s": D_ref_m2_s,
            "D_recovered_m2_s": D_meas,
            "relative_error_pct": 100.0 * (D_meas - D_ref_m2_s) / D_ref_m2_s,
            "delta_ref_m_at_1600rpm": delta_ref,
        }

    return {
        "plateau_E_V": plateau_E,
        "windows": rec,
        "levich": {**lev, "D_m2_s": D_meas, "i_lim_by_omega_A_m2": i_lim_by_omega.tolist()},
        "nernst_layer_m_at_1600rpm": delta_ref,
        "fe_tafel": fe_tafel,
        "her_tafel": her_tafel,
        "E_eq_fe_V": E_eq_fe,
        "E_eq_her_V": E_eq_her,
        "recovery_checks": checks,
    }


# ═════════════════════════════════════════════════════════════════════
#  Experimental design + scope
# ═════════════════════════════════════════════════════════════════════
def rde_experiment_design(
    *,
    omega_lo_rpm: float = 400.0,
    omega_hi_rpm: float = 2500.0,
    n_omega: int = 6,
    C_fe_M: float = 1.0,
    z: int = Z_FE,
    D_m2_s: float = D_FE2_25,
    nu_m2_s: float = KIN_VISC_WATER_25,
    T_C: float = 60.0,
) -> Dict[str, Any]:
    """Recommended RDE measurement matrix and gate rules.

    Returns a geometrically-spaced rotation-rate matrix, the expected Levich
    limiting-current spread (so the current meter and potentiostat resolution can
    be chosen to resolve it), the expected Nernst film thickness at each rate,
    and pass/fail decision rules for accepting the measurement.
    """
    omegas = np.geomspace(omega_lo_rpm, omega_hi_rpm, n_omega)
    D_T = diffusivity_Arrhenius(D_m2_s, T_C)
    nu_T = kinematic_viscosity_water_m2_s(T_C)
    i_lim = levich_limiting_current(omegas, z=z, D_m2_s=D_T, C_bulk_M=C_fe_M, nu_m2_s=nu_T)
    delta = nernst_layer_thickness_m(omegas, D_m2_s=D_T, nu_m2_s=nu_T)

    # A Levich plot whose highest and lowest points differ by <~3x needs a
    # current meter good to ~0.3% to be trusted; flag it.
    spread_ratio = float(i_lim.max() / i_lim.min())
    current_precision_req = 100.0 / spread_ratio  # ~ percentage

    gates = [
        "Levich plot R^2 >= 0.995 over the plateau (else re-run with more rotation rates / wider omega range)",
        "Recovered D within 25% of literature D_Fe2 (7.2e-10 m^2/s at 25 C, Arrhenius-scaled) or of the FE engine's calibration anchor",
        "Fe Tafel fit R^2 >= 0.98 and a slope in 0.06-0.18 V/decade (single-step 2e- transfer, alpha~0.5)",
        "HER Tafel fit R^2 >= 0.98 in a window where total current exceeds the fastest-rotation i_lim by >10% (Fe fully saturated)",
        "Koutecky-Levich intercept positive (1/i_k > 0); a negative intercept means the plateau potential was mis-placed",
    ]

    return {
        "rotation_matrix_rpm": omegas.tolist(),
        "expected_i_lim_A_m2": i_lim.tolist(),
        "expected_nernst_layer_m": delta.tolist(),
        "i_lim_spread_ratio": spread_ratio,
        "current_precision_requirement_pct": current_precision_req,
        "diffusivity_at_T_C": D_T,
        "kinematic_viscosity_at_T_C": nu_T,
        "gates": gates,
    }


def model_scope() -> Dict[str, Any]:
    """What this module does and does not compute (house scope contract)."""
    return {
        "computes": [
            "Levich limiting current i_lim = 0.62 z F D^(2/3) omega^(1/2) nu^(-1/6) C_bulk",
            "Diffusivity D from a Levich plot (i_lim vs omega^(1/2))",
            "Nernst film thickness delta on the RDE (the number that calibrates diffusion_layer_1d / transport.py)",
            "Koutecky-Levich kinetic-current correction (1/i_k = 1/i - 1/i_lim)",
            "Fe and HER Tafel slopes + exchange-current densities from a single RDE polarization set",
            "Automatic plateau / kinetic-window / HER-window recommendation from the data",
            "Synthetic RDE polarization simulator for method validation and experiment planning",
            "RDE measurement matrix with resolution requirements and gate rules",
        ],
        "does_not_compute": [
            "Rotation-rate-independent convection inside the film (assumed steady, well-established Cochran flow)",
            "Activity coefficients, complexation, or ion-pairing corrections to C_bulk",
            "Non-ideal Levich behaviour at very high omega (transition to turbulence)",
            "Migration enhancement of i_lim (supporting-electrolyte-free limit; see transport.py)",
            "Boron/borate or sulfate transport-limited corrections to the HER branch",
            "Anodic branch fitting (module is cathodic-only by design)",
        ],
        "calibration_required": [
            "One RDE polarization set in the actual bath to fix D (and hence delta) for diffusion_layer_1d",
            "Tafel slopes to be confirmed by Levich-line independence of the kinetic current",
            "Kinematic viscosity of the real electrolyte (the water-only value is a screening input)",
        ],
        "key_uncertainty": (
            "nu (kinematic viscosity) is taken as water-only here; real sulfate/chloride "
            "baths raise nu a few percent, which shifts D by ~nu^(1/4). Measure nu (Ostwald "
            "viscometer) before trusting absolute D; the method's internal cross-checks "
            "(Tafel linearity, Levich R^2) are what gate acceptance."
        ),
        "limitations": (
            "Screening model demonstrating the method on synthetic data. No wet-lab RDE "
            "current exists in this repository; the module converts a free model parameter "
            "(delta) into a measured one the first time it is run against real data."
        ),
    }
