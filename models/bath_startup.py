"""
Bath startup kinetics — Fe²⁺ air oxidation and ascorbic acid stabilization.

Models the homogeneous oxidation of Fe²⁺ to Fe³⁺ by dissolved oxygen in
acidic iron plating baths, and the stabilizing (sacrificial) effect of
ascorbic acid (C₆H₈O₆ → dehydroascorbic acid C₆H₆O₆).

Chemistry
---------
1. Fe²⁺ autoxidation:  4 Fe²⁺ + O₂ + 4 H⁺ → 4 Fe³⁺ + 2 H₂O
   Rate = k_ox · [Fe²⁺] · [O₂] · ([OH⁻] / [OH⁻]_ref)²
   k_ox ≈ 1e-4 M⁻¹ s⁻¹ at 25 °C, pH 2  (effective, empirical)

2. Ascorbic acid reduction:  2 Fe³⁺ + AA → 2 Fe²⁺ + DHA + 2 H⁺
   Rate = k_aa · [Fe³⁺] · [AA]  (fast, near-diffusion-limited)
   k_aa ≈ 1e3 M⁻¹ s⁻¹  (Hynes & Ratkovic 1987; Buettner 1993)

3. Ascorbic acid autoxidation:  AH⁻ + ½ O₂ → DHA + H₂O
   Rate = k_aa_ox · [AH⁻] · [O₂]   (slow at low pH)
   k_aa_ox ≈ 3.0 M⁻¹ s⁻¹  (intrinsic, for deprotonated AH⁻)
   pKa₁ = 4.17  →  at pH 2, [AH⁻]/[AA_total] ≈ 0.7%

Dissolved O₂ is replenished from headspace at a rate proportional to the
surface-area-to-volume ratio (kLa analog).

References
----------
* Sung & Morgan (1980), Environ. Sci. Technol. 14:590-594 — Fe²⁺ oxidation kinetics
* Stumm & Lee (1961), Ind. Eng. Chem. 53:143-146 — OH⁻ dependence
* Hynes & Ratkovic (1987) — ascorbate reduction of Fe³⁺
* Buettner (1993), Free Radic. Biol. Med. 14:425-428
* Khan & Martell (1967), J. Am. Chem. Soc. 89:4176 — AA autoxidation kinetics
* Weiss (1970), Deep-Sea Res. 17:721-735 — dissolved O₂ saturation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
from scipy.integrate import solve_ivp

from .electrochemistry import R_GAS

# ── Constants ────────────────────────────────────────────────────────────────

M_AA = 176.12          # g/mol  (ascorbic acid C₆H₈O₆)
ASCORBIC_PKA1 = 4.17   # pKa₁ of ascorbic acid (first dissociation)


# ── Temperature-dependent dissolved oxygen saturation ───────────────────────

def dissolved_o2_saturation_mol_L(T_C: float) -> float:
    """Dissolved O₂ saturation at 1 atm air in pure water (mol/L).

    Uses the Weiss (1970) equation for freshwater (S=0).  Valid 0–60 °C.
    Returns concentration in mol/L (converted from mL_STP/L via /22414).
    """
    T_K = T_C + 273.15
    # Weiss (1970) coefficients for O₂ in freshwater
    A1 = -173.4292
    A2 = 249.6339
    A3 = 143.3483
    A4 = -21.8492
    ln_C = A1 + A2 * (100.0 / T_K) + A3 * math.log(T_K / 100.0) + A4 * (T_K / 100.0)
    C_mL_per_L = math.exp(ln_C)  # mL(STP)/L at 1 atm total pressure (air mixture)
    # Convert to mol/L: 1 mol gas at STP = 22414 mL
    return C_mL_per_L / 22414.0


# ── Parameter dataclasses ────────────────────────────────────────────────────

@dataclass
class BathParams:
    """Parameters for the Fe²⁺ oxidation / ascorbic acid stabilization model.

    Attributes
    ----------
    fe2_0 : float
        Initial [Fe²⁺] (M). Typical 0.5–2 M.
    fe3_0 : float
        Initial [Fe³⁺] (M). Should be ~0 for fresh bath.
    aa_0 : float
        Initial ascorbic acid concentration (mol/L).
        1 g/L ≈ 5.68 mM.
    pH : float
        Bath pH. Typical 1.5–3.5.
    T_C : float
        Bath temperature (°C). Typical 25–60.
    k_ox_ref : float
        Fe²⁺ oxidation rate constant (M⁻¹ s⁻¹) at reference conditions.
    Ea_ox : float
        Activation energy for Fe²⁺ oxidation (J/mol).
    k_aa_reduce : float
        Ascorbic acid reduction rate constant (M⁻¹ s⁻¹).
    k_aa_oxidize : float
        Ascorbic acid autoxidation rate constant (M⁻¹ s⁻¹).
        Intrinsic rate for [AH⁻] + O₂; pH-dependent via pKa.
    sa_v_ratio : float
        Surface-area-to-volume ratio (cm⁻¹).
        Open beaker ~2 cm⁻¹, covered ~0.1 cm⁻¹.
    kLa_factor : float
        Mass transfer factor scaling kLa from SA/V (s⁻¹ per cm⁻¹).
    fe3_threshold : float
        Fe³⁺/Fe²⁺ ratio at which bath is considered degraded (default 0.05).
    """
    fe2_0: float = 1.0              # M
    fe3_0: float = 0.0              # M
    aa_0: float = 0.0               # mol/L (convert from g/L via g_per_L_to_mol_L)
    pH: float = 2.0
    T_C: float = 25.0

    # Rate constants
    k_ox_ref: float = 1.0e-4        # M⁻¹ s⁻¹ at 25°C, pH 2
    Ea_ox: float = 50_000.0         # J/mol (Sung & Morgan 1980)
    k_aa_reduce: float = 1.0e3      # M⁻¹ s⁻¹ (fast)
    k_aa_oxidize: float = 3.0       # M⁻¹ s⁻¹ (intrinsic for AH⁻; Khan & Martell 1967)

    # Transport
    sa_v_ratio: float = 1.0         # cm⁻¹  (moderate open container)
    kLa_factor: float = 1.0e-4      # s⁻¹ per cm⁻¹  (calibrated)

    # Threshold
    fe3_threshold: float = 0.05     # 5%


def g_per_L_to_mol_L(g_per_L: float) -> float:
    """Convert ascorbic acid from g/L to mol/L."""
    return g_per_L / M_AA


# ── Core kinetics ────────────────────────────────────────────────────────────

def fe2_oxidation_rate(fe2: float, o2: float, pH: float, T_C: float,
                       k_ox_ref: float, Ea_ox: float) -> float:
    """Rate of Fe²⁺ oxidation:  4 Fe²⁺ + O₂ + 4 H⁺ → 4 Fe³⁺ + 2 H₂O.

    rate = k_eff · [Fe²⁺] · [O₂]

    where k_eff = k_ox_ref · (10^(pH - 2))² · exp(Ea/R · (1/298.15 - 1/T))
    incorporates the [OH⁻]²/[OH⁻]_ref² pH dependence and Arrhenius
    temperature scaling.
    """
    OH = 10.0 ** (-(14.0 - pH))
    OH_ref = 10.0 ** (-12.0)  # [OH⁻] at pH 2
    T_K = T_C + 273.15
    T_ref = 298.15
    k_T = k_ox_ref * math.exp(Ea_ox / R_GAS * (1.0 / T_ref - 1.0 / T_K))
    oh_ratio_sq = (OH / OH_ref) ** 2
    return k_T * fe2 * o2 * oh_ratio_sq


def aa_reduction_rate(fe3: float, aa: float, k_aa: float) -> float:
    """Rate of ascorbic acid reducing Fe³⁺ → Fe²⁺.

    2 Fe³⁺ + AA → 2 Fe²⁺ + DHA + 2 H⁺
    rate = k_aa · [Fe³⁺] · [AA]  (consumption of Fe³⁺)
    """
    return k_aa * fe3 * aa


def aa_autoxidation_rate(aa: float, o2: float, pH: float, k_aa_ox: float) -> float:
    """Rate of ascorbic acid direct oxidation by dissolved O₂.

    AH⁻ + ½ O₂ → DHA + H₂O

    The intrinsic rate constant applies to the deprotonated form AH⁻.
    The fraction deprotonated at a given pH is:

        f_AH- = 1 / (1 + 10^(pKa - pH))

    At pH 2, pKa 4.17 → f ≈ 0.67%, so autoxidation is very slow.
    """
    f_dissoc = 1.0 / (1.0 + 10.0 ** (ASCORBIC_PKA1 - pH))
    return k_aa_ox * f_dissoc * aa * o2


# ── ODE system ───────────────────────────────────────────────────────────────

def _ode_rhs(t: float, y: np.ndarray, params: BathParams) -> np.ndarray:
    """Right-hand side of the coupled ODE system.

    y = [Fe²⁺, Fe³⁺, AA, O₂]
    """
    fe2, fe3, aa, o2 = y

    # Ensure non-negative concentrations
    fe2 = max(fe2, 0.0)
    fe3 = max(fe3, 0.0)
    aa = max(aa, 0.0)
    o2 = max(o2, 0.0)

    # Fe²⁺ oxidation (consumes Fe²⁺, produces Fe³⁺)
    r_ox = fe2_oxidation_rate(fe2, o2, params.pH, params.T_C,
                              params.k_ox_ref, params.Ea_ox)

    # AA reduction of Fe³⁺ (consumes AA, converts Fe³⁺ back to Fe²⁺)
    r_aa_red = aa_reduction_rate(fe3, aa, params.k_aa_reduce)

    # AA direct oxidation (consumes AA) — pH-dependent via [AH⁻]
    r_aa_ox = aa_autoxidation_rate(aa, o2, params.pH, params.k_aa_oxidize)

    # Dissolved O₂ replenishment from headspace
    o2_sat = dissolved_o2_saturation_mol_L(params.T_C)
    # Convert SA/V from cm⁻¹ to m⁻¹ for kLa
    kLa = params.kLa_factor * params.sa_v_ratio * 100.0  # s⁻¹
    o2_supply = kLa * (o2_sat - o2)

    # Species rate-of-change
    # Fe²⁺: consumed by oxidation, regenerated by AA reduction
    dfe2_dt = -r_ox + r_aa_red
    # Fe³⁺: produced by oxidation, consumed by AA reduction
    dfe3_dt = r_ox - r_aa_red
    # AA: consumed by reduction (2 Fe³⁺ per AA) and autoxidation
    daa_dt = -(r_aa_red / 2.0) - r_aa_ox
    # O₂: consumed by Fe²⁺ ox (1 O₂ per 4 Fe²⁺), AA ox (½ O₂ per AA), replenished
    do2_dt = -(r_ox / 4.0) - (r_aa_ox / 2.0) + o2_supply

    return np.array([dfe2_dt, dfe3_dt, daa_dt, do2_dt])


# ── Simulation result ───────────────────────────────────────────────────────

@dataclass
class BathSimulationResult:
    """Result of an ODE integration of the bath startup model.

    Attributes
    ----------
    time_hr : np.ndarray
        Time points (hours).
    fe2 : np.ndarray
        [Fe²⁺] at each time point (M).
    fe3 : np.ndarray
        [Fe³⁺] at each time point (M).
    aa : np.ndarray
        [Ascorbic acid] at each time point (mol/L).
    o2 : np.ndarray
        [Dissolved O₂] at each time point (mol/L).
    fe3_ratio : np.ndarray
        [Fe³⁺]/[Fe²⁺] ratio at each time point.
    time_to_threshold_hr : Optional[float]
        Hours until [Fe³⁺]/[Fe²⁺] > threshold. None if never reached.
    params : BathParams
        The parameters used for this simulation.
    """
    time_hr: np.ndarray
    fe2: np.ndarray
    fe3: np.ndarray
    aa: np.ndarray
    o2: np.ndarray
    fe3_ratio: np.ndarray
    time_to_threshold_hr: Optional[float]
    params: BathParams


def simulate_bath(params: BathParams, t_end_hr: float = 48.0,
                  dt_output_hr: float = 0.1) -> BathSimulationResult:
    """Integrate the bath kinetics ODE system.

    Parameters
    ----------
    params : BathParams
        Bath conditions and rate parameters.
    t_end_hr : float
        Simulation end time (hours).
    dt_output_hr : float
        Output time step (hours).

    Returns
    -------
    BathSimulationResult
        Time series of all species and derived quantities.
    """
    t_end_s = t_end_hr * 3600.0
    dt_s = dt_output_hr * 3600.0
    t_eval = np.arange(0, t_end_s + dt_s, dt_s)

    o2_sat = dissolved_o2_saturation_mol_L(params.T_C)

    y0 = np.array([params.fe2_0, params.fe3_0, params.aa_0, o2_sat])

    sol = solve_ivp(
        _ode_rhs,
        t_span=(0, t_end_s),
        y0=y0,
        args=(params,),
        method='LSODA',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-12,
        max_step=60.0,  # max 1 min steps
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    fe2 = sol.y[0]
    fe3 = sol.y[1]
    aa = sol.y[2]
    o2 = sol.y[3]
    time_hr = sol.t / 3600.0

    # Fe³⁺/Fe²⁺ ratio (avoid division by zero)
    fe3_ratio = np.where(fe2 > 1e-12, fe3 / fe2, np.inf)

    # Time to threshold
    threshold = params.fe3_threshold
    above = np.where(fe3_ratio > threshold)[0]
    if len(above) > 0:
        idx = above[0]
        if idx > 0:
            # Linear interpolation between bracketing points
            f = (threshold - fe3_ratio[idx - 1]) / (fe3_ratio[idx] - fe3_ratio[idx - 1])
            t_to = time_hr[idx - 1] + f * (time_hr[idx] - time_hr[idx - 1])
        else:
            t_to = time_hr[0]
        time_to_threshold = float(t_to)
    else:
        time_to_threshold = None

    return BathSimulationResult(
        time_hr=time_hr,
        fe2=fe2,
        fe3=fe3,
        aa=aa,
        o2=o2,
        fe3_ratio=fe3_ratio,
        time_to_threshold_hr=time_to_threshold,
        params=params,
    )


# ── Analysis helpers ─────────────────────────────────────────────────────────

def recommend_ascorbic_loading(pH: float = 2.0, T_C: float = 25.0,
                                fe2_0: float = 1.0,
                                sa_v_ratio: float = 1.0,
                                target_hr: float = 24.0,
                                max_ratio: float = 0.05) -> float:
    """Binary-search for the ascorbic acid loading (g/L) that keeps
    [Fe³⁺]/[Fe²⁺] below *max_ratio* for *target_hr* hours.

    Returns
    -------
    float
        Recommended ascorbic acid loading in g/L.
    """
    lo, hi = 0.0, 50.0  # g/L search range

    # Check that the upper bound works
    p_high = BathParams(fe2_0=fe2_0, aa_0=g_per_L_to_mol_L(hi),
                        pH=pH, T_C=T_C, sa_v_ratio=sa_v_ratio)
    res_high = simulate_bath(p_high, t_end_hr=target_hr + 0.1)
    if res_high.time_to_threshold_hr is not None:
        return hi  # even upper bound isn't enough — return max

    for _ in range(60):
        mid = (lo + hi) / 2.0
        p = BathParams(fe2_0=fe2_0, aa_0=g_per_L_to_mol_L(mid),
                       pH=pH, T_C=T_C, sa_v_ratio=sa_v_ratio)
        res = simulate_bath(p, t_end_hr=target_hr + 0.1)
        if res.time_to_threshold_hr is None or res.time_to_threshold_hr > target_hr:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.005:
            break

    return hi


def ascorbic_consumption_summary(params: BathParams, t_end_hr: float = 24.0) -> Dict[str, Optional[float]]:
    """Run a simulation and return a summary dict of consumption rates.

    Returns dict with keys:
    - aa_remaining_mol_L, aa_remaining_g_L
    - aa_consumed_g_L, consumption_rate_g_L_day
    - fe3_ratio_at_end
    - time_to_threshold_hr
    """
    res = simulate_bath(params, t_end_hr=t_end_hr)
    aa_consumed = params.aa_0 - res.aa[-1]  # mol/L consumed
    aa_consumed_g = aa_consumed * M_AA       # mol/L × g/mol = g/L

    return {
        "aa_remaining_mol_L": float(res.aa[-1]),
        "aa_remaining_g_L": float(res.aa[-1] * M_AA),
        "aa_consumed_g_L": float(max(aa_consumed_g, 0.0)),
        "consumption_rate_g_L_day": float(max(aa_consumed_g * 24.0 / t_end_hr, 0.0)),
        "fe3_ratio_at_end": float(res.fe3_ratio[-1]),
        "time_to_threshold_hr": res.time_to_threshold_hr,
    }


def sensitivity_ph(ph_values: List[float], T_C: float = 25.0,
                   fe2_0: float = 1.0, aa_g_L: float = 1.0,
                   sa_v_ratio: float = 1.0,
                   t_end_hr: float = 24.0) -> List[Dict[str, Optional[float]]]:
    """Sweep pH and compute stability metrics at each value."""
    results = []
    for pH in ph_values:
        p = BathParams(fe2_0=fe2_0, aa_0=g_per_L_to_mol_L(aa_g_L),
                       pH=pH, T_C=T_C, sa_v_ratio=sa_v_ratio)
        summary = ascorbic_consumption_summary(p, t_end_hr=t_end_hr)
        summary["pH"] = pH
        results.append(summary)
    return results


def sensitivity_temperature(T_values: List[float], pH: float = 2.0,
                            fe2_0: float = 1.0, aa_g_L: float = 1.0,
                            sa_v_ratio: float = 1.0,
                            t_end_hr: float = 24.0) -> List[Dict[str, Optional[float]]]:
    """Sweep temperature and compute stability metrics at each value."""
    results = []
    for T_C in T_values:
        p = BathParams(fe2_0=fe2_0, aa_0=g_per_L_to_mol_L(aa_g_L),
                       pH=pH, T_C=T_C, sa_v_ratio=sa_v_ratio)
        summary = ascorbic_consumption_summary(p, t_end_hr=t_end_hr)
        summary["T_C"] = T_C
        results.append(summary)
    return results
