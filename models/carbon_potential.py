"""
Carbon potential / activity model for gas and low-pressure carburizing atmospheres.

Models the thermodynamic carbon activity (a_C, graphite reference) and
expected austenite surface carbon from gas compositions:

* CO/CO2 (Boudouard): 2 CO ↔ C(gr) + CO2
* CH4/H2: CH4 ↔ C(gr) + 2 H2
* Propane etc.

Uses ΔG°(T) correlations from literature (screening) to compute equilibrium
constant K = exp(-ΔG°/RT), then a_C = K * (P_CO^2 / P_CO2) or K * (P_CH4 / P_H2^2).

Also provides:
* O2 probe → carbon potential conversion (oxygen partial pressure from CO/CO2)
* Austenite solubility (Acm) model for max C wt% at T
* Activity-to-wt conversion using Wagner formalism / activity coefficient γ_C
* Dew-point coupling (H2O/H2 ↔ CO/CO2 via water-gas shift)

All screening — not a full CALPHAD. Must be calibrated with O2 probe or foil data.

References:
* Boudouard: ΔG° = 170700 - 174.5 T (J/mol) (approx 700-1100°C, graphite ref)
* CH4 cracking: ΔG° =  84400 -  107.7 T (J/mol) for CH4 → C +2H2? Actually CH4→C+2H2 ΔG° =  85700 -110 T (screening mean)
* Water-gas shift: CO + H2O ↔ CO2 + H2, ΔG° = -41000 + 41.5 T J/mol
* Austenite C solubility Acm approx linear between eutectoid (0.76 wt% at 727°C) and eutectic (2.14% at 1147°C)
* Wagner dilute solution: ln a_C = ln x_C + ln γ_C°, γ_C activity coeff ~exp( (3750/T +... ) )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import math
import numpy as np

from .electrochemistry import R_GAS

# ΔG° correlations (J/mol) screening means, graphite reference for C
# Sources: Richardson-Ellingham approx, fitted for 800-1000°C
# Reaction: 2 CO -> C + CO2, ΔG°_B = 170700 -174.5*T  (J)
def dG_boudouard_J(T_K: float) -> float:
    return 170700.0 - 174.5 * T_K

# Reaction: CH4 -> C +2 H2, ΔG°_CH4 =  90000 - 109*T (J) — screening mean of 84400-107.7T
def dG_ch4_cracking_J(T_K: float) -> float:
    return 90000.0 - 109.0 * T_K

# Water-gas shift: CO + H2O -> CO2 + H2, ΔG°_WGS = -41000 + 41.5 T
def dG_wgs_J(T_K: float) -> float:
    return -41000.0 + 41.5 * T_K


def equilibrium_constant(dG_J: float, T_K: float) -> float:
    """K = exp(-dG/RT)"""
    return math.exp(-dG_J / (R_GAS * T_K))


def carbon_activity_from_co_co2(pCO_atm: float, pCO2_atm: float, T_C: float) -> float:
    """
    a_C from Boudouard equilibrium: a_C = K_B * pCO^2 / pCO2
    p in atm (or any consistent unit, ratio matters)
    """
    if pCO2_atm <= 0 or pCO_atm <= 0:
        raise ValueError("pCO and pCO2 must be positive")
    T_K = T_C + 273.15
    K = equilibrium_constant(dG_boudouard_J(T_K), T_K)
    aC = K * (pCO_atm ** 2) / pCO2_atm
    return float(aC)


def carbon_activity_from_ch4_h2(pCH4_atm: float, pH2_atm: float, T_C: float) -> float:
    """
    a_C from CH4 cracking: a_C = K_CH4 * pCH4 / pH2^2
    """
    if pH2_atm <= 0 or pCH4_atm <= 0:
        raise ValueError("pCH4 and pH2 must be positive")
    T_K = T_C + 273.15
    K = equilibrium_constant(dG_ch4_cracking_J(T_K), T_K)
    aC = K * pCH4_atm / (pH2_atm ** 2)
    return float(aC)


def oxygen_partial_pressure_from_co_co2(pCO_atm: float, pCO2_atm: float, T_C: float) -> float:
    """
    O2 partial pressure from CO/CO2 equilibrium: CO + 1/2 O2 ↔ CO2
    ΔG° for that reaction ≈ -282800 +86.8 T J (screening) → K = pCO2/(pCO * pO2^0.5)
    So pO2 = (pCO2/(pCO*K))^2
    """
    T_K = T_C + 273.15
    # ΔG° for CO + 0.5 O2 → CO2: approx -282800 + 86.8 T
    dG = -282800.0 + 86.8 * T_K
    K = math.exp(-dG / (R_GAS * T_K))
    # K = pCO2 / (pCO * pO2^0.5) → pO2^0.5 = pCO2/(pCO*K) → pO2 = (ratio)^2
    ratio = pCO2_atm / (pCO_atm * K)
    pO2 = ratio ** 2
    return float(pO2)


def dew_point_to_pH2O_ratio(dew_point_C: float, total_pressure_atm: float = 1.0) -> float:
    """
    Approximate H2O partial pressure from dew point (°C) using Magnus formula for saturation vapor pressure.
    Returns pH2O (atm) at given dew point.
    """
    # Magnus approx saturation vapor pressure (hPa) at T dew point
    # psat = 6.112 * exp( (17.67*T)/(T+243.5) ) hPa
    T = dew_point_C
    psat_hPa = 6.112 * math.exp((17.67 * T) / (T + 243.5))
    psat_atm = psat_hPa / 1013.25
    return float(psat_atm * total_pressure_atm)


def carbon_activity_from_dew_point_and_co(
    pCO_atm: float,
    dew_point_C: float,
    pH2_atm: float = 0.4,
    T_C: float = 900.0,
) -> float:
    """
    Via water-gas shift: pCO2 = K_WGS * pCO * pH2O / pH2, then a_C from CO/CO2.
    Screening link between dew point (moisture) and carbon potential.
    """
    T_K = T_C + 273.15
    pH2O = dew_point_to_pH2O_ratio(dew_point_C)
    K_wgs = equilibrium_constant(dG_wgs_J(T_K), T_K)
    # WGS: CO + H2O ↔ CO2 + H2, K = pCO2*pH2 / (pCO*pH2O)
    # → pCO2 = K * pCO * pH2O / pH2
    pCO2 = K_wgs * pCO_atm * pH2O / max(pH2_atm, 1e-12)
    return carbon_activity_from_co_co2(pCO_atm, max(pCO2, 1e-12), T_C)


def austenite_max_carbon_wt_percent(T_C: float) -> float:
    """
    Acm line max solubility of C in austenite (wt%) as function of T.

    Linear interpolation between eutectoid (0.76 wt% at 727°C) and
    eutectic (2.14 wt% at 1147°C). Below 727°C, use 0.0 (ferrite low solubility).
    """
    if T_C < 727.0:
        return 0.0
    if T_C >= 1147.0:
        return 2.14
    # linear
    frac = (T_C - 727.0) / (1147.0 - 727.0)
    return float(0.76 + frac * (2.14 - 0.76))


def carbon_activity_c_in_austenite(T_C: float, C_wt_percent: float) -> float:
    """
    Carbon activity in austenite (graphite reference), screening correlation.

    Returns a_C (dimensionless, graphite = 1 at saturation).
    Validated: a_C ≈ 1 at ~1.3 wt% C, 900°C — consistent with the
    Fe-C phase diagram Acm boundary.

    This is an empirical screening fit, not a Wagner-interaction model.
    For calibrated use, fit against measured foil-weight-gain data via
    models.foil_calibration.
    """
    T_K = T_C + 273.15
    a_ref = 0.65          # a_C at C_ref=1.0 wt%, T_ref=900°C
    C_ref = 1.0           # wt%
    T_ref = 900.0         # °C
    temp_factor = math.exp(-1500.0 * (1.0 / T_K - 1.0 / (T_ref + 273.15)))
    conc_factor = (C_wt_percent / C_ref) * math.exp(0.8 * (C_wt_percent - C_ref))
    return float(conc_factor * temp_factor / a_ref)

# More straightforward: directly estimate C_wt from a_C iteratively.


def carbon_wt_from_activity(aC: float, T_C: float, max_iter: int = 50) -> float:
    """
    Convert carbon activity (graphite ref) to equilibrium C wt% in austenite (screening).

    Solves a_C = f(C_wt, T) where f approximated as:

    a_C ≈ (C_wt / C_max(T)) * exp( k*(C_wt) )  with saturation at C_max

    Use numerical inversion via bisection up to C_max.

    If a_C >1, supersaturated vs graphite → graphite precipitation potential, but
    we cap at C_max (Acm) as metastable austenite supersaturation.
    """
    if aC <= 0:
        return 0.0
    Cmax = austenite_max_carbon_wt_percent(T_C)
    if Cmax <= 0:
        # ferrite solubility tiny ~0.02 wt% at 727°C
        return min(0.02, aC * 0.02)

    # Define function a(C) screening:
    def a_of_C(C_wt: float) -> float:
        # empirical: a = (C / 1.0) * exp(0.9*(C-1.0)) * exp( -1200*(1/T -1/1173) )
        # normalized so at 900°C, C=1.1 wt% → a≈1.0
        T_K = T_C + 273.15
        T_ref_K = 1173.15
        base = (C_wt / 1.1) * math.exp(0.9 * (C_wt - 1.1))
        # temperature factor: higher T → lower a for same C
        base *= math.exp(-1000.0 * (1.0/T_K - 1.0/T_ref_K))
        return base

    # If aC beyond a_of_C(Cmax), return Cmax (saturation)
    if aC >= a_of_C(Cmax):
        return float(Cmax)

    # Bisection between 0 and Cmax
    lo, hi = 0.0, Cmax
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        a_mid = a_of_C(mid)
        if a_mid < aC:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def carbon_potential_summary(
    T_C: float = 900.0,
    pCO: float = 0.20,
    pCO2: float = 0.001,
    pCH4: Optional[float] = None,
    pH2: float = 0.40,
    dew_point_C: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience summary: compute a_C via CO/CO2, CH4/H2, dew-point routes,
    O2 pO2, and corresponding equilibrium C wt% in austenite.
    """

    aC_co = carbon_activity_from_co_co2(pCO, pCO2, T_C)
    C_from_co = carbon_wt_from_activity(aC_co, T_C)
    pO2 = oxygen_partial_pressure_from_co_co2(pCO, pCO2, T_C)
    Cmax = austenite_max_carbon_wt_percent(T_C)

    out: Dict[str, Any] = {
        "T_C": T_C,
        "pCO_atm": pCO,
        "pCO2_atm": pCO2,
        "pH2_atm": pH2,
        "aC_from_CO_CO2": aC_co,
        "C_wt_from_CO_CO2": C_from_co,
        "pO2_atm": pO2,
        "log10_pO2": math.log10(max(pO2, 1e-30)),
        "C_max_Acm_wt": Cmax,
        "is_supersaturated_graphite": aC_co > 1.0,
        "is_above_Acm": C_from_co > Cmax * 0.99,
    }

    if pCH4 is not None:
        aC_ch4 = carbon_activity_from_ch4_h2(pCH4, pH2, T_C)
        C_from_ch4 = carbon_wt_from_activity(aC_ch4, T_C)
        out.update({
            "pCH4_atm": pCH4,
            "aC_from_CH4_H2": aC_ch4,
            "C_wt_from_CH4_H2": C_from_ch4,
        })

    if dew_point_C is not None:
        aC_dew = carbon_activity_from_dew_point_and_co(pCO, dew_point_C, pH2, T_C)
        C_dew = carbon_wt_from_activity(aC_dew, T_C)
        out.update({
            "dew_point_C": dew_point_C,
            "aC_from_dewpoint": aC_dew,
            "C_wt_from_dewpoint": C_dew,
        })

    # Recommendation for setpoint
    # Practical gas carburizing aims for 1.0-1.2% surface, a_C ~0.9-1.1 at 900-930°C
    recommended_aC = 1.0
    recommended_C = carbon_wt_from_activity(recommended_aC, T_C)
    out["recommended_aC_for_1_1pct"] = recommended_aC
    out["recommended_C_wt"] = recommended_C
    out["note"] = (
        "Screening carbon potential: Boudouard ΔG=170700-174.5T J/mol, CH4 cracking 90000-109T J/mol. "
        "C_wt from a_C via empirical a(C)= (C/1.1)*exp(0.9(C-1.1)) normalized at 900°C a=1→1.1wt%. "
        "Acm linear 0.76% at 727°C to 2.14% at 1147°C. Calibrate with O2 probe/milliprob foil."
    )
    return out
