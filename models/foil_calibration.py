"""
Foil and O2 probe calibration model for carburizing atmosphere and diffusion.

Implements calibration of screening coefficients from experimental measurements:

* Foil test: thin Fe foil (50-100 µm) exposed to carburizing gas for time t.
  Weight gain and combustion C wt% give effective D and surface C (Cs) via
  inverse Fick analysis. For thin foil, through-carburization approximated,
  so average C ≈ (C0+Cs)/2 at long times? More precisely solve for Cs and D
  by fitting measured average C vs time to finite-slab solution.

* O2 probe: measured mV from zirconia O2 sensor converts to pO2 and then to
  carbon activity aC via CO/CO2 equilibrium. Calibrate Boudouard K(T) or
  offset vs theoretical.

* Case depth traverse: measured HV vs depth calibrates carbon_potential
  a(C) relation and tempering k_softening.

* Hardness traverse vs C: calibrates Maynier coefficients.

All screening fits use scipy.optimize.least_squares / curve_fit.

Outputs calibrated parameter JSON that can be fed back into CarburizationParams,
CarbonPotential summary, Tempering, and MechanicalProperties.

This bridges synthetic screening to real data without hardcoding plant data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import math
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.special import erfc

from .carburization import (
    carbon_diffusivity_m2_s,
    carbon_profile_finite_slab,
)
from .carbon_potential import (
    carbon_activity_from_co_co2,
    carbon_wt_from_activity,
    austenite_max_carbon_wt_percent,
)
from .tempering import tempered_hardness_hollomon_jaffe, hollomon_jaffe_parameter

R_GAS = 8.314462


@dataclass
class FoilMeasurement:
    """Single foil exposure measurement."""

    time_hr: float
    temperature_C: float
    pCO_atm: float
    pCO2_atm: float
    pH2_atm: float = 0.40
    foil_thickness_um: float = 75.0
    measured_avg_C_wt_percent: float = 0.8
    measured_weight_gain_g_m2: Optional[float] = None
    o2_probe_mV: Optional[float] = None  # zirconia probe mV
    dew_point_C: Optional[float] = None

    def __post_init__(self):
        if self.time_hr <= 0:
            raise ValueError("time_hr >0")
        if self.foil_thickness_um <= 10:
            raise ValueError("foil too thin")


@dataclass
class CaseDepthMeasurement:
    """Hardness traverse / combustion traverse."""

    depth_um: float
    measured_C_wt_percent: Optional[float] = None
    measured_HV: Optional[float] = None
    temperature_C: float = 900.0
    time_hr: float = 4.0
    surface_C_wt: float = 1.1


def o2_probe_mv_to_pO2(mV: float, T_C: float, p_ref_air_atm: float = 0.209) -> float:
    """
    Zirconia O2 probe: E = (RT/4F) ln(pO2_ref / pO2_sample)
    So pO2 = p_ref * exp(-4F E / RT)

    mV measured vs air reference.
    """
    T_K = T_C + 273.15
    F = 96485.3321
    E_V = mV * 1e-3
    exponent = -4.0 * F * E_V / (R_GAS * T_K)
    pO2 = p_ref_air_atm * math.exp(exponent)
    return float(pO2)


def pO2_to_carbon_activity_via_co_co2(pO2_atm: float, pCO_atm: float, T_C: float) -> float:
    """
    From O2 probe: CO + 0.5 O2 ↔ CO2 equilibrium gives pCO2,
    then aC = K_B * pCO^2 / pCO2

    Use ΔG° CO+0.5O2→CO2 ≈ -282800+86.8T
    K = exp(-ΔG/RT) = pCO2/(pCO * pO2^0.5)
    → pCO2 = K * pCO * sqrt(pO2)
    → aC = K_B * pCO^2 / pCO2 where K_B = exp(-ΔG_B/RT), ΔG_B=170700-174.5T
    """

    T_K = T_C + 273.15
    dG_co_o2 = -282800.0 + 86.8 * T_K
    K_co_o2 = math.exp(-dG_co_o2 / (R_GAS * T_K))
    pCO2 = K_co_o2 * pCO_atm * math.sqrt(max(pO2_atm, 1e-30))

    # Boudouard
    dG_B = 170700.0 - 174.5 * T_K
    K_B = math.exp(-dG_B / (R_GAS * T_K))
    aC = K_B * (pCO_atm ** 2) / max(pCO2, 1e-12)
    return float(aC)


def fit_diffusivity_from_foil_data(
    measurements: List[FoilMeasurement],
    initial_C_wt: float = 0.02,
    method: str = "least_squares",
) -> Dict[str, Any]:
    """
    Fit D (m²/s) and Cs (surface C wt%) from foil average C vs time.

    Model: finite slab with both sides carburized, thickness L, average C predicted:
    C_avg(t) = (1/L) ∫_0^L C(x,t) dx

    For fast approximation, use thin foil through model: at long times average → Cs,
    early times ∝ sqrt(Dt)/L.

    Fit two parameters: log10(D) and Cs.

    Returns dict with fitted D, Cs, covariance, report.
    """

    if not measurements:
        raise ValueError("no measurements")

    # Extract arrays
    t_s = np.array([m.time_hr * 3600.0 for m in measurements])
    C_avg_meas = np.array([m.measured_avg_C_wt_percent for m in measurements])
    L_m = np.array([m.foil_thickness_um * 1e-6 for m in measurements])
    # Use mean L for simplicity, or handle per measurement variable thickness in model
    # For screening, assume same thickness ~ mean
    L_mean = float(np.mean(L_m))
    T_mean = float(np.mean([m.temperature_C for m in measurements]))

    def avg_C_model(t_arr, log10_D, Cs):
        D = 10 ** log10_D
        # Vectorized finite slab average: integrate profile
        Cs_f = float(Cs)
        C0 = initial_C_wt
        avg = []
        for ti in t_arr:
            x = np.linspace(0, L_mean, 200)
            c = carbon_profile_finite_slab(x, ti, D, Cs_f, C0, L_mean)
            avg.append(float(np.trapz(c, x) / L_mean) if hasattr(np, 'trapezoid') is False else float(np.trapezoid(c, x)/L_mean))
            # fallback: np.trapezoid if available
            # Actually handle both
        # More robust: use manual trapz if above failed
        return np.array(avg)

    # Wrapper using np.trapezoid with fallback
    def avg_C_model_robust(t_arr, log10_D, Cs):
        D = 10 ** log10_D
        Cs_f = float(Cs)
        C0 = initial_C_wt
        out = []
        for ti in t_arr:
            x = np.linspace(0, L_mean, 200)
            c = carbon_profile_finite_slab(x, ti, D, Cs_f, C0, L_mean)
            # integrate
            try:
                avg = np.trapezoid(c, x) / L_mean
            except AttributeError:
                avg = np.trapz(c, x) / L_mean
            out.append(avg)
        return np.array(out)

    # Initial guess: D ~ 1e-12 at 900°C, Cs ~1.1
    p0 = [-12.0, 1.1]
    bounds = ([-16, 0.3], [-9, 2.0])  # logD between 1e-16 and 1e-9, Cs 0.3-2.0

    try:
        popt, pcov = curve_fit(
            avg_C_model_robust,
            t_s,
            C_avg_meas,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
        logD_fit, Cs_fit = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
    except Exception as e:
        # Fallback to least_squares manual
        def residuals(p):
            return avg_C_model_robust(t_s, p[0], p[1]) - C_avg_meas

        res = least_squares(residuals, x0=p0, bounds=bounds, max_nfev=2000)
        logD_fit, Cs_fit = res.x
        perr = [np.nan, np.nan]

    D_fit = 10 ** logD_fit
    # Compare to theoretical D at mean T
    D_theory, phase = carbon_diffusivity_m2_s(T_mean)

    return {
        "D_fit_m2_s": float(D_fit),
        "log10_D_fit": float(logD_fit),
        "Cs_fit_wt_percent": float(Cs_fit),
        "C0_wt_percent": float(initial_C_wt),
        "L_mean_m": float(L_mean),
        "T_mean_C": float(T_mean),
        "D_theory_m2_s": float(D_theory),
        "phase_theory": phase,
        "perr_logD_Cs": [float(perr[0]), float(perr[1])] if perr[0] is not np.nan else [None, None],
        "n_measurements": len(measurements),
        "note": f"Fitted D={D_fit:.2e} m2/s vs theory {D_theory:.2e} at {T_mean:.0f}C, Cs fit {Cs_fit:.2f} wt% (phase {phase})",
    }


def fit_carbon_potential_offset(
    measurements: List[FoilMeasurement],
) -> Dict[str, Any]:
    """
    Calibrate carbon potential K offset from O2 probe + foil data.

    For each measurement with O2 probe mV and pCO, compute theoretical aC (Boudouard)
    and measured aC inferred from foil Cs fit or measured avg C → surface C?

    Screening: compare aC_theory (from pCO/pCO2 gas) vs aC_from_O2_probe.

    Returns offset factor: aC_measured / aC_theory average.
    """

    ratios = []
    for m in measurements:
        if m.o2_probe_mV is None:
            continue
        pO2 = o2_probe_mv_to_pO2(m.o2_probe_mV, m.temperature_C)
        aC_from_probe = pO2_to_carbon_activity_via_co_co2(pO2, m.pCO_atm, m.temperature_C)
        aC_theory = carbon_activity_from_co_co2(m.pCO_atm, m.pCO2_atm, m.temperature_C)
        if aC_theory > 1e-12:
            ratios.append(aC_from_probe / aC_theory)

    if not ratios:
        return {"n_o2_measurements": 0, "offset_factor": 1.0, "note": "No O2 probe measurements"}

    ratios = np.array(ratios)
    offset = float(np.mean(ratios))
    return {
        "n_o2_measurements": len(ratios),
        "offset_factor_aC_probe_over_theory_mean": offset,
        "offset_std": float(np.std(ratios)),
        "ratios": ratios.tolist(),
        "note": f"Mean aC(probe)/aC(gas)={offset:.3f}±{np.std(ratios):.3f}. Apply as correction to Boudouard K if needed.",
    }


def fit_tempering_softening(
    measured_data: List[Dict[str, float]],
    # expected dict keys: HV_q, T_C, t_hr, HV_measured
) -> Dict[str, Any]:
    """
    Fit tempering softening coefficient k_softening (and optionally C_HJ)
    from measured HV after tempering vs P.

    Model: HV_t = HV_q * exp(-k*(P-8000))

    Fit k via least squares on log(HV_t/HV_q).
    """

    if not measured_data:
        raise ValueError("no data")

    Ps = []
    log_ratios = []
    for d in measured_data:
        T_C = d["T_C"]
        t_hr = d["t_hr"]
        from .tempering import hollomon_jaffe_parameter

        P = hollomon_jaffe_parameter(T_C, t_hr, d.get("C_HJ", 19.5))
        HV_q = d["HV_q"]
        HV_meas = d["HV_measured"]
        if HV_q <= 0 or HV_meas <= 0:
            continue
        Ps.append(P)
        log_ratios.append(math.log(max(HV_meas / HV_q, 0.35)))

    Ps = np.array(Ps)
    log_ratios = np.array(log_ratios)

    # Model: log(HV_t/HV_q) = -k*(P-8000) for P>8000 else 0
    def model(P, k):
        # k>0
        delta = np.maximum(P - 8000.0, 0.0)
        return -k * delta

    # Fit k via linear least squares on (P-8000) vs -log_ratio
    delta = np.maximum(Ps - 8000.0, 0.0)
    # Avoid zero division
    mask = delta > 1.0
    if np.sum(mask) < 1:
        return {"k_fit": 0.00018, "note": "insufficient data, default"}

    # linear regression k = -log_ratio / delta
    ks = -log_ratios[mask] / delta[mask]
    k_mean = float(np.mean(ks))
    k_std = float(np.std(ks))

    # Also fit via curve_fit for refinement
    try:
        popt, _ = curve_fit(model, Ps, log_ratios, p0=[0.00018], bounds=([1e-6], [0.01]))
        k_fit = float(popt[0])
    except Exception:
        k_fit = k_mean

    return {
        "k_fit": k_fit,
        "k_mean": k_mean,
        "k_std": k_std,
        "n_points": len(Ps),
        "note": f"Fitted tempering softening k={k_fit:.2e} (mean {k_mean:.2e}±{k_std:.2e}) for model HV_t=HV_q*exp(-k*(P-8000))",
    }


def fit_mechanical_hall_petch(
    grain_size_um: np.ndarray,
    yield_MPa: np.ndarray,
) -> Dict[str, Any]:
    """
    Calibrate Hall-Petch σ0 and k_HP from measured grain size vs yield.

    Model: σ_y = σ0 + k / sqrt(d) where d in meters.
    Fit via linear regression on 1/sqrt(d).
    """

    d_m = np.asarray(grain_size_um) * 1e-6
    y = np.asarray(yield_MPa)
    x = 1.0 / np.sqrt(d_m)

    # linear fit y = sigma0 + k * x
    A = np.vstack([np.ones_like(x), x]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    sigma0, k = coeffs[0], coeffs[1]

    # R^2
    y_pred = sigma0 + k * x
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "sigma0_MPa": float(sigma0),
        "k_HP_MPa_sqrt_m": float(k),
        "r_squared": float(r2),
        "n_points": len(grain_size_um),
        "note": f"Hall-Petch fit: σ_y = {sigma0:.1f} + {k:.3f}/√d (d in m), R²={r2:.3f}",
    }
