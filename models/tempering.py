"""
Tempering, retained-austenite, and Hollomon-Jaffe model for carburized
and as-deposited iron/steel.

Provides:

* Martensite start (Ms) temperature via Andrews / Payson-Savage empirical:
  Ms(°C) = 539 -423*C -30.4*Mn -17.7*Ni -12.1*Cr -7.5*Mo [+10 Co -7.5 Si]
* Koistinen-Marburger retained austenite: f_RA = exp(-α (Ms - T_q)), α≈0.011 K⁻¹
* Hollomon-Jaffe tempering parameter P = T*(C_HJ + log10(t)), C_HJ≈19.5-20
* Tempered hardness / YS softening: Maynier + Grange-Baughman / Kang-Lee
* Bainite/pearlite competition during slow quench
* Tempered case-core composite evolution
* Energy for tempering

All screening — calibrate with dilatometry, XRD RA, hardness traverses.

References (screening means):
* Andrews Ms: Ms(°C)=539-423C-30.4Mn-17.7Ni-12.1Cr-7.5Mo (°C, wt%)
* Koistinen-Marburger (1959): f_M =1- exp(-0.011*(Ms - Tq))
* Hollomon-Jaffe (1945): P = T(K) * (C_HJ + log10(t_s)), C≈19.5
* Grange et al. tempering of martensite: ΔHRC ≈ 0.5*P/1000? Simplified.
* Maynier hardness: base HV=127+949C+...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import math
import numpy as np

from .electrochemistry import R_GAS


@dataclass(frozen=True)
class AlloyComposition:
    """Alloy content wt% for Ms and hardenability."""

    C: float = 0.5
    Mn: float = 0.0
    Ni: float = 0.0
    Cr: float = 0.0
    Mo: float = 0.0
    Si: float = 0.0
    Co: float = 0.0

    def __post_init__(self):
        for k in ("C", "Mn", "Ni", "Cr", "Mo", "Si", "Co"):
            v = getattr(self, k)
            if v < 0 or v > 20:
                # allow 0-20 wt% but C typically <2.1
                if k == "C" and v > 6.67:
                    raise ValueError("C must be <6.67 wt%")
                # others okay up to 20
                pass


def martensite_start_C(chem: AlloyComposition) -> float:
    """
    Andrews linear Ms temperature (°C):
    Ms =539 -423*C -30.4*Mn -17.7*Ni -12.1*Cr -7.5*Mo +10*Co -7.5*Si
    Valid for 0.1-0.6% C low-alloy steels; extrapolation outside.
    """
    Ms = (
        539.0
        - 423.0 * chem.C
        - 30.4 * chem.Mn
        - 17.7 * chem.Ni
        - 12.1 * chem.Cr
        - 7.5 * chem.Mo
        - 7.5 * chem.Si
        + 10.0 * chem.Co
    )
    return float(Ms)


def retained_austenite_fraction_koistinen_marburger(
    Ms_C: float,
    Tq_C: float = 25.0,
    alpha_K_inv: float = 0.011,
) -> float:
    """
    Fraction of austenite remaining after quench to Tq:

    f_RA = exp(-α (Ms - Tq)) for Tq < Ms, else 1.0
    f_M = 1 - f_RA

    α≈0.011 K⁻¹ for many steels (0.008-0.015 range).
    T in °C (difference same in K).
    """

    if Tq_C >= Ms_C:
        return 1.0
    expo = -alpha_K_inv * (Ms_C - Tq_C)
    f_ra = math.exp(expo)
    return float(np.clip(f_ra, 0.0, 1.0))


def bainite_pearlite_fraction_quench(
    quench_rate_C_s: float,
    C_wt: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Screening non-martensite fraction vs quench rate.

    f_Mart ≈ 1 - exp(-quench/20) (critical ~20 C/s for plain C)
    Remainder split bainite/pearlite depending on C: higher C more bainite.

    Returns (f_mart, f_bainite, f_pearlite) fractions sum=1.
    """

    f_mart = 1.0 - math.exp(-quench_rate_C_s / 20.0)
    f_mart = float(np.clip(f_mart, 0.0, 1.0))
    rem = 1.0 - f_mart
    # Bainite favored at higher C and moderate rates
    f_bainite = rem * (0.3 + 0.4 * min(C_wt / 0.8, 1.0))
    f_pearlite = rem - f_bainite
    return f_mart, f_bainite, f_pearlite


def hollomon_jaffe_parameter(
    T_C: float,
    t_hr: float,
    C_HJ: float = 19.5,
) -> float:
    """
    P = T(K) * (C_HJ + log10(t_s))

    T_C tempering temperature °C, t_hr hours, C_HJ ~19.5-20.
    Returns P (K * log units). Typical P 8000-20000 for tempering 200-650°C.
    """
    T_K = T_C + 273.15
    t_s = max(t_hr * 3600.0, 1.0)
    P = T_K * (C_HJ + math.log10(t_s))
    return float(P)


def tempered_hardness_hollomon_jaffe(
    HV_as_quenched: float,
    P: float,
    k_softening: float = 0.00018,
    n: float = 1.0,
) -> float:
    """
    Screening tempered hardness drop:

    HV_t = HV_q * exp(-k * (P - P0)^n)  with P0 ~8000 (no tempering below).

    Or Grange: HRC drop ~ 0.35*(P-10000)/1000?

    Use exp form: HV_t = HV_q * (1 - 0.8*(1- exp(-(P-8000)/6000))) = earlier form,
    but provide parameterized.

    k_softening calibrated to give ~30% drop at P~15000 (500°C 1hr ≈ 15000).
    """

    P0 = 8000.0
    if P <= P0:
        return float(HV_as_quenched)
    delta = P - P0
    # softening factor
    # Use two-stage: f = exp(-k*delta^n)
    f = math.exp(-k_softening * (delta ** n))
    # More aggressive: limit to 0.35 floor
    f = max(f, 0.35)
    hv_t = HV_as_quenched * f
    # Pearlitic floor ~ 200 + 150*C
    hv_floor = 150.0
    return float(max(hv_t, hv_floor))


def tempered_yield_from_hardness(HV_kgf_mm2: float, factor: float = 3.0) -> float:
    """
    Tabor estimate: σ_y (MPa) ≈ HV * 9.81 / factor

    For tempered martensite, factor ~3.0-3.3.
    """
    return float(HV_kgf_mm2 * 9.80665 / factor)


def tempering_curve(
    HV_q: float = 800.0,
    chem: Optional[AlloyComposition] = None,
    C_HJ: float = 19.5,
    t_hr: float = 1.0,
    T_range_C: Tuple[float, float] = (150.0, 650.0),
    n_points: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Generate tempering curve HV vs T at fixed time.

    Returns dict with T_C, P, HV_tempered, YS_t, f_RA_decomposed estimate.
    """

    chem = chem or AlloyComposition(C=0.5)
    Ts = np.linspace(T_range_C[0], T_range_C[1], n_points)
    Ps = np.array([hollomon_jaffe_parameter(T, t_hr, C_HJ) for T in Ts])
    HVs = np.array([tempered_hardness_hollomon_jaffe(HV_q, P) for P in Ps])
    YSs = np.array([tempered_yield_from_hardness(hv) for hv in HVs])

    # RA decomposition: retained austenite decomposes during tempering 200-400°C
    # Screening: f_RA_t = f_RA_q * exp(- (T-200)/150) for T>200
    Ms = martensite_start_C(chem)
    f_RA_q = retained_austenite_fraction_koistinen_marburger(Ms, 25.0)
    f_RA_t = []
    for T in Ts:
        if T < 200:
            f_RA_t.append(f_RA_q)
        else:
            # decomposition starts at 200, half at 350°C
            f = f_RA_q * math.exp(-max(0.0, T - 200.0) / 150.0)
            f_RA_t.append(f)
    f_RA_t = np.array(f_RA_t)

    return {
        "T_C": Ts,
        "P": Ps,
        "HV_tempered": HVs,
        "YS_MPa": YSs,
        "f_RA_remaining": f_RA_t,
        "f_RA_as_quenched": np.full_like(Ts, f_RA_q),
        "Ms_C": np.full_like(Ts, Ms),
        "t_hr": np.full_like(Ts, t_hr),
    }


def case_hardness_after_tempering(
    c_profile_wt: np.ndarray,
    x_um: np.ndarray,
    temper_T_C: float = 180.0,
    temper_t_hr: float = 1.0,
    quench_rate_C_s: float = 200.0,
    chem_base: Optional[AlloyComposition] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given carbon profile (wt%) vs depth, compute as-quenched HV,
    retained austenite profile, and tempered HV.

    Returns (HV_q, f_RA_q, HV_tempered) arrays matching x_um.
    """

    # As-quenched HV from carbon (screening Maynier)
    # Use function from carburization module to avoid circular import: reimplement simple
    def hv_from_c(C):
        # Maynier simplified
        hv = 127.0 + 949.0 * np.asarray(C)
        hv = np.minimum(hv, 900.0)
        hv = np.maximum(hv, 80.0)
        # quench rate knockdown
        if quench_rate_C_s < 30:
            f_mart = 1.0 - np.exp(-quench_rate_C_s / 20.0)
            hv_bain = 200.0 + 200.0 * np.asarray(C)
            hv = f_mart * hv + (1 - f_mart) * hv_bain
        return hv

    HV_q = hv_from_c(c_profile_wt)

    # RA fraction per point based on local C
    chem_base = chem_base or AlloyComposition()
    f_RA = []
    for C in c_profile_wt:
        chem_local = AlloyComposition(
            C=float(C),
            Mn=chem_base.Mn,
            Ni=chem_base.Ni,
            Cr=chem_base.Cr,
            Mo=chem_base.Mo,
            Si=chem_base.Si,
        )
        Ms = martensite_start_C(chem_local)
        fra = retained_austenite_fraction_koistinen_marburger(Ms, 25.0)
        f_RA.append(fra)
    f_RA = np.array(f_RA)

    # Tempered hardness
    P = hollomon_jaffe_parameter(temper_T_C, temper_t_hr)
    HV_t = np.array([tempered_hardness_hollomon_jaffe(hv, P) for hv in HV_q])

    return HV_q, f_RA, HV_t


def recommended_tempering_for_target_hv(
    HV_q: float,
    target_HV: float,
    t_hr: float = 1.0,
    C_HJ: float = 19.5,
    T_min: float = 150.0,
    T_max: float = 700.0,
) -> Optional[float]:
    """
    Inverse: find tempering T to reach target HV at fixed time, via bisection.

    Returns T_C or None if not reachable in range.
    """

    if target_HV >= HV_q:
        return T_min if abs(target_HV - HV_q) < 10 else None

    def f(T):
        P = hollomon_jaffe_parameter(T, t_hr, C_HJ)
        hv_t = tempered_hardness_hollomon_jaffe(HV_q, P)
        return hv_t - target_HV

    # Check brackets
    f_min = f(T_min)
    f_max = f(T_max)
    if f_min * f_max > 0:
        # Both same sign, no root in interval
        return None

    # Bisection
    lo, hi = T_min, T_max
    for _ in range(40):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if abs(fm) < 1.0:
            return float(mid)
        if f(lo) * fm <= 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2.0)
