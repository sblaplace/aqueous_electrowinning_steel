"""
As-deposited Fe-C layer -> steel-grade router (coupling seam, Round 5, A2).

Why this module exists
----------------------
The product path has three carbon-bearing pipelines that do not talk:
``co_deposition.py`` (C in the layer), ``carbon_potential.py``/``carburization.py``
(post-deposition carburizing), and ``tempering.py``/``thermomechanical.py``
(final microstructure/mechanical). None maps an as-deposited carbon-bearing iron
layer through the Fe-C phase field to an actual steel grade.

This module is a **coupling seam** (not new materials physics): it takes deposit
carbon (from ``carbon_electrodeposition`` A1 or carburization), Mn/S/P (from
``bath_impurity_codeposition``), and a thermal path, and routes to phase
fractions + AISI grade + mechanicals, reusing the Fe-C phase-field logic already
implicit in the carburization/tempering kernels.

Screening flag
--------------
L1. Eutectoid temperature/composition are classic Fe-C values; the phase-fraction
mapping is a screening approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from models.carbon_electrodeposition import steel_grade_for_carbon

SCREENING_FLAG = "unvalidated (L1)"

# Fe-C equilibrium points (hypoeutectoid steel).
EUTECTOID_C_WT = 0.77
A1_TEMP_C = 727.0
A3_HYPEREUTECTOID_C_WT = 2.11


@dataclass
class AsDepositedGradeParams:
    """Screening parameters for the as-deposited grade router."""

    # Phase-fraction exponents (Avrami-like) for a given cooling path.
    martensite_cooling_factor: float = 0.90  # fraction martensite at fast quench
    # Mn and C strengthen ferrite/pearlite; Mn adds hardenability.
    mn_hardenability_coeff: float = 0.05


def phase_fractions(
    carbon_wt_percent: float,
    cooling: str = "slow",  # "slow" (furnace) | "fast" (quench)
    params: Optional[AsDepositedGradeParams] = None,
) -> dict:
    """
    Fe-C phase fractions for a given carbon content and cooling path.

    Returns dict with ferrite, pearlite, martensite, cementite fractions (0..1).
    """
    p = params or AsDepositedGradeParams()
    c = max(float(carbon_wt_percent), 0.0)

    if cooling == "fast":
        # Fast quench -> largely martensite (with residual austenite ignored here).
        mart = p.martensite_cooling_factor
        # Martensite hardness rises with C; above ~0.5 C retained austenite grows.
        # Screening: reduce martensite above 0.6 C to reflect retained austenite.
        if c > 0.6:
            mart *= (1.0 - 0.3 * (c - 0.6) / 1.0)
        mart = max(min(mart, 1.0), 0.0)
        return {"ferrite": round(1.0 - mart, 4), "pearlite": 0.0,
                "martensite": round(mart, 4), "cementite": 0.0}

    # Slow cooling: equilibrium ferrite + pearlite below eutectoid; above
    # eutectoid, pearlite + proeutectoid cementite.
    if c <= EUTECTOID_C_WT:
        # Hypoeutectoid: ferrite fraction = (0.77 - c)/(0.77 - 0.022)
        ferro = (EUTECTOID_C_WT - c) / (EUTECTOID_C_WT - 0.022)
        pearlite = 1.0 - ferro
        return {"ferrite": round(min(max(ferro, 0.0), 1.0), 4),
                "pearlite": round(min(max(pearlite, 0.0), 1.0), 4),
                "martensite": 0.0, "cementite": 0.0}
    else:
        # Hypereutectoid: pearlite + proeutectoid cementite.
        c_max = A3_HYPEREUTECTOID_C_WT
        cement = (c - EUTECTOID_C_WT) / (c_max - EUTECTOID_C_WT)
        cement = min(max(cement, 0.0), 1.0)
        return {"ferrite": 0.0, "pearlite": round(1.0 - cement, 4),
                "martensite": 0.0, "cementite": round(cement, 4)}


def as_deposited_grade(
    carbon_wt_percent: float,
    mn_wt_percent: float = 0.0,
    s_wt_percent: float = 0.0,
    p_wt_percent: float = 0.0,
    cooling: str = "slow",
    params: Optional[AsDepositedGradeParams] = None,
) -> dict:
    """
    Route an as-deposited carbon-bearing iron layer to a steel grade.

    Returns dict with grade, phase fractions, hardness/strength proxies, and
    quality flags (S/P deep-draw eligibility).
    """
    p = params or AsDepositedGradeParams()
    phases = phase_fractions(carbon_wt_percent, cooling, p)
    grade = steel_grade_for_carbon(carbon_wt_percent)

    # Hardness proxy: martensite hard and C-dependent; pearlite/ferrite soft.
    hv = (phases["martensite"] * (120 + 250 * carbon_wt_percent)
          + phases["pearlite"] * 180
          + phases["ferrite"] * 90
          + phases["cementite"] * 800)
    # Yield proxy: ferrite soft, martensite strong, C/Mn strengthen.
    ys = (phases["ferrite"] * 210
          + phases["pearlite"] * 330
          + phases["martensite"] * (700 + 500 * carbon_wt_percent)
          + 30 * mn_wt_percent)
    deep_draw = (s_wt_percent <= 0.02) and (p_wt_percent <= 0.02)
    return {
        "grade": grade,
        "phase_fractions": phases,
        "hardness_HV_proxy": round(hv, 1),
        "yield_strength_MPa_proxy": round(ys, 1),
        "deep_draw_eligible": bool(deep_draw),
        "cooling": cooling,
    }


def main() -> None:
    """CLI entrypoint for as-deposited grade routing."""
    print("=" * 70)
    print(" As-Deposited Fe-C -> Steel-Grade Router (Round 5, A2)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    for c in (0.02, 0.15, 0.45, 0.8):
        for cool in ("slow", "fast"):
            res = as_deposited_grade(c, cooling=cool)
            ph = res["phase_fractions"]
            print(f"  C={c:4.2f} {cool:4s} -> {res['grade']:<32s} "
                  f"HV={res['hardness_HV_proxy']:6.1f} "
                  f"YS={res['yield_strength_MPa_proxy']:6.1f} "
                  f"(F{ph['ferrite']:.0f} P{ph['pearlite']:.0f} M{ph['martensite']:.0f})")


if __name__ == "__main__":
    main()
