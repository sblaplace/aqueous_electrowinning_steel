"""
H₂ bubble detachment vs. deposit-front engulfment -> deposit porosity.

Why this module exists
----------------------
``gas_holdup.py`` tracks void fraction and current redistribution in the *channel*
and ``deposit_morphology.py`` classifies morphology — but nothing decides whether
a growing H₂ bubble is **captured by the advancing deposit front** (-> porosity,
pinholes, blisters) or detaches first. Deposit porosity is a first-order
product-quality metric (density, cold-roll ceiling, steel quality) and is
currently not predicted by any mechanism.

This module (Round 5, B3) computes a bubble detachment radius from surface
tension / contact angle, a deposit-front advance velocity from (j, FE), and a
capture criterion that yields a screening deposit-porosity fraction and a
pinhole/blister flag.

Screening flag
--------------
L1. The Fritz-type detachment-radius correlation and the capture geometry are
screening; calibrate against measured deposit density/porosity vs current density
and the bubble-departure size measured by the gas_holdup experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from models.thermodynamic_constants import FARADAY

SCREENING_FLAG = "unvalidated (L1)"

# Iron
M_FE = 55.845e-3  # kg/mol
RHO_FE = 7874.0   # kg/m³


@dataclass
class BubbleEngulfmentParams:
    """Screening parameters for H₂ bubble detachment and engulfment."""

    # Bubble detachment (Fritz-type): r_d = k_Fritz * theta_contact(rad) *
    #   sqrt( sigma / (g*(rho_l - rho_g)) ), k_Fritz ~ 0.0208.
    k_fritz: float = 0.0208
    sigma_surface_tension_N_m: float = 0.055  # acid sulfate, surfactant-laden
    rho_liquid_kg_m3: float = 1100.0
    theta_contact_deg: float = 90.0
    # Bubble surface coverage of the cathode (fraction); tied to HER rate / j.
    bubble_coverage_ref: float = 0.15
    j_coverage_ref_A_m2: float = 3000.0
    coverage_exponent: float = 0.5
    # Fraction of covered area that ends up captured once the front outruns the
    # bubble (depends on geometry; screening).
    capture_fraction_ref: float = 0.5
    # Blockage factor for the bubble diameter in the engulfment time scale.
    bubble_dia_factor: float = 2.0  # d_b ~ 2 * r_d


def bubble_detachment_radius_m(
    params: Optional[BubbleEngulfmentParams] = None,
) -> float:
    """Detachment radius of an H₂ bubble (m) from the Fritz correlation."""
    p = params or BubbleEngulfmentParams()
    theta_rad = math.radians(p.theta_contact_deg)
    g = 9.81
    return float(p.k_fritz * theta_rad * math.sqrt(
        p.sigma_surface_tension_N_m / (g * p.rho_liquid_kg_m3)))


def deposit_advance_velocity_m_s(
    j_Fe_A_m2: float,
    faradaic_efficiency: float = 1.0,
    params: Optional[BubbleEngulfmentParams] = None,
) -> float:
    """Deposit-front advance velocity (m/s) from the iron partial current."""
    j_fe = max(float(j_Fe_A_m2), 0.0)
    fe = min(max(float(faradaic_efficiency), 0.0), 1.0)
    # Fe²⁺ + 2e⁻ -> Fe ; v = j*FE*M/(2F*rho)
    return float(j_fe * fe * M_FE / (2.0 * FARADAY * RHO_FE))


def bubble_capture_porosity_fraction(
    j_Fe_A_m2: float,
    her_efficiency: float = 0.0,
    faradaic_efficiency: float = 1.0,
    params: Optional[BubbleEngulfmentParams] = None,
) -> dict:
    """
    Screening deposit-porosity fraction from H₂ bubble engulfment.

    Parameters
    ----------
    j_Fe_A_m2 : iron partial current (A/m²).
    her_efficiency : fraction of current going to HER (0..1). Determines the
        H₂ generation rate and hence bubble coverage.
    faradaic_efficiency : FE (Fe/total). Determines deposit advance rate.

    Returns
    -------
    dict with bubble_detachment_radius_um, bubble_coverage, deposit_velocity,
      engulfment_ratio, porosity_fraction (0..1), pinhole_blister_flag.
    """
    p = params or BubbleEngulfmentParams()
    j_total = max(float(j_Fe_A_m2) / max(float(faradaic_efficiency), 1e-9), 0.0)
    her = min(max(float(her_efficiency), 0.0), 1.0)

    r_d = bubble_detachment_radius_m(p)
    d_b = p.bubble_dia_factor * r_d

    # Bubble coverage rises with HER partial current.
    j_her = j_total * her
    coverage = min(p.bubble_coverage_ref * (j_her / p.j_coverage_ref_A_m2) ** p.coverage_exponent, 0.9)

    # Deposit advance velocity.
    v_dep = deposit_advance_velocity_m_s(j_Fe_A_m2, faradaic_efficiency, p)

    # Engulfment: if the deposit front advances a bubble diameter in less time
    # than the bubble takes to detach/grow away, it is captured. Screening proxy:
    # ratio = (time to cover a bubble) / (characteristic bubble dwell time).
    # Higher v_dep and larger coverage -> higher capture propensity.
    engulfment_ratio = 0.0
    if v_dep > 1e-12 and d_b > 0:
        engulfment_ratio = coverage * (d_b * v_dep) / 1e-9  # normalized, screening

    porosity = min(p.capture_fraction_ref * engulfment_ratio, 1.0)
    flag = porosity > 0.02  # 2 vol% screening pinhole/blister threshold

    return {
        "bubble_detachment_radius_um": r_d * 1e6,
        "bubble_coverage": float(coverage),
        "deposit_velocity_nm_s": v_dep * 1e9,
        "engulfment_ratio": float(engulfment_ratio),
        "porosity_fraction": float(porosity),
        "pinhole_blister_flag": bool(flag),
    }


def main() -> None:
    """CLI entrypoint for H₂ bubble engulfment / porosity."""
    print("=" * 70)
    print(" H₂ Bubble Engulfment -> Deposit Porosity (Round 5, B3)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    r = bubble_detachment_radius_m()
    print(f" Bubble detachment radius : {r*1e6:.1f} µm")

    for her in (0.05, 0.15, 0.30):
        res = bubble_capture_porosity_fraction(3000.0, her_efficiency=her)
        print(f"  HER eff = {her:4.2f}  -> coverage = {res['bubble_coverage']:.3f} "
              f"| porosity = {res['porosity_fraction']*100:5.2f} vol% "
              f"| {'BLISTER' if res['pinhole_blister_flag'] else 'ok'}")


if __name__ == "__main__":
    main()
