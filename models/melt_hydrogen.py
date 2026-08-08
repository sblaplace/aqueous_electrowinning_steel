"""
Hydrogen carried from an electrowon iron charge into the steel melt: white-spot
/ flake risk.

Why this module exists
----------------------
The program's near-term product (docs/RESEARCH_PROGRAM.md, Option A) is a
**melt-shop iron feedstock** — powder, flake or foil charged to an EAF/induction
furnace. None of the existing hydrogen models answers the melt-shop buyer's
question: *"your flake is H-rich — what does that do to my ingot?"*

The physics (Round 5, B2): liquid iron dissolves far more hydrogen than solid
iron. Under the Sieverts law, molten Fe holds on the order of 25 ppm H at
1600 °C / 1 atm H₂, while solid δ/γ retains only a few ppm. When an H-rich
electrowon charge melts and resolidifies, the excess hydrogen supersaturates and
exsolves as molecular H₂ in voids/microporosity, producing internal **flake
cracks / white spots / fish-eyes** — the classic hydrogen defect in large steel
sections.

This module converts a deposit H content into a charge H budget, a
liquid/solid solubility gap, a flake-risk index and a required pre-melt
bake-out (reusing hydrogen_trapping.py's bakeout) or melt-side degas.

Screening flag
--------------
L1 screening. Sieverts constants are literature anchors; the section-size /
flake-threshold mapping is a screening proxy to be tuned against foundry
experience.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SCREENING_FLAG = "unvalidated (L1)"

# Sieverts-law constant for H in liquid iron at 1600 C: C_H(ppm) = K * sqrt(p_H2/atm)
# Literature: liquid Fe at 1600 C ~ 25 ppm at 1 atm H2.
K_SIEVERTS_LIQUID_1600C = 25.0  # ppm H at 1 atm, 1600 C
# Solid (δ/γ) at melting point retains far less (~3 ppm at 1 atm).
K_SIEVERTS_SOLID_1530C = 3.0

# van't Hoff-style enthalpy for H dissolution into liquid Fe (J/mol); screening.
DH_H_DISSOLUTION_J_MOL = 60.0e3
R_GAS = 8.314


@dataclass
class MeltHydrogenParams:
    """Screening parameters for melt hydrogen / white-spot risk."""

    # Reference melting / handling temperature (C)
    t_melt_C: float = 1600.0
    t_solidus_C: float = 1530.0
    # Solid H retained at solidus, ppm (supersaturation reference).
    c_h_solid_ppm: float = 3.0
    # Threshold excess H (liquid minus retained solid) above which flake risk
    # becomes material in a heavy section (ppm).
    excess_h_flake_threshold_ppm: float = 8.0
    # Section-size sensitivity exponent (bigger section -> more flake-prone).
    section_size_ref_m: float = 0.3
    section_size_exp: float = 0.5
    # Fraction of flake H that actually reaches the melt from a charged deposit
    # (the rest desorbs from powder surface / off-gasses).
    charge_h_transfer_fraction: float = 0.6


def sieverts_H_ppm_liquid(temperature_C: float, p_h2_atm: float = 1.0) -> float:
    """Hydrogen solubility in liquid iron (ppm) by the Sieverts law."""
    # K(T) = K_ref * exp(-dH/R (1/T - 1/T_ref)); H solubility increases with T.
    t_k = temperature_C + 273.15
    t_ref_k = 1600.0 + 273.15
    k = K_SIEVERTS_LIQUID_1600C * math.exp(
        -DH_H_DISSOLUTION_J_MOL / R_GAS * (1.0 / t_k - 1.0 / t_ref_k))
    return float(k * math.sqrt(max(p_h2_atm, 1e-9)))


def melt_hydrogen_budget(
    c_h_deposit_ppm: float,
    charge_fraction: float = 1.0,
    p_h2_atm: float = 1.0,
    params: Optional[MeltHydrogenParams] = None,
) -> dict:
    """
    Hydrogen budget when an H-bearing electrowon charge is melted.

    Parameters
    ----------
    c_h_deposit_ppm : diffusible H in the as-deposited charge (ppm).
    charge_fraction : fraction of the furnace charge that is electrowon iron
        (the rest, e.g. scrap, carries its own H).
    p_h2_atm : H₂ partial pressure over the melt (roughly the H₂ from the
        deposit desorbing; 1 atm screening).

    Returns
    -------
    dict with:
      h_in_melt_ppm        : H delivered to the melt from the charge
      liquid_solubility_ppm: Sieverts solubility of liquid Fe at p_h2_atm
      retained_solid_ppm   : H the solid can retain (supersaturation baseline)
      excess_h_ppm         : liquid H minus retained solid H
      flake_risk_index     : 0..1 white-spot propensity (screening)
      needs_bake_or_degas  : bool, excess_h above flake threshold
    """
    p = params or MeltHydrogenParams()
    c_dep = max(float(c_h_deposit_ppm), 0.0)

    h_in_melt = c_dep * charge_fraction * p.charge_h_transfer_fraction
    liquid_sol = sieverts_H_ppm_liquid(p.t_melt_C, p_h2_atm)
    excess = max(h_in_melt - p.c_h_solid_ppm, 0.0)

    # Screening flake-risk index: rises with excess H and with section size.
    risk = (excess / p.excess_h_flake_threshold_ppm) ** p.section_size_exp
    risk = min(max(risk, 0.0), 1.0)
    needs = excess > p.excess_h_flake_threshold_ppm

    return {
        "h_in_melt_ppm": float(h_in_melt),
        "liquid_solubility_ppm": float(liquid_sol),
        "retained_solid_ppm": float(p.c_h_solid_ppm),
        "excess_h_ppm": float(excess),
        "flake_risk_index": float(risk),
        "needs_bake_or_degas": bool(needs),
    }


def required_bakeout_C_H_ppm(
    c_h_deposit_ppm: float,
    charge_fraction: float = 1.0,
    params: Optional[MeltHydrogenParams] = None,
) -> float:
    """
    Maximum deposit H content (ppm) that keeps the melt below the flake threshold.

    Inverts the budget: require excess_h <= threshold. Used to set a deposit H
    product spec for the feedstock business, and as the target for the
    hydrogen_trapping.py bake-out.
    """
    p = params or MeltHydrogenParams()
    allowed_excess = float(p.excess_h_flake_threshold_ppm)
    # excess = c_dep*frac*transfer - c_solid <= allowed_excess
    denom = max(charge_fraction * p.charge_h_transfer_fraction, 1e-12)
    return float((allowed_excess + p.c_h_solid_ppm) / denom)


def main() -> None:
    """CLI entrypoint for melt hydrogen / white-spot risk analysis."""
    print("=" * 70)
    print(" Melt Hydrogen -> White-Spot / Flake Risk (Round 5, B2)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    print(f" Liquid Fe Sieverts solubility @1600C/1atm : "
          f"{sieverts_H_ppm_liquid(1600.0):.1f} ppm")

    for c_h in (1.0, 5.0, 10.0, 25.0):
        res = melt_hydrogen_budget(c_h)
        print(f"\n Deposit H = {c_h:5.1f} ppm")
        print(f"   -> melt H = {res['h_in_melt_ppm']:.2f} ppm | excess = "
              f"{res['excess_h_ppm']:.2f} ppm | risk = {res['flake_risk_index']:.2f} "
              f"| {res['needs_bake_or_degas'] and 'BAKE/DEGAS' or 'OK'}")

    print(f"\n Allowable deposit H for flake-safe melt : "
          f"{required_bakeout_C_H_ppm(25.0):.1f} ppm max")


if __name__ == "__main__":
    main()
