"""
Recombination poisons and their control of absorbed hydrogen in the deposit.

Why this module exists
----------------------
``hydrogen_embrittlement.py`` treats the fraction of evolved hydrogen that
enters the iron lattice with a fixed screening value (``absorption_fraction``,
~0.05 at pH 3.5, scaled by pH / T / j). In a real cell the *dominant* control
on that fraction is the surface coverage of **cathodic recombination poisons** —
sulfide (S²⁻/HS⁻), arsenic, antimony, selenium, tellurium, phosphorus, cyanide —
which block the Tafel / Heyrovský H₂ recombination step and force a far larger
fraction of adsorbed H into the deposit. This is exactly the case for
electrowon iron made from spent pickle liquor / steel-mill dust feedstock,
where trace S in the bath is common.

This module (Round 5, B1) computes a poison coverage and a promotion factor that
replaces the fixed ``absorption_fraction`` in ``hydrogen_embrittlement.py``:

    feedstock S → bath S²⁻/HS⁻ → θ_poison(η) → absorption_promotion_factor → C_H

Scope / screening flag
----------------------
This is a **screening (L1) module**, per the repository convention: the numbers
are central values to be replaced by divided-cell permeation / bake-out
measurements (``hydrogen_trapping.py`` bakeout, ``run_hydrogen_embrittlement.py``).
The classical result — trace As/S raise permeation by orders of magnitude —
is robust; the exact promotion coefficients are not.

References
----------
* Bockris & Reddy, "Modern Electrochemistry."
* Zakroczymski, T. (1991) on the effect of S and As on hydrogen permeation.
* Smialowski, "Hydrogen in Steel" (1962) — recombination poisons / absorption.
* McCright & Staehle (1974) — H-entry promoters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from models.thermodynamic_constants import R_GAS

SCREENING_FLAG = "unvalidated (L1)"


@dataclass
class PoisonParams:
    """
    Screening adsorption and promotion parameters for cathodic recombination poisons.

    Each poison has:
    * an adsorption affinity K (L/mol), temperature-adjusted via van't Hoff from
      a reference value and an enthalpy dH;
    * a promotion coefficient p: the multiplicative boost to absorbed-H
      efficiency at full (monolayer) coverage of that poison.

    Values are screening anchors drawn from the H-permeation literature on steel
    (relative rankings: As and S are the strongest promoters; P and CN the
    weakest of the group). Calibrate against real permeation data.
    """

    # Adsorption affinity at 25 C (L/mol) and van't Hoff enthalpy (J/mol).
    # Units: L/mol of dissolved poison concentration (mol/L).
    K_ref_L_mol: Dict[str, float] = field(default_factory=lambda: {
        "sulfide": 8.0e2,     # S²⁻/HS⁻ strong adsorber
        "arsenic": 6.0e2,     # As(III)/AsH₃
        "antimony": 5.0e2,    # Sb
        "selenium": 4.0e2,    # Se
        "tellurium": 3.0e2,   # Te
        "phosphorus": 2.0e2,  # P
        "cyanide": 1.0e2,     # CN⁻
    })
    dH_ads_J_mol: Dict[str, float] = field(default_factory=lambda: {
        "sulfide": -25.0e3,
        "arsenic": -28.0e3,
        "antimony": -26.0e3,
        "selenium": -24.0e3,
        "tellurium": -23.0e3,
        "phosphorus": -20.0e3,
        "cyanide": -18.0e3,
    })
    # Promotion coefficient at full coverage (multiplicative boost to absorbed-H).
    p_promote: Dict[str, float] = field(default_factory=lambda: {
        "sulfide": 250.0,
        "arsenic": 800.0,
        "antimony": 300.0,
        "selenium": 200.0,
        "tellurium": 150.0,
        "phosphorus": 40.0,
        "cyanide": 25.0,
    })
    # Potential dependence exponent: more negative cathodic potential favours
    # poison adsorption/deposition. Factor = (|eta|/eta_ref)^exp, saturating.
    eta_ref_V: float = 0.30
    eta_exponent: float = 0.5
    # Hard cap on the total promotion factor (avoid unbounded screening blow-ups).
    max_promotion_factor: float = 2000.0
    t_ref_K: float = 298.15

    def poison_names(self) -> tuple:
        return tuple(self.K_ref_L_mol.keys())


def adsorption_constant_L_mol(name: str, temperature_C: float,
                              params: Optional[PoisonParams] = None) -> float:
    """Temperature-adjusted Langmuir adsorption constant for one poison."""
    p = params or PoisonParams()
    k_ref = p.K_ref_L_mol[name]
    dH = p.dH_ads_J_mol.get(name, -20.0e3)
    t_k = temperature_C + 273.15
    return k_ref * math.exp(-dH / R_GAS * (1.0 / t_k - 1.0 / p.t_ref_K))


def poison_coverage_fraction(
    concentrations_M: Dict[str, float],
    temperature_C: float,
    cathodic_overpotential_V: float = 0.0,
    params: Optional[PoisonParams] = None,
) -> Dict[str, float]:
    """
    Competitive Langmuir coverage of each poison on the cathode surface.

    Parameters
    ----------
    concentrations_M : dict of poison-name -> dissolved concentration (mol/L).
        Missing poisons are treated as absent (0 M).
    temperature_C : bath temperature.
    cathodic_overpotential_V : magnitude of the (negative) cathodic overpotential
        (pass a positive number). More negative polarisation enhances adsorption.
    params : PoisonParams or defaults.

    Returns
    -------
    dict of poison-name -> surface coverage fraction (0..1). Covers only the
    poisons present; the metal surface fraction left free is 1 - sum(theta).
    """
    p = params or PoisonParams()
    t_k = temperature_C + 273.15

    # Potential factor (saturating power law), applies uniformly to poisons that
    # are cathode-deposited / adsorbed under cathodic polarisation.
    eta = max(float(cathodic_overpotential_V), 0.0)
    pot_factor = min((eta / p.eta_ref_V) ** p.eta_exponent, 1.0)
    if eta <= 0.0:
        pot_factor = 0.0

    # Competitive Langmuir: theta_i = K_i * C_i / (1 + sum_j K_j * C_j)
    kc: Dict[str, float] = {}
    for name in p.poison_names():
        c = max(float(concentrations_M.get(name, 0.0)), 0.0)
        if c <= 0.0:
            kc[name] = 0.0
            continue
        k = adsorption_constant_L_mol(name, temperature_C, p)
        kc[name] = k * c * pot_factor

    denom = 1.0 + sum(kc.values())
    return {name: (val / denom if denom > 0 else 0.0) for name, val in kc.items()}


def absorption_promotion_factor(
    coverages: Dict[str, float],
    params: Optional[PoisonParams] = None,
) -> float:
    """
    Multiplicative boost to absorbed-hydrogen efficiency from poison coverage.

    Each poison at coverage theta contributes (1 + p_i * theta_i); the factors
    multiply and the total is capped at ``max_promotion_factor``. A clean surface
    (all coverages 0) returns exactly 1.0 (no change to the base absorption).
    """
    p = params or PoisonParams()
    factor = 1.0
    for name in p.poison_names():
        theta = min(max(float(coverages.get(name, 0.0)), 0.0), 1.0)
        if theta <= 0.0:
            continue
        factor *= (1.0 + p.p_promote[name] * theta)
    return float(min(factor, p.max_promotion_factor))


def poisoned_absorption_fraction(
    base_absorption_fraction: float,
    concentrations_M: Dict[str, float],
    temperature_C: float,
    cathodic_overpotential_V: float = 0.0,
    params: Optional[PoisonParams] = None,
) -> Dict[str, float]:
    """
    Apply the recombination-poison promotion to a base absorption fraction.

    Parameters
    ----------
    base_absorption_fraction : the value ``hydrogen_embrittlement.py`` currently
        computes (its screening ~0.001..0.20). Pass 0.0 to get promotion only.

    Returns
    -------
    dict with:
      coverages           : poison-name -> surface coverage
      promotion_factor    : multiplicative H-entry boost
      poisoned_absorption_fraction : base * promotion_factor (<= 1.0)
    """
    p = params or PoisonParams()
    coverages = poison_coverage_fraction(
        concentrations_M, temperature_C, cathodic_overpotential_V, p)
    factor = absorption_promotion_factor(coverages, p)
    poisoned = min(float(base_absorption_fraction) * factor, 1.0)
    return {
        "coverages": coverages,
        "promotion_factor": float(factor),
        "poisoned_absorption_fraction": poisoned,
    }


def sulfide_from_ppm(ppm_S_in_feed: float, dilution_factor: float = 1.0) -> float:
    """
    Rough bath sulfide concentration (mol/L) from a feedstock S content.

    ``ppm_S_in_feed`` is the sulfur in the feedstock (e.g. pickle-liquor sulfate,
    steel-mill dust). ``dilution_factor`` scales from feedstock to bath
    concentration. Screening only: the true bath S²⁻/HS⁻ level is set by the
    sulfate-reduction / redox state of the bath, which the closed-loop model
    should ultimately supply.

    Returns mol/L of reduced sulfur assumed present.
    """
    ppm = max(float(ppm_S_in_feed), 0.0)
    # ppm (mass/mass) -> g S per 1e6 g -> mol per ~1e6/rho mL ~ ~1e3 L at rho~1.
    mol_L_feed = ppm / (32.06 * 1000.0)  # ~ mol/L in the feedstock liquor
    return mol_L_feed / max(float(dilution_factor), 1e-9)


def main() -> None:
    """CLI entrypoint for recombination-poison H-entry analysis."""
    print("=" * 70)
    print(" Recombination Poisons -> Absorbed-H Promotion (Round 5, B1)")
    print("=" * 70)
    print(f" Screening flag      : {SCREENING_FLAG}")
    print(f" Poison parameters   : {PoisonParams().poison_names()}")

    # Illustrative feedstock: pickle liquor carrying S + a bit of As.
    feed_sulfide_M = sulfide_from_ppm(300.0)
    feed_arsenic_M = 5e-6
    concentrations = {"sulfide": feed_sulfide_M, "arsenic": feed_arsenic_M}
    print(f"\n Bath [S²⁻/HS⁻] ~ {feed_sulfide_M:.3e} mol/L  [As] ~ {feed_arsenic_M:.1e} mol/L")

    base = 0.05  # hydrogen_embrittlement.py screening value at pH 3.5
    for eta in (0.0, 0.15, 0.30):
        res = poisoned_absorption_fraction(
            base, concentrations, temperature_C=60.0,
            cathodic_overpotential_V=eta)
        cov = res["coverages"]
        print(f"\n |eta| = {eta:5.2f} V")
        print(f"   theta_sulfide = {cov['sulfide']:.3f}   theta_As = {cov['arsenic']:.3f}")
        print(f"   promotion factor = {res['promotion_factor']:.1f}x")
        print(f"   absorption_fraction {base} -> {res['poisoned_absorption_fraction']:.4f}")

    print("\n=> Without poison (clean bath) the promotion factor is 1.0; with a")
    print("   S/As-bearing pickle-liquor feed it can raise absorbed-H by 100-1000x.")


if __name__ == "__main__":
    main()
