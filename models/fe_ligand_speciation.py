"""
Fe(II)-ligand complexation: widen the deposition pH window.

Why this module exists
----------------------
The research report lists chelating ligands (citrate, glycine, gluconate, EDTA)
as the strategy for neutral/alkaline operation, but there is no module for
Fe(II)-ligand solution speciation and its effect on the deposition potential
and Fe(OH)₂ window. This is the chemistry that would let the program operate
outside the acidic HER-dominated regime.

The chemistry (Round 5, E2): a ligand L raises the pH at which Fe²⁺ precipitates
by lowering free a_Fe²⁺ via complexes FeL²⁺, FeL₂, protonated FeHL, etc.:

    Fe²⁺ + nL ⇌ FeL_n     (log β: glycine ~3-4, citrate ~4-5, EDTA ~14)
    pH_ppt(ligand)  >  pH_ppt(no ligand)

But complexation also shifts E_eq(Fe²⁺/Fe) negative (raises cell voltage) and
changes the interfacial pH/HER balance.

This module computes free a_Fe²⁺, the shifted precipitation pH, and the
Nernst shift of the deposition potential for a chosen ligand.

Screening flag
--------------
L1. Stability constants are literature screening values; refine against
measured Fe-ligand speciation / polarography on the actual bath.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from models.thermodynamic_constants import FARADAY, R_GAS, KSP_FEOH2_25

SCREENING_FLAG = "unvalidated (L1)"

# Overall formation constants log beta_n for Fe(L)_n (25 C, screening values).
LOG_BETA: Dict[str, Dict[int, float]] = {
    "glycine": {1: 3.8, 2: 6.9},
    "citrate": {1: 4.4, 2: 7.9},
    "gluconate": {1: 1.4, 2: 2.4},
    "EDTA": {1: 14.3},
    "none": {},
}


@dataclass
class LigandSpeciationParams:
    """Screening parameters for Fe-ligand speciation."""

    log_beta: Dict[str, Dict[int, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in LOG_BETA.items()})
    # Protonation (acid dissociation) of the free ligand, screening pKa.
    ligand_pka: Dict[str, float] = field(default_factory=lambda: {
        "glycine": 9.6, "citrate": 6.4, "gluconate": 3.9, "EDTA": 10.3, "none": 0.0})
    # Number of protons displaced per ligand bound (screening; affects E shift).
    protons_per_ligand: Dict[str, int] = field(default_factory=lambda: {
        "glycine": 1, "citrate": 2, "gluconate": 1, "EDTA": 3, "none": 0})
    t_ref_K: float = 298.15
    # Standard potential shift coefficient (V) per log-beta-weighted ligand
    # bound (screening Nernst-style correction).
    dE_per_logbeta_V: float = 0.0295  # ~ (RT/2F)·ln10 at 25 C


def formation_constant(ligand: str, n: int,
                       params: Optional[LigandSpeciationParams] = None) -> float:
    """Formation constant beta_n for Fe(L)_n (M^-n)."""
    p = params or LigandSpeciationParams()
    lb = p.log_beta.get(ligand, {}).get(n, 0.0)
    return 10.0 ** lb


def free_Fe2_concentration_M(
    total_Fe_M: float,
    ligand: str,
    total_ligand_M: float,
    params: Optional[LigandSpeciationParams] = None,
) -> dict:
    """
    Free aquated Fe²⁺ concentration (M) in the presence of ligand.

    Solves the Fe/L mass balance with a single dominant complex (FeL) for
    simplicity; the full multi-complex form is an obvious extension. Returns
    the free Fe, the bound fraction, and the complex concentration.
    """
    p = params or LigandSpeciationParams()
    fe_t = max(float(total_Fe_M), 0.0)
    l_t = max(float(total_ligand_M), 0.0)
    beta1 = formation_constant(ligand, 1, p)

    if fe_t <= 0.0:
        return {"free_fe2_M": 0.0, "bound_fraction": 0.0, "complex_M": 0.0}
    if beta1 <= 0.0 or l_t <= 0.0:
        return {"free_fe2_M": fe_t, "bound_fraction": 0.0, "complex_M": 0.0}

    # Mass balance: Fe_t = Fe_free + FeL ; L_t = L_free + FeL
    # FeL = Fe_free * beta1 * L_free ; and Fe_free = Fe_t - FeL.
    # Solve for L_free: L_t = L_free + Fe_t*beta1*L_free/(1+beta1*L_free)
    # -> iterative or quadratic. Use a small fixed-point for robustness.
    L_free = l_t
    for _ in range(50):
        denom = 1.0 + beta1 * L_free
        FeL = fe_t * beta1 * L_free / denom
        Fe_free = fe_t - FeL
        new_L = l_t - FeL
        if abs(new_L - L_free) < 1e-12 * max(l_t, 1e-9):
            L_free = new_L
            break
        L_free = 0.5 * (L_free + max(new_L, 0.0))

    denom = 1.0 + beta1 * L_free
    FeL = fe_t * beta1 * L_free / denom
    Fe_free = fe_t - FeL
    return {
        "free_fe2_M": float(max(Fe_free, 0.0)),
        "bound_fraction": float(min(max(FeL / fe_t, 0.0), 1.0)),
        "complex_M": float(max(FeL, 0.0)),
    }


def precipitation_pH(
    total_Fe_M: float,
    ligand: str = "none",
    total_ligand_M: float = 0.0,
    params: Optional[LigandSpeciationParams] = None,
) -> float:
    """pH at which Fe(OH)₂ precipitates, given the free Fe²⁺ activity."""
    p = params or LigandSpeciationParams()
    fe_free = free_Fe2_concentration_M(total_Fe_M, ligand, total_ligand_M, p)["free_fe2_M"]
    fe_free = max(fe_free, 1e-30)
    # Ksp = [Fe²⁺][OH⁻]²  =>  [OH⁻] = sqrt(Ksp/Fe)  =>  pH = 14 + log10[OH⁻]
    oh = math.sqrt(KSP_FEOH2_25 / fe_free)
    pH = 14.0 + math.log10(max(oh, 1e-30))
    return float(pH)


def deposition_potential_shift_V(
    total_Fe_M: float,
    ligand: str = "none",
    total_ligand_M: float = 0.0,
    params: Optional[LigandSpeciationParams] = None,
) -> float:
    """
    Nernst shift of E_eq(Fe²⁺/Fe) from complexation (V, negative = harder to reduce).

    Lower free Fe²⁺ activity shifts the reduction potential negative via
    ΔE = (RT/2F)·ln(a_Fe_free / Fe_total).
    """
    p = params or LigandSpeciationParams()
    fe_free = free_Fe2_concentration_M(total_Fe_M, ligand, total_ligand_M, p)["free_fe2_M"]
    fe_t = max(float(total_Fe_M), 1e-30)
    ratio = max(fe_free, 1e-30) / fe_t
    t_k = p.t_ref_K
    return float((R_GAS * t_k / (2.0 * FARADAY)) * math.log(ratio))


def ligand_window_summary(
    total_Fe_M: float,
    ligand: str,
    total_ligand_M: float,
    params: Optional[LigandSpeciationParams] = None,
) -> dict:
    """One-stop summary of the ligand's effect on the deposition window."""
    spe = free_Fe2_concentration_M(total_Fe_M, ligand, total_ligand_M, params)
    ph_ppt = precipitation_pH(total_Fe_M, ligand, total_ligand_M, params)
    ph_ppt_base = precipitation_pH(total_Fe_M, "none", 0.0, params)
    dE = deposition_potential_shift_V(total_Fe_M, ligand, total_ligand_M, params)
    return {
        "ligand": ligand,
        "free_fe2_M": spe["free_fe2_M"],
        "bound_fraction": spe["bound_fraction"],
        "precipitation_pH": ph_ppt,
        "precipitation_pH_no_ligand": ph_ppt_base,
        "pH_window_widening": ph_ppt - ph_ppt_base,
        "deposition_potential_shift_V": dE,
    }


def main() -> None:
    """CLI entrypoint for Fe-ligand speciation analysis."""
    print("=" * 70)
    print(" Fe(II)-Ligand Speciation: Widening the pH Window (Round 5, E2)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    for ligand in ("none", "glycine", "citrate", "EDTA"):
        res = ligand_window_summary(total_Fe_M=0.5, ligand=ligand, total_ligand_M=0.5)
        print(f"  {ligand:9s}: pH_ppt={res['precipitation_pH']:5.2f} "
              f"(base {res['precipitation_pH_no_ligand']:.2f}, +{res['pH_window_widening']:+.2f}) "
              f"ΔE={res['deposition_potential_shift_V']*1000:+.0f} mV "
              f"bound={res['bound_fraction']*100:.0f}%")


if __name__ == "__main__":
    main()
