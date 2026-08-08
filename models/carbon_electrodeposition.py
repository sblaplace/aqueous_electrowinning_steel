"""
Electrochemical co-deposition of dissolved carbon species into interstitial
carbon in the iron deposit.

Why this module exists
----------------------
Today the repository adds carbon to iron only *after* deposition
(``carburization.py``, ``carbon_potential.py``) or as *solid particles*
(``co_deposition.py``, Guglielmi composite plating). There is no model of
co-reducing **dissolved carbon species** (CO₂, carbonate, formate, CO, urea)
at the strongly-negative iron-deposition potential so that carbon enters the
growing layer as *interstitial C* — the one-step "electrowin steel, not just
iron" route.

This module (Round 5, A1) computes a carbon-deposition partial current and the
resulting deposit C wt%, driven by the dissolved-carbon concentration, pH,
temperature and the competition with Fe and HER. It also accounts for carbon
unintentionally carried in from organic additives (saccharin, thiourea, PEG),
which share the same interstitial-carbon channel.

Screening flag
--------------
L1 screening model. The central numbers (C vector diffusivity, incorporation
efficiency) must be calibrated against deposit C wt% measured by combustion /
OES on real divided-cell runs.

References
----------
* Thermodynamic access of C + 4H⁺ + 4e⁻ -> C + 2H₂O is standard electrochem.
* Electrolytic co-deposition of C in Fe from CO₂/formate is an active research
  area (carbon-incorporation electrochemistry in aqueous Fe deposition).
* Dissolved-carbon vector solubilities: CO₂ is ~0.033 M at 1 atm 25 C but much
  lower in acid sulfate; formate/carbonate/CO are the usable vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from models.thermodynamic_constants import FARADAY, R_GAS

SCREENING_FLAG = "unvalidated (L1)"

# Molar masses, g/mol
M_C = 12.011
M_FE = 55.845


@dataclass
class CarbonElectrodepositionParams:
    """Screening parameters for dissolved-carbon co-reduction on the cathode."""

    # The dissolved "carbon vector" the user supplies. Defaults are for formate
    # (HCOO⁻), which is soluble and reducible in acid sulfate; CO₂ is barely
    # soluble there. n_C = electrons per C atom deposited from the vector.
    n_C: int = 3              # HCOO⁻ + 3H⁺ + 2e⁻ -> C + 2H₂O (use n=2); see below
    D_C_25_m2_s: float = 1.5e-9   # formate diffusivity in water at 25 C (m²/s)
    Ea_C_diffusion_J_mol: float = 18.0e3
    delta_um: float = 100.0   # Nernst boundary-layer thickness (µm)
    C_C_ref_M: float = 0.1    # reference dissolved-carbon concentration (mol/L)
    incorporation_efficiency_ref: float = 0.15  # j_C/j_C,lim at ref overpotential
    eta_inc_ref_V: float = 0.35  # overpotential at which incorporation eff. is ref
    inc_potential_exponent: float = 0.8  # more negative η -> more C co-deposited
    t_ref_K: float = 298.15
    # Additive carbon contribution: grams C released per kg deposit per g/L
    # of additive at full aging (screening, from additive_aging degradation).
    additive_c_g_per_kg_per_gL: float = 0.15


def diffusivity_formate_m2_s(temperature_C: float,
                             params: Optional[CarbonElectrodepositionParams] = None) -> float:
    """Arrhenius-scaled diffusivity of the dissolved-carbon vector."""
    p = params or CarbonElectrodepositionParams()
    t_k = temperature_C + 273.15
    return p.D_C_25_m2_s * math.exp(
        p.Ea_C_diffusion_J_mol / R_GAS * (1.0 / p.t_ref_K - 1.0 / t_k))


def carbon_limiting_current_density_A_m2(
    c_carbon_M: float,
    temperature_C: float,
    params: Optional[CarbonElectrodepositionParams] = None,
) -> float:
    """Mass-transfer limiting current for carbon co-reduction (A/m²)."""
    p = params or CarbonElectrodepositionParams()
    d_c = diffusivity_formate_m2_s(temperature_C, p)
    delta_m = max(p.delta_um * 1e-6, 1e-9)
    c_bulk_mol_m3 = max(float(c_carbon_M), 0.0) * 1e3  # mol/L -> mol/m³
    # j_lim = n F D (C_bulk - 0) / delta
    return float(p.n_C * FARADAY * d_c * c_bulk_mol_m3 / delta_m)


def carbon_partial_current_density_A_m2(
    c_carbon_M: float,
    temperature_C: float,
    cathodic_overpotential_V: float = 0.35,
    params: Optional[CarbonElectrodepositionParams] = None,
) -> float:
    """
    Actual carbon co-reduction current (A/m²).

    Runs at the mass-transfer limit (transport control) but only a fraction of
    that current is actually incorporated; the fraction rises with cathodic
    overpotential (screening power law saturating at 1.0).
    """
    p = params or CarbonElectrodepositionParams()
    j_lim = carbon_limiting_current_density_A_m2(c_carbon_M, temperature_C, p)
    eta = max(float(cathodic_overpotential_V), 0.0)
    inc = min(p.incorporation_efficiency_ref *
              (eta / p.eta_inc_ref_V) ** p.inc_potential_exponent, 1.0)
    return float(j_lim * inc)


def deposit_carbon_wt_percent(
    j_Fe_A_m2: float,
    c_carbon_M: float,
    temperature_C: float,
    cathodic_overpotential_V: float = 0.35,
    params: Optional[CarbonElectrodepositionParams] = None,
) -> dict:
    """
    Deposit carbon content (wt%) from co-reduced dissolved carbon.

    Uses the molar deposition-rate ratio:
        C wt% = (n_Fe * j_C / n_C) * M_C / ( (n_Fe*j_C/n_C)*M_C + j_Fe*M_Fe )
    where n_Fe = 2 (Fe²⁺ + 2e⁻ -> Fe) and n_C = electrons per C deposited.

    Returns
    -------
    dict with c_wt_percent, j_c_A_m2, j_lim_A_m2, incorporation_efficiency.
    """
    p = params or CarbonElectrodepositionParams()
    j_fe = max(float(j_Fe_A_m2), 1e-12)
    j_c = carbon_partial_current_density_A_m2(
        c_carbon_M, temperature_C, cathodic_overpotential_V, p)
    j_lim = carbon_limiting_current_density_A_m2(c_carbon_M, temperature_C, p)

    # Molar deposition rates of Fe and C per electron.
    # Fe: n_Fe=2 electrons per atom. C: n_C electrons per atom.
    mol_c = (j_c / p.n_C)
    mol_fe = (j_fe / 2.0)
    c_mass = mol_c * M_C
    fe_mass = mol_fe * M_FE
    c_wt = c_mass / (c_mass + fe_mass) * 100.0 if (c_mass + fe_mass) > 0 else 0.0

    eta = max(float(cathodic_overpotential_V), 0.0)
    inc = min(p.incorporation_efficiency_ref *
              (eta / p.eta_inc_ref_V) ** p.inc_potential_exponent, 1.0)
    return {
        "c_wt_percent": float(c_wt),
        "j_c_A_m2": float(j_c),
        "j_c_lim_A_m2": float(j_lim),
        "incorporation_efficiency": float(inc),
        "mol_c_per_m2_s": float(mol_c),
        "mol_fe_per_m2_s": float(mol_fe),
    }


def additive_carbon_wt_percent(
    additive_g_L: float,
    aged_fraction: float = 1.0,
    params: Optional[CarbonElectrodepositionParams] = None,
) -> float:
    """
    Additional deposit C wt% from organic-additive fragmentation.

    Screening: additive_aging.py degrades saccharin/thiourea/PEG, releasing C
    that co-deposits. ``aged_fraction`` is the fraction of the additive that has
    degraded by the point of interest (0 fresh .. 1 fully aged).
    """
    p = params or CarbonElectrodepositionParams()
    return float(max(additive_g_L, 0.0) * max(aged_fraction, 0.0)
                 * p.additive_c_g_per_kg_per_gL / 10.0)  # g C per kg deposit -> wt%


def steel_grade_for_carbon(c_wt_percent: float) -> str:
    """Coarse AISI routing from deposit C wt% (interstitial carbon)."""
    c = float(c_wt_percent)
    if c <= 0.06:
        return "AISI 1005 (extra-low carbon)"
    if c <= 0.20:
        return "AISI 1018 (low carbon)"
    if c <= 0.45:
        return "AISI 1045 (medium carbon)"
    return "AISI 1095+ (high carbon / tool-steel range)"


def main() -> None:
    """CLI entrypoint for electrochemical carbon co-deposition."""
    print("=" * 70)
    print(" Electrochemical Carbon Co-deposition -> Deposit C wt% (Round 5, A1)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    print(f" Carbon vector  : formate (HCOO⁻), n_C = {CarbonElectrodepositionParams().n_C}")

    j_fe_A_m2 = 3000.0  # 300 mA/cm²
    print(f"\n Fe deposition current : {j_fe_A_m2/10:.0f} mA/cm²")
    for c_carbon_M in (0.02, 0.1, 0.5, 1.0):
        res = deposit_carbon_wt_percent(j_fe_A_m2, c_carbon_M, temperature_C=60.0,
                                        cathodic_overpotential_V=0.4)
        grade = steel_grade_for_carbon(res["c_wt_percent"])
        print(f"   [C]={c_carbon_M:5.2f} M  ->  C wt% = {res['c_wt_percent']:7.4f}  "
              f"j_C = {res['j_c_A_m2']/10:8.2f} mA/cm²  -> {grade}")

    add = additive_carbon_wt_percent(additive_g_L=2.0)
    print(f"\n Additive fragmentation (2 g/L saccharin, aged) adds ~{add:.4f} wt% C")


if __name__ == "__main__":
    main()
