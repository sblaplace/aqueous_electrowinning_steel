"""
Primary iron ore leaching kinetics and reductive leaching mechanics for the Dark Mill.

Physics and Hydrometallurgy
---------------------------
In an autonomous or centralized aqueous electrowinning plant (:mod:`models.dark_mill`,
:mod:`models.feedstock_logistics`), raw iron feedstock originates from primary iron
ores (hematite α-Fe₂O₃, magnetite Fe₃O₄, goethite α-FeOOH) or industrial byproducts
(bauxite residue / red mud, pickle liquor).

1. **Shrinking Core Model (SCM)**:
   For spherical mineral particles of initial radius r₀ reacting in acidic media:
   - **Surface chemical reaction control**:
       g(X) = 1 - (1 - X)^(1/3) = (k_chem · C_acid^n / (ρ_m · r₀)) · t
   - **Product-layer (ash) diffusion control**:
       p(X) = 1 - 3(1 - X)^(2/3) + 2(1 - X) = (6 D_e · C_acid / (ρ_m · r₀²)) · t

2. **The Hematite Dissolution Barrier**:
   Direct sulfuric acid dissolution of hematite (Fe₂O₃ + 3H₂SO₄ → Fe₂(SO₄)₃ + 3H₂O)
   is notoriously sluggish (high activation energy E_a ≈ 75–85 kJ/mol).  At 25 °C,
   hematite leaching takes weeks; at 80–90 °C, several hours.

3. **Reductive Leaching Catalysis**:
   Introducing a chemical reducing agent (scrap iron Fe⁰, sulfur dioxide SO₂, or
   ascorbic acid) accelerates dissolution by **10–100×**:
     Fe₂O₃ + Fe⁰ + 3 H₂SO₄ → 3 FeSO₄ + 3 H₂O
     Fe₂O₃ + SO₂ + H₂SO₄ → 2 FeSO₄ + 2 H₂O
   Reduction of surface Fe³⁺ to Fe²⁺ breaks strong Fe³⁺–O coordination bonds,
   releasing soluble Fe²⁺ directly into the electrowinning feedstock stream without
   producing parasitic Fe³⁺.

References
----------
* Levenspiel, O. (1999). "Chemical Reaction Engineering", 3rd ed., Wiley (Shrinking Core Model).
* Senanayake, G., & Muir, D. M. (1988). "Dissolution of iron oxides in sulfuric
  acid: Role of reducing agents." Hydrometallurgy, 21(2), 197–214.
* Cornell, R. M., & Schwertmann, U. (2003). "The Iron Oxides: Structure,
  Properties, Reactions, Occurrences and Uses." Wiley-VCH.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

# Physical and chemical constants
R_GAS = 8.314462618        # J/(mol·K)
T_REF = 298.15             # K

M_FE = 55.845e-3           # kg/mol
M_FE2O3 = 159.69e-3        # kg/mol (hematite)
M_FE3O4 = 231.53e-3        # kg/mol (magnetite)
M_FEOOH = 88.85e-3         # kg/mol (goethite)
M_H2SO4 = 98.079e-3        # kg/mol

RHO_HEMATITE = 5240.0      # kg/m³
RHO_MAGNETITE = 5175.0     # kg/m³
RHO_GOETHITE = 4280.0      # kg/m³


@dataclass(frozen=True)
class OreSpec:
    """Characteristics of the raw mineral feedstock."""

    mineral_type: Literal["hematite", "magnetite", "goethite"] = "hematite"
    particle_p80_um: float = 75.0      # Particle 80% passing size (µm)
    fe_grade_wt_percent: float = 65.0  # Fe content (wt%)
    gangue_silica_wt_percent: float = 4.5

    @property
    def particle_radius_m(self) -> float:
        """Representative particle radius r0 (m)."""
        return (self.particle_p80_um * 1e-6) / 2.0

    @property
    def molar_density_mol_m3(self) -> float:
        """Molar density of active iron oxide mineral (mol/m³)."""
        if self.mineral_type == "hematite":
            return RHO_HEMATITE / M_FE2O3
        elif self.mineral_type == "magnetite":
            return RHO_MAGNETITE / M_FE3O4
        else:
            return RHO_GOETHITE / M_FEOOH

    @property
    def activation_energy_kJ_mol(self) -> float:
        """Arrhenius activation energy for acid dissolution."""
        if self.mineral_type == "hematite":
            return 80.0
        elif self.mineral_type == "magnetite":
            return 65.0
        else:
            return 72.0


@dataclass
class LeachingResult:
    """Conversion, residence time, and chemical consumption from ore leaching."""

    mineral: str
    temperature_C: float
    acid_concentration_M: float
    reductant_present: bool
    residence_time_hours: float
    fe_recovery_fraction: float        # Extraction yield X (0 to 1)
    dissolved_fe_output_M: float
    fe2_to_fe3_product_ratio: float
    acid_consumed_kg_per_t_fe: float
    controlling_mechanism: str


def simulate_ore_leaching(
    ore: Optional[OreSpec] = None,
    temperature_C: float = 80.0,
    acid_concentration_M: float = 2.0,
    residence_time_hours: float = 4.0,
    use_reductant: bool = True,
) -> LeachingResult:
    """
    Simulate iron extraction from ore using the shrinking core kinetic model.

    Parameters
    ----------
    ore : OreSpec, optional
        Ore particle size and mineralogy.
    temperature_C : float
        Leaching vessel temperature (°C).
    acid_concentration_M : float
        Sulfuric acid concentration (mol/L).
    residence_time_hours : float
        Leach tank residence time (hours).
    use_reductant : bool
        Whether reducing agents (scrap Fe, SO2) are added.

    Returns
    -------
    LeachingResult
        Conversion fraction, dissolved Fe production, and acid consumption.
    """
    if ore is None:
        ore = OreSpec()

    t_k = max(float(temperature_C) + 273.15, 273.15)
    t_sec = max(float(residence_time_hours), 0.001) * 3600.0
    c_acid = max(float(acid_concentration_M), 0.05) * 1e3  # mol/m³
    r0 = ore.particle_radius_m
    rho_m = ore.molar_density_mol_m3

    # Arrhenius pre-exponential factor k0 (m/s) calibrated to experimental
    # hematite dissolution rates in H2SO4 (Senanayake & Muir 1988; Cornell & Schwertmann):
    # Yields k_chem(25 °C) ≈ 1.5e-10 m/s and k_chem(80 °C) ≈ 2.2e-8 m/s at Ea = 80 kJ/mol.
    k0 = 1.5e4  # m/s
    e_a = ore.activation_energy_kJ_mol * 1e3
    k_chem = k0 * math.exp(-e_a / (R_GAS * t_k))

    # Reductive leaching catalysis (Fe⁰, SO2, ascorbic acid) accelerates dissolution
    # by 10–50x by reducing surface Fe³⁺ to labile Fe²⁺
    if use_reductant:
        k_chem *= 25.0

    # Dimensionless time parameter for chemical reaction control:
    # tau_chem = (rho_m * r0) / (k_chem * C_acid)
    tau_chem = (rho_m * r0) / max(k_chem * c_acid, 1e-15)

    # Chemical reaction control conversion: 1 - (1 - X)^(1/3) = t / tau_chem
    ratio_chem = t_sec / max(tau_chem, 1e-12)
    if ratio_chem >= 1.0:
        x_chem = 1.0
    else:
        x_chem = 1.0 - ((1.0 - ratio_chem) ** 3)

    # Product layer pore diffusion control: tau_diff = (rho_m * r0^2) / (6 * D_e * C_acid)
    d_e = 5.0e-11 * (t_k / T_REF) * math.exp(-15e3 / (R_GAS * t_k))
    tau_diff = (rho_m * (r0 ** 2)) / max(6.0 * d_e * c_acid, 1e-15)

    # Mixed kinetic model: time t = tau_chem * g(X) + tau_diff * p(X)
    # Series resistance approximation for overall conversion:
    tau_total = tau_chem + 0.25 * tau_diff
    ratio_total = t_sec / max(tau_total, 1e-12)
    if ratio_total >= 1.0:
        x_final = 0.998
    else:
        x_final = 1.0 - ((1.0 - ratio_total) ** 3)
    x_final = min(max(x_final, 0.0), 0.999)

    # Controlling mechanism
    if tau_chem > tau_diff:
        mechanism = "Surface chemical reaction controlled"
    else:
        mechanism = "Product layer pore diffusion controlled"

    # Dissolved Fe product concentration and Fe²⁺/Fe³⁺ ratio
    # Stoichiometry: Fe2O3 + 3 H2SO4 -> Fe2(SO4)3 + 3 H2O (or + Fe0 -> 3 FeSO4)
    fe_output_M = min(1.8, (c_acid / 1.5) * x_final * 1e-3)
    if use_reductant:
        fe2_ratio = 0.95  # Reductive leaching yields predominantly Fe²⁺
        acid_consumed = 1750.0  # kg H2SO4 / tonne Fe
    else:
        fe2_ratio = 0.05  # Direct acid leach yields predominantly Fe³⁺
        acid_consumed = 2650.0

    return LeachingResult(
        mineral=ore.mineral_type,
        temperature_C=temperature_C,
        acid_concentration_M=acid_concentration_M,
        reductant_present=use_reductant,
        residence_time_hours=residence_time_hours,
        fe_recovery_fraction=x_final,
        dissolved_fe_output_M=fe_output_M,
        fe2_to_fe3_product_ratio=fe2_ratio,
        acid_consumed_kg_per_t_fe=acid_consumed,
        controlling_mechanism=mechanism,
    )


def main() -> None:
    """CLI entrypoint for ore leaching simulation."""
    print("=================================================================")
    print(" Primary Iron Ore Shrinking Core Leaching & Reductive Kinetics")
    print("=================================================================")
    ore = OreSpec(mineral_type="hematite", particle_p80_um=75.0)
    for t_hr in [1.0, 2.0, 4.0, 6.0]:
        res_direct = simulate_ore_leaching(ore, temperature_C=80.0, residence_time_hours=t_hr, use_reductant=False)
        res_red = simulate_ore_leaching(ore, temperature_C=80.0, residence_time_hours=t_hr, use_reductant=True)
        print(f" Residence time = {t_hr:3.1f} hr | Direct Acid Extr = {res_direct.fe_recovery_fraction*100:5.1f}% | Reductive Extr = {res_red.fe_recovery_fraction*100:5.1f}%")


if __name__ == "__main__":
    main()
