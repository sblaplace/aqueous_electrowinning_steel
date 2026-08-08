"""
Crystallite coalescence-induced intrinsic tensile stress and Griffith cracking mechanics.

Physics and Chemistry
---------------------
The mechanical integrity and harvestability of electrodeposited iron foils are heavily
dictated by intrinsic residual stress. This module models the physical birth of tensile
stress during the initial impingement of 3D Volmer–Weber crystallite islands:

1. **Chaudhari–Windischmann Coalescence Stress Model**:
   As discrete iron nuclei grow, they touch and form grain boundaries. To reduce their
   high surface energy, the crystallites deform elastically to close the remaining
   gap (separation distance δ ≈ 0.1–0.2 nm), generating a massive tensile stress at the boundary:
     σ_coal = [E / (1 - ν)] * (δ / L) * (Δγ / γ_s)
   where:
     - E is Young's Modulus of the deposit (~200 GPa for iron, temperature-dependent)
     - ν is Poisson's ratio (~0.29 for bcc iron)
     - δ is the inter-crystallite gap constraint (~0.15 nm)
     - L is the lateral grain size (m)
     - Δγ / γ_s is the normalized surface energy reduction (typically ~1.0)

2. **Hall-Petch Yield Strength Coupling**:
   The same grain size L that governs coalescence stress determines the yield strength
   of the deposit via the Hall-Petch relationship:
     σ_y = σ_0 + k_HP * L^(-1/2)
   For iron, typical screening values are σ_0 ≈ 70 MPa and k_HP ≈ 600 MPa·nm^(1/2).
   If σ_coal exceeds σ_y, the film undergoes localized plastic deformation or micro-cracking
   at the grain boundaries.

3. **Griffith Critical Cracking and Peeling Thickness**:
   The elastic strain energy stored per unit area in a film of thickness h under stress σ is:
     U = [σ² * h * (1 - ν)] / E
   According to Griffith fracture mechanics, the film will spontaneously crack or peel from
   the substrate if the stored energy exceeds the interfacial fracture energy G_c (work of
   adhesion, typically 5–20 J/m²):
     h_crit_crack = [G_c * E] / [σ² * (1 - ν)]

References
----------
* Windischmann, H. (1992). "Intrinsic Stress in Sputtered Thin Films." Critical Reviews in
  Solid State and Materials Sciences, 17(6), 547–596.
* Chaudhari, P. (1972). "Hillock Growth in Thin Films." J. Appl. Phys., 43(11), 4306.
* Griffith, A. A. (1921). "The Phenomena of Rupture and Flow in Solids." Phil. Trans. R.
  Soc. Lond. A, 221, 163–198.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CoalescenceStressParams:
    """Material and interfacial parameters for intrinsic stress and fracture calculations."""

    youngs_modulus_0_GPa: float = 200.0   # Young's Modulus of bcc iron at 0 °C (GPa)
    youngs_temp_coeff_GPa_K: float = 0.05 # Thermal reduction coefficient of E (GPa/K)
    poisson_ratio: float = 0.29           # Poisson's ratio of bcc iron
    gap_delta_nm: float = 0.15            # Inter-crystallite snap gap (nm)
    energy_ratio: float = 1.0             # Surface energy reduction ratio (Δγ / γ_s)
    sigma_0_MPa: float = 70.0             # Lattice friction stress (MPa)
    k_hp_MPa_nm12: float = 600.0          # Hall-Petch stress intensity (MPa·nm^(1/2))
    g_c_J_m2: float = 12.0                # Interfacial fracture energy / work of adhesion (J/m²)


def get_temperature_dependent_youngs_modulus_Pa(
    temperature_C: float,
    params: Optional[CoalescenceStressParams] = None,
) -> float:
    """Calculate the temperature-dependent Young's Modulus of iron (Pa)."""
    if params is None:
        params = CoalescenceStressParams()

    t_k = temperature_C + 273.15
    e_gpa = params.youngs_modulus_0_GPa - params.youngs_temp_coeff_GPa_K * (t_k - 273.15)
    return max(e_gpa, 50.0) * 1e9  # Convert GPa to Pa


@dataclass
class CoalescenceStressResult:
    """Solved intrinsic stress, yield strength, and fracture limits."""

    grain_size_um: float
    youngs_modulus_GPa: float
    coalescence_stress_MPa: float
    hall_petch_yield_strength_MPa: float
    is_plastically_deformed: bool       # True if σ_coal > σ_y
    stored_strain_energy_per_um_J_m2: float  # Stored energy per micrometer of film thickness
    critical_crack_thickness_um: float  # Safe film thickness limit (h_crit) before cracking/peeling


def analyze_coalescence_stress(
    grain_size_um: float,
    temperature_C: float,
    params: Optional[CoalescenceStressParams] = None,
) -> CoalescenceStressResult:
    """
    Evaluate the coalescence-induced intrinsic tensile stress and fracture limits.

    Parameters
    ----------
    grain_size_um : float
        Lateral grain size of the deposit (µm).
    temperature_C : float
        Electrolyte / deposit temperature (°C).
    params : CoalescenceStressParams, optional
        Material parameters.

    Returns
    -------
    CoalescenceStressResult
        Mechanical stress and fracture metrics.
    """
    if params is None:
        params = CoalescenceStressParams()

    g_size_m = max(float(grain_size_um), 1e-3) * 1e-6  # µm -> m
    e_mod = get_temperature_dependent_youngs_modulus_Pa(temperature_C, params)

    # 1. Coalescence Stress (Chaudhari model)
    # σ_coal = [E / (1 - ν)] * (δ / L) * (Δγ / γ_s)
    delta_m = params.gap_delta_nm * 1e-9
    sigma_coal_Pa = (e_mod / (1.0 - params.poisson_ratio)) * (delta_m / g_size_m) * params.energy_ratio
    sigma_coal_MPa = sigma_coal_Pa * 1e-6

    # 2. Hall-Petch Yield Strength
    # L in nm for Hall-Petch parameter scaling
    g_size_nm = max(float(grain_size_um), 1e-3) * 1e3
    sigma_y_MPa = params.sigma_0_MPa + params.k_hp_MPa_nm12 / math.sqrt(g_size_nm)

    is_plastic = sigma_coal_MPa > sigma_y_MPa

    # 3. Elastic Strain Energy and Griffith crack limit
    # h_crit = [G_c * E] / [σ² * (1 - ν)]
    numerator = params.g_c_J_m2 * e_mod
    denominator = (sigma_coal_Pa ** 2) * (1.0 - params.poisson_ratio)
    
    if denominator > 1e-6:
        h_crit_m = numerator / denominator
        # Energy stored per µm (1e-6 m) of film: U = σ² * h * (1 - ν) / E
        energy_per_um = ((sigma_coal_Pa ** 2) * 1e-6 * (1.0 - params.poisson_ratio)) / e_mod
    else:
        h_crit_m = float("inf")
        energy_per_um = 0.0

    return CoalescenceStressResult(
        grain_size_um=grain_size_um,
        youngs_modulus_GPa=e_mod * 1e-9,
        coalescence_stress_MPa=sigma_coal_MPa,
        hall_petch_yield_strength_MPa=sigma_y_MPa,
        is_plastically_deformed=is_plastic,
        stored_strain_energy_per_um_J_m2=energy_per_um,
        critical_crack_thickness_um=h_crit_m * 1e6 if h_crit_m != float("inf") else float("inf"),
    )


def main() -> None:
    """CLI entrypoint for coalescence stress and fracture analysis."""
    print("=================================================================")
    print(" Coalescence Stress, Hall-Petch, and Griffith Cracking Solver")
    print("=================================================================")
    params = CoalescenceStressParams()
    print(f"Interfacial Fracture Energy G_c: {params.g_c_J_m2:.1f} J/m²")
    print(f"Poisson's Ratio                : {params.poisson_ratio:.2f}")
    print(f"Snap Gap Delta                 : {params.gap_delta_nm:.2f} nm\n")

    print("Grain Size vs Mechanical Properties (T = 60°C):")
    for d in [0.05, 0.1, 0.5, 1.0, 5.0]:
        res = analyze_coalescence_stress(d, 60.0, params)
        print(f"  Grain size L = {d:5.2f} µm:")
        print(f"    Coalescence Stress: {res.coalescence_stress_MPa:6.1f} MPa | Yield Strength: {res.hall_petch_yield_strength_MPa:6.1f} MPa | Plastic? {res.is_plastically_deformed}")
        print(f"    Griffith Crack Limit (h_crit): {res.critical_crack_thickness_um:6.1f} µm\n")


if __name__ == "__main__":
    main()
