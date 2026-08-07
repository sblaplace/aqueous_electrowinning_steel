"""
Hierarchical hydrogen trapping spectrum and transient McNabb–Foster / Oriani bakeout.

Physics and Metallurgy
----------------------
Hydrogen in electrodeposited bcc α-iron is not uniformly distributed in the
interstitial lattice; it resides in a discrete **hierarchy of microstructural traps**
with distinct binding energies E_b:

1. **Reversible (Diffusible) Traps** (E_b ≈ 20–35 kJ/mol):
   - Dislocation cores (E_b ≈ 26 kJ/mol)
   - Low-angle grain boundaries (E_b ≈ 20 kJ/mol)
   - Elastic tensile stress fields (E_b ≈ 22 kJ/mol)
   *These traps exchange hydrogen reversibly with the lattice at room temperature
   and are the sole cause of delayed hydrogen-induced cracking / embrittlement.*

2. **Irreversible (Deep) Traps** (E_b ≈ 55–95 kJ/mol):
   - Cementite interfaces (Fe₃C, E_b ≈ 65 kJ/mol)
   - Oxide and hydroxide inclusions (FeOOH, Fe₃O₄, E_b ≈ 75 kJ/mol)
   - Nanovoids and blister cavities (E_b ≈ 90 kJ/mol)
   *These traps hold hydrogen permanently at room and moderate temperatures (<350 °C),
   acting as benign sinks that reduce diffusible hydrogen content.*

3. **Oriani Local Thermodynamic Equilibrium**:
   Between lattice concentration C_L and trap fractional coverage θ_i:
     θ_i / (1 - θ_i) = (C_L / N_L) · exp(E_b,i / (R T))
   where N_L = 8.46 × 10²⁸ sites/m³ (2 octahedral interstitial sites per bcc unit cell).

4. **McNabb–Foster Transient De-Embrittlement Bakeout**:
   During thermal baking (150–220 °C, ASTM F519 / AMS 2759):
   - Reversible traps thermally release hydrogen into the mobile lattice C_L.
   - Hydrogen diffuses to the foil surfaces and desorbs into ambient air.
   - Irreversible traps retain their hydrogen without contributing to room-temperature
     mobile diffusible hydrogen C_H,diff.

References
----------
* Oriani, R. A. (1970). "The diffusion and trapping of hydrogen in steel."
  Acta Metall., 18(1), 147–157.
* McNabb, A., & Foster, P. K. (1963). "A new analysis of the diffusion of
  hydrogen in iron and ferritic steels." Trans. Metall. Soc. AIME, 227, 618–627.
* Pressouyre, G. M. (1979). "A classification of hydrogen traps in steels."
  Metall. Trans. A, 10(10), 1571–1573.
* ASTM F519 / AMS 2759/9 — Standard Test Method for Mechanical Hydrogen
  Embrittlement Evaluation and Post-Plating Bakeout Schedules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Physical constants
FARADAY = 96485.33212      # C/mol
R_GAS = 8.314462618        # J/(mol·K)
T_REF = 298.15             # K
RHO_FE = 7874.0            # kg/m³
M_FE = 55.845e-3           # kg/mol
M_H = 1.008e-3             # kg/mol

# Lattice parameters of bcc α-iron
A0_BCC_M = 0.2866e-9       # Lattice parameter (m)
# Interstitial site density: 2 atoms/unit cell, 6 tetrahedral sites/cell => N_L = 6 / a^3
N_L_M3 = 6.0 / (A0_BCC_M ** 3)  # ≈ 2.54e29 sites/m³ (or 8.46e28 octahedral)
N_L_OCT_M3 = 8.46e28       # Standard octahedral site density (sites/m³)

# Lattice diffusivity of H in pure bcc iron (Kiuchi & McLellan 1983)
D0_LATTICE_M2_S = 7.3e-8   # m²/s
Q_LATTICE_KJ_MOL = 4.6     # Activation energy (kJ/mol)


@dataclass(frozen=True)
class TrapSiteCategory:
    """Individual hydrogen trap category with characteristic binding energy."""

    name: str
    binding_energy_kJ_mol: float
    site_density_m3: float
    is_reversible_at_room_temp: bool


def default_trap_hierarchy(
    dislocation_density_m2: float = 1e15,
    grain_size_um: float = 2.0,
    carbide_volume_fraction: float = 0.01,
) -> List[TrapSiteCategory]:
    """Generate default trap spectrum based on electrodeposit microstructure."""
    # Dislocation trap density: ~1 site per Burgers vector along dislocation line
    b_m = 0.25e-9
    n_disl = dislocation_density_m2 / b_m  # sites/m³

    # Grain boundary trap density: area per volume ≈ 3 / (d_grain)
    d_m = grain_size_um * 1e-6
    n_gb = (3.0 / max(d_m, 1e-8)) * (1.0 / (b_m ** 2))

    # Carbide interface traps (if carbon/carbides present)
    n_carbide = carbide_volume_fraction * 1e27

    return [
        TrapSiteCategory("dislocation_cores", 26.0, n_disl, True),
        TrapSiteCategory("grain_boundaries", 20.0, n_gb, True),
        TrapSiteCategory("elastic_stress_fields", 22.0, 1e25, True),
        TrapSiteCategory("cementite_interfaces", 65.0, n_carbide, False),
        TrapSiteCategory("oxide_inclusions", 75.0, 5e23, False),
        TrapSiteCategory("nanovoids", 90.0, 1e23, False),
    ]


def lattice_diffusivity_m2_s(temperature_C: float) -> float:
    """Compute intrinsic lattice diffusivity of H in bcc iron (m²/s)."""
    t_k = max(float(temperature_C) + 273.15, 200.0)
    return D0_LATTICE_M2_S * math.exp(-(Q_LATTICE_KJ_MOL * 1e3) / (R_GAS * t_k))


def effective_trapped_diffusivity_m2_s(
    temperature_C: float,
    traps: Optional[List[TrapSiteCategory]] = None,
    c_lattice_mol_m3: float = 0.1,
) -> float:
    """
    Compute apparent effective diffusivity D_eff(T) modified by trap occupancy.

    Oriani formula: D_eff = D_L / (1 + sum( (N_i / N_L) * exp(E_b,i/RT) / (1 + (C_L/N_L)exp(E_b/RT))^2 ))
    """
    if traps is None:
        traps = default_trap_hierarchy()

    t_k = max(float(temperature_C) + 273.15, 200.0)
    d_l = lattice_diffusivity_m2_s(temperature_C)

    rt = R_GAS * t_k
    sum_trap_terms = 0.0

    for trap in traps:
        exp_factor = math.exp(min((trap.binding_energy_kJ_mol * 1e3) / rt, 100.0))
        # Apparent trapping factor
        theta_denom = 1.0 + (c_lattice_mol_m3 / N_L_OCT_M3) * exp_factor
        trap_term = (trap.site_density_m3 / N_L_OCT_M3) * exp_factor / (theta_denom ** 2)
        sum_trap_terms += trap_term

    return d_l / (1.0 + sum_trap_terms)


@dataclass
class BakeoutScheduleResult:
    """Optimized thermal de-embrittlement bakeout schedule."""

    foil_thickness_um: float
    bake_temperature_C: float
    initial_total_H_ppm_wt: float
    initial_diffusible_H_ppm_wt: float
    final_diffusible_H_ppm_wt: float
    irreversible_trapped_H_ppm_wt: float
    required_bake_time_hours: float
    d_eff_at_bake_temp_m2_s: float
    is_embrittlement_safe: bool        # True if final diffusible H <= 0.10 ppm wt
    astm_f519_compliance: str


def compute_bakeout_schedule(
    foil_thickness_um: float = 100.0,
    total_initial_H_ppm_wt: float = 5.0,
    bake_temperature_C: float = 190.0,
    target_diffusible_H_ppm_wt: float = 0.10,
    traps: Optional[List[TrapSiteCategory]] = None,
) -> BakeoutScheduleResult:
    """
    Compute the required bakeout duration to desorb diffusible hydrogen below target.

    Uses Fickian slab desorption with trap-modified effective diffusivity at bake temperature.
    """
    if traps is None:
        traps = default_trap_hierarchy()

    l_half_m = (foil_thickness_um * 1e-6) / 2.0  # Diffusion distance to closest foil surface

    # Partition initial hydrogen between reversible (diffusible) and deep irreversible traps
    # Screening partition: ~80% reversible in fine electrodeposited Fe, ~20% irreversible
    c_diff_init = total_initial_H_ppm_wt * 0.80
    c_irrev = total_initial_H_ppm_wt * 0.20

    # Effective diffusivity at bake temperature
    d_eff_bake = effective_trapped_diffusivity_m2_s(bake_temperature_C, traps)

    # Slab desorption series: C_avg(t) / C_0 ≈ (8/pi^2) * exp(-pi^2 * D * t / (4 * L_half^2))
    ratio = max(target_diffusible_H_ppm_wt / max(c_diff_init, 1e-6), 1e-6)
    if ratio >= 1.0:
        t_sec = 0.0
    else:
        # ln(ratio * pi^2 / 8) = -pi^2 * D * t / (4 * L_half^2)
        argument = (ratio * (math.pi ** 2)) / 8.0
        t_sec = (-math.log(max(argument, 1e-9)) * 4.0 * (l_half_m ** 2)) / ((math.pi ** 2) * max(d_eff_bake, 1e-16))

    t_hours = t_sec / 3600.0
    # Minimum ASTM industrial bakeout duration is 1.0 hour
    t_recommended = max(t_hours, 1.0)

    is_safe = target_diffusible_H_ppm_wt <= 0.10

    if t_recommended <= 4.0:
        compliance = "Standard ASTM F519 bakeout (1-4 hr)"
    elif t_recommended <= 24.0:
        compliance = "Extended heavy-section bakeout (4-24 hr)"
    else:
        compliance = "Severe entrapment; higher bake temperature recommended"

    return BakeoutScheduleResult(
        foil_thickness_um=foil_thickness_um,
        bake_temperature_C=bake_temperature_C,
        initial_total_H_ppm_wt=total_initial_H_ppm_wt,
        initial_diffusible_H_ppm_wt=c_diff_init,
        final_diffusible_H_ppm_wt=target_diffusible_H_ppm_wt,
        irreversible_trapped_H_ppm_wt=c_irrev,
        required_bake_time_hours=t_recommended,
        d_eff_at_bake_temp_m2_s=d_eff_bake,
        is_embrittlement_safe=is_safe,
        astm_f519_compliance=compliance,
    )
