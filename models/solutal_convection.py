"""
Solutal buoyancy, mixed convection (Gr_m / Re²), and boundary layer stability on vertical cathodes.

Physics and Hydrodynamics
-------------------------
In vertical-channel iron electrowinning cells (:mod:`models.boundary_layer`,
:mod:`models.transport`, :mod:`models.cell_architecture`), mass transport is not
governed by pure forced convection alone; it is a **coupled mixed-convection**
regime driven by the interplay of forced electrolyte flow and solutal buoyancy.

1. **Solutal Density Depletion**:
   As Fe²⁺ ions (molar mass M = 55.85 g/mol) are reduced and deposited at the
   cathode, the electrolyte inside the Nernst boundary layer becomes severely
   depleted in heavy iron salt compared to the dense bulk electrolyte:
     Δρ_solutal = ρ_bulk - ρ_surf = ρ₀ · β_c · (C_bulk - C_surf)
   where β_c ≈ 0.055 L/mol is the solutal densification coefficient.  The density
   difference reaches **Δρ ≈ 25–70 kg/m³**, creating a powerful **upward buoyant force**.

2. **Solutal Grashof and Richardson Numbers**:
   - Solutal Grashof number:
       Gr_m = (g · β_c · ΔC · H³) / ν²
     For a commercial cathode height H = 0.8–1.2 m, Gr_m reaches **10⁹ to 10¹¹**
     (fully turbulent natural convection).
   - Solutal Richardson number:
       Ri_m = Gr_m / Re²
     where Re = (u₀ · d_h) / ν.

3. **Flow Regimes and Boundary Layer Stability**:
   - **Co-current upward flow (Aiding mixed convection)**:
     Forced velocity and solutal buoyancy act in the same direction, thinning the
     boundary layer and enhancing mass transport:
       Sh_eff = (Sh_forced³ + Sh_natural³)^(1/3)

   - **Counter-current downward flow (Opposing mixed convection)**:
     Downward forced flow opposes the upward solutal plume.  When Ri_m ≈ 1.0, the
     opposing momentum forces balance, causing **boundary layer separation, flow
     reversal, and stagnant recirculation zones**.  This triggers severe local
     iron starvation, high overpotentials, and nodular/dendritic deposits.

4. **Critical Velocity Criterion**:
   To prevent flow reversal and guarantee stable boundary layers in downward-flow cells:
     u_forced > u_crit = √(g · β_c · ΔC · H)

References
----------
* Ibl, N., & Muller, R. H. (1958). "Studies of natural convection at vertical
  electrodes." J. Electrochem. Soc., 105(6), 346–353.
* Wragg, A. A. (1977). "Combined free and forced convection in electrochemical
  reactors." Electrochim. Acta, 22(10), 1145–1152.
* Selman, J. R., & Newman, J. (1971). "Free-convection mass transfer with a
  supporting electrolyte." J. Electrochem. Soc., 118(7), 1070.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional


# Physical and gravitational constants
G_GRAVITY = 9.80665        # m/s²
RHO_WATER = 1000.0         # kg/m³


@dataclass(frozen=True)
class SolutalChannelParams:
    """Dimensions and fluid properties of the vertical cathode channel."""

    cathode_height_m: float = 1.0          # Active electrode height H (m)
    interelectrode_gap_m: float = 0.003   # Gap width d (m)
    electrolyte_density_kg_m3: float = 1200.0 # Bulk 1.5 M FeSO4 density
    kinematic_viscosity_m2_s: float = 1.25e-6 # Electrolyte viscosity (m²/s)
    fe_diffusivity_m2_s: float = 6.5e-10  # Fe2+ diffusivity in bulk
    solutal_expansion_coeff_L_mol: float = 0.055 # β_c (L/mol)


@dataclass
class MixedConvectionResult:
    """Mixed convection regime and boundary layer stability metrics."""

    forced_velocity_m_s: float
    flow_direction: str                 # "upward (aiding)" or "downward (opposing)"
    bulk_fe2_mol_L: float
    surface_fe2_mol_L: float
    density_depletion_kg_m3: float
    grashof_number_Gr_H: float          # Plate-scale Grashof number (H-based, ~10^10 - 10^11)
    grashof_number_Gr_dh: float         # Channel-gap Grashof number (d_h-based, ~10^4 - 10^5)
    reynolds_number_Re: float
    richardson_number_Ri_m: float
    buoyancy_velocity_m_s: float
    critical_antireversal_velocity_m_s: float
    is_flow_reversal_threat: bool       # True if downward flow with Ri_m > 0.5
    effective_sherwood_number: float
    effective_boundary_layer_um: float
    convective_regime: str


def solve_solutal_mixed_convection(
    forced_velocity_m_s: float = 0.15,
    flow_direction: Literal["upward", "downward"] = "upward",
    bulk_fe2_mol_L: float = 1.50,
    surface_fe2_mol_L: float = 0.40,
    params: Optional[SolutalChannelParams] = None,
) -> MixedConvectionResult:
    """
    Solve the coupled solutal buoyancy and mixed convection on a vertical cathode.

    Parameters
    ----------
    forced_velocity_m_s : float
        Superficial electrolyte velocity in channel (m/s).
    flow_direction : str
        "upward" (co-current with buoyancy) or "downward" (opposing).
    bulk_fe2_mol_L : float
        Bulk Fe²⁺ concentration (mol/L).
    surface_fe2_mol_L : float
        Cathode surface Fe²⁺ concentration (mol/L).
    params : SolutalChannelParams, optional
        Electrode and electrolyte transport properties.

    Returns
    -------
    MixedConvectionResult
        Hydrodynamic stability, Grashof/Richardson numbers, and effective boundary layer.
    """
    if params is None:
        params = SolutalChannelParams()

    u0 = max(float(forced_velocity_m_s), 0.001)
    c_bulk = max(float(bulk_fe2_mol_L), 0.0)
    c_surf = max(min(float(surface_fe2_mol_L), c_bulk), 0.0)
    delta_c = c_bulk - c_surf  # mol/L

    h = params.cathode_height_m
    nu = params.kinematic_viscosity_m2_s
    d_fe = params.fe_diffusivity_m2_s
    d_h = 2.0 * params.interelectrode_gap_m  # Hydraulic diameter of parallel plate slot
    sc = nu / d_fe

    # Solutal density reduction: Δρ = ρ₀ * β_c * ΔC
    delta_rho = params.electrolyte_density_kg_m3 * (params.solutal_expansion_coeff_L_mol * delta_c)

    # Plate-scale Solutal Grashof number (height-based, characterizes turbulent vertical buoyant plume):
    # Gr_H = (g * β_c * ΔC * H³) / ν²
    beta_c_m3_mol = params.solutal_expansion_coeff_L_mol * 1e-3
    delta_c_mol_m3 = delta_c * 1e3
    gr_h = (G_GRAVITY * beta_c_m3_mol * delta_c_mol_m3 * (h ** 3)) / max(nu ** 2, 1e-18)

    # Channel-gap Solutal Grashof number (gap-based, characterizes lateral slot confinement):
    # Gr_dh = (g * β_c * ΔC * d_h³) / ν²
    gr_dh = (G_GRAVITY * beta_c_m3_mol * delta_c_mol_m3 * (d_h ** 3)) / max(nu ** 2, 1e-18)

    # Reynolds number in channel slot: Re = u₀ * d_h / ν
    re = (u0 * d_h) / max(nu, 1e-12)

    # Solutal Richardson number in duct: Ri_m = Gr_dh / Re²
    ri_m = gr_dh / max(re ** 2, 1e-6)

    # Characteristic natural convection buoyancy velocity: u_buoy = √(g * β_c * ΔC * H)
    u_buoy = math.sqrt(max(G_GRAVITY * beta_c_m3_mol * delta_c_mol_m3 * h, 1e-9))
    u_crit = math.sqrt(max(G_GRAVITY * beta_c_m3_mol * delta_c_mol_m3 * d_h, 1e-9))  # Slot critical velocity

    # Natural convection Sherwood number (Ibl & Muller correlation for vertical plate).
    # Sh_nat is evaluated on the PLATE height H (it enters via Gr_H, the plate-scale
    # Grashof), so it must be re-referenced to the channel gap before it can be combined
    # with the d_h-based forced Sh and divided into d_h for the boundary layer.
    # Sh_dh = Sh_H * (d_h / H)  (equivalent boundary-layer thickness on either reference).
    ra_h = gr_h * sc
    if ra_h < 1e9:
        sh_nat_H = 0.67 * (ra_h ** 0.25)
    else:
        sh_nat_H = 0.15 * (ra_h ** (1.0 / 3.0))
    sh_nat = sh_nat_H * (d_h / max(h, 1e-12))

    # Forced convection Sherwood number (Graetz/Leveque or turbulent duct):
    # Sh_forced = 1.85 * (Re * Sc * d_h / H)^(1/3)
    gz = max((re * sc * d_h) / max(h, 1e-4), 1.0)
    sh_forced = 1.85 * (gz ** (1.0 / 3.0))

    if flow_direction == "upward":
        # Aiding convection: cubic addition
        sh_eff = ((sh_forced ** 3) + (sh_nat ** 3)) ** (1.0 / 3.0)
        is_reversal = False
        if ri_m > 2.0:
            regime = "Natural convection dominated (strong solutal plume aiding flow)"
        elif ri_m > 0.2:
            regime = "Mixed convection (coupled buoyant-forced enhancement)"
        else:
            regime = "Forced convection dominated"
    else:
        # Downward opposing flow
        is_reversal = (ri_m >= 0.5) or (u0 < u_crit)
        if is_reversal:
            sh_eff = max(sh_nat * 0.50, 5.0)  # Significant transport penalty due to recirculation
            regime = "Flow reversal & stagnation threat (opposing solutal buoyancy)"
        else:
            diff_cubed = max((sh_forced ** 3) - (sh_nat ** 3), 0.0)
            sh_eff = max(diff_cubed ** (1.0 / 3.0), sh_nat * 0.8)
            regime = "Forced flow overcoming opposing solutal buoyancy"

    # Effective Nernst diffusion layer thickness: δ = d_h / Sh
    delta_eff_m = d_h / max(sh_eff, 1.0)
    delta_eff_um = delta_eff_m * 1e6

    return MixedConvectionResult(
        forced_velocity_m_s=u0,
        flow_direction=f"{flow_direction} ({'aiding' if flow_direction == 'upward' else 'opposing'})",
        bulk_fe2_mol_L=c_bulk,
        surface_fe2_mol_L=c_surf,
        density_depletion_kg_m3=delta_rho,
        grashof_number_Gr_H=gr_h,
        grashof_number_Gr_dh=gr_dh,
        reynolds_number_Re=re,
        richardson_number_Ri_m=ri_m,
        buoyancy_velocity_m_s=u_buoy,
        critical_antireversal_velocity_m_s=u_crit,
        is_flow_reversal_threat=is_reversal,
        effective_sherwood_number=sh_eff,
        effective_boundary_layer_um=delta_eff_um,
        convective_regime=regime,
    )


def main() -> None:
    """CLI entrypoint for solutal mixed convection on vertical cathodes."""
    print("=================================================================")
    print(" Solutal Buoyancy & Mixed Convection on Vertical Cathodes")
    print("=================================================================")
    print("Comparison of upward (aiding) vs downward (opposing) forced flow:")
    res_up = solve_solutal_mixed_convection(forced_velocity_m_s=0.15, flow_direction="upward")
    res_down = solve_solutal_mixed_convection(forced_velocity_m_s=0.04, flow_direction="downward")
    print(f"  Upward (0.15 m/s)  : delta = {res_up.effective_boundary_layer_um:.1f} µm | {res_up.convective_regime}")
    print(f"  Downward (0.04 m/s): delta = {res_down.effective_boundary_layer_um:.1f} µm | {res_down.convective_regime}")
    print(f"  Critical antireversal velocity: {res_down.critical_antireversal_velocity_m_s:.3f} m/s")


if __name__ == "__main__":
    main()
