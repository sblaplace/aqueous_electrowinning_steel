"""
Marangoni & electrocapillary surface-tension-driven flows in the cell.

Why this module exists
----------------------
``solutal_convection.py`` (V3) and ``mhd_convection.py`` capture buoyancy and
Lorentz stirring. Missing are **surface-tension-driven flows**: thermocapillary
(temperature gradient along the interface), solutocapillary (surfactant/additive
concentration gradient), and **electrocapillary** (potential-dependent surface
tension via the Lippmann equation). These stir the boundary layer, thin the
diffusion-layer thickness δ, and alter the local current distribution —
especially in the additive-laden, non-isothermal industrial cell the reviews'
temperature-gradient items set up.

The physics (Round 5, D1): the metal/electrolyte surface tension depends on
potential (electrocapillary maximum near the pzc), temperature, and additive
coverage. A gradient in any of these drives a Marangoni shear stress at the
interface:  τ_Marangoni = ∇γ(φ_M, T, Γ_org, Γ_cl).

This module computes the surface-tension gradient and a screening Marangoni
velocity that can be combined with buoyancy to estimate an effective δ_eff.

Screening flag
--------------
L1. Surface-tension coefficients and the Marangoni-velocity closure are
screening; calibrate against flow/limiting-current measurements on the
reference cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SCREENING_FLAG = "unvalidated (L1)"

# Electroneutrality-free, hydrodynamic reference values.
RHO_ELECTROLYTE_KG_M3 = 1100.0
NU_VISCOSITY_M2_S = 1.2e-6


@dataclass
class MarangoniParams:
    """Screening parameters for surface-tension-gradient flows."""

    # Surface tension temperature coefficient (N/m per K); negative = tension
    # falls as temperature rises.
    d_gamma_dT_N_m_K: float = -1.5e-4
    # Solutocapillary: tension change per unit additive coverage (N/m).
    d_gamma_dGamma_N_m: float = -4.0e-2
    # Electrocapillary (Lippmann): tension change per volt (N/m per V),
    # around the pzc. Negative slope on one side of pzc.
    d_gamma_dE_N_m_V: float = -0.10
    # Charge density term for the Lippmann equation (C/m²), screening.
    q_m_C_m2: float = 0.20
    # Reference boundary-layer thickness (µm) before Marangoni stirring.
    delta_ref_um: float = 100.0
    # Marangoni-velocity to delta reduction: d_eff = delta / (1 + k*v_M/v_ref)^m.
    marangoni_mixing_coeff: float = 2.0
    v_ref_m_s: float = 1e-4


def surface_tension_N_m(
    potential_V: float = 0.0,
    temperature_C: float = 60.0,
    additive_coverage_fraction: float = 0.0,
    params: Optional[MarangoniParams] = None,
) -> float:
    """Surface tension of the metal/electrolyte interface (N/m)."""
    p = params or MarangoniParams()
    t_ref = 60.0
    return float(
        0.35                                          # base Fe/electrolyte tension (N/m)
        + p.d_gamma_dT_N_m_K * (temperature_C - t_ref)
        + p.d_gamma_dGamma_N_m * max(additive_coverage_fraction, 0.0)
        + p.d_gamma_dE_N_m_V * potential_V
        + 0.5 * p.q_m_C_m2 * potential_V ** 2          # Lippmann curvature
    )


def surface_tension_gradient_N_m2(
    temperature_gradient_K_m: float = 0.0,
    additive_gradient_1_m: float = 0.0,
    potential_gradient_V_m: float = 0.0,
    params: Optional[MarangoniParams] = None,
) -> float:
    """Gradient of surface tension along the interface (N/m²)."""
    p = params or MarangoniParams()
    return float(
        p.d_gamma_dT_N_m_K * temperature_gradient_K_m
        + p.d_gamma_dGamma_N_m * additive_gradient_1_m
        + p.d_gamma_dE_N_m_V * potential_gradient_V_m
    )


def marangoni_velocity_m_s(
    temperature_gradient_K_m: float = 0.0,
    additive_gradient_1_m: float = 0.0,
    potential_gradient_V_m: float = 0.0,
    params: Optional[MarangoniParams] = None,
) -> float:
    """Screening Marangoni surface velocity (m/s)."""
    p = params or MarangoniParams()
    grad = surface_tension_gradient_N_m2(
        temperature_gradient_K_m, additive_gradient_1_m, potential_gradient_V_m, p)
    # Boundary-layer scaling: v_M ~ (d_gamma/dx) * delta / (mu) ; screening.
    delta = p.delta_ref_um * 1e-6
    mu = RHO_ELECTROLYTE_KG_M3 * NU_VISCOSITY_M2_S
    return max(abs(grad) * delta / max(mu, 1e-9), 0.0)


def effective_diffusion_layer_um(
    temperature_gradient_K_m: float = 0.0,
    additive_gradient_1_m: float = 0.0,
    potential_gradient_V_m: float = 0.0,
    forced_flow_velocity_m_s: float = 0.0,
    params: Optional[MarangoniParams] = None,
) -> dict:
    """
    Boundary-layer thickness after Marangoni stirring (µm).

    Returns
    -------
    dict with marangoni_velocity_m_s, mixing_ratio, delta_effective_um, and
      vs_buoyancy ratio placeholder.
    """
    p = params or MarangoniParams()
    v_m = marangoni_velocity_m_s(
        temperature_gradient_K_m, additive_gradient_1_m, potential_gradient_V_m, p)
    v_total = max(forced_flow_velocity_m_s + v_m, 1e-12)
    mixing = p.marangoni_mixing_coeff * (v_total / max(p.v_ref_m_s, 1e-12))
    d_eff = p.delta_ref_um / (1.0 + mixing) ** 0.5
    return {
        "marangoni_velocity_m_s": v_m,
        "mixing_ratio": mixing,
        "delta_effective_um": d_eff,
        "delta_reduction_factor": d_eff / max(p.delta_ref_um, 1e-12),
    }


def main() -> None:
    """CLI entrypoint for Marangoni surface-flow analysis."""
    print("=" * 70)
    print(" Marangoni & Electrocapillary Surface Flows (Round 5, D1)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    for dT in (0.0, 2.0, 10.0):  # K/m vertical gradient
        res = effective_diffusion_layer_um(temperature_gradient_K_m=dT)
        print(f"  dT/dx={dT:5.1f} K/m -> v_M={res['marangoni_velocity_m_s']:.2e} m/s "
              f"d_eff={res['delta_effective_um']:6.1f} µm "
              f"(x{res['delta_reduction_factor']:.3f})")


if __name__ == "__main__":
    main()
