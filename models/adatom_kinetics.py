"""
Adatom surface-diffusion / kink-step kinetic incorporation barrier.

Why this module exists
----------------------
The charge-transfer kinetics in ``kinetics.py``/``bdd_kinetics.py`` end at the
surface; ``pulse.py``'s off-time "healing" and ``mullins_sekerka.py``'s smoothing
both implicitly rely on **surface diffusion**, but there is no explicit
adatom/step model. This leaves an unallocated part of the cathodic overpotential
(the "crystallization overpotential") and makes the off-time smoothing in pulse
plating unquantified.

The physics (Round 5, C3): crystal growth is limited by (1) charge transfer,
(2) transport, and (3) **surface diffusion of adatoms to step/kink sites +
kink incorporation**. The third adds a crystallization overpotential

    η_cryst ≈ (RT/F)·ln(1 + j / j_0,surf),   j_0,surf ∝ D_s·c_adatom·ρ_kink

where D_s is the surface diffusivity (temperature- and additive-suppressed)
and ρ_kink is the kink/step density. This is the physical knob additive levelers
turn, and it sets the off-time surface-diffusion length that ``pulse.py`` /
``mullins_sekerka.py`` consume.

Screening flag
--------------
L1. Surface diffusivity pre-factors and kink density are screening; calibrate
against pulse-reverse morphology and EIS-derived crystallization resistance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from models.thermodynamic_constants import R_GAS, FARADAY

SCREENING_FLAG = "unvalidated (L1)"


@dataclass
class AdatomParams:
    """Screening parameters for adatom surface diffusion and incorporation."""

    d_s0_m2_s: float = 1.0e-5       # surface-diffusivity pre-exponential (m²/s)
    e_surf_diff_J_mol: float = 45.0e3
    c_adatom_sat_M: float = 1e-4    # saturation adatom surface concentration
    rho_kink_ref: float = 1.0e8     # kink/step density (1/m), screening
    # Additive suppression of surface diffusion (per unit coverage).
    additive_suppress_factor: float = 5.0
    # Prefactor for the surface exchange current (A/m²).
    j0_surf_prefactor_A_m2: float = 1.0e4
    t_ref_K: float = 298.15


def surface_diffusivity_m2_s(
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[AdatomParams] = None,
) -> float:
    """Temperature- and additive-dependent surface diffusivity (m²/s)."""
    p = params or AdatomParams()
    t_k = temperature_C + 273.15
    d_s = p.d_s0_m2_s * math.exp(-p.e_surf_diff_J_mol / (R_GAS * t_k))
    theta = min(max(float(additive_coverage_fraction), 0.0), 0.999)
    d_s /= (1.0 + p.additive_suppress_factor * theta)
    return float(max(d_s, 1e-25))


def surface_exchange_current_A_m2(
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[AdatomParams] = None,
) -> float:
    """Surface (crystallization) exchange current density (A/m²)."""
    p = params or AdatomParams()
    d_s = surface_diffusivity_m2_s(temperature_C, additive_coverage_fraction, p)
    return float(p.j0_surf_prefactor_A_m2 * d_s / p.d_s0_m2_s)


def crystallization_overpotential_V(
    j_Fe_A_m2: float,
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[AdatomParams] = None,
) -> dict:
    """
    Crystallization (adatom incorporation) overpotential (V).

    Returns
    -------
    dict with overpotential_V, surface_exchange_A_m2, surface_diffusivity_m2_s.
    """
    p = params or AdatomParams()
    t_k = temperature_C + 273.15
    j0 = surface_exchange_current_A_m2(temperature_C, additive_coverage_fraction, p)
    j_fe = max(float(j_Fe_A_m2), 0.0)
    eta = (R_GAS * t_k / (2.0 * FARADAY)) * math.log(1.0 + j_fe / max(j0, 1e-30))
    d_s = surface_diffusivity_m2_s(temperature_C, additive_coverage_fraction, p)
    return {
        "crystallization_overpotential_V": float(max(eta, 0.0)),
        "surface_exchange_A_m2": float(j0),
        "surface_diffusivity_m2_s": d_s,
    }


def off_time_healing_length_m(
    off_time_s: float,
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[AdatomParams] = None,
) -> float:
    """Surface-diffusion length during a pulse off-time (m)."""
    d_s = surface_diffusivity_m2_s(temperature_C, additive_coverage_fraction, params)
    return float(math.sqrt(4.0 * d_s * max(float(off_time_s), 0.0)))


def main() -> None:
    """CLI entrypoint for adatom / crystallization kinetics."""
    print("=" * 70)
    print(" Adatom Surface-Diffusion & Kink Incorporation (Round 5, C3)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    for j in (1000.0, 3000.0, 10000.0):
        res = crystallization_overpotential_V(j, 60.0)
        print(f"  j_Fe={j/10:5.0f} mA/cm² -> eta_cryst={res['crystallization_overpotential_V']*1000:6.2f} mV")
    l = off_time_healing_length_m(0.01, 60.0)
    print(f"\n  Off-time healing length (10 ms) = {l*1e6:.2f} µm")


if __name__ == "__main__":
    main()
