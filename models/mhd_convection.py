"""Magnetohydrodynamic (MHD) convection and Lorentz-driven mass transport.

Why this module exists
----------------------
At crate or tankhouse scale (or even RC-1 with steel reinforcement or
external magnets), the interaction of current density **j** with a
magnetic field **B** produces a Lorentz body force **j × B** that
drives additional convective rolls in the catholyte channel.  This
augments the existing gas-holdup self-stirring (`gas_holdup.py`) and
solutal convection (`solutal_convection.py`), thinning the diffusion
layer and boosting Faradaic efficiency at high current density.

This is a classic industrial electrowinning effect (zinc, copper) that
is completely absent from the current model suite.  The module provides:

* A screening-level Lorentz velocity scale.
* An effective boundary-layer thinning factor.
* A simple 1-D channel MHD flow model.
* Coupling hooks into `gas_holdup.py`, `diffusion_layer_1d.py`, and
  `cell_architecture.py` for scale-up studies.
* A measurement protocol that replaces the screening constants.

Scope: screening Level-1 model.  No MHD data exists in the repository.
All numbers are transferred from water-electrolysis / chlor-alkali
literature or simple dimensional analysis.  The dominant uncertainty
is the effective magnetic field strength inside a real cell (Earth
field + steel-induced field + any applied magnets).

Sign convention: positive current density is cathodic (Fe deposition).
B is the component perpendicular to j (usually horizontal across the
channel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict



# ─── Physical constants (screening values) ───────────────────────────
MU0 = 4.0 * math.pi * 1e-7          # H/m, vacuum permeability
RHO_LIQUID = 1200.0                 # kg/m³ (sulfate bath ~60 °C)
MU_LIQUID = 7.0e-4                  # Pa·s
SIGMA_LIQUID = 13.5                 # S/m (typical conductivity)

# Screening Earth + stray field inside a cell (Tesla).
# Earth field ~30–60 µT; steel reinforcement and busbars can raise
# the local |B| to 0.1–0.5 mT.  Applied permanent magnets for flow
# control can reach 10–50 mT.
B_EARTH_TYPICAL_T = 4.0e-5
B_STRAY_TYPICAL_T = 2.0e-4
B_APPLIED_TYPICAL_T = 0.0           # set >0 for deliberate MHD stirring

# Lorentz force scaling constant (screening).
# u_L ~ (B * j * h²) / (ρ * ν)   (dimensional analysis for channel)
# where h is the characteristic length (channel depth or electrode gap).
LORENTZ_VELOCITY_FACTOR = 1.0       # calibration knob (0.5–2.0 typical)


@dataclass
class MHDGeometry:
    """Channel geometry for MHD calculations (re-uses ChannelGeometry spirit)."""
    height_m: float = 0.050
    width_m: float = 0.020
    depth_m: float = 0.003          # characteristic length for Lorentz force
    electrode_gap_m: float = 0.020

    @property
    def characteristic_length_m(self) -> float:
        return self.depth_m


@dataclass
class MHDSolution:
    """Result of an MHD convection calculation."""
    b_field_T: float
    j_mean_A_m2: float
    lorentz_velocity_m_s: float
    effective_delta_reduction_factor: float
    additional_mass_transfer_coeff_m_s: float
    notes: str = ""


def lorentz_velocity_scale(
    j_A_m2: float,
    b_T: float,
    char_length_m: float,
    rho: float = RHO_LIQUID,
    mu: float = MU_LIQUID,
    factor: float = LORENTZ_VELOCITY_FACTOR,
) -> float:
    """Screening Lorentz-driven velocity scale (m/s).

    u_L ≈ factor * (B * j * L²) / (ρ * ν)
    where ν = μ/ρ.
    """
    if j_A_m2 <= 0 or b_T <= 0 or char_length_m <= 0:
        return 0.0
    nu = mu / rho
    u = factor * (b_T * j_A_m2 * char_length_m**2) / (rho * nu)
    return float(max(u, 0.0))


def mhd_boundary_layer_reduction(
    u_lorentz_m_s: float,
    delta_forced_m: float,
    diffusivity_m2_s: float = 7.2e-10,
) -> float:
    """Approximate thinning factor on the diffusion layer due to MHD flow.

    Simple engineering closure: effective k_mhd ≈ u_L * (D / δ_forced)^{0.5}
    → δ_eff = D / (k_forced + k_mhd)
    Returns the ratio δ_eff / δ_forced (always ≤ 1).
    """
    if u_lorentz_m_s <= 0 or delta_forced_m <= 0:
        return 1.0
    k_forced = diffusivity_m2_s / delta_forced_m
    # Rough scaling for additional mass transfer from Lorentz flow
    k_mhd = u_lorentz_m_s * math.sqrt(diffusivity_m2_s / delta_forced_m)
    k_total = math.hypot(k_forced, k_mhd)
    return float(diffusivity_m2_s / (k_total * delta_forced_m))


def compute_mhd_solution(
    j_mean_mA_cm2: float = 100.0,
    b_field_T: float = B_STRAY_TYPICAL_T,
    geometry: MHDGeometry = MHDGeometry(),
    temperature_C: float = 60.0,
    delta_forced_m: float = 50.0e-6,
    diffusivity_m2_s: float = 7.2e-10,
    use_applied_magnet: bool = False,
) -> MHDSolution:
    """Convenience wrapper that returns a complete MHD result."""
    j_A_m2 = float(j_mean_mA_cm2) * 10.0
    b = float(b_field_T)
    if use_applied_magnet:
        b = max(b, B_APPLIED_TYPICAL_T)

    u_L = lorentz_velocity_scale(
        j_A_m2, b, geometry.characteristic_length_m
    )

    reduction = mhd_boundary_layer_reduction(
        u_L, delta_forced_m, diffusivity_m2_s
    )

    # Additional mass-transfer coefficient from MHD flow
    k_mhd = u_L * math.sqrt(diffusivity_m2_s / delta_forced_m) if u_L > 0 else 0.0

    notes = (
        f"MHD screening (B={b*1000:.2f} mT). "
        f"Earth+stray field assumed unless applied magnet requested."
    )

    return MHDSolution(
        b_field_T=b,
        j_mean_A_m2=j_A_m2,
        lorentz_velocity_m_s=u_L,
        effective_delta_reduction_factor=reduction,
        additional_mass_transfer_coeff_m_s=k_mhd,
        notes=notes,
    )


def mhd_enhanced_delta(
    delta_forced_m: float,
    mhd_solution: MHDSolution,
) -> float:
    """Return the MHD-thinned diffusion layer thickness."""
    return float(delta_forced_m * mhd_solution.effective_delta_reduction_factor)


# ─── Integration helper for gas_holdup.py and diffusion_layer_1d.py ───
def effective_mass_transfer_with_mhd(
    delta_forced_m: float,
    j_mA_cm2: float,
    b_field_T: float = B_STRAY_TYPICAL_T,
    geometry: MHDGeometry = MHDGeometry(),
    diffusivity_m2_s: float = 7.2e-10,
) -> float:
    """Drop-in replacement for forced-only δ in transport models.

    Returns an effective diffusion-layer thickness that includes both
    forced convection and MHD stirring.
    """
    sol = compute_mhd_solution(
        j_mean_mA_cm2=j_mA_cm2,
        b_field_T=b_field_T,
        geometry=geometry,
        delta_forced_m=delta_forced_m,
        diffusivity_m2_s=diffusivity_m2_s,
    )
    return mhd_enhanced_delta(delta_forced_m, sol)


# ─── Measurement protocol (replaces screening assumptions) ────────────
def measurement_protocol() -> Dict[str, Any]:
    """Cheap experiment to replace the MHD screening constants."""
    return {
        "title": "MHD convection measurement on RC-1 or crate-scale cell",
        "objective": (
            "Quantify the additional mass-transport benefit (or penalty) "
            "from Lorentz forces in the actual iron bath and cell geometry. "
            "Replace B_stay, Lorentz factor, and δ-reduction scaling."
        ),
        "estimated_cost_usd": 1200,
        "estimated_duration_days": 4,
        "prerequisite": "Transparent reference cell + weak permanent magnet array (optional)",
        "measurements": [
            {
                "quantity": "Local velocity field with and without applied B",
                "method": "Particle image velocimetry (PIV) or simple dye streak lines",
                "resolution_required": "±10 % on velocity",
                "calibrates": "Lorentz velocity scale factor",
            },
            {
                "quantity": "Limiting current density (or FE at high j) vs applied B",
                "method": "RDE or segmented cathode with variable magnet placement",
                "resolution_required": "±3 % on i_lim",
                "calibrates": "Boundary-layer reduction factor",
            },
            {
                "quantity": "Stray magnetic field inside operating cell",
                "method": "Hall probe at multiple locations (busbars, steel frame)",
                "resolution_required": "±5 µT",
                "calibrates": "B_stay value",
            },
        ],
        "decision_rules": {
            "confirm": "Measured δ reduction within ±25 % of prediction at 100–300 mA/cm²",
            "recalibrate": "Systematic offset → refit Lorentz factor and B_stay",
            "escalate": "Observed large-scale rolls or flow reversal not captured by 1-D model",
        },
    }


def model_scope() -> Dict[str, Any]:
    return {
        "provenance": "Screening Level-1 MHD model. No iron-bath MHD data exists.",
        "computes": [
            "Lorentz velocity scale from j × B",
            "Effective diffusion-layer thinning",
            "Additional mass-transfer coefficient",
            "Drop-in δ_eff for diffusion_layer_1d and gas_holdup",
        ],
        "does_not_compute": [
            "Full 3-D MHD flow field",
            "Induced magnetic field from the current itself",
            "Turbulent transition criteria",
            "Interaction with gas bubbles (see future MHD+gas_holdup coupling)",
        ],
        "dominant_uncertainty": "Local |B| inside the cell (Earth + stray).",
        "replaced_by": "measurement_protocol()",
        "level": 1,
    }
