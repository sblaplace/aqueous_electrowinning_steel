"""Bath rheology and non-Newtonian viscosity model for concentrated iron electrolytes.

Why this module exists
----------------------
The Fe²⁺ + anion + particle-laden + surfactant bath is **not** Newtonian
water.  Shear-thinning or yield-stress behaviour changes:

* Bubble rise velocity and gas hold-up (gas_holdup.py)
* Boundary-layer thickness (diffusion_layer_1d.py)
* Membrane electro-osmotic flow (membrane_transport.py)
* Solutal convection strength

This module supplies a simple Herschel–Bulkley / Carreau–Yasuda closure
that can be dropped into the existing transport and two-phase modules.

Scope: screening Level-1 model.  No rheology data for the exact bath
exists in the repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BathRheologyParams:
    """Rheological parameters for the iron bath."""
    # Herschel–Bulkley parameters (screening values for 1–2 M Fe²⁺ + additives)
    yield_stress_Pa: float = 0.05          # Pa (small but non-zero with particles)
    consistency_index_Pa_s_n: float = 0.0012
    flow_index_n: float = 0.85             # <1 → shear-thinning
    density_kg_m3: float = 1200.0
    temperature_C: float = 60.0

    # Particle volume fraction (from Fe(OH)₃ or inclusions)
    phi_particles: float = 0.01


def herschel_bulkley_viscosity(
    shear_rate_s: float,
    params: BathRheologyParams,
) -> float:
    """Herschel–Bulkley apparent viscosity (Pa·s)."""
    if shear_rate_s <= 0:
        return float('inf')
    tau_y = params.yield_stress_Pa
    K = params.consistency_index_Pa_s_n
    n = params.flow_index_n

    if shear_rate_s * params.density_kg_m3 < 1e-6:   # very low shear
        return K * (shear_rate_s ** (n - 1)) + tau_y / max(shear_rate_s, 1e-6)

    tau = tau_y + K * (shear_rate_s ** n)
    return tau / shear_rate_s


def effective_viscosity_for_gas_holdup(
    params: BathRheologyParams,
    shear_rate_s: float = 10.0,   # typical wall shear in channel
) -> float:
    """Return viscosity to use in gas_holdup.py terminal velocity calculations."""
    return herschel_bulkley_viscosity(shear_rate_s, params)


def viscosity_correction_for_delta(
    params: BathRheologyParams,
    base_nu_m2_s: float = 7.0e-4 / 1200.0,
    shear_rate_s: float = 10.0,
) -> float:
    """Return corrected kinematic viscosity for boundary-layer calculations."""
    eta = herschel_bulkley_viscosity(shear_rate_s, params)
    return eta / params.density_kg_m3


def model_scope() -> Dict[str, Any]:
    return {
        "provenance": "Screening Level-1 rheology model.",
        "computes": [
            "Herschel–Bulkley apparent viscosity",
            "Effective viscosity for gas_holdup and diffusion_layer",
        ],
        "does_not_compute": [
            "Full thixotropy / time-dependent rheology",
            "Temperature-dependent parameters (future work)",
        ],
        "level": 1,
    }
