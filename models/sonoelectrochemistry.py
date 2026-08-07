"""Sonoelectrochemistry / ultrasonic agitation physics for iron electrowinning.

Why this module exists
----------------------
Ultrasonic agitation (horns, transducers, or bath sonication) produces
acoustic streaming and cavitation micro-jets that can:

* Reduce effective diffusion-layer thickness by 2–5×.
* Degass H₂ bubbles (reducing surface coverage and ohmic penalty).
* Suppress dendritic growth via micro-jets.
* Improve deposit morphology and adhesion.

This is orthogonal to pulse-reverse, levelers, and the new MHD module.
It is a practical, low-cost lever for high-rate plating that the
current suite does not model.

The module provides:
* Acoustic streaming velocity and effective δ reduction.
* Cavitation threshold and micro-jet velocity estimates.
* Degassing enhancement factor for gas_holdup.py.
* Coupling to diffusion_layer_1d.py and pulse.py.
* Measurement protocol.

Scope: screening Level-1 model.  No ultrasonic data for this bath exists.
All correlations are transferred from water electrolysis / high-rate
plating literature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .electrochemistry import FARADAY

# ─── Screening physical constants ────────────────────────────────────
RHO = 1200.0          # kg/m³
MU = 7.0e-4           # Pa·s
SIGMA = 0.070         # N/m
D_FE2 = 7.2e-10       # m²/s (infinite dilution)

# Typical ultrasonic parameters (bench-scale)
US_FREQUENCY_HZ = 20_000
US_POWER_W = 100.0
US_TRANSDUCER_AREA_M2 = 0.001     # ~3 cm diameter horn

# Acoustic streaming velocity scaling (screening)
STREAMING_VELOCITY_FACTOR = 0.8   # 0.5–1.5 literature range

# Cavitation threshold pressure (Pa) for aqueous sulfate ~60 °C
CAVITATION_THRESHOLD_PA = 1.5e5


@dataclass
class UltrasonicParameters:
    """Ultrasonic operating point."""
    frequency_hz: float = US_FREQUENCY_HZ
    power_w: float = US_POWER_W
    transducer_area_m2: float = US_TRANSDUCER_AREA_M2
    distance_to_electrode_m: float = 0.02   # typical horn-to-cathode gap


@dataclass
class SonoelectroResult:
    """Result of ultrasonic agitation calculation."""
    acoustic_streaming_velocity_m_s: float
    effective_delta_reduction_factor: float
    microjet_velocity_m_s: float
    degassing_factor: float          # multiplier on bubble departure rate
    cavitation_active: bool
    notes: str = ""


def acoustic_streaming_velocity(
    power_w: float,
    transducer_area_m2: float,
    frequency_hz: float = US_FREQUENCY_HZ,
    rho: float = RHO,
    factor: float = STREAMING_VELOCITY_FACTOR,
) -> float:
    """Screening acoustic streaming velocity (m/s).

    u_ac ≈ factor * sqrt( (2 * I) / (rho * c) )   (simplified)
    where I = power / area is acoustic intensity.
    """
    if power_w <= 0 or transducer_area_m2 <= 0:
        return 0.0
    intensity = power_w / transducer_area_m2
    # Speed of sound in water ~1480 m/s; screening value
    c = 1480.0
    u = factor * math.sqrt(2 * intensity / (rho * c))
    return float(max(u, 0.0))


def cavitation_microjet_velocity(
    power_w: float,
    transducer_area_m2: float,
    threshold_pa: float = CAVITATION_THRESHOLD_PA,
) -> float:
    """Rough estimate of cavitation micro-jet velocity (m/s)."""
    if power_w <= 0:
        return 0.0
    intensity = power_w / transducer_area_m2
    # Corrected screening: compare intensity directly to a threshold intensity
    threshold_intensity = 15.0   # W/m² screening value
    if intensity < threshold_intensity:
        return 0.0
    excess = intensity - threshold_intensity
    delta_p = excess * 1e4   # screening Pa conversion
    u_jet = math.sqrt(2 * max(delta_p, 0) / RHO)
    return float(min(u_jet, 300.0))  # cap below supersonic for screening


def ultrasonic_delta_reduction(
    u_streaming_m_s: float,
    delta_forced_m: float,
    diffusivity_m2_s: float = D_FE2,
) -> float:
    """Return δ_eff / δ_forced due to acoustic streaming."""
    if u_streaming_m_s <= 0 or delta_forced_m <= 0:
        return 1.0
    k_forced = diffusivity_m2_s / delta_forced_m
    # Additional mass transfer from streaming (order-of-magnitude)
    k_us = u_streaming_m_s * math.sqrt(diffusivity_m2_s / delta_forced_m)
    k_total = math.hypot(k_forced, k_us)
    return float(diffusivity_m2_s / (k_total * delta_forced_m))


def degassing_enhancement(
    u_streaming_m_s: float,
    u_microjet_m_s: float,
    base_departure_rate: float = 1.0,
) -> float:
    """Multiplier on bubble departure rate (degassing benefit)."""
    # Streaming + micro-jets both help detach bubbles earlier
    enhancement = 1.0 + 0.8 * (u_streaming_m_s + 0.3 * u_microjet_m_s)
    return float(min(enhancement, 4.0))   # cap at 4×


def compute_sonoelectro_result(
    params: UltrasonicParameters = UltrasonicParameters(),
    delta_forced_m: float = 50.0e-6,
    diffusivity_m2_s: float = D_FE2,
) -> SonoelectroResult:
    """Full sonoelectrochemistry calculation."""
    u_stream = acoustic_streaming_velocity(
        params.power_w, params.transducer_area_m2, params.frequency_hz
    )
    u_jet = cavitation_microjet_velocity(params.power_w, params.transducer_area_m2)

    reduction = ultrasonic_delta_reduction(u_stream, delta_forced_m, diffusivity_m2_s)
    degas = degassing_enhancement(u_stream, u_jet)

    cav_active = u_jet > 0.0

    notes = (
        f"US power {params.power_w} W at {params.frequency_hz/1000:.0f} kHz. "
        f"Streaming {u_stream*1000:.1f} mm/s, micro-jets {'active' if cav_active else 'inactive'}."
    )

    return SonoelectroResult(
        acoustic_streaming_velocity_m_s=u_stream,
        effective_delta_reduction_factor=reduction,
        microjet_velocity_m_s=u_jet,
        degassing_factor=degas,
        cavitation_active=cav_active,
        notes=notes,
    )


def ultrasonic_enhanced_delta(
    delta_forced_m: float,
    result: SonoelectroResult,
) -> float:
    """Convenience: return the thinned diffusion layer."""
    return float(delta_forced_m * result.effective_delta_reduction_factor)


# ─── Drop-in helper for existing transport / gas models ──────────────
def effective_delta_with_ultrasound(
    delta_forced_m: float,
    us_power_w: float = 100.0,
    us_area_m2: float = 0.001,
    frequency_hz: float = US_FREQUENCY_HZ,
    diffusivity_m2_s: float = D_FE2,
) -> float:
    """Drop-in replacement for forced-only δ."""
    params = UltrasonicParameters(
        frequency_hz=frequency_hz,
        power_w=us_power_w,
        transducer_area_m2=us_area_m2,
    )
    res = compute_sonoelectro_result(params, delta_forced_m, diffusivity_m2_s)
    return ultrasonic_enhanced_delta(delta_forced_m, res)


# ─── Measurement protocol ────────────────────────────────────────────
def measurement_protocol() -> Dict[str, Any]:
    return {
        "title": "Ultrasonic agitation benefit on RC-1 cell",
        "objective": (
            "Measure the improvement in FE, morphology, and bubble "
            "departure when ultrasonic agitation is added to the reference cell."
        ),
        "estimated_cost_usd": 800,
        "estimated_duration_days": 3,
        "prerequisite": "RC-1 with transparent window + 20–40 kHz ultrasonic horn",
        "measurements": [
            {
                "quantity": "Limiting current / FE vs US power",
                "method": "Galvanostatic runs at 100–400 mA/cm² with/without US",
                "resolution_required": "±2 % FE",
                "calibrates": "δ reduction factor and degassing enhancement",
            },
            {
                "quantity": "Bubble departure diameter and frequency",
                "method": "High-speed video through window",
                "resolution_required": "≥1000 fps",
                "calibrates": "degassing_factor",
            },
            {
                "quantity": "Deposit morphology (SEM) with/without US",
                "method": "Post-run SEM + roughness profilometry",
                "resolution_required": "Standard lab SEM",
                "calibrates": "dendrite suppression claim",
            },
        ],
        "decision_rules": {
            "confirm": "≥15 % FE gain or 30 % thinner effective δ at 200 mA/cm²",
            "recalibrate": "Smaller benefit → adjust streaming factor",
            "escalate": "Cavitation damage to deposit or excessive heating",
        },
    }


def model_scope() -> Dict[str, Any]:
    return {
        "provenance": "Screening Level-1 sonoelectrochemistry model.",
        "computes": [
            "Acoustic streaming velocity",
            "Effective diffusion-layer thinning",
            "Cavitation micro-jet velocity",
            "Degassing enhancement factor (bubble departure)",
            "Drop-in δ_eff for diffusion_layer_1d / gas_holdup",
        ],
        "does_not_compute": [
            "Full acoustic field simulation",
            "Standing-wave patterns in the channel",
            "Erosion / pitting from prolonged cavitation",
            "Interaction with pulse waveforms (future work)",
        ],
        "dominant_uncertainty": "Acoustic streaming velocity factor and transducer-to-electrode coupling efficiency.",
        "replaced_by": "measurement_protocol()",
        "level": 1,
    }
