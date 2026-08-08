"""Electrochemical Quartz Crystal Microbalance (EQCM) metrology for real-time deposit mass and hydrogen inventory.

Why this module exists
----------------------
The program's primary artifact is "a weighed, characterized iron deposit with a
closed charge/mass/electrolyte balance".  Post-run gravimetry is the current
standard, but EQCM gives **in-situ, real-time** mass gain + simultaneous HER
detection via frequency shift and dissipation.  It also quantifies diffusible
vs trapped hydrogen (linking to hydrogen_trapping.py and internal_stress.py).

This module supplies:
* Sauerbrey mass calculation + viscoelastic correction.
* Synthetic EQCM data generator for calibration_pipeline and run_record.
* Hydrogen-trapping frequency-shift model.
* Drop-in hooks for reference_cell_pipeline and digital_twin.

Scope: screening Level-1 model.  No real EQCM data exists in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


from .electrochemistry import FARADAY

# ─── EQCM constants (AT-cut 5 MHz crystal, screening) ────────────────
F0_HZ = 5_000_000.0          # fundamental frequency
C_SENSITIVITY_HZ_CM2_NG = 56.6   # Sauerbrey sensitivity (Hz cm² / ng) for 5 MHz
RHO_Q = 2.648e3              # kg/m³ quartz
MU_Q = 2.947e10              # Pa quartz shear modulus

# Trapped-H ceiling from dissipation (screening; calibration target ~200 ppm)
H_TRAP_MAX_PPM = 300.0

@dataclass
class EQCMResult:
    """Result of an EQCM measurement / simulation."""
    frequency_shift_Hz: float
    mass_gain_ug_cm2: float
    viscoelastic_correction: float
    trapped_h_ppm: float
    notes: str = ""


def sauerbrey_mass(
    delta_f_Hz: float,
    area_cm2: float = 0.2,
) -> float:
    """Mass gain from frequency shift (µg/cm²) using Sauerbrey equation."""
    if delta_f_Hz >= 0:
        return 0.0
    # Δm = - (Δf / C) * area   but we want per cm²
    mass_ug_cm2 = -delta_f_Hz / C_SENSITIVITY_HZ_CM2_NG
    return float(mass_ug_cm2)


def viscoelastic_correction(
    delta_f_Hz: float,
    dissipation: float,
    film_thickness_um: float = 10.0,
) -> float:
    """Simple viscoelastic correction factor (1.0 = rigid film)."""
    # Rough model: correction grows with thickness and dissipation
    if film_thickness_um < 1.0 or dissipation < 1e-6:
        return 1.0
    correction = 1.0 + 0.15 * (film_thickness_um / 10.0) * (dissipation / 1e-4)
    return float(min(correction, 1.8))


def trapped_hydrogen_from_dissipation(
    dissipation: float,
    deposit_mass_ug_cm2: float,
    max_h_ppm: float = H_TRAP_MAX_PPM,
    ref_dissipation: float = 5e-5,
) -> float:
    """Estimate trapped H concentration (ppm by mass) from dissipation.

    Dissipation is a proxy for deposit disorder / internal defects that trap
    diffusible hydrogen, so trapped H scales with dissipation relative to a
    reference level, saturating at a physical ceiling (H_TRAP_MAX_PPM).  It is
    a *concentration*, so it is independent of total deposit mass — the deposit
    must merely exist (mass > 0).
    """
    if deposit_mass_ug_cm2 <= 0:
        return 0.0
    if ref_dissipation <= 0:
        return 0.0
    h_ppm = max_h_ppm * (dissipation / ref_dissipation)
    return float(min(max(h_ppm, 0.0), max_h_ppm))


def simulate_eqcm_run(
    charge_density_C_cm2: float,
    fe_efficiency: float = 0.85,
    trapped_h_ppm: float = 200.0,
    film_thickness_um: float = 15.0,
    dissipation: float = 3e-5,
) -> EQCMResult:
    """Generate a synthetic EQCM result for a plating run."""
    # Mass from Fe only (Faraday)
    # Q (C/cm²) * M (g/mol) / (2F) → g/cm², then ×1e6 → µg/cm²
    m_fe_g_cm2 = (charge_density_C_cm2 * fe_efficiency * 55.845) / (2 * FARADAY)
    m_fe_ug_cm2 = m_fe_g_cm2 * 1_000_000.0
    # Add trapped H contribution to effective mass (small)
    m_total_ug_cm2 = m_fe_ug_cm2 * (1 + trapped_h_ppm * 1e-6)

    # Frequency shift (negative for mass gain)
    delta_f = -m_total_ug_cm2 * C_SENSITIVITY_HZ_CM2_NG

    visc_corr = viscoelastic_correction(delta_f, dissipation, film_thickness_um)
    h_est = trapped_hydrogen_from_dissipation(dissipation, m_total_ug_cm2)

    return EQCMResult(
        frequency_shift_Hz=delta_f,
        mass_gain_ug_cm2=m_total_ug_cm2,
        viscoelastic_correction=visc_corr,
        trapped_h_ppm=h_est,
        notes=f"Simulated run: FE={fe_efficiency:.2f}, H={trapped_h_ppm} ppm",
    )


def model_scope() -> Dict[str, Any]:
    return {
        "provenance": "Screening Level-1 EQCM model.",
        "computes": [
            "Sauerbrey mass from frequency shift",
            "Viscoelastic correction",
            "Trapped hydrogen estimate from dissipation",
            "Synthetic data for calibration and digital twin",
        ],
        "level": 1,
    }
