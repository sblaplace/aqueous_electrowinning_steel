"""Canonical thermodynamic and transport constants for the aqueous cell.

The repository historically carried several copies of the same electrochemical
constants.  That is dangerous in a coupled model: a few millivolts in a
standard potential or a second temperature correction on conductivity can move
an energy result without any change in the physical input.

This module is deliberately small and dependency-free.  Values are screening
anchors, not a replacement for a temperature/composition database.  Every
caller should still report the provenance and uncertainty of a fitted value.
"""

from __future__ import annotations

import math

R_GAS = 8.314462618  # J mol-1 K-1
FARADAY = 96485.33212  # C mol-1
T_REF_K = 298.15

# Reduction potentials, V vs SHE, at the conventional 25 C standard state.
# Keep one value for the shared Fe2+/Fe couple.  The former speciation path
# used -0.447 V while kinetics/electrochemistry used -0.440 V.
E0_FE_REDUCTION_V = -0.440
E0_OER_V = 1.229
E0_FE3_FE2_V = 0.771

# Aqueous equilibria, 25 C, concentration/molar standard-state screening
# anchors.  Non-ideal calculations should apply activities and a measured
# composition where available.
KA_HSO4_25 = 1.05e-2
KA_BORIC_25 = 5.8e-10
KW_25 = 1.0e-14
KSP_FEOH2_25 = 4.87e-17
LOGKSP_FEOH2_25 = math.log10(KSP_FEOH2_25)

# Approximate reaction enthalpies used only for van't Hoff screening updates.
DH_HSO4_J_MOL = -22.4e3
DH_BORIC_J_MOL = 13.8e3
DH_FEOH2_DISSOLUTION_J_MOL = 22.0e3

# Infinite-dilution diffusivity anchors at 25 C, m2 s-1.
D_FE2_25 = 7.2e-10
D_H_25 = 9.31e-9
D_OH_25 = 5.27e-9
D_HSO4_25 = 1.33e-9
D_SO4_25 = 1.07e-9
D_NA_25 = 1.33e-9
D_H3BO3_25 = 0.92e-9
D_H2BO3_25 = 1.00e-9
DIFFUSION_EA_J_MOL = 18.0e3


def vanthoff_constant(K_ref: float, T_K: float, dH_J_mol: float,
                      T_ref_K: float = T_REF_K) -> float:
    """Return ``K(T)`` from a constant-enthalpy van't Hoff approximation."""
    return float(K_ref * math.exp(-dH_J_mol / R_GAS * (1.0 / T_K - 1.0 / T_ref_K)))


def ksp_feoh2(T_K: float) -> float:
    """Temperature-corrected Fe(OH)2 Ksp, screening value."""
    return vanthoff_constant(KSP_FEOH2_25, T_K, DH_FEOH2_DISSOLUTION_J_MOL)


def kw_water(T_K: float) -> float:
    """Temperature-corrected water autoprotolysis constant, screening value."""
    # Approximate enthalpy for water dissociation; use the same convention as
    # the prior speciation implementation.
    return vanthoff_constant(KW_25, T_K, 55.8e3)


def diffusivity_at_temperature(D25_m2_s: float, T_K: float,
                                Ea_J_mol: float = DIFFUSION_EA_J_MOL) -> float:
    """Arrhenius-scaled diffusivity from the shared 25 C anchor."""
    return float(D25_m2_s * math.exp(Ea_J_mol / R_GAS * (1.0 / T_REF_K - 1.0 / T_K)))


def buffer_capacity_M_per_pH(
    pH: float,
    temperature_C: float,
    total_sulfate_M: float = 0.0,
    total_borate_M: float = 0.0,
) -> float:
    """Approximate acid/base buffer capacity of a sulfate/borate bath.

    The result is ``d(mol acid-equivalent/L) / d(pH)``.  It includes the
    HSO4-/SO4(2-) and B(OH)3/B(OH)4- pairs plus water.  It intentionally does
    not pretend that total boric-acid concentration is itself a buffer
    capacity; at pH 2, 0.4 M boric acid contributes only about 5e-8 M/pH.

    This is a concentration-scale screening calculation.  A production
    implementation should evaluate the same derivative from the full
    activity-based speciation Jacobian and include all ligands/solids.
    """
    if not 0.0 <= pH <= 14.0:
        raise ValueError("pH must lie between 0 and 14")
    if total_sulfate_M < 0.0 or total_borate_M < 0.0:
        raise ValueError("total sulfate and borate must be non-negative")

    T_K = float(temperature_C) + 273.15
    H = 10.0 ** (-float(pH))
    ka_s = vanthoff_constant(KA_HSO4_25, T_K, DH_HSO4_J_MOL)
    ka_b = vanthoff_constant(KA_BORIC_25, T_K, DH_BORIC_J_MOL)
    kw = kw_water(T_K)

    # For HA <-> H+ + A-, beta = 2.303 C Ka H/(Ka+H)^2.
    beta_sulfate = 2.303 * total_sulfate_M * ka_s * H / (ka_s + H) ** 2
    beta_borate = 2.303 * total_borate_M * ka_b * H / (ka_b + H) ** 2
    beta_water = 2.303 * (H + kw / max(H, 1e-30))
    return float(max(beta_sulfate + beta_borate + beta_water, 0.0))


__all__ = [
    "R_GAS", "FARADAY", "T_REF_K",
    "E0_FE_REDUCTION_V", "E0_OER_V", "E0_FE3_FE2_V",
    "KA_HSO4_25", "KA_BORIC_25", "KW_25", "KSP_FEOH2_25",
    "LOGKSP_FEOH2_25", "DH_HSO4_J_MOL", "DH_BORIC_J_MOL",
    "DH_FEOH2_DISSOLUTION_J_MOL", "D_FE2_25", "D_H_25", "D_OH_25",
    "D_HSO4_25", "D_SO4_25", "D_NA_25", "D_H3BO3_25", "D_H2BO3_25",
    "DIFFUSION_EA_J_MOL", "vanthoff_constant", "ksp_feoh2", "kw_water",
    "diffusivity_at_temperature", "buffer_capacity_M_per_pH",
]
