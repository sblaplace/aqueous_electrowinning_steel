"""Deposit internal stress: Stoney/bent-strip measurement theory, mechanism
prediction, and the coupon-curvature experiment.

Closes the gap named in ``docs/RESEARCH_PROGRAM.md`` (Missing Physics item 7):

    "Deposit internal stress. Still missing. No Stoney/bent-strip model
    exists; run_mechanical_properties.py explicitly disclaims texture and
    residual stress. Internal stress governs whether thick deposits crack or
    curl off the substrate, and it is measurable for ~$200 (Tier 0 item 6).
    This is now the largest un-modelled physics gap on the product path."

``adhesion_peel.py`` already carries the *terms* — Hoffman grain-coalescence,
hydrogen-effusion, thermal-mismatch — but it consumes grain size and hydrogen
content as inputs and ships a bare inverse-Stoney one-liner.  This module is
the standalone home of the stress problem:

1. **Measurement theory.** Forward and inverse Stoney, the bent-strip
   cantilever forms (Brenner–Senderoff convention), an exact two-layer
   laminate curvature for the finite-thickness correction, and a GUM-style
   uncertainty budget — so the ~$200 coupon experiment returns a number with
   a closed error budget, not a vibe.

2. **Mechanism prediction.** ``deposit_stress_from_conditions`` derives grain
   size, diffusible hydrogen, and thickness from *plating conditions* via the
   existing models (mechanical_properties, hydrogen_embrittlement, Faraday's
   law), then decomposes the residual stress, including the two empirical
   electroforming knobs the literature actually uses: saccharin stress relief
   and chloride-bath shift (Di Bari, *Modern Electroplating*).

3. **Peel handoff.** ``peel_verdict_from_conditions`` maps corrected stress
   components back onto equivalent (grain, C_H) inputs and runs
   ``adhesion_peel.evaluate_peel``, so a bath condition — not an assumed
   stress — decides the drum-and-strip verdict.

4. **The experiment.** ``coupon_curvature_protocol`` specifies the Tier-0
   bent-strip/Stoney measurement set: shim geometry, plating plan, instrument
   resolution requirements, uncertainty budget, and kill/confirm/redirect
   rules.  Screening mechanics: no iron internal-stress data exists in this
   repository yet.

Stress sign convention throughout: **positive = tensile**, and tensile film
stress bends the coupon with the film side convex (positive curvature).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .adhesion_peel import (
    E_FE_GPA,
    HOFFMAN_DELTA_M,
    NU_FE,
    SUBSTRATES,
    PeelConditions,
    SubstrateSpec,
    biaxial_modulus_Pa,
    energy_release_rate,
    evaluate_peel,
    hoffman_intrinsic_stress_MPa,
    hydrogen_stress_MPa,
    stoney_stress_MPa,  # the thin-film inverse; re-exported as our own
    thermal_mismatch_stress_MPa,
)

# ═════════════════════════════════════════════════════════════════════
#  Coupon and measurement defaults
# ═════════════════════════════════════════════════════════════════════
# The default coupon is 316L shim stock: it is the same material as the
# Day-1 gravimetric FE coupons (docs/FIRST_LAB_DAY.md R3/R4), cheap, and
# available in precise thicknesses.  A Ti shim variant covers the drum
# surface (adhesion_peel's reference substrate).
COUPON_E_GPA = 193.0          # 316L, Young's modulus
COUPON_NU = 0.29              # 316L, Poisson ratio
COUPON_THICKNESS_MM = 0.4     # stiff enough that 400 MPa films stay small-deflection
COUPON_LENGTH_MM = 60.0       # cantilever gauge length
COUPON_WIDTH_MM = 10.0

# Instrument resolutions (order-of-magnitude, for the uncertainty budget)
DIAL_GAUGE_RESOLUTION_UM = 10.0       # cheap dial indicator, ~$25
PROFILOMETER_RESOLUTION_UM = 1.0      # stylus profilometer trace (shared)

# ═════════════════════════════════════════════════════════════════════
#  Empirical electroforming knobs (engineering estimates, screening level)
# ═════════════════════════════════════════════════════════════════════
# Saccharin is the classical stress reliever of Ni/Fe electroforming
# (Di Bari, Modern Electroplating ch. on electroforming; MEMS/LIGA iron
# literature).  It acts on the *intrinsic* term: at a few g/L it converts
# strongly tensile deposits toward compression.  Functional form and
# constants are our screening fit, not a measurement.
SACCHARIN_RELIEF_MAX = 0.80            # max fractional relief of intrinsic stress
SACCHARIN_REF_G_L = 1.5                # concentration at ~63% of max relief
# Chloride baths plate with lower tensile (often compressive) stress than
# sulfate baths at equal current density — same source family.  Screening
# shift applied to the intrinsic term, MPa.
CHLORIDE_SHIFT_MPa = 30.0

# Stress evolution with thickness: many iron/steel electrodeposits start
# with an interfacial transient (often compressive) and relax onto a
# plateau.  h_char is the relaxation length of that transient (screening).
STRESS_INTERFACE_DEFAULT_MPa = -40.0   # compressive interfacial transient
STRESS_CHARACTERISTIC_UM = 10.0        # relaxation length


# ═════════════════════════════════════════════════════════════════════
#  1. Measurement theory: Stoney forward, bent strip, finite thickness
# ═════════════════════════════════════════════════════════════════════

def curvature_from_stress_MPa(
    sigma_MPa: float,
    film_thickness_m: float,
    substrate_thickness_m: float,
    substrate_E_GPa: float = COUPON_E_GPA,
    substrate_nu: float = COUPON_NU,
) -> float:
    """Forward Stoney: film stress → coupon curvature change (1/m).

    ``κ = 6 σ (1 − ν_s) h_f / (E_s h_s²)``

    This is the thin-film form (h_f ≪ h_s).  Positive σ (tensile) gives
    positive curvature (film side convex).  This is the prediction side of
    the experiment: *before* plating, given a mechanism estimate of σ, it
    says whether the curvature will be measurable.
    """
    if film_thickness_m <= 0 or substrate_thickness_m <= 0:
        raise ValueError("thicknesses must be positive")
    return (
        6.0 * sigma_MPa * 1e6 * (1.0 - substrate_nu) * film_thickness_m
        / (substrate_E_GPa * 1e9 * substrate_thickness_m ** 2)
    )


def cantilever_deflection_m(
    sigma_MPa: float,
    gauge_length_m: float = COUPON_LENGTH_MM * 1e-3,
    film_thickness_m: float = 25e-6,
    substrate_thickness_m: float = COUPON_THICKNESS_MM * 1e-3,
    substrate_E_GPa: float = COUPON_E_GPA,
    substrate_nu: float = COUPON_NU,
) -> float:
    """Free-end deflection of a clamped coupon under film stress (m).

    For a cantilever with small deflection, κ ≈ 2δ/L², so
    ``δ = 3 σ (1 − ν_s) h_f L² / (E_s h_s²)``.  This is the quantity a dial
    gauge or profilometer actually reads.
    """
    if gauge_length_m <= 0:
        raise ValueError("gauge length must be positive")
    kappa = curvature_from_stress_MPa(
        sigma_MPa,
        film_thickness_m,
        substrate_thickness_m,
        substrate_E_GPa=substrate_E_GPa,
        substrate_nu=substrate_nu,
    )
    return kappa * gauge_length_m ** 2 / 2.0


def bent_strip_stress_MPa(
    deflection_m: float,
    gauge_length_m: float = COUPON_LENGTH_MM * 1e-3,
    film_thickness_m: float = 25e-6,
    substrate_thickness_m: float = COUPON_THICKNESS_MM * 1e-3,
    substrate_E_GPa: float = COUPON_E_GPA,
    substrate_nu: float = COUPON_NU,
) -> float:
    """Inverse bent-strip (Brenner–Senderoff convention): deflection → σ (MPa).

    ``σ = E_s h_s² δ / (3 (1 − ν_s) h_f L²)``

    Consistent with :func:`stoney_stress_MPa` under the small-deflection
    cantilever relation κ = 2δ/L²; the round trip is exact.
    """
    if deflection_m == 0:
        return 0.0
    if gauge_length_m <= 0 or film_thickness_m <= 0 or substrate_thickness_m <= 0:
        raise ValueError("geometry must be positive")
    sigma_Pa = (
        substrate_E_GPa * 1e9 * substrate_thickness_m ** 2 * deflection_m
        / (3.0 * (1.0 - substrate_nu) * film_thickness_m * gauge_length_m ** 2)
    )
    return sigma_Pa / 1e6


# ═════════════════════════════════════════════════════════════════════
#  2. Finite-thickness correction (exact two-layer laminate)
# ═════════════════════════════════════════════════════════════════════

def _two_layer_curvature_per_stress(
    sigma_true_MPa: float,
    film_thickness_m: float,
    substrate_thickness_m: float,
    film_E_GPa: float = E_FE_GPA,
    film_nu: float = NU_FE,
    substrate_E_GPa: float = COUPON_E_GPA,
    substrate_nu: float = COUPON_NU,
) -> float:
    """Exact biaxial two-layer curvature for a film with eigenstress (1/m).

    Solves the Euler–Bernoulli laminate: the film carries eigenstress
    ``σ* = M_f ε*``, the neutral axis sits where the elastic first moment
    vanishes, and the curvature follows from moment equilibrium.  Reduces to
    thin-film Stoney as h_f → 0 (tested).  Positive σ* → positive curvature.
    """
    M_s = biaxial_modulus_Pa(substrate_E_GPa, substrate_nu)
    M_f = biaxial_modulus_Pa(film_E_GPa, film_nu)
    H = substrate_thickness_m + film_thickness_m
    # Neutral axis (elastic first moment of area = 0):
    z0 = (
        M_s * substrate_thickness_m ** 2
        + M_f * (H ** 2 - substrate_thickness_m ** 2)
    ) / (2.0 * (M_s * substrate_thickness_m + M_f * film_thickness_m))
    # First moment of the film about the neutral axis:
    I_film = film_thickness_m * (
        substrate_thickness_m + film_thickness_m / 2.0 - z0
    )
    # Second moment of the whole laminate about the neutral axis:
    def _second_moment(a: float, b: float) -> float:
        return ((b - z0) ** 3 - (a - z0) ** 3) / 3.0

    I_tot = (
        M_s * _second_moment(0.0, substrate_thickness_m)
        + M_f * _second_moment(substrate_thickness_m, H)
    )
    eigenstrain = sigma_true_MPa * 1e6 / M_f
    kappa = M_f * eigenstrain * I_film / I_tot
    return float(kappa)


def finite_thickness_correction(
    film_thickness_m: float,
    substrate_thickness_m: float,
    film_E_GPa: float = E_FE_GPA,
    film_nu: float = NU_FE,
    substrate_E_GPa: float = COUPON_E_GPA,
    substrate_nu: float = COUPON_NU,
) -> float:
    """Correction factor C = σ_true / σ_Stoney-inferred for finite h_f.

    Thin-film Stoney ignores the film's own stiffness, so for thick films
    the curvature-based inference *underestimates* the true stress.  C → 1
    as h_f → 0 and grows with h_f/h_s; the correction is computed from the
    exact laminate, not from a truncated series.
    """
    if film_thickness_m <= 0 or substrate_thickness_m <= 0:
        raise ValueError("thicknesses must be positive")
    sigma_ref = 100.0  # MPa — linear problem, any reference value works
    kappa = _two_layer_curvature_per_stress(
        sigma_ref,
        film_thickness_m,
        substrate_thickness_m,
        film_E_GPa=film_E_GPa,
        film_nu=film_nu,
        substrate_E_GPa=substrate_E_GPa,
        substrate_nu=substrate_nu,
    )
    sigma_inferred = stoney_stress_MPa(
        kappa,
        substrate_thickness_m,
        film_thickness_m,
        substrate_E_GPa=substrate_E_GPa,
        substrate_nu=substrate_nu,
    )
    return float(sigma_ref / sigma_inferred)


def stoney_stress_finite_thickness_MPa(
    curvature_change_per_m: float,
    substrate_thickness_m: float,
    film_thickness_m: float,
    film_E_GPa: float = E_FE_GPA,
    film_nu: float = NU_FE,
    substrate_E_GPa: float = COUPON_E_GPA,
    substrate_nu: float = COUPON_NU,
) -> float:
    """Stoney inverse with the finite-thickness correction applied (MPa)."""
    thin = stoney_stress_MPa(
        curvature_change_per_m,
        substrate_thickness_m,
        film_thickness_m,
        substrate_E_GPa=substrate_E_GPa,
        substrate_nu=substrate_nu,
    )
    corr = finite_thickness_correction(
        film_thickness_m,
        substrate_thickness_m,
        film_E_GPa=film_E_GPa,
        film_nu=film_nu,
        substrate_E_GPa=substrate_E_GPa,
        substrate_nu=substrate_nu,
    )
    return thin * corr


# ═════════════════════════════════════════════════════════════════════
#  3. Uncertainty budget (GUM, independent inputs)
# ═════════════════════════════════════════════════════════════════════

def stress_uncertainty_MPa(
    deflection_m: float,
    u_deflection_m: float,
    gauge_length_m: float = COUPON_LENGTH_MM * 1e-3,
    u_gauge_length_m: float = 0.5e-3,
    film_thickness_m: float = 25e-6,
    u_film_thickness_m: float = 1.0e-6,
    substrate_thickness_m: float = COUPON_THICKNESS_MM * 1e-3,
    u_substrate_thickness_m: float = 2.0e-6,
    substrate_E_GPa: float = COUPON_E_GPA,
    u_substrate_E_GPa: float = 5.0,
    substrate_nu: float = COUPON_NU,
    u_substrate_nu: float = 0.01,
) -> Dict[str, Any]:
    """Standard-uncertainty budget for the bent-strip stress (MPa).

    ``σ = E_s h_s² δ / (3 (1 − ν_s) h_f L²)``; the relative sensitivity of
    σ to each input is its exponent, so the combined standard uncertainty is

    ``u(σ)/σ = sqrt[(uδ/δ)² + (2·uh_s/h_s)² + (uh_f/h_f)²
                    + (uE/E)² + (uν/(1−ν))² + (2·uL/L)²]``.

    Returns the budget with per-term contributions and the dominant term, so
    the protocol can say *which measurement to spend the money on*.
    """
    if deflection_m == 0:
        raise ValueError("zero deflection gives no stress signal")
    sigma = bent_strip_stress_MPa(
        deflection_m,
        gauge_length_m=gauge_length_m,
        film_thickness_m=film_thickness_m,
        substrate_thickness_m=substrate_thickness_m,
        substrate_E_GPa=substrate_E_GPa,
        substrate_nu=substrate_nu,
    )
    terms = {
        "deflection": (u_deflection_m / deflection_m) ** 2,
        "substrate_thickness": (2.0 * u_substrate_thickness_m / substrate_thickness_m) ** 2,
        "film_thickness": (u_film_thickness_m / film_thickness_m) ** 2,
        "substrate_modulus": (u_substrate_E_GPa / substrate_E_GPa) ** 2,
        "poisson": (u_substrate_nu / (1.0 - substrate_nu)) ** 2,
        "gauge_length": (2.0 * u_gauge_length_m / gauge_length_m) ** 2,
    }
    rel = math.sqrt(sum(terms.values()))
    contributions = {k: sigma * math.sqrt(v) for k, v in terms.items()}
    return {
        "sigma_MPa": float(sigma),
        "u_sigma_MPa": float(sigma * rel),
        "relative_uncertainty": float(rel),
        "contributions_MPa": contributions,
        "dominant_uncertainty": max(contributions, key=contributions.get),
    }


def stress_resolution_MPa(
    deflection_resolution_m: float,
    gauge_length_m: float = COUPON_LENGTH_MM * 1e-3,
    film_thickness_m: float = 25e-6,
    substrate_thickness_m: float = COUPON_THICKNESS_MM * 1e-3,
    substrate_E_GPa: float = COUPON_E_GPA,
    substrate_nu: float = COUPON_NU,
) -> float:
    """Smallest resolvable stress change for an instrument resolution (MPa).

    The deflection term only — the resolution floor of the instrument,
    independent of the magnitude being measured.
    """
    return abs(
        bent_strip_stress_MPa(
            deflection_resolution_m,
            gauge_length_m=gauge_length_m,
            film_thickness_m=film_thickness_m,
            substrate_thickness_m=substrate_thickness_m,
            substrate_E_GPa=substrate_E_GPa,
            substrate_nu=substrate_nu,
        )
    )


def deflection_resolution_for_stress_um(
    target_stress_resolution_MPa: float,
    gauge_length_m: float = COUPON_LENGTH_MM * 1e-3,
    film_thickness_m: float = 25e-6,
    substrate_thickness_m: float = COUPON_THICKNESS_MM * 1e-3,
    substrate_E_GPa: float = COUPON_E_GPA,
    substrate_nu: float = COUPON_NU,
) -> float:
    """Deflection resolution (µm) needed to resolve a given stress (MPa)."""
    if target_stress_resolution_MPa <= 0:
        raise ValueError("target resolution must be positive")
    delta_m = (
        3.0
        * (1.0 - substrate_nu)
        * film_thickness_m
        * gauge_length_m ** 2
        * target_stress_resolution_MPa
        * 1e6
        / (substrate_E_GPa * 1e9 * substrate_thickness_m ** 2)
    )
    return delta_m * 1e6


# ═════════════════════════════════════════════════════════════════════
#  4. Mechanism prediction: plating conditions → residual stress
# ═════════════════════════════════════════════════════════════════════

def _resolve_substrate(substrate: Union[str, SubstrateSpec]) -> SubstrateSpec:
    if isinstance(substrate, SubstrateSpec):
        return substrate
    if substrate not in SUBSTRATES:
        raise ValueError(f"unknown substrate {substrate!r}; have {sorted(SUBSTRATES)}")
    return SUBSTRATES[substrate]


def deposit_stress_from_conditions(
    j_mA_cm2: float = 100.0,
    current_efficiency_percent: float = 85.0,
    deposition_time_s: float = 900.0,
    waveform: str = "dc",
    j_peak_mA_cm2: Optional[float] = None,
    duty_cycle: float = 1.0,
    bath_pH: float = 3.0,
    temperature_C: float = 60.0,
    ambient_temperature_C: float = 25.0,
    substrate: Union[str, SubstrateSpec] = "ti_passive_tio2",
    saccharin_g_L: float = 0.0,
    chloride_bath: bool = False,
    hydrogen_fraction_effused: float = 1.0,
    relaxation_distance_m: float = HOFFMAN_DELTA_M,
    include_point_defect_stress: bool = False,
    additive_coverage_fraction: float = 0.0,
) -> Dict[str, Any]:
    """Predict residual stress end-to-end from *plating conditions* (MPa).

    Closes the loop the peel module deliberately left open: where
    ``adhesion_peel.residual_stress`` takes grain size and hydrogen content
    as inputs, this derives them from how the cell is run —

    * grain size ← ``mechanical_properties.estimate_grain_size_um``
    * diffusible H ← ``hydrogen_embrittlement.hydrogen_uptake_from_electrolysis``
    * thickness ← Faraday's law at the given j, FE and time

    — then decomposes the stress into intrinsic (Hoffman), hydrogen-effusion
    and thermal-mismatch terms, and applies the two empirical electroforming
    corrections the bath actually controls: saccharin relief of the intrinsic
    term and the chloride-bath compressive shift.  Both corrections are
    engineering estimates and are reported separately so they can never be
    mistaken for measurement.

    Returns signed components (positive = tensile) plus the derived
    quantities and sources, so a single operating point propagates
    consistently from cell conditions to stress.
    """
    from .electrochemistry import FARADAY, M_FE
    from .hydrogen_embrittlement import hydrogen_uptake_from_electrolysis
    from .mechanical_properties import estimate_grain_size_um

    if j_mA_cm2 <= 0:
        raise ValueError("current density must be positive")
    if not 0.0 < current_efficiency_percent <= 100.0:
        raise ValueError("current efficiency must lie in (0, 100]")
    if saccharin_g_L < 0:
        raise ValueError("saccharin concentration must be non-negative")

    fe = current_efficiency_percent / 100.0
    j_A_m2 = j_mA_cm2 * 10.0
    sub = _resolve_substrate(substrate)

    # Derived deposit state from the existing models (same calls the peel
    # module's conditions_from_deposition makes — consistency is tested).
    grain_um = estimate_grain_size_um(
        j_avg_mA_cm2=j_mA_cm2,
        j_peak_mA_cm2=j_peak_mA_cm2,
        duty_cycle=duty_cycle,
        waveform=waveform,  # type: ignore[arg-type]
        temperature_C=temperature_C,
    )
    h_uptake = hydrogen_uptake_from_electrolysis(
        current_density_mA_cm2=j_mA_cm2,
        deposition_time_s=deposition_time_s,
        her_efficiency=max(1.0 - fe, 1e-4),
        bath_pH=bath_pH,
        temperature_C=temperature_C,
    )
    C_H_ppm = h_uptake["C_H_diffusible_ppm"]
    thickness_um = (
        j_A_m2 * fe * deposition_time_s * M_FE / (2.0 * FARADAY) / 7874.0 * 1e6
    )

    # Mechanism terms — the same physics as adhesion_peel.residual_stress.
    intrinsic = hoffman_intrinsic_stress_MPa(
        grain_um, relaxation_distance_m=relaxation_distance_m
    )
    hydrogen = hydrogen_stress_MPa(C_H_ppm, hydrogen_fraction_effused)
    thermal = thermal_mismatch_stress_MPa(
        temperature_C - ambient_temperature_C,
        alpha_substrate_per_K=sub.thermal_expansion_per_K,
    )

    # Empirical bath corrections (screening estimates; reported separately).
    relief = (
        SACCHARIN_RELIEF_MAX * (1.0 - math.exp(-saccharin_g_L / SACCHARIN_REF_G_L))
        if saccharin_g_L > 0
        else 0.0
    )
    chloride_shift = -CHLORIDE_SHIFT_MPa if chloride_bath else 0.0
    intrinsic_corrected = intrinsic * (1.0 - relief) + chloride_shift

    # Round 5 (C2): non-equilibrium point-defect intrinsic stress, opt-in.
    point_defect_MPa = 0.0
    if include_point_defect_stress:
        from .point_defect_stress import defect_injection_stress_MPa

        # Screening overpotential estimate from current density (Tafel-like).
        eta = 0.05 + 0.06 * math.log10(max(j_mA_cm2 / 100.0, 1.0))
        pd = defect_injection_stress_MPa(
            eta, temperature_C, deposition_time_s,
            additive_coverage_fraction=additive_coverage_fraction,
        )
        point_defect_MPa = pd["net_stress_MPa"]

    total = intrinsic_corrected + hydrogen + thermal + point_defect_MPa
    contributions = {
        "intrinsic": abs(intrinsic_corrected),
        "point_defect": abs(point_defect_MPa),
        "hydrogen": abs(hydrogen),
        "thermal": abs(thermal),
    }
    dominant = max(contributions, key=contributions.get)
    components = {
        "intrinsic_MPa": float(intrinsic_corrected),
        "hydrogen_MPa": float(hydrogen),
        "thermal_MPa": float(thermal),
        "total_MPa": float(total),
    }
    if include_point_defect_stress:
        components["point_defect_MPa"] = float(point_defect_MPa)
    return {
        "components": components,
        "dominant_mechanism": dominant,
        "sign": "tensile" if total >= 0 else "compressive",
        "derived": {
            "grain_size_um": float(grain_um),
            "C_H_diffusible_ppm": float(C_H_ppm),
            "thickness_um": float(thickness_um),
            "deposition_rate_um_hr": float(
                thickness_um / (deposition_time_s / 3600.0)
            ),
        },
        "corrections": {
            "saccharin_g_L": float(saccharin_g_L),
            "intrinsic_relief_fraction": float(relief),
            "chloride_shift_MPa": float(chloride_shift),
            "raw_intrinsic_MPa": float(intrinsic),
        },
        "substrate": sub.id,
        "sources": {
            "grain_size": "mechanical_properties.estimate_grain_size_um",
            "hydrogen": "hydrogen_embrittlement.hydrogen_uptake_from_electrolysis",
            "thickness": "Faraday's law at the supplied j, FE and time",
            "intrinsic": "adhesion_peel.hoffman_intrinsic_stress_MPa",
            "hydrogen_stress": "adhesion_peel.hydrogen_stress_MPa",
            "thermal": "adhesion_peel.thermal_mismatch_stress_MPa",
            "saccharin": "internal_stress screening fit (Di Bari electroforming)",
            "chloride": "internal_stress screening shift (Di Bari electroforming)",
        },
    }


def equivalent_grain_and_hydrogen(
    corrected_intrinsic_MPa: float,
    corrected_hydrogen_MPa: float,
    relaxation_distance_m: float = HOFFMAN_DELTA_M,
    E_GPa: float = E_FE_GPA,
    nu: float = NU_FE,
) -> Tuple[float, float]:
    """Map corrected stress components back to equivalent (grain, C_H).

    ``adhesion_peel.evaluate_peel`` rebuilds stress from
    :class:`PeelConditions` (grain size, hydrogen content), so corrections
    from this module must be expressed *as* those inputs before they can
    reach the peel verdict.  Both mechanisms are linear in their input, so
    the mapping is exact:

    * intrinsic: σ_i = E′·Δ/d  ⇒  d_eff = E′·Δ / σ′_i  (clamped ≥ 1e3 µm
      when the corrected intrinsic term is zero or compressive, so it
      contributes ~nothing through the peel module)
    * hydrogen:   σ_H = k·C_H  ⇒  C_H_eff = σ′_H / k
    """
    Eprime = biaxial_modulus_Pa(E_GPa, nu)
    if corrected_intrinsic_MPa > 0:
        d_eff = Eprime * relaxation_distance_m / (corrected_intrinsic_MPa * 1e6)
        d_eff = min(max(d_eff * 1e6, 1e-3), 1e3)  # µm, bounded
    else:
        d_eff = 1e3  # negligible intrinsic contribution
    k = hydrogen_stress_MPa(1.0, fraction_effused=1.0, E_GPa=E_GPa, nu=nu)  # MPa/ppm
    C_H_eff = max(corrected_hydrogen_MPa / k, 0.0)
    return float(d_eff), float(C_H_eff)


# ═════════════════════════════════════════════════════════════════════
#  5. Stress evolution with thickness
# ═════════════════════════════════════════════════════════════════════

def stress_evolution(
    plateau_stress_MPa: float,
    thickness_um: float,
    interface_stress_MPa: float = STRESS_INTERFACE_DEFAULT_MPa,
    characteristic_thickness_um: float = STRESS_CHARACTERISTIC_UM,
) -> Dict[str, float]:
    """Local and Stoney-measured (average) stress at a given thickness (MPa).

    ``σ_loc(h) = σ_pl + (σ_if − σ_pl)·exp(−h/h_char)`` relaxes the
    interfacial transient onto the plateau; the coupon curvature integrates
    the whole film, so the Stoney experiment reads the thickness-averaged
    stress ``⟨σ⟩(h) = (1/h)∫σ_loc dh``.  The difference between the two is
    exactly why a *thickness sweep* is part of the protocol: the average
    hides the transient, and the transient is what governs adhesion of thin
    deposits.
    """
    if thickness_um <= 0:
        raise ValueError("thickness must be positive")
    if characteristic_thickness_um <= 0:
        raise ValueError("characteristic thickness must be positive")
    local = plateau_stress_MPa + (interface_stress_MPa - plateau_stress_MPa) * math.exp(
        -thickness_um / characteristic_thickness_um
    )
    # ∫₀ʰ σ_loc dh / h  =  σ_pl + (σ_if − σ_pl)·(h_char/h)·(1 − e^(−h/h_char))
    if math.isclose(thickness_um, 0.0):
        average = interface_stress_MPa
    else:
        average = plateau_stress_MPa + (
            interface_stress_MPa - plateau_stress_MPa
        ) * (characteristic_thickness_um / thickness_um) * (
            1.0 - math.exp(-thickness_um / characteristic_thickness_um)
        )
    return {
        "local_MPa": float(local),
        "average_MPa": float(average),
        "plateau_MPa": float(plateau_stress_MPa),
        "interface_MPa": float(interface_stress_MPa),
    }


def stress_profile(
    plateau_stress_MPa: float,
    h_max_um: float,
    n_points: int = 200,
    interface_stress_MPa: float = STRESS_INTERFACE_DEFAULT_MPa,
    characteristic_thickness_um: float = STRESS_CHARACTERISTIC_UM,
) -> Dict[str, np.ndarray]:
    """Vectorised stress-vs-thickness profile for plotting.

    Returns ``h_um``, ``local_MPa`` and the thickness-averaged (Stoney
    measurable) stress ``average_MPa`` over each thickness.
    """
    h = np.linspace(1e-6, h_max_um, n_points)
    local = plateau_stress_MPa + (
        interface_stress_MPa - plateau_stress_MPa
    ) * np.exp(-h / characteristic_thickness_um)
    average = plateau_stress_MPa + (
        interface_stress_MPa - plateau_stress_MPa
    ) * (characteristic_thickness_um / h) * (1.0 - np.exp(-h / characteristic_thickness_um))
    return {"h_um": h, "local_MPa": local, "average_MPa": average}


def driving_force_from_stress(
    sigma_MPa: float,
    thickness_um: float,
    E_GPa: float = E_FE_GPA,
    nu: float = NU_FE,
) -> float:
    """Strain-energy release rate G = (1−ν)σ²h/E (J/m²).

    Re-export of ``adhesion_peel.energy_release_rate`` so the stress module
    can speak directly in peel-driving terms.  G scales with σ² — which is
    why a 30% stress reduction is a 50% reduction in the peel driving force.
    """
    return energy_release_rate(sigma_MPa, thickness_um, E_GPa=E_GPa, nu=nu)


# ═════════════════════════════════════════════════════════════════════
#  6. Peel handoff: bath conditions → drum verdict
# ═════════════════════════════════════════════════════════════════════

def peel_verdict_from_conditions(
    j_mA_cm2: float = 100.0,
    current_efficiency_percent: float = 85.0,
    deposition_time_s: float = 900.0,
    waveform: str = "dc",
    j_peak_mA_cm2: Optional[float] = None,
    duty_cycle: float = 1.0,
    bath_pH: float = 3.0,
    temperature_C: float = 60.0,
    ambient_temperature_C: float = 25.0,
    substrate: Union[str, SubstrateSpec] = "ti_passive_tio2",
    saccharin_g_L: float = 0.0,
    chloride_bath: bool = False,
    peel_angle_deg: float = 90.0,
) -> Dict[str, Any]:
    """Full pipeline: plating conditions → stress → peel verdict.

    Chains :func:`deposit_stress_from_conditions` →
    :func:`equivalent_grain_and_hydrogen` →
    ``adhesion_peel.evaluate_peel``, so the drum-and-strip verdict is a
    function of *how the cell is run*, including the bath knobs (saccharin,
    chloride) that the peel module cannot see directly.  Returns the stress
    decomposition, the equivalent inputs actually handed to the peel model,
    and the full peel result.
    """
    from .mechanical_properties import MechanicalPropertiesModel

    sub = _resolve_substrate(substrate)
    sd = deposit_stress_from_conditions(
        j_mA_cm2=j_mA_cm2,
        current_efficiency_percent=current_efficiency_percent,
        deposition_time_s=deposition_time_s,
        waveform=waveform,
        j_peak_mA_cm2=j_peak_mA_cm2,
        duty_cycle=duty_cycle,
        bath_pH=bath_pH,
        temperature_C=temperature_C,
        ambient_temperature_C=ambient_temperature_C,
        substrate=sub,
        saccharin_g_L=saccharin_g_L,
        chloride_bath=chloride_bath,
    )
    comp = sd["components"]
    grain_eff, C_H_eff = equivalent_grain_and_hydrogen(
        comp["intrinsic_MPa"], comp["hydrogen_MPa"]
    )
    mech = MechanicalPropertiesModel().predict(
        j_avg_mA_cm2=j_mA_cm2,
        j_peak_mA_cm2=j_peak_mA_cm2,
        duty_cycle=duty_cycle,
        waveform=waveform,  # type: ignore[arg-type]
        ni_wt_percent=0.0,
        carbon_wt_percent=0.0,
        current_efficiency_percent=current_efficiency_percent,
        temperature_C=temperature_C,
    )
    conditions = PeelConditions(
        thickness_um=max(sd["derived"]["thickness_um"], 0.05),
        grain_size_um=grain_eff,
        C_H_ppm=C_H_eff,
        bath_temperature_C=temperature_C,
        ambient_temperature_C=ambient_temperature_C,
        peel_angle_deg=peel_angle_deg,
        foil_yield_strength_MPa=mech.sigma_y_MPa,
    )
    result = evaluate_peel(sub, conditions)
    return {
        "inputs": {
            "j_mA_cm2": j_mA_cm2,
            "current_efficiency_percent": current_efficiency_percent,
            "deposition_time_s": deposition_time_s,
            "waveform": waveform,
            "bath_pH": bath_pH,
            "temperature_C": temperature_C,
            "substrate": sub.id,
            "saccharin_g_L": saccharin_g_L,
            "chloride_bath": chloride_bath,
        },
        "stress": sd,
        "equivalent_inputs_to_peel_model": {
            "grain_size_um": float(grain_eff),
            "C_H_diffusible_ppm": float(C_H_eff),
            "note": (
                "evaluate_peel rebuilds stress from these; the mapping from "
                "corrected stress components is exact for both mechanisms."
            ),
        },
        "peel": result.to_dict(),
        "verdict": result.outcome,
        "peelable": result.peelable,
        "good_for_flake_harvest": result.good_for_flake_harvest,
    }


# ═════════════════════════════════════════════════════════════════════
#  7. The coupon-curvature experiment (Tier 0 item 6, ~$200)
# ═════════════════════════════════════════════════════════════════════

def coupon_curvature_protocol(
    j_mA_cm2: float = 100.0,
    current_efficiency_percent: float = 85.0,
    temperature_C: float = 60.0,
    substrate: Union[str, SubstrateSpec] = "ti_passive_tio2",
) -> Dict[str, Any]:
    """Specify the bent-strip/Stoney coupon experiment (the ~$200 Tier 0 item).

    ``docs/RESEARCH_PROGRAM.md`` Tier 0 item 6: *"Deposit on one face of a
    thin shim, measure curvature, get internal stress in real time as a
    function of thickness and waveform."*  This returns the executable
    version: coupon geometry, plating plan, instruments, the uncertainty
    budget, and what each measurement replaces in the model set.  It is the
    stress-side complement of ``adhesion_peel.coupon_test_protocol`` (which
    measures the *interface*; this measures the *film*).
    """
    sub = _resolve_substrate(substrate)
    sd = deposit_stress_from_conditions(
        j_mA_cm2=j_mA_cm2,
        current_efficiency_percent=current_efficiency_percent,
        deposition_time_s=1800.0,  # 30 min → ~50 µm at 85% FE
        temperature_C=temperature_C,
        substrate=sub,
    )
    sigma_ref = sd["components"]["total_MPa"]
    thickness_um = sd["derived"]["thickness_um"]

    # Measurability at the default coupon geometry
    delta_m = cantilever_deflection_m(
        sigma_ref,
        gauge_length_m=COUPON_LENGTH_MM * 1e-3,
        film_thickness_m=thickness_um * 1e-6,
        substrate_thickness_m=COUPON_THICKNESS_MM * 1e-3,
    )
    delta_um = delta_m * 1e6
    budget = stress_uncertainty_MPa(
        deflection_m=max(delta_m, 1e-9),
        u_deflection_m=DIAL_GAUGE_RESOLUTION_UM * 1e-6,
        film_thickness_m=thickness_um * 1e-6,
    )
    res_dial = stress_resolution_MPa(
        DIAL_GAUGE_RESOLUTION_UM * 1e-6, film_thickness_m=thickness_um * 1e-6
    )
    res_prof = stress_resolution_MPa(
        PROFILOMETER_RESOLUTION_UM * 1e-6, film_thickness_m=thickness_um * 1e-6
    )

    return {
        "title": "Bent-strip / Stoney coupon-curvature set (Tier 0 item 6)",
        "gates": (
            "docs/RESEARCH_PROGRAM.md Missing Physics item 7; Tier 0 item 6; "
            "adhesion_peel.calibration_required ('Residual stress by coupon "
            "curvature (Stoney) in the actual bath')"
        ),
        "runs_alongside": (
            "The Day-1 Hull cell and adhesion coupon set (docs/FIRST_LAB_DAY.md) "
            "— same bath, same rectifier, same session; the 316L shim is the "
            "same material as the R3/R4 gravimetric coupons."
        ),
        "coupons": [
            {
                "material": "316L shim stock",
                "thickness_mm": 0.2,
                "role": "thin — maximizes deflection sensitivity, stays in Stoney validity to ~20 µm",
                "n_replicates": 3,
            },
            {
                "material": "316L shim stock",
                "thickness_mm": 0.4,
                "role": "reference geometry for the uncertainty budget; matches FE coupons",
                "n_replicates": 3,
            },
            {
                "material": "Titanium shim (drum surface)",
                "thickness_mm": 0.4,
                "role": "stress on the actual drum candidate (adhesion_peel reference substrate)",
                "n_replicates": 3,
            },
        ],
        "plating_plan": [
            {
                "run": "DC baseline",
                "conditions": (
                    f"{j_mA_cm2:g} mA/cm², 50–{temperature_C:g} °C, bath B0 "
                    "per FIRST_LAB_DAY.md, mask one face, plate 10–100 µm "
                    "(thickness series by time)"
                ),
                "yields": "σ(h) — the thickness sweep resolves the interfacial transient",
            },
            {
                "run": "Waveform comparison",
                "conditions": "DC vs PE vs PRE at equal average j (same thickness)",
                "yields": "σ(waveform) — does pulsing buy stress relief along with FE?",
            },
            {
                "run": "Saccharin ±",
                "conditions": "±2 g/L saccharin in a fresh B0 aliquot",
                "yields": "intrinsic-term relief constant for this bath (calibrates the screening fit)",
            },
        ],
        "measurements": [
            {
                "measurement": "Cantilever deflection δ before/after plating",
                "instrument": (
                    "Dial gauge (10 µm) or stylus profilometer (1 µm) over "
                    f"{COUPON_LENGTH_MM:g} mm gauge length; 3 traces per coupon"
                ),
                "yields": "σ via bent_strip_stress_MPa() / stoney_stress_MPa()",
                "replaces_in_model": (
                    "HOFFMAN_DELTA_M and the whole forward intrinsic estimate"
                ),
            },
            {
                "measurement": "Deposit thickness by mass and cross-section",
                "instrument": "Analytical balance + metallographic mount",
                "yields": "h_f for Stoney; Faradaic cross-check",
                "replaces_in_model": "Faraday-law thickness assumption",
            },
            {
                "measurement": "Diffusible hydrogen (optional on one coupon)",
                "instrument": "Inert-gas fusion analyser (outsourced)",
                "yields": "C_H to separate hydrogen stress from intrinsic stress",
                "replaces_in_model": (
                    "hydrogen_embrittlement.hydrogen_uptake_from_electrolysis "
                    "absorption-fraction screening factor"
                ),
            },
        ],
        "uncertainty_budget": {
            "reference_stress_MPa": float(sigma_ref),
            "reference_thickness_um": float(thickness_um),
            "deflection_at_reference_um": float(delta_um),
            "small_deflection_valid": bool(delta_um / (COUPON_LENGTH_MM * 1e3) < 0.1),
            "u_sigma_dial_gauge_MPa": float(budget["u_sigma_MPa"]),
            "stress_resolution_dial_gauge_MPa": float(res_dial),
            "stress_resolution_profilometer_MPa": float(res_prof),
            "dominant_uncertainty": budget["dominant_uncertainty"],
            "note": (
                "If the reference deflection is below the instrument floor, "
                "thin the shim (δ ∝ h_s⁻²) or lengthen the gauge (δ ∝ L²)."
            ),
        },
        "decision_rules": [
            {
                "if": "measured σ within ±50% of the model decomposition at ≥3 thicknesses",
                "then": (
                    "model confirmed as a screening tool; proceed to use σ(h) "
                    "in the peel window (adhesion_peel) with measured inputs"
                ),
                "class": "confirm",
            },
            {
                "if": "σ > 400 MPa tensile at the 25 µm drum target, or self-delamination observed",
                "then": (
                    "foil branch at risk from the film side; prioritize H "
                    "suppression (waveform, temperature, additives) before "
                    "any drum scale-up"
                ),
                "class": "kill-or-redirect (foil branch)",
            },
            {
                "if": "σ(h) trend or sign disagrees with the mechanism decomposition",
                "then": (
                    "recalibrate the intrinsic and hydrogen terms against the "
                    "measured C_H and grain size before using the peel model"
                ),
                "class": "redirect",
            },
        ],
        "budget_usd": {
            "shim_stock_and_masking": 60,
            "dial_gauge_or_profilometer_access": 120,
            "consumables": 20,
            "total": 200,
            "note": "Shared Day-1 items (rectifier, bath, balance) excluded; hydrogen analysis extra if run.",
        },
        "replaces_in_model_set": [
            "HOFFMAN_DELTA_M (intrinsic term) — via measured σ at known grain size",
            "absorption-fraction screening factor (hydrogen term) — via measured C_H",
            "the forward residual-stress estimate used by adhesion_peel",
            "plastic amplification cross-check for the peel interface (paired with adhesion_peel.coupon_test_protocol)",
        ],
    }


# ═════════════════════════════════════════════════════════════════════
#  8. Scope contract
# ═════════════════════════════════════════════════════════════════════

def model_scope() -> Dict[str, Any]:
    """What this module does and does not compute (house scope contract)."""
    return {
        "computes": [
            "Forward Stoney: stress → curvature → cantilever deflection",
            "Inverse Stoney / bent-strip: deflection → stress",
            "Exact two-layer laminate curvature and the finite-thickness correction",
            "GUM uncertainty budget for the measured stress",
            "Stress decomposition from plating conditions (intrinsic, hydrogen, thermal)",
            "Saccharin and chloride bath corrections (screening estimates)",
            "Stress evolution with thickness (local vs Stoney-averaged)",
            "Peel-driving force G(σ, h) and the drum verdict from bath conditions",
            "The coupon-curvature experiment design with decision rules",
        ],
        "does_not_compute": [
            "Texture (crystallographic) stress contributions",
            "Local stress at defects, pinholes, edge build-up, nodules",
            "Time-resolved in-situ stress transients (only end-state + exponential relaxation)",
            "Additives beyond saccharin/chloride (e.g. organic brighteners, thiourea)",
            "Stress relief by annealing or hydrogen bake-out kinetics",
            "Ductility-limited cracking (see mechanical_properties.py)",
        ],
        "calibration_required": [
            "Measured σ(h) by coupon curvature in the actual bath (the protocol above)",
            "Diffusible H by thermal desorption on the same coupons",
            "Grain size by metallography on the same coupons",
            "Saccharin relief constant for this bath chemistry",
        ],
        "key_uncertainty": (
            "The screening constants: HOFFMAN_DELTA_M, the saccharin relief "
            "fit, the chloride shift, and the absorption-fraction factor in "
            "hydrogen_embrittlement. All four are literature-order estimates, "
            "and all four are replaced by the coupon protocol's measurements."
        ),
        "limitations": (
            "Screening mechanics with literature values and engineering "
            "estimates. No iron internal-stress data exists in this "
            "repository; the protocol is specified so the first measured "
            "number replaces the widest assumption."
        ),
    }
