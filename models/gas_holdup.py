"""Two-phase gas hold-up in the cathode channel: the last unmodelled field.

Closes the gap named in three places at once:

* ``docs/NEXT_STEPS.md`` §3.3 — "measure flow and gas hold-up and calibrate a
  reduced-order model for local boundary-layer thickness, bubble coverage and
  detachment, mixing and dead zones";
* ``docs/REFERENCE_CELL_DESIGN_BASIS.md`` line 42 — the vertical-channel,
  bottom-inlet orientation "deliberately gives generated gas a buoyant escape
  path; **this is a design choice to be observed, not a validated bubble
  model**";
* ``docs/SIM_THEORY_CONFIDENCE.md`` claim 2 — passes only with the caveat
  "gas/flow **partially** modeled".

Before this module the repository's entire bubble physics was
``anode.py:bubble_fraction`` — an empirical saturating coverage θ(j, T) applied
to the **anode only**, as a local surface correction to the ohmic drop.  Three
things were missing, and all three get worse in exactly the direction the
program wants to push (higher j):

1. **No cathodic gas source at all**, even though HER is the program's central
   loss mechanism.  The ``1 − FE`` that ``diffusion_layer_1d.py`` computes was
   never turned into gas volume anywhere.
2. **No axial hold-up profile.**  In a vertical channel with bottom inlet, gas
   accumulates upward, so void fraction — and therefore electrolyte
   resistivity — is a function of height.  The top of the electrode is not the
   same cell as the bottom.  ``scale_up.py``'s Wagner-number treatment cannot
   express this: it assumes a gas-free, uniform-conductivity electrolyte.
3. **No feedback into FE.**  Bubbles do two opposing things — they blanket
   active area and add resistance (bad) and they stir the diffusion layer
   (good).  ``δ`` was a fixed input everywhere in the suite.

What this module computes
-------------------------

* Faradaic gas generation for H₂ (cathode, from the HER current) and O₂
  (anode, from the OER current), as *wet* gas at cell temperature — bubbles
  leave saturated with water vapour, which matters at 60 °C.
* A one-dimensional **drift-flux** void-fraction profile up the channel
  (Zuber–Findlay), closed with a Fritz detachment diameter and a
  Harmathy/Stokes terminal rise velocity.
* **Bruggeman** effective conductivity κ(1−ε)^1.5 and the resulting axial
  resistivity profile and extra ohmic burden.
* A **coupled current redistribution**: the electrodes are equipotential, so
  current leaves the gassy top of the channel for the clear bottom — which
  changes where the gas is made, which changes the hold-up.  Solved as a fixed
  point.
* **Bubble-induced mass transfer** (Stephan–Vogt form) giving an effective
  diffusion-layer thickness δ_eff that is *thinner* than the forced-convection
  value, and which feeds back into the FE engine.
* A **self-consistent segment-wise FE solve**: each height segment runs the
  1-D diffusion-layer model at its own local j and its own δ_eff, and the
  area-averaged FE closes the loop against the gas it generates.
* **Hydrogen safety**: headspace concentration, LFL margin and the dilution
  airflow needed to hold a stated fraction of the lower flammable limit — the
  physics behind ``reference_cell_rc1.yaml``'s flat
  ``maximum_hydrogen_design_rate_L_h`` scalar.

Status: **screening two-phase model, Level 0.**  No gas hold-up, bubble size,
coverage or void-fraction data exists in this repository.  Every correlation
here is transferred from water electrolysis, chlor-alkali or air-water column
practice.  ``measurement_protocol()`` specifies the cheap experiment that
replaces the assumptions, and the design basis's escalation rule stands: go to
two-phase CFD only if the transparent reference cell shows behaviour this
reduced-order model cannot represent.

Sign and orientation convention: ``y`` runs from 0 at the channel inlet
(bottom) to ``height_m`` at the outlet (top); gas flows with +y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from .electrochemistry import FARADAY, R_GAS

# ─── Physical constants and default fluid properties ──────────────────

G_ACCEL = 9.80665            # m/s², standard gravity
P_ATM_PA = 101325.0          # Pa, standard atmosphere

Z_H2 = 2                     # electrons per H₂ (2 H⁺ + 2e⁻ → H₂)
Z_O2 = 4                     # electrons per O₂ (2 H₂O → O₂ + 4 H⁺ + 4e⁻)
M_H2_KG_MOL = 2.016e-3
M_O2_KG_MOL = 31.998e-3

#: Lower flammable limit of hydrogen in air (volume fraction).
LFL_H2_VOL_FRAC = 0.04
#: Upper flammable limit of hydrogen in air (volume fraction).
UFL_H2_VOL_FRAC = 0.75

#: Sulfate bath liquid properties (screening values near 60 °C).
RHO_LIQUID_KG_M3 = 1200.0    # ~1 M FeSO₄ + Na₂SO₄ + H₃BO₃
MU_LIQUID_PA_S = 7.0e-4      # dynamic viscosity at 60 °C
SIGMA_LIQUID_N_M = 0.070     # surface tension, aqueous sulfate

#: Apparent Fritz contact angle for H₂ on a wetted plated-iron cathode (deg).
#:
#: Deliberately small.  Fritz was derived for boiling, where contact angles of
#: 40-50° give millimetre bubbles; electrolytic hydrogen on a well-wetted metal
#: detaches one to two orders of magnitude smaller (tens to a few hundred µm in
#: the water-electrolysis literature).  Using an *apparent* angle of a few
#: degrees reproduces that range while keeping a single physically-structured
#: correlation.  This parameter is the single largest lever on predicted void
#: fraction — halving the diameter cuts the Stokes rise velocity fourfold and
#: roughly quadruples hold-up — which is why ``measurement_protocol`` puts
#: bubble sizing on the list.
CONTACT_ANGLE_H2_DEG = 3.0
#: Apparent Fritz contact angle for O₂ on an oxide anode (degrees).
#: Oxygen bubbles on oxide anodes are consistently observed larger than
#: hydrogen bubbles on metal cathodes at matched current.
CONTACT_ANGLE_O2_DEG = 5.0

#: Drag coefficient used in the crossflow shear-detachment check.
SHEAR_DRAG_COEFFICIENT = 1.0

#: Zuber–Findlay distribution parameter for bubbly flow in a narrow channel.
DISTRIBUTION_PARAMETER_C0 = 1.2

#: Bruggeman exponent for a dispersed non-conducting phase in a conductor.
BRUGGEMAN_EXPONENT = 1.5

#: Screening saturating-coverage parameters for H₂ on the cathode.
#: Same functional form as ``anode.bubble_fraction`` (tanh saturation), with a
#: higher ceiling: cathodic H₂ evolves more bubbles per coulomb than O₂ (two
#: electrons per molecule instead of four) at comparable current.
THETA_MAX_H2 = 0.30
J_CHAR_COVERAGE_MA_CM2 = 150.0

#: Stephan–Vogt bubble-induced mass-transfer correlation coefficients.
VOGT_COEFFICIENT = 0.93
VOGT_RE_EXPONENT = 0.5
VOGT_SC_EXPONENT = 0.487


# ─── Gas generation ───────────────────────────────────────────────────

def water_vapor_pressure_Pa(temperature_C: float) -> float:
    """Saturation vapour pressure of water (Pa) — Buck (1981).

    Bubbles leaving a 60 °C cell are saturated, so ~20 % of the gas volume
    at the vent is water, not hydrogen.  Ignoring this over-reads the
    hydrogen concentration in a headspace safety calculation and under-reads
    the volumetric gas flux driving hold-up.
    """
    T = float(temperature_C)
    return 611.21 * math.exp((18.678 - T / 234.5) * (T / (257.14 + T)))


def faradaic_gas_flow_mol_s(current_A: float, z: int, faradaic_fraction: float = 1.0) -> float:
    """Molar gas generation rate (mol/s) from Faraday's law.

    Parameters
    ----------
    current_A
        Total cell current (A).
    z
        Electrons per gas molecule (2 for H₂, 4 for O₂).
    faradaic_fraction
        Fraction of the current going to this gas.  For cathodic hydrogen
        this is ``1 − FE``.
    """
    if current_A < 0.0:
        raise ValueError("current_A must be non-negative")
    if z <= 0:
        raise ValueError("z must be positive")
    if not 0.0 <= faradaic_fraction <= 1.0:
        raise ValueError("faradaic_fraction must lie in [0, 1]")
    return current_A * faradaic_fraction / (z * FARADAY)


def gas_volumetric_flow_m3_s(
    mol_s: float,
    temperature_C: float = 60.0,
    pressure_Pa: float = P_ATM_PA,
    water_saturated: bool = True,
) -> float:
    """Ideal-gas volumetric flow (m³/s) of a wet gas stream.

    With ``water_saturated=True`` the dry gas is expanded by the water it
    carries: the dry species occupies only ``(P − p_sat)/P`` of the total
    pressure, so the total wet volume is larger by ``P / (P − p_sat)``.
    """
    if mol_s < 0.0:
        raise ValueError("mol_s must be non-negative")
    T = temperature_C + 273.15
    dry = mol_s * R_GAS * T / pressure_Pa
    if not water_saturated:
        return dry
    p_sat = water_vapor_pressure_Pa(temperature_C)
    if p_sat >= pressure_Pa:
        raise ValueError("water saturation pressure exceeds total pressure (boiling)")
    return dry * pressure_Pa / (pressure_Pa - p_sat)


def hydrogen_flow_L_h(
    current_A: float,
    current_efficiency: float,
    temperature_C: float = 60.0,
    pressure_Pa: float = P_ATM_PA,
    water_saturated: bool = True,
) -> float:
    """Cathodic hydrogen flow (L/h) at cell conditions.

    Note this is *wet gas at cell temperature*, which is the quantity that
    matters for hold-up and for vent sizing.  It is deliberately not the same
    normalisation as ``reference_cell_design.hydrogen_rate_L_h``, which
    reports dry gas at 25 °C / 1 atm.
    """
    mol_s = faradaic_gas_flow_mol_s(current_A, Z_H2, 1.0 - current_efficiency)
    m3_s = gas_volumetric_flow_m3_s(mol_s, temperature_C, pressure_Pa, water_saturated)
    return m3_s * 1000.0 * 3600.0


def oxygen_flow_L_h(
    current_A: float,
    oer_fraction: float = 1.0,
    temperature_C: float = 60.0,
    pressure_Pa: float = P_ATM_PA,
    water_saturated: bool = True,
) -> float:
    """Anodic oxygen flow (L/h) at cell conditions."""
    mol_s = faradaic_gas_flow_mol_s(current_A, Z_O2, oer_fraction)
    m3_s = gas_volumetric_flow_m3_s(mol_s, temperature_C, pressure_Pa, water_saturated)
    return m3_s * 1000.0 * 3600.0


# ─── Bubble mechanics ─────────────────────────────────────────────────

def fritz_detachment_diameter_m(
    contact_angle_deg: float = CONTACT_ANGLE_H2_DEG,
    sigma_N_m: float = SIGMA_LIQUID_N_M,
    rho_liquid: float = RHO_LIQUID_KG_M3,
    rho_gas: float = 0.07,
) -> float:
    """Fritz (1935) bubble departure diameter (m).

    ``d = 0.0208 · θ[deg] · sqrt(σ / (g · Δρ))``

    A static force balance between surface tension holding the bubble on the
    electrode and buoyancy pulling it off.  It ignores the shearing action of
    forced flow, which in a real channel detaches bubbles *earlier* and
    smaller — so this is a conservative (large-bubble) estimate.
    """
    if contact_angle_deg <= 0.0:
        raise ValueError("contact_angle_deg must be positive")
    delta_rho = max(rho_liquid - rho_gas, 1e-6)
    return 0.0208 * contact_angle_deg * math.sqrt(sigma_N_m / (G_ACCEL * delta_rho))


def shear_detachment_diameter_m(
    liquid_velocity_m_s: float,
    sigma_N_m: float = SIGMA_LIQUID_N_M,
    rho_liquid: float = RHO_LIQUID_KG_M3,
    drag_coefficient: float = SHEAR_DRAG_COEFFICIENT,
) -> float:
    """Bubble diameter at which crossflow drag overcomes surface-tension pinning (m).

    Balances the drag force on a bubble sitting in the flow,
    ``F_d ≈ C_d (ρ u²/2)(π d²/4)``, against the surface-tension retention
    force ``F_σ ≈ π d σ``, giving

    ``d_shear = 8 σ / (C_d ρ u²)``.

    Forced flow strips bubbles off *before* buoyancy would, so the operative
    departure diameter is the smaller of the Fritz (quiescent) and shear
    values.  At RC-1's 0.07 m/s this is a weak constraint; in the
    high-velocity channels a scaled cell would need, it dominates — which is
    itself a useful design result, because smaller bubbles mean lower hold-up.
    """
    if liquid_velocity_m_s <= 0.0:
        return math.inf
    if drag_coefficient <= 0.0:
        raise ValueError("drag_coefficient must be positive")
    return float(8.0 * sigma_N_m / (drag_coefficient * rho_liquid * liquid_velocity_m_s ** 2))


def departure_diameter_m(
    liquid_velocity_m_s: float = 0.0,
    contact_angle_deg: float = CONTACT_ANGLE_H2_DEG,
    sigma_N_m: float = SIGMA_LIQUID_N_M,
    rho_liquid: float = RHO_LIQUID_KG_M3,
    rho_gas: float = 0.07,
) -> float:
    """Operative bubble departure diameter (m): the smaller of Fritz and shear."""
    d_fritz = fritz_detachment_diameter_m(contact_angle_deg, sigma_N_m, rho_liquid, rho_gas)
    d_shear = shear_detachment_diameter_m(liquid_velocity_m_s, sigma_N_m, rho_liquid)
    return float(min(d_fritz, d_shear))


def terminal_rise_velocity_m_s(
    d_bubble_m: float,
    rho_liquid: float = RHO_LIQUID_KG_M3,
    rho_gas: float = 0.07,
    mu_liquid: float = MU_LIQUID_PA_S,
    sigma_N_m: float = SIGMA_LIQUID_N_M,
) -> float:
    """Terminal rise velocity of an isolated bubble (m/s).

    Takes the smaller of two limiting regimes, which is the standard way to
    span the transition without a piecewise Reynolds criterion:

    * **Stokes** (small, spherical, viscosity-controlled):
      ``u = g Δρ d² / (18 μ)``
    * **Harmathy** (larger, distorted, surface-tension-controlled):
      ``u = 1.53 (σ g Δρ / ρ_l²)^0.25``, which is diameter-independent.

    For the ~0.2 mm hydrogen bubbles this cell produces, Stokes is the
    binding branch at small sizes and Harmathy caps the velocity above
    roughly 1 mm.
    """
    if d_bubble_m <= 0.0:
        raise ValueError("d_bubble_m must be positive")
    delta_rho = max(rho_liquid - rho_gas, 1e-6)
    u_stokes = G_ACCEL * delta_rho * d_bubble_m ** 2 / (18.0 * mu_liquid)
    u_harmathy = 1.53 * (sigma_N_m * G_ACCEL * delta_rho / rho_liquid ** 2) ** 0.25
    return float(min(u_stokes, u_harmathy))


def drift_flux_void_fraction(
    j_gas_m_s: float,
    j_liquid_m_s: float,
    u_rise_m_s: float,
    C0: float = DISTRIBUTION_PARAMETER_C0,
) -> float:
    """Zuber–Findlay drift-flux void fraction (dimensionless).

    ``ε = j_g / (C₀ (j_g + j_l) + u_∞)``

    The gas moves faster than the mixture for two reasons: it is concentrated
    in the fast-moving channel core (the distribution parameter ``C₀ > 1``)
    and it buoys relative to the liquid (the drift velocity ``u_∞``).  Both
    reduce the void fraction below the no-slip homogeneous value
    ``j_g/(j_g+j_l)``, which is why a homogeneous model over-predicts the
    ohmic penalty.
    """
    if j_gas_m_s < 0.0:
        raise ValueError("j_gas_m_s must be non-negative")
    if j_liquid_m_s < 0.0:
        raise ValueError("j_liquid_m_s must be non-negative")
    if C0 <= 0.0:
        raise ValueError("C0 must be positive")
    denom = C0 * (j_gas_m_s + j_liquid_m_s) + u_rise_m_s
    if denom <= 0.0:
        return 0.0
    return float(min(j_gas_m_s / denom, 0.999))


def bruggeman_conductivity(kappa_S_m: float, void_fraction: float,
                           exponent: float = BRUGGEMAN_EXPONENT) -> float:
    """Effective conductivity of a bubbly electrolyte (S/m).

    ``κ_eff = κ (1 − ε)^1.5`` — Bruggeman's relation for a dispersion of
    non-conducting spheres.  The 1.5 exponent is the theoretical value for
    dilute spheres; measured electrolysis cells often sit between 1.5 and 2.5
    because bubbles flatten against the electrode and cluster.
    """
    if kappa_S_m <= 0.0:
        raise ValueError("kappa_S_m must be positive")
    eps = min(max(void_fraction, 0.0), 0.999)
    return float(kappa_S_m * (1.0 - eps) ** exponent)


def surface_coverage_fraction(
    j_gas_mA_cm2: float,
    theta_max: float = THETA_MAX_H2,
    j_char_mA_cm2: float = J_CHAR_COVERAGE_MA_CM2,
) -> float:
    """Fraction of electrode area blanketed by adhering bubbles.

    ``θ = θ_max · tanh(j_gas / j_char)`` — the same saturating form used by
    ``anode.bubble_fraction``, kept deliberately consistent so the cathode and
    anode treatments are comparable.  Driven by the **gas-producing** current
    density, not the total: a cell at 95 % FE has almost no cathodic coverage
    even at high j, which is a real and favourable coupling.

    Screening correlation.  Vogt's reviews put steady-state coverage on
    gas-evolving electrodes anywhere from a few percent to above 0.5
    depending on wettability and flow; that spread is the dominant
    uncertainty in this module and is what ``measurement_protocol`` targets.
    """
    if theta_max < 0.0 or theta_max >= 1.0:
        raise ValueError("theta_max must lie in [0, 1)")
    if j_char_mA_cm2 <= 0.0:
        raise ValueError("j_char_mA_cm2 must be positive")
    j = max(float(j_gas_mA_cm2), 0.0)
    return float(theta_max * math.tanh(j / j_char_mA_cm2))


def vogt_mass_transfer_coefficient_m_s(
    j_gas_superficial_m_s: float,
    d_bubble_m: float,
    diffusivity_m2_s: float,
    nu_m2_s: float,
) -> float:
    """Bubble-induced mass-transfer coefficient (m/s), Stephan–Vogt form.

    ``Sh = 0.93 Re_g^0.5 Sc^0.487``  with  ``Re_g = v_g d_b / ν``

    where ``v_g`` is the volumetric gas flux per unit electrode area.  The
    physical picture is microconvection: each departing bubble drags fresh
    electrolyte into the space it vacated, so a gassing electrode stirs its
    own diffusion layer.  This is the mechanism that makes high-rate
    electrolysis cells less transport-limited than their nominal flow
    suggests, and it is the reason bubbles are not purely a penalty.
    """
    if d_bubble_m <= 0.0 or diffusivity_m2_s <= 0.0 or nu_m2_s <= 0.0:
        raise ValueError("d_bubble_m, diffusivity and nu must be positive")
    v_g = max(float(j_gas_superficial_m_s), 0.0)
    if v_g <= 0.0:
        return 0.0
    Re = v_g * d_bubble_m / nu_m2_s
    Sc = nu_m2_s / diffusivity_m2_s
    Sh = VOGT_COEFFICIENT * Re ** VOGT_RE_EXPONENT * Sc ** VOGT_SC_EXPONENT
    return float(Sh * diffusivity_m2_s / d_bubble_m)


def combined_boundary_layer_m(
    delta_forced_m: float,
    k_bubble_m_s: float,
    diffusivity_m2_s: float,
) -> float:
    """Effective diffusion-layer thickness combining forced and bubble convection.

    Mass-transfer coefficients from independent mechanisms are superposed in
    quadrature (``k = sqrt(k_f² + k_b²)``), the usual engineering closure for
    mixed convection, and converted back to a Nernst thickness
    ``δ_eff = D / k``.  The result is always thinner than the forced-flow
    value, and never thinner than the bubble-only value.
    """
    if delta_forced_m <= 0.0:
        raise ValueError("delta_forced_m must be positive")
    if diffusivity_m2_s <= 0.0:
        raise ValueError("diffusivity_m2_s must be positive")
    k_forced = diffusivity_m2_s / delta_forced_m
    k_total = math.hypot(k_forced, max(k_bubble_m_s, 0.0))
    return float(diffusivity_m2_s / k_total)


# ─── Channel description ──────────────────────────────────────────────

@dataclass
class ChannelGeometry:
    """Vertical electrode channel — the RC-1 configuration by default.

    Defaults mirror ``processes/reference_cell_rc1.yaml``: a 50 mm tall,
    20 mm wide electrode in a 3 mm deep channel, bottom inlet, top outlet.
    """

    height_m: float = 0.050            # electrode height, flow direction
    width_m: float = 0.020             # electrode width
    depth_m: float = 0.003             # channel depth (also the gas rise path)
    interelectrode_gap_m: float = 0.020
    liquid_flow_L_min: float = 0.25

    def __post_init__(self) -> None:
        for name in ("height_m", "width_m", "depth_m", "interelectrode_gap_m"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.liquid_flow_L_min < 0.0:
            raise ValueError("liquid_flow_L_min must be non-negative")

    @property
    def electrode_area_m2(self) -> float:
        return self.height_m * self.width_m

    @property
    def electrode_area_cm2(self) -> float:
        return self.electrode_area_m2 * 1e4

    @property
    def cross_section_m2(self) -> float:
        """Channel flow cross-section (m²)."""
        return self.width_m * self.depth_m

    @property
    def hydraulic_diameter_m(self) -> float:
        return 2.0 * self.width_m * self.depth_m / (self.width_m + self.depth_m)

    @property
    def superficial_liquid_velocity_m_s(self) -> float:
        """Liquid superficial velocity up the channel (m/s)."""
        q_m3_s = self.liquid_flow_L_min / 1000.0 / 60.0
        return q_m3_s / self.cross_section_m2


@dataclass
class HoldupProfile:
    """Axial two-phase state of one electrode channel."""

    y_m: np.ndarray                       # segment midpoint heights
    j_mA_cm2: np.ndarray                  # local current density
    gas_fraction_of_current: np.ndarray   # local gas-producing current fraction
    superficial_gas_velocity_m_s: np.ndarray
    void_fraction: np.ndarray
    kappa_eff_S_m: np.ndarray
    surface_coverage: np.ndarray
    delta_eff_m: np.ndarray
    bubble_diameter_m: float
    rise_velocity_m_s: float
    kappa_bulk_S_m: float
    geometry: ChannelGeometry = field(repr=False)

    @property
    def outlet_void_fraction(self) -> float:
        return float(self.void_fraction[-1])

    @property
    def mean_void_fraction(self) -> float:
        return float(np.mean(self.void_fraction))

    @property
    def conductivity_penalty(self) -> float:
        """Ratio of gas-free to mean effective conductivity (≥ 1)."""
        return float(self.kappa_bulk_S_m / np.mean(self.kappa_eff_S_m))

    @property
    def current_uniformity(self) -> float:
        """min(j)/max(j) over the electrode height (1 = perfectly uniform)."""
        return float(np.min(self.j_mA_cm2) / np.max(self.j_mA_cm2))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "y_mm": [round(v * 1000.0, 3) for v in self.y_m],
            "j_mA_cm2": [round(float(v), 3) for v in self.j_mA_cm2],
            "void_fraction": [round(float(v), 5) for v in self.void_fraction],
            "kappa_eff_S_m": [round(float(v), 4) for v in self.kappa_eff_S_m],
            "surface_coverage": [round(float(v), 5) for v in self.surface_coverage],
            "delta_eff_um": [round(float(v) * 1e6, 3) for v in self.delta_eff_m],
            "bubble_diameter_um": round(self.bubble_diameter_m * 1e6, 2),
            "rise_velocity_mm_s": round(self.rise_velocity_m_s * 1000.0, 3),
            "outlet_void_fraction": round(self.outlet_void_fraction, 5),
            "mean_void_fraction": round(self.mean_void_fraction, 5),
            "conductivity_penalty": round(self.conductivity_penalty, 4),
            "current_uniformity": round(self.current_uniformity, 4),
        }


# ─── Hold-up profile at fixed current distribution ────────────────────

def holdup_profile(
    j_mA_cm2: Sequence[float] | float,
    gas_current_fraction: Sequence[float] | float,
    geometry: ChannelGeometry = ChannelGeometry(),
    temperature_C: float = 60.0,
    kappa_S_m: float = 13.5,
    pressure_Pa: float = P_ATM_PA,
    z_gas: int = Z_H2,
    contact_angle_deg: float = CONTACT_ANGLE_H2_DEG,
    delta_forced_m: float = 50.0e-6,
    diffusivity_m2_s: float = 7.2e-10,
    n_segments: int = 12,
    theta_max: float = THETA_MAX_H2,
    water_saturated: bool = True,
) -> HoldupProfile:
    """Integrate void fraction up the channel for a given current distribution.

    Gas generated below a given height has to pass through it, so the
    superficial gas velocity accumulates:

    ``v_g(y) = (1/A_c) ∫₀^y  q_gas'(y') dy'``

    with ``q_gas'`` the volumetric gas production per unit height.  Void
    fraction then follows pointwise from the drift-flux closure, and the
    Bruggeman relation converts it to a local conductivity.

    ``j_mA_cm2`` and ``gas_current_fraction`` may each be a scalar (uniform)
    or a per-segment array.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be at least 1")

    j_arr = np.full(n_segments, float(j_mA_cm2)) if np.isscalar(j_mA_cm2) \
        else np.asarray(j_mA_cm2, dtype=float)
    f_arr = np.full(n_segments, float(gas_current_fraction)) if np.isscalar(gas_current_fraction) \
        else np.asarray(gas_current_fraction, dtype=float)
    if j_arr.size != n_segments or f_arr.size != n_segments:
        raise ValueError("j and gas_current_fraction must be scalar or length n_segments")
    if np.any(j_arr < 0.0):
        raise ValueError("current density must be non-negative")
    if np.any(f_arr < 0.0) or np.any(f_arr > 1.0):
        raise ValueError("gas_current_fraction must lie in [0, 1]")

    dy = geometry.height_m / n_segments
    y_mid = (np.arange(n_segments) + 0.5) * dy
    seg_area_m2 = dy * geometry.width_m

    # Per-segment gas volumetric production (m³/s)
    seg_current_A = j_arr * 10.0 * seg_area_m2           # mA/cm² → A/m² → A
    seg_gas_m3_s = np.array([
        gas_volumetric_flow_m3_s(
            faradaic_gas_flow_mol_s(float(I), z_gas, float(f)),
            temperature_C, pressure_Pa, water_saturated,
        )
        for I, f in zip(seg_current_A, f_arr)
    ])

    # Cumulative gas passing each segment midpoint: everything below, plus
    # half of the local segment's own production.
    cumulative = np.cumsum(seg_gas_m3_s) - 0.5 * seg_gas_m3_s
    v_gas = cumulative / geometry.cross_section_m2

    v_liq = geometry.superficial_liquid_velocity_m_s
    d_bubble = departure_diameter_m(
        liquid_velocity_m_s=v_liq, contact_angle_deg=contact_angle_deg
    )
    u_rise = terminal_rise_velocity_m_s(d_bubble)

    eps = np.array([drift_flux_void_fraction(float(vg), v_liq, u_rise) for vg in v_gas])
    kappa_eff = np.array([bruggeman_conductivity(kappa_S_m, float(e)) for e in eps])

    # Surface coverage responds to the *local* gas-producing current density.
    j_gas_local = j_arr * f_arr
    theta = np.array([surface_coverage_fraction(float(jg), theta_max=theta_max) for jg in j_gas_local])

    # Bubble microconvection thins the diffusion layer.
    nu = MU_LIQUID_PA_S / RHO_LIQUID_KG_M3
    # Local gas flux per unit electrode area (not the accumulated channel flux):
    # microconvection is driven by locally departing bubbles.
    v_gas_local = seg_gas_m3_s / seg_area_m2
    k_bubble = np.array([
        vogt_mass_transfer_coefficient_m_s(float(v), d_bubble, diffusivity_m2_s, nu)
        for v in v_gas_local
    ])
    delta_eff = np.array([
        combined_boundary_layer_m(delta_forced_m, float(k), diffusivity_m2_s)
        for k in k_bubble
    ])

    return HoldupProfile(
        y_m=y_mid,
        j_mA_cm2=j_arr,
        gas_fraction_of_current=f_arr,
        superficial_gas_velocity_m_s=v_gas,
        void_fraction=eps,
        kappa_eff_S_m=kappa_eff,
        surface_coverage=theta,
        delta_eff_m=delta_eff,
        bubble_diameter_m=d_bubble,
        rise_velocity_m_s=u_rise,
        kappa_bulk_S_m=kappa_S_m,
        geometry=geometry,
    )


# ─── Current redistribution ───────────────────────────────────────────

def solve_current_distribution(
    j_mean_mA_cm2: float,
    kappa_eff_S_m: Sequence[float],
    geometry: ChannelGeometry = ChannelGeometry(),
    tafel_slope_V: float = 0.120,
    j_ref_mA_cm2: float = 1.0,
    surface_coverage: Optional[Sequence[float]] = None,
    extra_area_resistance_ohm_m2: float = 8.0e-4,
) -> np.ndarray:
    """Redistribute current over height given an axial conductivity profile.

    The electrodes are metal, hence equipotential, so every height segment
    sits at the same total driving voltage:

    ``V = η_act(j_i) + j_i · (L/κ_eff,i + R_other)``

    with a Tafel activation term ``η = b·log₁₀(j/j_ref)`` and the segment's own
    bubbly ohmic path.  Gassy segments are more resistive, so they take less
    current; the current they give up appears in the clear segments below.
    Bubble surface coverage additionally shrinks the active area, raising the
    *true* local current density on the metal that remains — this is folded
    into the activation term.

    Returns the per-segment nominal (geometric) current density, normalised so
    the area average equals ``j_mean_mA_cm2``.
    """
    kappa = np.asarray(kappa_eff_S_m, dtype=float)
    if kappa.size == 0:
        raise ValueError("kappa_eff_S_m must be non-empty")
    if np.any(kappa <= 0.0):
        raise ValueError("kappa_eff_S_m must be positive")
    if j_mean_mA_cm2 <= 0.0:
        raise ValueError("j_mean_mA_cm2 must be positive")
    if tafel_slope_V <= 0.0:
        raise ValueError("tafel_slope_V must be positive")

    n = kappa.size
    theta = np.zeros(n) if surface_coverage is None else np.asarray(surface_coverage, dtype=float)
    if theta.size != n:
        raise ValueError("surface_coverage must match kappa_eff_S_m length")
    active = np.clip(1.0 - theta, 0.02, 1.0)

    R_ohm = geometry.interelectrode_gap_m / kappa + extra_area_resistance_ohm_m2  # Ω·m²

    def j_at_V(V: float, i: int) -> float:
        """Local geometric current density (A/m²) at driving voltage V."""
        def residual(j: float) -> float:
            j_true = j / active[i]
            eta = tafel_slope_V * math.log10(max(j_true, 1e-12) / (j_ref_mA_cm2 * 10.0))
            return eta + j * R_ohm[i] - V
        lo, hi = 1e-9, 1e6
        if residual(lo) > 0.0:
            return lo
        if residual(hi) < 0.0:
            return hi
        return float(brentq(residual, lo, hi, xtol=1e-10, rtol=1e-10))

    target_A_m2 = j_mean_mA_cm2 * 10.0

    def mean_residual(V: float) -> float:
        return float(np.mean([j_at_V(V, i) for i in range(n)]) - target_A_m2)

    V_lo, V_hi = -5.0, 50.0
    if mean_residual(V_lo) > 0.0:
        V_lo = -50.0
    if mean_residual(V_hi) < 0.0:
        V_hi = 500.0
    V_star = float(brentq(mean_residual, V_lo, V_hi, xtol=1e-10, rtol=1e-10))

    j_local = np.array([j_at_V(V_star, i) for i in range(n)]) / 10.0  # → mA/cm²
    # Renormalise against residual solver tolerance so the ledger closes exactly.
    j_local *= j_mean_mA_cm2 / float(np.mean(j_local))
    return j_local


# ─── Coupled solve ────────────────────────────────────────────────────

@dataclass
class CoupledGasResult:
    """Self-consistent two-phase operating point for one channel."""

    j_mean_mA_cm2: float
    profile: HoldupProfile
    segment_FE: np.ndarray
    area_average_FE: float
    FE_no_bubbles: float
    delta_forced_m: float
    ohmic_penalty_V: float
    ohmic_gas_free_V: float
    hydrogen_flow_L_h: float
    converged: bool
    iterations: int

    @property
    def FE_shift(self) -> float:
        """Change in area-average FE attributable to bubbles (percentage points)."""
        return (self.area_average_FE - self.FE_no_bubbles) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "j_mean_mA_cm2": round(self.j_mean_mA_cm2, 3),
            "area_average_FE": round(self.area_average_FE, 5),
            "FE_no_bubbles": round(self.FE_no_bubbles, 5),
            "FE_shift_percentage_points": round(self.FE_shift, 4),
            "segment_FE": [round(float(v), 5) for v in self.segment_FE],
            "delta_forced_um": round(self.delta_forced_m * 1e6, 3),
            "ohmic_penalty_V": round(self.ohmic_penalty_V, 5),
            "ohmic_gas_free_V": round(self.ohmic_gas_free_V, 5),
            "hydrogen_flow_L_h_wet_at_T": round(self.hydrogen_flow_L_h, 4),
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "profile": self.profile.to_dict(),
        }


def _default_fe_model(
    j_mA_cm2: float,
    delta_m: float,
    temperature_C: float,
    fe_conc_M: float,
    pH_bulk: float,
) -> float:
    """FE from the repository's gating 1-D diffusion-layer engine."""
    from .diffusion_layer_1d import DiffusionLayer1D

    model = DiffusionLayer1D(
        fe_conc_M=fe_conc_M,
        pH_bulk=pH_bulk,
        temperature_C=temperature_C,
        delta_m=delta_m,
        fast_mode=True,
    )
    return float(model.solve(j_mA_cm2).current_efficiency)


def solve_coupled(
    j_mean_mA_cm2: float = 100.0,
    geometry: ChannelGeometry = ChannelGeometry(),
    temperature_C: float = 60.0,
    kappa_S_m: float = 13.5,
    delta_forced_m: float = 50.0e-6,
    fe_conc_M: float = 1.0,
    pH_bulk: float = 2.0,
    n_segments: int = 6,
    max_iterations: int = 8,
    tol: float = 2.0e-3,
    fe_model: Optional[Callable[..., float]] = None,
    relaxation: float = 0.5,
) -> CoupledGasResult:
    """Fixed-point solve of the gas ↔ current ↔ FE loop.

    The circular dependency this resolves:

    ``FE  →  H₂ generation  →  void fraction ε(y)  →  κ_eff(y)  →  j(y)``
    and simultaneously
    ``gas flux  →  bubble microconvection  →  δ_eff(y)  →  FE(y)``

    Iterated under relaxation until the segment FE vector stops moving.  The
    ``fe_model`` hook takes ``(j_mA_cm2, delta_m, temperature_C, fe_conc_M,
    pH_bulk)`` and returns Faradaic efficiency; it defaults to
    ``diffusion_layer_1d.DiffusionLayer1D`` in fast mode, and is injectable so
    tests and sweeps can substitute a cheap surrogate.
    """
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0, 1]")
    if n_segments < 1:
        raise ValueError("n_segments must be at least 1")

    model = fe_model if fe_model is not None else _default_fe_model

    # Baseline: no bubbles anywhere — uniform current, forced-convection δ.
    fe_flat = float(model(j_mean_mA_cm2, delta_forced_m, temperature_C, fe_conc_M, pH_bulk))

    fe_seg = np.full(n_segments, fe_flat)
    j_seg = np.full(n_segments, float(j_mean_mA_cm2))
    profile = None
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        profile = holdup_profile(
            j_mA_cm2=j_seg,
            gas_current_fraction=1.0 - fe_seg,
            geometry=geometry,
            temperature_C=temperature_C,
            kappa_S_m=kappa_S_m,
            delta_forced_m=delta_forced_m,
            n_segments=n_segments,
        )
        j_new = solve_current_distribution(
            j_mean_mA_cm2=j_mean_mA_cm2,
            kappa_eff_S_m=profile.kappa_eff_S_m,
            geometry=geometry,
            surface_coverage=profile.surface_coverage,
        )
        fe_new = np.array([
            float(model(float(j), float(d), temperature_C, fe_conc_M, pH_bulk))
            for j, d in zip(j_new, profile.delta_eff_m)
        ])

        delta_fe = float(np.max(np.abs(fe_new - fe_seg)))
        delta_j = float(np.max(np.abs(j_new - j_seg)) / j_mean_mA_cm2)

        fe_seg = (1.0 - relaxation) * fe_seg + relaxation * fe_new
        j_seg = (1.0 - relaxation) * j_seg + relaxation * j_new

        # Require at least two passes: the first only measures the distance
        # from the bubble-free initial guess, which is not a convergence test.
        if iterations >= 2 and delta_fe < tol and delta_j < tol:
            converged = True
            break

    assert profile is not None
    # Refresh the profile so the returned state matches the converged vectors.
    profile = holdup_profile(
        j_mA_cm2=j_seg,
        gas_current_fraction=1.0 - fe_seg,
        geometry=geometry,
        temperature_C=temperature_C,
        kappa_S_m=kappa_S_m,
        delta_forced_m=delta_forced_m,
        n_segments=n_segments,
    )

    fe_avg = float(np.mean(fe_seg * j_seg) / np.mean(j_seg))

    # Ohmic burden: area-average of j·L/κ, with and without gas.
    j_A_m2 = j_seg * 10.0
    ohmic_gas = float(np.mean(j_A_m2 * geometry.interelectrode_gap_m / profile.kappa_eff_S_m))
    ohmic_free = float(j_mean_mA_cm2 * 10.0 * geometry.interelectrode_gap_m / kappa_S_m)

    current_A = j_mean_mA_cm2 * 10.0 * geometry.electrode_area_m2
    h2_L_h = hydrogen_flow_L_h(current_A, fe_avg, temperature_C)

    return CoupledGasResult(
        j_mean_mA_cm2=float(j_mean_mA_cm2),
        profile=profile,
        segment_FE=fe_seg,
        area_average_FE=fe_avg,
        FE_no_bubbles=fe_flat,
        delta_forced_m=delta_forced_m,
        ohmic_penalty_V=ohmic_gas - ohmic_free,
        ohmic_gas_free_V=ohmic_free,
        hydrogen_flow_L_h=h2_L_h,
        converged=converged,
        iterations=iterations,
    )


# ─── Hydrogen safety ──────────────────────────────────────────────────

@dataclass
class HydrogenSafetyResult:
    """Headspace/vent hydrogen assessment for one cell."""

    hydrogen_flow_L_h: float
    dilution_flow_L_h: float
    hydrogen_vol_fraction: float
    fraction_of_LFL: float
    target_fraction_of_LFL: float
    required_dilution_flow_L_h: float
    flammable: bool
    acceptable: bool
    headspace_L: Optional[float]
    time_to_LFL_min: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hydrogen_flow_L_h": round(self.hydrogen_flow_L_h, 4),
            "dilution_flow_L_h": round(self.dilution_flow_L_h, 2),
            "hydrogen_vol_percent": round(self.hydrogen_vol_fraction * 100.0, 4),
            "fraction_of_LFL": round(self.fraction_of_LFL, 4),
            "target_fraction_of_LFL": self.target_fraction_of_LFL,
            "required_dilution_flow_L_h": round(self.required_dilution_flow_L_h, 2),
            "flammable_mixture": bool(self.flammable),
            "acceptable": bool(self.acceptable),
            "headspace_L": self.headspace_L,
            "time_to_LFL_min_unventilated": (
                None if self.time_to_LFL_min is None else round(self.time_to_LFL_min, 2)
            ),
        }


def hydrogen_safety(
    current_A: float,
    current_efficiency: float,
    dilution_flow_L_h: float = 0.0,
    temperature_C: float = 60.0,
    target_fraction_of_LFL: float = 0.25,
    headspace_L: Optional[float] = None,
    water_saturated: bool = True,
) -> HydrogenSafetyResult:
    """Headspace hydrogen concentration, LFL margin and required dilution.

    Puts physics behind ``reference_cell_rc1.yaml``'s flat
    ``maximum_hydrogen_design_rate_L_h`` scalar.  Hydrogen is flammable in air
    from 4 vol % to 75 vol %; the conventional design target for a vented
    enclosure is 25 % of the lower limit, i.e. 1 vol %.

    Dilution requirement follows from a steady mixed-headspace balance:
    ``x = Q_H2 / (Q_H2 + Q_air)``, so holding ``x ≤ x_target`` needs
    ``Q_air ≥ Q_H2 (1/x_target − 1)``.

    ``time_to_LFL_min`` is the *unventilated* accumulation time in a sealed
    headspace of the given volume — the number that matters if the extract
    fan fails, and the reason the design basis says never use a closed gas
    path.
    """
    if not 0.0 < target_fraction_of_LFL <= 1.0:
        raise ValueError("target_fraction_of_LFL must lie in (0, 1]")
    if dilution_flow_L_h < 0.0:
        raise ValueError("dilution_flow_L_h must be non-negative")

    q_h2 = hydrogen_flow_L_h(current_A, current_efficiency, temperature_C,
                             water_saturated=water_saturated)
    total = q_h2 + dilution_flow_L_h
    x = 0.0 if total <= 0.0 else q_h2 / total

    x_target = target_fraction_of_LFL * LFL_H2_VOL_FRAC
    required = q_h2 * (1.0 / x_target - 1.0) if q_h2 > 0.0 else 0.0

    t_to_lfl = None
    if headspace_L is not None:
        if headspace_L <= 0.0:
            raise ValueError("headspace_L must be positive")
        if q_h2 > 0.0:
            t_to_lfl = (LFL_H2_VOL_FRAC * headspace_L) / q_h2 * 60.0

    return HydrogenSafetyResult(
        hydrogen_flow_L_h=q_h2,
        dilution_flow_L_h=dilution_flow_L_h,
        hydrogen_vol_fraction=x,
        fraction_of_LFL=x / LFL_H2_VOL_FRAC,
        target_fraction_of_LFL=target_fraction_of_LFL,
        required_dilution_flow_L_h=required,
        flammable=LFL_H2_VOL_FRAC <= x <= UFL_H2_VOL_FRAC,
        acceptable=x <= x_target,
        headspace_L=headspace_L,
        time_to_LFL_min=t_to_lfl,
    )


# ─── Screening sweeps ─────────────────────────────────────────────────

def current_density_sweep(
    j_values_mA_cm2: Sequence[float] = (50.0, 100.0, 200.0, 300.0, 400.0),
    current_efficiency: float = 0.85,
    geometry: ChannelGeometry = ChannelGeometry(),
    temperature_C: float = 60.0,
    kappa_S_m: float = 13.5,
    n_segments: int = 12,
) -> List[Dict[str, Any]]:
    """Uncoupled hold-up screen across current density at fixed FE.

    Cheap: no FE engine calls.  Shows how void fraction, conductivity penalty
    and current non-uniformity grow toward the kill-criterion current density
    of 300 mA/cm².
    """
    rows = []
    for j in j_values_mA_cm2:
        prof = holdup_profile(
            j_mA_cm2=float(j),
            gas_current_fraction=1.0 - current_efficiency,
            geometry=geometry,
            temperature_C=temperature_C,
            kappa_S_m=kappa_S_m,
            n_segments=n_segments,
        )
        j_redist = solve_current_distribution(
            j_mean_mA_cm2=float(j),
            kappa_eff_S_m=prof.kappa_eff_S_m,
            geometry=geometry,
            surface_coverage=prof.surface_coverage,
        )
        rows.append({
            "j_mA_cm2": float(j),
            "current_efficiency": current_efficiency,
            "outlet_void_fraction": prof.outlet_void_fraction,
            "mean_void_fraction": prof.mean_void_fraction,
            "conductivity_penalty": prof.conductivity_penalty,
            "surface_coverage_max": float(np.max(prof.surface_coverage)),
            "delta_eff_min_um": float(np.min(prof.delta_eff_m) * 1e6),
            "current_uniformity": float(np.min(j_redist) / np.max(j_redist)),
            "j_top_mA_cm2": float(j_redist[-1]),
            "j_bottom_mA_cm2": float(j_redist[0]),
        })
    return rows


def height_scaling_screen(
    heights_m: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 1.00),
    j_mA_cm2: float = 300.0,
    current_efficiency: float = 0.85,
    base_geometry: ChannelGeometry = ChannelGeometry(),
    temperature_C: float = 60.0,
    kappa_S_m: float = 13.5,
    n_segments: int = 16,
    uniformity_floor: float = 0.90,
) -> List[Dict[str, Any]]:
    """How far can the electrode grow before gas hold-up breaks uniformity?

    This is the scale-up question the reference cell cannot answer by itself:
    RC-1's 50 mm electrode is short enough that hold-up is nearly invisible,
    while a tankhouse-scale 1 m plate accumulates gas over twenty times the
    path.  Liquid flow is scaled with height to hold the same superficial
    velocity, so the only thing changing is the gas accumulation path.
    """
    rows = []
    for h in heights_m:
        scale = h / base_geometry.height_m
        geom = ChannelGeometry(
            height_m=float(h),
            width_m=base_geometry.width_m,
            depth_m=base_geometry.depth_m,
            interelectrode_gap_m=base_geometry.interelectrode_gap_m,
            liquid_flow_L_min=base_geometry.liquid_flow_L_min,
        )
        prof = holdup_profile(
            j_mA_cm2=j_mA_cm2,
            gas_current_fraction=1.0 - current_efficiency,
            geometry=geom,
            temperature_C=temperature_C,
            kappa_S_m=kappa_S_m,
            n_segments=n_segments,
        )
        j_redist = solve_current_distribution(
            j_mean_mA_cm2=j_mA_cm2,
            kappa_eff_S_m=prof.kappa_eff_S_m,
            geometry=geom,
            surface_coverage=prof.surface_coverage,
        )
        uni = float(np.min(j_redist) / np.max(j_redist))
        rows.append({
            "height_mm": round(h * 1000.0, 1),
            "area_scale_vs_RC1": round(scale, 2),
            "outlet_void_fraction": prof.outlet_void_fraction,
            "conductivity_penalty": prof.conductivity_penalty,
            "current_uniformity": uni,
            "passes_uniformity_floor": uni >= uniformity_floor,
        })
    return rows


# ─── The experiment that replaces the assumptions ─────────────────────

def measurement_protocol() -> Dict[str, Any]:
    """Tier-0 gas hold-up measurement: what to measure, and what it decides.

    Deliberately cheap.  Every quantity here is obtainable from the
    transparent reference cell already specified in
    ``docs/REFERENCE_CELL_DESIGN_BASIS.md`` plus a graduated cylinder, a
    manometer leg and a phone camera — no PIV, no tomography.
    """
    return {
        "title": "RC-1 gas hold-up and bubble characterisation",
        "objective": (
            "Replace the four transferred correlations in gas_holdup.py "
            "(detachment diameter, distribution parameter C0, Bruggeman "
            "exponent, surface coverage θ_max) with measured values for the "
            "actual sulfate bath and cathode surface."
        ),
        "estimated_cost_usd": 450,
        "estimated_duration_days": 3,
        "prerequisite": "RC-1 water flow test and transparent channel window",
        "measurements": [
            {
                "quantity": "Volumetric gas rate at the cathode vent",
                "method": "Inverted graduated cylinder / water displacement, "
                          "temperature and barometric pressure recorded",
                "resolution_required": "±2 % of flow",
                "calibrates": "Faradaic gas closure and, with the charge "
                              "ledger, an independent check on FE",
            },
            {
                "quantity": "Void fraction, channel-averaged",
                "method": "Level swell / manometric: liquid level rise in the "
                          "riser with current on versus off, at fixed flow",
                "resolution_required": "±0.5 mm level, giving ε to ~±0.01",
                "calibrates": "Distribution parameter C0 in the drift-flux closure",
            },
            {
                "quantity": "Bubble departure diameter and coverage",
                "method": "Backlit high-frame-rate phone video through the "
                          "channel window; frame-count sizing on 200+ bubbles",
                "resolution_required": "≥240 fps, ≥20 px per bubble diameter",
                "calibrates": "Fritz contact angle and θ_max in surface_coverage_fraction",
            },
            {
                "quantity": "Effective conductivity in situ",
                "method": "High-frequency AC resistance (EIS series-R intercept) "
                          "between reference probes, current on versus off",
                "resolution_required": "±3 % on R_s",
                "calibrates": "Bruggeman exponent — the direct measurement of "
                              "the ohmic penalty this module predicts",
            },
            {
                "quantity": "Axial current non-uniformity",
                "method": "Segmented cathode (3-5 electrically isolated strips, "
                          "individual shunts) at fixed total current",
                "resolution_required": "±1 % per segment",
                "calibrates": "solve_current_distribution end-to-end; this is "
                              "the only measurement that tests the coupling "
                              "rather than a single correlation",
            },
        ],
        "decision_rules": {
            "confirm": (
                "Measured outlet void fraction and segment current spread within "
                "±30 % of prediction at 100 and 300 mA/cm² — the reduced-order "
                "model is adequate and the design basis's no-CFD rule holds."
            ),
            "recalibrate": (
                "Systematic offset but correct trend with j and height — refit "
                "C0, Bruggeman exponent and θ_max, keep the model structure."
            ),
            "escalate": (
                "Observed maldistribution, dead zones, slugging or gas "
                "channelling that a 1-D axial model cannot represent — this is "
                "the specific trigger for two-phase CFD named in "
                "docs/REFERENCE_CELL_DESIGN_BASIS.md line 108."
            ),
        },
        "safety_note": (
            "Gas measurement train must not add meaningful backpressure, and "
            "the collection vessel must not create a closed hydrogen path. "
            "Displacement collection vents to the same safe exhaust as the cell."
        ),
    }


def model_scope() -> Dict[str, Any]:
    """Machine-readable statement of what this model is and is not."""
    return {
        "provenance": (
            "Screening two-phase model, Level 0. No gas hold-up, bubble size, "
            "surface coverage or void-fraction data exists in this repository. "
            "Drift-flux, Fritz, Harmathy, Bruggeman and Stephan-Vogt "
            "correlations are transferred from water electrolysis, "
            "chlor-alkali and air-water column practice, not fitted to an "
            "iron sulfate cell."
        ),
        "computes": [
            "Faradaic H2/O2 generation as wet gas at cell temperature",
            "Drift-flux void fraction profile up a vertical channel",
            "Bruggeman effective conductivity and axial ohmic burden",
            "Bubble surface coverage and active-area reduction",
            "Stephan-Vogt bubble microconvection and effective boundary layer",
            "Equipotential-electrode current redistribution over height",
            "Self-consistent gas <-> current <-> FE fixed point",
            "Headspace hydrogen concentration, LFL margin, dilution flow",
            "Electrode-height scaling limit set by hold-up non-uniformity",
        ],
        "does_not_compute": [
            "Radial/lateral bubble distribution or wall-vs-core segregation",
            "Bubble coalescence, breakup and size distribution evolution",
            "Slug or churn-turbulent regime transitions",
            "Gas blinding of the membrane or anolyte-side crossover effects",
            "Dissolved-gas supersaturation and nucleation kinetics",
            "Transient/startup gas dynamics or flow instability",
            "Dead zones and maldistribution from real manifold geometry",
            "Hydrogen absorption into the deposit (see hydrogen_embrittlement.py)",
        ],
        "dominant_uncertainty": (
            "Surface coverage ceiling theta_max and the Bruggeman exponent. "
            "Literature coverage on gas-evolving electrodes spans a few "
            "percent to above 0.5, and measured Bruggeman exponents span "
            "1.5-2.5. Both act directly on the ohmic penalty and the "
            "active-area correction."
        ),
        "replaced_by": "measurement_protocol() — ~$450, 3 days, RC-1 hardware",
        "level": 0,
        "gap_closed": [
            "docs/NEXT_STEPS.md section 3.3 (hydrodynamics, bubbles)",
            "docs/REFERENCE_CELL_DESIGN_BASIS.md line 42 (no validated bubble model)",
            "docs/SIM_THEORY_CONFIDENCE.md claim 2 (gas/flow partially modeled)",
        ],
    }
