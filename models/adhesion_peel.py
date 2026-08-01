"""
Deposit adhesion and peel mechanics — the gating unknown for continuous foil.

Why this module exists
----------------------
``models/cell_architecture.py`` screened five reactor types and found that
only *continuously harvested* cells clear the areal-productivity target.  Of
those, the rotating cylinder makes powder (a feedstock-path answer) and
**drum-and-strip is the only screened route to continuous coherent foil**.
That route rests on one assumption which the architecture module itself flags
and explicitly refuses to compute:

    ``limitations``: "Peelability of iron from the drum is UNVERIFIED - the
    gating risk"

    ``model_scope()["does_not_compute"]``: "Deposit adhesion or peelability -
    the gating unknown for drums"

``docs/PROGRAM_SUMMARY.md`` carries the same hole into the program gates:
gate 2 asks for "an iron-on-titanium peel/adhesion coupon" because "it gates
the entire continuous-foil architecture branch."  Nothing in ``models/``
touched it.  This module closes that gap: it turns "does iron peel?" into a
computable *window* with named failure modes, and it sizes the coupon
experiment that would falsify it.

The physics chain
-----------------
Peeling is an energy-balance competition between what the film has stored and
what the interface costs to break.

1. **Residual stress** (:func:`residual_stress`).  Electrodeposited iron from
   sulfate baths is strongly *tensile* — the deposit wants to contract.  Three
   additive contributions:

   * *Intrinsic / grain-coalescence* stress, Hoffman's grain-boundary
     relaxation model, ``σ_i = E' · Δ / d`` — finer grains, higher stress.
     This is the dominant term for nanocrystalline electrodeposits.
   * *Hydrogen* stress.  Codeposited H expands the lattice; when it effuses
     after deposition the constrained film is left in additional tension.
   * *Thermal mismatch* on cooling from bath to ambient,
     ``σ_th = E' · (α_f − α_s) · ΔT``.

2. **Stored elastic energy → driving force** (:func:`energy_release_rate`).
   A residually stressed film of thickness ``h`` releases, per unit area of
   new interface, ``G_ss = (1 − ν) σ² h / E``.  Note the **linear thickness
   dependence**: a thick deposit peels itself off, a thin one does not.  This
   is why copper-foil drums run thin foil and why the same drum may hold onto
   a 5 µm iron film and spit off a 100 µm one.

3. **Interfacial toughness** (:func:`interfacial_toughness`).  The resistance
   term.  Its thermodynamic floor is the Dupré work of adhesion
   ``W = γ_f + γ_s − γ_fs``, of order 0.3–3 J/m².  Measured peel toughness of
   ductile metal films is 10–1000× larger because the *film* dissipates
   plastic work at the crack tip.  The model carries that amplification
   explicitly rather than hiding it in a fitted constant, and applies a
   Rice–Wang-type hydrogen knockdown: segregated H lowers the interfacial
   separation energy.

4. **Peel mechanics** (:func:`peel_force_per_width`).  Steady-state peel at
   angle θ: ``G = (P/b)(1 − cos θ) + G_residual``.  Residual stress *assists*
   the peel, so the machine only has to supply the difference.

5. **The web must survive its own peel** (:func:`evaluate_peel`).  The peel
   force is carried by the foil cross-section: ``σ_web = (P/b)/h``.  If that
   exceeds the deposit's own yield strength the foil tears instead of peeling
   — and thin, hydrogen-charged, nanocrystalline iron is exactly the material
   most likely to do so.  **This is the constraint that makes iron different
   from copper**, and it is why the peel question cannot be answered by
   adhesion alone.

The result is a *window*, not a number: the deposit must release hard enough
to be strippable and hold on well enough to be wound.

Outcomes
--------
=========================  =================================================
``bonded_no_release``      Peel force exceeds what a winder can apply; the
                           deposit stays on the drum.  Foil route dead.
``cohesive_failure_in_film``  The interface is tougher than the foil itself,
                           so the crack leaves the interface and runs through
                           the deposit: metal stays on the drum and the strip
                           comes off in fragments.  This is how a *well
                           bonded* deposit fails, and it is the outcome the
                           work-of-adhesion number alone cannot predict.
``tears_before_peel``      Peel is possible in principle but the required web
                           stress exceeds the foil's strength.  Foil route
                           dead unless the deposit is toughened or thickened.
``marginal_peel``          Inside the window but with little margin on tear
                           or on control.
``clean_peel``             The target: controlled, continuous strip.
``spontaneous_delamination``  Stored energy alone exceeds interfacial
                           toughness; the deposit self-releases.  **Fatal for
                           foil, excellent for a flake/powder harvester** —
                           the same computation therefore also scores the
                           feedstock (Option A) path.
=========================  =================================================

Scope and honesty
-----------------
This is a **screening** model at the evidence tier of ``cell_architecture.py``
and ``deposit_morphology.py``.  No wet-lab iron adhesion data exists in this
repository, and none is invented here.  Surface energies are literature
values for clean surfaces; real drums carry adsorbates, and the plastic
amplification factor is the single least-constrained quantity in the chain
(it spans an order of magnitude in the peel-mechanics literature).  Every
substrate in the library therefore carries an ``evidence_level`` and a
provenance string, and :func:`coupon_test_protocol` specifies the measurement
that would replace the estimate.  Treat outputs as hypothesis ranking, not as
a peel-strength prediction.

References
----------
- Stoney, G.G. (1909) *The tension of metallic films deposited by
  electrolysis*, Proc. R. Soc. Lond. A 82, 172.
- Hoffman, R.W. (1976) *Stresses in thin films: the relevance of grain
  boundaries and impurities*, Thin Solid Films 34, 185 — ``σ = E'Δ/d``.
- Hutchinson, J.W., Suo, Z. (1992) *Mixed mode cracking in layered
  materials*, Adv. Appl. Mech. 29, 63 — steady-state ``G`` for films.
- Kendall, K. (1975) *Thin-film peeling — the elastic term*, J. Phys. D 8,
  1449 — the ``(1 − cos θ)`` peel balance.
- Kim, K.-S., Aravas, N. (1988) *Elastoplastic analysis of the peel test*,
  Int. J. Solids Struct. 24, 417 — why measured peel work exceeds ``W_adh``.
- Rice, J.R., Wang, J.-S. (1989) *Embrittlement of interfaces by solute
  segregation*, Mater. Sci. Eng. A 107, 23 — H knockdown of interface energy.
- Weil, R. (1970/1971) *The origins of stress in electrodeposits*, Plating
  57–58 — tensile stress in electrodeposited iron-group metals.
- Electrodeposited copper foil practice: titanium drum with a passive TiO₂
  release layer, continuous peel at 30–120 A/dm² (industrial practice).

See ``models/README.md`` for the model-scope contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

# ─── Iron film elastic and thermal properties ────────────────────────
E_FE_GPA = 211.0            # Young's modulus, polycrystalline bcc Fe
NU_FE = 0.29                # Poisson ratio
ALPHA_FE_PER_K = 11.8e-6    # linear thermal expansion, 1/K
GAMMA_FE_J_M2 = 2.4         # surface energy of Fe(110) — matches
                            # deposit_morphology.GAMMA_FE_SURFACE

# ─── Substrate thermal expansion (drum/cathode materials), 1/K ───────
ALPHA_TI_PER_K = 8.6e-6
ALPHA_SS316_PER_K = 16.0e-6
ALPHA_CU_PER_K = 16.5e-6

# ─── Hoffman intrinsic-stress relaxation distance ────────────────────
# Grain-boundary relaxation distance Δ; literature range 0.02–0.20 nm for
# electrodeposited iron-group metals.  0.10 nm reproduces the few-hundred-MPa
# tensile stresses reported for sulfate iron deposits at ~0.1–1 µm grain size
# (≈300 MPa at 0.1 µm grain, ≈60 MPa at 0.5 µm).
HOFFMAN_DELTA_M = 0.10e-9

# ─── Hydrogen stress coefficient ─────────────────────────────────────
# Partial molar volume of H in bcc Fe ≈ 2.0e-6 m³/mol.  Constrained lattice
# expansion by dissolved H, then effusion, leaves the film in tension.
V_H_PARTIAL_M3_MOL = 2.0e-6
M_H_KG_MOL = 1.008e-3
RHO_FE = 7874.0             # kg/m³ — matches electrochemistry.RHO_FE

# ─── Hydrogen knockdown of interfacial toughness (Rice–Wang type) ────
# The floor is reached near ~70 ppm with these constants; below that the
# knockdown must actually vary, or the model would report hydrogen as
# irrelevant across the whole range electrowon iron plausibly occupies.
H_KNOCKDOWN_COEFF = 0.20    # fractional Γ loss per ln-decade of H above C0
H_KNOCKDOWN_C0_PPM = 1.0
H_KNOCKDOWN_FLOOR = 0.15    # Γ never falls below this fraction of Γ_dry

# ─── Plastic-zone confinement in a thin film ─────────────────────────
# Crack-tip plastic dissipation in the peeling film cannot exceed what the
# film's own thickness can accommodate: a very thin foil has no room for a
# developed plastic zone, so its peel work collapses towards the
# thermodynamic work of adhesion (Kim & Aravas 1988; Wei & Hutchinson 1997).
# The dissipation therefore scales with h up to a saturation thickness of
# order the plastic-zone size.
PLASTIC_ZONE_THICKNESS_UM = 50.0

# ─── Machine limits for a foil winding line ──────────────────────────
# What a web-handling line can actually do, independent of chemistry.
# Electroformed-foil lines run web tensions of order 10-100 N/m on 10-35 µm
# foil. 200 N/m is a generous ceiling for peel plus tension: above it the
# drive, nip and tracking hardware becomes a different and much heavier
# machine, and the web starts to stretch over the idlers. The ceiling is a
# *machine* limit, not a materials one, so it is exposed as a field on
# PeelConditions and swept in the driver rather than buried here.
MAX_WINDER_TENSION_N_PER_M = 200.0    # practical peel/tension ceiling
MIN_CONTROLLABLE_TENSION_N_PER_M = 5.0  # below this the web flaps free
WEB_STRESS_SAFETY_FACTOR = 0.5        # fraction of yield allowed in the web

Bonding = Literal["metallic", "oxide_weak", "oxide_passive", "release_coating"]
EvidenceLevel = Literal["commercial", "pilot", "lab", "concept"]
Outcome = Literal[
    "bonded_no_release",
    "cohesive_failure_in_film",
    "tears_before_peel",
    "marginal_peel",
    "clean_peel",
    "spontaneous_delamination",
]


# ═════════════════════════════════════════════════════════════════════
#  Substrate library
# ═════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SubstrateSpec:
    """A candidate cathode/drum surface for iron deposition.

    Attributes
    ----------
    surface_energy_J_m2
        Surface energy of the *outermost* layer actually contacted — the
        passive oxide on titanium or stainless, not the bare metal beneath.
    interface_energy_J_m2
        Fe/substrate interfacial energy γ_fs entering Dupré's relation.  For
        weakly bonded oxide couples this is close to ``γ_f + γ_s`` (little
        adhesion); for metallic couples it is small.
    plastic_amplification
        Ratio of measured peel toughness to thermodynamic work of adhesion,
        capturing crack-tip plastic dissipation in the film.  This is the
        least-constrained parameter in the model; see ``model_scope()``.
    roughness_Ra_um
        Surface roughness.  Mechanical interlocking raises effective
        toughness — a ground-and-polished drum is deliberately smooth.
    electrically_conductive
        A release layer that is not conductive cannot be a cathode.  This
        kills the most attractive-sounding fixes (PTFE, polymer release).
    """

    id: str
    name: str
    surface_energy_J_m2: float
    interface_energy_J_m2: float
    bonding: Bonding
    plastic_amplification: float
    roughness_Ra_um: float
    thermal_expansion_per_K: float
    electrically_conductive: bool
    evidence_level: EvidenceLevel
    provenance: str
    notes: str = ""
    limitations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.surface_energy_J_m2 <= 0:
            raise ValueError("surface energy must be positive")
        if self.plastic_amplification < 1.0:
            raise ValueError("plastic amplification cannot be below 1")
        if self.roughness_Ra_um < 0:
            raise ValueError("roughness must be non-negative")


#: Candidate drum/cathode surfaces.  Surface energies are clean-surface
#: literature values; interface energies and amplification factors are
#: engineering estimates by bonding class, not measurements for iron.
SUBSTRATES: Dict[str, SubstrateSpec] = {
    "ti_passive_tio2": SubstrateSpec(
        id="ti_passive_tio2",
        name="Titanium drum with passive TiO₂ (Cu-foil practice)",
        surface_energy_J_m2=0.45,     # rutile/amorphous TiO₂, hydrated
        interface_energy_J_m2=2.55,   # weakly bonded: W ≈ 0.30 J/m²
        bonding="oxide_passive",
        plastic_amplification=25.0,
        roughness_Ra_um=0.15,         # ground and polished foil drum
        thermal_expansion_per_K=ALPHA_TI_PER_K,
        electrically_conductive=True,
        evidence_level="commercial",
        provenance=(
            "Commercial for COPPER foil, not for iron. The passive TiO₂ film "
            "is what makes Cu foil peelable; whether Fe behaves the same on "
            "the same surface is the untested assumption this module exists "
            "to expose."
        ),
        notes=(
            "The reference case. Conductive through the thin passive film, "
            "mechanically robust, already the industry standard drum."
        ),
        limitations=[
            "Transferred from copper; iron may reduce or penetrate the oxide",
            "Fe(II)-containing acid can locally break down TiO₂ passivity",
            "Oxide thickness drifts with service and with cathodic polarisation",
        ],
    ),
    "ti_bare_etched": SubstrateSpec(
        id="ti_bare_etched",
        name="Titanium, etched/depassivated (metallic contact)",
        surface_energy_J_m2=2.0,
        interface_energy_J_m2=1.4,    # W ≈ 3.0 J/m² — strong metallic bond
        bonding="metallic",
        plastic_amplification=80.0,
        roughness_Ra_um=0.8,
        thermal_expansion_per_K=ALPHA_TI_PER_K,
        electrically_conductive=True,
        evidence_level="lab",
        provenance="Clean-metal surface energies; metallic-couple estimate.",
        notes=(
            "The failure mode of the passive drum rather than a design "
            "choice: if cathodic polarisation strips the TiO₂, this is the "
            "surface the deposit actually sees."
        ),
        limitations=[
            "Etched Ti repassivates in seconds in aerated electrolyte",
            "Strong adhesion is the point of failure, not a feature",
        ],
    ),
    "stainless_316_passive": SubstrateSpec(
        id="stainless_316_passive",
        name="316L stainless, passive Cr₂O₃ (tankhouse blank practice)",
        surface_energy_J_m2=0.9,
        interface_energy_J_m2=2.5,    # W ≈ 0.80 J/m²
        bonding="oxide_passive",
        plastic_amplification=40.0,
        roughness_Ra_um=0.4,
        thermal_expansion_per_K=ALPHA_SS316_PER_K,
        electrically_conductive=True,
        evidence_level="commercial",
        provenance=(
            "Zinc/copper tankhouse permanent cathode blanks are stripped "
            "mechanically after batch plating — commercial, but batch and "
            "with far thicker deposits than a foil drum."
        ),
        notes=(
            "The known-good industrial answer for BATCH stripping. Higher "
            "thermal expansion than Ti adds mismatch stress on cool-down, "
            "which assists release."
        ),
        limitations=[
            "Demonstrated for batch strip, not continuous peel",
            "Fe-on-Fe-bearing substrate risks epitaxial/metallurgical bonding",
        ],
    ),
    "chromium_plated": SubstrateSpec(
        id="chromium_plated",
        name="Hard-chromium plated mandrel (electroforming practice)",
        surface_energy_J_m2=0.7,      # passive Cr₂O₃ over hard Cr
        interface_energy_J_m2=2.7,    # W ≈ 0.40 J/m²
        bonding="oxide_weak",
        plastic_amplification=15.0,
        roughness_Ra_um=0.10,
        thermal_expansion_per_K=6.2e-6,
        electrically_conductive=True,
        evidence_level="commercial",
        provenance=(
            "Standard electroforming release surface (nickel electroforming "
            "mandrels). Commercial for Ni; transferred to Fe here."
        ),
        notes=(
            "Electroforming's answer to exactly this problem: a conductive, "
            "hard, low-adhesion passive surface engineered for part release. "
            "The most credible alternative to the TiO₂ drum."
        ),
        limitations=[
            "Hexavalent-chromium plating is a regulatory liability",
            "Chromium passivity in hot acidic Fe(II) sulfate is unverified",
        ],
    ),
    "copper_substrate": SubstrateSpec(
        id="copper_substrate",
        name="Copper cathode (metallic, high adhesion)",
        surface_energy_J_m2=1.8,
        interface_energy_J_m2=1.2,    # W ≈ 3.0 J/m²
        bonding="metallic",
        plastic_amplification=90.0,
        roughness_Ra_um=0.5,
        thermal_expansion_per_K=ALPHA_CU_PER_K,
        electrically_conductive=True,
        evidence_level="lab",
        provenance="Clean-metal surface energies; metallic-couple estimate.",
        notes=(
            "Included as the deliberate negative control: Fe bonds well to "
            "Cu. A coupon set that cannot distinguish this from TiO₂ has not "
            "measured adhesion, it has measured noise."
        ),
        limitations=[
            "Cu contamination of the deposit violates the hot-shortness spec",
            "Not a candidate drum material — a control only",
        ],
    ),
    "ptfe_release_coating": SubstrateSpec(
        id="ptfe_release_coating",
        name="PTFE / polymer release coating",
        surface_energy_J_m2=0.02,
        interface_energy_J_m2=2.4,    # W ≈ 0.02 J/m² — essentially non-adhesive
        bonding="release_coating",
        plastic_amplification=4.0,
        roughness_Ra_um=0.05,
        thermal_expansion_per_K=100e-6,
        electrically_conductive=False,
        evidence_level="concept",
        provenance="Polymer surface-energy literature.",
        notes=(
            "The obvious fix, and the reason to model conductivity "
            "explicitly: a non-conductive release layer cannot pass current "
            "and therefore cannot be the cathode. Kept in the library so the "
            "screen rejects it on physics rather than by omission."
        ),
        limitations=[
            "Electrically insulating — disqualifying for a cathode surface",
            "Conductive-filled variants trade release for percolation paths",
        ],
    ),
}


# ═════════════════════════════════════════════════════════════════════
#  Elastic helpers
# ═════════════════════════════════════════════════════════════════════

def plane_strain_modulus_Pa(
    E_GPa: float = E_FE_GPA,
    nu: float = NU_FE,
) -> float:
    """E' = E / (1 − ν²), the plane-strain modulus (Pa)."""
    if E_GPa <= 0:
        raise ValueError("modulus must be positive")
    if not -1.0 < nu < 0.5:
        raise ValueError("Poisson ratio out of physical range")
    return E_GPa * 1e9 / (1.0 - nu ** 2)


def biaxial_modulus_Pa(
    E_GPa: float = E_FE_GPA,
    nu: float = NU_FE,
) -> float:
    """E/(1 − ν) — the modulus governing equibiaxial film stress (Pa)."""
    if E_GPa <= 0:
        raise ValueError("modulus must be positive")
    if not -1.0 < nu < 0.5:
        raise ValueError("Poisson ratio out of physical range")
    return E_GPa * 1e9 / (1.0 - nu)


# ═════════════════════════════════════════════════════════════════════
#  1. Residual stress
# ═════════════════════════════════════════════════════════════════════

def hoffman_intrinsic_stress_MPa(
    grain_size_um: float,
    relaxation_distance_m: float = HOFFMAN_DELTA_M,
    E_GPa: float = E_FE_GPA,
    nu: float = NU_FE,
) -> float:
    """Grain-coalescence tensile stress, ``σ = E/(1−ν) · Δ/d`` (MPa).

    Hoffman's picture: adjacent grains snap together across a boundary gap of
    order Δ as the film coalesces, and the constrained film is left in
    tension.  The ``1/d`` scaling is the operationally important part —
    **the fine-grained deposits that plating conditions favour are exactly
    the ones with the highest self-peeling driving force.**

    Positive returned values are tensile.
    """
    if grain_size_um <= 0:
        raise ValueError("grain size must be positive")
    if relaxation_distance_m < 0:
        raise ValueError("relaxation distance must be non-negative")
    d_m = grain_size_um * 1e-6
    sigma_Pa = biaxial_modulus_Pa(E_GPa, nu) * relaxation_distance_m / d_m
    return sigma_Pa / 1e6


def hydrogen_stress_MPa(
    C_H_ppm: float,
    fraction_effused: float = 1.0,
    E_GPa: float = E_FE_GPA,
    nu: float = NU_FE,
) -> float:
    """Tensile stress left behind when codeposited hydrogen effuses (MPa).

    Dissolved H dilates the lattice by ``V_H`` per mole.  A film bonded to a
    rigid substrate cannot contract when that H leaves, so the lost
    dilatation appears as biaxial tension:

    ``σ_H = E/(1−ν) · (1/3) · (V_H · n_H / V) · f_effused``

    where ``n_H/V`` is the H molar density in the deposit.  This term is the
    direct coupling between HER — the program's central efficiency problem —
    and mechanical release: **hydrogen the process is trying to suppress for
    Faradaic reasons also drives the deposit off the drum.**
    """
    if C_H_ppm < 0:
        raise ValueError("hydrogen content must be non-negative")
    if not 0.0 <= fraction_effused <= 1.0:
        raise ValueError("fraction_effused must lie in [0, 1]")
    # ppm by mass → mol H per m³ of deposit
    mass_frac = C_H_ppm * 1e-6
    n_H_per_m3 = mass_frac * RHO_FE / M_H_KG_MOL
    volumetric_strain = V_H_PARTIAL_M3_MOL * n_H_per_m3
    linear_strain = volumetric_strain / 3.0
    sigma_Pa = biaxial_modulus_Pa(E_GPa, nu) * linear_strain * fraction_effused
    return sigma_Pa / 1e6


def thermal_mismatch_stress_MPa(
    delta_T_K: float,
    alpha_film_per_K: float = ALPHA_FE_PER_K,
    alpha_substrate_per_K: float = ALPHA_TI_PER_K,
    E_GPa: float = E_FE_GPA,
    nu: float = NU_FE,
) -> float:
    """Thermal-mismatch stress on cooling from bath to ambient (MPa).

    ``σ_th = E/(1−ν) · (α_f − α_s) · ΔT`` with ``ΔT = T_bath − T_ambient``
    positive for cooling.  A substrate that shrinks *more* than iron (316L,
    copper) puts the film into compression and squeezes it off; titanium
    shrinks less, holding iron in tension.
    """
    sigma_Pa = (
        biaxial_modulus_Pa(E_GPa, nu)
        * (alpha_film_per_K - alpha_substrate_per_K)
        * delta_T_K
    )
    return sigma_Pa / 1e6


def residual_stress(
    grain_size_um: float = 0.5,
    C_H_ppm: float = 0.0,
    bath_temperature_C: float = 60.0,
    ambient_temperature_C: float = 25.0,
    substrate: Optional[SubstrateSpec] = None,
    hydrogen_fraction_effused: float = 1.0,
    relaxation_distance_m: float = HOFFMAN_DELTA_M,
) -> Dict[str, float]:
    """Total residual stress in the deposit, decomposed by mechanism (MPa).

    Returns each contribution separately so the dominant term is visible
    rather than buried in a single number.
    """
    sub = substrate or SUBSTRATES["ti_passive_tio2"]
    intrinsic = hoffman_intrinsic_stress_MPa(
        grain_size_um, relaxation_distance_m=relaxation_distance_m
    )
    hydrogen = hydrogen_stress_MPa(C_H_ppm, hydrogen_fraction_effused)
    thermal = thermal_mismatch_stress_MPa(
        bath_temperature_C - ambient_temperature_C,
        alpha_substrate_per_K=sub.thermal_expansion_per_K,
    )
    total = intrinsic + hydrogen + thermal
    contributions = {
        "intrinsic": abs(intrinsic),
        "hydrogen": abs(hydrogen),
        "thermal": abs(thermal),
    }
    dominant = max(contributions, key=contributions.get)
    return {
        "intrinsic_MPa": float(intrinsic),
        "hydrogen_MPa": float(hydrogen),
        "thermal_MPa": float(thermal),
        "total_MPa": float(total),
        "dominant_mechanism": dominant,
        "sign": "tensile" if total >= 0 else "compressive",
    }


def stoney_stress_MPa(
    curvature_change_per_m: float,
    substrate_thickness_m: float,
    film_thickness_m: float,
    substrate_E_GPa: float = 116.0,   # Ti coupon default
    substrate_nu: float = 0.32,
) -> float:
    """Stoney's equation — film stress from *measured* coupon curvature (MPa).

    ``σ = E_s h_s² Δκ / (6 (1 − ν_s) h_f)``

    This is the inverse of the forward model above and the reason the coupon
    experiment is worth running: a strip of shim stock, a deposit, and a
    surface profilometer measure the driving force directly, with no
    dependence on the Hoffman relaxation distance.
    """
    if substrate_thickness_m <= 0 or film_thickness_m <= 0:
        raise ValueError("thicknesses must be positive")
    if film_thickness_m > substrate_thickness_m / 10.0:
        # Stoney assumes a thin film on a thick substrate.
        pass  # not an error; flagged by callers
    sigma_Pa = (
        substrate_E_GPa * 1e9
        * substrate_thickness_m ** 2
        * curvature_change_per_m
        / (6.0 * (1.0 - substrate_nu) * film_thickness_m)
    )
    return sigma_Pa / 1e6


def stoney_validity(
    substrate_thickness_m: float,
    film_thickness_m: float,
    max_thickness_ratio: float = 0.1,
) -> Dict[str, Any]:
    """Check the thin-film assumption behind :func:`stoney_stress_MPa`."""
    ratio = film_thickness_m / substrate_thickness_m
    return {
        "thickness_ratio": float(ratio),
        "max_thickness_ratio": max_thickness_ratio,
        "valid": bool(ratio <= max_thickness_ratio),
        "note": (
            "Stoney assumes h_film << h_substrate. Above the ratio limit the "
            "measured curvature underestimates stress and a finite-thickness "
            "correction is required."
        ),
    }


# ═════════════════════════════════════════════════════════════════════
#  2. Driving force
# ═════════════════════════════════════════════════════════════════════

def energy_release_rate(
    stress_MPa: float,
    thickness_um: float,
    E_GPa: float = E_FE_GPA,
    nu: float = NU_FE,
) -> float:
    """Steady-state strain-energy release rate of a stressed film (J/m²).

    ``G_ss = (1 − ν) σ² h / E`` for an equibiaxially stressed film that fully
    relaxes on delamination (Hutchinson & Suo).  Both signs of stress store
    energy, so a compressive film can buckle-delaminate just as a tensile one
    can peel.

    The **linear dependence on thickness** is the design lever: doubling the
    foil thickness doubles the self-peeling driving force at fixed stress.
    """
    if thickness_um <= 0:
        raise ValueError("thickness must be positive")
    sigma_Pa = stress_MPa * 1e6
    h_m = thickness_um * 1e-6
    return (1.0 - nu) * sigma_Pa ** 2 * h_m / (E_GPa * 1e9)


def critical_thickness_um(
    stress_MPa: float,
    toughness_J_m2: float,
    E_GPa: float = E_FE_GPA,
    nu: float = NU_FE,
) -> float:
    """Thickness at which the film spontaneously delaminates (µm).

    Inverting ``G_ss = Γ``:  ``h_c = Γ E / ((1 − ν) σ²)``.  Below ``h_c`` the
    deposit stays put no matter how badly it is bonded; above it, the deposit
    comes off by itself.  For a foil line this is the **upper** bound on
    thickness; for a flake harvester it is the target to exceed.
    """
    if toughness_J_m2 <= 0:
        raise ValueError("toughness must be positive")
    if stress_MPa == 0:
        return math.inf
    sigma_Pa = abs(stress_MPa) * 1e6
    h_m = toughness_J_m2 * E_GPa * 1e9 / ((1.0 - nu) * sigma_Pa ** 2)
    return h_m * 1e6


# ═════════════════════════════════════════════════════════════════════
#  3. Interfacial toughness
# ═════════════════════════════════════════════════════════════════════

def dupre_work_of_adhesion(
    gamma_film_J_m2: float,
    gamma_substrate_J_m2: float,
    gamma_interface_J_m2: float,
) -> float:
    """Dupré: ``W = γ_f + γ_s − γ_fs`` (J/m²), clipped at zero."""
    return max(0.0, gamma_film_J_m2 + gamma_substrate_J_m2 - gamma_interface_J_m2)


def girifalco_good_work_of_adhesion(
    gamma_film_J_m2: float,
    gamma_substrate_J_m2: float,
    interaction_parameter: float = 1.0,
) -> float:
    """Geometric-mean estimate ``W = 2 φ √(γ_f γ_s)`` (J/m²).

    Used as a sanity bracket on the library's tabulated interface energies:
    a dispersive-only estimate should not exceed the Dupré value for a
    metallic couple, nor fall far below it for a passive oxide.
    """
    if gamma_film_J_m2 < 0 or gamma_substrate_J_m2 < 0:
        raise ValueError("surface energies must be non-negative")
    return 2.0 * interaction_parameter * math.sqrt(
        gamma_film_J_m2 * gamma_substrate_J_m2
    )


def hydrogen_toughness_knockdown(
    C_H_ppm: float,
    coefficient: float = H_KNOCKDOWN_COEFF,
    C0_ppm: float = H_KNOCKDOWN_C0_PPM,
    floor: float = H_KNOCKDOWN_FLOOR,
) -> float:
    """Fractional retention of interfacial toughness under H segregation.

    Rice–Wang thermodynamics: solute segregated to an interface lowers its
    separation energy.  Screening form
    ``f = 1 − k·ln(1 + C_H/C₀)``, clipped to ``[floor, 1]``.

    The sign matters for the program's story: hydrogen *helps* release and
    *hurts* the foil.  Both effects are represented, and they act against
    each other in :func:`evaluate_peel`.
    """
    if C_H_ppm < 0:
        raise ValueError("hydrogen content must be non-negative")
    f = 1.0 - coefficient * math.log1p(C_H_ppm / C0_ppm)
    return float(np.clip(f, floor, 1.0))


def confined_plastic_amplification(
    amplification: float,
    thickness_um: float,
    plastic_zone_um: float = PLASTIC_ZONE_THICKNESS_UM,
) -> float:
    """Plastic amplification available to a film of finite thickness.

    ``φ_eff = 1 + (φ − 1)·min(1, h / h_plastic)``

    A peeling film dissipates plastic work in a zone that cannot be larger
    than the film itself.  Thin foil therefore peels closer to its
    thermodynamic work of adhesion, and peel work rises with thickness until
    the zone is fully developed — the thickness dependence of peel strength
    that is measured routinely and would otherwise be missing here.
    """
    if amplification < 1.0:
        raise ValueError("plastic amplification cannot be below 1")
    if thickness_um <= 0 or plastic_zone_um <= 0:
        raise ValueError("thicknesses must be positive")
    confinement = min(1.0, thickness_um / plastic_zone_um)
    return 1.0 + (amplification - 1.0) * confinement


def interfacial_toughness(
    substrate: SubstrateSpec,
    C_H_ppm: float = 0.0,
    thickness_um: float = PLASTIC_ZONE_THICKNESS_UM,
    gamma_film_J_m2: float = GAMMA_FE_J_M2,
    roughness_reference_um: float = 0.1,
    roughness_coefficient: float = 0.6,
    amplification_override: Optional[float] = None,
) -> Dict[str, float]:
    """Interfacial fracture toughness Γ of the Fe/substrate interface (J/m²).

    ``Γ = W_adh × φ_eff(h) × (1 + k_r · Ra/Ra_ref) × f_H``

    * ``W_adh`` is the thermodynamic floor (Dupré).
    * ``φ_eff`` is crack-tip plastic dissipation in the film — 10–100× for a
      fully developed plastic zone, reduced by
      :func:`confined_plastic_amplification` for thin foil.  It is the
      dominant uncertainty here.
    * The roughness term is mechanical interlocking; it is why foil drums are
      polished and why an as-machined coupon will over-report adhesion.
    * ``f_H`` is the hydrogen knockdown.
    """
    W = dupre_work_of_adhesion(
        gamma_film_J_m2,
        substrate.surface_energy_J_m2,
        substrate.interface_energy_J_m2,
    )
    phi = (
        substrate.plastic_amplification
        if amplification_override is None
        else float(amplification_override)
    )
    phi_eff = confined_plastic_amplification(phi, thickness_um)
    roughness_factor = 1.0 + roughness_coefficient * (
        substrate.roughness_Ra_um / roughness_reference_um
    )
    f_H = hydrogen_toughness_knockdown(C_H_ppm)
    gamma_i = W * phi_eff * roughness_factor * f_H
    return {
        "work_of_adhesion_J_m2": float(W),
        "plastic_amplification": float(phi),
        "effective_amplification": float(phi_eff),
        "roughness_factor": float(roughness_factor),
        "hydrogen_retention": float(f_H),
        "toughness_J_m2": float(gamma_i),
    }


# ═════════════════════════════════════════════════════════════════════
#  4. Peel mechanics
# ═════════════════════════════════════════════════════════════════════

def peel_force_per_width(
    toughness_J_m2: float,
    peel_angle_deg: float = 90.0,
    residual_G_J_m2: float = 0.0,
) -> float:
    """Steady-state peel force per unit width (N/m).

    Kendall's energy balance for a flexible strip:
    ``G = (P/b)(1 − cos θ) + G_residual``, so

    ``P/b = max(0, (Γ − G_residual)) / (1 − cos θ)``

    Stored residual energy pays part of the bill; when it covers the whole
    interfacial toughness the required force is zero and the deposit
    self-releases.
    """
    if not 0.0 < peel_angle_deg <= 180.0:
        raise ValueError("peel angle must lie in (0, 180] degrees")
    if toughness_J_m2 < 0:
        raise ValueError("toughness must be non-negative")
    denom = 1.0 - math.cos(math.radians(peel_angle_deg))
    if denom <= 1e-12:
        return math.inf
    return max(0.0, toughness_J_m2 - residual_G_J_m2) / denom


def film_tearing_energy_J_m2(
    yield_strength_MPa: float,
    elongation_fraction: float,
    thickness_um: float,
    shape_factor: float = 2.0,
) -> float:
    """Essential work of fracture of the foil itself (J/m²).

    ``W_film ≈ k · σ_y · ε_f · h``

    The energy required to drive a crack *through* the deposit rather than
    along its interface.  Like ``G``, it scales with thickness — thin,
    hydrogen-charged, nanocrystalline iron has very little of it.

    This is the quantity that decides whether a peel is clean.  An interface
    tougher than the film does not hold the film on; it redirects the crack
    into the film, so the deposit fragments and leaves metal on the drum.
    Work of adhesion alone cannot predict that, which is why copper-foil
    experience does not transfer to iron by analogy.
    """
    if thickness_um <= 0:
        raise ValueError("thickness must be positive")
    if yield_strength_MPa <= 0:
        raise ValueError("yield strength must be positive")
    if not 0.0 <= elongation_fraction <= 1.0:
        raise ValueError("elongation fraction must lie in [0, 1]")
    return (
        shape_factor
        * yield_strength_MPa * 1e6
        * elongation_fraction
        * thickness_um * 1e-6
    )


def web_stress_MPa(peel_force_N_per_m: float, thickness_um: float) -> float:
    """Tensile stress carried by the peeled web (MPa).

    ``σ_web = (P/b) / h``.  The peel force is reacted by the foil's own cross
    section, so a thin foil sees a large stress for a modest peel force.
    """
    if thickness_um <= 0:
        raise ValueError("thickness must be positive")
    return peel_force_N_per_m / (thickness_um * 1e-6) / 1e6


# ═════════════════════════════════════════════════════════════════════
#  Operating conditions and the combined evaluation
# ═════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PeelConditions:
    """Deposition and harvesting conditions for one peel evaluation."""

    thickness_um: float = 25.0            # foil-drum target (cell_architecture)
    grain_size_um: float = 0.5
    C_H_ppm: float = 2.0
    bath_temperature_C: float = 60.0
    ambient_temperature_C: float = 25.0
    peel_angle_deg: float = 90.0
    foil_yield_strength_MPa: float = 400.0
    foil_elongation_fraction: float = 0.05
    hydrogen_fraction_effused: float = 1.0
    max_winder_tension_N_per_m: float = MAX_WINDER_TENSION_N_PER_M
    min_controllable_tension_N_per_m: float = MIN_CONTROLLABLE_TENSION_N_PER_M
    web_stress_safety_factor: float = WEB_STRESS_SAFETY_FACTOR

    def __post_init__(self) -> None:
        if self.thickness_um <= 0:
            raise ValueError("thickness must be positive")
        if self.grain_size_um <= 0:
            raise ValueError("grain size must be positive")
        if self.foil_yield_strength_MPa <= 0:
            raise ValueError("yield strength must be positive")
        if not 0.0 < self.foil_elongation_fraction <= 1.0:
            raise ValueError("elongation fraction must lie in (0, 1]")
        if not 0.0 < self.web_stress_safety_factor <= 1.0:
            raise ValueError("safety factor must lie in (0, 1]")


@dataclass
class PeelResult:
    """Outcome of one substrate × condition peel evaluation."""

    substrate_id: str
    substrate_name: str
    evidence_level: str
    conductive: bool

    # Stress state
    residual_stress_MPa: float
    stress_breakdown: Dict[str, float]

    # Energies
    driving_force_J_m2: float
    work_of_adhesion_J_m2: float
    toughness_J_m2: float
    film_tearing_energy_J_m2: float
    hydrogen_retention: float

    # Mechanics
    peel_force_N_per_m: float
    web_stress_MPa: float
    allowable_web_stress_MPa: float
    critical_thickness_um: float

    # Verdict
    outcome: Outcome
    self_release_ratio: float      # G_ss / Γ ; ≥1 → spontaneous
    cohesive_ratio: float          # Γ / W_film ; ≥1 → crack runs in the film
    tear_margin: float             # allowable web stress / required
    reasons: List[str] = field(default_factory=list)

    @property
    def peelable(self) -> bool:
        """True when a controlled continuous strip is predicted."""
        return self.outcome in ("clean_peel", "marginal_peel")

    @property
    def good_for_flake_harvest(self) -> bool:
        """True when the deposit self-releases — the powder/flake path."""
        return self.outcome == "spontaneous_delamination"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "substrate_id": self.substrate_id,
            "substrate_name": self.substrate_name,
            "evidence_level": self.evidence_level,
            "electrically_conductive": self.conductive,
            "residual_stress_MPa": round(self.residual_stress_MPa, 1),
            "stress_breakdown_MPa": {
                k: round(v, 1) for k, v in self.stress_breakdown.items()
                if isinstance(v, (int, float))
            },
            "driving_force_G_J_m2": round(self.driving_force_J_m2, 4),
            "work_of_adhesion_J_m2": round(self.work_of_adhesion_J_m2, 3),
            "interfacial_toughness_J_m2": round(self.toughness_J_m2, 2),
            "film_tearing_energy_J_m2": round(self.film_tearing_energy_J_m2, 2),
            "hydrogen_retention_factor": round(self.hydrogen_retention, 3),
            "peel_force_N_per_m": round(self.peel_force_N_per_m, 1),
            "web_stress_MPa": round(self.web_stress_MPa, 1),
            "allowable_web_stress_MPa": round(self.allowable_web_stress_MPa, 1),
            "critical_thickness_um": (
                None if math.isinf(self.critical_thickness_um)
                else round(self.critical_thickness_um, 1)
            ),
            "self_release_ratio": round(self.self_release_ratio, 4),
            "cohesive_ratio": round(self.cohesive_ratio, 4),
            "tear_margin": (
                None if math.isinf(self.tear_margin) else round(self.tear_margin, 3)
            ),
            "outcome": self.outcome,
            "peelable": self.peelable,
            "good_for_flake_harvest": self.good_for_flake_harvest,
            "reasons": self.reasons,
        }

    def summary(self) -> str:
        crit = (
            "∞" if math.isinf(self.critical_thickness_um)
            else f"{self.critical_thickness_um:,.1f} µm"
        )
        return "\n".join([
            f"{self.substrate_name}",
            f"  evidence          {self.evidence_level}"
            f"   conductive: {self.conductive}",
            f"  residual stress   {self.residual_stress_MPa:,.0f} MPa "
            f"({self.stress_breakdown.get('dominant_mechanism', '?')}-dominated)",
            f"  driving force G   {self.driving_force_J_m2:.3f} J/m²",
            f"  toughness Γ       {self.toughness_J_m2:.2f} J/m² "
            f"(W_adh {self.work_of_adhesion_J_m2:.2f})",
            f"  film tearing W    {self.film_tearing_energy_J_m2:.2f} J/m²",
            f"  G/Γ               {self.self_release_ratio:.3f}"
            f"   Γ/W_film {self.cohesive_ratio:.3f}",
            f"  peel force        {self.peel_force_N_per_m:,.0f} N/m",
            f"  web stress        {self.web_stress_MPa:,.0f} MPa "
            f"(allowable {self.allowable_web_stress_MPa:,.0f})",
            f"  critical thickness{crit:>12}",
            f"  outcome           {self.outcome}",
        ])


def evaluate_peel(
    substrate: SubstrateSpec,
    conditions: Optional[PeelConditions] = None,
    amplification_override: Optional[float] = None,
) -> PeelResult:
    """Full peel evaluation for one substrate at one operating point.

    Chains residual stress → driving force → toughness → peel force → web
    stress, then classifies the outcome against both the machine's limits and
    the foil's own strength.
    """
    cond = conditions or PeelConditions()

    stress = residual_stress(
        grain_size_um=cond.grain_size_um,
        C_H_ppm=cond.C_H_ppm,
        bath_temperature_C=cond.bath_temperature_C,
        ambient_temperature_C=cond.ambient_temperature_C,
        substrate=substrate,
        hydrogen_fraction_effused=cond.hydrogen_fraction_effused,
    )
    sigma = stress["total_MPa"]

    G = energy_release_rate(sigma, cond.thickness_um)
    tough = interfacial_toughness(
        substrate,
        C_H_ppm=cond.C_H_ppm,
        thickness_um=cond.thickness_um,
        amplification_override=amplification_override,
    )
    gamma_i = tough["toughness_J_m2"]

    W_film = film_tearing_energy_J_m2(
        cond.foil_yield_strength_MPa,
        cond.foil_elongation_fraction,
        cond.thickness_um,
    )

    P = peel_force_per_width(gamma_i, cond.peel_angle_deg, residual_G_J_m2=G)
    sigma_web = web_stress_MPa(P, cond.thickness_um) if math.isfinite(P) else math.inf
    allowable = cond.foil_yield_strength_MPa * cond.web_stress_safety_factor
    h_crit = critical_thickness_um(sigma, gamma_i)
    ratio = G / gamma_i if gamma_i > 0 else math.inf
    cohesive_ratio = gamma_i / W_film if W_film > 0 else math.inf
    tear_margin = allowable / sigma_web if sigma_web > 0 else math.inf

    outcome, reasons = _classify(
        substrate=substrate,
        conditions=cond,
        peel_force_N_per_m=P,
        web_stress=sigma_web,
        allowable_web_stress=allowable,
        self_release_ratio=ratio,
        cohesive_ratio=cohesive_ratio,
    )

    return PeelResult(
        substrate_id=substrate.id,
        substrate_name=substrate.name,
        evidence_level=substrate.evidence_level,
        conductive=substrate.electrically_conductive,
        residual_stress_MPa=sigma,
        stress_breakdown=stress,
        driving_force_J_m2=G,
        work_of_adhesion_J_m2=tough["work_of_adhesion_J_m2"],
        toughness_J_m2=gamma_i,
        film_tearing_energy_J_m2=W_film,
        hydrogen_retention=tough["hydrogen_retention"],
        peel_force_N_per_m=P,
        web_stress_MPa=sigma_web,
        allowable_web_stress_MPa=allowable,
        critical_thickness_um=h_crit,
        outcome=outcome,
        self_release_ratio=ratio,
        cohesive_ratio=cohesive_ratio,
        tear_margin=tear_margin,
        reasons=reasons,
    )


def _classify(
    substrate: SubstrateSpec,
    conditions: PeelConditions,
    peel_force_N_per_m: float,
    web_stress: float,
    allowable_web_stress: float,
    self_release_ratio: float,
    cohesive_ratio: float,
) -> tuple:
    """Assign an outcome label and the reasons behind it.

    Order matters: self-release is checked first because a deposit that has
    already fallen off cannot be peeled; cohesive failure is checked next
    because a crack that has left the interface is no longer a peel at all;
    and a non-conductive surface is rejected before any mechanics are
    considered meaningful.
    """
    reasons: List[str] = []

    if not substrate.electrically_conductive:
        reasons.append(
            "Surface is electrically insulating and cannot serve as a cathode "
            "— disqualified regardless of its release behaviour."
        )

    if self_release_ratio >= 1.0:
        reasons.append(
            f"Stored elastic energy exceeds interfacial toughness "
            f"(G/Γ = {self_release_ratio:.2f}): the deposit releases without "
            f"applied force. Fatal for continuous foil, ideal for a "
            f"flake/powder harvester."
        )
        return "spontaneous_delamination", reasons

    if cohesive_ratio >= 1.0:
        reasons.append(
            f"Interfacial toughness exceeds the deposit's own tearing energy "
            f"(Γ/W_film = {cohesive_ratio:.2f}): the crack leaves the "
            f"interface and runs through the foil. The deposit fragments and "
            f"leaves metal on the drum — a bonded failure that no amount of "
            f"winder force fixes."
        )
        return "cohesive_failure_in_film", reasons

    if peel_force_N_per_m > conditions.max_winder_tension_N_per_m:
        reasons.append(
            f"Required peel force {peel_force_N_per_m:,.0f} N/m exceeds the "
            f"winder ceiling {conditions.max_winder_tension_N_per_m:,.0f} N/m: "
            f"the deposit stays on the drum."
        )
        return "bonded_no_release", reasons

    if web_stress > allowable_web_stress:
        reasons.append(
            f"Peel demands {web_stress:,.0f} MPa in the web against an "
            f"allowable {allowable_web_stress:,.0f} MPa "
            f"({conditions.web_stress_safety_factor:.0%} of a "
            f"{conditions.foil_yield_strength_MPa:,.0f} MPa yield): the foil "
            f"tears before the interface does."
        )
        return "tears_before_peel", reasons

    margin = allowable_web_stress / web_stress if web_stress > 0 else math.inf
    if peel_force_N_per_m < conditions.min_controllable_tension_N_per_m:
        reasons.append(
            f"Peel force {peel_force_N_per_m:,.1f} N/m is below the "
            f"controllable-tension floor "
            f"{conditions.min_controllable_tension_N_per_m:,.1f} N/m: release "
            f"happens but the web cannot be tracked or tensioned reliably."
        )
        return "marginal_peel", reasons
    if margin < 2.0:
        reasons.append(
            f"Inside the window but with only {margin:.2f}× margin on web "
            f"tearing — insufficient for a production line given the "
            f"uncertainty in the plastic amplification factor."
        )
        return "marginal_peel", reasons
    if self_release_ratio > 0.5:
        reasons.append(
            f"Inside the window but G/Γ = {self_release_ratio:.2f}; a modest "
            f"increase in thickness or stress tips the deposit into "
            f"uncontrolled self-release."
        )
        return "marginal_peel", reasons
    if cohesive_ratio > 0.5:
        reasons.append(
            f"Inside the window but Γ/W_film = {cohesive_ratio:.2f}; the "
            f"interface is nearly as tough as the foil, so any embrittlement "
            f"of the deposit sends the crack into the film instead."
        )
        return "marginal_peel", reasons

    reasons.append(
        f"Controlled strip predicted: {peel_force_N_per_m:,.0f} N/m peel "
        f"force, {margin:.1f}× margin against web tearing, G/Γ = "
        f"{self_release_ratio:.2f}."
    )
    return "clean_peel", reasons


# ═════════════════════════════════════════════════════════════════════
#  Screens and sweeps
# ═════════════════════════════════════════════════════════════════════

def screen_substrates(
    conditions: Optional[PeelConditions] = None,
    substrates: Optional[Dict[str, SubstrateSpec]] = None,
) -> List[PeelResult]:
    """Evaluate every substrate in the library, ranked most→least peelable.

    Ranking: viable peel windows first (widest tear margin first), then
    spontaneous release, then hard failures.
    """
    cond = conditions or PeelConditions()
    lib = substrates or SUBSTRATES
    results = [evaluate_peel(s, cond) for s in lib.values()]

    order = {
        "clean_peel": 0,
        "marginal_peel": 1,
        "spontaneous_delamination": 2,
        "tears_before_peel": 3,
        "cohesive_failure_in_film": 4,
        "bonded_no_release": 5,
    }

    def key(r: PeelResult):
        margin = r.tear_margin if math.isfinite(r.tear_margin) else 1e9
        return (0 if r.conductive else 1, order[r.outcome], -margin)

    return sorted(results, key=key)


def thickness_sweep(
    substrate: SubstrateSpec,
    thicknesses_um: Optional[np.ndarray] = None,
    conditions: Optional[PeelConditions] = None,
) -> Dict[str, Any]:
    """Sweep deposit thickness — the single strongest lever on release.

    Driving force grows linearly with thickness while toughness does not, so
    every substrate has a thickness above which it self-releases.  The web
    stress falls with thickness at fixed peel force, so thin foil tears.
    **The foil window is bounded from both sides**, and this sweep locates it.
    """
    cond = conditions or PeelConditions()
    if thicknesses_um is None:
        thicknesses_um = np.logspace(math.log10(1.0), math.log10(500.0), 60)
    thicknesses_um = np.asarray(thicknesses_um, dtype=float)

    rows = [
        evaluate_peel(
            substrate,
            PeelConditions(**{**cond.__dict__, "thickness_um": float(t)}),
        )
        for t in thicknesses_um
    ]
    viable = [t for t, r in zip(thicknesses_um, rows) if r.peelable]
    return {
        "substrate_id": substrate.id,
        "thickness_um": thicknesses_um.tolist(),
        "driving_force_J_m2": [r.driving_force_J_m2 for r in rows],
        "toughness_J_m2": [r.toughness_J_m2 for r in rows],
        "peel_force_N_per_m": [r.peel_force_N_per_m for r in rows],
        "web_stress_MPa": [r.web_stress_MPa for r in rows],
        "outcome": [r.outcome for r in rows],
        "viable_thickness_min_um": float(min(viable)) if viable else None,
        "viable_thickness_max_um": float(max(viable)) if viable else None,
    }


def hydrogen_sweep(
    substrate: SubstrateSpec,
    C_H_ppm_values: Optional[np.ndarray] = None,
    conditions: Optional[PeelConditions] = None,
) -> Dict[str, Any]:
    """Sweep codeposited hydrogen — the two-sided coupling to HER.

    Hydrogen simultaneously *raises* the driving force (effusion tension) and
    *lowers* the interfacial toughness (Rice–Wang), so release gets easier
    monotonically.  It also embrittles the foil, which this sweep exposes via
    the tear margin if a yield-strength knockdown is supplied by the caller.
    """
    cond = conditions or PeelConditions()
    if C_H_ppm_values is None:
        C_H_ppm_values = np.logspace(-1, 2, 40)
    C_H_ppm_values = np.asarray(C_H_ppm_values, dtype=float)

    rows = [
        evaluate_peel(
            substrate,
            PeelConditions(**{**cond.__dict__, "C_H_ppm": float(c)}),
        )
        for c in C_H_ppm_values
    ]
    return {
        "substrate_id": substrate.id,
        "C_H_ppm": C_H_ppm_values.tolist(),
        "residual_stress_MPa": [r.residual_stress_MPa for r in rows],
        "toughness_J_m2": [r.toughness_J_m2 for r in rows],
        "self_release_ratio": [r.self_release_ratio for r in rows],
        "peel_force_N_per_m": [r.peel_force_N_per_m for r in rows],
        "outcome": [r.outcome for r in rows],
    }


def grain_size_sweep(
    substrate: SubstrateSpec,
    grain_sizes_um: Optional[np.ndarray] = None,
    conditions: Optional[PeelConditions] = None,
) -> Dict[str, Any]:
    """Sweep grain size — where plating conditions enter the peel problem.

    Grain size is set by current density and waveform
    (``mechanical_properties.estimate_grain_size_um``), and it drives the
    Hoffman intrinsic stress as ``1/d``.  The pulse waveforms that refine
    grain for strength therefore also raise the self-release driving force:
    **the mechanical-property optimum and the peel optimum pull in opposite
    directions**, and this sweep quantifies the conflict.
    """
    cond = conditions or PeelConditions()
    if grain_sizes_um is None:
        grain_sizes_um = np.logspace(math.log10(0.05), math.log10(10.0), 50)
    grain_sizes_um = np.asarray(grain_sizes_um, dtype=float)

    rows = [
        evaluate_peel(
            substrate,
            PeelConditions(**{**cond.__dict__, "grain_size_um": float(d)}),
        )
        for d in grain_sizes_um
    ]
    return {
        "substrate_id": substrate.id,
        "grain_size_um": grain_sizes_um.tolist(),
        "residual_stress_MPa": [r.residual_stress_MPa for r in rows],
        "driving_force_J_m2": [r.driving_force_J_m2 for r in rows],
        "self_release_ratio": [r.self_release_ratio for r in rows],
        "outcome": [r.outcome for r in rows],
    }


def amplification_robustness(
    substrate: SubstrateSpec,
    conditions: Optional[PeelConditions] = None,
    factors: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """How the verdict moves with the least-known parameter.

    The plastic amplification factor φ spans roughly an order of magnitude in
    the peel-mechanics literature.  Rather than report a single verdict from a
    point estimate, this sweeps φ and reports the *fraction of the plausible
    range* that supports each outcome.  A verdict that survives the whole
    range is worth acting on; one that does not is a request for the coupon
    experiment.
    """
    cond = conditions or PeelConditions()
    if factors is None:
        factors = np.logspace(math.log10(2.0), math.log10(200.0), 40)
    factors = np.asarray(factors, dtype=float)

    outcomes = [
        evaluate_peel(substrate, cond, amplification_override=float(f)).outcome
        for f in factors
    ]
    counts: Dict[str, int] = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    fractions = {k: v / len(outcomes) for k, v in counts.items()}
    dominant = max(fractions, key=fractions.get)
    return {
        "substrate_id": substrate.id,
        "amplification": factors.tolist(),
        "outcome": outcomes,
        "outcome_fractions": fractions,
        "dominant_outcome": dominant,
        "verdict_robust": bool(fractions[dominant] >= 0.8),
        "note": (
            "Fraction of the plausible plastic-amplification range (2–200×) "
            "producing each outcome. verdict_robust is True only when one "
            "outcome holds across ≥80% of the range."
        ),
    }


# ═════════════════════════════════════════════════════════════════════
#  Integration with the deposition models
# ═════════════════════════════════════════════════════════════════════

def conditions_from_deposition(
    j_mA_cm2: float = 100.0,
    current_efficiency_percent: float = 85.0,
    deposition_time_s: float = 900.0,
    waveform: str = "dc",
    j_peak_mA_cm2: Optional[float] = None,
    duty_cycle: float = 1.0,
    bath_pH: float = 3.0,
    temperature_C: float = 60.0,
    peel_angle_deg: float = 90.0,
) -> Dict[str, Any]:
    """Build :class:`PeelConditions` from *plating* conditions.

    This is the module's integration point.  Rather than asking the user for
    grain size, hydrogen content, thickness and yield strength — all of which
    are consequences of how the cell is run — it derives them from the
    existing models:

    * grain size ← ``mechanical_properties.estimate_grain_size_um``
    * deposit thickness ← Faraday's law at the given ``j``, FE and time
    * diffusible H ← ``hydrogen_embrittlement.hydrogen_uptake_from_electrolysis``
    * foil yield strength ← ``mechanical_properties.MechanicalPropertiesModel``

    so that a single operating point propagates consistently from cell
    conditions to a peel verdict.
    """
    from .electrochemistry import FARADAY, M_FE
    from .hydrogen_embrittlement import hydrogen_uptake_from_electrolysis
    from .mechanical_properties import (
        MechanicalPropertiesModel,
        estimate_grain_size_um,
    )

    if j_mA_cm2 <= 0:
        raise ValueError("current density must be positive")
    if not 0.0 < current_efficiency_percent <= 100.0:
        raise ValueError("current efficiency must lie in (0, 100]")

    fe = current_efficiency_percent / 100.0
    j_A_m2 = j_mA_cm2 * 10.0

    # Faraday's law → deposit thickness
    mass_kg_m2 = j_A_m2 * fe * deposition_time_s * M_FE / (2.0 * FARADAY)
    thickness_um = mass_kg_m2 / RHO_FE * 1e6

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
        thickness_um=max(thickness_um, 0.05),
        grain_size_um=grain_um,
        C_H_ppm=h_uptake["C_H_diffusible_ppm"],
        bath_temperature_C=temperature_C,
        peel_angle_deg=peel_angle_deg,
        foil_yield_strength_MPa=mech.sigma_y_MPa,
    )
    return {
        "conditions": conditions,
        "derived": {
            "thickness_um": float(thickness_um),
            "grain_size_um": float(grain_um),
            "C_H_diffusible_ppm": float(h_uptake["C_H_diffusible_ppm"]),
            "foil_yield_strength_MPa": float(mech.sigma_y_MPa),
            "deposition_rate_um_hr": float(thickness_um / (deposition_time_s / 3600.0)),
        },
        "sources": {
            "grain_size": "mechanical_properties.estimate_grain_size_um",
            "hydrogen": "hydrogen_embrittlement.hydrogen_uptake_from_electrolysis",
            "yield_strength": "mechanical_properties.MechanicalPropertiesModel",
            "thickness": "Faraday's law at the supplied j, FE and time",
        },
    }


def foil_route_verdict(
    conditions: Optional[PeelConditions] = None,
    substrate_id: str = "ti_passive_tio2",
) -> Dict[str, Any]:
    """Go/no-go for the drum-and-strip (continuous foil) architecture branch.

    ``cell_architecture.py`` found drum-and-strip to be the only screened
    route to continuous coherent foil and marked its peelability UNVERIFIED.
    This function returns the branch verdict that module could not compute,
    together with the robustness of that verdict against the plastic
    amplification uncertainty and the fallback if the branch fails.
    """
    cond = conditions or PeelConditions()
    substrate = SUBSTRATES[substrate_id]
    result = evaluate_peel(substrate, cond)
    robustness = amplification_robustness(substrate, cond)
    screen = screen_substrates(cond)
    alternatives = [
        r.substrate_id for r in screen
        if r.peelable and r.conductive and r.substrate_id != substrate_id
    ]
    self_releasing = [r.substrate_id for r in screen if r.good_for_flake_harvest]

    if result.peelable and robustness["verdict_robust"]:
        verdict = "proceed"
    elif result.peelable or alternatives:
        verdict = "proceed_with_coupon_test"
    elif self_releasing:
        verdict = "pivot_to_flake_harvest"
    else:
        verdict = "no_go"

    return {
        "reference_substrate": substrate_id,
        "outcome": result.outcome,
        "verdict": verdict,
        "verdict_robust_to_amplification": robustness["verdict_robust"],
        "dominant_outcome_over_amplification_range":
            robustness["dominant_outcome"],
        "outcome_fractions": robustness["outcome_fractions"],
        "peelable_alternatives": alternatives,
        "self_releasing_substrates": self_releasing,
        "result": result.to_dict(),
        "interpretation": _verdict_text(verdict, result, alternatives, self_releasing),
    }


def _verdict_text(
    verdict: str,
    result: PeelResult,
    alternatives: List[str],
    self_releasing: List[str],
) -> str:
    if verdict == "proceed":
        return (
            "The continuous-foil branch survives the screen on its reference "
            "surface across the plausible amplification range. The coupon "
            "test becomes a confirmation, not a gate."
        )
    if verdict == "proceed_with_coupon_test":
        alt = ", ".join(alternatives) if alternatives else "none"
        return (
            "The foil branch is neither confirmed nor killed by the screen: "
            f"the reference surface returns '{result.outcome}' and viable "
            f"alternatives are [{alt}]. The verdict moves within the "
            "plausible plastic-amplification range, so the peel coupon is "
            "the cheapest way to resolve it and must precede any drum spend."
        )
    if verdict == "pivot_to_flake_harvest":
        return (
            "No screened surface supports controlled continuous peel at these "
            "conditions, but "
            f"[{', '.join(self_releasing)}] self-release. That is the "
            "feedstock (Option A) path: harvest flake, skip the winder, and "
            "let the melt shop do the metallurgy — which the program already "
            "names as its primary route."
        )
    return (
        "Neither controlled peel nor clean self-release is predicted at these "
        "conditions. The deposit stays bonded or tears; both outcomes kill "
        "continuous harvesting on the screened surfaces."
    )


# ═════════════════════════════════════════════════════════════════════
#  The experiment this module is asking for
# ═════════════════════════════════════════════════════════════════════

def coupon_test_protocol(
    conditions: Optional[PeelConditions] = None,
) -> Dict[str, Any]:
    """Specify the peel/curvature coupon experiment that resolves the branch.

    ``docs/PROGRAM_SUMMARY.md`` gate 2 already asks for this test and calls it
    "nearly free". This returns the executable version: which coupons, which
    measurements, what each one replaces in the model, and what result would
    kill the foil branch.
    """
    cond = conditions or PeelConditions()
    return {
        "title": "Iron-on-substrate adhesion coupon set (Day-1 add-on)",
        "gates": (
            "docs/PROGRAM_SUMMARY.md gate 2; cell_architecture.py "
            "drum_and_strip branch"
        ),
        "runs_alongside": (
            "The Phase-II Hull cell order in docs/FIRST_LAB_DAY.md — same "
            "bath, same rectifier, same session."
        ),
        "coupons": [
            {
                "substrate": s.id,
                "role": (
                    "reference drum surface" if s.id == "ti_passive_tio2"
                    else "negative control (expect strong bond)"
                    if s.bonding == "metallic"
                    else "alternative release surface"
                ),
                "n_replicates": 3,
                "evidence_level": s.evidence_level,
            }
            for s in SUBSTRATES.values()
            if s.electrically_conductive
        ],
        "measurements": [
            {
                "measurement": "90° peel test, ASTM B571 / D6862 geometry",
                "instrument": "Load cell on a motorised stage, 10 mm strip",
                "yields": "Peel force per width → interfacial toughness Γ",
                "replaces_in_model": (
                    "plastic_amplification — the least-constrained parameter "
                    "in interfacial_toughness()"
                ),
            },
            {
                "measurement": "Coupon curvature before and after deposition",
                "instrument": (
                    "Stylus profilometer or optical flat on 100 µm shim stock"
                ),
                "yields": "Residual stress via stoney_stress_MPa()",
                "replaces_in_model": (
                    "HOFFMAN_DELTA_M and the whole forward residual-stress "
                    "estimate"
                ),
            },
            {
                "measurement": "Deposit thickness by mass and by cross-section",
                "instrument": "Analytical balance + metallographic mount",
                "yields": "h for G = (1−ν)σ²h/E, and a Faradaic cross-check",
                "replaces_in_model": "Faraday-law thickness assumption",
            },
            {
                "measurement": "Thermal-desorption or hot-extraction hydrogen",
                "instrument": "Inert-gas fusion analyser (outsourced)",
                "yields": "Diffusible H (ppm)",
                "replaces_in_model": (
                    "hydrogen_embrittlement.hydrogen_uptake_from_electrolysis "
                    "absorption-fraction estimate"
                ),
            },
            {
                "measurement": "Spontaneous-release thickness ladder",
                "instrument": "Same cell, four plating durations",
                "yields": "Observed critical thickness h_c",
                "replaces_in_model": (
                    "critical_thickness_um() — a direct, single-number test "
                    "of the whole energy balance"
                ),
            },
        ],
        "decision_rules": {
            "kills_foil_branch": (
                "Iron on passive TiO₂ requires >"
                f"{cond.max_winder_tension_N_per_m:,.0f} N/m to peel, or the "
                "peeled strip fractures at any thickness that plates in a "
                "reasonable drum residence time."
            ),
            "confirms_foil_branch": (
                "A reproducible peel force between "
                f"{cond.min_controllable_tension_N_per_m:,.0f} and "
                f"{cond.max_winder_tension_N_per_m:,.0f} N/m with an intact "
                "strip and ≥2× margin to web yielding."
            ),
            "redirects_to_flake": (
                "Deposits self-release below the target foil thickness — "
                "adopt the flake/powder harvest path and delete the winder."
            ),
        },
        "estimated_cost_usd": {
            "coupon_stock_and_masking": 250,
            "peel_fixture_and_load_cell": 900,
            "profilometry_access": 0,
            "hydrogen_analysis_outsourced": 600,
            "total": 1750,
        },
        "estimated_duration_days": 3,
        "why_now": (
            "It is the cheapest experiment that can delete an entire "
            "architecture branch. Every drum, winder and tension-control "
            "decision downstream is contingent on its result, and the "
            "screen's verdict is not robust to the one parameter it cannot "
            "estimate from first principles."
        ),
    }


def comparison_table(results: List[PeelResult]) -> str:
    """Fixed-width substrate comparison for console/report output."""
    header = (
        f"{'Substrate':<40} {'Evid.':<11} {'σ_res':>8} {'G':>8} {'Γ':>8} "
        f"{'G/Γ':>7} {'P/b':>10} {'Outcome':<24}"
    )
    sep = "─" * len(header)
    lines = [header, sep]
    for r in results:
        p = "∞" if math.isinf(r.peel_force_N_per_m) else f"{r.peel_force_N_per_m:,.0f}"
        lines.append(
            f"{r.substrate_name[:39]:<40} {r.evidence_level:<11} "
            f"{r.residual_stress_MPa:>8,.0f} "
            f"{r.driving_force_J_m2:>8.3f} "
            f"{r.toughness_J_m2:>8.2f} "
            f"{r.self_release_ratio:>7.3f} "
            f"{p:>10} "
            f"{r.outcome:<24}"
        )
    lines.append("")
    lines.append(
        "σ_res in MPa (tensile +), G and Γ in J/m², P/b peel force in N/m"
    )
    return "\n".join(lines)


def model_scope() -> Dict[str, Any]:
    """Machine-readable statement of what this model is and is not."""
    return {
        "provenance": (
            "Screening model. No wet-lab iron adhesion data exists in this "
            "repository. Surface energies are clean-surface literature "
            "values; interface energies and plastic amplification factors "
            "are engineering estimates by bonding class, transferred from "
            "copper-foil and electroforming practice."
        ),
        "computes": [
            "Residual stress decomposed into intrinsic, hydrogen and thermal",
            "Steady-state strain-energy release rate G(σ, h)",
            "Interfacial toughness from work of adhesion × plastic × roughness",
            "Hydrogen knockdown of interfacial toughness (Rice-Wang form)",
            "Steady-state peel force per width at arbitrary peel angle",
            "Web stress and the tear-before-peel criterion",
            "Critical self-delamination thickness",
            "Substrate ranking and the drum-and-strip branch verdict",
            "Stoney inversion of measured coupon curvature",
        ],
        "does_not_compute": [
            "Mode-mixity of the interfacial crack (assumed steady-state peel)",
            "Rate/temperature dependence of the plastic dissipation term",
            "Adhesion evolution over drum service life or passivation drift",
            "Localised adhesion failure: pinholes, edge build-up, nodules",
            "Deposit morphology (see deposit_morphology.py)",
            "Whether the peeled foil is metallurgically acceptable",
        ],
        "calibration_required": [
            "90° peel toughness of iron on the actual drum surface",
            "Residual stress by coupon curvature (Stoney) in the actual bath",
            "Hoffman relaxation distance Δ for this deposit",
            "Diffusible hydrogen by thermal desorption",
            "Observed critical self-release thickness",
        ],
        "key_uncertainty": (
            "plastic_amplification spans roughly an order of magnitude in the "
            "peel literature and is the single parameter most likely to flip "
            "the verdict. amplification_robustness() reports whether it does."
        ),
    }
