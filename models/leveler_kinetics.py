"""
Additive / leveler Langmuir adsorption kinetics (CHEM_PHYS_REVIEW.md §2.6).

Why this module exists
----------------------
§2.6 calls out that the whole morphology/stress programme is currently driven
by a *single* knob — ``internal_stress.saccharin_g_L`` — where the lab will
actually control a *package* of additives, each with a literature-documented
mechanism.  This module replaces that one lumped parameter with mechanism-level
adsorption kinetics:

  * :func:`langmuir_coverage` / :func:`gamma_adsorbed` — a **Langmuir isotherm**
    per additive (saccharin, thiourea, PEG, coumarin, chloride) giving the
    adsorbed coverage θ_i and areal density Γ_i = Γ_max,i·θ_i from a bath
    concentration (g/L → mol/L via each additive's molar mass), with an
    Arrhenius temperature correction on the adsorption constant.
  * :func:`nucleation_rate_multiplier` — a **Γ-dependent nucleation rate**.
    Levelers / brighteners block growth sites on stable low-index facets and
    force the growing layer to renucleate; the more that is adsorbed, the finer
    (and more numerous) the nuclei.  This is the term that takes the existing
    morphology screen from "coarse or fine?" toward "what package reaches
    structural grade?".
  * :func:`h_recomb_fraction` / :func:`h_recomb_overpotential_reduction_V` —
    a **Γ-dependent H-recombination** term.  Sulphur/thiourea-type levelers and
    grain refiners catalyse the 2 H_ads → H₂ recombination at the surface, so a
    larger Γ leaves *less* H to be codeposited / trapped and lowers the
    overpotential the HER branch must pay — the "stress relief by hydrogen
    recombination catalysis" mechanism §2.6 names.
  * :func:`stress_relief_fraction` — a Γ-dependent relief of the *intrinsic*
    deposit stress, aggregating the whole package (saccharin stays the dominant
    reliever) and saturating at ``RELIEF_MAX``.
  * :func:`carbon_incorporation_blocking` — the same organic coverage that
    refines the grain also blocks the Guglielmi carbon-particle anchoring step,
    so heavy leveler packages suppress particle co-deposition.
  * :func:`structural_grade_score` — folds relief + grain refinement + H
    removal into one screening number so packages can be ranked.

Scope & conventions
-------------------
L0 screening scaffold (matches the rest of the twin stack).  Every screening
constant carries ``SCREENING_FLAG`` so the confidence ledger can find it at
audit time.  No data has been fit; the constants are the uncertain ±30 %
literature-scale numbers documented per additive.  Everything here is **opt-in
/ additive**: the base ``internal_stress`` / ``deposit_morphology`` /
``co_deposition`` code paths are byte-identical when no additive package is
supplied (see the ``additive_package=None`` defaults in the parent modules).

Input convention
----------------
An *additive package* is ``dict[str, float]`` mapping additive id (one of
``SUPPORTED_ADDITIVES``) to bath concentration in **g/L**, e.g.::

    {"saccharin": 1.5, "thiourea": 0.05, "peg": 0.2, "chloride": 2.0}

``surface_coverages(pkg, ...)`` resolves every entry to θ_i / Γ_i and returns
the aggregate ``AdditivePackage`` used by the rest of the stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

SCREENING_FLAG = "unvalidated (L1)"

# ─── Physical constants ─────────────────────────────────────────────
N_AVOGADRO = 6.02214076e23   # 1/mol
EPSILON_0 = 8.8541878128e-12  # F/m (vacuum permittivity)
DEBYE = 3.33564e-30            # C·m (one Debye)

# Saturation of the aggregate intrinsic-stress relief.  Matches the value the
# single-saccharin screening fit in internal_stress.SACCHARIN_RELIEF_MAX so a
# saccharin-only leveler package is consistent with the existing knob.
RELIEF_MAX = 0.80
# θ scale at which the multiplicative relief reaches ~63 % of RELIEF_MAX.
RELIEF_THETA_REF = 0.60

# Max fraction of particle strong-adsorption a full organic monolayer blocks.
# Matches ``co_deposition.GuglielmiCarbonIncorporation.ORGANIC_BLOCK_MAX`` so a
# package's ``carbon_incorporation_blocking`` is what a caller feeds straight
# into that model's ``organic_coverage_theta``.
CARBON_BLOCK_MAX = 0.70


@dataclass(frozen=True)
class AdditiveSpec:
    """Screening-level physical parameters for one additive.

    All numbers are the uncertain (±30 %) literature-scale L0 screening values;
    none are fit to wet-lab data (none exists yet).
    """

    id: str
    molar_mass_g_mol: float        # to convert g/L → mol/L
    gamma_max_mol_m2: float        # monolayer areal density
    K_ad_L_mol: float              # Langmuir affinity at T_ref
    dH_ads_J_mol: float            # isosteric enthalpy for the T-correction
    dipole_D: float                # effective dipole (organic ψ₁ shift)
    # Mechanism strengths (fractional response at a full monolayer, θ_i = 1):
    relief_coeff: float            # fraction of intrinsic-stress relief
    nucleation_coeff: float        # additive contribution to J_nuc multiplier
    h_recomb_coeff: float          # fraction of adsorbed H catalytically recombined
    # Role tag for report legibility.
    role: str = "leveler/brightener"


# Screening isotherm parameters.  K_ad (L/mol) is the binding affinity; bigger
# = adsorbs from lower bulk concentration.  Molar masses from standard tables.
ADDITIVE_SPECS: Dict[str, AdditiveSpec] = {
    "saccharin": AdditiveSpec(
        id="saccharin",
        molar_mass_g_mol=183.18,
        gamma_max_mol_m2=8.0e-6,
        K_ad_L_mol=300.0,
        dH_ads_J_mol=-25.0e3,
        dipole_D=2.0,
        relief_coeff=0.80,
        nucleation_coeff=2.0,
        h_recomb_coeff=0.40,
        role="leveler/brightener",
    ),
    "thiourea": AdditiveSpec(
        id="thiourea",
        molar_mass_g_mol=76.12,
        gamma_max_mol_m2=9.0e-6,
        K_ad_L_mol=5000.0,
        dH_ads_J_mol=-30.0e3,
        dipole_D=3.0,
        relief_coeff=0.60,
        nucleation_coeff=3.0,
        h_recomb_coeff=0.60,
        role="leveler/grain refiner",
    ),
    "peg": AdditiveSpec(
        id="peg",
        molar_mass_g_mol=2000.0,     # representative low-MW suppressor chain
        gamma_max_mol_m2=2.0e-7,     # few long chains per m² but large footprint
        K_ad_L_mol=5000.0,
        dH_ads_J_mol=-35.0e3,
        dipole_D=0.8,
        relief_coeff=0.35,
        nucleation_coeff=1.5,
        h_recomb_coeff=0.30,
        role="suppressor",
    ),
    "coumarin": AdditiveSpec(
        id="coumarin",
        molar_mass_g_mol=146.14,
        gamma_max_mol_m2=8.0e-6,
        K_ad_L_mol=800.0,
        dH_ads_J_mol=-28.0e3,
        dipole_D=1.5,
        relief_coeff=0.50,
        nucleation_coeff=4.0,
        h_recomb_coeff=0.20,
        role="brightener",
    ),
    "chloride": AdditiveSpec(
        id="chloride",
        molar_mass_g_mol=35.45,
        gamma_max_mol_m2=1.0e-5,
        K_ad_L_mol=50.0,
        dH_ads_J_mol=-15.0e3,
        dipole_D=1.2,
        relief_coeff=0.30,
        nucleation_coeff=1.2,
        h_recomb_coeff=0.30,
        role="anion co-adsorbate",
    ),
}

SUPPORTED_ADDITIVES = tuple(sorted(ADDITIVE_SPECS))

T_REF_K = 298.15


def is_supported(additive_id: str) -> bool:
    return additive_id in ADDITIVE_SPECS


# ─── Langmuir isotherm ──────────────────────────────────────────────
def adsorption_constant_at_T(
    K_ref_L_mol: float,
    temperature_C: float,
    dH_ads_J_mol: float = -25.0e3,
) -> float:
    """Arrhenius temperature correction to a Langmuir adsorption constant.

    ``K(T) = K_ref · exp(−ΔH_ads/R · (1/T − 1/T_ref))``.  Exothermic adsorption
    (ΔH_ads < 0) weakens binding as T rises.
    """
    T = float(temperature_C) + 273.15
    return K_ref_L_mol * math.exp(
        -dH_ads_J_mol / 8.31446261815324 * (1.0 / T - 1.0 / T_REF_K)
    )


def langmuir_coverage(K_L_mol: float, c_mol_L: float) -> float:
    """Langmuir fractional coverage θ = K·c / (1 + K·c) ∈ [0, 1)."""
    if c_mol_L < 0.0:
        raise ValueError("concentration must be non-negative")
    if K_L_mol < 0.0:
        raise ValueError("adsorption constant must be non-negative")
    if c_mol_L == 0.0:
        return 0.0
    return float(K_L_mol * c_mol_L / (1.0 + K_L_mol * c_mol_L))


def gamma_adsorbed(
    spec: AdditiveSpec,
    c_g_L: float,
    temperature_C: float = 60.0,
) -> float:
    """Adsorbed areal density Γ = Γ_max·θ (mol/m²) for one additive.

    Converts the bath concentration from g/L to mol/L using the additive's
    molar mass, temperature-corrects the affinity, then applies Langmuir.
    """
    if c_g_L < 0.0:
        raise ValueError("concentration must be non-negative")
    c_mol_L = c_g_L / spec.molar_mass_g_mol
    K = adsorption_constant_at_T(spec.K_ad_L_mol, temperature_C, spec.dH_ads_J_mol)
    theta = langmuir_coverage(K, c_mol_L)
    return spec.gamma_max_mol_m2 * theta


def coverage_per_additive(
    spec: AdditiveSpec,
    c_g_L: float,
    temperature_C: float = 60.0,
) -> Dict[str, float]:
    """θ, Γ (mol/m²) and K for one additive's current bath concentration."""
    c_mol_L = c_g_L / spec.molar_mass_g_mol
    K = adsorption_constant_at_T(spec.K_ad_L_mol, temperature_C, spec.dH_ads_J_mol)
    theta = langmuir_coverage(K, c_mol_L)
    return {
        "c_g_L": c_g_L,
        "c_mol_L": c_mol_L,
        "K_ad_L_mol": K,
        "theta": theta,
        "gamma_mol_m2": spec.gamma_max_mol_m2 * theta,
    }


# ─── Aggregate additive package ─────────────────────────────────────
@dataclass
class AdditivePackage:
    """Resolved surface state of an additive package.

    ``theta_by_id`` holds per-additive Langmuir coverage; ``theta_organic`` is
    the joint probability that at least one additive occupies a site
    (1 − ∏(1−θ_i)) — the site a growing adatom or a Guglielmi particle actually
    competes against.  ``nucleation_multiplier``, ``h_recomb_fraction``,
    ``h_recomb_overpotential_reduction_V`` and ``stress_relief_fraction`` are
    the package-level mechanism outputs the rest of the twin feeds on.  All are
    monotone non-decreasing in every additive's Γ and equal their trivial
    values (1.0 / 0.0) for the null package, preserving the no-additive path.
    """

    additive_specs: Dict[str, AdditiveSpec] = field(default_factory=dict)
    theta_by_id: Dict[str, float] = field(default_factory=dict)
    gamma_by_id: Dict[str, float] = field(default_factory=dict)
    theta_organic: float = 0.0
    gamma_organic_mol_m2: float = 0.0

    # Mechanism outputs
    nucleation_multiplier: float = 1.0
    h_recomb_fraction: float = 0.0
    h_recomb_overpotential_reduction_V: float = 0.0
    stress_relief_fraction: float = 0.0
    carbon_incorporation_blocking: float = 1.0

    def summary(self) -> Dict[str, Any]:
        return {
            "theta_by_id": {k: round(v, 4) for k, v in self.theta_by_id.items()},
            "gamma_mol_m2_by_id": {k: f"{v:.2e}" for k, v in self.gamma_by_id.items()},
            "theta_organic": round(self.theta_organic, 4),
            "nucleation_multiplier": round(self.nucleation_multiplier, 3),
            "h_recomb_fraction": round(self.h_recomb_fraction, 3),
            "h_recomb_overpotential_reduction_V": round(
                self.h_recomb_overpotential_reduction_V, 4
            ),
            "stress_relief_fraction": round(self.stress_relief_fraction, 3),
            "carbon_incorporation_blocking": round(
                self.carbon_incorporation_blocking, 3
            ),
        }


def resolve_package(
    additive_pkg: Optional[Dict[str, float]] = None,
    temperature_C: float = 60.0,
) -> AdditivePackage:
    """Resolve a bath additive package (g/L per id) to mechanism outputs.

    The null package (``None`` or ``{}``) returns the identity state
    (nucleation_multiplier 1, no relief, no H recombination), so downstream
    modules keep the no-additive path byte-identical.

    Parameters
    ----------
    additive_pkg : dict[str, float] or None
        Additive id → bath concentration in g/L.  Unknown ids raise ``ValueError``.
    temperature_C : float
        Bath temperature; enters the Arrhenius adsorption-constant correction.
    """
    pkg = additive_pkg if additive_pkg is not None else {}
    specs: Dict[str, AdditiveSpec] = {}
    theta_by_id: Dict[str, float] = {}
    gamma_by_id: Dict[str, float] = {}

    for aid, c_g_L in pkg.items():
        if not is_supported(aid):
            raise ValueError(
                f"unsupported additive {aid!r}; have {list(SUPPORTED_ADDITIVES)}"
            )
        if c_g_L < 0.0:
            raise ValueError(f"additive {aid}: concentration must be non-negative")
        spec = ADDITIVE_SPECS[aid]
        cov = coverage_per_additive(spec, c_g_L, temperature_C)
        specs[aid] = spec
        theta_by_id[aid] = cov["theta"]
        gamma_by_id[aid] = cov["gamma_mol_m2"]

    if not theta_by_id:
        return AdditivePackage()

    # Joint occupancy: at least one additive on a site.
    joint_exclusion = 1.0
    for theta in theta_by_id.values():
        joint_exclusion *= (1.0 - theta)
    theta_organic = 1.0 - joint_exclusion
    gamma_organic_mol_m2 = sum(gamma_by_id.values())

    # Mechanism closures (each additive's coeff reaches its full value at θ=1).
    relief_eff = sum(ADDITIVE_SPECS[a].relief_coeff * th for a, th in theta_by_id.items())
    relief = RELIEF_MAX * (1.0 - math.exp(-relief_eff / RELIEF_THETA_REF))

    nuc_mult = 1.0 + sum(
        ADDITIVE_SPECS[a].nucleation_coeff * th for a, th in theta_by_id.items()
    )
    # Cap so extreme packages cannot drive the ratio absurdly high.
    nuc_mult = min(nuc_mult, 10.0)

    f_recomb = min(
        sum(ADDITIVE_SPECS[a].h_recomb_coeff * th for a, th in theta_by_id.items()),
        0.99,
    )
    # Screening scale: a fully-catalysing monolayer ≈ 150 mV of HER
    # overpotential saved (uncertain L1 number).
    h_eta_reduction_V = f_recomb * 0.150

    # Blocking of carbon-particle anchoring: an organic film at θ_organic
    # blocks up to CARBON_BLOCK_MAX of the strong-adsorption step; the value is
    # the *fraction of incorporation left*, i.e. 1 − CARBON_BLOCK_MAX·θ_org.
    # θ_org = 0 → 1.0 (no suppression).  Consistent with
    # ``co_deposition.organic_anchoring_blocking_factor``.
    carbon_blocking = 1.0 - CARBON_BLOCK_MAX * theta_organic

    return AdditivePackage(
        additive_specs=dict(specs),
        theta_by_id=theta_by_id,
        gamma_by_id=gamma_by_id,
        theta_organic=theta_organic,
        gamma_organic_mol_m2=gamma_organic_mol_m2,
        nucleation_multiplier=nuc_mult,
        h_recomb_fraction=f_recomb,
        h_recomb_overpotential_reduction_V=h_eta_reduction_V,
        stress_relief_fraction=min(relief, RELIEF_MAX),
        carbon_incorporation_blocking=carbon_blocking,
    )


# Convenience accessors matching the §2.6 mechanism names.
def surface_coverages(
    additive_pkg: Optional[Dict[str, float]] = None,
    temperature_C: float = 60.0,
) -> AdditivePackage:
    """Alias for :func:`resolve_package` — the resolved coverage state."""
    return resolve_package(additive_pkg, temperature_C)


def nucleation_rate_multiplier(
    additive_pkg: Optional[Dict[str, float]] = None,
    temperature_C: float = 60.0,
    resolved: Optional[AdditivePackage] = None,
) -> float:
    """Γ-dependent nucleation-rate multiplier (≥ 1) for an additive package.

    Levelers / brighteners raise the nucleation rate relative to growth by
    blocking growth sites on stable facets; the multiplier feeds the
    morphology ``nucleation_rate_ratio`` term.  Null package → 1.0.
    """
    if resolved is not None:
        return resolved.nucleation_multiplier
    return resolve_package(additive_pkg, temperature_C).nucleation_multiplier


def h_recomb_fraction(
    additive_pkg: Optional[Dict[str, float]] = None,
    temperature_C: float = 60.0,
    resolved: Optional[AdditivePackage] = None,
) -> float:
    """Γ-dependent fraction of codepositable H catalytically recombined away.

    S/thiourea-type levelers catalyse 2H_ads → H₂, so a larger Γ leaves less H
    to be absorbed/trapped — the hydrogen half of "stress relief by hydrogen
    recombination catalysis".  Null package → 0.0.
    """
    if resolved is not None:
        return resolved.h_recomb_fraction
    return resolve_package(additive_pkg, temperature_C).h_recomb_fraction


def h_recomb_overpotential_reduction_V(
    additive_pkg: Optional[Dict[str, float]] = None,
    temperature_C: float = 60.0,
    resolved: Optional[AdditivePackage] = None,
) -> float:
    """Γ-dependent reduction in the HER H-recombination overpotential (V)."""
    if resolved is not None:
        return resolved.h_recomb_overpotential_reduction_V
    return resolve_package(additive_pkg, temperature_C).h_recomb_overpotential_reduction_V


def stress_relief_fraction(
    additive_pkg: Optional[Dict[str, float]] = None,
    temperature_C: float = 60.0,
    resolved: Optional[AdditivePackage] = None,
) -> float:
    """Γ-dependent fractional relief of intrinsic deposit stress (≤ RELIEF_MAX)."""
    if resolved is not None:
        return resolved.stress_relief_fraction
    return resolve_package(additive_pkg, temperature_C).stress_relief_fraction


def carbon_incorporation_blocking(
    additive_pkg: Optional[Dict[str, float]] = None,
    temperature_C: float = 60.0,
    resolved: Optional[AdditivePackage] = None,
) -> float:
    """Multiplicative factor (≤ 1) on Guglielmi carbon incorporation.

    An organic film blocks the particle strong-adsorption (anchoring) step, so
    heavy leveler packages suppress particle co-deposition.  Null → 1.0.
    """
    if resolved is not None:
        return resolved.carbon_incorporation_blocking
    return resolve_package(additive_pkg, temperature_C).carbon_incorporation_blocking


# ─── Structural-grade scoring ───────────────────────────────────────
# Weights turning the three mechanism axes into a single structural-grade
# screening number (normalized so a perfect-but-unreachable package → 1.0).
W_RELIEF = 0.40   # low intrinsic stress
W_GRAIN = 0.35    # fine, coherent grain (nucleation refinement)
W_H = 0.25        # low trapped hydrogen

# Grain-refinement normalizer: the multiplier at which the deposit is
# convincingly refined (fine but still coherent) — screening scale.
NUC_REF = 2.0
# H-removal normalizer for the score.
H_REF = 0.5


def structural_grade_score(
    additive_pkg: Optional[Dict[str, float]] = None,
    temperature_C: float = 60.0,
    resolved: Optional[AdditivePackage] = None,
) -> Dict[str, Any]:
    """Rank an additive package against 'structural grade'.

    Structural grade in aqueous electrowinning means low residual stress (so a
    drum/strip won't peel), fine-but-coherent grain (so tensile properties are
    smooth), and low trapped hydrogen (so embrittlement doesn't bite).  This
    folds the package's relief, grain-refinement and H-removal into one number:

        score = W_RELIEF·(relief/RELIEF_MAX)
              + W_GRAIN·min((J_nuc−1)/(NUC_REF−1), 1)
              + W_H·min(f_recomb/H_REF, 1)

    Returns a dict with the score, a coarse verdict, and the underlying axes so
    the answer to "which package gets us to structural grade?" is auditable.
    """
    pkg = resolved if resolved is not None else resolve_package(additive_pkg, temperature_C)
    relief_axis = pkg.stress_relief_fraction / RELIEF_MAX
    grain_axis = min((pkg.nucleation_multiplier - 1.0) / (NUC_REF - 1.0), 1.0)
    h_axis = min(pkg.h_recomb_fraction / H_REF, 1.0)
    score = W_RELIEF * relief_axis + W_GRAIN * grain_axis + W_H * h_axis

    if score >= 0.80:
        verdict = "structural grade"
    elif score >= 0.55:
        verdict = "near-structural"
    elif score >= 0.35:
        verdict = "functional but not structural"
    else:
        verdict = "screening (below structural)"

    return {
        "score": float(score),
        "verdict": verdict,
        "axes": {
            "relief": float(relief_axis),
            "grain_refinement": float(grain_axis),
            "h_removal": float(h_axis),
        },
        "weights": {
            "relief": W_RELIEF,
            "grain": W_GRAIN,
            "h": W_H,
        },
        "package": pkg.summary(),
    }


def compare_packages(
    packages: Dict[str, Dict[str, float]],
    temperature_C: float = 60.0,
) -> Dict[str, Any]:
    """Rank several named additive packages by structural grade."""
    rows = []
    for name, pkg in packages.items():
        res = structural_grade_score(pkg, temperature_C)
        rows.append(
            {
                "name": name,
                "package": pkg,
                "score": res["score"],
                "verdict": res["verdict"],
                "axes": res["axes"],
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"ranked": rows, "temperature_C": temperature_C}


__all__ = [
    "SCREENING_FLAG",
    "RELIEF_MAX",
    "RELIEF_THETA_REF",
    "CARBON_BLOCK_MAX",
    "AdditiveSpec",
    "ADDITIVE_SPECS",
    "SUPPORTED_ADDITIVES",
    "T_REF_K",
    "N_AVOGADRO",
    "EPSILON_0",
    "DEBYE",
    "is_supported",
    "adsorption_constant_at_T",
    "langmuir_coverage",
    "gamma_adsorbed",
    "coverage_per_additive",
    "AdditivePackage",
    "resolve_package",
    "surface_coverages",
    "nucleation_rate_multiplier",
    "h_recomb_fraction",
    "h_recomb_overpotential_reduction_V",
    "stress_relief_fraction",
    "carbon_incorporation_blocking",
    "structural_grade_score",
    "compare_packages",
]
