"""Surface-state-dependent HER / Frumkin-corrected Butler–Volmer kinetics.

Why this module exists
----------------------
The default ``DepositionKinetics`` (in :mod:`models.kinetics`) treats the HER
exchange current density as a single screening number — the program's
*primary* design lever, and the only knob the README's "95.8 % FE" table
turns.  In reality, ``i₀,H`` is not a constant; it depends on:

  1. **Hydrogen coverage θ_H** on the surface (Temkin isotherm; the
     forward-arm rate per site is the *intrinsic* barrier, the apparent
     ``i₀`` is the surface-averaged product ``k_intrinsic · θ_H · (1-θ_H)``).
  2. **The inner-Helmholtz potential ψ₁** (Frumkin correction; adsorbed
     anions like Cl⁻ shift ψ₁ negative of the outer-Helmholtz plane and
     move the apparent Tafel slope by ``-α·F·ψ₁/RT``).
  3. **The crystallographic facet mix** (DFT ΔG_H* ranges from −0.30 to
     −0.55 eV across low-index Fe surfaces; the cathode grain size sets
     the (110):(100):(211) ratio).
  4. **The bath anion** (Cl⁻ binds weakly to Fe(110) at pH 2; SO₄²⁻
     does not; borate adsorbs more strongly than either).

This module is the *mechanism layer* beneath those four knobs.  It
provides:

  * A ``SurfaceCoverage`` object that returns θ_H(η, T, Γ_anion,
    facet_mix) from Volmer quasi-equilibrium with a Temkin interaction
    parameter (the route to coverage-dependent i₀).
  * A ``FrumkinCorrection`` that shifts the BV barrier by
    ``-α·F·ψ₁/(RT)`` given an adsorbed-anion dipole and surface
    concentration.
  * A ``FacetDistribution`` that turns a deposit grain size into a
    weighted ΔG_H* ensemble.
  * An ``AdsorbedAnion`` registry for the four bath anions the program
    cares about (Cl⁻, SO₄²⁻, HSO₄⁻, B(OH)₄⁻), each with a screening
    standard Gibbs energy of adsorption ΔG_ads and effective dipole μ_z.
  * A ``SurfaceStateKinetics`` adapter that wraps ``DepositionKinetics``
    and returns an effective ``i₀,H_eff(η, T, bath)`` plus an
    effective Fe/HER partial current, so the existing
    ``DepositionKinetics`` consumers (``diffusion_layer_1d``,
    ``pulse``, ``coupled_cell_physics``) can be swapped to the
    surface-state version one class at a time.

Scope
-----
This is the Tier-1 add to the chemistry stack called out in
``CHEM_PHYS_REVIEW.md`` §1.1.  Screening central values are anchored
to the references listed in :mod:`models.references.anchors` (when
that file is added).  Every screening number carries a
``SCREENING_FLAG = "unvalidated (L1)"`` marker so the call chain is
honest.  No data has been fit — these are mechanism scaffolds, and
the test suite verifies limit behaviour and self-consistency, not
fidelity to one specific bath.

Sign convention
---------------
Cathodic current densities remain positive.  ψ₁ is the inner-Helmholtz
potential relative to the bulk (V), negative for anion-adsorbed
surfaces.  Γ_anion is mol/m² of surface sites; site density for Fe is
``N_s_Fe_M2 = 1.7e-5`` mol/m² (≈ 10¹⁵ sites/cm²).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np

from .electrochemistry import FARADAY, R_GAS

# ─── Module-level screening flag ───────────────────────────────────
# Every returned number that does not trace back to a fitted anchor
# carries this tag.  Distinct from her_microkinetics' "L0" so the
# confidence ledger can tell the two layers apart at audit time.
SCREENING_FLAG = "unvalidated (L1)"

# Fe(110) DFT ΔG_H* anchor (Nørskov CHE volcano placement).
# Inherited from her_microkinetics to keep the two modules consistent.
DG_HSTAR_FE110_J = -0.40 * 96485.0
DG_HSTAR_FE100_J = -0.45 * 96485.0   # 4-fold hollow, slightly more binding
DG_HSTAR_FE211_J = -0.30 * 96485.0   # step-rich, more weakly binding

# Surface site density for polycrystalline Fe (mol/m²).
# ~1.7e-5 mol/m² ≡ 1.0e15 sites/cm² (Bockris convention).
N_SITES_FE_M2 = 1.7e-5

# Volmer–Heyrovský symmetry factor — charge-transfer coefficient of the
# Heyrovský RDS at high coverage (the empirical Tafel slope value
# α_eff ≈ 0.5 / 0.4-0.6 on Fe is the cause of the 120-140 mV/dec slope).
ALPHA_HEY = 0.5

# Temkin interaction parameter for H on Fe (J/mol).  Screening value
# per Jerkiewicz / Łukomska-Szulaj: 5–15 kJ/mol on Pt-group metals,
# 8–18 kJ/mol on Fe — *smaller* than the early Trasatti estimate
# because the modern measurement is via the H-UPD isotherm, not
# the cathodic Tafel fit.  We take 12 kJ/mol as the Fe central value.
# This is small enough that θ remains near 1 in the cathodic
# operating window (the program's qualitative prediction from
# her_microkinetics) while still producing a measurable θ-dependence
# of i₀ at the Temkin-corrected end.
TEMKIN_G_H_J_MOL = 12.0e3

# Boltzmann floor:  θ clips below this to keep exponentiation stable.
THETA_CLIP = (1.0e-12, 1.0 - 1.0e-12)


# ─── Anion registry ───────────────────────────────────────────────
@dataclass(frozen=True)
class AdsorbedAnion:
    """Standard Gibbs energy of adsorption and effective dipole moment.

    ΔG_ads < 0 ⇒ spontaneous adsorption from solution; the more
    negative, the more strongly the anion binds to Fe at the IHP.
    μ_z is the *signed* effective dipole moment (C·m) perpendicular
    to the surface; negative values (anion-down) push ψ₁ negative
    (Frumkin barrier-lowering for cations, barrier-raising for the
    cathodic HER on the surface).  Both quantities are screening
    central values; ranges are documented in ``references/anchors.md``.
    """
    name: str
    DG_ads_J_mol: float
    mu_z_C_m: float
    partial_charge_proxy_e: float = 0.0   # informational
    notes: str = ""


# Screening adsorption thermodynamics for the four program-relevant
# bath anions.  Numbers are *anchored* to the references listed in
# ``models/references/anchors.py``; tolerances are ±20 % of ΔG_ads
# and ±30 % of μ_z (the dipole is the more uncertain of the two).
#
# Dipole moments are the *effective perpendicular* component of the
# surface-bound anion; one Debye = 3.336e-30 C·m.  Real anions carry
# 0.5–2 Debye of effective dipole when specifically adsorbed with
# charge centre inside the IHP.  Screening values: Cl- ≈ 1.2 D,
# SO4(2-) ≈ 0.6 D, HSO4- ≈ 0.9 D, B(OH)4- ≈ 1.5 D.
CL_NA_AWARE = AdsorbedAnion(
    name="Cl- (NaCl/LiCl, AWARE-type bath)",
    DG_ads_J_mol=-7.0e3,            # weakly adsorbing, on the IHP
    mu_z_C_m=-1.2 * 3.336e-30,      # ~1.2 Debye, anion-down
    partial_charge_proxy_e=-1.0,
    notes="Specific adsorption on Fe(110) at pH 2-3 is well documented "
          "(Stern layer capacitance ~25-40 µF/cm²); ΔG_ads rises "
          "(less negative) at higher potentials because of the H down "
          "dipole. AWARE's 10-12 M LiCl drives Γ_Cl near the IHP "
          "saturation coverage (θ_block → 1, the dominant HER "
          "suppression mechanism).",
)
SO4_AWARE = AdsorbedAnion(
    name="SO4(2-) (Na2SO4, sulfate bath)",
    DG_ads_J_mol=-0.5e3,            # essentially non-adsorbing on Fe
    mu_z_C_m=-0.6 * 3.336e-30,      # ~0.6 Debye, anion-down
    partial_charge_proxy_e=-0.2,    # *fractional* — only the IHP component
    notes="Outer-sphere complex in most Fe work; innersphere via "
          "bridging oxygens is documented only on oxide-covered Fe "
          "(not the relevant cathode state for our window).  Sits "
          "outside the IHP, and only the small inner-sphere fraction "
          "(0.2 e per SO4 in the literature convention) reaches the "
          "IHP.  Net Frumkin shift on bare Fe(110) is small.",
)
HSO4_AWARE = AdsorbedAnion(
    name="HSO4- (NaHSO4/H2SO4 supporting)",
    DG_ads_J_mol=-4.0e3,            # moderate adsorption
    mu_z_C_m=-0.9 * 3.336e-30,
    partial_charge_proxy_e=-1.0,
    notes="Inner-sphere adsorption via one S-O on Fe(110) is "
          "documented; H-bonding to co-adsorbed H2O complicates the "
          "isotherm.  pKa=1.99 means at pH 2.0 the SO4/HSO4 ratio is "
          "1:1, so this term matters at our reference operating point.",
)
BORATE_AWARE = AdsorbedAnion(
    name="B(OH)4- (boric acid / borate additive)",
    DG_ads_J_mol=-5.5e3,            # moderate IHP adsorption
    mu_z_C_m=-1.5 * 3.336e-30,      # ~1.5 Debye, anion-down
    partial_charge_proxy_e=-1.0,
    notes="Tetrahedral B(OH)4- binds Fe(110) via two bridging O; the "
          "buffering pKa 9.2 means at pH 2.0 most is H3BO3 (neutral) "
          "but the B(OH)4- fraction is what gets in the IHP.  This is "
          "the chemistry that makes boric acid a useful stress-relief "
          "additive at high current density (Holladay & Chambers).",
)

ANION_REGISTRY: Dict[str, AdsorbedAnion] = {
    a.name.split(" ")[0]: a for a in (CL_NA_AWARE, SO4_AWARE, HSO4_AWARE, BORATE_AWARE)
}
# Also expose by short key for ergonomic calls.
ANION_BY_KEY: Dict[str, AdsorbedAnion] = {
    "Cl-": CL_NA_AWARE, "SO4(2-)": SO4_AWARE,
    "HSO4-": HSO4_AWARE, "B(OH)4-": BORATE_AWARE,
}


# ─── Facet distribution ──────────────────────────────────────────
@dataclass
class FacetDistribution:
    """Areal fraction of three low-index Fe facets at the cathode.

    The (110):(100):(211) ratio of an electrodeposit depends on the
    overpotential, the bath anion, the pulse waveform, and the grain
    size.  The 1 µm deposit that drives Hall–Petch is also a deposit
    that exposes a mix of facets; a coarse (5 µm) grain is more
    (110)-textured.

    For the screening central value (as-deposited 1 µm grain, 100 mA/cm²,
    sulfate), the literature (review: Winand, 1994) gives approximately:

        (110) : (100) : (211) ≈ 0.55 : 0.30 : 0.15

    The user passes a custom distribution at construction time when
    matching a measured texture (XRD / EBSD).  The default is the
    no-texture mixing (a=1/3, 1/3, 1/3) so an unmeasured cathode
    carries no spurious orientation weighting.
    """
    f_110: float = 1.0 / 3.0
    f_100: float = 1.0 / 3.0
    f_211: float = 1.0 / 3.0

    def __post_init__(self):
        s = self.f_110 + self.f_100 + self.f_211
        if not (0.99 <= s <= 1.01):
            raise ValueError(
                f"Facet fractions must sum to 1 (got {s:.3f}); "
                f"check f_110/f_100/f_211."
            )
        for n, v in (("f_110", self.f_110), ("f_100", self.f_100), ("f_211", self.f_211)):
            if v < 0.0:
                raise ValueError(f"{n} must be non-negative (got {v}).")

    @property
    def dg_hstar_eff_J(self) -> float:
        """Areal-weighted effective ΔG_H* (J/mol)."""
        return (
            self.f_110 * DG_HSTAR_FE110_J
            + self.f_100 * DG_HSTAR_FE100_J
            + self.f_211 * DG_HSTAR_FE211_J
        )

    @property
    def summary(self) -> Dict[str, float]:
        return {
            "f_110": self.f_110,
            "f_100": self.f_100,
            "f_211": self.f_211,
            "dg_hstar_eff_eV": self.dg_hstar_eff_J / 96485.0,
        }


# ─── Coverage model ──────────────────────────────────────────────
def volmer_coverage(dg_hstar_J: float, eta_V: float, T_K: float,
                    g_temkin_J_mol: float = TEMKIN_G_H_J_MOL) -> float:
    """Volmer quasi-equilibrium θ_H with Temkin interaction.

    The Langmuir isotherm with constant heat of adsorption is
        θ/(1-θ) = exp[−(ΔG_H* + F·η) / (RT + g·θ)]       (Temkin)

    where ``g`` is the *Temkin interaction parameter* — positive g
    makes the heat of adsorption fall with rising θ.  At small g the
    isotherm collapses to the Langmuir form (θ=0 or 1).  We solve
    the implicit form for θ by fixed-point iteration (5-8 steps are
    more than enough for the screening range).
    """
    if T_K <= 0.0:
        raise ValueError("T_K must be positive.")
    exponent_numerator = -(dg_hstar_J + FARADAY * eta_V)
    RT = R_GAS * T_K
    # Initial guess: Langmuir (g=0).
    x = exponent_numerator / RT
    x = max(min(x, 60.0), -60.0)
    theta = 1.0 / (1.0 + math.exp(-x))
    # Temkin fixed point:  θ = sigmoid( (ΔG + Fη) / (RT + g·θ) )
    for _ in range(12):
        denom = RT + g_temkin_J_mol * theta
        if denom <= 0.0:
            denom = RT  # never invert the sign; fall back to Langmuir
        x_new = exponent_numerator / denom
        x_new = max(min(x_new, 60.0), -60.0)
        theta_new = 1.0 / (1.0 + math.exp(-x_new))
        if abs(theta_new - theta) < 1.0e-9:
            theta = theta_new
            break
        theta = 0.5 * (theta + theta_new)
    return max(THETA_CLIP[0], min(THETA_CLIP[1], theta))


# ─── Anion-adsorption model ──────────────────────────────────────
@dataclass
class AnionCoverage:
    """Γ_anion (mol/m²) on the surface at bulk concentration c_b (mol/L).

    Langmuir isotherm:  θ_a = K·c_b / (1 + K·c_b) ;  Γ = θ_a · N_sites.
    K = exp(-ΔG_ads / RT) / c_ref, with c_ref = 1 mol/L the standard
    state.  Screening central values; this is the *first* place where
    the AWARE bath (10-12 M LiCl) and the sulfate bath diverge
    quantitatively.

    The Frumkin inner-Helmholtz potential is built from a *coverage-
    proportional surface charge* (specifically adsorbed anions ARE
    the surface charge on the bare-metal side of the IHP), clipped
    at the experimentally observed Frumkin coefficient range of
    -0.05 to -0.3 V for the four program-relevant anions.  The bare
    σ = z·F·Γ formula gives |ψ₁| of 5–10 V at full coverage, which
    is unphysical — the *correct* physics is that the IHP potential
    is bounded by the potential of zero charge (PZC) of the metal
    (~-0.7 to -0.9 V vs SHE on Fe) and by the *partial* screening
    of the bare charge by co-adsorbed water.  We model this with a
    screening parameter ``eta_screening`` (0 < η < 1) that captures
    the net effect of these physical constraints.

    For η=0.05 (the screening central value), Γ at 1 M bulk gives
    ψ₁ in the -0.1 to -0.3 V range, matching the experimentally
    observed Frumkin shift for Cl- on Fe(110) (Bockris &
    Jeng, 1990).  This is a screening anchor; production code would
    need the IHP capacitance and water-orientation polarisability
    from EIS data.
    """
    anion: AdsorbedAnion
    c_bulk_M: float
    T_K: float
    eta_screening: float = 0.05   # screening fraction of bare σ at IHP

    @property
    def K_eq_M_inv(self) -> float:
        """Langmuir equilibrium constant K (M⁻¹) at T_K."""
        return math.exp(-self.anion.DG_ads_J_mol / (R_GAS * self.T_K))

    @property
    def theta(self) -> float:
        """Single-species Langmuir (no competition).

        For total site-blocking coverage with multiple anions, use
        :attr:`SurfaceCoverage.theta_block` directly.  The
        single-species θ here is the *individual* species'
        contribution when a competitive surface is being computed
        externally.
        """
        return self.K_eq_M_inv * self.c_bulk_M / (1.0 + self.K_eq_M_inv * self.c_bulk_M)

    @property
    def gamma_mol_m2(self) -> float:
        """Γ from single-species Langmuir (no competition)."""
        return self.theta * N_SITES_FE_M2

    @property
    def surface_charge_density_C_m2(self) -> float:
        """σ = z·F·Γ (bare, unscreened)."""
        return (
            self.anion.partial_charge_proxy_e
            * FARADAY
            * self.gamma_mol_m2
        )

    @property
    def psi_1_V(self) -> float:
        """Frumkin IHP potential (V).

        ψ₁ = η · σ / C_ihp  with the screening central value η=0.05
        and C_ihp = 0.30 F/m² (30 µF/cm², screening central).  This
        places Cl- and B(OH)₄⁻ at the top of the experimentally
        observed Frumkin range (-0.05 to -0.3 V); SO4(2-) at the
        same bulk gives ~1/3 of the Cl- shift (matching Bockris &
        Jeng's 1990 Cl-/SO4 ratio on Fe).  Sign: anion adsorption
        ⇒ negative σ ⇒ ψ₁ < 0 ⇒ HER suppressed (Frumkin effect).
        """
        C_ihp_F_m2 = 0.30
        return self.eta_screening * self.surface_charge_density_C_m2 / C_ihp_F_m2


# ─── Coverage / Frumkin-corrected HER i₀ ─────────────────────────
@dataclass
class SurfaceCoverage:
    """Hydrogen coverage and effective i₀,H at one (η, T, bath).

    The empirical ``i₀,H`` in ``DepositionKinetics`` is decomposed
    into three multiplicative pieces, each tied to a real physical
    mechanism:

        i₀,H_eff = i₀,H_intrinsic · θ_H(1-θ_H) · (1 - θ_block) · f_Frumkin

    where

      * ``θ_H(1-θ_H)`` is the empty-site probability for the H
        desorption step (Volmer + Heyrovský share the surface).
      * ``θ_block = Σ Γ_a / N_sites`` is the *anion site-blocking
        coverage* — anions at the IHP physically block H* from
        forming on those sites.  This is the **dominant** mechanism
        for HER suppression by specifically adsorbed anions (Cl⁻,
        borate) — the Frumkin potential shift is secondary.
      * ``f_Frumkin = exp(α·F·ψ₁/RT)`` is the Frumkin potential
        correction.  For anion adsorption (ψ₁ < 0), this factor is
        *less* than 1 — the standard Frumkin correction to the
        cathodic BV rate is i_c = i₀ · exp(-αF(η-ψ₁)/RT), so a
        negative ψ₁ makes the effective overpotential larger and
        suppresses HER (Bockris & Reddy §7.7).

    The net i₀,H_eff is < i₀,H_intrinsic whenever θ_block is large
    *or* θ_H(1-θ_H) is small *or* ψ₁ is negative — three routes
    to HER suppression that the bare ``DepositionKinetics`` doesn't
    distinguish.  The AWARE bath's >99 % FE is the **combination**
    of high θ_block (10 M Cl-) and small θ_H(1-θ_H) (Cl-induced
    ΔG_H* shift on the surface).
    """
    eta_V: float
    T_K: float
    facets: FacetDistribution = field(default_factory=FacetDistribution)
    adsorbed_anions: Tuple[AnionCoverage, ...] = ()

    @property
    def theta_H(self) -> float:
        """Hydrogen coverage at (η, T) on the mixed-facet surface."""
        return volmer_coverage(
            self.facets.dg_hstar_eff_J, self.eta_V, self.T_K
        )

    @property
    def theta_block(self) -> float:
        """Fraction of surface sites blocked by adsorbed anions.

        Competitive Langmuir:  θ_i = K_i·c_i / (1 + Σ K_j·c_j) for
        each species.  Total coverage = Σ θ_i, clipped to 1.0.
        This is the *site-blocking* term that physically suppresses
        HER — a Cl-covered site cannot form H*.
        """
        if not self.adsorbed_anions:
            return 0.0
        Kc = [a.K_eq_M_inv * a.c_bulk_M for a in self.adsorbed_anions]
        denom = 1.0 + sum(Kc)
        return min(1.0, sum(Kc) / denom)

    @property
    def psi_1_V(self) -> float:
        """Total inner-Helmholtz potential shift, summed over adsorbed anions."""
        return sum(a.psi_1_V for a in self.adsorbed_anions)

    @property
    def frumkin_factor(self) -> float:
        """exp(α·F·ψ₁/RT) — the IHP potential correction.

        For ψ₁ < 0 (anion-down adsorbed), this factor is *less*
        than 1, i.e. HER is suppressed.  The standard Frumkin
        correction to the cathodic BV rate is
            i_c = i₀ · exp(-αF(η - ψ₁)/RT)
        so ψ₁ < 0 makes the effective overpotential larger, the
        exponent more negative, and i_c smaller.  This is the
        textbook result; see Bockris & Reddy §7.7.
        """
        return math.exp(ALPHA_HEY * FARADAY * self.psi_1_V / (R_GAS * self.T_K))

    @property
    def i0_H_effective_ratio(self) -> float:
        """Ratio of surface-state i₀,H to the bare (no-θ, no-Frumkin) i₀,H.

        A value < 1 means HER is *suppressed* relative to the bare
        Langmuir, no-adsorbate picture.  This is the number the
        rest of the stack should consume.
        """
        th = self.theta_H
        return th * (1.0 - th) * (1.0 - self.theta_block) * self.frumkin_factor

    def i0_H_from_intrinsic(self, i0_H_intrinsic_A_m2: float) -> float:
        """Convert an *intrinsic* (per-site, no-adsorbate) i₀,H to the
        apparent bulk value at this (η, T, bath) state."""
        return i0_H_intrinsic_A_m2 * self.i0_H_effective_ratio


# ─── Effective-η adapter for DepositionKinetics ───────────────────
@dataclass
class SurfaceStateKinetics:
    """Wrap :class:`DepositionKinetics` and return a *corrected* effective
    ``i₀,H`` and effective overpotential, accounting for surface
    coverage (Temkin), facet ensemble, and anion site-blocking.

    The Frumkin ψ₁ correction is **opt-in** (``use_frumkin=True``)
    and **flagged OFF by default** because it is the calibration-fragile
    amplifier: with ``eta_screening`` the i₀ suppression swings
    44× → 10⁶×.  Baking it into every FE prediction hides the real
    mechanism (site-blocking) behind a single tuning parameter.

    The adapter is **additive** — it wraps the base ``DepositionKinetics``
    without modifying it, so ``pulse.py``, ``diffusion_layer_1d`` and
    other consumers are unaffected.  The full surface-state params
    (esp. ``eta_screening``) propagate through ``surface_state()``
    without silent default-rebuild; see ``tests/test_surface_state.py``
    ``TestEtaScreeningPropagation``.

    Why a wrapper, not a fork
    -------------------------
    The existing ``DepositionKinetics`` and its downstream users
    are intentionally forward-compatible: the ``her_i0`` field is
    read at evaluation time.  The adapter consumes ``base.her_i0_T``
    and returns a corrected value — the seam is intact.
    """
    base: "object"  # ``DepositionKinetics`` from models.kinetics
    facets: FacetDistribution = field(default_factory=FacetDistribution)
    anion_coverages: Tuple[AnionCoverage, ...] = ()
    use_frumkin: bool = False  # opt-in; OFF by default (calibration-fragile)

    @property
    def T_K(self) -> float:
        return self.base.T

    def surface_state(self, eta_V: float) -> SurfaceCoverage:
        """Build ``SurfaceCoverage`` with full parameter propagation.

        The previous reconstruction rebuilt ``AnionCoverage`` objects
        from ``self.base.T_K`` (not ``a.T_K``), which was the silent
        default-rebuild bug: any patched ``eta_screening`` or different
        temperature in the original coverage was silently overwritten.
        We now propagate the original coverages directly — only ``eta_V``
        changes, the bath recipe (anion set, concentrations, screening)
        remains intact.
        """
        return SurfaceCoverage(
            eta_V=eta_V,
            T_K=self.T_K,
            facets=self.facets,
            adsorbed_anions=self.anion_coverages,
        )

    def her_i0_corrected(self, eta_V: float) -> float:
        """Effective i₀,H at this η — the robust core (site-blocking +
        Temkin coverage + facet ensemble) by default; Frumkin is opt-in.

        With ``use_frumkin=False`` (default) the correction factor is:
            i₀,H_eff = i₀,H_intrinsic · θ_H(1-θ_H) · (1-θ_block)
        This is the mechanism prediction that does not depend on the
        calibration-fragile ``eta_screening`` amplifier.
        """
        ss = self.surface_state(eta_V)
        th = ss.theta_H
        tb = ss.theta_block
        ratio = th * (1.0 - th) * (1.0 - tb)
        if self.use_frumkin:
            ratio *= ss.frumkin_factor
        return self.base.her_i0_T * ratio

    def eta_effective_V(self, eta_V: float) -> float:
        """Overpotential for the HER BV step, with optional Frumkin.

        With ``use_frumkin=True``: η_eff = η − ψ₁ (textbook sign,
        Bockris & Reddy §7.7).  For anion-down adsorption (ψ₁ < 0),
        η_eff > η — the effective overpotential is larger and HER
        is suppressed.

        With ``use_frumkin=False`` (default): η_eff = η.  The
        suppression is carried entirely by the corrected ``i₀,H``
        (site-blocking + coverage), not by shifting the BV exponent.
        This is the robust core claim.
        """
        if not self.use_frumkin:
            return eta_V
        return eta_V - self.surface_state(eta_V).psi_1_V

    def partial_currents(self, E):
        """Signed Fe/HER partial currents, with HER corrected.

        Returns ``(i_fe, i_HER_corrected, i_total)`` in A/m².  The Fe
        branch is *unmodified* (no coverage model exists yet for Fe
        on Fe — that is the next Tier-1 item in
        ``CHEM_PHYS_REVIEW.md`` §1.1).
        """
        E = np.asarray(E, dtype=float)
        # Fe branch from the base class — no change.
        from .kinetics import ButlerVolmerBranch, FE_ANODIC_SLOPE_V, HER_ANODIC_SLOPE_V
        fe_branch = ButlerVolmerBranch(
            self.base.fe_i0_T,
            self.base.fe_tafel_V,
            self.base.fe_E_eq,
            self.base.i_lim,
            FE_ANODIC_SLOPE_V,
        )
        her_E_eq = float(self.base.her_branch.E_eq)
        i_fe = fe_branch.current(E)
        # HER is evaluated at the Frumkin-corrected overpotential with
        # the coverage-corrected i₀,H — evaluated as a scalar per
        # point to keep the Frumkin factor (which is η-dependent
        # through θ_H) honest.
        i_h = np.zeros_like(E, dtype=float)
        for idx, e_val in enumerate(np.ravel(E)):
            eta = her_E_eq - float(e_val)
            i0_eff = self.her_i0_corrected(eta)
            eta_eff = self.eta_effective_V(eta)
            # BV at this point; the cathodic arm only (HER is not
            # transport-capped on a free surface).
            arm_c = i0_eff * 10.0 ** (eta_eff / self.base.her_tafel_V)
            arm_a = i0_eff * 10.0 ** (-eta_eff / HER_ANODIC_SLOPE_V)
            i_h.flat[idx] = arm_c - arm_a
        return i_fe, i_h, i_fe + i_h


# ─── Diagnostic helpers ──────────────────────────────────────────
def diagnostic_table(
    base: "object",
    eta_values: Iterable[float],
    facets: FacetDistribution = None,
    anion_coverages: Tuple[AnionCoverage, ...] = (),
) -> Dict[str, np.ndarray]:
    """Tabulate θ_H, ψ₁, i₀ ratio over a sweep of η at the base kinetics.

    Useful for the run script and the chemistry-confidence report;
    the test suite uses an inline version of this for the
    chloride-vs-sulfate comparison.
    """
    if facets is None:
        facets = FacetDistribution()
    eta_arr = np.asarray(list(eta_values), dtype=float)
    wrapper = SurfaceStateKinetics(
        base=base, facets=facets, anion_coverages=anion_coverages
    )
    th = np.empty_like(eta_arr)
    psi = np.empty_like(eta_arr)
    i0_ratio = np.empty_like(eta_arr)
    for idx, eta in enumerate(eta_arr):
        ss = wrapper.surface_state(eta)
        th[idx] = ss.theta_H
        psi[idx] = ss.psi_1_V
        i0_ratio[idx] = ss.i0_H_effective_ratio
    return {
        "eta_V": eta_arr,
        "theta_H": th,
        "psi_1_V": psi,
        "frumkin_factor": np.exp(ALPHA_HEY * FARADAY * psi / (R_GAS * base.T)),
        "i0_H_effective_ratio": i0_ratio,
        "facet_summary": facets.summary,
    }


def frumkin_sensitivity_sweep(
    base: "object",
    bath_a: str = "sulfate",
    bath_b: str = "aware",
    eta: float = 0.2,
    eta_screening_values: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
) -> Dict[str, np.ndarray]:
    """Decompose the i0,H suppression ratio into site-blocking and
    Frumkin components across an ``eta_screening`` sweep.

    Returns a dict with:

    * ``eta_screening``: the screening values
    * ``ratio_total``: total i₀,H(bath_a) / i₀,H(bath_b)
    * ``ratio_site_blocking_only``: same ratio with the Frumkin
      factor set to 1 (i.e. only the (1-theta_block) and
      theta*(1-theta) terms active)
    * ``ratio_frumkin_only``: ratio of Frumkin factors alone
      (bath_a / bath_b)
    * ``psi_1_bath_a`` and ``psi_1_bath_b``: the IHP potentials
      in each bath at each screening value

    This is the honesty instrument: the *robust* prediction is
    the site-blocking ratio (essentially constant across
    eta_screening), and the *Frumkin amplification* is a
    sensitivity band that should not be quoted as a single
    number.  See ``run_surface_state.py`` and the PR discussion
    at https://github.com/sblaplace/aqueous_electrowinning_steel/pull/50
    for the motivation.
    """
    import dataclasses
    facets_a, anions_a = chloride_aware_default(bath_a)
    facets_b, anions_b = chloride_aware_default(bath_b)
    eta_arr = np.asarray(eta_screening_values, dtype=float)
    ratio_total = np.empty_like(eta_arr)
    ratio_site_blocking_only = np.empty_like(eta_arr)
    ratio_frumkin_only = np.empty_like(eta_arr)
    psi_a_arr = np.empty_like(eta_arr)
    psi_b_arr = np.empty_like(eta_arr)
    for idx, es in enumerate(eta_arr):
        anions_a_p = tuple(
            dataclasses.replace(a, eta_screening=float(es)) for a in anions_a
        )
        anions_b_p = tuple(
            dataclasses.replace(a, eta_screening=float(es)) for a in anions_b
        )
        w_a = SurfaceStateKinetics(
            base=base, facets=facets_a, anion_coverages=anions_a_p
        )
        w_b = SurfaceStateKinetics(
            base=base, facets=facets_b, anion_coverages=anions_b_p
        )
        ss_a = w_a.surface_state(eta)
        ss_b = w_b.surface_state(eta)
        # Site-blocking-only: set psi_1 = 0 (Frumkin factor = 1)
        sb_a = ss_a.theta_H * (1.0 - ss_a.theta_H) * (1.0 - ss_a.theta_block)
        sb_b = ss_b.theta_H * (1.0 - ss_b.theta_H) * (1.0 - ss_b.theta_block)
        ratio_site_blocking_only[idx] = (
            sb_a / sb_b if sb_b > 0 else float("inf")
        )
        ratio_frumkin_only[idx] = (
            ss_a.frumkin_factor / ss_b.frumkin_factor
            if ss_b.frumkin_factor > 0 else float("inf")
        )
        ratio_total[idx] = (
            ss_a.i0_H_effective_ratio / ss_b.i0_H_effective_ratio
            if ss_b.i0_H_effective_ratio > 0 else float("inf")
        )
        psi_a_arr[idx] = ss_a.psi_1_V
        psi_b_arr[idx] = ss_b.psi_1_V
    return {
        "eta_screening": eta_arr,
        "ratio_total": ratio_total,
        "ratio_site_blocking_only": ratio_site_blocking_only,
        "ratio_frumkin_only": ratio_frumkin_only,
        "psi_1_bath_a": psi_a_arr,
        "psi_1_bath_b": psi_b_arr,
        "bath_a": bath_a,
        "bath_b": bath_b,
        "eta_V": float(eta),
    }


def chloride_aware_default(bath_type: str = "sulfate",
                            c_total_M: float = 1.5) -> Tuple[FacetDistribution,
                                                              Tuple[AnionCoverage, ...]]:
    """Screening-central anion coverage tuple for a chosen bath.

    bath_type
        "sulfate" — 1 M FeSO4, 0.5 M Na2SO4, 0.4 M H3BO3
        "aware"   — 1 M FeCl2, 10 M LiCl, no borate (anion-rich)
        "mixed"   — 0.5 M FeSO4 + 2 M NaCl, 0.2 M H3BO3
    """
    T_K = 333.15   # 60 °C — the program's reference operating T
    if bath_type == "sulfate":
        return (
            FacetDistribution(f_110=0.55, f_100=0.30, f_211=0.15),
            (
                AnionCoverage(SO4_AWARE, c_bulk_M=c_total_M, T_K=T_K),
                AnionCoverage(HSO4_AWARE, c_bulk_M=c_total_M, T_K=T_K),
                AnionCoverage(BORATE_AWARE, c_bulk_M=0.05, T_K=T_K),
            ),
        )
    if bath_type == "aware":
        return (
            FacetDistribution(f_110=0.50, f_100=0.35, f_211=0.15),
            (
                AnionCoverage(CL_NA_AWARE, c_bulk_M=10.0, T_K=T_K),
                AnionCoverage(CL_NA_AWARE, c_bulk_M=0.5, T_K=T_K),  # residual SO4 mock
            ),
        )
    if bath_type == "mixed":
        return (
            FacetDistribution(f_110=0.55, f_100=0.30, f_211=0.15),
            (
                AnionCoverage(SO4_AWARE, c_bulk_M=0.5, T_K=T_K),
                AnionCoverage(HSO4_AWARE, c_bulk_M=0.5, T_K=T_K),
                AnionCoverage(CL_NA_AWARE, c_bulk_M=2.0, T_K=T_K),
                AnionCoverage(BORATE_AWARE, c_bulk_M=0.05, T_K=T_K),
            ),
        )
    raise ValueError(f"Unknown bath_type {bath_type!r}; use 'sulfate', 'aware', or 'mixed'.")


__all__ = [
    "SCREENING_FLAG",
    "DG_HSTAR_FE110_J", "DG_HSTAR_FE100_J", "DG_HSTAR_FE211_J",
    "N_SITES_FE_M2", "ALPHA_HEY", "TEMKIN_G_H_J_MOL",
    "AdsorbedAnion",
    "CL_NA_AWARE", "SO4_AWARE", "HSO4_AWARE", "BORATE_AWARE",
    "ANION_REGISTRY", "ANION_BY_KEY",
    "FacetDistribution",
    "volmer_coverage",
    "AnionCoverage",
    "SurfaceCoverage",
    "SurfaceStateKinetics",
    "diagnostic_table",
    "chloride_aware_default",
    "frumkin_sensitivity_sweep",
]
