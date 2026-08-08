"""
Potential- and double-layer-aware charge-transfer kinetics (CHEM_PHYS_REVIEW.md §2.3).

Why this module exists
----------------------
The base ``DepositionKinetics`` (models.kinetics) holds the Tafel slopes and
symmetry factors α *constant*.  §2.3 calls out that in reality α — and hence
the BV slope RT/αF — is mildly potential-dependent (Marcus-like, most apparent
at high |η| ≳ 200 mV), and that the pre-exponential carries an entropic /
activation term that varies with double-layer structure (the Frumkin
correction).  This module supplies the *mechanism layer* for that:

  * :func:`alpha_eff` — a Marcus-like, potential- and double-layer-dependent
    effective transfer coefficient α_eff(η, ψ₁).  It declines with cathodic
    overpotential (the "alpha drop"), and grows in magnitude as ψ₁ is shifted
    (Frumkin).
  * :func:`heat_of_activation` — the pre-exponential "heat-of-activation"
    correction: the Marcus quadratic barrier ΔG‡ = (λ + nF·η_eff)²/(4λ)
    re-expressed as a multiplier on i₀ relative to the equilibrium barrier
    λ/4.  It is *exactly* the BV exponent at small η_eff and adds the
    quadratic (η²) Marcus deviation at high η_eff, which is the part that
    makes high-|η| predictions more honest.
  * :func:`organic_psi1_shift` — model leveler / additive adsorption (how
    saccharin, thiourea, chloride actually work) as a *double-layer* shift:
    ψ₁ is shifted by the adsorbed organic dipole layer
    ``Δψ₁ = Γ_organic · N_A · μ_dipole / (ε₀·ε_r)``.
  * :class:`FrumkinCorrectedBV` — a signed Butler–Volmer branch whose Tafel
    slope is re-derived per-point from α_eff(η,ψ₁) and whose pre-exponential
    carries the heat-of-activation correction.  Used to wire α_eff into the
    HER and Fe BV branches behind an opt-in flag, leaving the default path
    byte-identical.

Scope
-----
L0 screening scaffold (matches the rest of the twin stack).  Every screening
number carries a ``SCREENING_FLAG`` so the confidence ledger can find it at
audit time.  No data has been fit.  The test suite verifies limit behaviour
and self-consistency (monotonic α direction, leveler ψ₁ shift, flag-off =
default), not fidelity to one specific bath.

Sign convention
---------------
Cathodic current densities remain positive.  η is the *cathodic* overpotential
(positive; E_eq − E).  ψ₁ is the inner-Helmholtz potential relative to bulk (V),
negative for anion / organic-dipole-down adsorption.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .electrochemistry import FARADAY, R_GAS

SCREENING_FLAG = "unvalidated (L1)"

# ─── Physical constants ─────────────────────────────────────────────
N_AVOGADRO = 6.02214076e23   # 1/mol
EPSILON_0 = 8.8541878128e-12  # F/m (vacuum permittivity)
DEBYE = 3.33564e-30            # C·m (one Debye)

# ─── Screening central values ───────────────────────────────────────
# Reorganization energy λ for Fe²⁺/Fe (and HER on Fe) charge transfer.
# Metal-deposition / outer-sphere λ's are typically 1–2 eV; we take
# 1.5 eV × F as the central value (unvalidated L1).
LAMBDA_DEFAULT_J_MOL = 1.5 * FARADAY
ALPHA0_DEFAULT = 0.5
ALPHA_MIN_DEFAULT = 0.05        # hard floor so α stays positive/sane
T_REF_K = 298.15
# Organic-layer dielectric constant used in the leveler dipole shift
# (screenings commonly cite ε_r ≈ 3–10 for an adsorbed organic film;
#  we take 6 as central).
EPS_R_ORGANIC_DEFAULT = 6.0


def alpha_eff(
    eta_V: float,
    psi_1_V: float = 0.0,
    n: int = 1,
    lambda_J_mol: float = LAMBDA_DEFAULT_J_MOL,
    alpha0: float = ALPHA0_DEFAULT,
    alpha_min: float = ALPHA_MIN_DEFAULT,
) -> float:
    """Marcus-like, double-layer-aware effective transfer coefficient.

    α_eff(η, ψ₁) = α₀ − (n·F·η_eff) / (2·λ),  η_eff = η − ψ₁

    where η_eff is the overpotential at the reaction plane (Frumkin-corrected
    by ψ₁).  It declines linearly with cathodic overpotential — the
    "alpha drop" that makes high-|η| (> 200 mV) BV more honest — and the
    Frumkin shift ψ₁ enters through η_eff (a more negative ψ₁ raises η_eff and
    hence lowers α_eff further).  Clamped to ``[alpha_min, 1.0]``.

    Parameters
    ----------
    eta_V : float
        Cathodic overpotential (V), positive.
    psi_1_V : float
        Inner-Helmholtz-plane potential (V) relative to bulk; negative for
        anion / organic-dipole-down adsorption.
    n : int
        Electrons transferred in the rate-determining step (HER n=1, Fe n=2
        for the full two-electron step — use the RDS value when known).
    lambda_J_mol : float
        Reorganization energy (J/mol).
    alpha0 : float
        Symmetry factor at η_eff = 0 (≈ 0.5).
    alpha_min : float
        Hard lower clamp so α stays positive.
    """
    eta_eff = float(eta_V) - float(psi_1_V)
    alpha = alpha0 - (n * FARADAY * eta_eff) / (2.0 * lambda_J_mol)
    return float(max(alpha_min, min(1.0, alpha)))


def tafel_slope_from_alpha(alpha: float, n: int = 1, T_K: float = T_REF_K) -> float:
    """Cathodic Tafel slope (V/decade) implied by a transfer coefficient.

    b = 2.303·R·T / (α·n·F)  — used to turn each α_eff into the local slope
    the BV branch actually integrates.
    """
    return 2.303 * R_GAS * T_K / (alpha * n * FARADAY)


def heat_of_activation(
    eta_V: float,
    psi_1_V: float = 0.0,
    n: int = 1,
    lambda_J_mol: float = LAMBDA_DEFAULT_J_MOL,
    T_K: float = T_REF_K,
) -> float:
    """Multiplicative i₀ correction from the Marcus *quadratic* barrier.

    The full Marcus barrier for the reduction branch is
        ΔG‡(η,ψ₁) = (λ − n·F·η_eff)² / (4λ),   η_eff = η − ψ₁
    whose linear-in-η term (−nF·η_eff/2) is *already* carried by the α_eff
    slope (at α₀=0.5, 10^(η_eff/b) = exp(+nF·η_eff/2RT)).  The part of the
    barrier that constant-α BV omits is the quadratic curvature
        ΔG‡_quad = (n·F·η_eff)² / (4λ)
    This correction is therefore the pure heat-of-activation deviation from
    constant-α BV:
        f_act = exp[ −(n·F·η_eff)² / (4·λ·R·T) ]   ≤ 1 for all η_eff ≠ 0.

    At small |η| this is ≈ 1 (BV recovered); at high |η| it bends the branch
    down (the "inverted-region" curvature), the honest high-η (>200 mV)
    behaviour §2.3 asks for, *on top of* the α_eff slope drop.
    """
    eta_eff = float(eta_V) - float(psi_1_V)
    lam = float(lambda_J_mol)
    if lam <= 0.0:
        raise ValueError("lambda_J_mol must be positive")
    dG_quad = (n * FARADAY * eta_eff) ** 2 / (4.0 * lam)
    return float(math.exp(-dG_quad / (R_GAS * T_K)))


def organic_psi1_shift(
    gamma_organic_mol_m2: float,
    mu_dipole_C_m: float,
    eps_r: float = EPS_R_ORGANIC_DEFAULT,
    eps_0: float = EPSILON_0,
    n_avogadro: float = N_AVOGADRO,
) -> float:
    """Double-layer shift from an adsorbed organic leveler / additive.

    A dipole layer of areal density Γ·N_A (dipoles/m²), each of effective
    dipole moment μ_z normal to the surface, produces a Helmholtz potential
    step
        Δψ₁ = Γ · N_A · μ_z / (ε₀·ε_r)   (V).

    This is the mechanism by which saccharin, thiourea, chloride, coumarin
    really act: they adsorb with a characteristic dipole, shifting the
    inner-Helmholtz potential (and hence α_eff through η_eff) rather than by
    a single lumped ``additive`` factor.  μ_z is the *signed* effective
    dipole (C·m); a dipole-down organic (negative end toward the metal →
    negative μ_z convention here) shifts ψ₁ negative, which increases η_eff
    and further lowers α_eff (better leveling), consistent with the
    Frumkin picture in models.surface_state.

    Screening values for effective dipoles (the uncertain ±30 % number):
    chloride ≈ 1.2 D, saccharin ≈ 2 D, thiourea ≈ 2–4 D, coumarin ≈ 1.5 D.
    """
    return -float(gamma_organic_mol_m2 * n_avogadro * mu_dipole_C_m) / (eps_0 * eps_r)


# ─── Opt-in double-layer / alpha-eff state ─────────────────────────
@dataclass
class FrumkinParams:
    """Screening-level double-layer + Marcus parameters for one branch.

    Exposed to let callers keep default path untouched while opting in
    through ``DepositionKinetics.use_frumkin_alpha_eff``.  ψ₁ here is the
    *resolved* inner-Helmholtz potential (V) — e.g. from
    ``surface_state.SurfaceCoverage.psi_1_V`` plus any
    :func:`organic_psi1_shift` from a leveler.
    """

    psi_1_V: float = 0.0
    n: int = 1
    alpha0: float = ALPHA0_DEFAULT
    alpha_min: float = ALPHA_MIN_DEFAULT
    lambda_J_mol: float = LAMBDA_DEFAULT_J_MOL
    T_K: float = T_REF_K


@dataclass
class FrumkinCorrectedBV:
    """Signed Butler–Volmer branch with α_eff(η,ψ₁) + heat-of-activation.

    Mirrors ``kinetics.ButlerVolmerBranch``'s signed current convention
    (cathodic positive, exactly 0 at E_eq, net oxidation anodic of it) but
    re-derives the local Tafel slope from :func:`alpha_eff` and multiplies the
    pre-exponential by :func:`heat_of_activation` at every point.  This is
    the "wire α_eff into the HER and Fe BV branches behind a flag" object:
    leave ``frumkin`` as None in the parent to get the unmodified branch.

    Parameters
    ----------
    i0 : float
        Exchange current density (A/m²), already Arrhenius-scaled to T.
    E_eq : float
        Equilibrium potential of the couple (V vs SHE).
    params : FrumkinParams
        α_eff / double-layer parameters for this branch.
    i_lim : float or None
        Diffusion-limiting current density (A/m²); blends the cathodic arm
        only (as ``ButlerVolmerBranch``).
    anodic_alpha0 : float
        Symmetry factor for the anodic (oxidation) arm at η_eff=0; the anodic
        slope is held constant (no anodic α-drop in this screening tier).
    """

    i0: float
    E_eq: float
    params: FrumkinParams
    i_lim: Optional[float] = None
    anodic_alpha0: float = ALPHA0_DEFAULT

    def _arm_c(self, eta_V):
        """Cathodic (magnitude) factor: i0 · f_act · 10^(η_eff/b(η_eff))."""
        a = alpha_eff(
            eta_V,
            self.params.psi_1_V,
            self.params.n,
            self.params.lambda_J_mol,
            self.params.alpha0,
            self.params.alpha_min,
        )
        b = tafel_slope_from_alpha(a, self.params.n, self.params.T_K)
        eta_eff = float(eta_V) - self.params.psi_1_V
        f_act = heat_of_activation(
            eta_V,
            self.params.psi_1_V,
            self.params.n,
            self.params.lambda_J_mol,
            self.params.T_K,
        )
        return self.i0 * f_act * 10.0 ** (eta_eff / b), a

    def _arm_a(self, eta_V):
        """Anodic (oxidation) magnitude factor."""
        b_a = tafel_slope_from_alpha(self.anodic_alpha0, self.params.n, self.params.T_K)
        eta_eff = float(eta_V) - self.params.psi_1_V
        # oxidation arm carries the same heat-of-activation correction
        f_act = heat_of_activation(
            eta_V,
            self.params.psi_1_V,
            self.params.n,
            self.params.lambda_J_mol,
            self.params.T_K,
        )
        return self.i0 * f_act * 10.0 ** (-eta_eff / b_a)

    def current(self, E):
        """Signed current density (A/m²); cathodic positive."""
        E = np.asarray(E, dtype=float)
        eta = self.E_eq - E
        i = np.empty_like(eta, dtype=float)
        for idx, e_val in enumerate(np.ravel(eta)):
            arm_c, _ = self._arm_c(e_val)
            arm_a = self._arm_a(e_val)
            i.flat[idx] = arm_c - arm_a
        i = i.reshape(np.shape(eta))
        if self.i_lim is None:
            return i
        i_cat = np.where(i > 0.0, i, 0.0)
        blended = 1.0 / (1.0 / np.maximum(i_cat, 1e-30) + 1.0 / self.i_lim)
        return np.where(i > 0.0, blended, i)

    def current_magnitude(self, E):
        """Cathodic magnitude (A/m²); 0 anodic of E_eq."""
        return np.where(self.current(E) > 0.0, self.current(E), 0.0)


__all__ = [
    "SCREENING_FLAG",
    "N_AVOGADRO",
    "EPSILON_0",
    "DEBYE",
    "LAMBDA_DEFAULT_J_MOL",
    "ALPHA0_DEFAULT",
    "ALPHA_MIN_DEFAULT",
    "T_REF_K",
    "EPS_R_ORGANIC_DEFAULT",
    "alpha_eff",
    "tafel_slope_from_alpha",
    "heat_of_activation",
    "organic_psi1_shift",
    "FrumkinParams",
    "FrumkinCorrectedBV",
]
