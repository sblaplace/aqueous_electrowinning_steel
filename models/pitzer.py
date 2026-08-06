"""Pitzer ion-interaction activity model for Fe(II)-sulfate electrolytes.

Multicomponent Pitzer equations for the Fe²⁺–Na⁺–H⁺ ∥ SO₄²⁻–HSO₄⁻
system that constitutes the aqueous iron electrowinning bath.  This module
replaces the Davies-equation activity model previously used in
``speciation.py``, which is only valid to I ≈ 0.5 mol/kg while the
reference bath sits at I ≈ 2–4 mol/kg — far outside its calibrated range
(the result was a spurious prediction that ~97 % of dissolved iron is
bound as neutral FeSO₄⁰ pairs).

Conventions
-----------
* Work in **molality** (mol / kg H₂O).  ``speciation.py`` converts the
  molar bath recipe to molal via a documented density estimate.
* Standard state: 1 mol/kg, hypothetically ideal.  Single-ion activity
  coefficients are conventional (no MacInnes scaling applied).
* β² terms with α₂ = 12.0 apply only to 2–2 electrolytes (FeSO₄ association
  is absorbed into the virial coefficients — NO explicit ion-pair species
  is used; see module data block).
* Unsymmetrical same-sign mixing (electrostatic ᵈθ terms) is included via
  the Pitzer (1975) J-function fits, exactly as in PHREEQC's pitzer.dat
  implementation.  This matters here because Na⁺/Fe²⁺ (1:2 charge) mix.

Parameter provenance (25 °C)
-----------------------------
* Fe²⁺–SO₄²⁻ : Pitzer (1991) tabulation (Pitzer & Mayorga basis);
  validated below against the Kobylin et al. (2011) assessment anchor
  γ±(FeSO₄, 0.1 m) = 0.164 (Reardon & Beckie 1987 used 0.161).
* Na⁺–SO₄²⁻, Na⁺–HSO₄⁻, H⁺–SO₄²⁻, H⁺–HSO₄⁻ : Harvie, Møller & Weare
  (1984) set, as distributed in PHREEQC pitzer.dat.
* Fe²⁺–HSO₄⁻ : Pitzer (1991, p. 105) / Reardon & Beckie (1987, p. 2362).
* θ(H⁺,Na⁺) : PHREEQC pitzer.dat (Harvie-Møller-Weare).
* θ(Na⁺,Fe²⁺) : from Charykova et al. (2010) Na₂SO₄–FeSO₄ fits, as tabulated
  in the Sandia/WIPP Fe(II) solubility model documentation.
* θ(SO₄²⁻,HSO₄⁻) and all ψ triplets: 0.0 default (insufficient data for this
  specific system in the canonical databases; the nearest published value
  θ ≈ −0.1354 for SO₄²⁻–HSO₄⁻ is noted).  At the bath's pH ≥ 2 the HSO₄⁻
  population is ≤ a few mol% of total sulfate, so these terms are second
  order; they are exposed as parameters for refinement.

Temperature
-----------
The Debye–Hückel slope Aφ(T) is computed exactly from the water dielectric
constant and density.  The interaction parameters themselves are 25 °C
values; their enthalpic derivatives are not yet wired in (Reardon & Beckie
1987 provide fitted functions over 10–60 °C — a follow-up).  Over the
bath's 25–90 °C envelope the binary parameters drift slowly for sulfate
salts; flag results outside 10–60 °C as extrapolated.

References
----------
* Pitzer, K. S. (1973) J. Phys. Chem. 77, 268 — thermodynamics of
  electrolytes.
* Pitzer, K. S. (1991) in "Activity Coefficients in Electrolyte Solutions",
  2nd ed., CRC Press, ch. 3 — equations used here.
* Harvie, C. E., Møller, N., Weare, J. H. (1984) Geochim. Cosmochim. Acta
  48, 723 — parameter set (PHREEQC pitzer.dat).
* Reardon, E. J., Beckie, R. D. (1987) Geochim. Cosmochim. Acta 51,
  2355–2368 — FeSO₄–H₂SO₄–H₂O system.
* Kobylin, P., Sippola, H., Taskinen, P. (2011) Geochim. Cosmochim. Acta —
  FeSO₄–H₂O thermodynamic assessment; source of the γ±(0.1 m)=0.164 anchor.
* Charykova, M. V. et al. (2010) Russ. J. Appl. Chem. — Na-Fe sulfate mixing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

# ─── Debye–Hückel osmotic slope Aφ(T) ────────────────────────────────

_N_A = 6.02214076e23          # 1/mol
_E_CHARGE = 1.602176634e-19   # C
_EPS0 = 8.8541878128e-12      # C^2 / (J m)
_K_B = 1.380649e-23           # J/K

B_DH = 1.2                    # Pitzer's universal b parameter (kg/mol)^0.5


def water_dielectric(T_C: float) -> float:
    """Static dielectric constant of liquid water (Malmberg–Maryott-form fit).

    Same polynomial family as ``speciation.davies_A``; valid ≈ 0–100 °C.
    """
    return 87.740 - 0.4008 * T_C + 9.398e-4 * T_C**2 - 1.410e-6 * T_C**3


def water_density_kg_L(T_C: float) -> float:
    """Density of pure water, kg/L.  Simple fit, ±0.2 % over 0–100 °C."""
    T = T_C
    rho = 0.9998395 + 6.7982999e-5 * T - 9.1060256e-6 * T**2 + 1.0052729e-7 * T**3
    return max(rho, 0.95)


def A_phi(T_C: float = 25.0) -> float:
    """Debye–Hückel osmotic-coefficient slope Aφ in (kg/mol)^0.5.

    A_γ(molar) = (2π N_A d_w)^0.5 (e² / 4π ε₀ ε_r k_B T)^1.5   (d_w in kg/m³,
    I in mol/L), and Aφ (molal) = A_γ / (3 √d_w) with d_w in kg/L for the
    mol/L → mol/kg mapping.

    Sanity: Aφ(25 °C) = 0.3915 (literature).
    """
    T_K = T_C + 273.15
    eps_r = water_dielectric(T_C)
    d_kg_L = water_density_kg_L(T_C)
    pre = (2.0 * math.pi * _N_A * (d_kg_L * 1000.0)) ** 0.5
    coul = (_E_CHARGE**2 / (4.0 * math.pi * _EPS0 * eps_r * _K_B * T_K)) ** 1.5
    a_gamma_molar = pre * coul
    return float(a_gamma_molar / (3.0 * math.sqrt(d_kg_L)))


# ─── Pitzer function machinery ───────────────────────────────────────

def _g(x: float) -> float:
    """g(x) = 2 [1 − (1 + x) e^{−x}] / x²  (Pitzer's g)."""
    if x <= 0.0:
        return 1.0
    return 2.0 * (1.0 - (1.0 + x) * math.exp(-x)) / (x * x)


def _gp(x: float) -> float:
    """g′(x) = −2 [1 − (1 + x + x²/2) e^{−x}] / x²  (Pitzer's g prime)."""
    if x <= 0.0:
        return -1.0 / 3.0
    return -2.0 * (1.0 - (1.0 + x + 0.5 * x * x) * math.exp(-x)) / (x * x)


def _J(x: float) -> float:
    """Pitzer (1975) electrostatic unsymmetric-mixing integral J(x).

    J(x) = x / (4 + 4.581 x^{−0.7237} e^{−0.0120 x^{0.528}})
    (the same Chebyshev-derived fit used in PHREEQC pitzer.c).
    """
    return x / (4.0 + 4.581 * x ** (-0.7237) * math.exp(-0.0120 * x ** 0.528))


def _Jp(x: float) -> float:
    """dJ/dx by exact analytic differentiation of the Pitzer fit."""
    b = 4.581 * math.exp(-0.0120 * x ** 0.528)
    D = 4.0 + b * x ** (-0.7237)
    Dp = b * x ** (-0.7237) * (-0.7237 / x - 0.0120 * 0.528 * x ** (-0.472))
    return (D - x * Dp) / (D * D)


def _etheta(z_i: int, z_j: int, I: float, Aphi: float) -> float:
    """ᵈθ_ij(I) — electrostatic unsymmetric mixing term (Pitzer 1975)."""
    if z_i == z_j:
        return 0.0
    x_ij = 6.0 * z_i * z_j * Aphi * math.sqrt(I)
    x_ii = 6.0 * z_i * z_i * Aphi * math.sqrt(I)
    x_jj = 6.0 * z_j * z_j * Aphi * math.sqrt(I)
    return (z_i * z_j / (4.0 * I)) * (_J(x_ij) - 0.5 * _J(x_ii) - 0.5 * _J(x_jj))


def _etheta_prime(z_i: int, z_j: int, I: float, Aphi: float) -> float:
    """dᵈθ_ij/dI (Pitzer 1975)."""
    et = _etheta(z_i, z_j, I, Aphi)
    if et == 0.0:
        return 0.0
    x_ij = 6.0 * z_i * z_j * Aphi * math.sqrt(I)
    x_ii = 6.0 * z_i * z_i * Aphi * math.sqrt(I)
    x_jj = 6.0 * z_j * z_j * Aphi * math.sqrt(I)
    j2 = x_ij * _Jp(x_ij) - 0.5 * x_ii * _Jp(x_ii) - 0.5 * x_jj * _Jp(x_jj)
    return -et / I + (z_i * z_j / (8.0 * I * I)) * j2


# ─── Parameter database ─────────────────────────────────────────────

@dataclass(frozen=True)
class PitzerPair:
    """Cation–anion binary interaction parameters (molal scale, 25 °C)."""
    beta0: float
    beta1: float
    beta2: float
    Cphi: float
    alpha1: float = 2.0
    alpha2: float = 12.0
    ref: str = ""


# Key: (cation, anion).  Charges keyed in PITZER_CHARGES below.
PITZER_BINARY: Dict[Tuple[str, str], PitzerPair] = {
    # 2–2 sulfate — α1 = 1.4 is the Pitzer convention for 2–2 electrolytes.
    # FeSO4 association is absorbed in β2; no explicit pair species.
    ("Fe2+", "SO4-2"): PitzerPair(
        0.2568, 3.063, -42.42, 0.0213, alpha1=1.4,
        ref="Pitzer (1991) tabulation; γ±(0.1 m)→0.16 matches Kobylin et al. (2011)",
    ),
    # Harvie-Møller-Weare (1984) — PHREEQC pitzer.dat set.
    ("Na+", "SO4-2"): PitzerPair(0.01958, 1.113, 0.0, 0.00497, ref="Harvie et al. 1984 / PHREEQC pitzer.dat"),
    ("Na+", "HSO4-"): PitzerPair(0.0454, 0.398, 0.0, 0.0, ref="Harvie et al. 1984 / PHREEQC pitzer.dat"),
    ("H+", "SO4-2"): PitzerPair(0.0298, 0.0, 0.0, 0.0438, ref="Harvie et al. 1984 / PHREEQC pitzer.dat"),
    ("H+", "HSO4-"): PitzerPair(0.2065, 0.5556, 0.0, 0.0, ref="Harvie et al. 1984 / PHREEQC pitzer.dat"),
    # Reardon & Beckie (1987, p. 2362) via Pitzer (1991, p. 105).
    ("Fe2+", "HSO4-"): PitzerPair(0.4273, 3.48, 0.0, 0.0, ref="Pitzer 1991 / Reardon & Beckie 1987"),
    # ── Implementation-validation species (also chemically relevant) ──
    # NaCl: canonical 1–1 test electrolyte; Cl− also enters the AWARE
    # chloride-route baths.  MgSO4: canonical 2–2 test electrolyte with
    # well-tabulated γ±; Mg²⁺ is also a plausible waste-feed contaminant.
    ("Na+", "Cl-"): PitzerPair(0.0765, 0.2664, 0.0, 0.00127, ref="Harvie et al. 1984 / PHREEQC pitzer.dat"),
    ("Mg2+", "SO4-2"): PitzerPair(0.2210, 3.343, -37.23, 0.025, alpha1=1.4,
                                    ref="Pitzer 1991 tabulation (test electrolyte)"),
}

# Same-sign θ mixing (molal scale).  Missing pairs default to 0.0 and are
# flagged in the module docstring.
PITZER_THETA: Dict[Tuple[str, str], float] = {
    ("H+", "Na+"): 0.036,      # PHREEQC pitzer.dat
    ("Fe2+", "Na+"): 0.10945,  # Charykova et al. 2010 (via Sandia/WIPP tabulation)
    # θ(SO4-2, HSO4-): R&B/Charykova report ≈ −0.1354; canonical pitzer.dat
    # omits it (0.0).  Negligible at bath pH ≥ 2; exposed for refinement.
}

# Same-sign/opposite-charge ψ triplets — all default 0.0 (see docstring).
PITZER_PSI: Dict[Tuple[str, str, str], float] = {}

PITZER_CHARGES: Dict[str, int] = {
    "Fe2+": 2,
    "Na+": 1,
    "H+": 1,
    "Mg2+": 2,
    "SO4-2": -2,
    "HSO4-": -1,
    "Cl-": -1,
}


# ─── Solver ──────────────────────────────────────────────────────────

@dataclass
class PitzerSolution:
    """Result of a Pitzer activity-coefficient evaluation (molal basis)."""
    ionic_strength_molal: float
    total_molality: float
    gamma: Dict[str, float]          # conventional single-ion γ (molal scale)
    activity: Dict[str, float]       # a_i = γ_i · m_i
    osmotic_coefficient: float       # φ
    water_activity: float            # a_w = exp(−φ Σm / 55.5084)


def solve_pitzer(
    molalities: Dict[str, float],
    T_C: float = 25.0,
    theta: Dict[Tuple[str, str], float] | None = None,
    psi: Dict[Tuple[str, str, str], float] | None = None,
) -> PitzerSolution:
    """Evaluate Pitzer activity coefficients for a molal composition.

    Parameters
    ----------
    molalities : dict species -> molality (mol/kg H2O).  Keys must appear
        in ``PITZER_CHARGES``; electroneutrality is the caller's
        responsibility (checked to 1e-4 rel. tolerance and warned, not
        enforced).
    T_C : temperature.  Only Aφ responds to T; binary parameters are the
        25 °C set (see module docstring — 10–60 °C is the reliable window).

    Returns
    -------
    PitzerSolution with per-ion γ, activities, osmotic coefficient and
    water activity.
    """
    theta = dict(PITZER_THETA if theta is None else theta)
    psi = dict(PITZER_PSI if psi is None else psi)

    A = A_phi(T_C)
    z = PITZER_CHARGES

    m = {s: molalities.get(s, 0.0) for s in molalities}
    cations = [s for s in molalities if z[s] > 0]
    anions = [s for s in molalities if z[s] < 0]
    # Interaction terms only involve species actually present (m > 0);
    # zero-molality species still get a well-defined (trace) gamma.
    cations_p = [s for s in cations if m.get(s, 0.0) > 0.0]
    anions_p = [s for s in anions if m.get(s, 0.0) > 0.0]

    I = 0.5 * sum(m[s] * z[s] ** 2 for s in m)
    I = max(I, 1e-12)
    sqrtI = math.sqrt(I)
    Z = sum(m[s] * abs(z[s]) for s in m)
    m_tot = sum(m[s] for s in m)

    # Electroneutrality sanity check (relative to total charge)
    q = sum(m[s] * z[s] for s in m)
    q_scale = sum(m[s] * abs(z[s]) for s in m)
    if q_scale > 0 and abs(q) / q_scale > 1e-4:
        import warnings

        warnings.warn(
            f"solve_pitzer: solution not electroneutral (charge imbalance {q:.4g} molal eq); "
            "results are physically meaningless for the imbalanced recipe."
        )

    # ── Binary pair functions ────────────────────────────────────────
    def _pair(c: str, a: str) -> PitzerPair:
        p = PITZER_BINARY.get((c, a))
        if p is None:
            raise KeyError(f"No Pitzer parameters for ({c}, {a})")
        return p

    def B_gamma(c: str, a: str) -> float:
        if m.get(c, 0.0) <= 0.0 or m.get(a, 0.0) <= 0.0:
            # No contribution (all uses are m-scaled), but a parameter row
            # may be missing for absent species — return a neutral value.
            return 0.0
        p = _pair(c, a)
        return p.beta0 + p.beta1 * _g(p.alpha1 * sqrtI) + p.beta2 * _g(p.alpha2 * sqrtI)

    def B_prime(c: str, a: str) -> float:
        if m.get(c, 0.0) <= 0.0 or m.get(a, 0.0) <= 0.0:
            return 0.0
        p = _pair(c, a)
        return (p.beta1 * _gp(p.alpha1 * sqrtI) + p.beta2 * _gp(p.alpha2 * sqrtI)) / I

    def B_phi(c: str, a: str) -> float:
        if m.get(c, 0.0) <= 0.0 or m.get(a, 0.0) <= 0.0:
            return 0.0
        p = _pair(c, a)
        return p.beta0 + p.beta1 * math.exp(-p.alpha1 * sqrtI) + p.beta2 * math.exp(-p.alpha2 * sqrtI)

    def C_val(c: str, a: str) -> float:
        """C_ca = Cφ_ca / (2 √|z_c z_a|).  0.0 for absent species."""
        if m.get(c, 0.0) <= 0.0 or m.get(a, 0.0) <= 0.0:
            return 0.0
        p = _pair(c, a)
        return p.Cphi / (2.0 * math.sqrt(abs(z[c] * z[a])))

    # ── Same-sign mixing (θ + electrostatic ᵈθ) ──────────────────────
    def _theta_keyed(s1: str, s2: str) -> float:
        return theta.get((s1, s2), theta.get((s2, s1), 0.0))

    def _psi_keyed(s1: str, s2: str, s3: str) -> float:
        # canonicalize triplet key order-insensitively
        for key in ((s1, s2, s3), (s2, s1, s3), (s3, s2, s1), (s1, s3, s2),
                    (s2, s3, s1), (s3, s1, s2)):
            if key in psi:
                return psi[key]
        return 0.0

    def Phi_same(s1: str, s2: str) -> float:
        """Φ for activity coefficients (θ + ᵈθ + I ᵈθ′)."""
        e = _etheta(abs(z[s1]), abs(z[s2]), I, A)
        ep = _etheta_prime(abs(z[s1]), abs(z[s2]), I, A)
        return _theta_keyed(s1, s2) + e + I * ep

    def Phi_phi(s1: str, s2: str) -> float:
        """Φφ for the osmotic coefficient (θ + ᵈθ)."""
        return _theta_keyed(s1, s2) + _etheta(abs(z[s1]), abs(z[s2]), I, A)

    def Phi_prime(s1: str, s2: str) -> float:
        return _etheta_prime(abs(z[s1]), abs(z[s2]), I, A)

    # ── Long-range term F ────────────────────────────────────────────
    f_gamma = -A * (sqrtI / (1.0 + B_DH * sqrtI) + (2.0 / B_DH) * math.log(1.0 + B_DH * sqrtI))

    F = f_gamma
    for c in cations_p:
        for a in anions_p:
            F += m[c] * m[a] * B_prime(c, a)
    for i, c1 in enumerate(cations_p):
        for c2 in cations_p[i + 1:]:
            F += m[c1] * m[c2] * Phi_prime(c1, c2)
    for i, a1 in enumerate(anions_p):
        for a2 in anions_p[i + 1:]:
            F += m[a1] * m[a2] * Phi_prime(a1, a2)

    C_sum = sum(m[c] * m[a] * C_val(c, a) for c in cations_p for a in anions_p)

    # ── Single-ion activity coefficients ─────────────────────────────
    gamma: Dict[str, float] = {}

    for M in cations:
        ln_g = z[M] ** 2 * F
        for a in anions:
            ln_g += m[a] * (2.0 * B_gamma(M, a) + Z * C_val(M, a))
        for c in cations:
            if c == M:
                continue
            psi_sum = sum(m[a] * _psi_keyed(M, c, a) for a in anions)
            ln_g += m[c] * (2.0 * Phi_same(M, c) + psi_sum)
        for i, a1 in enumerate(anions):
            for a2 in anions[i + 1:]:
                ln_g += m[a1] * m[a2] * _psi_keyed(M, a1, a2)
        ln_g += z[M] * C_sum
        gamma[M] = math.exp(ln_g)

    for X in anions:
        ln_g = z[X] ** 2 * F
        for c in cations:
            ln_g += m[c] * (2.0 * B_gamma(c, X) + Z * C_val(c, X))
        for a in anions:
            if a == X:
                continue
            psi_sum = sum(m[c] * _psi_keyed(X, a, c) for c in cations)
            ln_g += m[a] * (2.0 * Phi_same(X, a) + psi_sum)
        for i, c1 in enumerate(cations):
            for c2 in cations[i + 1:]:
                ln_g += m[c1] * m[c2] * _psi_keyed(c1, c2, X)
        ln_g += abs(z[X]) * C_sum
        gamma[X] = math.exp(ln_g)

    activity = {s: gamma[s] * m[s] for s in gamma}

    # ── Osmotic coefficient & water activity ─────────────────────────
    brackets = -A * I ** 1.5 / (1.0 + B_DH * sqrtI)
    for c in cations_p:
        for a in anions_p:
            brackets += m[c] * m[a] * (B_phi(c, a) + Z * C_val(c, a) *
                                       (2.0 * math.sqrt(abs(z[c] * z[a]))))
    for i, c1 in enumerate(cations_p):
        for c2 in cations_p[i + 1:]:
            brackets += m[c1] * m[c2] * (Phi_phi(c1, c2) +
                                         sum(m[a] * _psi_keyed(c1, c2, a) for a in anions_p))
    for i, a1 in enumerate(anions_p):
        for a2 in anions_p[i + 1:]:
            brackets += m[a1] * m[a2] * (Phi_phi(a1, a2) +
                                         sum(m[c] * _psi_keyed(a1, a2, c) for c in cations_p))

    phi = 1.0 + (2.0 / max(m_tot, 1e-30)) * brackets
    a_w = math.exp(-phi * m_tot / 55.50844)

    return PitzerSolution(
        ionic_strength_molal=I,
        total_molality=m_tot,
        gamma=gamma,
        activity=activity,
        osmotic_coefficient=phi,
        water_activity=a_w,
    )


def mean_activity_coefficient_pure(
    cation: str, anion: str, molality: float, T_C: float = 25.0
) -> float:
    """Stoichiometric mean activity coefficient γ± of a pure salt.

    Evaluated through the full multicomponent machinery so that the pure
    case and the mixture share one code path.  γ± = (γ+^{ν+} γ−^{ν−})^{1/ν}
    on the *chemical* (fully dissociated) convention.
    """
    zc, za = PITZER_CHARGES[cation], abs(PITZER_CHARGES[anion])
    # Salt stoichiometry from electroneutrality, reduced to lowest terms:
    #   nu_c : nu_a = |za| : zc   (e.g. Na2SO4 → 2:1, FeSO4 → 1:1)
    g = math.gcd(za, zc)
    nu_c, nu_a = za // g, zc // g
    m_c = nu_c * molality
    m_a = nu_a * molality
    sol = solve_pitzer({cation: m_c, anion: m_a}, T_C=T_C)
    nu = nu_c + nu_a
    return (sol.gamma[cation] ** nu_c * sol.gamma[anion] ** nu_a) ** (1.0 / nu)
