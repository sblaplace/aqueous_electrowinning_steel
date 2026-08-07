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
constant and density.  Binary interaction parameters are anchored at 25 °C;
``PitzerPair.at_T`` applies a per-parameter polynomial T-correction when
the pair ships verified coefficients, in one of two named bases:

* ``t_form="eq36"`` — the EQ3/6-Sandia form
  ``p(T) = a + c1·(1/T − 1/Tr) + c2·ln(T/Tr) + c3·(T − Tr)`` (Tr = 298.15 K),
  with ``a`` the value at Tr (slots for documented re-fits).
* ``t_form="mtd"`` — the MTDATA/NPL form ``p(T) = A + B·T + D·T² + F/T``
  (T in K), used to ship published coefficient tables **verbatim**.

An all-zero row always means "frozen at the base value", so pairs without
tables stay byte-identical to the frozen-parameter behavior.

**Shipped T-table (2026-08-06): Fe²⁺–SO₄²⁻ from Kobylin, Sippola &
Taskinen (2011).**  The Kobylin CALPHAD assessment of the FeSO4–H2O binary
(their Table 6, MTDATA form) is wired verbatim with
``t_range_C = (10, 90)`` — the window jointly certified by the R&B (binary
10–90 °C) and Kobylin assessments; the Kobylin model itself spans −2–220 °C
with self-declared sparsity above 100 °C.  Verification on this repo's
machinery: γ±(FeSO4, 0.1 m, 25 °C) = 0.163 vs the published anchor 0.164
(≈0.7 %; Reardon & Beckie 1987: 0.161 — both inside the 0.150–0.164
assessment spread).  **Honest provenance note:** the original
Reardon & Beckie (1987, GCA 51:2355–2368) β(T)/Cφ(T) functions are
paywalled and could not be transcribed-verified in this environment; what
is publicly verifiable is their 25 °C projection (β⁰=0.2568, β¹=3.063,
β²=−42 — retained as the pair's ``beta*``/``Cphi`` base fields, the
Pitzer (1991) anchor set) and the abstract-published Ksp(T) relations.
Kobylin et al. document an internal enthalpy inconsistency in the R&B fits
(ΔHs ≈ 16.1 vs 21.2 kJ/mol) that the 2011 re-assessment resolved, so the
Kobylin set is the defensible "verified" wiring; its 25 °C projection
(β⁰=0.3194, β¹=2.2621, β²=−16.2142, Cφ=−0.0159) *supersedes* the anchor
set at every temperature, 25 °C included — a ~2.5 % shift of γ± at 0.1 m,
inside the mutual spread of the two assessments.  Fe²⁺–HSO₄⁻ and the
H⁺–SO₄²⁻/HSO₄⁻ binaries remain frozen (no publicly verified T-functions;
Kobylin et al. 2012's ternary A+F/T set is paywalled).
Treat results outside 10–90 °C as extrapolated (``at_T`` warns).

The **acceptance gate** that vetted this table remains in place for future
tables (e.g. a transcribed R&B 1987 set): ``register_t_coeff_table`` /
``apply_t_coeff_library`` / ``verify_t_coeff_table`` /
``revert_t_coeff_library`` with ``FESO4_GAMMA_ANCHORS`` and
``RB1987_ACCEPTANCE_ANCHORS`` (still pending); see
``docs/PITZER_TCOEFF_ACCEPTANCE.md``.

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
from dataclasses import dataclass, replace
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

# Anchor temperature of the tabulated binary parameters (25 °C).
PITZER_T_REF_K = 298.15

# One all-zero 4-coefficient row: "parameter frozen to its 25 °C value".
_ZERO_T_ROW: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
_ZERO_T_COEFFS: Tuple[Tuple[float, float, float, float], ...] = (_ZERO_T_ROW,) * 4


@dataclass(frozen=True)
class PitzerPair:
    """Cation–anion binary interaction parameters (molal scale, 25 °C anchor).

    ``t_coeffs`` carries the optional temperature dependence (2026-08
    framework): one 4-coefficient row per parameter, in the order
    (β⁰, β¹, β², Cφ), interpreted by ``t_form``:

    * ``t_form="eq36"`` (default) — rows ``(a, c1, c2, c3)``, EQ3/6-Sandia:
      ``p(T) = a + c1·(1/T − 1/Tr) + c2·ln(T/Tr) + c3·(T − Tr)``
      with Tr = 298.15 K and ``a`` the 25 °C value (which should coincide
      with the base field).
    * ``t_form="mtd"`` — rows ``(A, B, D, F)``, MTDATA/NPL:
      ``p(T) = A + B·T + D·T² + F/T``  (T in K).

      Verbatim-published tables (the shipped Fe²⁺–SO₄²⁻ Kobylin et al. 2011
      set) ship in this form; their 25 °C projection *supersedes* the
      ``beta*``/``Cphi`` base fields at every temperature — the base fields
      then document the previously-shipped anchor set only (see the module
      docstring's Temperature section).

    An all-zero row always means "frozen at the base value" under either
    form.  α₁/α₂ belong to the functional form itself and get no T-form.

    ``t_range_C`` is the validity window certified by the table's
    provenance (``None`` for the shipped frozen tables).  Evaluating an
    evolved (non-frozen) pair outside it raises an extrapolation warning.
    """
    beta0: float
    beta1: float
    beta2: float
    Cphi: float
    alpha1: float = 2.0
    alpha2: float = 12.0
    ref: str = ""
    t_coeffs: Tuple[Tuple[float, float, float, float], ...] = _ZERO_T_COEFFS
    t_range_C: "Tuple[float, float] | None" = None
    t_form: str = "eq36"

    def at_T(self, T_C: float) -> "PitzerPair":
        """Return a copy with β⁰/β¹/β²/Cφ evaluated at ``T_C``.

        With every ``t_coeffs`` row zero this returns ``self`` unchanged,
        so default (frozen) results stay byte-identical.
        """
        if all(c == 0.0 for row in self.t_coeffs for c in row):
            return self
        if self.t_range_C is not None and not (
            self.t_range_C[0] <= T_C <= self.t_range_C[1]
        ):
            import warnings

            warnings.warn(
                f"PitzerPair({self.ref or 'unnamed'}): T = {T_C} °C is outside "
                f"the table's certified window {self.t_range_C} °C — "
                "extrapolated; treat results as unverified."
            )
        T = T_C + 273.15
        Tr = PITZER_T_REF_K

        def evolve(base: float, row: Tuple[float, float, float, float]) -> float:
            if all(c == 0.0 for c in row):
                return base
            if self.t_form == "mtd":
                a, b, d, f = row
                return a + b * T + d * T * T + f / T
            # EQ3/6-Sandia form
            a, c1, c2, c3 = row
            return (a + c1 * (1.0 / T - 1.0 / Tr)
                    + c2 * math.log(T / Tr) + c3 * (T - Tr))

        evolved = tuple(
            evolve(base, row)
            for base, row in zip(
                (self.beta0, self.beta1, self.beta2, self.Cphi), self.t_coeffs
            )
        )
        return replace(
            self, beta0=evolved[0], beta1=evolved[1],
            beta2=evolved[2], Cphi=evolved[3],
        )


# Key: (cation, anion).  Charges keyed in PITZER_CHARGES below.
PITZER_BINARY: Dict[Tuple[str, str], PitzerPair] = {
    # 2–2 sulfate — α1 = 1.4 is the Pitzer convention for 2–2 electrolytes.
    # FeSO4 association is absorbed in β2; no explicit pair species.
    # Base values (25 °C anchor): Pitzer (1991) tabulation of the
    # Reardon & Beckie (1987) fit.  Since 2026-08-06 the pair ships the
    # Kobylin, Sippola & Taskinen (2011, CALPHAD 35:499–511, Table 6)
    # VERIFIED temperature functions VERBATIM (MTDATA form
    # p = A + B·T + D·T² + F/T, T in K; their table prints D·10⁻⁵ and
    # F·10³ — transcription validated by reproducing their anchor
    # γ±(0.1 m, 25 °C) = 0.164 to ≈0.7 %, see FESO4_GAMMA_ANCHORS and
    # docs/PITZER_TCOEFF_ACCEPTANCE.md).  The Kobylin 25 °C projection
    # (0.3194/2.2621/−16.2142/−0.0159) supersedes the base set at all T
    # (module docstring, Temperature section).
    ("Fe2+", "SO4-2"): PitzerPair(
        0.2568, 3.063, -42.42, 0.0213, alpha1=1.4,
        ref="Pitzer (1991)/R&B-1987 25 °C anchor + Kobylin et al. (2011) T-functions (verbatim, mtd form)",
        t_coeffs=(
            (5.1934, -0.0161, 1.8349e-5, -508.3),    # β⁰: A, B, D, F
            (15.8514, 0.0085, -6.0442e-5, -3205.3),  # β¹: A, B, D, F
            (-16.2142, 0.0, 0.0, 0.0),               # β²: constant
            (-0.0588, 0.0, 0.0, 12.8),               # Cφ: A + F/T
        ),
        t_range_C=(10.0, 90.0),
        t_form="mtd",
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
    # LiCl: canonical 1–1 electrolyte, the AWARE supporting salt.  Pitzer
    # 1991 tabulation (p. 105) of the Holmes-Couture 25 °C fit.
    ("Li+", "Cl-"): PitzerPair(0.1494, 0.3074, 0.0, 0.00359, ref="Pitzer (1991, p.105) / Holmes-Couture 25 °C"),
    # HCl: the AWARE pH-adjustment acid.  Pitzer 1991 p. 105.
    ("H+", "Cl-"): PitzerPair(0.1775, 0.2945, 0.0, 0.0008, ref="Pitzer (1991, p.105) / Holmes 25 °C"),
    # Li2SO4: the Li+ analogue of Na2SO4 — used when the AWARE feed
    # is delivered as Li2SO4 before chloride exchange.  Pitzer 1991
    # tabulation.
    ("Li+", "SO4-2"): PitzerPair(-0.0624, 0.1503, 0.0, -0.00319, ref="Pitzer (1991) / Filippov 25 °C"),
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

# Snapshot of the database exactly as shipped (2026-08-06: Fe–SO4 carries
# its accepted Kobylin table).  ``revert_t_coeff_library`` restores from
# this, i.e. revert means "back to the shipped state", not "back to zero
# T-coefficients".
PITZER_BINARY_SHIPPED: Dict[Tuple[str, str], PitzerPair] = dict(PITZER_BINARY)


# ─── Verified T-coefficient table registry & acceptance gate (2026-08) ──

@dataclass(frozen=True)
class TCoeffTable:
    """A candidate verified T-coefficient table for one binary pair.

    Fields
    ------
    cation, anion : the pair it applies to (key of ``PITZER_BINARY``).
    t_coeffs : four rows in the order (β⁰, β¹, β², Cφ); each row is
        ``(a, c1, c2, c3)`` for ``t_form="eq36"`` or ``(A, B, D, F)`` for
        ``t_form="mtd"`` — the ``PitzerPair.at_T`` bases.
    t_range_C : provenance-certified validity window.
    provenance : REQUIRED exact citation, including the table/equation
        numbers the values were transcribed from.  This is the whole point
        of the gate: "verified" means traceable reproduction, not a number
        of unknown parse lineage.
    t_form : ``"eq36"`` (default) or ``"mtd"``.
    """
    cation: str
    anion: str
    t_coeffs: Tuple[Tuple[float, float, float, float], ...]
    t_range_C: Tuple[float, float]
    provenance: str
    t_form: str = "eq36"


@dataclass(frozen=True)
class GammaAnchor:
    """One published mean-activity-coefficient anchor a T-table must hit."""
    t_C: float
    molality: float
    gamma_expected: float
    rel_tol: float
    source: str


# γ± anchors for the Fe–SO4 pair, from the published record.  The shipped
# Kobylin table must (and does) reproduce BOTH through this module's
# machinery: γ±(0.1 m, 25 °C) = 0.1628 computed → 0.7 % vs Kobylin's
# anchor, 1.1 % vs Reardon & Beckie's — each well inside its tolerance
# (see docs/PITZER_TCOEFF_ACCEPTANCE.md).
FESO4_GAMMA_ANCHORS: Tuple[GammaAnchor, ...] = (
    GammaAnchor(25.0, 0.1, 0.164, 0.02,
                "Kobylin, Sippola & Taskinen (2011), CALPHAD 35:499-511, "
                "assessment anchor γ±(FeSO4, 0.1 m, 25 °C) = 0.164"),
    GammaAnchor(25.0, 0.1, 0.161, 0.02,
                "Reardon & Beckie (1987), GCA 51:2355-2368 — "
                "γ±(FeSO4, 0.1 m, 25 °C) = 0.161"),
)


KOBYLIN2011_FESO4_TTABLE = TCoeffTable(
    cation="Fe2+", anion="SO4-2",
    t_coeffs=(
        (5.1934, -0.0161, 1.8349e-5, -508.3),    # β⁰: A, B, D, F
        (15.8514, 0.0085, -6.0442e-5, -3205.3),  # β¹: A, B, D, F
        (-16.2142, 0.0, 0.0, 0.0),               # β²: constant
        (-0.0588, 0.0, 0.0, 12.8),               # Cφ: A + F/T
    ),
    t_range_C=(10.0, 90.0),
    provenance=(
        "Kobylin, Sippola & Taskinen (2011), 'Thermodynamic modelling of "
        "aqueous Fe(II) sulfate solutions', CALPHAD 35(4):499-511, Table 6, "
        "verbatim in the paper's MTDATA form p = A + B·T + D·T² + F/T "
        "(printed column scales D·10⁻⁵, F·10³; their Table 6).  Functional "
        "form confirmed from Kobylin's doctoral dissertation (Aalto 2013, "
        "eq. 26); same 2–2 electrolyte α₁=1.4/α₂=12 convention as Table 1 "
        "here (Pitzer/HMW).  Validated in-repo by reproducing both γ±(0.1 m, "
        "25 °C) anchors (0.164 Kobylin / 0.161 R&B) to ≈0.7 %/1.1 %."
    ),
    t_form="mtd",
)

# The library ships with the table that has PASSED the gate (applied to
# PITZER_BINARY above).  The R&B (1987) β(T)/Cφ(T) functions remain a
# shopping-list item (paywalled; see RB1987_ACCEPTANCE_ANCHORS comment and
# docs/PITZER_TCOEFF_ACCEPTANCE.md).
T_COEFF_LIBRARY: Dict[str, TCoeffTable] = {
    "kobylin-2011-feso4": KOBYLIN2011_FESO4_TTABLE,
}

# Acceptance anchors for a future transcribed R&B FeSO4 T-set: gamma and
# copperas solubility vs T over 10–60 °C.  Empty by design — populated only
# when values with publication-grade provenance (table/equation numbers
# from the 1987 paper itself) exist.  tests/test_pitzer_tcoeffs.py pins
# this pending state so a future table installation cannot sneak in
# untested.
RB1987_ACCEPTANCE_ANCHORS: Tuple[GammaAnchor, ...] = ()


def register_t_coeff_table(name: str, table: TCoeffTable) -> None:
    """Register a candidate T-coefficient table (does not apply it)."""
    if not table.provenance:
        raise ValueError(
            "TCoeffTable.provenance is required — unverified tables are "
            "exactly what this gate refuses.")
    if table.t_form not in ("eq36", "mtd"):
        raise ValueError(f"t_form must be 'eq36' or 'mtd', got {table.t_form!r}")
    expected_key = (table.cation, table.anion)
    if expected_key not in PITZER_BINARY:
        raise KeyError(f"No PitzerPair for {expected_key} in PITZER_BINARY")
    if len(table.t_coeffs) != 4 or any(len(r) != 4 for r in table.t_coeffs):
        raise ValueError("t_coeffs must be 4 rows × 4 coefficients")
    if name in T_COEFF_LIBRARY:
        raise ValueError(f"T-coefficient table {name!r} already registered")
    T_COEFF_LIBRARY[name] = table


def apply_t_coeff_library(name: str) -> None:
    """Install a registered table's coefficients into its PITZER_BINARY pair.

    Raises on out-of-window evaluation later via the pair's ``t_range_C``.
    Call ``revert_t_coeff_library`` to restore the pair as shipped.
    """
    table = T_COEFF_LIBRARY[name]
    key = (table.cation, table.anion)
    PITZER_BINARY[key] = replace(
        PITZER_BINARY[key],
        t_coeffs=table.t_coeffs,
        t_range_C=table.t_range_C,
        t_form=table.t_form,
        ref=(PITZER_BINARY[key].ref + f" [{name} T-coeffs: {table.provenance}]").strip(),
    )


def revert_t_coeff_library(name: str) -> None:
    """Restore the table's pair to its shipped state (PITZER_BINARY_SHIPPED).

    For the Fe–SO4 pair the shipped state *includes* the accepted Kobylin
    et al. (2011) table, so reverting ships-frozen behaviour is wrong —
    the module snapshot is the source of truth.
    """
    table = T_COEFF_LIBRARY[name]
    key = (table.cation, table.anion)
    PITZER_BINARY[key] = PITZER_BINARY_SHIPPED[key]


def verify_t_coeff_table(
    table: TCoeffTable,
    anchors: Tuple[GammaAnchor, ...],
) -> Dict[str, object]:
    """Check a candidate table against published γ± anchors.

    Temporarily installs the table, evaluates
    ``mean_activity_coefficient_pure`` at each anchor, and restores the
    frozen set afterwards.  Returns a report dict:
    ``{"passed": bool, "rows": [{t_C, molality, expected, actual, rel_err,
    source}], "provenance": str}``.

    NOTE: passing here establishes only that the table reproduces the given
    anchors.  Verification *of provenance* additionally requires that the
    anchor values themselves come from the source publication (see
    ``FESO4_GAMMA_ANCHORS`` / ``RB1987_ACCEPTANCE_ANCHORS`` /
    docs/PITZER_TCOEFF_ACCEPTANCE.md).
    """
    key = (table.cation, table.anion)
    if key not in PITZER_BINARY:
        raise KeyError(f"No PitzerPair for {key} in PITZER_BINARY")
    saved = PITZER_BINARY[key]
    try:
        PITZER_BINARY[key] = replace(
            saved, t_coeffs=table.t_coeffs, t_range_C=table.t_range_C,
            t_form=table.t_form)
        rows = []
        passed = True
        for a in anchors:
            actual = mean_activity_coefficient_pure(
                table.cation, table.anion, a.molality, T_C=a.t_C)
            rel_err = abs(actual - a.gamma_expected) / a.gamma_expected
            ok = rel_err <= a.rel_tol
            passed = passed and ok
            rows.append({
                "t_C": a.t_C, "molality": a.molality,
                "expected": a.gamma_expected, "actual": actual,
                "rel_err": rel_err, "source": a.source, "passed": ok,
            })
    finally:
        PITZER_BINARY[key] = saved
    return {"passed": passed, "rows": rows, "provenance": table.provenance}

PITZER_CHARGES: Dict[str, int] = {
    "Fe2+": 2,
    "Na+": 1,
    "H+": 1,
    "Mg2+": 2,
    "Li+": 1,   # AWARE supporting cation; registered for the chloride-bath solver
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
    T_C : temperature.  Aφ responds to T exactly; binary parameters respond
        through ``PitzerPair.at_T`` — the Fe²⁺–SO₄²⁻ pair carries the
        verified Kobylin et al. (2011) T-functions verbatim (t_form="mtd"),
        the other shipped pairs stay frozen at their 25 °C set (see module
        docstring — 10–90 °C is the certified window for the Fe–SO4 table).

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
        # 2026-08: T-evolved copy (identity when the pair ships frozen
        # 25 °C parameters, so default runs stay byte-identical).
        return p.at_T(T_C)

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
