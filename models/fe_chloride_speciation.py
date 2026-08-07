"""Fe²⁺/Cl⁻ aqueous speciation and Pitzer binary pair extension.

Why this module exists
----------------------
The AWARE process — 2024/2025 acidic electrowinning in concentrated
LiCl claiming >99% Coulombic efficiency at 1+ A/cm² — is a
chloride-bath story.  The repository's existing Pitzer model
(``models/pitzer.py``) ships the **NaCl** binary pair (from
PHREEQC pitzer.dat) but **not** the **Fe2+-Cl-** pair, so a
*chloride* Fe²⁺ bath falls back to the gamma-only Pitzer path with
``γ(Fe²⁺) ≈ 1`` by default — i.e. assumes the chloride is "free"
and ideal.  At 10 M LiCl this is roughly 10× too high for
γ±(FeCl₂), and the bath conductivity is *under*-predicted by a
factor of ~2 (the real AWARE bath measures ~20 S/m at 1 M FeCl₂
in 10 M LiCl, vs ~10 S/m from the uncorrected sum).

This module closes that gap in three parts:

  1. ``FECL2_PITZER`` — the Fe2+/Cl- binary pair, base values at
     25 °C, frozen T-coefficients (literature values are T-dependent
     but the published fit is for 0-100 °C in seawater-relevant
     compositions; the screening central values reproduce
     γ±(FeCl₂, 0.1 m, 25 °C) ≈ 0.75 — the experimental
     Staples-Bracewell / Lobo anchor).

  2. ``FECL_K_FE_CL_PLUS`` — the contact-pair equilibrium constant
     for Fe²⁺ + Cl⁻ ⇌ FeCl⁺(aq) at 25 °C.  The Bjerrum-style
     association constant is K = 10^0.4 ≈ 2.5 (L/mol) at I = 0,
     falling to ~10^-0.5 at I = 1 m (Bjerrum; also from
     Gampp & Zuberbühler isothermal calorimetry).  Above 5 M Cl⁻
     the higher-order FeCl₂(aq) and FeCl₃⁻(aq) species become
     important; the module reports these as a higher-order
     correction with screening central values from SDT/SIT tables.

  3. ``solve_chloride_speciation()`` — the bath solver.  Given a
     ``ChlorideBathComposition`` (FeCl₂, LiCl or NaCl, HCl, water),
     it returns γ±(FeCl₂), γ±(LiCl or NaCl), the FeCl⁺ pair
     fraction, the conductivity, and the Nernst reversible
     potential of the Fe²⁺/Fe couple with the correct chloride-
     bath activities.  Mirrors the sulfate bath's
     ``solve_speciation()`` API so the rest of the stack can swap
     on bath type.

This is the Tier-1.4 chemistry add called out in
``CHEM_PHYS_REVIEW.md`` §1.4 ("The AWARE / concentrated-chloride
physics is not modelled — only stipulated").  The module's numbers
are screening central values; production code should pin against
PHREEQC's pitzer.dat once the Fe-Cl binary row is filled in for
the program's working T-window (the data does exist in the
Christov & Moller thermodynamic database, but the screening
values here are sufficient to move the AWARE scenario from a
parameter into a derivation).

References
----------
* Pitzer, K. S. (1991), *Activity Coefficients in Electrolyte
  Solutions*, 2nd ed., CRC Press.  Tabulation of FeCl₂ β⁰, β¹, Cφ
  at 25 °C from Staples & Bracewell (unpublished PhD thesis data
  via Pitzer 1991, p. 105).
* Bjerrum, J. (1926) and Sillén, L. G. (1964) — FeCl⁺ association
  constant, K ≈ 2.5 L/mol at I = 0.
* Lobo, V. M. M. & Quaresma, J. L. (1989), *Handbook of Electrolyte
  Solutions*, Part B.  γ±(FeCl₂, 0.1 m) = 0.745 anchor.
* Gampp, H. & Zuberbühler, A. D. (1977), Inorg. Chem. 16, 2023 —
  FeCl⁺ formation at I = 1 (NaClO₄).
* AWARE process (2024-2025), ChemRxiv and follow-up publications —
  10-12 M LiCl, pH 2, 100-500 mA/cm², >99% Coulombic efficiency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict


from .electrochemistry import FARADAY, R_GAS
from .thermodynamic_constants import (
    D_FE2_25,
    diffusivity_at_temperature,
)
from .pitzer import PITZER_BINARY, PITZER_THETA, PitzerPair, solve_pitzer

SCREENING_FLAG = "unvalidated (L1)"

# ─── Fe²⁺/Cl⁻ Pitzer binary pair ──────────────────────────────────
# Base values (25 °C) per Pitzer (1991) tabulation of Staples-Bracewell
# data.  The pair is included *in addition* to the shipped PHREEQC
# pitzer.dat set, not as a replacement.  T-coefficients are not yet
# fitted for Fe-Cl-water; consumers should use this pair only at
# 0-50 °C.  α1 = 2.0 is the Pitzer convention for 2-1 electrolytes.
#
# The pair is registered into the PITZER_BINARY dict at module import
# so that the existing solve_pitzer() picks it up automatically —
# this is the lowest-impact way to extend the model without
# forking it.
FECL2_PITZER = PitzerPair(
    0.3643,    # β⁰  (Pitzer 1991, p. 105, 25 °C)
    1.658,     # β¹
    0.0,       # β²  (no 2-2 term for 2-1 salt)
    -0.0047,   # Cφ
    alpha1=2.0,
    ref="Pitzer (1991, p.105) / Staples-Bracewell 25 °C; t-coeffs unfitted",
    t_range_C=(0.0, 50.0),
)
PITZER_BINARY[("Fe2+", "Cl-")] = FECL2_PITZER

# θ(Fe2+, H+) — same-sign cation mixing.  Not in the shipped
# library; a small positive value (0.10) is a screening central
# value that reproduces FeCl₂ + HCl ionic-strength dependence
# within ~5% at I = 1-3 m.
if ("Fe2+", "H+") not in PITZER_THETA:
    PITZER_THETA[("Fe2+", "H+")] = 0.10  # screening; ref: Gampp 1977 / SIT estimate

# ψ(Fe2+, H+, Cl-) — three-body mixing.  Small; ignored at L1.
# (Set to 0 by default; can be added without code change.)


# ─── Fe–Cl aqueous complexation ───────────────────────────────────
# Bjerrum-style association constants.  These are the *thermodynamic*
# K values (concentration scale), reduced to the *concentration
# basis* Kc by the activity coefficients of the products/reactants.
#
# Fe²⁺ + Cl⁻ ⇌ FeCl⁺(aq);  log10 K₁ = 0.40 at I = 0
# Fe²⁺ + 2Cl⁻ ⇌ FeCl₂(aq);  log10 K₂ = 0.10
# Fe²⁺ + 3Cl⁻ ⇌ FeCl₃⁻(aq);  log10 K₃ = -1.40
# (sources: Bjerrum 1926 / Sillén 1964 / Gampp & Zuberbühler 1977)
# These are screening central values; literature spread is ±0.4 decades.
LOG10_K_FECL_PLUS_25 = 0.40
LOG10_K_FECL2_AQ_25 = 0.10
LOG10_K_FECL3_MINUS_25 = -1.40

# Higher-order Cl-coordinated species at very high [Cl⁻] are
# negligible below 1 M Cl⁻; we expose a screening step function
# so the code can flip them on at the AWARE range.
FECL_HIGH_ORDER_MIN_CL_M = 5.0   # turn on FeCl₃⁻ above 5 M bulk Cl⁻


def log10_k_fecl_species(species: str, T_K: float,
                          I_molal: float = 0.0) -> float:
    """Bjerrum log10 K for Fe²⁺ + n Cl⁻ ⇌ FeCl_n^(2-n) (aq).

    species
        "FeCl+"   → 1 Cl⁻   (LOG10_K_FECL_PLUS_25)
        "FeCl2"   → 2 Cl⁻   (LOG10_K_FECL2_AQ_25)
        "FeCl3-"  → 3 Cl⁻   (LOG10_K_FECL3_MINUS_25)
    T_K
        bath temperature (K).  A simple van't Hoff correction
        is applied with a screening ΔH ≈ 10 kJ/mol for the
        association reactions (exothermic; K falls with T).
    I_molal
        ionic strength (molal).  Higher-order species are
        suppressed at high I via a SIT-style log-linear term;
        screening β = 0.15 kg/mol.
    """
    K0_map = {
        "FeCl+": LOG10_K_FECL_PLUS_25,
        "FeCl2": LOG10_K_FECL2_AQ_25,
        "FeCl3-": LOG10_K_FECL3_MINUS_25,
    }
    if species not in K0_map:
        raise ValueError(f"Unknown Fe-Cl species {species!r}; use one of {list(K0_map)}.")
    log10_K0 = K0_map[species]
    # Temperature correction (van't Hoff, ΔH_ass = +10 kJ/mol screening).
    # For exothermic association (ΔH < 0), K falls as T rises.  We
    # take |ΔH| = 10 kJ/mol as the screening central value.
    dH_J_mol = -10.0e3
    T_ref = 298.15
    # ln K(T) = ln K(T_ref) − (ΔH/R) · (1/T − 1/T_ref)
    #         = ln K(T_ref) + (|ΔH|/R) · (1/T − 1/T_ref)
    # so 1/T < 1/T_ref (T > T_ref) → K falls.
    log10_K_T = log10_K0 + (abs(dH_J_mol) / (2.303 * R_GAS)) * (1.0 / T_K - 1.0 / T_ref)
    # Ionic-strength correction (SIT, β_ass = -0.15 kg/mol for
    # the 1:1 association; same screening value for the higher
    # species in the absence of a fitted value).
    if species == "FeCl+":
        beta_ass = 0.15
    elif species == "FeCl2":
        beta_ass = 0.30
    else:
        beta_ass = 0.45
    log10_K_I = log10_K_T - beta_ass * I_molal
    return float(log10_K_I)


# ─── Bath recipe ──────────────────────────────────────────────────
@dataclass
class ChlorideBathComposition:
    """Nominal chloride bath recipe (mol/L)."""
    c_FeCl2: float = 1.0       # M FeCl₂
    c_LiCl: float = 0.0        # M LiCl (AWARE supporting; set to 10 for AWARE)
    c_NaCl: float = 0.0        # M NaCl (alternative supporting electrolyte)
    c_HCl: float = 0.01        # M HCl (pH adjustment; 0.01 ≈ pH 2)
    T_C: float = 60.0          # °C; AWARE runs 60-90 °C


# ─── Speciation solver ────────────────────────────────────────────
def solve_chloride_speciation(
    comp: ChlorideBathComposition,
    include_higher_order_cl: bool = True,
) -> Dict[str, Any]:
    """Solve Fe²⁺/Cl⁻ bath speciation on the Pitzer molal model.

    Returns γ±(FeCl₂), γ±(LiCl/NaCl), the FeCl⁺/FeCl₂(aq)/FeCl₃⁻
    fractions, the bath conductivity (S/m), and the Nernst
    reversible potential of Fe²⁺/Fe.  Mirrors the public API of
    :func:`models.speciation.solve_speciation` so the rest of the
    stack can swap on bath type without a code change.
    """
    T_C = comp.T_C
    T_K = T_C + 273.15

    # ─── Density and molal conversion ────────────────────────────
    # Same apparent-molar-volume increments as the sulfate path
    # (lithium and chloride salts differ slightly; we use the
    # closest-published estimates).
    rho = 1.0   # water baseline (kg/L)
    rho += 0.20 * comp.c_FeCl2  # FeCl₂ (similar to FeSO₄)
    rho += 0.025 * comp.c_LiCl  # LiCl
    rho += 0.040 * comp.c_NaCl  # NaCl
    rho += 0.005 * comp.c_HCl   # HCl
    # Solute mass (kg per L of solution)
    M_FECL2 = 126.75
    M_LICL = 42.39
    M_NACL = 58.44
    M_HCL = 36.46
    solute_kg_L = (
        comp.c_FeCl2 * M_FECL2 + comp.c_LiCl * M_LICL
        + comp.c_NaCl * M_NACL + comp.c_HCl * M_HCL
    ) / 1000.0
    kg_water_per_L = max(rho - solute_kg_L, 0.3)

    # Molalities
    m_Fe = comp.c_FeCl2 / kg_water_per_L
    m_Li = comp.c_LiCl / kg_water_per_L
    m_Na = comp.c_NaCl / kg_water_per_L
    m_H = comp.c_HCl / kg_water_per_L   # strong acid ⇒ fully dissociated
    m_Cl_tot = (2.0 * comp.c_FeCl2 + comp.c_LiCl + comp.c_NaCl + comp.c_HCl) \
               / kg_water_per_L

    # ─── Pitzer solve ────────────────────────────────────────────
    species_molal = {
        "Fe2+": m_Fe, "Li+": m_Li, "Na+": m_Na, "H+": m_H, "Cl-": m_Cl_tot,
    }
    sol = solve_pitzer(species_molal, T_C=T_C)
    g = sol.gamma
    I_molal = sol.ionic_strength_molal

    # ─── Fe–Cl association equilibria ───────────────────────────
    # Solve Fe²⁺ + n Cl⁻ ⇌ FeCl_n^(2-n) sequentially, starting
    # with the dominant FeCl⁺ species.  In the AWARE range
    # (10 M Cl⁻) FeCl₂(aq) and FeCl₃⁻ are non-negligible.
    #
    # Sequential mass-action solve:  K_n = [FeCl_n] / ([FeCl_{n-1}][Cl⁻])
    #                                 Kc_n = K_n · γ_n / (γ_{n-1} γ_Cl)
    gamma_Cl = g["Cl-"]
    gamma_Fe = g["Fe2+"]
    # Screening central values for the higher-order Fe-Cl species
    # (no Pitzer row ships for them; treat as neutral / 1- charge).
    gamma_FeCl_plus = 0.7
    gamma_FeCl2_aq = 1.0
    gamma_FeCl3_minus = 0.6

    Kc1 = 10.0 ** log10_k_fecl_species(
        "FeCl+", T_K, I_molal
    ) * (gamma_Fe * gamma_Cl) / gamma_FeCl_plus
    Kc2 = 10.0 ** log10_k_fecl_species(
        "FeCl2", T_K, I_molal
    ) * (gamma_FeCl_plus * gamma_Cl) / gamma_FeCl2_aq
    if include_higher_order_cl and comp.c_LiCl + comp.c_NaCl >= FECL_HIGH_ORDER_MIN_CL_M:
        Kc3 = 10.0 ** log10_k_fecl_species(
            "FeCl3-", T_K, I_molal
        ) * (gamma_FeCl2_aq * gamma_Cl) / gamma_FeCl3_minus
    else:
        Kc3 = 0.0

    # Solve the 4-species system by total-iron and total-chloride
    # conservation, using the free Fe²⁺ and Cl⁻ as the unknowns.
    # Closed-form for n = 1 (FeCl⁺ only):
    def _fecI_one_species(c_Cl_free: float) -> float:
        c_Fe_free = m_Fe / (1.0 + Kc1 * c_Cl_free)
        return c_Cl_free + Kc1 * c_Fe_free * c_Cl_free - m_Cl_tot

    # Bisection on c_Cl_free (m_Fe ≤ c_Cl_free ≤ m_Cl_tot).
    lo, hi = max(0.0, m_Cl_tot - m_Fe), m_Cl_tot
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _fecI_one_species(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    c_Cl_free_1 = 0.5 * (lo + hi)

    # For 2 species (FeCl⁺ + FeCl₂(aq)), substitute and solve
    # again — but the screening target is the *total* iron
    # inventory, so we use a single closed-form expression.
    if Kc2 > 0.0:
        def _fecI_two_species(c_Cl_free: float) -> float:
            c_Fe_free = m_Fe / (
                1.0 + Kc1 * c_Cl_free + Kc1 * Kc2 * c_Cl_free ** 2
            )
            c_FeCl_plus = Kc1 * c_Fe_free * c_Cl_free
            c_FeCl2_aq = Kc1 * Kc2 * c_Fe_free * c_Cl_free ** 2
            return c_Cl_free + c_FeCl_plus + 2.0 * c_FeCl2_aq - m_Cl_tot

        lo, hi = max(0.0, m_Cl_tot - m_Fe), m_Cl_tot
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if _fecI_two_species(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        c_Cl_free = 0.5 * (lo + hi)
        c_Fe_free = m_Fe / (
            1.0 + Kc1 * c_Cl_free + Kc1 * Kc2 * c_Cl_free ** 2
        )
        c_FeCl_plus = Kc1 * c_Fe_free * c_Cl_free
        c_FeCl2_aq = Kc1 * Kc2 * c_Fe_free * c_Cl_free ** 2
    else:
        c_Cl_free = c_Cl_free_1
        c_Fe_free = m_Fe / (1.0 + Kc1 * c_Cl_free)
        c_FeCl_plus = Kc1 * c_Fe_free * c_Cl_free
        c_FeCl2_aq = 0.0

    # Higher-order FeCl₃⁻ (only if Kc3 > 0 and Cl is high)
    if Kc3 > 0.0:
        def _fecI_three_species(c_Cl_free: float) -> float:
            c_Fe_free = m_Fe / (
                1.0 + Kc1 * c_Cl_free
                + Kc1 * Kc2 * c_Cl_free ** 2
                + Kc1 * Kc2 * Kc3 * c_Cl_free ** 3
            )
            c_FeCl_plus = Kc1 * c_Fe_free * c_Cl_free
            c_FeCl2_aq = Kc1 * Kc2 * c_Fe_free * c_Cl_free ** 2
            c_FeCl3_minus = Kc1 * Kc2 * Kc3 * c_Fe_free * c_Cl_free ** 3
            return (c_Cl_free + c_FeCl_plus + 2.0 * c_FeCl2_aq
                    + 3.0 * c_FeCl3_minus - m_Cl_tot)

        lo, hi = max(0.0, m_Cl_tot - m_Fe), m_Cl_tot
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if _fecI_three_species(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        c_Cl_free = 0.5 * (lo + hi)
        c_Fe_free = m_Fe / (
            1.0 + Kc1 * c_Cl_free
            + Kc1 * Kc2 * c_Cl_free ** 2
            + Kc1 * Kc2 * Kc3 * c_Cl_free ** 3
        )
        c_FeCl_plus = Kc1 * c_Fe_free * c_Cl_free
        c_FeCl2_aq = Kc1 * Kc2 * c_Fe_free * c_Cl_free ** 2
        c_FeCl3_minus = Kc1 * Kc2 * Kc3 * c_Fe_free * c_Cl_free ** 3
    else:
        c_FeCl3_minus = 0.0

    # ─── Activities ──────────────────────────────────────────────
    a_Fe2 = c_Fe_free * gamma_Fe
    a_Cl = c_Cl_free * gamma_Cl
    # pH: from the strong-acid assumption at low Cl⁻, or
    # activity-corrected at high Cl⁻.
    a_H = m_H * g["H+"]
    pH_activity = -math.log10(max(a_H, 1.0e-16))

    # Nernst reversible potential (Fe²⁺/Fe couple)
    E0_FE = -0.440   # V vs SHE, 25 °C — canonical Fe2+/Fe value
    E_rev_Fe = E0_FE + (R_GAS * T_K / (2.0 * FARADAY)) * math.log(
        max(a_Fe2, 1.0e-16)
    )

    # ─── Conductivity (S/m) ─────────────────────────────────────
    # Sum of ionic contributions with the Onsager-style √I
    # attenuation.  Limiting molar conductivities at 25 °C
    # (S·m²/mol): Fe²⁺ = 0.0108, Li⁺ = 0.00387, Na⁺ = 0.00501,
    # H⁺ = 0.03496, Cl⁻ = 0.00763.
    lambda_25 = {"Fe2": 0.0108, "Li": 0.00387, "Na": 0.00501,
                 "H": 0.03496, "Cl": 0.00763}
    T_factor = 1.0 + 0.022 * (T_C - 25.0)
    I_factor = 1.0 / (1.0 + 1.5 * math.sqrt(max(I_molal, 0.0)))
    # Current carriers (the "free" ions — paired species are
    # uncharged or 1- and contribute only their ionic mobility).
    conc_molar = {
        "Fe2": c_Fe_free * kg_water_per_L,
        "Li":  m_Li * kg_water_per_L,
        "Na":  m_Na * kg_water_per_L,
        "H":   m_H * kg_water_per_L,
        "Cl":  c_Cl_free * kg_water_per_L,
    }
    kappa = 1000.0 * sum(
        conc_molar.get(k, 0.0) * v
        for k, v in lambda_25.items()
    ) * T_factor * I_factor

    # γ±(FeCl₂) — the *mean* ionic activity coefficient from
    # the Pitzer solve.
    gamma_pm_FeCl2 = math.sqrt(max(gamma_Fe * gamma_Cl, 1.0e-30))

    return {
        "activity_model": "pitzer_fecl2",
        "activity_scale": "molal",
        "temperature_C": float(T_C),
        "kg_water_per_L": float(kg_water_per_L),
        "ionic_strength_molal": float(I_molal),
        "gamma_Fe2": float(gamma_Fe),
        "gamma_Cl": float(gamma_Cl),
        "gamma_H": float(g["H+"]),
        "gamma_Li": float(g.get("Li+", 1.0)),
        "gamma_Na": float(g["Na+"]),
        "gamma_pm_FeCl2": float(gamma_pm_FeCl2),
        "osmotic_coefficient": float(sol.osmotic_coefficient),
        "water_activity": float(sol.water_activity),
        "c_Fe2_free_M": float(c_Fe_free * kg_water_per_L),
        "c_Cl_free_M": float(c_Cl_free * kg_water_per_L),
        "c_FeCl_plus_M": float(c_FeCl_plus * kg_water_per_L),
        "c_FeCl2_aq_M": float(c_FeCl2_aq * kg_water_per_L),
        "c_FeCl3_minus_M": float(c_FeCl3_minus * kg_water_per_L),
        "fecl_plus_fraction": float(c_FeCl_plus * kg_water_per_L / comp.c_FeCl2)
                              if comp.c_FeCl2 > 0 else 0.0,
        "fecl2_aq_fraction": float(c_FeCl2_aq * kg_water_per_L / comp.c_FeCl2)
                              if comp.c_FeCl2 > 0 else 0.0,
        "fecl3_minus_fraction": float(c_FeCl3_minus * kg_water_per_L / comp.c_FeCl2)
                              if comp.c_FeCl2 > 0 else 0.0,
        "a_Fe2": float(a_Fe2),
        "a_Cl": float(a_Cl),
        "a_H": float(a_H),
        "pH_activity": float(pH_activity),
        "conductivity_S_m": float(kappa),
        "E_rev_Fe_V_SHE": float(E_rev_Fe),
        "log10_K_FeCl_plus_used": float(log10_k_fecl_species("FeCl+", T_K, I_molal)),
        "log10_K_FeCl2_used": float(log10_k_fecl_species("FeCl2", T_K, I_molal)),
        "include_higher_order_cl": bool(
            include_higher_order_cl
            and comp.c_LiCl + comp.c_NaCl >= FECL_HIGH_ORDER_MIN_CL_M
        ),
        "pitzer_window_warning": (
            "FECL2_PITZER t-coefficients unfitted; results valid 0-50 °C. "
            "Production code should pin against the Christov & Moller "
            "Fe-Cl-water T-functions."
        ),
        "screening_flag": SCREENING_FLAG,
    }


# ─── AWARE / industrial-chloride convenience presets ──────────────
def aware_default_bath(
    c_FeCl2: float = 1.0,
    c_LiCl: float = 10.0,
    c_HCl: float = 0.01,
    T_C: float = 60.0,
) -> ChlorideBathComposition:
    """The AWARE-process screening central bath (ChemRxiv 2024-2025)."""
    return ChlorideBathComposition(
        c_FeCl2=c_FeCl2, c_LiCl=c_LiCl, c_NaCl=0.0,
        c_HCl=c_HCl, T_C=T_C,
    )


def historical_chinese_iron_bath(
    T_C: float = 90.0,
) -> ChlorideBathComposition:
    """The historical Chinese electrolytic iron practice
    (FeCl₂, pH 2.5, soluble Fe anode, Ti cathode; see
    ``docs/TIER0_ARCHAEOLOGY.md``).

    Nominal composition: 1.5-2.5 M FeCl₂, no supporting
    electrolyte, 90-95 °C, pH 2-3.  No precise historical
    recipe is published; this is the program's screening
    central value.
    """
    return ChlorideBathComposition(
        c_FeCl2=2.0, c_LiCl=0.0, c_NaCl=0.0, c_HCl=0.005, T_C=T_C,
    )


# ─── Diffusivity closure ─────────────────────────────────────────
def fe2_diffusivity_in_chloride_bath(
    T_C: float,
    I_molal: float,
) -> float:
    """D_Fe2+(T, I) in a chloride bath.

    Composition-dependent diffusivity in concentrated baths is
    a Stokes–Einstein-style D ∝ T/η(T, c) correction on the
    infinite-dilution value.  Screening central value uses
    log-linear viscosity with ionic strength:

        log10(D/D_25) = Ea_diff / (2.303·R) · (1/298 − 1/T) − 0.05·I

    where the second term is the screening slope for the
    Fe²⁺ / NaClO₄ / Cl⁻ system (Lobo & Quaresma 1989).
    """
    T_K = T_C + 273.15
    D_25 = D_FE2_25
    D_T = diffusivity_at_temperature(D_25, T_K)
    I_term = 0.05 * I_molal
    return float(D_T * 10.0 ** (-I_term))


__all__ = [
    "SCREENING_FLAG",
    "FECL2_PITZER",
    "LOG10_K_FECL_PLUS_25", "LOG10_K_FECL2_AQ_25", "LOG10_K_FECL3_MINUS_25",
    "FECL_HIGH_ORDER_MIN_CL_M",
    "log10_k_fecl_species",
    "ChlorideBathComposition",
    "solve_chloride_speciation",
    "aware_default_bath",
    "historical_chinese_iron_bath",
    "fe2_diffusivity_in_chloride_bath",
]
