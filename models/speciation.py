"""Multicomponent concentrated electrolyte speciation and activity model.

Solves non-ideal ion activities, bisulfate dissociation, ionic strength,
electrolyte conductivity, and thermodynamic Nernst reversible potentials
for aqueous iron electrowinning baths.

Two activity models are available:

* ``model="pitzer"`` (default) — the multicomponent Pitzer ion-interaction
  model from ``models/pitzer.py`` (Fe²⁺–Na⁺–H⁺ ∥ SO₄²⁻–HSO₄⁻), valid to
  the multi-molal ionic strengths of real electrowinning baths.  In this
  convention ferrous sulfate association is absorbed into the 2–2-salt
  virial coefficients — iron is treated as fully dissociated Fe²⁺ and the
  resulting single-ion activity coefficients (γ±(FeSO₄, 1 m) ≈ 0.05)
  carry the non-ideality.  A secondary *contact-pair estimate* of
  FeSO₄⁰(aq) is still reported for diagnostics/conductivity, but it no
  longer drives activities.

  This replaces the previous default, which applied the **Davies**
  equation (valid to I ≈ 0.5 mol/kg) inside a self-consistent pair
  equilibrium at I ≈ 1.5–2.5 mol/kg.  In that regime Davies badly
  overestimates γ of the divalent ions (γ₂ ≈ 0.68 instead of ≈ 0.05),
  which forced ~97 % of the iron into phantom FeSO₄⁰ pairs, depressed
  the computed conductivity ~2×, and shifted the Nernst potential
  ~40–60 mV too negative.  The Davies numbers were outside their
  calibrated range; the Pitzer path supersedes them.  The old behaviour
  is preserved under ``model="davies"`` for comparison and regression
  archaeology only.

* ``model="davies"`` — legacy Davies + explicit FeSO₄⁰ pairing model.

Standard state: molal (1 mol/kg H₂O) in Pitzer mode; activities are
conventional single-ion values.  The molar ↔ molal conversion uses a
documented apparent-molar-volume density estimate.

References:
- Pitzer, K. S. (1991). Activity Coefficients in Electrolyte Solutions. CRC Press.
- Harvie, Møller & Weare (1984); Reardon & Beckie (1987); Kobylin et al. (2011)
  — see ``models/pitzer.py`` for parameter provenance and validation anchors.
- Davies, C. W. (1962). Ion Association. Butterworths, London. (legacy path)
- Beverskog, B., & Puigdomenech, I. (1996). Revised Pourbaix diagrams for iron.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from . import pitzer as _pitzer

# Physical constants
R = 8.314462618  # J/(mol*K)
F = 96485.33212  # C/mol
E0_FE2_FE = -0.447  # V vs SHE for Fe2+ + 2e- -> Fe at 25 °C (molar std state)
KW_25 = 1.0e-14
KSP_FE_OH2_25 = 4.87e-17  # Ksp for Fe(OH)2 at 25 °C
K_HSO4_25 = 1.05e-2  # Ka2 for HSO4- -> H+ + SO4(2-) at 25 °C
K_FESO4_PAIR_25 = 200.0  # Thermodynamic K for FeSO4(aq) contact pair (25 °C)
KA_BORIC_25 = 5.8e-10  # B(OH)3 + H2O -> H+ + B(OH)4-, pKa ≈ 9.24 at 25 °C

# Molar masses (g/mol) for density/molality conversion
M_FESO4 = 151.908
M_NA2SO4 = 142.04
M_H2SO4 = 98.079
M_H3BO3 = 61.83


@dataclass
class SolutionComposition:
    """Nominal bath recipe concentrations (mol/L or M)."""
    c_FeSO4: float = 1.0      # M FeSO4
    c_Na2SO4: float = 0.5     # M Na2SO4 (supporting electrolyte)
    c_H2SO4: float = 0.01     # M H2SO4 (pH adjustment)
    c_H3BO3: float = 0.4      # M H3BO3 (boric acid buffer)
    T_C: float = 50.0         # °C


# ─── Legacy Davies machinery (kept for A/B comparison) ──────────────

def davies_A(T_C: float) -> float:
    """Temperature-dependent Davies / Debye-Hückel A parameter (kg^0.5 / mol^0.5)."""
    T_K = T_C + 273.15
    # Empirical fit for dielectric constant of water vs temperature
    epsilon = 87.740 - 0.4008 * T_C + 9.398e-4 * (T_C ** 2) - 1.410e-6 * (T_C ** 3)
    # A = 1.8246e6 / (epsilon * T_K)^1.5
    A = 1.8246e6 / ((epsilon * T_K) ** 1.5)
    return float(A)


def davies_gamma(z: int, I: float, A: float) -> float:
    """Davies equation for single ion activity coefficient gamma_i.

    log10(gamma_i) = -A * z^2 * (sqrt(I)/(1 + sqrt(I)) - 0.3 * I)

    Valid to I ≈ 0.5 mol/kg.  Concentrated FeSO4 baths sit at I ≈ 1.5–4
    mol/kg where this expression is uncalibrated and wrong — use the
    Pitzer path (default) for process numbers.
    """
    if I <= 0.0:
        return 1.0
    sqrt_I = math.sqrt(I)
    log_gamma = -A * (z ** 2) * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I)
    return float(10.0 ** log_gamma)


# ─── Shared helpers ─────────────────────────────────────────────────

def estimate_solution_density_kg_L(comp: SolutionComposition) -> float:
    """Approximate solution density (kg/L) from apparent-molar-volume increments.

    ρ ≈ ρ_water + Σ_i c_i · (Δρ/Δc)_i with literature-consistent
    increments (kg/L per mol/L, ~25 °C):

        FeSO4  : 0.20   (1 M FeSO4 solution ≈ 1.20 kg/L, ~13 wt %)
        Na2SO4 : 0.12   (1 M Na2SO4 solution ≈ 1.12 kg/L)
        H2SO4  : 0.06   (dilute acid)
        H3BO3  : 0.025  (boric acid)

    Screening-level estimate (±1–2 %); a measured bath density supersedes
    it.  Temperature dependence of the increments is neglected.
    """
    rho = _pitzer.water_density_kg_L(comp.T_C)
    rho += 0.20 * comp.c_FeSO4 + 0.12 * comp.c_Na2SO4
    rho += 0.06 * comp.c_H2SO4 + 0.025 * comp.c_H3BO3
    return float(rho)


def molar_to_molal(comp: SolutionComposition, rho_kg_L: float) -> Dict[str, float]:
    """Convert the molar recipe to species molalities (mol/kg H2O).

    Water mass per litre of solution = ρ − Σ c_salt·M_salt.
    """
    solute_mass = (
        comp.c_FeSO4 * M_FESO4 + comp.c_Na2SO4 * M_NA2SO4
        + comp.c_H2SO4 * M_H2SO4 + comp.c_H3BO3 * M_H3BO3
    ) / 1000.0  # kg per litre of solution
    kg_water_per_L = max(rho_kg_L - solute_mass, 0.3)

    m_Fe = comp.c_FeSO4 / kg_water_per_L
    m_Na = 2.0 * comp.c_Na2SO4 / kg_water_per_L
    m_H_from_acid = 2.0 * comp.c_H2SO4 / kg_water_per_L
    m_SO4_tot = (comp.c_FeSO4 + comp.c_Na2SO4 + comp.c_H2SO4) / kg_water_per_L
    return {
        "m_Fe_tot": m_Fe,
        "m_Na_tot": m_Na,
        "m_H_acid": m_H_from_acid,
        "m_SO4_tot": m_SO4_tot,
        "kg_water_per_L": kg_water_per_L,
    }


def _conductivity_S_m(
    conc_molar: Dict[str, float], I_molal: float, T_C: float
) -> float:
    """Electrolyte conductivity estimate (S/m) via ionic mobilities.

    κ = Σ_i λ°ᵢ cᵢ · T_factor / (1 + B_ATT·√I)

    with limiting molar conductivities at 25 °C (S·m²/mol), a viscosity
    temperature factor (~2.2 %/K), and a Debye–Hückel–Onsager-style
    relaxation/electrophoretic attenuation.  The attenuation coefficient
    B_ATT = 1.5 (kg/mol)^0.5 is calibrated so that pure 1 M FeSO4 gives
    ≈ 5.5–6 S/m (measured ≈ 5–6 S/m at 25 °C) and 1 M Na2SO4 ≈ 6–7 S/m
    (measured ≈ 7.5 S/m).  Concentrations are the *current-carrying*
    (unpaired) ions from the contact-pair estimate in Pitzer mode.

    Screening precision ≈ ±15–20 %.  A measured bath conductivity should
    supersede this estimate in any calibrated run.
    """
    lambda_25 = {"Fe2": 0.0108, "Na": 0.00501, "H": 0.03496,
                 "SO4": 0.0160, "HSO4": 0.0050}
    T_factor = 1.0 + 0.022 * (T_C - 25.0)
    I_factor = 1.0 / (1.0 + 1.5 * math.sqrt(max(I_molal, 0.0)))

    kappa = 1000.0 * (
        conc_molar.get("Fe2", 0.0) * lambda_25["Fe2"]
        + conc_molar.get("Na", 0.0) * lambda_25["Na"]
        + conc_molar.get("H", 0.0) * lambda_25["H"]
        + conc_molar.get("SO4", 0.0) * lambda_25["SO4"]
        + conc_molar.get("HSO4", 0.0) * lambda_25["HSO4"]
    ) * T_factor * I_factor  # S/m
    return float(kappa)


def _nernst_and_precip(a_Fe2: float, T_K: float) -> Dict[str, float]:
    """Nernst potential and Fe(OH)2 precipitation pH from Fe2+ activity."""
    E_rev_Fe = E0_FE2_FE + (R * T_K / (2.0 * F)) * math.log(max(1e-12, a_Fe2))
    Kw_T = KW_25 * math.exp((-55800.0 / R) * (1.0 / T_K - 1.0 / 298.15))
    Ksp_T = KSP_FE_OH2_25 * math.exp((-25000.0 / R) * (1.0 / T_K - 1.0 / 298.15))
    a_OH_precip = math.sqrt(max(1e-30, Ksp_T / max(1e-12, a_Fe2)))
    a_H_precip = Kw_T / max(1e-30, a_OH_precip)
    pH_precip = -math.log10(max(1e-14, a_H_precip))
    return {
        "E_rev_Fe_V_SHE": float(E_rev_Fe),
        "pH_precip_Fe_OH2": float(pH_precip),
    }


# ─── Pitzer-based speciation (default) ───────────────────────────────

def _solve_speciation_pitzer(comp: SolutionComposition) -> Dict[str, Any]:
    """Solve bath speciation on the Pitzer molal activity model.

    Equilibrium solved explicitly: HSO4- ⇌ H+ + SO4(2-) (Ka2, van't Hoff).
    Fe(II) is fully dissociated Fe2+ (association absorbed in the 2–2-salt
    Pitzer virial terms).  A secondary FeSO4^0(aq) contact-pair estimate
    (K = 200 kg/mol with Pitzer single-ion gammas) is produced for
    diagnostics and the conductivity budget only.
    """
    T_K = comp.T_C + 273.15
    rho = estimate_solution_density_kg_L(comp)
    mm = molar_to_molal(comp, rho)
    kw = mm["kg_water_per_L"]

    m_Fe = mm["m_Fe_tot"]
    m_Na = mm["m_Na_tot"]
    m_H_acid = mm["m_H_acid"]
    m_SO4_tot = mm["m_SO4_tot"]

    # Ka2 at temperature (van't Hoff, dH0 ≈ −22.4 kJ/mol)
    Ka2_T = K_HSO4_25 * math.exp((-22400.0 / R) * (1.0 / T_K - 1.0 / 298.15))

    # ── Solve bisulfate dissociation with Pitzer activities ──────────
    #   Ka2 = a_H · a_SO4 / a_HSO4 ;  h = m_HSO4
    h = m_H_acid * 0.9  # initial guess: nearly complete second dissociation
    sol = None
    for _ in range(40):
        h = min(max(h, 0.0), m_H_acid)
        m_H = m_H_acid - h
        m_SO4 = m_SO4_tot - h
        sol = _pitzer.solve_pitzer(
            {"Fe2+": m_Fe, "Na+": m_Na, "H+": max(m_H, 1e-16),
             "SO4-2": m_SO4, "HSO4-": max(h, 1e-16)},
            T_C=comp.T_C,
        )
        g = sol.gamma

        def resid(hh: float) -> float:
            mH = max(m_H_acid - hh, 1e-16)
            mS = max(m_SO4_tot - hh, 1e-16)
            lhs = g["H+"] * mH * g["SO4-2"] * mS
            rhs = Ka2_T * g["HSO4-"] * max(hh, 1e-16)
            return lhs - rhs

        lo, hi = 1e-16, m_H_acid + 1e-12
        if resid(1e-16) < 0.0:
            h_new = 1e-16
        else:
            for _bis in range(80):
                mid = 0.5 * (lo + hi)
                if resid(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            h_new = 0.5 * (lo + hi)
        if abs(h_new - h) < 1e-10 * max(h, 1e-12):
            h = h_new
            break
        h = h_new

    # Final gamma evaluation at converged h
    m_H = max(m_H_acid - h, 1e-16)
    m_SO4 = m_SO4_tot - h
    sol = _pitzer.solve_pitzer(
        {"Fe2+": m_Fe, "Na+": m_Na, "H+": m_H, "SO4-2": m_SO4,
         "HSO4-": max(h, 1e-16)},
        T_C=comp.T_C,
    )
    g = sol.gamma

    # ── pH, including a boric-acid-only fallback when no acid is dosed ──
    a_H = g["H+"] * m_H
    if comp.c_H2SO4 <= 0.0:
        # B(OH)3 + H2O ⇌ H+ + B(OH)4−;  a_H ≈ sqrt(Ka_b · a_H3BO3)
        Kab_T = KA_BORIC_25 * math.exp((-13800.0 / R) * (1.0 / T_K - 1.0 / 298.15))
        m_H3BO3 = comp.c_H3BO3 / kw
        a_H = math.sqrt(max(Kab_T, 1e-30) * max(m_H3BO3, 1e-12))
        m_H = a_H / max(g["H+"], 1e-6)
    pH_act = -math.log10(max(a_H, 1e-16))
    c_H_molar = m_H * kw
    pH_conc = -math.log10(max(c_H_molar, 1e-16))

    # ── Secondary contact-pair estimate (diagnostic / conductivity) ──
    # K_pair(conc) = K_thermo · γ_Fe · γ_SO4 / γ_neutral(≈1).  With honest
    # 2–2-salt gammas this yields a modest contact-pair fraction
    # (~10–25 %), consistent with spectroscopic estimates for divalent
    # sulfate systems — NOT the 97 % the Davies path produced.
    K_pair_T = K_FESO4_PAIR_25 * math.exp((8000.0 / R) * (1.0 / T_K - 1.0 / 298.15))
    Kc_pair = K_pair_T * g["Fe2+"] * g["SO4-2"]
    pair_frac = Kc_pair * m_SO4 / (1.0 + Kc_pair * m_SO4) if m_SO4 > 0 else 0.0
    c_Fe_unpaired = comp.c_FeSO4 * (1.0 - pair_frac)
    c_SO4_unpaired = max(m_SO4 * kw - comp.c_FeSO4 * pair_frac, 0.0)

    conductivity = _conductivity_S_m(
        {"Fe2": c_Fe_unpaired, "Na": comp.c_Na2SO4 * 2.0, "H": c_H_molar,
         "SO4": c_SO4_unpaired, "HSO4": h * kw},
        sol.ionic_strength_molal, comp.T_C,
    )

    a_Fe2 = g["Fe2+"] * m_Fe
    thermo = _nernst_and_precip(a_Fe2, T_K)

    return {
        "activity_model": "pitzer",
        "activity_scale": "molal",
        "temperature_C": comp.T_C,
        "solution_density_kg_L": float(rho),
        "ionic_strength_M": float(sol.ionic_strength_molal * kw),  # mol/L-scale
        "ionic_strength_molal": float(sol.ionic_strength_molal),
        "gamma_Fe2": float(g["Fe2+"]),
        "gamma_H": float(g["H+"]),
        "gamma_SO4": float(g["SO4-2"]),
        "gamma_Na": float(g["Na+"]),
        "gamma_HSO4": float(g["HSO4-"]),
        "osmotic_coefficient": float(sol.osmotic_coefficient),
        "water_activity": float(sol.water_activity),
        "c_Fe2_free_M": float(comp.c_FeSO4),  # fully dissociated convention
        "c_SO4_free_M": float(m_SO4 * kw),
        "c_H_free_M": float(c_H_molar),
        "c_HSO4_free_M": float(h * kw),
        "c_FeSO4_pair_M": float(comp.c_FeSO4 * pair_frac),
        "fe2_pair_percentage": float(100.0 * pair_frac),
        "fe2_pair_basis": "secondary contact-pair estimate (K=200, Pitzer gammas)",
        "a_Fe2": float(a_Fe2),
        "a_H": float(a_H),
        "pH_activity": float(pH_act),
        "pH_concentration": float(pH_conc),
        "conductivity_S_m": conductivity,
        "pitzer_parameter_temperature_window": (
            "Fe2+-SO4(2-): verified Kobylin et al. (2011) T-functions, 10-90 °C; "
            "other binaries frozen at 25 °C (see docs/PITZER_TCOEFF_ACCEPTANCE.md)"
        ),
        **thermo,
    }


# ─── Legacy Davies path (unchanged) ─────────────────────────────────

def _solve_speciation_davies(comp: SolutionComposition, max_iter: int = 200, tol: float = 1e-6) -> Dict[str, Any]:
    """Iteratively solve ionic equilibria with Davies activity coefficients.

    Equilibria:
    1) HSO4- <-> H+ + SO4(2-)   (Ka2)
    2) Fe(2+) + SO4(2-) <-> FeSO4(aq)   (K_pair)

    NOTE: Davies is uncalibrated above I ≈ 0.5 mol/kg; the concentrated
    sulfate baths of this program sit at I ≈ 1.5–4 mol/kg where this path
    over-binds ~97 % of the iron as FeSO4^0 pairs.  Retained for
    regression archaeology and A/B comparison only — prefer the default
    Pitzer path for process numbers.
    """
    T_K = comp.T_C + 273.15
    A_dh = davies_A(comp.T_C)

    Ka2_T = K_HSO4_25 * math.exp((-22400.0 / R) * (1.0 / T_K - 1.0 / 298.15))
    K_pair_T = K_FESO4_PAIR_25 * math.exp((8000.0 / R) * (1.0 / T_K - 1.0 / 298.15))

    c_Fe2 = comp.c_FeSO4
    c_Na = 2.0 * comp.c_Na2SO4
    c_H = 2.0 * comp.c_H2SO4
    c_HSO4 = 0.0
    c_FeSO4_pair = 0.0
    c_SO4 = comp.c_FeSO4 + comp.c_Na2SO4 + comp.c_H2SO4

    I = 0.5 * (4.0 * c_Fe2 + 1.0 * c_Na + 1.0 * c_H + 4.0 * c_SO4)

    for _ in range(max_iter):
        I_prev = I

        gamma1 = davies_gamma(1, I, A_dh)
        gamma2 = davies_gamma(2, I, A_dh)
        gamma0 = 1.0

        Kc_HSO4 = Ka2_T * gamma1 / (gamma1 * gamma2)
        Kc_pair = K_pair_T * (gamma2 * gamma2) / gamma0

        for _inner in range(20):
            c_Fe2 = comp.c_FeSO4 / (1.0 + Kc_pair * c_SO4)
            c_FeSO4_pair = comp.c_FeSO4 - c_Fe2

            c_H = (2.0 * comp.c_H2SO4) / (1.0 + c_SO4 / Kc_HSO4)
            c_HSO4 = 2.0 * comp.c_H2SO4 - c_H

            c_SO4_new = (comp.c_FeSO4 + comp.c_Na2SO4 + comp.c_H2SO4) - c_HSO4 - c_FeSO4_pair
            c_SO4_new = max(1e-8, c_SO4_new)

            c_SO4 = 0.5 * (c_SO4 + c_SO4_new)

        I_new = 0.5 * (4.0 * c_Fe2 + 1.0 * c_Na + 1.0 * c_H + 1.0 * c_HSO4 + 4.0 * c_SO4)
        I = max(1e-6, 0.5 * (I + I_new))

        if abs(I - I_prev) / I < tol:
            break

    gamma_Fe2 = davies_gamma(2, I, A_dh)
    gamma_H = davies_gamma(1, I, A_dh)
    gamma_SO4 = davies_gamma(2, I, A_dh)

    a_Fe2 = gamma_Fe2 * c_Fe2
    a_H = gamma_H * c_H if c_H > 0.0 else None
    pH_act = -math.log10(max(1e-14, a_H)) if a_H else float("nan")
    c_H_safe = max(c_H, 0.0)
    pH_conc = -math.log10(max(1e-14, c_H)) if c_H > 0 else float("nan")
    if c_H <= 0.0:
        # Boric-acid-only pH fallback (no sulfuric acid dosed).
        Kab_T = KA_BORIC_25 * math.exp((-13800.0 / R) * (1.0 / T_K - 1.0 / 298.15))
        a_H = math.sqrt(Kab_T * max(comp.c_H3BO3, 1e-12))
        pH_act = -math.log10(a_H)
        pH_conc = pH_act

    thermo = _nernst_and_precip(a_Fe2, T_K)

    lambda_25 = {"Fe2": 0.0108, "Na": 0.00501, "H": 0.03496, "SO4": 0.0160, "HSO4": 0.0050}
    T_factor = 1.0 + 0.022 * (comp.T_C - 25.0)
    I_factor = 1.0 / (1.0 + 0.45 * math.sqrt(I))
    kappa = 1000.0 * (
        c_Fe2 * lambda_25["Fe2"] + c_Na * lambda_25["Na"] + c_H_safe * lambda_25["H"]
        + c_SO4 * lambda_25["SO4"] + c_HSO4 * lambda_25["HSO4"]
    ) * T_factor * I_factor

    return {
        "activity_model": "davies",
        "temperature_C": comp.T_C,
        "ionic_strength_M": float(I),
        "davies_A": float(A_dh),
        "gamma_Fe2": float(gamma_Fe2),
        "gamma_H": float(gamma_H),
        "gamma_SO4": float(gamma_SO4),
        "c_Fe2_free_M": float(c_Fe2),
        "c_SO4_free_M": float(c_SO4),
        "c_H_free_M": float(c_H_safe),
        "c_HSO4_free_M": float(c_HSO4),
        "c_FeSO4_pair_M": float(c_FeSO4_pair),
        "a_Fe2": float(a_Fe2),
        "a_H": float(a_H),
        "pH_activity": float(pH_act),
        "pH_concentration": float(pH_conc),
        "conductivity_S_m": float(kappa),
        "fe2_pair_percentage": float(100.0 * c_FeSO4_pair / comp.c_FeSO4) if comp.c_FeSO4 > 0 else 0.0,
        **thermo,
    }


def solve_speciation(
    comp: SolutionComposition,
    max_iter: int = 200,
    tol: float = 1e-6,
    model: str = "pitzer",
) -> Dict[str, Any]:
    """Solve bath speciation and activities.

    Parameters
    ----------
    comp : SolutionComposition
        Nominal bath recipe (molar).
    model : {"pitzer", "davies"}
        Activity model.  Default ``"pitzer"`` (multicomponent Pitzer,
        valid at electrowinning-relevant ionic strengths).  ``"davies"``
        retains the legacy behaviour for comparison only.

    Returns
    -------
    dict with free concentrations, activity coefficients, activities,
    ionic strength, conductivity, Nernst potential, Fe(OH)2 precipitation
    pH, water activity/osmotic coefficient (Pitzer mode), and provenance
    metadata.
    """
    if model == "pitzer":
        return _solve_speciation_pitzer(comp)
    if model == "davies":
        return _solve_speciation_davies(comp, max_iter=max_iter, tol=tol)
    raise ValueError(f"unknown activity model: {model!r} (expected 'pitzer' or 'davies')")


def speciation_temperature_sweep(
    comp: SolutionComposition,
    T_min: float = 20.0,
    T_max: float = 80.0,
    num: int = 13,
    model: str = "pitzer",
) -> Dict[str, np.ndarray]:
    """Perform speciation sweep across temperature range."""
    temps = np.linspace(T_min, T_max, num)
    res_list = []
    for T in temps:
        c = SolutionComposition(c_FeSO4=comp.c_FeSO4, c_Na2SO4=comp.c_Na2SO4,
                                c_H2SO4=comp.c_H2SO4, c_H3BO3=comp.c_H3BO3, T_C=float(T))
        res_list.append(solve_speciation(c, model=model))

    keep = [k for k in res_list[0] if isinstance(res_list[0][k], (int, float))]
    out: Dict[str, np.ndarray] = {"temperature_C": temps}
    for k in keep:
        if k != "temperature_C":
            out[k] = np.array([r[k] for r in res_list], dtype=float)
    return out
