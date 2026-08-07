"""Bath impurity co-deposition — S, P, Mn, Si, B — and AISI 10xx routing.

Why this module exists
----------------------
The existing ``impurity_codeposition.py`` models the *metallic* impurities
Cu / Ni / Zn / Pb / Sn via Butler–Volmer partial currents capped by
Koutecky–Levich transport.  Real electrolytic iron also picks up the
steel-grade-defining non-Ni/C impurities — Mn, S, P, Si, B — that decide
which AISI grade a deposit routes to:

* **Mn** (substitutional, metallic) co-deposits as Mn²⁺ → Mn, same
  Butler–Volmer mechanism as Cu/Ni/Zn.
* **S**, **P**, **Si**, **B** enter chiefly as *anionic / oxy-anion*
  species (S²⁻/SO₃²⁻, PO₄/phosphide, SiO₃²⁻/silicate) via surface adsorption
  (Langmuir), driven by concentration and reduction/incorporation at the
  growing interface — the framework the Langmuir machinery in
  ``co_deposition.py`` / the adsorption picture in ``impurity_codeposition.py``
  already models.

This is the Tier 1.2 placeholder (CHEM_PHYS_REVIEW.md §1.2): it extends the
``BathKinetics`` framework to S/P/Mn/Si/B and reports the deposit content
that routes a foil to **AISI 1005 vs 1018 vs low-sulfur deep-drawing**.

This is a screening module.  All numbers carry
``SCREENING_FLAG = "unvalidated (L1)"`` and are **not** gate evidence; real
grade routing needs combustion/inert-gas OES analysis of the deposit.

References (screening calibrations)
-----------------------------------
* Brenner, *Electrodeposition of Alloys* (1963) — impurity co-deposition.
* Schlesinger & Paunovic, *Modern Electroplating* 5th ed. (2010).
* AISI/SAE standard carbon-steel composition limits (10xx series):
  Mn ≤ 0.35 % (1005) / 0.60–0.90 % (1018), P ≤ 0.040 %, S ≤ 0.050 %
  (general) and ≤ 0.01–0.02 % for low-sulfur deep-drawing grades.
* Task spec: t_f0a8f3a0 — Solid-phase Fe chemistry, Tier 1.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional, Tuple

import numpy as np

from .electrochemistry import FARADAY, R_GAS, E0_FE, M_FE, Z_FE
from .impurity_codeposition import BathKinetics as _BathKinetics
from .impurity_codeposition import limiting_current_density  # noqa: F401  (re-export parity)

# ── Honesty flag ─────────────────────────────────────────────────────────────
SCREENING_FLAG = "unvalidated (L1)"

# ── Metallic impurity: manganese (cationic Butler–Volmer path) ──────────────
M_MN = 54.938e-3            # kg/mol
Z_MN = 2                    # Mn²⁺ → Mn
E0_MN = -1.185              # V vs. SHE (Mn²⁺/Mn)
D_MN_DEFAULT = 0.71e-9      # m²/s

# ── Anionic / oxy-anion impurities: Langmuir adsorption path ────────────────
# ``M_*`` = molar mass of the *element* as it appears in the deposit.
M_S = 32.06e-3              # kg/mol  (sulfur)
M_P = 30.974e-3             # kg/mol  (phosphorus)
M_SI = 28.085e-3            # kg/mol  (silicon)
M_B = 10.81e-3              # kg/mol  (boron)

# Langmuir adsorption coefficients K (per ppm) and incorporation scaling.
# Screening central values; K_* = 1/C_half, so a larger K = more uptake at
# low bath concentration.
K_S_PER_PPM = 0.02
K_P_PER_PPM = 0.01
K_SI_PER_PPM = 0.008
K_B_PER_PPM = 0.012

# Transfer efficiency: fraction of *adsorbed* anionic impurity that is
# electrochemically reduced / incorporated into the deposit per area-time.
ANION_ABSORPTION_BASE = 1.0e-3   # dimensionless overall uptake scale


# ── AISI routing thresholds (wt%) ───────────────────────────────────────────
AISI_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "AISI_1005": {"c_min": 0.0,   "c_max": 0.06,  "mn_max": 0.35, "p_max": 0.040, "s_max": 0.050},
    "AISI_1006": {"c_min": 0.0,   "c_max": 0.08,  "mn_max": 0.25, "p_max": 0.040, "s_max": 0.050},
    "AISI_1010": {"c_min": 0.08,  "c_max": 0.13,  "mn_max": 0.60, "p_max": 0.040, "s_max": 0.050},
    "AISI_1018": {"c_min": 0.15,  "c_max": 0.20,  "mn_max": 0.90, "p_max": 0.040, "s_max": 0.050},
}
LOW_SULFUR_S_MAX = 0.020       # deep-drawing ceiling on S
LOW_PHOSPHORUS_P_MAX = 0.020   # deep-drawing ceiling on P


# ── Kinetic parameter set (extends the BathKinetics framework) ──────────────


@dataclass(frozen=True)
class BathImpurityKinetics:
    """Kinetic parameters for S/P/Mn/Si/B co-deposition.

    Mirrors ``impurity_codeposition.BathKinetics`` for the metallic (Mn)
    channel and adds Langmuir adsorption coefficients for the anionic
    (S/P/Si/B) channels.  Users may supply their own instance to calibrate.
    """

    # Mn channel — Butler–Volmer (same as BathKinetics' cu/ni/zn nomenclature)
    mn_i0: float = 8.0e-3
    mn_tafel: float = 0.120
    mn_lim_factor: float = 1.0

    # Anionic Langmuir uptake coefficients (per ppm) and absorption scale
    k_s: float = K_S_PER_PPM
    k_p: float = K_P_PER_PPM
    k_si: float = K_SI_PER_PPM
    k_b: float = K_B_PER_PPM
    anion_absorption: float = ANION_ABSORPTION_BASE

    # Conductivity factor (multiplier on diffusivity, parity with BathKinetics)
    conductivity_factor: float = 1.0


# Sulfate-default parameter set (screening)
SULFATE_IMPURITY_KINETICS = BathImpurityKinetics()
CHLORIDE_IMPURITY_KINETICS = BathImpurityKinetics(mn_i0=1.6e-2, mn_tafel=0.110,
                                                  conductivity_factor=1.3)

IMPURITY_KINETICS: Dict[str, BathImpurityKinetics] = {
    "sulfate": SULFATE_IMPURITY_KINETICS,
    "chloride": CHLORIDE_IMPURITY_KINETICS,
}


# ── Langmuir helper ─────────────────────────────────────────────────────────


def langmuir_coverage(conc_ppm: float, k_per_ppm: float) -> float:
    """Single-species Langmuir surface coverage (0–1): θ = KC/(1+KC)."""
    kc = max(k_per_ppm, 0.0) * max(conc_ppm, 0.0)
    return float(kc / (1.0 + kc))


# ── Main model ──────────────────────────────────────────────────────────────


@dataclass
class BathImpurityCoDeposition:
    """Predict S/P/Mn/Si/B uptake in an iron electrodeposit.

    Parameters
    ----------
    fe_conc_M : float
        Bulk Fe²⁺ concentration (mol/L).
    mn_ppm, s_ppm, p_ppm, si_ppm, b_ppm : float
        Bath impurity concentrations (ppm by mass).
    pH, temperature_C, boundary_layer_m : bath state (Mn channel).
    bath_type : {"sulfate", "chloride"}
        Electrolyte family (selects default kinetics).
    custom_kinetics : BathImpurityKinetics or None
        Override built-in kinetic set.
    """

    fe_conc_M: float = 1.0
    mn_ppm: float = 100.0
    s_ppm: float = 50.0
    p_ppm: float = 30.0
    si_ppm: float = 20.0
    b_ppm: float = 5.0
    pH: float = 3.0
    temperature_C: float = 60.0
    boundary_layer_m: float = 5e-5
    bath_type: Literal["sulfate", "chloride"] = "sulfate"
    custom_kinetics: Optional[BathImpurityKinetics] = None

    @property
    def T_K(self) -> float:
        return self.temperature_C + 273.15

    @property
    def kinetics(self) -> BathImpurityKinetics:
        if self.custom_kinetics is not None:
            return self.custom_kinetics
        return IMPURITY_KINETICS[self.bath_type]

    # --- Mn channel: Butler–Volmer partial current (mirror BathKinetics) ----

    @staticmethod
    def _ppm_to_mol_per_m3(ppm: float, molar_mass_kg: float) -> float:
        """ppm (mg/L) → mol/m³."""
        return ppm * 1e-3 / molar_mass_kg

    def _mn_partial_current(self, E_V: float) -> float:
        """Mn²⁺ → Mn partial current (A/m²) via cathodic Tafel + Koutecky–Levich."""
        kin = self.kinetics
        c = self._ppm_to_mol_per_m3(self.mn_ppm, M_MN)
        if c <= 0.0:
            return 0.0
        E_eq = E0_MN + (R_GAS * self.T_K / (Z_MN * FARADAY)) * np.log(
            max(c / 1000.0, 1e-30))
        eta = E_eq - E_V
        if eta <= 0.0:
            return 0.0
        i_kin = kin.mn_i0 * 10.0 ** (eta / kin.mn_tafel)
        eff_delta = self.boundary_layer_m / kin.conductivity_factor
        i_lim = limiting_current_density(c, D_MN_DEFAULT, eff_delta, Z_MN)
        return float(1.0 / (1.0 / max(i_kin, 1e-30) + 1.0 / max(i_lim, 1e-30)))

    def _fe_current(self, j_mA_cm2: float) -> Tuple[float, float]:
        """Fe cathode potential and partial current at the applied j
        (assume applied ≈ Fe current)."""
        kin = self.kinetics
        j_A_m2 = j_mA_cm2 * 10.0
        c_fe = self.fe_conc_M * 1000.0
        E_eq = E0_FE + (R_GAS * self.T_K / (Z_FE * FARADAY)) * np.log(
            max(c_fe / 1000.0, 1e-30))
        # use impurity_k.kinetics' mn_i0 as a stand-in for the Fe i0 surrogate:
        # in the parent module the Fe cathode potential comes from Fe's own
        # Tafel; we reuse a representative Fe i0 here.
        fe_i0 = 1.0e-2
        E_V = E_eq - 0.120 * np.log10(max(j_A_m2 / fe_i0, 1e-30))
        i_fe = j_A_m2
        return E_V, i_fe

    # --- Anionic channel: Langmuir incorporation ---------------------------

    def _anion_mass_rate_kg_m2_s(
        self, conc_ppm: float, k_per_ppm: float, m_element: float,
        j_mA_cm2: float,
    ) -> float:
        """Mass incorporation rate (kg/m²/s) for an anionic impurity.

        Langmuir coverage θ of the adsorbed species, folded into a rate
        proportional to the Fe deposition flux (co-incorporation during
        growth).  Scaling with j keeps the rate bounded at high field.
        """
        theta = langmuir_coverage(conc_ppm, k_per_ppm)
        m_fe = j_mA_cm2 * 10.0 * M_FE / (2.0 * FARADAY)
        j_scale = (j_mA_cm2 / 100.0) ** 0.5 if j_mA_cm2 > 0 else 0.0
        uptake = self.kinetics.anion_absorption * theta * j_scale
        # Convert to a mass rate of the *element* co-deposited per Fe mass.
        return float(uptake * m_fe * (m_element / M_FE))

    def deposit_composition(self, j_mA_cm2: float) -> Dict[str, Any]:
        """Deposit S/P/Mn/Si/B content (wt% and ppm) at a given current density.

        Returns
        -------
        dict with ``mn/s/p/si/b`` ``_wt_percent`` and ``_in_ppm``,
        the Mn partial current, bath_type and current density, and the
        ``flag``.
        """
        E_V, i_fe = self._fe_current(j_mA_cm2)

        # Mn — mass rate from partial current (mol → kg)
        i_mn = self._mn_partial_current(E_V)
        m_mn = i_mn / (Z_MN * FARADAY) * M_MN

        # Anionics — Langmuir mass rates
        m_s = self._anion_mass_rate_kg_m2_s(self.s_ppm, self.kinetics.k_s,
                                            M_S, j_mA_cm2)
        m_p = self._anion_mass_rate_kg_m2_s(self.p_ppm, self.kinetics.k_p,
                                            M_P, j_mA_cm2)
        m_si = self._anion_mass_rate_kg_m2_s(self.si_ppm, self.kinetics.k_si,
                                             M_SI, j_mA_cm2)
        m_b = self._anion_mass_rate_kg_m2_s(self.b_ppm, self.kinetics.k_b,
                                            M_B, j_mA_cm2)

        m_fe = i_fe / (Z_FE * FARADAY) * M_FE
        total = m_fe + m_mn + m_s + m_p + m_si + m_b
        if total <= 0.0:
            raise ValueError("Fe mass flux is zero; cannot compute composition")

        def _wt(m: float) -> float:
            return 100.0 * m / total

        return {
            "potential_V": E_V,
            "fe_wt_percent": _wt(m_fe),
            "mn_wt_percent": _wt(m_mn),
            "s_wt_percent": _wt(m_s),
            "p_wt_percent": _wt(m_p),
            "si_wt_percent": _wt(m_si),
            "b_wt_percent": _wt(m_b),
            "mn_in_ppm": _wt(m_mn) * 1e4,
            "s_in_ppm": _wt(m_s) * 1e4,
            "p_in_ppm": _wt(m_p) * 1e4,
            "si_in_ppm": _wt(m_si) * 1e4,
            "b_in_ppm": _wt(m_b) * 1e4,
            "total_impurity_wt": _wt(m_mn) + _wt(m_s) + _wt(m_p) + _wt(m_si) + _wt(m_b),
            "mn_current_A_m2": i_mn,
            "bath_type": self.bath_type,
            "current_density_mA_cm2": j_mA_cm2,
            "flag": SCREENING_FLAG,
        }


# ── AISI 10xx routing ───────────────────────────────────────────────────────


def route_steel_grade(
    c_wt_percent: float,
    mn_wt_percent: float,
    p_wt_percent: float,
    s_wt_percent: float,
    si_wt_percent: float = 0.0,
) -> Dict[str, Any]:
    """Route a deposit's S/P/Mn/C composition to an AISI 10xx grade.

    The impurity module's job (CHEM_PHYS_REVIEW.md §1.2): S, P, Mn, Si decide
    whether a foil routes to AISI 1005 vs 1018 vs a low-sulfur deep-drawing
    grade.  Resolution order:

    1. If S or P exceeds the general ceiling → "resulfurized / not
       deep-drawing" (free-machining or rejected for deep-draw).
    2. If both S and P are at the low-sulfur deep-drawing level → flag as
       deep-drawing candidate (S and P are the discriminator species).
    3. Otherwise pick the best-fit 10xx grade by C and Mn ranges.

    Parameters
    ----------
    c_wt, mn_wt, p_wt, s_wt, si_wt : float
        Deposit composition (wt%).

    Returns
    -------
    dict with ``grade``, ``category`` (one of "deep_drawing",
    "resulfurized", "carbon"), ``reason`` and the composition echoed back.
    """
    grades = [
        ("AISI_1005", AISI_THRESHOLDS["AISI_1005"]),
        ("AISI_1006", AISI_THRESHOLDS["AISI_1006"]),
        ("AISI_1010", AISI_THRESHOLDS["AISI_1010"]),
        ("AISI_1018", AISI_THRESHOLDS["AISI_1018"]),
    ]

    # 1 — general ceilings
    if s_wt_percent > AISI_THRESHOLDS["AISI_1018"]["s_max"]:
        return {
            "grade": "resulfurized",
            "category": "resulfurized",
            "reason": f"S {s_wt_percent:.3f}% exceeds the 0.050% general ceiling — "
                      f"not a clean deep-drawing grade (free-machining / reject).",
            "s_wt_percent": s_wt_percent, "p_wt_percent": p_wt_percent,
            "mn_wt_percent": mn_wt_percent, "c_wt_percent": c_wt_percent,
            "si_wt_percent": si_wt_percent, "flag": SCREENING_FLAG,
        }
    if p_wt_percent > AISI_THRESHOLDS["AISI_1018"]["p_max"]:
        return {
            "grade": "high-phosphorus",
            "category": "resulfurized",
            "reason": f"P {p_wt_percent:.3f}% exceeds the 0.040% ceiling.",
            "s_wt_percent": s_wt_percent, "p_wt_percent": p_wt_percent,
            "mn_wt_percent": mn_wt_percent, "c_wt_percent": c_wt_percent,
            "si_wt_percent": si_wt_percent, "flag": SCREENING_FLAG,
        }

    # 2 — low-sulfur deep-drawing
    if s_wt_percent <= LOW_SULFUR_S_MAX and p_wt_percent <= LOW_PHOSPHORUS_P_MAX:
        return {
            "grade": "low-sulfur deep-drawing",
            "category": "deep_drawing",
            "reason": f"S {s_wt_percent:.3f}% ≤ {LOW_SULFUR_S_MAX:.3f}% and "
                      f"P {p_wt_percent:.3f}% ≤ {LOW_PHOSPHORUS_P_MAX:.3f}% — "
                      f"clean enough for deep drawing.",
            "s_wt_percent": s_wt_percent, "p_wt_percent": p_wt_percent,
            "mn_wt_percent": mn_wt_percent, "c_wt_percent": c_wt_percent,
            "si_wt_percent": si_wt_percent, "flag": SCREENING_FLAG,
        }

    # 3 — best-fit 10xx by C & Mn
    best, best_name, best_score = None, None, None
    for name, th in grades:
        c_ok = th["c_min"] <= c_wt_percent <= th["c_max"]
        mn_ok = mn_wt_percent <= th["mn_max"]
        if c_ok and mn_ok:
            return {
                "grade": name, "category": "carbon",
                "reason": f"C {c_wt_percent:.3f}% within {th['c_min']}-{th['c_max']}% "
                          f"and Mn {mn_wt_percent:.3f}% ≤ {th['mn_max']}%.",
                "s_wt_percent": s_wt_percent, "p_wt_percent": p_wt_percent,
                "mn_wt_percent": mn_wt_percent, "c_wt_percent": c_wt_percent,
                "si_wt_percent": si_wt_percent, "flag": SCREENING_FLAG,
            }

    return {
        "grade": "out-of-spec-10xx",
        "category": "carbon",
        "reason": "Deposit composition does not fall in a standard 10xx C/Mn box.",
        "s_wt_percent": s_wt_percent, "p_wt_percent": p_wt_percent,
        "mn_wt_percent": mn_wt_percent, "c_wt_percent": c_wt_percent,
        "si_wt_percent": si_wt_percent, "flag": SCREENING_FLAG,
    }
