"""
Anode kinetics for aqueous iron electrowinning.

Implements the anode half of the full cell, covering:

* Oxygen Evolution Reaction (OER) on dimensionally-stable anodes (DSAs) —
  IrO₂–Ta₂O₅ on Ti in acidic / neutral baths, and Ni–Co spinel oxides in
  alkaline baths.
* Chlorine Evolution Reaction (CER) — 2Cl⁻ → Cl₂ + 2e⁻ — in high-chloride
  AWARE-type acidic electrolytes where it competes with OER and may be the
  thermodynamically favoured anodic process.
* Ohmic drop from bubble-induced electrolyte resistance.
* Concentration (gas-diffusion) overpotential from O₂ transport away from
  the anode surface.

The anode overpotential is decomposed as:

    η_anode = η_activation + η_concentration + η_bubble

where each term is a function of current density, electrolyte composition,
temperature, and anode material type.  When CER is active, the anode
potential is the mixed potential at which the sum of anodic currents equals
the applied current density.

References
----------
* Trasatti, S. (2000). Electrocatalysis by oxides — attempt at a unifying
  approach. J. Electroanal. Chem., 503, 251–268.
* Spijkerman, J.J. & Abbott, D.F. (2025). Iridium-based OER catalysts.
  Handbook of Electrochemistry (2nd ed.).
* Yuan et al. (2009). Iron electrowinning from NaOH solution.
  Hydrometallurgy, 98, 10–15.
* AWARE process (2024–2025). Acidic electrowinning in anion-rich
  electrolytes. ChemRxiv / follow-up publications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import numpy as np
from scipy.optimize import brentq

from .electrochemistry import FARADAY, R_GAS
from .pourbaix import oer_line


T_REF = 298.15          # K  — reference temperature
P_O2_STD = 1.0          # bar — standard O₂ partial pressure
P_CL2_STD = 1.0         # bar — standard Cl₂ partial pressure

# ─── Thermodynamic equilibrium potentials (V vs. SHE) ─────────────────

# OER in acid:  2H₂O → O₂ + 4H⁺ + 4e⁻
E0_OER_ACIDIC = 1.229

# OER in base: 4OH⁻ → O₂ + 2H₂O + 4e⁻  (Nernst-corrected by pH)
# E_eq(base) = 0.401 V vs. SHE at pH 14, 25 °C
# General:     E_eq = 1.229 − 0.0591·pH  (at 25 °C)
E0_OER_ALKALINE = 0.401   # vs. SHE at pH 14, 25 °C; use Nernst shift

# CER:         2Cl⁻ → Cl₂ + 2e⁻
E0_CER = 1.360            # vs. SHE at a_Cl- = 1 M, 25 °C (AWARE bath)


# ─── Physical constants ─────────────────────────────────────────────────

# R = 8.314 J/(mol·K) imported from electrochemistry
# F = 96485 C/mol
# RT/F at 298.15 K = 0.025693 V
RT_F_298 = R_GAS * T_REF / FARADAY


# ─── Anode material definitions ────────────────────────────────────────

@dataclass
class AnodeMaterial:
    """
    Kinetic and material properties for one anode type.

    Parameters
    ----------
    name : str
        Human-readable label.
    oer_i0 : float
        OER exchange current density (A/m²).
    oer_tafel_V : float
        Anodic Tafel slope for OER (V/decade).  Typical range 0.040–0.120 V.
    cer_i0 : float or None
        CER exchange current density (A/m²).  None = CER not active.
    cer_tafel_V : float
        Anodic Tafel slope for CER (V/decade).
    cer_n : int
        Electrons transferred per CER mole reaction (2 for 2Cl⁻ → Cl₂).
    oer_n : int
        Electrons transferred per OER mole reaction (4 for 2H₂O → O₂).
    max_bubble_fraction : float
        Maximum gas-void fraction achievable on this surface (0–1).
        Controls the upper bound of bubble-induced resistance.
    temperature_C : float
        Operating temperature (°C).  Affects exchange current via Arrhenius.
    references : str
        Literature source(s) for the parameters.
    """

    name: str
    oer_i0: float                      # A/m² at T_REF
    oer_tafel_V: float                  # V/decade
    cer_i0: Optional[float] = None     # A/m²; None = CER suppressed
    cer_tafel_V: float = 0.040         # V/decade
    cer_n: int = 2
    oer_n: int = 4
    max_bubble_fraction: float = 0.15  # maximum void fraction
    temperature_C: float = 60.0       # operating temperature
    oer_ea_kj_mol: float = 40.0     # kJ/mol — activation energy for OER on IrO₂
    references: str = ""

    @property
    def T(self) -> float:
        return self.temperature_C + 273.15

    def oer_i0_at_T(self) -> float:
        """Exchange current at operating temperature (Arrhenius correction)."""
        ea = self.oer_ea_kj_mol * 1000.0   # J/mol
        return self.oer_i0 * np.exp(-ea / R_GAS * (1.0 / self.T - 1.0 / T_REF))


# ─── Catalogue of well-characterised anode materials ──────────────────

# 1. IrO₂–Ta₂O₅ DSA on Ti — acidic or neutral baths (Trasatti 2000;
#    Spijkerman & Abbott 2025)
DSA_IRO2_TA2O5: AnodeMaterial = AnodeMaterial(
    name="IrO₂–Ta₂O₅ / Ti DSA",
    oer_i0=1e-3,                # A/m² — low-index estimate for IrO₂
    oer_tafel_V=0.060,          # V/decade (40–70 mV typical)
    cer_i0=1e-1,                # A/m² — CER is fast on IrO₂
    cer_tafel_V=0.040,          # V/decade
    cer_n=2,
    oer_n=4,
    max_bubble_fraction=0.12,
    temperature_C=60.0,
    oer_ea_kj_mol=40.0,
    references="Trasatti (2000) J. Electroanal. Chem.; Spijkerman & Abbott (2025)",
)

# 2. Ni–Co spinel (NiCo₂O₄) — alkaline baths (Yuan & Haarberg 2009;
#    Kempler et al. 2025).  Ni–Co spinels show very low OER overpotential.
NICO_SPINEL: AnodeMaterial = AnodeMaterial(
    name="NiCo₂O₄ / Ni foam",
    oer_i0=1e-1,                # A/m² — much higher than IrO₂ in base
    oer_tafel_V=0.060,          # V/decade
    cer_i0=None,                # not relevant in alkaline
    cer_tafel_V=0.040,
    cer_n=2,
    oer_n=4,
    max_bubble_fraction=0.10,
    temperature_C=90.0,
    oer_ea_kj_mol=30.0,
    references="Yuan & Haarberg (2009) Hydrometallurgy; Kempler et al. (2025) ACS Nano",
)

# 3. NiFe LDH — alkaline, next-generation low-cost alternative
NIFE_LDH: AnodeMaterial = AnodeMaterial(
    name="NiFe LDH / Ni foam",
    oer_i0=1e0,                 # A/m² — very active
    oer_tafel_V=0.040,          # V/decade (as low as 30–35 mV reported)
    cer_i0=None,
    cer_tafel_V=0.040,
    cer_n=2,
    oer_n=4,
    max_bubble_fraction=0.08,
    temperature_C=80.0,
    oer_ea_kj_mol=20.0,
    references="Song & Hu (2014) ChemSusChem; Trotochaud et al. (2014) J. Am. Chem. Soc.",
)

# 4. Pt/Ti — laboratory reference anode; fast OER but expensive and Pt
#    dissolves at high potentials.  Used for benchmarking.
PT_TI: AnodeMaterial = AnodeMaterial(
    name="Pt / Ti (reference)",
    oer_i0=1e-1,                # A/m² — Pt OER is moderate
    oer_tafel_V=0.060,          # V/decade
    cer_i0=1e0,                 # A/m² — Pt is excellent for CER
    cer_tafel_V=0.040,
    cer_n=2,
    oer_n=4,
    max_bubble_fraction=0.08,
    temperature_C=25.0,
    oer_ea_kj_mol=50.0,
    references="Angerstein-Kozlowska et al. (1973) Comprehensive Treatise of Electrochemistry",
)


# ─── Bubble resistance model ───────────────────────────────────────────

def bubble_fraction(
    j_mA_cm2: float,
    temperature_C: float = 60.0,
    anode_material: str = "IrO2",
) -> float:
    """
    Void fraction of bubbles on an anode surface at a given current density.

    θ(j) = θ_max · tanh(j / j_char)  (smooth, saturating)

    where θ_max is the material-specific maximum and j_char is a
    characteristic current density (empirically ≈ 50–200 mA/cm²).

    Bubbles increase the electrolyte Ohmic drop approximately linearly
    with θ: R_bubble = R_electrolyte · θ / (1 − θ).

    Parameters
    ----------
    j_mA_cm2 : float
        Anodic current density (mA/cm²).
    temperature_C : float
        Electrolyte temperature (°C).  Higher T → smaller bubbles → lower θ.
    anode_material : str
        "IrO2", "NiCo", "NiFe", or "Pt".  Controls θ_max.

    Returns
    -------
    float
        Void fraction (0–1).
    """
    material_maxima = {
        "IrO2": 0.12,
        "NiCo": 0.10,
        "NiFe": 0.08,
        "Pt":   0.08,
    }
    theta_max = material_maxima.get(anode_material, 0.10)

    # Temperature correction: higher T reduces bubble size and detachment rate.
    # Approximate scaling: θ ∝ exp(−β·T) with β ≈ 0.007 K⁻¹
    T_corr = np.exp(-0.007 * (temperature_C - 25.0))
    theta_max *= T_corr

    # Characteristic current density for bubble coverage saturation
    j_char = 150.0   # mA/cm²
    return float(theta_max * np.tanh(j_mA_cm2 / j_char))


def bubble_resistance_multiplier(theta: float) -> float:
    """
    Multiplicative factor for electrolyte resistance due to bubble coverage.

    R_bubble = R_electrolyte · θ / (1 − θ)

    This extra resistance is dominated by the gas-liquid interface at the
    electrode surface and the disrupted ionic paths between bubbles.

    Parameters
    ----------
    theta : float
        Bubble void fraction (0–1).

    Returns
    -------
    float
        Multiplier on electrolyte Ohmic drop (dimensionless, ≥ 1).
    """
    if theta >= 1.0:
        return 1e6   # electrically insulating — physically unrealistic
    return 1.0 + theta / max(1.0 - theta, 1e-6)


# ─── Concentration overpotential ───────────────────────────────────────

def concentration_overpotential_oer(
    j_mA_cm2: float,
    temperature_C: float = 60.0,
    boundary_layer_m: float = 5e-5,
    diffusivity_O2_m2_s: float = 2.0e-9,
    bulk_O2_mol_m3: float = 0.25,   # ≈ 8 mg/L dissolved O₂ in aerated water
) -> float:
    """
    Concentration overpotential at the OER anode (V).

    At high current densities the O₂ bubbles generated at the surface
    create a diffusion barrier, effectively lowering the local O₂ activity
    and requiring a more-positive anode potential to sustain the same
    current density.

    Model: linear stagnant film, O₂ diffusion away from surface.
    η_conc = (RT / nF) · ln(a_O2,bulk / a_O2,surface)

    The surface O₂ activity approaches zero at the limiting current,
    giving η_conc,max = (RT / nF) · ln(a_O2,bulk / 1e-9) ≈ 0.05–0.09 V.

    Parameters
    ----------
    j_mA_cm2 : float
        Anodic current density (mA/cm²).
    temperature_C : float
        Temperature (°C) for RT/nF correction.
    boundary_layer_m : float
        Anode diffusion layer thickness (m).
    diffusivity_O2_m2_s : float
        O₂ diffusion coefficient in the electrolyte.
    bulk_O2_mol_m3 : float
        Bulk dissolved O₂ concentration (mol/m³).

    Returns
    -------
    float
        Concentration overpotential (V), positive.
    """
    if j_mA_cm2 <= 0.0:
        return 0.0

    n = 4  # OER electrons
    T = temperature_C + 273.15
    RT_nF = R_GAS * T / (n * FARADAY)

    # OER limiting current (O₂ diffusion away from surface)
    # Flux of O₂ leaving = j_anode / (4F)  (mol/(m²·s))
    # i_lim = 4 F D C_bulk / δ
    i_lim = n * FARADAY * diffusivity_O2_m2_s * bulk_O2_mol_m3 / boundary_layer_m
    j_A_m2 = j_mA_cm2 * 10.0

    if j_A_m2 >= i_lim:
        # Mass-transport limited; cap η_conc
        return float(RT_nF * np.log(bulk_O2_mol_m3 / 1e-9))

    # a_O2,surface / a_O2,bulk = 1 − j / i_lim  (linear profile approximation)
    ratio = max(1e-9 / bulk_O2_mol_m3, 1.0 - j_A_m2 / i_lim)
    return float(RT_nF * np.abs(np.log(ratio)))


# ─── Core anode kinetics class ─────────────────────────────────────────

@dataclass
class AnodeKinetics:
    """
    Complete anode model for aqueous iron electrowinning.

    Combines OER and (optionally) CER Tafel kinetics, bubble resistance,
    and O₂ mass-transport concentration overpotential to compute the total
    anodic overpotential at any current density.

    The mixed OER/CER potential is found by Newton iteration on the current
    balance i_OER(E) + i_CER(E) = i_applied.  When CER is not active the
    model reduces to a pure OER anode.

    Parameters
    ----------
    material : AnodeMaterial
        Anode material specification.
    electrolyte_type : {"acidic", "alkaline", "acidic_chloride"}
        Electrolyte regime, which determines the dominant OER equilibrium
        reaction and whether CER is thermodynamically accessible.
    pH : float
        Bulk electrolyte pH (used for Nernst correction of E_eq).
    a_Cl_molar : float, default 0.0
        Molar activity of Cl⁻ (mol/L).  Set > 0 to activate CER kinetics.
        For the AWARE process set a_Cl_molar ≈ 10–12 (concentrated LiCl).
    boundary_layer_m : float
        Anode diffusion layer thickness (m).
    electrolyte_conductivity_S_m : float
        Electrolyte ionic conductivity (S/m).  Used for Ohmic correction
        when bubble_fraction is applied.
    electrolyte_resistivity_ohm_m2 : float
        Area-specific electrolyte resistance (Ω·m²) between electrodes.
    """

    material: AnodeMaterial
    electrolyte_type: Literal["acidic", "alkaline", "acidic_chloride"] = "alkaline"
    pH: float = 14.0
    a_Cl_molar: float = 0.0          # activates CER when > 0
    boundary_layer_m: float = 5e-5
    electrolyte_conductivity_S_m: float = 10.0
    electrolyte_resistivity_ohm_m2: float = 0.001  # Ω·m² (0.01 Ω·cm², anode half-cell)

    # ─── Derived properties ─────────────────────────────────────────

    @property
    def T(self) -> float:
        return self.material.T

    @property
    def _oer_n(self) -> int:
        return self.material.oer_n

    @property
    def cer_active(self) -> bool:
        """True when CER is thermodynamically and kinetically accessible."""
        if self.electrolyte_type == "acidic_chloride":
            return self.a_Cl_molar > 0.05
        return False

    # ─── Equilibrium potentials ──────────────────────────────────────

    def oer_equilibrium(self) -> float:
        """
        OER Nernst equilibrium potential (V vs. SHE) at the operating pH.

        In acid:  E = 1.229 − 0.0591·pH  (25 °C, p_O2 = 1 bar)
        In base: E = 0.401  (pH-independent in 4OH⁻ → O₂ form)

        The base expression is derived from the acid form by adding
        4 × (−0.0591·pH) + 0.0591·pH = −0.0591·pH per electron × 4.
        """
        T = self.T
        # 2H₂O → O₂ + 4H⁺ + 4e⁻  (works for all pH)
        return float(oer_line(self.pH, T, P_O2_STD))

    def cer_equilibrium(self) -> float:
        """
        CER Nernst equilibrium potential (V vs. SHE).

        2Cl⁻ → Cl₂ + 2e⁻

        E = E°_CER + (RT/2F)·ln(a_Cl² / p_Cl2)
          ≈ E°_CER + (RT/F)·ln(a_Cl)  (at p_Cl2 = 1 bar)
        """
        if self.a_Cl_molar <= 0.0:
            return float("nan")
        T = self.T
        RT_nF = R_GAS * T / (self.material.cer_n * FARADAY)
        return E0_CER + RT_nF * np.log(self.a_Cl_molar ** 2 / P_CL2_STD)

    # ─── Tafel branch currents ────────────────────────────────────────

    def _oer_current(self, E: float) -> float:
        """
        OER anodic current density (A/m²) at potential E (V vs. SHE).

        i = i₀ · 10^{η / b}   where η = E − E_eq(OER)
        """
        i0 = self.material.oer_i0_at_T()
        eta = E - self.oer_equilibrium()
        if eta <= 0.0:
            return 0.0
        return float(i0 * 10.0 ** (eta / self.material.oer_tafel_V))

    def _cer_current(self, E: float) -> float:
        """
        CER anodic current density (A/m²) at potential E (V vs. SHE).

        i = i₀ · 10^{η / b}   where η = E − E_eq(CER)
        """
        if not self.cer_active:
            return 0.0
        i0 = self.material.cer_i0 or 1e-3
        E_eq = self.cer_equilibrium()
        eta = E - E_eq
        if eta <= 0.0:
            return 0.0
        return float(i0 * 10.0 ** (eta / self.material.cer_tafel_V))

    def total_anodic_current(self, E: float) -> float:
        """Sum of OER and CER currents at potential E."""
        return self._oer_current(E) + self._cer_current(E)

    # ─── Overpotential decomposition ─────────────────────────────────

    def overpotential_at_current(self, j_mA_cm2: float) -> dict:
        """
        Compute anode overpotential components at a given current density.

        Parameters
        ----------
        j_mA_cm2 : float
            Total anodic current density (mA/cm²).

        Returns
        -------
        dict
            Keys: ``total_V``, ``eta_activation_V``, ``eta_concentration_V``,
            ``eta_bubble_V``, ``E_anode_V``, ``E_eq_V``, ``i_oer_A_m2``,
            ``i_cer_A_m2``, ``cer_fraction``, ``bubble_fraction``.
        """
        if j_mA_cm2 <= 0.0:
            return {
                "total_V": 0.0,
                "eta_activation_V": 0.0,
                "eta_concentration_V": 0.0,
                "eta_bubble_V": 0.0,
                "E_anode_V": self.oer_equilibrium(),
                "E_eq_V": self.oer_equilibrium(),
                "i_oer_A_m2": 0.0,
                "i_cer_A_m2": 0.0,
                "cer_fraction": 0.0,
                "bubble_fraction": 0.0,
            }

        target = j_mA_cm2 * 10.0   # A/m²

        # Find anode potential that satisfies i_total = target
        E_eq = self.oer_equilibrium()
        # Bracket: anode must be above E_eq by at least 0.05 V
        lo, hi = E_eq + 0.05, E_eq + 5.0

        def residual(E):
            return self.total_anodic_current(E) - target

        # If neither bound works, expand
        if residual(hi) > 0:
            hi = E_eq + 10.0
        if residual(lo) < 0:
            lo = E_eq + 0.01

        E_anode = float(brentq(residual, lo, hi, xtol=1e-9))

        # Component overpotentials
        eta_act = max(E_anode - E_eq, 0.0)
        eta_conc = concentration_overpotential_oer(
            j_mA_cm2,
            self.material.temperature_C,
            self.boundary_layer_m,
        )

        # Bubble resistance
        anode_key = "IrO2"
        if "NiCo" in self.material.name:
            anode_key = "NiCo"
        elif "NiFe" in self.material.name:
            anode_key = "NiFe"
        elif "Pt" in self.material.name:
            anode_key = "Pt"

        theta = bubble_fraction(j_mA_cm2, self.material.temperature_C, anode_key)
        R_mult = bubble_resistance_multiplier(theta)
        eta_bubble = self.electrolyte_resistivity_ohm_m2 * (R_mult - 1.0) * target

        # CER contribution
        i_cer = self._cer_current(E_anode)
        i_oer = self._oer_current(E_anode)
        cer_frac = i_cer / max(target, 1e-30)

        return {
            "total_V": float(eta_act + eta_conc + max(eta_bubble, 0.0)),
            "eta_activation_V": float(eta_act),
            "eta_concentration_V": float(eta_conc),
            "eta_bubble_V": float(max(eta_bubble, 0.0)),
            "E_anode_V": float(E_anode),
            "E_eq_V": float(E_eq),
            "i_oer_A_m2": float(i_oer),
            "i_cer_A_m2": float(i_cer),
            "cer_fraction": float(cer_frac),
            "bubble_fraction": float(theta),
        }

    # ─── Full anode polarization curve ──────────────────────────────

    def polarization_curve(
        self,
        j_range: Optional[Iterable[float]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return anode polarization curve.

        Returns (j_mA_cm2, E_anode_V, eta_total_V, eta_act_V, cer_fraction).

        Parameters
        ----------
        j_range : iterable, optional
            Current densities (mA/cm²).  Defaults to 1–600 mA/cm².

        Returns
        -------
        j : ndarray
            Current densities (mA/cm²).
        E : ndarray
            Anode potentials vs. SHE (V).
        eta_total : ndarray
            Total overpotential (V).
        eta_act : ndarray
            Activation overpotential (V).
        cer_frac : ndarray
            Fraction of current going to CER (0–1).
        """
        if j_range is None:
            j_range = np.linspace(1.0, 600.0, 300)
        j = np.asarray(list(j_range), dtype=float)

        E = np.empty_like(j)
        eta_total = np.empty_like(j)
        eta_act = np.empty_like(j)
        cer_frac = np.empty_like(j)

        for idx, jval in enumerate(j):
            res = self.overpotential_at_current(jval)
            E[idx] = res["E_anode_V"]
            eta_total[idx] = res["total_V"]
            eta_act[idx] = res["eta_activation_V"]
            cer_frac[idx] = res["cer_fraction"]

        return j, E, eta_total, eta_act, cer_frac

    # ─── Convenience single-value solver ─────────────────────────────

    def eta_anode(self, j_mA_cm2: float) -> float:
        """
        Total anode overpotential (V) at a given current density.

        Convenience wrapper around :meth:`overpotential_at_current`.
        """
        return self.overpotential_at_current(j_mA_cm2)["total_V"]

    def E_anode(self, j_mA_cm2: float) -> float:
        """
        Anode potential vs. SHE (V) at a given current density.
        """
        return self.overpotential_at_current(j_mA_cm2)["E_anode_V"]

    # ─── Process metrics ─────────────────────────────────────────────

    def O2_production_rate_mol_m2_hr(self, j_mA_cm2: float) -> float:
        """Molar O₂ evolution rate at the anode surface (mol/(m²·hr))."""
        res = self.overpotential_at_current(j_mA_cm2)
        # OER: n_Fe = 4, n_O2 = 1; flux_O2 = i_OER / (4F)
        return float(res["i_oer_A_m2"]) / (4.0 * FARADAY) * 3600.0

    def Cl2_production_rate_mol_m2_hr(self, j_mA_cm2: float) -> float:
        """Molar Cl₂ evolution rate at the anode surface (mol/(m²·hr))."""
        res = self.overpotential_at_current(j_mA_cm2)
        # CER: n = 2; flux_Cl2 = i_CER / (2F)
        return float(res["i_cer_A_m2"]) / (2.0 * FARADAY) * 3600.0

    # ─── Summary report ──────────────────────────────────────────────

    def summary(self, j_mA_cm2: float = 100.0) -> dict:
        """
        Full anode summary at a representative current density.

        Parameters
        ----------
        j_mA_cm2 : float
            Operating current density (mA/cm²).

        Returns
        -------
        dict
            Formatted results dictionary.
        """
        r = self.overpotential_at_current(j_mA_cm2)
        return {
            "Anode material": self.material.name,
            "Electrolyte type": self.electrolyte_type,
            "j (mA/cm²)": j_mA_cm2,
            "E_eq OER (V vs SHE)": round(r["E_eq_V"], 3),
            "E_anode (V vs SHE)": round(r["E_anode_V"], 3),
            "η_activation (V)": round(r["eta_activation_V"], 3),
            "η_concentration (V)": round(r["eta_concentration_V"], 3),
            "η_bubble (V)": round(r["eta_bubble_V"], 3),
            "η_anode total (V)": round(r["total_V"], 3),
            "i_OER (A/m²)": round(r["i_oer_A_m2"], 1),
            "i_CER (A/m²)": round(r["i_cer_A_m2"], 1),
            "CER fraction (%)": round(r["cer_fraction"] * 100, 1),
            "Bubble fraction (%)": round(r["bubble_fraction"] * 100, 1),
            "O₂ rate (mol/m²·hr)": round(self.O2_production_rate_mol_m2_hr(j_mA_cm2), 3),
            "Cl₂ rate (mol/m²·hr)" if self.cer_active else "Cl₂ rate (mol/m²·hr)":
                round(self.Cl2_production_rate_mol_m2_hr(j_mA_cm2), 3),
        }


# ─── Full-cell integration helpers ────────────────────────────────────

def anode_eta_from_lookup(
    anode: AnodeKinetics,
    j_mA_cm2: float,
) -> float:
    """
    Return anode overpotential (V) at current density j.

    This is the standard interface used by CellVoltageModel when an
    AnodeKinetics object is supplied instead of a fixed eta_anode value.
    """
    return anode.eta_anode(j_mA_cm2)


def full_cell_voltage(
    anode: AnodeKinetics,
    E_cathode_eq: float,
    E_cathode_actual: float,
    ir_drop: float,
    j_mA_cm2: float,
) -> dict:
    """
    Compute full-cell voltage components from anode and cathode models.

    V_cell = (E_anode − E_cathode) + ir_drop

    Parameters
    ----------
    anode : AnodeKinetics
        Anode model.
    E_cathode_eq : float
        Cathode equilibrium potential (V vs. SHE).
    E_cathode_actual : float
        Operating cathode potential (V vs. SHE).
    ir_drop : float
        Ohmic drop across electrolyte + membrane (V).
    j_mA_cm2 : float
        Operating current density (mA/cm²).

    Returns
    -------
    dict
        Voltage components: ``V_cell``, ``E_thermo``, ``eta_anode``, ``eta_cathode``,
        ``ir_drop``, ``E_anode``, ``E_cathode``.
    """
    anode_result = anode.overpotential_at_current(j_mA_cm2)
    eta_a = anode_result["total_V"]
    E_a = anode_result["E_anode_V"]

    eta_c = max(E_cathode_eq - E_cathode_actual, 0.0)
    E_thermo = abs(E_a - E_cathode_eq)

    return {
        "V_cell": float(E_a - E_cathode_actual + ir_drop),
        "E_thermo": float(E_thermo),
        "eta_anode": float(eta_a),
        "eta_cathode": float(eta_c),
        "ir_drop": float(ir_drop),
        "E_anode": float(E_a),
        "E_cathode": float(E_cathode_actual),
    }
