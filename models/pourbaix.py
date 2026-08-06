"""
Fe-H2O Pourbaix (potential-pH) speciation for aqueous iron electrowinning.

Implements a simplified but thermodynamically consistent Fe-H2O diagram at
298.15 K (extensible to elevated temperature via the Nernst term), plus the
hydrogen and oxygen evolution lines that bound the water stability window.

Species considered
------------------
    Fe(s), Fe2+(aq), Fe3+(aq), Fe(OH)2(s), Fe(OH)3(s), HFeO2-(aq)

Conventions
-----------
    * All potentials are V vs. SHE.
    * Activity of dissolved species is set by `activity` (default 1e-6 M,
      the classical Pourbaix convention for "corrosion" boundaries).
      For electrowinning, a process-relevant activity (0.1-2 M) is more
      informative; pass it explicitly.

References
----------
    Pourbaix, M. "Atlas of Electrochemical Equilibria in Aqueous Solutions" (1974).
    Beverskog & Puigdomenech, Corros. Sci. 38 (1996) 2121 - revised Fe-H2O diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .electrochemistry import FARADAY, R_GAS

# ─── Standard potentials / equilibrium constants at 298.15 K ──────────
E0_FE2_FE = -0.440      # Fe2+ + 2e- -> Fe(s)                 V vs. SHE
E0_FE3_FE2 = 0.771      # Fe3+ + e-  -> Fe2+                  V vs. SHE
E0_FEOH3_FE2 = 1.057    # Fe(OH)3 + 3H+ + e- -> Fe2+ + 3H2O   V vs. SHE
E0_FEOH2_FE = -0.047    # Fe(OH)2 + 2H+ + 2e- -> Fe + 2H2O    V vs. SHE
E0_HFEO2_FE = -0.909    # HFeO2- + 3H+ + 2e- -> Fe + 2H2O     V vs. SHE
E0_FEOH3_FEOH2 = 0.271  # Fe(OH)3 + H+ + e- -> Fe(OH)2 + H2O  V vs. SHE

# Solubility products and hydrolysis constants (log10, 25 °C)
LOGKSP_FEOH2 = -16.31    # Fe(OH)2 <-> Fe2+ + 2OH-   (Ksp = 4.87e-17)
LOGKSP_FEOH3 = -38.55    # Fe(OH)3 <-> Fe3+ + 3OH-   (Ksp = 2.79e-39)
LOGK_FEOH2_HFEO2 = -18.30  # Fe(OH)2 <-> HFeO2- + H+
KW = 14.0                # -log10 Kw at 298 K

T_REF = 298.15

# Fe(OH)₂ solubility has a strong temperature dependence: it becomes *more*
# soluble as T rises (the dissolution enthalpy is positive, endothermic).
# Screening central value ΔH_sol ≈ +22 kJ/mol for
# Fe(OH)₂(s) ⇌ Fe²⁺ + 2 OH⁻, anchored at the 25 °C pKsp = 16.31 and chosen
# to reproduce the ~5× Ksp rise between 25 and 60 °C reported in the
# FeSO₄/Fe(OH)₂ solubility literature (Ball & Nordstrom; Kobylin et al.).
# This is a van 't Hoff screening value, not a fitted thermodynamic model.
DH_SOL_FEOH2_J_MOL = 22.0e3


def logksp_feoh2(T: float = T_REF, dH_J_mol: float = DH_SOL_FEOH2_J_MOL) -> float:
    """log₁₀ Ksp of Fe(OH)₂ at temperature T (K), van 't Hoff corrected.

    Fe(OH)₂ is more soluble at elevated temperature (endothermic
    dissolution), so Ksp rises and the Fe²⁺/Fe(OH)₂ precipitation boundary
    shifts to higher pH as the bath warms.  The previous code used the
    25 °C Ksp unchanged at 60–90 °C, biasing the precipitation criterion.
    """
    return LOGKSP_FEOH2 - (dH_J_mol / (2.303 * R_GAS)) * (1.0 / T - 1.0 / T_REF)


def ksp_feoh2(T: float = T_REF, dH_J_mol: float = DH_SOL_FEOH2_J_MOL) -> float:
    """Ksp of Fe(OH)₂ (mol/L)³ at temperature T (K)."""
    return 10.0 ** logksp_feoh2(T, dH_J_mol)


def _slope(T: float, n_h: int, n_e: int) -> float:
    """Nernst pH-slope (V per pH unit) for a reaction with n_h protons, n_e electrons."""
    return -(np.log(10.0) * R_GAS * T / FARADAY) * (n_h / n_e)


def nernst_pH_line(E0: float, pH, T: float = T_REF, n_h: int = 0, n_e: int = 1,
                   log_activity: float = 0.0, n_species: int = 1):
    """
    Potential of a redox couple as a function of pH.

    E = E0 + slope * pH + (2.303 RT / n_e F) * n_species * log10(a)

    `n_species` is the stoichiometric coefficient of the dissolved species,
    POSITIVE if it appears on the oxidised (left) side of the reduction
    reaction and NEGATIVE if it is a reduction product.
    """
    pH = np.asarray(pH, dtype=float)
    prefactor = np.log(10.0) * R_GAS * T / FARADAY
    return (
        E0
        + _slope(T, n_h, n_e) * pH
        + prefactor * (n_species / n_e) * log_activity
    )


def her_line(pH, T: float = T_REF, p_H2: float = 1.0):
    """Hydrogen evolution equilibrium line: 2H+ + 2e- -> H2."""
    pH = np.asarray(pH, dtype=float)
    prefactor = np.log(10.0) * R_GAS * T / FARADAY
    return -prefactor * pH - (prefactor / 2.0) * np.log10(p_H2)


def oer_line(pH, T: float = T_REF, p_O2: float = 1.0):
    """Oxygen evolution equilibrium line: O2 + 4H+ + 4e- -> 2H2O."""
    pH = np.asarray(pH, dtype=float)
    prefactor = np.log(10.0) * R_GAS * T / FARADAY
    return 1.229 - prefactor * np.asarray(pH, dtype=float) + (prefactor / 4.0) * np.log10(p_O2)


@dataclass
class FePourbaix:
    """
    Fe-H2O Pourbaix diagram evaluator.

    Parameters
    ----------
    activity : float
        Activity (approx. molarity) of dissolved Fe species. Default 1e-6
        (Pourbaix convention); use 0.1-2.0 for electrowinning baths.
    temperature_C : float
        Temperature in degrees Celsius.
    """

    activity: float = 1e-6
    temperature_C: float = 25.0

    @property
    def T(self) -> float:
        return self.temperature_C + 273.15

    @property
    def log_a(self) -> float:
        return float(np.log10(self.activity))

    # ─── Individual boundaries ────────────────────────────────────────
    def E_Fe2_Fe(self, pH=0.0):
        """Fe2+/Fe: pH-independent, activity-dependent."""
        return nernst_pH_line(E0_FE2_FE, pH, self.T, n_h=0, n_e=2,
                              log_activity=self.log_a, n_species=1)

    def E_Fe3_Fe2(self, pH=0.0):
        """Fe3+/Fe2+: pH-independent; equal activities assumed."""
        return nernst_pH_line(E0_FE3_FE2, pH, self.T, n_h=0, n_e=1,
                              log_activity=0.0)

    def E_FeOH2_Fe(self, pH):
        """Fe(OH)2/Fe: 2 protons, 2 electrons -> -59 mV/pH."""
        return nernst_pH_line(E0_FEOH2_FE, pH, self.T, n_h=2, n_e=2)

    def E_FeOH3_Fe2(self, pH):
        """Fe(OH)3/Fe2+: 3 protons, 1 electron -> -177 mV/pH."""
        return nernst_pH_line(E0_FEOH3_FE2, pH, self.T, n_h=3, n_e=1,
                              log_activity=self.log_a, n_species=-1)

    def E_FeOH3_FeOH2(self, pH):
        """Fe(OH)3/Fe(OH)2: 1 proton, 1 electron -> -59 mV/pH."""
        return nernst_pH_line(E0_FEOH3_FEOH2, pH, self.T, n_h=1, n_e=1)

    def E_HFeO2_Fe(self, pH):
        """HFeO2-/Fe (alkaline ferrite): 3 protons, 2 electrons."""
        return nernst_pH_line(E0_HFEO2_FE, pH, self.T, n_h=3, n_e=2,
                              log_activity=self.log_a, n_species=1)

    # ─── Vertical (pH-only) boundaries ────────────────────────────────
    @property
    def pH_Fe2_FeOH2(self) -> float:
        """pH at which Fe2+ hydrolyses to Fe(OH)2 at the given activity."""
        return KW + 0.5 * (LOGKSP_FEOH2 - self.log_a)

    @property
    def pH_Fe3_FeOH3(self) -> float:
        """pH at which Fe3+ hydrolyses to Fe(OH)3."""
        return KW + (LOGKSP_FEOH3 - self.log_a) / 3.0

    @property
    def pH_FeOH2_HFeO2(self) -> float:
        """pH above which Fe(OH)2 redissolves as ferrite HFeO2-."""
        return -LOGK_FEOH2_HFEO2 + self.log_a

    # ─── Practical electrowinning diagnostics ─────────────────────────
    def deposition_potential(self, pH: float) -> float:
        """
        Potential at which metallic Fe becomes stable at this pH, i.e. the
        lower boundary of the immunity domain (max of the relevant couples).
        """
        if pH < self.pH_Fe2_FeOH2:
            return float(self.E_Fe2_Fe(pH))
        if pH < self.pH_FeOH2_HFeO2:
            return float(self.E_FeOH2_Fe(pH))
        return float(self.E_HFeO2_Fe(pH))

    def her_margin(self, pH: float) -> float:
        """
        Thermodynamic penalty for HER competition (V).

        Positive value = how far negative of the reversible hydrogen potential
        one must polarise to deposit iron. Smaller is better.
        """
        return float(her_line(pH, self.T) - self.deposition_potential(pH))

    def dominant_species(self, pH: float, E: float) -> str:
        """Return the dominant Fe species at a given (pH, potential) point."""
        if E < self.deposition_potential(pH):
            return "Fe(s)"
        if pH < self.pH_Fe2_FeOH2:
            if E > self.E_FeOH3_Fe2(pH) and pH > self.pH_Fe3_FeOH3:
                return "Fe(OH)3(s)"
            if E > self.E_Fe3_Fe2(pH):
                return "Fe3+"
            return "Fe2+"
        if pH < self.pH_FeOH2_HFeO2:
            if E > self.E_FeOH3_FeOH2(pH):
                return "Fe(OH)3(s)"
            return "Fe(OH)2(s)"
        return "HFeO2-"

    def water_window(self, pH: float) -> Tuple[float, float]:
        """(HER line, OER line) potentials bounding water stability at this pH."""
        return float(her_line(pH, self.T)), float(oer_line(pH, self.T))

    def summary(self, pH_points=(0.0, 2.0, 7.0, 10.0, 14.0)) -> Dict[float, dict]:
        """Tabulate deposition potential and HER margin across pH."""
        out = {}
        for pH in pH_points:
            out[pH] = {
                "E_deposition (V vs SHE)": round(self.deposition_potential(pH), 3),
                "E_HER (V vs SHE)": round(float(her_line(pH, self.T)), 3),
                "HER margin (V)": round(self.her_margin(pH), 3),
                "stable Fe species above E_dep": self.dominant_species(
                    pH, self.deposition_potential(pH) + 0.05
                ),
            }
        return out
