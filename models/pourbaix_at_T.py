"""
Fe-H2O Pourbaix boundaries recomputed at operating temperature.

`models/pourbaix.py` (the 25 °C module) evaluates the Nernst *slope* at T for
every line but keeps each standard potential E0 pinned to its 25 °C value.
At the operating temperature (60 °C, allowed up to 90 °C) that is the large
omitted temperature effect: the E0 intercepts of Fe2+/Fe, Fe(OH)2/Fe,
Fe3+/Fe2+, Fe(OH)3/Fe2+, HFeO2-/Fe and the OER all *drift* with T, and they
drift in *different* directions from the steepening H2 line.  The gap between
Fe2+ deposition and hydrogen evolution is the program's central lever, so this
additive module recomputes every boundary line with the temperature-dependent
standard Gibbs energies already implicit in the van't Hoff / Ellingham
framework.  The 25 °C module is left untouched.

Framework
---------
For each reduction reaction the temperature enters through the standard
reaction entropy in the constant-ΔH, ΔS (Ellingham / van't Hoff) form of the
reaction Gibbs energy::

    ΔG_rxn(T) = ΔH_rxn,ref − T·ΔS_rxn
    E0(T)      = −ΔG_rxn(T) / (n_e F)
               = E0(ref) + (ΔS_rxn / (n_e F))·(T − T_ref)

so E0(T) is linear in T with slope dE0/dT = ΔS_rxn/(n_e·F), anchored *exactly*
on the repo's 25 °C E0 values (recovered at T = 298.15 K, no regression).  The
Nernst pH slope then steepens with T as in the legacy module::

    E(pH, T) = E0(T) − (2.303·R·T/F)·(n_H/n_e)·pH
                   + (2.303·R·T/F)·(n_species/n_e)·log10(a)

Reference-electrode convention
------------------------------
Potentials are V vs. SHE with the standard convention S°(H+) = S°(e−) = 0, and
the SHE is *0 V at every temperature*.  The H+/H2 line therefore keeps
E(pH=0, T) = 0 and only steepens with T (the same behaviour already pinned in
the legacy `her_line`).  Because every Fe couple and the H2 line are referenced
to the same SHE, the *gap* between Fe deposition and HER is convention-free and
is the physically meaningful quantity this module makes visible.

Standard molar entropies S°(298.15 K), J mol⁻¹ K⁻¹ (screening anchors)
-----------------------------------------------------------------------
    Fe(s) 27.28 | Fe2+(aq) −137.7 | Fe3+(aq) −315.9 | Fe(OH)2(s) 88.0
    Fe(OH)3(s) 106.7 | HFeO2−(aq) −25* | H2O(l) 69.95 | O2(g) 205.15
    H+(aq) 0 | e−(aq) 0

*The HFeO2−(aq) entropy is poorly tabulated; −25 J mol⁻¹ K⁻¹ is a screening
mid-range estimate.  It only enters the far-alkaline HFeO2−/Fe line, not the
central Fe2+/Fe or Fe(OH)2/Fe levers, and is flagged here rather than hidden.

References
----------
    Bard, A. J., Parsons, R., Jordan, J. "Standard Potentials in Aqueous
        Solution" (1985).
    Pourbaix, M. "Atlas of Electrochemical Equilibria in Aqueous Solutions"
        (1974).
    Beverskog, B., Puigdomenech, I. Corros. Sci. 38 (1996) 2121 (revised
        Fe–H2O diagram at elevated temperature).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .electrochemistry import FARADAY, R_GAS
from .thermodynamic_constants import E0_OER_V, kw_water, T_REF_K
from .pourbaix import (
    E0_FE2_FE,
    E0_FE3_FE2,
    E0_FEOH2_FE,
    E0_FEOH3_FE2,
    E0_FEOH3_FEOH2,
    E0_HFEO2_FE,
    her_line,
    logksp_feoh2,
    LOGKSP_FEOH3,
    LOGK_FEOH2_HFEO2,
)

# ─── Standard molar entropies, J mol-1 K-1 ────────────────────────────────
S_FE_S = 27.28
S_FE2_AQ = -137.7
S_FE3_AQ = -315.9
S_FEOH2_S = 88.0
S_FEOH3_S = 106.7
S_HFEO2_AQ = -25.0          # screening estimate; see module docstring
S_H2O_L = 69.95
S_O2_G = 205.15

# Reaction entropies (J mol-1 K-1) for the reduction half-reactions as written.
DS_FE2_FE = S_FE_S - S_FE2_AQ                                   # +164.98
DS_FE3_FE2 = S_FE2_AQ - S_FE3_AQ                                 # +178.20
DS_FEOH2_FE = S_FE_S + 2.0 * S_H2O_L - S_FEOH2_S                 # +79.18
DS_FEOH3_FE2 = S_FE2_AQ + 3.0 * S_H2O_L - S_FEOH3_S              # -34.55
DS_FEOH3_FEOH2 = S_FEOH2_S + S_H2O_L - S_FEOH3_S                 # +51.25
DS_HFEO2_FE = S_FE_S + 2.0 * S_H2O_L - S_HFEO2_AQ                # +192.18
DS_OER = 2.0 * S_H2O_L - S_O2_G                                  # -65.25

# Screening dissolution enthalpies for the vertical (pH-only) boundaries,
# used only for the van't Hoff vertical-boundary shifts.  Endothermic
# (positive) like the Fe(OH)2 case already in the repo.
DH_FEOH3_DISSOLUTION_J_MOL = 55.0e3   # Fe(OH)3(s) -> Fe3+ + 3OH-
DH_FEOH2_HFEO2_J_MOL = 35.0e3         # Fe(OH)2(s) -> HFeO2- + H+


def e0_at_T(E0_ref: float, dS_rxn: float, n_e: int, T: float = T_REF_K) -> float:
    """Standard potential of a couple at temperature T (K), vs. SHE.

    Linear (constant-ΔH, ΔS) temperature correction anchored on ``E0_ref``.
    """
    return float(E0_ref + (dS_rxn / (n_e * FARADAY)) * (T - T_REF_K))


def nernst_pH_line_at_T(E0: float, pH, T: float = T_REF_K, n_h: int = 0,
                        n_e: int = 1, log_activity: float = 0.0,
                        n_species: int = 1):
    """Boundary line potential at ``pH`` and temperature ``T`` (same form as
    ``pourbaix.nernst_pH_line`` but with the T-corrected intercept ``E0``)."""
    pH = np.asarray(pH, dtype=float)
    prefactor = np.log(10.0) * R_GAS * T / FARADAY
    return (
        E0
        - prefactor * (n_h / n_e) * pH
        + prefactor * (n_species / n_e) * log_activity
    )


@dataclass
class PourbaixAtT:
    """Fe-H2O Pourbaix at operating temperature.

    Mirrors the API of ``FePourbaix`` (so it can be swapped in where the 25 °C
    object is used) but recomputes every standard potential at ``T``.

    Parameters
    ----------
    activity : float
        Activity (approx. molarity) of dissolved Fe species.  Default 1e-6
        (classical Pourbaix convention); use 0.1-2.0 for electrowinning baths.
    temperature_C : float
        Operating temperature in °C (default 60, the program's operating T;
        allowed up to 90).
    """

    activity: float = 1e-6
    temperature_C: float = 60.0

    @property
    def T(self) -> float:
        return self.temperature_C + 273.15

    @property
    def log_a(self) -> float:
        return float(np.log10(self.activity))

    # ─── Temperature-corrected standard potentials ─────────────────────
    def E0_Fe2_Fe(self) -> float:
        return e0_at_T(E0_FE2_FE, DS_FE2_FE, 2, self.T)

    def E0_Fe3_Fe2(self) -> float:
        return e0_at_T(E0_FE3_FE2, DS_FE3_FE2, 1, self.T)

    def E0_FeOH2_Fe(self) -> float:
        return e0_at_T(E0_FEOH2_FE, DS_FEOH2_FE, 2, self.T)

    def E0_FeOH3_Fe2(self) -> float:
        return e0_at_T(E0_FEOH3_FE2, DS_FEOH3_FE2, 1, self.T)

    def E0_FeOH3_FeOH2(self) -> float:
        return e0_at_T(E0_FEOH3_FEOH2, DS_FEOH3_FEOH2, 1, self.T)

    def E0_HFeO2_Fe(self) -> float:
        return e0_at_T(E0_HFEO2_FE, DS_HFEO2_FE, 2, self.T)

    def E0_OER_std(self) -> float:
        return e0_at_T(E0_OER_V, DS_OER, 4, self.T)

    # ─── Boundary lines (V vs SHE) ─────────────────────────────────────
    def E_Fe2_Fe(self, pH=0.0):
        """Fe2+/Fe: pH-independent, activity-dependent; shifts *up* with T."""
        return nernst_pH_line_at_T(
            self.E0_Fe2_Fe(), pH, self.T, n_h=0, n_e=2,
            log_activity=self.log_a, n_species=1)

    def E_Fe3_Fe2(self, pH=0.0):
        """Fe3+/Fe2+: pH-independent; shifts *up* with T (strongly)."""
        return nernst_pH_line_at_T(
            self.E0_Fe3_Fe2(), pH, self.T, n_h=0, n_e=1, log_activity=0.0)

    def E_FeOH2_Fe(self, pH):
        """Fe(OH)2/Fe: 2H+/2e- -> -59 mV/pH at 25 C; shifts *up* with T."""
        return nernst_pH_line_at_T(
            self.E0_FeOH2_Fe(), pH, self.T, n_h=2, n_e=2)

    def E_FeOH3_Fe2(self, pH):
        """Fe(OH)3/Fe2+: 3H+/1e-; shifts *down* with T (only negative slope)."""
        return nernst_pH_line_at_T(
            self.E0_FeOH3_Fe2(), pH, self.T, n_h=3, n_e=1,
            log_activity=self.log_a, n_species=-1)

    def E_FeOH3_FeOH2(self, pH):
        """Fe(OH)3/Fe(OH)2: 1H+/1e-; shifts *up* with T."""
        return nernst_pH_line_at_T(
            self.E0_FeOH3_FeOH2(), pH, self.T, n_h=1, n_e=1)

    def E_HFeO2_Fe(self, pH):
        """HFeO2-/Fe (alkaline ferrite): 3H+/2e-; shifts *up* with T."""
        return nernst_pH_line_at_T(
            self.E0_HFeO2_Fe(), pH, self.T, n_h=3, n_e=2,
            log_activity=self.log_a, n_species=1)

    def E_HER(self, pH):
        """Hydrogen line at operating T (SHE = 0 V at every T; slope steepens)."""
        return her_line(pH, self.T)

    def E_OER(self, pH, p_O2: float = 1.0):
        """Oxygen line at operating T (E0 drifts down, slope steepens)."""
        pH = np.asarray(pH, dtype=float)
        prefactor = np.log(10.0) * R_GAS * self.T / FARADAY
        return (self.E0_OER_std()
                - prefactor * pH
                + (prefactor / 4.0) * np.log10(p_O2))

    # ─── Vertical (pH-only) boundaries, T-corrected ────────────────────
    def _pKw(self) -> float:
        return -float(np.log10(kw_water(self.T)))

    @property
    def pH_Fe2_FeOH2(self) -> float:
        """pH at which Fe2+ hydrolyses to Fe(OH)2 at this T (shifts left)."""
        return self._pKw() + 0.5 * (logksp_feoh2(self.T) - self.log_a)

    @property
    def pH_Fe3_FeOH3(self) -> float:
        """pH at which Fe3+ hydrolyses to Fe(OH)3 at this T (screening dH)."""
        logksp_3 = LOGKSP_FEOH3 - (DH_FEOH3_DISSOLUTION_J_MOL / (2.303 * R_GAS)) \
            * (1.0 / self.T - 1.0 / T_REF_K)
        return self._pKw() + (logksp_3 - self.log_a) / 3.0

    @property
    def pH_FeOH2_HFeO2(self) -> float:
        """pH above which Fe(OH)2 redissolves as HFeO2- at this T (screening)."""
        logk = LOGK_FEOH2_HFEO2 - (DH_FEOH2_HFEO2_J_MOL / (2.303 * R_GAS)) \
            * (1.0 / self.T - 1.0 / T_REF_K)
        return -logk + self.log_a

    # ─── Practical electrowinning diagnostics ──────────────────────────
    def deposition_potential(self, pH: float) -> float:
        """Lower boundary of the Fe immunity domain at this pH and T."""
        if pH < self.pH_Fe2_FeOH2:
            return float(self.E_Fe2_Fe(pH))
        if pH < self.pH_FeOH2_HFeO2:
            return float(self.E_FeOH2_Fe(pH))
        return float(self.E_HFeO2_Fe(pH))

    def her_margin(self, pH: float) -> float:
        """Thermodynamic penalty for HER competition (V) at operating T.

        Positive = how far negative of the reversible H2 potential one must
        polarise to deposit iron.  This is the program's central lever; it
        *narrows* as T rises (Fe lines shift up, H2 line steepens down).
        """
        return float(self.E_HER(pH) - self.deposition_potential(pH))

    def summary(self, pH_points=(0.0, 2.0, 7.0, 10.0, 14.0)) -> Dict[float, dict]:
        out = {}
        for pH in pH_points:
            out[pH] = {
                "E_deposition (V vs SHE)": round(self.deposition_potential(pH), 3),
                "E_HER (V vs SHE)": round(float(self.E_HER(pH)), 3),
                "HER margin (V)": round(self.her_margin(pH), 3),
            }
        return out


def boundary_sweep(T_C: float = 60.0, activity: float = 1.0, pH=(0.0, 7.0)):
    """Tabulate the five boundary lines at a temperature for a quick check."""
    p = PourbaixAtT(activity=activity, temperature_C=T_C)
    ph = float(pH[0])
    return {
        "temperature_C": T_C,
        "E0_Fe2_Fe": round(p.E0_Fe2_Fe(), 3),
        "E0_Fe3_Fe2": round(p.E0_Fe3_Fe2(), 3),
        "E0_FeOH2_Fe": round(p.E0_FeOH2_Fe(), 3),
        "E0_FeOH3_Fe2": round(p.E0_FeOH3_Fe2(), 3),
        "E0_HFeO2_Fe": round(p.E0_HFeO2_Fe(), 3),
        f"E_HER@pH{ph}": round(float(p.E_HER(ph)), 3),
        f"E_OER@pH{ph}": round(float(p.E_OER(ph)), 3),
        f"HER_margin@pH{ph}": round(p.her_margin(ph), 3),
        "pH_Fe2_FeOH2": round(p.pH_Fe2_FeOH2, 2),
    }


if __name__ == "__main__":
    print("Fe-H2O Pourbaix at operating temperature (a = 1 M), vs 25 C:")
    print(f"{'line':<16}{'25 C':>10}{'60 C':>10}{'90 C':>10}")
    keys = ["E0_Fe2_Fe", "E0_Fe3_Fe2", "E0_FeOH2_Fe", "E0_FeOH3_Fe2",
            "E0_HFeO2_Fe", "E_HER@pH2.0", "E_OER@pH2.0", "HER_margin@pH2.0",
            "pH_Fe2_FeOH2"]
    rows = {k: [] for k in keys}
    for Tc in (25.0, 60.0, 90.0):
        sweep = boundary_sweep(Tc, pH=(2.0,))
        for k in keys:
            rows[k].append(sweep[k])
    for k in keys:
        print(f"{k:<16}{rows[k][0]:>10}{rows[k][1]:>10}{rows[k][2]:>10}")
