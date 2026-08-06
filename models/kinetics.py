"""
Butler-Volmer kinetics for competing Fe deposition and hydrogen evolution.

The central quantity for aqueous iron electrowinning is the *current
efficiency* - the fraction of the applied cathodic current that goes into
Fe2+ + 2e- -> Fe rather than the parasitic hydrogen evolution reaction (HER).
This module computes that partition from first principles using Tafel /
Butler-Volmer kinetics with an optional mass-transport limit on the iron
branch.

Sign convention
---------------
    Cathodic current densities are reported as POSITIVE magnitudes
    (A/m^2 unless stated otherwise).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import brentq

from .electrochemistry import FARADAY, M_FE, R_GAS, Z_FE
from .pourbaix import her_line

T_REF = 298.15

# ─── Arrhenius temperature corrections for kinetics ──────────────────
#
# Exchange current densities in this repository are anchored at the bath
# reference temperature I0_REF_K = 323.15 K (50 °C) — i.e. the literature
# i0 values configured in this module represent the kinetics *at the
# electrolyte reference condition*, not at 25 °C.
#
# Apparent activation energies (screening values, ±50 %):
#   Fe²⁺ + 2e⁻ → Fe on Fe, sulfate media: Ea ≈ 50 kJ/mol
#     (metal-deposition apparent Ea's are typically 40–60 kJ/mol;
#      e.g. Axt & Grant-style ferrous sulfate polarization data)
#   HER on Fe, mildly acidic media: Ea ≈ 60 kJ/mol
#     (HER apparent Ea's on iron-group metals in acid are typically
#      50–90 kJ/mol)
# Consequence: the HER branch (Ea ≈ 60) is more temperature-activated
# than Fe deposition (Ea ≈ 50), so at galvanostatic, HER-active
# conditions FE falls modestly with temperature — the CE-peaks-at-
# moderate-T behaviour long known for ferrous/zinc electrowinning — while
# transport-limited cases can still improve via D(T).
EA_FE_DEPOSITION_J_MOL = 50.0e3   # J/mol
EA_HER_ON_FE_J_MOL = 60.0e3       # J/mol
I0_REF_K = 323.15                 # K (50 °C)
EA_DIFFUSION_J_MOL = 18.0e3       # J/mol (aqueous diffusivities)
D_REF_K = 298.15                  # K — diffusivity literature anchor (25 °C)

# ─── Butler–Volmer anodic-branch slopes ──────────────────────────────
#
# The Tafel form i = i0·10^((E_eq−E)/b_c) has two classic artefacts the
# full Butler–Volmer form removes: it gives i(E_eq) = i0 ≠ 0 (no
# thermodynamic equilibrium) and it has no dissolution branch anodic of
# E_eq.  The anodic-branch slope is fixed once at 25 °C from the charge-
# transfer bookkeeping α_a·n = n − α_c·n, with α_c·n read off the 25 °C
# cathodic slope via the RT/F relation:
#     2.303 R T / F  =  b(25 °C)·(α·n)
#   Fe  (n=2):  b_c=0.120 → α_c·n = b25/0.120 → α_a·n = 2 − α_c·n → b_a
#   HER (n=1):  b_c=0.140 → α_c   = b25/0.140 → α_a   = 1 − α_c   → b_a
_FE_B25 = 2.303 * R_GAS * 298.15 / FARADAY
FE_ANODIC_SLOPE_V = _FE_B25 / (2.0 - _FE_B25 / 0.120)
HER_ANODIC_SLOPE_V = _FE_B25 / (1.0 - _FE_B25 / 0.140)



def arrhenius_i0(i0_ref: float, T_K: float, Ea_J_mol: float,
                 T_ref_K: float = I0_REF_K) -> float:
    """Exchange current density at temperature T_K.

    i0(T) = i0(T_ref) · exp[ Ea/R · (1/T_ref − 1/T) ]
    """
    return float(i0_ref * math.exp((Ea_J_mol / R_GAS) * (1.0 / T_ref_K - 1.0 / T_K)))


def arrhenius_diffusivity(D_ref: float, T_K: float,
                          Ea_J_mol: float = EA_DIFFUSION_J_MOL,
                          T_ref_K: float = D_REF_K) -> float:
    """Diffusivity at temperature T_K (Arrhenius; same convention as
    ``diffusion_layer_1d._diffusivity_T``)."""
    return float(D_ref * math.exp((Ea_J_mol / R_GAS) * (1.0 / T_ref_K - 1.0 / T_K)))


def limiting_current_density(
    concentration_mol_m3: float,
    diffusivity_m2_s: float = 7.2e-10,
    boundary_layer_m: float = 5e-5,
    z: int = Z_FE,
) -> float:
    """
    Levich-type diffusion-limited current density (A/m^2).

    i_lim = z F D C / delta
    """
    return z * FARADAY * diffusivity_m2_s * concentration_mol_m3 / boundary_layer_m


@dataclass
class TafelBranch:
    """
    A single cathodic Tafel branch.

    Parameters
    ----------
    i0 : float
        Exchange current density (A/m^2).
    tafel_slope_V : float
        Cathodic Tafel slope (V/decade), positive.
    E_eq : float
        Equilibrium potential of the couple (V vs. SHE).
    i_lim : float or None
        Diffusion-limiting current density (A/m^2). None = kinetics only.
    """

    i0: float
    tafel_slope_V: float
    E_eq: float
    i_lim: Optional[float] = None

    def current(self, E):
        """Cathodic current density magnitude (A/m^2) at electrode potential E."""
        E = np.asarray(E, dtype=float)
        eta = self.E_eq - E  # cathodic overpotential, positive when E < E_eq
        i_kin = self.i0 * 10.0 ** (np.clip(eta, 0.0, None) / self.tafel_slope_V)
        i_kin = np.where(eta > 0.0, i_kin, self.i0 * 10.0 ** (eta / self.tafel_slope_V))
        if self.i_lim is None:
            return i_kin
        # Koutecky-Levich mixed control
        return 1.0 / (1.0 / np.maximum(i_kin, 1e-30) + 1.0 / self.i_lim)


@dataclass
class ButlerVolmerBranch:
    """
    A full Butler–Volmer branch (cathodic positive, signed).

    Extends ``TafelBranch`` with the anodic (reverse) exponential so that
    the current is thermodynamically consistent at equilibrium —
    ``current(E_eq) == 0`` exactly — and so that the branch is *signed*:
    anodic of ``E_eq`` the net current is negative (dissolution for Fe,
    H₂ ionisation for HER), which the Tafel form cannot represent.  At
    operating cathodic overpotentials (|η| ≳ 150 mV) the reverse term is
    ≲1e-4 of the forward one, so results differ from ``TafelBranch`` only
    near equilibrium.

    Parameters
    ----------
    i0 : float
        Exchange current density (A/m^2).
    tafel_slope_V : float
        Cathodic Tafel slope (V/decade), positive.
    E_eq : float
        Equilibrium potential of the couple (V vs. SHE).
    i_lim : float or None
        Diffusion-limiting current density (A/m^2); blends the cathodic
        arm only (dissolution is never transport-capped here).
    anodic_slope_V : float or None
        Anodic Tafel slope (V/decade), positive. ``None`` disables the
        reverse term (fallback to the cathodic slope value) for callers
        that only need the signed form near equilibrium.
    """

    i0: float
    tafel_slope_V: float
    E_eq: float
    i_lim: Optional[float] = None
    anodic_slope_V: Optional[float] = None

    def current(self, E):
        """Signed current density (A/m^2) at electrode potential E.

        Cathodic positive; negative anodic of ``E_eq`` (net oxidation).
        """
        E = np.asarray(E, dtype=float)
        eta = self.E_eq - E  # cathodic overpotential, positive when E < E_eq
        b_a = self.anodic_slope_V if self.anodic_slope_V is not None else self.tafel_slope_V
        i_kin = self.i0 * (10.0 ** (eta / self.tafel_slope_V)
                           - 10.0 ** (-eta / b_a))
        if self.i_lim is None:
            return i_kin
        # Koutecky-Levich blend caps only the cathodic arm; the anodic
        # (dissolution) arm is returned unchanged.
        i_cat = np.where(i_kin > 0.0, i_kin, 0.0)
        blended = 1.0 / (1.0 / np.maximum(i_cat, 1e-30) + 1.0 / self.i_lim)
        return np.where(i_kin > 0.0, blended, i_kin)

    def current_scaled(self, E, forward_scale: float = 1.0):
        """BV current with the CATHODIC (forward) arm multiplied by
        ``forward_scale`` (2026-08; the anodic arm is never scaled).

        First-order surface-activity correction for transport-resolved
        solvers: as the surface reactant concentration falls, the reduction
        rate falls with it (scale ≈ a_surf/a_ref), while the oxidation arm —
        whose reactant is the solid surface itself — is unchanged.  This
        keeps a resolved-film solver's wall flux mass-consistent through
        deep depletion without re-introducing a transport cap.  Any
        ``i_lim`` blend is bypassed (returns the uncapped form).
        """
        E = np.asarray(E, dtype=float)
        eta = self.E_eq - E
        b_a = self.anodic_slope_V if self.anodic_slope_V is not None else self.tafel_slope_V
        return (forward_scale * self.i0 * 10.0 ** (eta / self.tafel_slope_V)
                - self.i0 * 10.0 ** (-eta / b_a))

    def current_magnitude(self, E):
        """Cathodic magnitude (A/m^2); 0 anodic of E_eq."""
        return np.where(self.current(E) > 0.0, self.current(E), 0.0)


def surface_bv_branches(
    pH: float,
    temperature_C: float,
    fe_i0: float,
    her_i0: float,
    fe_tafel_V: float = 0.120,
    her_tafel_V: float = 0.140,
    fe_E_eq: float = -0.440,
    fe_i0_Ea_J_mol: float = EA_FE_DEPOSITION_J_MOL,
    her_i0_Ea_J_mol: float = EA_HER_ON_FE_J_MOL,
    kinetics_ref_K: float = I0_REF_K,
) -> "tuple[ButlerVolmerBranch, ButlerVolmerBranch]":
    """Signed Butler–Volmer Fe/HER pair with NO transport cap (2026-08).

    For transport-resolved solvers (``pulse.py``'s Crank–Nicolson film, or
    any code that owns its concentration profile): the branches are pure
    surface-kinetics rate laws evaluated at the caller's surface state and
    temperature.  Transport capping is the caller's job — blending a
    Koutecký–Levich cap on top of a resolved film would count depletion
    twice.  ``DepositionKinetics`` remains the right entry point for
    bulk-driven (mixed-control) calculations.

    i0 values follow the module convention (anchored at ``kinetics_ref_K``,
    Arrhenius-scaled to ``temperature_C``); the Fe equilibrium potential is
    the canonical fixed screening value (surface-activity Nernst corrections
    are a documented limitation of the whole stack, not of this function).

    Returns ``(fe_branch, her_branch)`` — both signed: cathodic positive,
    exactly 0 at their equilibrium potentials, net oxidation anodic of them.
    """
    T_K = temperature_C + 273.15
    fe = ButlerVolmerBranch(
        arrhenius_i0(fe_i0, T_K, fe_i0_Ea_J_mol, kinetics_ref_K),
        fe_tafel_V, fe_E_eq, None, FE_ANODIC_SLOPE_V,
    )
    her = ButlerVolmerBranch(
        arrhenius_i0(her_i0, T_K, her_i0_Ea_J_mol, kinetics_ref_K),
        her_tafel_V, float(her_line(pH, T_K)), None, HER_ANODIC_SLOPE_V,
    )
    return fe, her


@dataclass
class DepositionKinetics:
    """
    Competing Fe deposition and HER on a cathode.

    Default parameters correspond to iron deposition on an iron substrate in a
    mildly acidic sulfate bath; `her_i0` should be lowered (1e-4 - 1e-6 A/m^2)
    to represent high-overpotential cathodes (Pb, Zn, Cd-like) or additive-
    blocked surfaces, which is the primary lever for current efficiency.

    Parameters
    ----------
    pH : float
        Bulk electrolyte pH (used for the HER equilibrium potential).
    temperature_C : float
        Temperature in degrees Celsius.
    fe_i0, her_i0 : float
        Exchange current densities (A/m^2) **at the kinetics reference
        temperature** ``kinetics_ref_K`` (default 50 °C).  At any other
        temperature they are Arrhenius-scaled with
        ``fe_i0_Ea_J_mol`` / ``her_i0_Ea_J_mol``.
    fe_tafel_V, her_tafel_V : float
        Cathodic Tafel slopes (V/decade).
    fe_conc_M : float
        Bulk Fe2+ concentration (mol/L), used for the limiting current.
    boundary_layer_m : float
        Nernst diffusion layer thickness (m); agitation reduces this.
    fe_i0_Ea_J_mol, her_i0_Ea_J_mol : float
        Apparent activation energies of the two branches (J/mol).
    kinetics_ref_K : float
        Anchor temperature for the i0 values (K).  Results at 50 °C are
        identical to the pre-Arrhenius model by construction.
    diffusivity_Ea_J_mol : float
        Activation energy for Fe2+ diffusivity (J/mol); default
        diffusivity is the 25 °C literature value.
    """

    pH: float = 2.0
    temperature_C: float = 60.0
    fe_i0: float = 1.0e-2
    her_i0: float = 1.0e-3
    fe_tafel_V: float = 0.120
    her_tafel_V: float = 0.140
    fe_conc_M: float = 1.0
    diffusivity_m2_s: float = 7.2e-10
    boundary_layer_m: float = 5e-5
    fe_E_eq: float = -0.440
    fe_i0_Ea_J_mol: float = EA_FE_DEPOSITION_J_MOL
    her_i0_Ea_J_mol: float = EA_HER_ON_FE_J_MOL
    kinetics_ref_K: float = I0_REF_K
    diffusivity_Ea_J_mol: float = EA_DIFFUSION_J_MOL
    # Full Butler-Volmer branches (2026-08 default).  The cathodic Tafel
    # slope is retained for the forward arm; the anodic (reverse) arm is
    # derived once at 25 °C (FE_ANODIC_SLOPE_V / HER_ANODIC_SLOPE_V).
    # BV fixes i(E_eq)=0 and represents dissolution anodic of E_eq.
    use_butler_volmer: bool = True
    fe_anodic_slope_V: float = FE_ANODIC_SLOPE_V
    her_anodic_slope_V: float = HER_ANODIC_SLOPE_V

    @property
    def T(self) -> float:
        return self.temperature_C + 273.15

    @property
    def fe_i0_T(self) -> float:
        """Fe exchange current density Arrhenius-scaled to T."""
        return arrhenius_i0(self.fe_i0, self.T, self.fe_i0_Ea_J_mol, self.kinetics_ref_K)

    @property
    def her_i0_T(self) -> float:
        """HER exchange current density Arrhenius-scaled to T."""
        return arrhenius_i0(self.her_i0, self.T, self.her_i0_Ea_J_mol, self.kinetics_ref_K)

    @property
    def D_fe_T(self) -> float:
        """Fe2+ diffusivity Arrhenius-scaled to T (25 °C anchor)."""
        return arrhenius_diffusivity(self.diffusivity_m2_s, self.T,
                                     self.diffusivity_Ea_J_mol)

    @property
    def i_lim(self) -> float:
        """Diffusion-limited Fe deposition current density (A/m^2)."""
        return limiting_current_density(
            self.fe_conc_M * 1000.0, self.D_fe_T, self.boundary_layer_m
        )

    @property
    def fe_branch(self) -> TafelBranch:
        return TafelBranch(self.fe_i0_T, self.fe_tafel_V, self.fe_E_eq, self.i_lim)

    @property
    def her_branch(self) -> TafelBranch:
        return TafelBranch(
            self.her_i0_T, self.her_tafel_V, float(her_line(self.pH, self.T))
        )

    @property
    def fe_branch_bv(self) -> ButlerVolmerBranch:
        return ButlerVolmerBranch(
            self.fe_i0_T, self.fe_tafel_V, self.fe_E_eq,
            self.i_lim, self.fe_anodic_slope_V,
        )

    @property
    def her_branch_bv(self) -> ButlerVolmerBranch:
        return ButlerVolmerBranch(
            self.her_i0_T, self.her_tafel_V, float(her_line(self.pH, self.T)),
            None, self.her_anodic_slope_V,
        )

    # ─── Partial currents ─────────────────────────────────────────────
    def partial_currents(self, E):
        """Return (i_Fe, i_HER, i_total) in A/m^2 at potential E (V vs. SHE).

        With ``use_butler_volmer`` (default) the individual branches are
        SIGNED: anodic of a branch's E_eq it is negative (Fe dissolution /
        H₂ ionisation), so ``i_total`` and ``i_Fe`` can go anodic.  Current
        efficiency is only a meaningful galvanostatic quantity where both
        partial currents are cathodic.
        """
        if self.use_butler_volmer:
            i_fe = self.fe_branch_bv.current(E)
            i_h = self.her_branch_bv.current(E)
        else:
            i_fe = self.fe_branch.current(E)
            i_h = self.her_branch.current(E)
        return i_fe, i_h, i_fe + i_h

    def current_efficiency(self, E):
        """Faradaic current efficiency for Fe at potential E (fraction 0-1).

        Defined where the net current is cathodic (both partial currents
        cathodic); anodic of E_eq(Fe) this is a signed Fe fraction, not a
        meaningful current efficiency.
        """
        i_fe, _, i_tot = self.partial_currents(E)
        return i_fe / np.maximum(i_tot, 1e-30)

    # ─── Galvanostatic operation ──────────────────────────────────────
    def potential_at_current(self, j_mA_cm2: float) -> float:
        """
        Solve for the cathode potential that delivers a given total cathodic
        current density (mA/cm^2 input; 1 mA/cm^2 = 10 A/m^2).
        """
        target = j_mA_cm2 * 10.0

        def f(E):
            return self.partial_currents(E)[2] - target

        lo, hi = -3.0, self.her_branch.E_eq
        if f(hi) > 0:
            hi = self.her_branch.E_eq + 0.5
        return float(brentq(f, lo, hi, xtol=1e-9))

    def efficiency_at_current(self, j_mA_cm2: float) -> float:
        """Current efficiency at a given galvanostatic current density."""
        return float(self.current_efficiency(self.potential_at_current(j_mA_cm2)))

    def polarization_curve(self, E_range: Optional[Iterable[float]] = None):
        """
        Return (E, i_Fe, i_HER, i_total, CE) arrays over a potential sweep.
        Currents in A/m^2 (cathodic positive; with Butler–Volmer branches,
        i_Fe/i_HER go negative anodic of their E_eq).  CE = i_Fe/i_total is
        a galvanostatic concept, defined only where both partial currents
        are cathodic; anodic of E_eq(Fe) it is a signed ratio, not a
        current efficiency.
        """
        if E_range is None:
            E_range = np.linspace(-1.6, -0.3, 400)
        E = np.asarray(list(E_range), dtype=float)
        i_fe, i_h, i_tot = self.partial_currents(E)
        return E, i_fe, i_h, i_tot, i_fe / np.maximum(i_tot, 1e-30)

    def efficiency_sweep(self, j_values_mA_cm2: Iterable[float]):
        """Current efficiency vs. applied current density."""
        js = np.asarray(list(j_values_mA_cm2), dtype=float)
        return js, np.array([self.efficiency_at_current(j) for j in js])

    # ─── Derived process metrics ──────────────────────────────────────
    def deposition_rate_um_hr(self, j_mA_cm2: float, rho_kg_m3: float = 7874.0) -> float:
        """Linear deposition rate (um/hr) at a given current density."""
        ce = self.efficiency_at_current(j_mA_cm2)
        j_A_m2 = j_mA_cm2 * 10.0
        mass_flux = j_A_m2 * ce * M_FE / (Z_FE * FARADAY)  # kg/(m^2 s)
        return mass_flux / rho_kg_m3 * 3600.0 * 1e6

    def hydrogen_flux_mol_m2_hr(self, j_mA_cm2: float) -> float:
        """H2 generation rate at the cathode (mol/(m^2 hr))."""
        E = self.potential_at_current(j_mA_cm2)
        _, i_h, _ = self.partial_currents(E)
        return float(i_h) / (2.0 * FARADAY) * 3600.0

    def summary(self, j_mA_cm2: float = 100.0) -> dict:
        E = self.potential_at_current(j_mA_cm2)
        i_fe, i_h, i_tot = self.partial_currents(E)
        return {
            "j applied (mA/cm²)": j_mA_cm2,
            "E cathode (V vs SHE)": round(E, 3),
            "η_Fe (V)": round(self.fe_E_eq - E, 3),
            "i_Fe (A/m²)": round(float(i_fe), 1),
            "i_HER (A/m²)": round(float(i_h), 1),
            "i_lim,Fe (A/m²)": round(self.i_lim, 1),
            "Current efficiency (%)": round(float(i_fe / i_tot) * 100, 1),
            "Deposition rate (µm/hr)": round(self.deposition_rate_um_hr(j_mA_cm2), 1),
            "H₂ flux (mol/m²/hr)": round(self.hydrogen_flux_mol_m2_hr(j_mA_cm2), 2),
        }
