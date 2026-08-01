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

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import brentq

from .electrochemistry import FARADAY, M_FE, Z_FE
from .pourbaix import her_line

T_REF = 298.15


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
        Exchange current densities (A/m^2).
    fe_tafel_V, her_tafel_V : float
        Cathodic Tafel slopes (V/decade).
    fe_conc_M : float
        Bulk Fe2+ concentration (mol/L), used for the limiting current.
    boundary_layer_m : float
        Nernst diffusion layer thickness (m); agitation reduces this.
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

    @property
    def T(self) -> float:
        return self.temperature_C + 273.15

    @property
    def i_lim(self) -> float:
        """Diffusion-limited Fe deposition current density (A/m^2)."""
        return limiting_current_density(
            self.fe_conc_M * 1000.0, self.diffusivity_m2_s, self.boundary_layer_m
        )

    @property
    def fe_branch(self) -> TafelBranch:
        return TafelBranch(self.fe_i0, self.fe_tafel_V, self.fe_E_eq, self.i_lim)

    @property
    def her_branch(self) -> TafelBranch:
        return TafelBranch(
            self.her_i0, self.her_tafel_V, float(her_line(self.pH, self.T))
        )

    # ─── Partial currents ─────────────────────────────────────────────
    def partial_currents(self, E):
        """Return (i_Fe, i_HER, i_total) in A/m^2 at potential E (V vs. SHE)."""
        i_fe = self.fe_branch.current(E)
        i_h = self.her_branch.current(E)
        return i_fe, i_h, i_fe + i_h

    def current_efficiency(self, E):
        """Faradaic current efficiency for Fe at potential E (fraction 0-1)."""
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
        Currents in A/m^2.
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
