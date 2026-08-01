"""
Steady one-dimensional cathode boundary-layer model.

This module adds the first local-pH correction to the lumped kinetics model.
It treats the cathode diffusion layer as a stagnant film and couples:

* hydroxide generation by HER (2 OH- per H2, or i_H/F),
* Fe2+ consumption by deposition (i_Fe/(2F)), and
* Fe(OH)2 precipitation when the local solubility product is exceeded.

The model is deliberately a screening model, not a full Nernst--Planck
solver: migration, convection inside the film, and transient nucleation are
not represented.  Its purpose is to expose the direction and magnitude of
local-pH and precipitation feedbacks before experimental calibration.

Concentrations are internal SI units (mol/m^3); pH is based on the equivalent
molar concentration in mol/L.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import brentq

from .electrochemistry import E0_FE, FARADAY, R_GAS
from .kinetics import limiting_current_density
from .pourbaix import LOGKSP_FEOH2, her_line

KW = 1.0e-14


@dataclass
class BoundaryLayerState:
    """Operating point returned by :class:`CathodeBoundaryLayer`."""

    potential_V: float
    applied_current_A_m2: float
    fe_current_A_m2: float
    her_current_A_m2: float
    current_efficiency: float
    bulk_pH: float
    surface_pH: float
    bulk_fe_M: float
    surface_fe_M: float
    surface_oh_M: float
    fe_transport_limit_A_m2: float
    precipitation_active: bool
    feoh2_supersaturation: float

    @property
    def local_pH_rise(self) -> float:
        """Surface pH minus bulk pH."""
        return self.surface_pH - self.bulk_pH


@dataclass
class CathodeBoundaryLayer:
    """Steady film model coupled to competing Fe/HER electrode kinetics.

    Parameters
    ----------
    bulk_pH : float
        Bulk electrolyte pH.
    fe_conc_M : float
        Bulk Fe2+ concentration in mol/L.
    boundary_layer_m : float
        Film thickness. Agitation reduces this value.
    temperature_C : float
        Temperature used in the Nernst equation and HER line.
    fe_i0, her_i0 : float
        Exchange current densities in A/m2.
    fe_tafel_V, her_tafel_V : float
        Cathodic Tafel slopes in V/decade.
    diffusivity_fe_m2_s, diffusivity_oh_m2_s : float
        Effective film diffusivities.
    """

    bulk_pH: float = 2.0
    fe_conc_M: float = 1.0
    boundary_layer_m: float = 5e-5
    temperature_C: float = 60.0
    fe_i0: float = 1e-2
    her_i0: float = 1e-6
    fe_tafel_V: float = 0.120
    her_tafel_V: float = 0.140
    diffusivity_fe_m2_s: float = 7.2e-10
    diffusivity_oh_m2_s: float = 5.3e-9
    max_iterations: int = 200
    convergence_tol: float = 1e-8

    @property
    def T(self) -> float:
        return self.temperature_C + 273.15

    @property
    def bulk_oh_M(self) -> float:
        return 10.0 ** (self.bulk_pH - 14.0)

    @property
    def fe_transport_limit_A_m2(self) -> float:
        return limiting_current_density(
            self.fe_conc_M * 1000.0,
            self.diffusivity_fe_m2_s,
            self.boundary_layer_m,
        )

    def _fe_equilibrium_potential(self, fe_surface_M: float) -> float:
        """Nernst potential for Fe2+/Fe, with activity approximated by M."""
        activity = max(fe_surface_M, 1e-15)
        return E0_FE + (R_GAS * self.T / (2.0 * FARADAY)) * np.log(activity)

    def _tafel_current(self, E: float, i0: float, slope: float, E_eq: float) -> float:
        eta = E_eq - E
        if eta <= 0.0:
            return float(i0 * 10.0 ** (eta / slope))
        return float(i0 * 10.0 ** (eta / slope))

    def _state_at_potential(self, E: float) -> BoundaryLayerState:
        """Iterate surface concentrations at a specified cathode potential."""
        fe_surface_M = max(self.fe_conc_M * 0.99, 1e-12)
        surface_pH = self.bulk_pH
        precipitation_seen = False
        max_supersaturation = 0.0

        for _ in range(self.max_iterations):
            oh_surface_M = max(10.0 ** (surface_pH - 14.0), self.bulk_oh_M)
            her_eq = float(her_line(surface_pH, self.T))
            i_her = self._tafel_current(E, self.her_i0, self.her_tafel_V, her_eq)

            # Fe flux depletes the film. Kinetic current is capped by the
            # bulk-to-surface transport limit, as in the Koutecky picture.
            i_fe_kin = self._tafel_current(
                E, self.fe_i0, self.fe_tafel_V,
                self._fe_equilibrium_potential(fe_surface_M),
            )
            i_fe = min(i_fe_kin, self.fe_transport_limit_A_m2)
            fe_transport_M = max(
                self.fe_conc_M
                - i_fe * self.boundary_layer_m
                / (2.0 * FARADAY * self.diffusivity_fe_m2_s * 1000.0),
                1e-15,
            )

            # Fe(OH)2(s) equilibrium: Ksp = [Fe2+][OH-]^2.
            feoh2_eq_M = (10.0 ** LOGKSP_FEOH2) / (oh_surface_M**2)
            precipitation = fe_transport_M > feoh2_eq_M
            precipitation_seen = precipitation_seen or precipitation
            max_supersaturation = max(
                max_supersaturation,
                fe_transport_M * oh_surface_M**2 / (10.0 ** LOGKSP_FEOH2),
            )
            next_fe = min(fe_transport_M, feoh2_eq_M) if precipitation else fe_transport_M

            # HER produces OH- at i_H/F. Convert mol/m3 to mol/L by /1000.
            oh_surface_from_flux_M = self.bulk_oh_M + (
                i_her * self.boundary_layer_m
                / (FARADAY * self.diffusivity_oh_m2_s * 1000.0)
            )
            next_pH = 14.0 + np.log10(max(oh_surface_from_flux_M, 1e-15))

            if (
                abs(next_fe - fe_surface_M) < self.convergence_tol
                and abs(next_pH - surface_pH) < self.convergence_tol
            ):
                fe_surface_M, surface_pH = next_fe, next_pH
                break
            fe_surface_M = 0.5 * fe_surface_M + 0.5 * next_fe
            surface_pH = 0.5 * surface_pH + 0.5 * next_pH

        oh_surface_M = 10.0 ** (surface_pH - 14.0)
        i_her = self._tafel_current(
            E, self.her_i0, self.her_tafel_V, float(her_line(surface_pH, self.T))
        )
        i_fe = min(
            self._tafel_current(
                E, self.fe_i0, self.fe_tafel_V,
                self._fe_equilibrium_potential(fe_surface_M),
            ),
            self.fe_transport_limit_A_m2,
        )
        supersaturation = (
            fe_surface_M * oh_surface_M**2 / (10.0 ** LOGKSP_FEOH2)
        )
        total = i_fe + i_her
        return BoundaryLayerState(
            potential_V=E,
            applied_current_A_m2=total,
            fe_current_A_m2=i_fe,
            her_current_A_m2=i_her,
            current_efficiency=i_fe / max(total, 1e-30),
            bulk_pH=self.bulk_pH,
            surface_pH=surface_pH,
            bulk_fe_M=self.fe_conc_M,
            surface_fe_M=fe_surface_M,
            surface_oh_M=oh_surface_M,
            fe_transport_limit_A_m2=self.fe_transport_limit_A_m2,
            # The fixed-point iteration may clamp Fe2+ to solubility and then
            # show a sub-saturated final state; retain whether precipitation
            # was encountered during the coupled iteration.
            precipitation_active=precipitation_seen or supersaturation >= 1.0,
            feoh2_supersaturation=float(max(max_supersaturation, supersaturation, 0.0)),
        )

    def solve(self, j_mA_cm2: float) -> BoundaryLayerState:
        """Solve the cathode potential for an applied current density."""
        if j_mA_cm2 <= 0.0:
            raise ValueError("j_mA_cm2 must be positive")
        target = j_mA_cm2 * 10.0

        def residual(E: float) -> float:
            return self._state_at_potential(E).applied_current_A_m2 - target

        E = brentq(residual, -3.0, 0.2, xtol=1e-9)
        return self._state_at_potential(float(E))

    def concentration_profiles(
        self, state: BoundaryLayerState, points: int = 101
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return distance, Fe2+, and OH- profiles from cathode to bulk.

        The profiles are linear film solutions and use the solved partial
        currents. Distance is measured outward from the cathode in metres.
        """
        if points < 2:
            raise ValueError("points must be at least 2")
        x = np.linspace(0.0, self.boundary_layer_m, points)
        fe = state.surface_fe_M + (
            self.fe_conc_M - state.surface_fe_M
        ) * x / self.boundary_layer_m
        oh = state.surface_oh_M + (
            self.bulk_oh_M - state.surface_oh_M
        ) * x / self.boundary_layer_m
        return x, fe, oh

    def efficiency_sweep(
        self, j_values_mA_cm2: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return current densities and local-pH-corrected efficiencies."""
        js = np.asarray(list(j_values_mA_cm2), dtype=float)
        states = [self.solve(float(j)) for j in js]
        return js, np.array([s.current_efficiency for s in states])
