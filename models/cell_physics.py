"""
Unified cell physics solver for aqueous iron electrowinning.

Connects speciation, transport (Nernst-Planck), and cell voltage into a
single self-consistent model.  Given a bath recipe and operating conditions,
predicts V_cell, FE, transport limits, and surface chemistry — the numbers
that dark_mill.py currently assumes.

Data flow:
    BathRecipe + ProcessConditions
        → solve_speciation()         activities, free [Fe²⁺], conductivity
        → NernstPlanckFilm.solve(j)  FE, surface pH, transport limit
        → CellVoltageModel.V_cell    voltage decomposition
        → OperatingPoint             everything at one j
        → sweep()                    OperatingWindow across j range
        → find_optimal_j()           best operating point for areal productivity

References
----------
- transport.py — steady 1-D Nernst-Planck film model
- speciation.py — Davies activity coefficients, HSO4⁻/FeSO4 pairing
- electrochemistry.py — CellVoltageModel, conductivity, diffusivity
- kinetics.py — DepositionKinetics, Tafel branches
- anode.py — AnodeKinetics, bubble resistance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

from .speciation import SolutionComposition, solve_speciation
from .transport import NernstPlanckFilm, NernstPlanckState
from .electrochemistry import (
    CellVoltageModel, MembraneModel, specific_energy_kWh_per_t, FARADAY, M_FE, Z_FE,
)
from .kinetics import DepositionKinetics


# ─── Inputs ───────────────────────────────────────────────────────

@dataclass
class BathRecipe:
    """Bath composition — the chemistry input."""
    c_FeSO4_M: float = 1.0          # mol/L FeSO4
    c_Na2SO4_M: float = 0.5         # mol/L supporting electrolyte
    c_H2SO4_M: float = 0.01         # mol/L (pH adjustment)
    c_H3BO3_M: float = 0.4          # mol/L boric acid buffer
    pH: float = 2.0                 # bulk pH (sets H2SO4 if not overridden)

    def to_speciation(self, T_C: float) -> SolutionComposition:
        return SolutionComposition(
            c_FeSO4=self.c_FeSO4_M,
            c_Na2SO4=self.c_Na2SO4_M,
            c_H2SO4=self.c_H2SO4_M,
            c_H3BO3=self.c_H3BO3_M,
            T_C=T_C,
        )


@dataclass
class CellGeometry:
    """Physical cell configuration."""
    interelectrode_gap_m: float = 0.02     # 2 cm default
    membrane: bool = True                  # divided cell
    membrane_area_resistance_ohm_m2: float = 3.0e-4  # Nafion N117 at 50°C
    contact_resistance_ohm_m2: float = 5.0e-4
    anode_bubble_fraction: float = 0.10


@dataclass
class ProcessConditions:
    """Operating conditions for the cell."""
    temperature_C: float = 50.0
    boundary_layer_m: float = 50e-6        # 50 µm (moderate agitation)
    flow_regime: str = "moderate"           # "still", "moderate", "vigorous"

    # Kinetic parameters (literature defaults for Fe/FeSO4 on Fe cathode)
    fe_i0: float = 1.0e-2                  # A/m² exchange current density
    her_i0: float = 1.0e-6                 # A/m² — suppressed HER (additive/overpotential)
    fe_tafel_V: float = 0.120
    her_tafel_V: float = 0.140


# ─── Outputs ──────────────────────────────────────────────────────

@dataclass
class OperatingPoint:
    """Self-consistent physics result at one current density."""
    j_mA_cm2: float                        # applied current density

    # From transport (Nernst-Planck)
    current_efficiency: float              # FE fraction (0-1)
    surface_pH: float
    surface_fe_M: float
    transport_limit_mA_cm2: float          # migration-enhanced limit
    diffusion_limit_mA_cm2: float          # Levich limit
    migration_enhancement: float           # transport_limit / diffusion_limit
    feoh2_supersaturation: float
    film_potential_drop_V: float
    precipitation_active: bool

    # From cell voltage model
    V_cell: float                          # total cell voltage (V)
    V_decomposition: Dict[str, float]      # E_cathode, E_anode, η, IR breakdown

    # Derived
    specific_energy_kWh_t: float           # 0.96 × V_cell / FE × 1000
    deposition_rate_um_hr: float           # linear growth rate

    # From speciation
    free_fe2_activity: float               # activity-corrected [Fe2+]
    conductivity_S_m: float                # electrolyte conductivity
    speciation: Dict[str, Any]             # full speciation result

    # Convergence
    transport_converged: bool


@dataclass
class OperatingWindow:
    """Sweep of OperatingPoint across current densities."""
    j_range_mA_cm2: np.ndarray
    points: List[OperatingPoint]

    @property
    def FE_array(self) -> np.ndarray:
        return np.array([p.current_efficiency for p in self.points])

    @property
    def V_cell_array(self) -> np.ndarray:
        return np.array([p.V_cell for p in self.points])

    @property
    def energy_array(self) -> np.ndarray:
        return np.array([p.specific_energy_kWh_t for p in self.points])

    @property
    def transport_limit_mA_cm2(self) -> float:
        """Minimum transport limit across the sweep."""
        return min(p.transport_limit_mA_cm2 for p in self.points)

    @property
    def j_max_feasible(self) -> float:
        """Highest j where FE > 50% and no precipitation."""
        for p in reversed(self.points):
            if p.current_efficiency > 0.50 and not p.precipitation_active:
                return p.j_mA_cm2
        return 0.0

    def optimal_j(self, min_FE: float = 0.70) -> Optional[OperatingPoint]:
        """Best operating point: max j with FE ≥ min_FE and no precipitation."""
        candidates = [
            p for p in self.points
            if p.current_efficiency >= min_FE and not p.precipitation_active
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.j_mA_cm2)


# ─── Physics Solver ───────────────────────────────────────────────

class CellPhysics:
    """
    Unified physics solver.  Connects speciation → transport → voltage.

    Usage:
        physics = CellPhysics(bath, geometry, conditions)
        point = physics.solve_at_j(100.0)    # mA/cm²
        window = physics.sweep()              # full operating window
        optimal = physics.find_optimal_j()    # best operating point
    """

    def __init__(
        self,
        bath: BathRecipe,
        geometry: CellGeometry = CellGeometry(),
        conditions: ProcessConditions = ProcessConditions(),
    ):
        self.bath = bath
        self.geometry = geometry
        self.conditions = conditions

        # Run speciation once (depends on T and composition, not j)
        self._spec = solve_speciation(bath.to_speciation(conditions.temperature_C))

        # Cache derived quantities
        self._free_fe2_M = self._spec.get("c_Fe2_free_M", bath.c_FeSO4_M)
        self._activity_fe2 = self._spec.get("a_Fe2", self._free_fe2_M)
        self._conductivity = self._spec.get("conductivity_S_m", 10.0)
        self._gamma2 = self._spec.get("gamma_Fe2", 1.0)

    def _build_transport(self) -> NernstPlanckFilm:
        """Build the Nernst-Planck film model.

        Uses NOMINAL Fe concentration for transport (FeSO4(aq) pairs
        dissociate at the cathode as Fe²⁺ is consumed — the transport
        limit is set by total Fe(II), not free Fe²⁺).
        Uses speciation-corrected conductivity for the voltage model.
        """
        return NernstPlanckFilm(
            bulk_pH=self.bath.pH,
            fe_conc_M=self.bath.c_FeSO4_M,   # total Fe(II), not free
            support_conc_M=self.bath.c_Na2SO4_M,
            boundary_layer_m=self.conditions.boundary_layer_m,
            temperature_C=self.conditions.temperature_C,
            fe_i0=self.conditions.fe_i0,
            her_i0=self.conditions.her_i0,
            fe_tafel_V=self.conditions.fe_tafel_V,
            her_tafel_V=self.conditions.her_tafel_V,
            grid_points=61,                    # faster than default 121
        )

    def _build_voltage_model(
        self, j_mA_cm2: float, cathode_overpotential_V: float
    ) -> CellVoltageModel:
        """Build the cell voltage model with transport-corrected parameters."""
        g = self.geometry
        return CellVoltageModel(
            E_cathode_eq=self._spec.get("E_rev_Fe_V_SHE", -0.440),
            eta_cathode=cathode_overpotential_V,
            temperature_C=self.conditions.temperature_C,
            fe2_conc_M=self._free_fe2_M,
            electrolyte_conductivity_S_m=self._conductivity,
            interelectrode_gap_m=g.interelectrode_gap_m,
            contact_resistance_ohm_m2=g.contact_resistance_ohm_m2,
            bubble_fraction=g.anode_bubble_fraction,
            divided_cell=g.membrane,
            membrane=MembraneModel(R_membrane_ohm_m2=g.membrane_area_resistance_ohm_m2) if g.membrane else None,
            j_operating_mA_cm2=j_mA_cm2,
        )

    def solve_at_j(self, j_mA_cm2: float) -> OperatingPoint:
        """
        Solve the full physics at one current density.

        Returns a self-consistent OperatingPoint with FE, V_cell,
        transport limits, and surface chemistry all computed from physics.
        """
        transport = self._build_transport()
        np_state: NernstPlanckState = transport.solve(j_mA_cm2)

        # Cathode overpotential from transport model
        # (the transport solver finds E_cathode that gives the target j)
        E_cathode_eq = self._spec.get("E_rev_Fe_V_SHE", -0.440)
        eta_cathode = max(E_cathode_eq - np_state.potential_V, 0.0)

        # Build voltage model with physics-derived overpotential
        vm = self._build_voltage_model(j_mA_cm2, eta_cathode)
        V_cell = vm.V_cell
        V_decomp = vm.V_decomposition

        # Deposition rate
        fe = np_state.current_efficiency
        j_A_m2 = j_mA_cm2 * 10.0
        mass_flux = j_A_m2 * fe * M_FE / (Z_FE * FARADAY)  # kg/(m²·s)
        rho = 7874.0  # kg/m³
        dep_rate = mass_flux / rho * 3600.0 * 1e6  # µm/hr

        return OperatingPoint(
            j_mA_cm2=j_mA_cm2,
            current_efficiency=fe,
            surface_pH=np_state.surface_pH,
            surface_fe_M=np_state.surface_fe_M,
            transport_limit_mA_cm2=np_state.transport_limit_A_m2 / 10.0,
            diffusion_limit_mA_cm2=np_state.diffusion_limit_A_m2 / 10.0,
            migration_enhancement=np_state.migration_enhancement,
            feoh2_supersaturation=np_state.feoh2_supersaturation,
            film_potential_drop_V=np_state.film_potential_drop_V,
            precipitation_active=np_state.precipitation_active,
            V_cell=V_cell,
            V_decomposition=V_decomp,
            specific_energy_kWh_t=specific_energy_kWh_per_t(V_cell, fe),
            deposition_rate_um_hr=dep_rate,
            free_fe2_activity=self._activity_fe2,
            conductivity_S_m=self._conductivity,
            speciation=self._spec,
            transport_converged=np_state.converged,
        )

    def sweep(
        self,
        j_min: float = 10.0,
        j_max: float = 400.0,
        n_points: int = 10,
    ) -> OperatingWindow:
        """
        Sweep across current densities to map the operating window.

        Returns an OperatingWindow with FE(j), V_cell(j), energy(j).
        """
        j_range = np.linspace(j_min, j_max, n_points)
        points = []
        for j in j_range:
            try:
                pt = self.solve_at_j(float(j))
                points.append(pt)
            except (ValueError, RuntimeError):
                # Solver failed — transport limit exceeded or convergence failure
                break

        return OperatingWindow(j_range_mA_cm2=j_range[:len(points)], points=points)

    def find_optimal_j(
        self,
        min_FE: float = 0.70,
        j_min: float = 10.0,
        j_max: float = 400.0,
        n_points: int = 10,
    ) -> Optional[OperatingPoint]:
        """
        Find the optimal current density: max j with FE ≥ min_FE
        and no Fe(OH)₂ precipitation.

        Uses a fast kinetics model (DepositionKinetics) for the sweep,
        then validates with the full Nernst-Planck transport solver at
        the optimal point.  This avoids running the expensive ODE solver
        at every sweep point.
        """
        # Fast sweep with DepositionKinetics (no ODE, instant)
        dk = DepositionKinetics(
            pH=self.bath.pH,
            temperature_C=self.conditions.temperature_C,
            fe_i0=self.conditions.fe_i0,
            her_i0=self.conditions.her_i0,
            fe_tafel_V=self.conditions.fe_tafel_V,
            her_tafel_V=self.conditions.her_tafel_V,
            fe_conc_M=self.bath.c_FeSO4_M,
            boundary_layer_m=self.conditions.boundary_layer_m,
        )

        j_range = np.linspace(j_min, j_max, n_points)
        best_j = None
        for j in j_range:
            try:
                fe = dk.efficiency_at_current(float(j))
                if fe >= min_FE:
                    # Quick V_cell estimate (thermodynamic + Tafel overpotential + IR)
                    E_cathode = dk.potential_at_current(float(j))
                    eta_cathode = max(dk.fe_E_eq - E_cathode, 0.0)
                    V_est = 1.72 + eta_cathode + 0.4 + 0.3 + (float(j) * 10.0 * 0.02) / (self._conductivity * 0.9)
                    energy_est = specific_energy_kWh_per_t(V_est, fe)
                    if energy_est <= 5000:  # practical energy ceiling
                        best_j = float(j)
            except (ValueError, RuntimeError):
                break

        if best_j is None:
            return None

        # Validate with full Nernst-Planck at the optimal point
        try:
            return self.solve_at_j(best_j)
        except (ValueError, RuntimeError):
            return None

    def summary(self, j_mA_cm2: float = 100.0) -> Dict[str, Any]:
        """Human-readable summary at a given current density."""
        pt = self.solve_at_j(j_mA_cm2)
        return {
            "Bath": {
                "FeSO4 (M)": self.bath.c_FeSO4_M,
                "Na2SO4 (M)": self.bath.c_Na2SO4_M,
                "pH": self.bath.pH,
                "T (°C)": self.conditions.temperature_C,
                "Free [Fe²⁺] (M)": round(self._free_fe2_M, 3),
                "γ(Fe²⁺)": round(self._gamma2, 3),
                "Conductivity (S/m)": round(self._conductivity, 1),
            },
            "Operating point": {
                "j (mA/cm²)": j_mA_cm2,
                "FE (%)": round(pt.current_efficiency * 100, 1),
                "V_cell (V)": round(pt.V_cell, 3),
                "Energy (kWh/t)": round(pt.specific_energy_kWh_t, 0),
                "Deposition (µm/hr)": round(pt.deposition_rate_um_hr, 1),
            },
            "Transport": {
                "i_lim diffusion (mA/cm²)": round(pt.diffusion_limit_mA_cm2, 0),
                "i_lim with migration (mA/cm²)": round(pt.transport_limit_mA_cm2, 0),
                "Migration enhancement": f"{pt.migration_enhancement:.2f}×",
                "Surface pH": round(pt.surface_pH, 2),
                "Surface [Fe²⁺] (M)": round(pt.surface_fe_M, 3),
                "Fe(OH)₂ supersaturation": f"{pt.feoh2_supersaturation:.2g}",
                "Precipitation": "YES" if pt.precipitation_active else "no",
            },
            "Voltage breakdown": pt.V_decomposition,
        }
