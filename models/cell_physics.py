"""
Unified cell physics solver for aqueous iron electrowinning.

Connects speciation, transport (Nernst-Planck), and cell voltage into a
single self-consistent model.  Given a bath recipe and operating conditions,
predicts V_cell, FE, transport limits, and surface chemistry — the numbers
that dark_mill.py currently assumes.

Data flow:
    BathRecipe or BathSpec + ProcessConditions
        → BathSpec.solve_bulk_speciation()  sulfate/chloride/ammonium/O₂ diagnostics
        → chemistry-rich current ledger     ORR and Fe³⁺ side branches, if enabled
        → DiffusionLayer1D.solve(j_core)    Fe/HER FE, surface pH, transport limit
        → CellVoltageModel.V_cell           voltage decomposition at applied j
        → OperatingPoint                    FE/V/deposit/current-breakdown at one j
        → sweep()                           OperatingWindow across j range
        → find_optimal_j()                  best operating point for areal productivity

References
----------
- transport.py — steady 1-D Nernst-Planck film model
- speciation.py — Pitzer activity coefficients, HSO4⁻ dissociation
  (legacy Davies + explicit FeSO4⁰ pairing retained for A/B only)
- electrochemistry.py — CellVoltageModel, conductivity, diffusivity
- kinetics.py — DepositionKinetics, Tafel branches
- anode.py — AnodeKinetics, bubble resistance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Literal
import numpy as np

from .speciation import SolutionComposition
from .bath_spec import BathSpec
from .transport import NernstPlanckFilm, NernstPlanckState
from .diffusion_layer_1d import DiffusionLayer1D, DiffusionLayerResult
from .electrochemistry import (
    CellVoltageModel, MembraneModel, specific_energy_kWh_per_t, FARADAY, M_FE, Z_FE,
    E0_FE,
)
from .kinetics import DepositionKinetics
from .anode import AnodeKinetics, AnodeMaterial, DSA_IRO2_TA2O5
from .dissolved_oxygen import (
    cathodic_orr_limiting_current_A_m2,
    dissolved_oxygen_solubility_M,
    homogeneous_fe2_oxidation_rate_M_s,
)
from .fe3_shuttle import (
    D_FE3_REF_M2_S,
    ShuttleParams,
    ShuttleScenario,
    steady_state as fe3_shuttle_steady_state,
)


# ─── Inputs ───────────────────────────────────────────────────────

@dataclass
class BathRecipe:
    """Bath composition — the chemistry input."""
    c_FeSO4_M: float = 1.0          # mol/L FeSO4
    c_Na2SO4_M: float = 0.5         # mol/L supporting electrolyte
    c_H2SO4_M: float = 0.01         # mol/L (pH adjustment)
    c_H3BO3_M: float = 0.4          # mol/L boric acid additive; capacity is derived
    pH: float = 2.0                 # declared/measured bulk electrolyte pH boundary

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
    """Physical cell configuration.

    ``anode_chemistry`` is explicit because soluble Fe and inert OER anodes
    have different thermodynamics, gas production, acid balances, and
    voltage.  The default remains the historical inert/OER screening branch.
    """
    interelectrode_gap_m: float = 0.02     # 2 cm default
    membrane: bool = True                  # divided cell
    membrane_area_resistance_ohm_m2: float = 3.0e-4  # configured comparator
    contact_resistance_ohm_m2: float = 5.0e-4
    anode_bubble_fraction: float = 0.10
    anode_chemistry: Literal["inert", "soluble", "fixed"] = "inert"
    anode_fe2_conc_M: float = 1.0
    anode_fe_dissolution_i0_A_m2: float = 10.0

    def __post_init__(self) -> None:
        if self.interelectrode_gap_m <= 0.0:
            raise ValueError("interelectrode_gap_m must be positive")
        if self.membrane_area_resistance_ohm_m2 < 0.0:
            raise ValueError("membrane_area_resistance_ohm_m2 must be non-negative")
        if self.contact_resistance_ohm_m2 < 0.0:
            raise ValueError("contact_resistance_ohm_m2 must be non-negative")
        if not 0.0 <= self.anode_bubble_fraction < 1.0:
            raise ValueError("anode_bubble_fraction must lie in [0, 1)")
        if self.anode_chemistry not in ("inert", "soluble", "fixed"):
            raise ValueError("anode_chemistry must be 'inert', 'soluble', or 'fixed'")
        if self.anode_fe2_conc_M <= 0.0:
            raise ValueError("anode_fe2_conc_M must be positive")
        if self.anode_fe_dissolution_i0_A_m2 <= 0.0:
            raise ValueError("anode_fe_dissolution_i0_A_m2 must be positive")


@dataclass
class ProcessConditions:
    """Operating conditions for the cell.

    ``chemistry_coupling="legacy"`` preserves the historical cell solve.  Set
    ``chemistry_coupling="rich"`` (or use :meth:`rich`) to attach the shared
    :class:`models.bath_spec.BathSpec` inventory to the central prediction path:
    surface-state HER, FeSO₄⁰ pairing, Fe(OH)₂ film diagnostics, dissolved-O₂
    ORR, and Fe³⁺ shuttle losses are all evaluated from one bath state.
    """
    temperature_C: float = 50.0
    boundary_layer_m: float = 50e-6        # 50 µm (moderate agitation)
    flow_regime: str = "moderate"           # "still", "moderate", "vigorous"
    # The reactive film path carries bisulfate/borate chemistry and the
    # activity correction into the cathode film.  ``dilute_np`` remains as an
    # A/B fallback for legacy comparisons.
    transport_model: Literal["reactive", "dilute_np"] = "reactive"

    # Chemistry coupling mode.  Individual booleans below are additive: they
    # can enable one correction in legacy mode, while rich mode enables all of
    # them unless the caller chooses a new ProcessConditions object.
    chemistry_coupling: Literal["legacy", "rich"] = "legacy"
    surface_state_her: bool = False
    fe_pair_correction: bool = False
    feoh2_film: bool = False
    dissolved_oxygen: bool = False
    fe3_shuttle: bool = False

    # Kinetic parameters (literature defaults for Fe/FeSO4 on Fe cathode)
    fe_i0: float = 1.0e-2                  # A/m² exchange current density
    her_i0: float = 1.0e-6                 # A/m² — suppressed HER (additive/overpotential)
    fe_tafel_V: float = 0.120
    her_tafel_V: float = 0.140

    @classmethod
    def rich(cls, **kwargs: Any) -> "ProcessConditions":
        """Convenience constructor for the chemistry-rich central path."""
        kwargs.setdefault("chemistry_coupling", "rich")
        return cls(**kwargs)

    def chemistry_enabled(self, feature: str) -> bool:
        """Return True when a named chemistry correction should be active."""
        if self.chemistry_coupling == "rich":
            return True
        return bool(getattr(self, feature))


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

    # Precipitation sink (screening) — set when precipitation_sink=True
    precipitation_flux_mol_m2_s: float = 0.0
    precipitation_fraction: float = 0.0
    sludge_rate_g_m2_s: float = 0.0
    deposition_rate_net_um_hr: float = 0.0  # after subtracting Fe lost to Fe(OH)2 sludge
    # NOTE: specific_energy_kWh_t still uses gross FE (V/FE); net rate is for
    # mass-balance diagnostics. At RC-1 (S~1e-8) gross==net; they diverge only
    # when precipitation is active, which RC-1 never reaches.

    # Chemistry-rich current accounting.  ``current_efficiency`` above is the
    # applied-current FE after these parasitic branches are subtracted; the
    # legacy transport-only FE is retained here for A/B residual dashboards.
    applied_current_A_m2: float = 0.0
    transport_current_efficiency: float = 0.0
    fe_current_A_m2: float = 0.0
    her_current_A_m2: float = 0.0
    orr_current_A_m2: float = 0.0
    fe3_shuttle_current_A_m2: float = 0.0
    parasitic_current_A_m2: float = 0.0
    current_breakdown_A_m2: Dict[str, float] = None
    current_breakdown_fraction: Dict[str, float] = None
    chemistry_diagnostics: Dict[str, Any] = None


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
        bath: BathRecipe | BathSpec,
        geometry: CellGeometry = CellGeometry(),
        conditions: ProcessConditions = ProcessConditions(),
    ):
        # ``BathRecipe`` remains supported for backward compatibility; the new
        # ``BathSpec`` is retained as the richer chemistry inventory.  Internally
        # the sulfate-native film still needs the legacy recipe view.
        if isinstance(bath, BathSpec):
            self.bath_spec = bath
            self.bath = BathRecipe(**bath.to_legacy_bath_recipe_kwargs())
        else:
            self.bath = bath
            self.bath_spec = BathSpec.from_legacy_recipe(
                bath, temperature_C=conditions.temperature_C
            )
        self.geometry = geometry
        self.conditions = conditions

        # Run bulk speciation once (depends on T and composition, not j).  The
        # BathSpec path enriches this with chloride/ammonium/O₂ diagnostics when
        # those inventories are present; legacy sulfate recipes produce the same
        # Pitzer keys the old solver exposed.
        self._spec = self.bath_spec.solve_bulk_speciation(conditions.temperature_C)
        # A recipe can specify both acid loading and a measured/setpoint pH.
        # Keep the declared pH as the transport boundary condition, but expose
        # the activity-pH mismatch instead of silently pretending the two are
        # the same state.  The calibration pipeline can later choose which
        # quantity is authoritative for a run.
        pH_activity = self._spec.get("pH_activity")
        if pH_activity is not None:
            self._spec["recipe_pH"] = float(self.bath.pH)
            self._spec["pH_activity_delta_from_recipe"] = float(pH_activity - self.bath.pH)
            self._spec["pH_consistency_warning"] = bool(abs(pH_activity - self.bath.pH) > 0.25)
            self._spec["pH_boundary_source"] = "BathRecipe.pH (declared/measured setpoint)"

        # Cache derived quantities
        self._free_fe2_M = self._spec.get("c_Fe2_free_M", self.bath.c_FeSO4_M)
        self._activity_fe2 = self._spec.get("a_Fe2", self._free_fe2_M)
        self._conductivity = self._spec.get("conductivity_S_m", 10.0)
        self._gamma2 = self._spec.get("gamma_Fe2", 1.0)

    def _build_transport(self) -> DiffusionLayer1D | NernstPlanckFilm:
        """Build the selected cathode-film model.

        The default reactive path transports conserved Fe/sulfate/borate
        components and applies the Pitzer activity correction to the Fe
        interfacial equilibrium.  The dilute NP path remains available for
        controlled A/B comparisons with the pre-upgrade implementation.
        """
        common = dict(
            pH_bulk=self.bath.pH,
            fe_conc_M=self.bath.c_FeSO4_M,
            support_conc_M=self.bath.c_Na2SO4_M,
            delta_m=self.conditions.boundary_layer_m,
            temperature_C=self.conditions.temperature_C,
            fe_i0=self.conditions.fe_i0,
            her_i0=self.conditions.her_i0,
            fe_tafel_V=self.conditions.fe_tafel_V,
            her_tafel_V=self.conditions.her_tafel_V,
        )
        if self.conditions.transport_model == "reactive":
            return DiffusionLayer1D(
                **common,
                buffer_conc_M=self.bath.c_H3BO3_M,
                activity_model="pitzer",
                grid_points=61,
                fast_mode=True,
                surface_state=self.conditions.chemistry_enabled("surface_state_her"),
                bath_type=self.bath_spec.surface_state_bath_type(),
                fes04_pair_correction=self.conditions.chemistry_enabled("fe_pair_correction"),
                feoh2_film=self.conditions.chemistry_enabled("feoh2_film"),
            )
        if self.conditions.transport_model == "dilute_np":
            return NernstPlanckFilm(
                bulk_pH=self.bath.pH,
                fe_conc_M=self.bath.c_FeSO4_M,
                support_conc_M=self.bath.c_Na2SO4_M,
                boundary_layer_m=self.conditions.boundary_layer_m,
                temperature_C=self.conditions.temperature_C,
                fe_i0=self.conditions.fe_i0,
                her_i0=self.conditions.her_i0,
                fe_tafel_V=self.conditions.fe_tafel_V,
                her_tafel_V=self.conditions.her_tafel_V,
                grid_points=61,
            )
        raise ValueError(f"unknown transport_model: {self.conditions.transport_model!r}")

    def _build_anode(self) -> Optional[AnodeKinetics]:
        """Build a first-principles anode object for explicit anode modes.

        Both branches now use a first-principles AnodeKinetics object so
        the cell voltage is computed from material-specific Tafel kinetics
        plus concentration and bubble terms, not a fixed 0.40 V.  The
        inert/OER branch uses the DSA IrO₂–Ta₂O₅ screening catalogue
        entry (Trasatti 2000) at the operating T/pH; the soluble branch
        uses the Fe → Fe²⁺ dissolution stoichiometry.  Pass
        ``anode_chemistry="soluble"`` for the latter.  The fixed-η
        fallback is retained only when ``CellGeometry`` explicitly asks
        for it via ``anode_chemistry="fixed"`` (tests / legacy A/B).

        At the coupled-screen optimum (150 mA/cm², 1.5 mm, low contact)
        the DSA correction is <1 mV; at the theory-confidence reference
        (96.7 mA/cm², 20 mm gap, divided cell) the pH-corrected DSA
        (E_eq≈1.10 V at pH 2) lowers V_cell by ~0.14 V (5.388→5.246 V)
        relative to the fixed 1.229 V / 0.40 V fallback — the honest,
        material-specific cost that the energy number is most sensitive to.
        """
        if self.geometry.anode_chemistry == "soluble":
            material = AnodeMaterial(
                name="Soluble Fe anode",
                oer_i0=1.0,
                oer_tafel_V=0.060,
                temperature_C=self.conditions.temperature_C,
                references="screening soluble-Fe branch; calibrate dissolution i0/Tafel",
            )
            return AnodeKinetics(
                material=material,
                electrolyte_type="acidic",
                pH=self.bath.pH,
                boundary_layer_m=self.conditions.boundary_layer_m,
                electrolyte_conductivity_S_m=self._conductivity,
                anode_chemistry="soluble",
                fe2_conc_M=self.geometry.anode_fe2_conc_M,
                fe_dissolution_i0=self.geometry.anode_fe_dissolution_i0_A_m2,
            )
        if self.geometry.anode_chemistry == "fixed":
            return None
        # Default: inert DSA — first-principles OER kinetics
        base = DSA_IRO2_TA2O5
        material = AnodeMaterial(
            name=base.name,
            oer_i0=base.oer_i0,
            oer_tafel_V=base.oer_tafel_V,
            cer_i0=base.cer_i0,
            cer_tafel_V=base.cer_tafel_V,
            cer_n=base.cer_n,
            oer_n=base.oer_n,
            max_bubble_fraction=base.max_bubble_fraction,
            temperature_C=self.conditions.temperature_C,
            oer_ea_kj_mol=base.oer_ea_kj_mol,
            references=base.references,
        )
        return AnodeKinetics(
            material=material,
            electrolyte_type="acidic",
            pH=self.bath.pH,
            boundary_layer_m=self.conditions.boundary_layer_m,
            electrolyte_conductivity_S_m=self._conductivity,
            anode_chemistry="inert",
        )

    def _build_voltage_model(
        self, j_mA_cm2: float, cathode_overpotential_V: float
    ) -> CellVoltageModel:
        """Build the cell voltage model with transport-corrected parameters."""
        g = self.geometry
        return CellVoltageModel(
            # ``E_cathode_eq`` is the shared standard-state potential;
            # ``fe2_conc_M`` carries the Pitzer activity correction exactly
            # once inside CellVoltageModel.
            E_cathode_eq=E0_FE,
            eta_cathode=cathode_overpotential_V,
            temperature_C=self.conditions.temperature_C,
            # fe2_conc_M drives the Nernst term (γ≡1 inside CellVoltageModel),
            # so it must receive the speciation ACTIVITY a_Fe2+, not a
            # nominal concentration.  With the Pitzer model (2026-08 default)
            # a_Fe2+ is small (≈0.05) and this term matters; the superseded
            # Davies path smuggled the same effect in via "free" [Fe²⁺].
            fe2_conc_M=self._activity_fe2,
            electrolyte_conductivity_S_m=self._conductivity,
            electrolyte_conductivity_at_temperature=True,
            interelectrode_gap_m=g.interelectrode_gap_m,
            contact_resistance_ohm_m2=g.contact_resistance_ohm_m2,
            bubble_fraction=g.anode_bubble_fraction,
            divided_cell=g.membrane,
            membrane=MembraneModel(R_membrane_ohm_m2=g.membrane_area_resistance_ohm_m2) if g.membrane else None,
            anode=self._build_anode(),
            j_operating_mA_cm2=j_mA_cm2,
        )

    def _enabled_chemistry_features(self) -> Dict[str, bool]:
        """Feature switches actually active for this solver instance."""
        return {
            "surface_state_her": self.conditions.chemistry_enabled("surface_state_her"),
            "fe_pair_correction": self.conditions.chemistry_enabled("fe_pair_correction"),
            "feoh2_film": self.conditions.chemistry_enabled("feoh2_film"),
            "dissolved_oxygen": self.conditions.chemistry_enabled("dissolved_oxygen"),
            "fe3_shuttle": self.conditions.chemistry_enabled("fe3_shuttle"),
        }

    def _chemistry_parasitics_A_m2(self, j_mA_cm2: float) -> tuple[Dict[str, float], Dict[str, Any]]:
        """Current branches not represented inside the Fe/HER film solve.

        The diffusion-layer solver returns a self-consistent Fe/HER split for
        the current it is asked to carry.  Oxygen reduction and Fe³⁺ reduction
        are additional cathodic branches that consume applied galvanostatic
        current first; the Fe/HER film is solved on the remaining current.
        A 95 % cap prevents an uncalibrated side-branch estimate from driving
        the main solve to zero current while preserving the warning in the
        diagnostics.
        """
        target_A_m2 = j_mA_cm2 * 10.0
        raw = {"orr_A_m2": 0.0, "fe3_shuttle_A_m2": 0.0}
        diagnostics: Dict[str, Any] = {
            "chemistry_coupling": self.conditions.chemistry_coupling,
            "enabled_features": self._enabled_chemistry_features(),
            "bath_inventory": self.bath_spec.feature_inventory(),
            "bath_spec": self.bath_spec.to_record(),
        }

        ionic_strength = float(
            self._spec.get("ionic_strength_M", self._spec.get("ionic_strength_molal", 0.0))
        )
        o2_fraction = float(self.bath_spec.dissolved_o2_fraction_sat)
        if self.conditions.chemistry_enabled("dissolved_oxygen") and o2_fraction > 0.0:
            params = self.bath_spec.oxygen_params(
                T_C=self.conditions.temperature_C,
                ionic_strength_M=ionic_strength,
                boundary_layer_m=self.conditions.boundary_layer_m,
            )
            raw["orr_A_m2"] = cathodic_orr_limiting_current_A_m2(params, o2_fraction)
            o2_sat = dissolved_oxygen_solubility_M(
                self.conditions.temperature_C, ionic_strength
            )
            diagnostics["dissolved_oxygen"] = {
                "bulk_M": o2_sat * o2_fraction,
                "saturation_M": o2_sat,
                "fraction_sat": o2_fraction,
                "orr_limiting_current_A_m2_uncapped": raw["orr_A_m2"],
                "fe3_generation_M_s": homogeneous_fe2_oxidation_rate_M_s(params, o2_fraction),
                "basis": "mass-transfer-limited ORR branch subtracted before Fe/HER solve",
            }

        if self.conditions.chemistry_enabled("fe3_shuttle"):
            direct_i = 0.0
            if self.bath_spec.fe3_M > 0.0:
                km = D_FE3_REF_M2_S / self.conditions.boundary_layer_m
                direct_i = FARADAY * km * self.bath_spec.fe3_M * 1000.0

            steady_i = 0.0
            steady_diag: Dict[str, Any] | None = None
            if o2_fraction > 0.0:
                md = dict(self.bath_spec.metadata)
                shuttle_params = ShuttleParams(
                    temperature_C=self.conditions.temperature_C,
                    pH=self.bath.pH,
                    fe2_M=self.bath_spec.fe2_total_M,
                    cathode_area_m2=float(md.get("cathode_area_m2", 1.0e-3)),
                    catholyte_volume_L=float(md.get("catholyte_volume_L", 0.5)),
                    boundary_layer_m=self.conditions.boundary_layer_m,
                )
                scenario = ShuttleScenario("bath_spec_dissolved_o2", o2_fraction)
                steady_diag = fe3_shuttle_steady_state(shuttle_params, scenario)
                steady_i = float(steady_diag["i_shuttle_A_m2"])

            # If a measured Fe³⁺ inventory is present, use it; otherwise use the
            # steady-state O₂-generation estimate as a screening branch.
            raw["fe3_shuttle_A_m2"] = direct_i if self.bath_spec.fe3_M > 0.0 else steady_i
            diagnostics["fe3_shuttle"] = {
                "direct_from_measured_fe3_A_m2": direct_i,
                "steady_state_from_o2_A_m2": steady_i,
                "used_A_m2_uncapped": raw["fe3_shuttle_A_m2"],
                "steady_state_diagnostics": steady_diag,
                "basis": "Fe3+ + e- -> Fe2+ mass-transfer-limited shuttle",
            }

        raw_total = sum(raw.values())
        cap_A_m2 = 0.95 * target_A_m2
        scale = 1.0 if raw_total <= cap_A_m2 or raw_total <= 0.0 else cap_A_m2 / raw_total
        currents = {k: v * scale for k, v in raw.items()}
        diagnostics["parasitic_current_uncapped_A_m2"] = raw_total
        diagnostics["parasitic_current_cap_A_m2"] = cap_A_m2
        diagnostics["parasitic_current_scale"] = scale
        diagnostics["parasitic_current_capped"] = bool(scale < 1.0)
        return currents, diagnostics

    def solve_at_j(self, j_mA_cm2: float) -> OperatingPoint:
        """
        Solve the full physics at one applied current density.

        In legacy mode the applied current is the Fe/HER film current, matching
        the historical solver.  In chemistry-rich mode the shared ``BathSpec``
        first estimates additional cathodic branches (ORR and Fe³⁺ shuttle),
        subtracts them from the galvanostatic current, and then solves the
        Fe/HER diffusion layer on the remaining current.  The reported
        ``current_efficiency`` is always Fe current divided by the *applied*
        current.
        """
        if j_mA_cm2 <= 0.0:
            raise ValueError("j_mA_cm2 must be positive")

        applied_A_m2 = j_mA_cm2 * 10.0
        parasitic, chemistry_diag = self._chemistry_parasitics_A_m2(j_mA_cm2)
        orr_A_m2 = float(parasitic.get("orr_A_m2", 0.0))
        fe3_A_m2 = float(parasitic.get("fe3_shuttle_A_m2", 0.0))
        parasitic_A_m2 = orr_A_m2 + fe3_A_m2
        fe_her_target_A_m2 = max(applied_A_m2 - parasitic_A_m2, 1.0e-9)
        fe_her_target_mA_cm2 = fe_her_target_A_m2 / 10.0

        transport = self._build_transport()
        transport_result = transport.solve(fe_her_target_mA_cm2)

        if isinstance(transport_result, DiffusionLayerResult):
            # The reactive film solver carries the activity-corrected
            # equilibrium and the richer local chemistry itself.
            potential_V = transport_result.V_cathode_V
            transport_fe = transport_result.current_efficiency
            fe_current_A_m2 = transport_result.fe_current_A_m2
            her_current_A_m2 = transport_result.her_current_A_m2
            surface_pH = transport_result.surface_pH
            surface_fe_M = transport_result.surface_fe_M
            transport_limit_A_m2 = transport_result.transport_limit_A_m2
            diffusion_limit_A_m2 = transport_result.diffusion_limit_A_m2
            film_potential_drop_V = transport_result.film_potential_drop_V
            supersaturation = transport_result.feoh2_supersaturation
            precipitation_active = transport_result.precipitation_active
            converged = transport_result.converged
        else:
            np_state: NernstPlanckState = transport_result
            potential_V = np_state.potential_V
            transport_fe = np_state.current_efficiency
            fe_current_A_m2 = np_state.fe_current_A_m2
            her_current_A_m2 = np_state.her_current_A_m2
            surface_pH = np_state.surface_pH
            surface_fe_M = np_state.surface_fe_M
            transport_limit_A_m2 = np_state.transport_limit_A_m2
            diffusion_limit_A_m2 = np_state.diffusion_limit_A_m2
            film_potential_drop_V = np_state.film_potential_drop_V
            supersaturation = np_state.feoh2_supersaturation
            precipitation_active = np_state.precipitation_active
            converged = np_state.converged

        # Fast film solves can miss the galvanostatic target by ~0.1 %.  Keep
        # the solved potential/surface chemistry, but rescale the Fe/HER
        # branch ledger to the exact current left after ORR/Fe³⁺ parasitics so
        # the operating-point charge accounting always closes.
        solved_fe_her_A_m2 = fe_current_A_m2 + her_current_A_m2
        branch_rescale = 1.0
        if solved_fe_her_A_m2 > 0.0:
            branch_rescale = fe_her_target_A_m2 / solved_fe_her_A_m2
            fe_current_A_m2 *= branch_rescale
            her_current_A_m2 *= branch_rescale

        # Cathode overpotential from the same bulk activity used by the
        # voltage model.  The Fe(OH)₂ passivation-film resistance is a cathodic
        # overpotential increment and therefore enters the same voltage model
        # term, preserving decomposition closure.
        E_cathode_eq = self._spec.get("E_rev_Fe_V_SHE", E0_FE)
        film_eta_V = float(getattr(transport_result, "feoh2_film_overpotential_V", 0.0))
        eta_cathode = max(E_cathode_eq - potential_V, 0.0) + film_eta_V

        # Build voltage model with physics-derived overpotential and the full
        # applied current for IR losses.
        vm = self._build_voltage_model(j_mA_cm2, eta_cathode)
        V_cell = vm.V_cell
        V_decomp = vm.V_decomposition

        # Applied-current FE after O₂/Fe³⁺ parasitics.
        fe = fe_current_A_m2 / max(applied_A_m2, 1e-30)
        fe = float(max(0.0, min(1.0, fe)))

        # Deposition rate from the actual Fe partial current.
        mass_flux = fe_current_A_m2 * M_FE / (Z_FE * FARADAY)  # kg/(m²·s)
        rho = 7874.0  # kg/m³
        dep_rate = mass_flux / rho * 3600.0 * 1e6  # µm/hr

        # Precipitation sink from the reactive film (if available)
        precip_flux = getattr(transport_result, "precipitation_flux_mol_m2_s", 0.0)
        precip_frac = getattr(transport_result, "precipitation_fraction", 0.0)
        sludge_g = getattr(transport_result, "sludge_rate_g_m2_s", 0.0)
        # Net deposition after subtracting Fe that precipitates as sludge.
        fe_mol_flux_total = fe_current_A_m2 / (Z_FE * FARADAY)
        fe_mol_flux_net = max(fe_mol_flux_total - precip_flux, 0.0)
        dep_rate_net = fe_mol_flux_net * (55.845e-3) / 7874.0 * 3600.0 * 1e6

        current_breakdown = {
            "applied_A_m2": float(applied_A_m2),
            "Fe_deposition_A_m2": float(fe_current_A_m2),
            "HER_A_m2": float(her_current_A_m2),
            "ORR_A_m2": float(orr_A_m2),
            "Fe3_shuttle_A_m2": float(fe3_A_m2),
        }
        assigned = sum(v for k, v in current_breakdown.items() if k != "applied_A_m2")
        current_breakdown["unassigned_A_m2"] = float(applied_A_m2 - assigned)
        current_fraction = {
            k.replace("_A_m2", "_fraction"): float(v / max(applied_A_m2, 1e-30))
            for k, v in current_breakdown.items()
            if k != "applied_A_m2"
        }
        chemistry_diag["transport_core_current_mA_cm2"] = fe_her_target_mA_cm2
        chemistry_diag["transport_branch_current_rescale"] = float(branch_rescale)
        chemistry_diag["transport_current_efficiency_Fe_over_FeHER"] = float(transport_fe)
        chemistry_diag["reactive_film"] = {
            "her_i0_surface_state_ratio": float(
                getattr(transport_result, "her_i0_surface_state_ratio", 1.0)
            ),
            "fe_pair_fraction_bulk": float(getattr(transport_result, "fe_pair_fraction_bulk", 0.0)),
            "feoh2_film_overpotential_V": film_eta_V,
            "surface_state_bath_type": self.bath_spec.surface_state_bath_type(),
        }

        return OperatingPoint(
            j_mA_cm2=j_mA_cm2,
            current_efficiency=fe,
            surface_pH=surface_pH,
            surface_fe_M=surface_fe_M,
            transport_limit_mA_cm2=transport_limit_A_m2 / 10.0,
            diffusion_limit_mA_cm2=diffusion_limit_A_m2 / 10.0,
            migration_enhancement=transport_limit_A_m2 / max(diffusion_limit_A_m2, 1e-30),
            feoh2_supersaturation=supersaturation,
            film_potential_drop_V=film_potential_drop_V,
            precipitation_active=precipitation_active,
            precipitation_flux_mol_m2_s=float(precip_flux),
            precipitation_fraction=float(precip_frac),
            sludge_rate_g_m2_s=float(sludge_g),
            V_cell=V_cell,
            V_decomposition=V_decomp,
            specific_energy_kWh_t=specific_energy_kWh_per_t(V_cell, fe),
            deposition_rate_um_hr=dep_rate,
            deposition_rate_net_um_hr=float(dep_rate_net),
            free_fe2_activity=self._activity_fe2,
            conductivity_S_m=self._conductivity,
            speciation=self._spec,
            transport_converged=converged,
            applied_current_A_m2=float(applied_A_m2),
            transport_current_efficiency=float(transport_fe),
            fe_current_A_m2=float(fe_current_A_m2),
            her_current_A_m2=float(her_current_A_m2),
            orr_current_A_m2=float(orr_A_m2),
            fe3_shuttle_current_A_m2=float(fe3_A_m2),
            parasitic_current_A_m2=float(parasitic_A_m2),
            current_breakdown_A_m2=current_breakdown,
            current_breakdown_fraction=current_fraction,
            chemistry_diagnostics=chemistry_diag,
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

        Uses a fast kinetics model (DepositionKinetics) for the coarse
        pre-screen, then validates with the configured reactive/dilute film
        solver at the optimal point.  The coarse model is intentionally a
        search acceleration only; reported operating points always come from
        the configured film model.
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

        # Validate with full Nernst-Planck at the optimal point.  In chemistry-
        # rich mode the validation may include ORR/Fe³⁺ parasitics that the
        # coarse Fe/HER-only pre-screen did not see, so re-check the FE floor.
        try:
            pt = self.solve_at_j(best_j)
            return pt if pt.current_efficiency >= min_FE else None
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
                "FE applied-current (%)": round(pt.current_efficiency * 100, 1),
                "FE Fe/(Fe+HER) before parasitics (%)": round(pt.transport_current_efficiency * 100, 1),
                "V_cell (V)": round(pt.V_cell, 3),
                "Energy (kWh/t)": round(pt.specific_energy_kWh_t, 0),
                "Deposition (µm/hr)": round(pt.deposition_rate_um_hr, 1),
            },
            "Current breakdown (A/m²)": {
                k: round(v, 6) for k, v in (pt.current_breakdown_A_m2 or {}).items()
            },
            "Chemistry coupling": {
                "mode": self.conditions.chemistry_coupling,
                "enabled": pt.chemistry_diagnostics.get("enabled_features", {}) if pt.chemistry_diagnostics else {},
                "surface-state bath type": (
                    pt.chemistry_diagnostics.get("reactive_film", {}).get("surface_state_bath_type")
                    if pt.chemistry_diagnostics else None
                ),
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
