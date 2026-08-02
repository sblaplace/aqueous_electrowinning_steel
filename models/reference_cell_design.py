"""Reference-cell design synthesis for the deployable RC-1 apparatus.

This module turns the controlled ``processes/reference_cell_rc1.yaml`` input
into an executable design calculation.  It combines the existing cell-physics
prediction with geometry-derived hydraulic, electrical, thermal, and gas-load
checks, then ranks candidate area/channel/flow choices against the declared
hardware constraints.

It is deliberately a *design* tool, not a second digital twin.  The selected
configuration is the same configuration passed to :mod:`twin_physics`,
:mod:`digital_twin`, and :mod:`operating_twin` once the apparatus operates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import math

import yaml

from .cell_physics import BathRecipe, CellGeometry, CellPhysics, ProcessConditions
from .electrochemistry import FARADAY

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "processes" / "reference_cell_rc1.yaml"
RHO_ELECTROLYTE_KG_M3 = 1_050.0  # conservative sulfate-bath engineering estimate
R_GAS_J_MOL_K = 8.314462
T_STANDARD_K = 298.15
THERMONEUTRAL_FE_V = 1.28


@dataclass(frozen=True)
class ReferenceCellConfig:
    """Controlled RC-1 design input parsed from YAML."""

    configuration_id: str
    active_area_cm2: float
    channel_length_m: float
    channel_width_m: float
    nominal_channel_depth_m: float
    catholyte_volume_L: float
    anolyte_volume_L: float
    flow_range_L_min: tuple[float, float]
    max_current_A: float
    max_current_density_mA_cm2: float
    max_voltage_V: float
    temperature_range_C: tuple[float, float]
    target_temperature_C: float
    max_pressure_drop_Pa: float
    max_hydrogen_design_rate_L_h: float
    bath: BathRecipe
    conditions: ProcessConditions
    geometry: CellGeometry
    candidate_active_areas_cm2: tuple[float, ...]
    candidate_channel_depths_m: tuple[float, ...]
    candidate_flows_L_min: tuple[float, ...]

    @property
    def total_volume_L(self) -> float:
        return self.catholyte_volume_L + self.anolyte_volume_L

    def operating_current_A(self, current_density_mA_cm2: float) -> float:
        return current_density_mA_cm2 * self.active_area_cm2 / 1_000.0


def _tuple_of_floats(value: Iterable[Any], name: str) -> tuple[float, ...]:
    result = tuple(float(v) for v in value)
    if not result or any(v <= 0 for v in result):
        raise ValueError(f"{name} must contain positive values")
    return result


def load_reference_cell_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ReferenceCellConfig:
    """Load and validate the controlled RC-1 input file.

    The parser intentionally validates the values that set the deployable
    hardware envelope.  It does not infer missing geometry or silently use a
    generic pilot-scale twin default.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("configuration_id") != "RC-1":
        raise ValueError("reference-cell configuration must identify RC-1")

    cathode = raw["cell_stack"]["cathode"]
    channels = raw["cell_stack"]["channels"]["cathode"]
    bop = raw["balance_of_plant"]
    electrical = bop["electrical"]
    constraints = bop["design_constraints"]
    chemistry = raw["chemistry"]
    bath_raw = chemistry["bath"]
    transport = chemistry["transport"]
    kinetics = chemistry["kinetics"]

    area = float(cathode["active_area_cm2"])
    calculated_area = float(cathode["active_length_mm"]) * float(cathode["active_width_mm"]) / 100.0
    if not math.isclose(area, calculated_area, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("cathode active_area_cm2 must equal active_length_mm × active_width_mm / 100")

    temp_range = _tuple_of_floats(bop["thermal"]["qualified_temperature_range_C"], "temperature range")
    if len(temp_range) != 2 or temp_range[0] >= temp_range[1]:
        raise ValueError("qualified_temperature_range_C must be an ordered [min, max]")
    flow_range = _tuple_of_floats(bop["recirculation"]["installed_flow_range_L_min"], "flow range")
    if len(flow_range) != 2 or flow_range[0] >= flow_range[1]:
        raise ValueError("installed_flow_range_L_min must be an ordered [min, max]")

    max_current = float(electrical["normal_current_ceiling_A"])
    max_j = float(electrical["current_density_ceiling_mA_cm2"])
    if max_current <= 0 or max_j <= 0:
        raise ValueError("current limits must be positive")
    if max_j * area / 1_000.0 > max_current + 1e-12:
        raise ValueError("configured active area/current-density ceiling exceeds normal current ceiling")

    target_T = float(constraints["target_temperature_C"])
    if not temp_range[0] <= target_T <= temp_range[1]:
        raise ValueError("target temperature must lie in the qualified range")

    bath = BathRecipe(
        c_FeSO4_M=float(bath_raw["fe_so4_M"]),
        c_Na2SO4_M=float(bath_raw["na2_so4_M"]),
        c_H2SO4_M=float(bath_raw["h2_so4_M"]),
        c_H3BO3_M=float(bath_raw["h3_bo3_M"]),
        pH=float(bath_raw["bulk_pH"]),
    )
    conditions = ProcessConditions(
        temperature_C=target_T,
        boundary_layer_m=float(transport["nominal_boundary_layer_um"]) * 1e-6,
        fe_i0=float(kinetics["fe_i0_A_m2"]),
        her_i0=float(kinetics["her_i0_A_m2"]),
        fe_tafel_V=float(kinetics["fe_tafel_V_dec"]),
        her_tafel_V=float(kinetics["her_tafel_V_dec"]),
    )
    geometry = CellGeometry(interelectrode_gap_m=float(channels["depth_mm"]) * 1e-3)

    return ReferenceCellConfig(
        configuration_id=raw["configuration_id"],
        active_area_cm2=area,
        channel_length_m=float(channels["length_mm"]) * 1e-3,
        channel_width_m=float(channels["width_mm"]) * 1e-3,
        nominal_channel_depth_m=float(channels["depth_mm"]) * 1e-3,
        catholyte_volume_L=float(bop["catholyte_volume_L"]),
        anolyte_volume_L=float(bop["anolyte_volume_L"]),
        flow_range_L_min=(flow_range[0], flow_range[1]),
        max_current_A=max_current,
        max_current_density_mA_cm2=max_j,
        max_voltage_V=float(electrical["conservative_voltage_hard_ceiling_V"]),
        temperature_range_C=(temp_range[0], temp_range[1]),
        target_temperature_C=target_T,
        max_pressure_drop_Pa=float(constraints["maximum_channel_pressure_drop_kPa"]) * 1_000.0,
        max_hydrogen_design_rate_L_h=float(bop["gas"]["maximum_hydrogen_design_rate_L_h_at_25C_1atm"]),
        bath=bath,
        conditions=conditions,
        geometry=geometry,
        candidate_active_areas_cm2=_tuple_of_floats(constraints["candidate_active_areas_cm2"], "candidate areas"),
        candidate_channel_depths_m=tuple(v * 1e-3 for v in _tuple_of_floats(
            constraints["candidate_channel_depths_mm"], "candidate channel depths")),
        candidate_flows_L_min=_tuple_of_floats(constraints["candidate_flow_rates_L_min"], "candidate flows"),
    )


@dataclass(frozen=True)
class CandidateDesign:
    active_area_cm2: float
    channel_depth_m: float
    flow_L_min: float


@dataclass(frozen=True)
class DesignEvaluation:
    candidate: CandidateDesign
    current_A: float
    current_density_mA_cm2: float
    cell_voltage_V: float
    faradaic_efficiency: float
    specific_energy_kWh_t: float
    deposit_rate_um_hr: float
    superficial_velocity_m_s: float
    reynolds_number: float
    pressure_drop_Pa: float
    channel_residence_time_s: float
    heat_generation_W: float
    h2_rate_L_h: float
    h2_design_rate_L_h: float
    feasible: bool
    failures: tuple[str, ...]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": {
                "active_area_cm2": self.candidate.active_area_cm2,
                "channel_depth_mm": self.candidate.channel_depth_m * 1e3,
                "flow_L_min": self.candidate.flow_L_min,
            },
            "operating": {
                "current_A": self.current_A,
                "current_density_mA_cm2": self.current_density_mA_cm2,
                "cell_voltage_V": self.cell_voltage_V,
                "faradaic_efficiency": self.faradaic_efficiency,
                "specific_energy_kWh_t": self.specific_energy_kWh_t,
                "deposit_rate_um_hr": self.deposit_rate_um_hr,
            },
            "hydraulics": {
                "superficial_velocity_m_s": self.superficial_velocity_m_s,
                "reynolds_number": self.reynolds_number,
                "pressure_drop_Pa": self.pressure_drop_Pa,
                "channel_residence_time_s": self.channel_residence_time_s,
            },
            "utilities_and_gas": {
                "heat_generation_W": self.heat_generation_W,
                "h2_rate_L_h": self.h2_rate_L_h,
                "h2_design_rate_L_h": self.h2_design_rate_L_h,
            },
            "feasible": self.feasible,
            "failures": list(self.failures),
            "score": self.score,
        }


def dynamic_viscosity_Pa_s(temperature_C: float) -> float:
    """Water-viscosity proxy used only for early hydraulic sizing.

    Andrade's common water correlation uses a base-10 exponent and an offset
    temperature.  It gives about 0.47 mPa·s at 60 °C; using ``exp(247.8/T)``
    would understate viscosity by roughly an order of magnitude and falsely
    label the RC-1 channel as turbulent.
    """
    T_K = temperature_C + 273.15
    return 2.414e-5 * 10.0 ** (247.8 / (T_K - 140.0))


def channel_hydraulics(config: ReferenceCellConfig, candidate: CandidateDesign) -> Dict[str, float]:
    """Calculate rectangular-channel flow quantities for one cathode channel.

    The pressure-drop relation is the fully developed laminar rectangular
    channel approximation.  It is a sizing check; fittings, manifolds,
    bubbles, and deposit growth are deliberately excluded and remain a design
    margin / future CFD item.
    """
    w, h, L = config.channel_width_m, candidate.channel_depth_m, config.channel_length_m
    q_m3_s = candidate.flow_L_min / 60_000.0
    area = w * h
    velocity = q_m3_s / area
    hydraulic_diameter = 2.0 * w * h / (w + h)
    mu = dynamic_viscosity_Pa_s(config.target_temperature_C)
    reynolds = RHO_ELECTROLYTE_KG_M3 * velocity * hydraulic_diameter / mu
    # Correction is valid when h <= w; all configured RC-1 candidates meet it.
    correction = 1.0 - 0.63 * h / w
    if correction <= 0:
        raise ValueError("channel geometry is outside rectangular-channel approximation")
    pressure = 12.0 * mu * L * q_m3_s / (w * h ** 3 * correction)
    residence = L * w * h / q_m3_s
    return {
        "superficial_velocity_m_s": velocity,
        "hydraulic_diameter_m": hydraulic_diameter,
        "reynolds_number": reynolds,
        "pressure_drop_Pa": pressure,
        "channel_residence_time_s": residence,
    }


def hydrogen_rate_L_h(current_A: float, her_fraction: float) -> float:
    """Cathode hydrogen rate at 25 °C/1 atm from the HER current fraction."""
    moles_per_s = max(0.0, current_A * her_fraction) / (2.0 * FARADAY)
    return moles_per_s * R_GAS_J_MOL_K * T_STANDARD_K / 101_325.0 * 1_000.0 * 3_600.0


def evaluate_design(
    config: ReferenceCellConfig,
    candidate: CandidateDesign,
    current_density_mA_cm2: Optional[float] = None,
    operating_point: Any = None,
) -> DesignEvaluation:
    """Evaluate a candidate at the required design current density.

    ``operating_point`` supports design-space reuse: area and flow do not
    change the present cell-physics solve at fixed current density and channel
    gap, so the synthesis routine evaluates each unique gap once.
    """
    j = config.max_current_density_mA_cm2 if current_density_mA_cm2 is None else current_density_mA_cm2
    current_A = j * candidate.active_area_cm2 / 1_000.0
    if operating_point is None:
        geometry = replace(config.geometry, interelectrode_gap_m=candidate.channel_depth_m)
        physics = CellPhysics(config.bath, geometry, config.conditions)
        point = physics.solve_at_j(j)
    else:
        point = operating_point
    hydraulic = channel_hydraulics(config, candidate)
    h2_actual = hydrogen_rate_L_h(current_A, 1.0 - point.current_efficiency)
    h2_design = hydrogen_rate_L_h(current_A, 1.0)
    heat = max(0.0, current_A * (point.V_cell - THERMONEUTRAL_FE_V))

    failures: List[str] = []
    if current_A > config.max_current_A + 1e-12:
        failures.append("current_limit")
    if j > config.max_current_density_mA_cm2 + 1e-12:
        failures.append("current_density_limit")
    if point.V_cell > config.max_voltage_V:
        failures.append("voltage_limit")
    if hydraulic["pressure_drop_Pa"] > config.max_pressure_drop_Pa:
        failures.append("pressure_drop_limit")
    if hydraulic["reynolds_number"] > 2_100.0:
        failures.append("hydraulic_regime_outside_laminar_model")
    # The controlled YAML declares the all-HER design limit at 25 °C / 1 atm.
    if h2_design > config.max_hydrogen_design_rate_L_h + 1e-9:
        failures.append("hydrogen_design_limit")

    # Prefer the declared baseline where feasible; the search explores useful
    # alternatives without silently redefining the controlled RC-1 intent.
    area_penalty = abs(candidate.active_area_cm2 - config.active_area_cm2) / config.active_area_cm2
    depth_penalty = abs(candidate.channel_depth_m - config.nominal_channel_depth_m) / config.nominal_channel_depth_m
    flow_target = sum(config.flow_range_L_min) / 2.0
    flow_penalty = abs(candidate.flow_L_min - flow_target) / flow_target
    utility_penalty = hydraulic["pressure_drop_Pa"] / config.max_pressure_drop_Pa + heat / 15.0
    score = 100.0 * area_penalty + 20.0 * depth_penalty + 5.0 * flow_penalty + utility_penalty

    return DesignEvaluation(
        candidate=candidate,
        current_A=current_A,
        current_density_mA_cm2=j,
        cell_voltage_V=point.V_cell,
        faradaic_efficiency=point.current_efficiency,
        specific_energy_kWh_t=point.specific_energy_kWh_t,
        deposit_rate_um_hr=point.deposition_rate_um_hr,
        superficial_velocity_m_s=hydraulic["superficial_velocity_m_s"],
        reynolds_number=hydraulic["reynolds_number"],
        pressure_drop_Pa=hydraulic["pressure_drop_Pa"],
        channel_residence_time_s=hydraulic["channel_residence_time_s"],
        heat_generation_W=heat,
        h2_rate_L_h=h2_actual,
        h2_design_rate_L_h=h2_design,
        feasible=not failures,
        failures=tuple(failures),
        score=score,
    )


def build_reference_cell_process_model(
    config: ReferenceCellConfig,
    cache_path: Optional[str] = None,
):
    """Build the physics surrogate on the RC-1, 300 mA/cm² design envelope.

    Imports lazily so a geometry/BOM review can use the lightweight hydraulic
    helpers without forcing an offline surrogate build.  The grid explicitly
    includes the program's 300 mA/cm² decision duty; the generic twin's 250
    mA/cm² ceiling is not used for RC-1.
    """
    from .twin_physics import CellProcessModel

    return CellProcessModel(
        bath=config.bath,
        geometry=config.geometry,
        conditions=config.conditions,
        j_grid=(50.0, 100.0, 200.0, 300.0),
        T_grid=(config.temperature_range_C[0], config.target_temperature_C, config.temperature_range_C[1]),
        fe2_grid=(0.5, 1.0, 1.5),
        cache_path=cache_path,
    )


def build_reference_cell_digital_twin(config: ReferenceCellConfig, model=None):
    """Create the existing EKF twin with RC-1 dimensions and inventory.

    This is the bridge from design synthesis to runtime estimation: a 3 A
    reading maps to 300 mA/cm² on a 10 cm² coupon, rather than to the generic
    pilot-scale twin's one-square-metre default.
    """
    from .digital_twin import DigitalTwin

    flow_L_hr = sum(config.flow_range_L_min) / 2.0 * 60.0
    design_point = {
        "temperature_C": config.target_temperature_C,
        "pH": config.bath.pH,
        "j_avg_mA_cm2": config.max_current_density_mA_cm2,
        "electrode_area_m2": config.active_area_cm2 / 10_000.0,
        "electrolyte_volume_L": config.total_volume_L,
        "fe2_M": config.bath.c_FeSO4_M,
        "recirculation_flow_L_hr": flow_L_hr,
        "reservoir_volume_L": config.catholyte_volume_L,
        "catholyte_volume_L": config.catholyte_volume_L,
        "anolyte_volume_L": config.anolyte_volume_L,
        "fe2_reservoir_M": config.bath.c_FeSO4_M,
        "pH_reservoir": config.bath.pH,
        "T_reservoir_C": config.target_temperature_C,
        "buffer_capacity_beta": config.bath.c_H3BO3_M,
    }
    return DigitalTwin(design_point=design_point, model=model)


def build_reference_cell_operating_twin(config: ReferenceCellConfig):
    """Create the advisory safety twin with the RC-1 hardware envelope.

    The returned twin remains in advisory mode; this function does not arm
    hardware control or substitute for the independent shutdown channel.
    """
    from .operating_twin import OperatingTwin, TwinConfig

    return OperatingTwin(TwinConfig(
        cell_id=config.configuration_id,
        max_current_A=config.max_current_A,
        max_current_density_mA_cm2=config.max_current_density_mA_cm2,
        max_voltage_V=config.max_voltage_V,
        min_temperature_C=config.temperature_range_C[0],
        max_temperature_C=config.temperature_range_C[1],
        min_fe2_M=0.2,
        max_fe2_M=2.0,
        min_pH=0.5,
        max_pH=5.0,
        target_current_A=config.max_current_A,
        target_temperature_C=config.target_temperature_C,
    ))


def enumerate_candidates(config: ReferenceCellConfig) -> List[CandidateDesign]:
    """Return the declared RC-1 design-space candidates inside installed flow range."""
    return [
        CandidateDesign(area, depth, flow)
        for area in config.candidate_active_areas_cm2
        for depth in config.candidate_channel_depths_m
        for flow in config.candidate_flows_L_min
        if config.flow_range_L_min[0] <= flow <= config.flow_range_L_min[1]
    ]


def synthesize_reference_cell_design(
    config: ReferenceCellConfig,
    candidates: Optional[Iterable[CandidateDesign]] = None,
) -> Dict[str, Any]:
    """Evaluate candidates and return the selected deployable RC-1 design."""
    candidate_list = list(candidates) if candidates is not None else enumerate_candidates(config)
    points_by_depth = {}
    for depth in {c.channel_depth_m for c in candidate_list}:
        geometry = replace(config.geometry, interelectrode_gap_m=depth)
        points_by_depth[depth] = CellPhysics(config.bath, geometry, config.conditions).solve_at_j(
            config.max_current_density_mA_cm2
        )
    evaluations = [
        evaluate_design(config, c, operating_point=points_by_depth[c.channel_depth_m])
        for c in candidate_list
    ]
    feasible = [e for e in evaluations if e.feasible]
    if not feasible:
        raise RuntimeError("no RC-1 candidate satisfies declared electrical, hydraulic, and gas constraints")
    selected = min(feasible, key=lambda e: e.score)
    return {
        "configuration_id": config.configuration_id,
        "design_status": "screening_design_for_procurement",
        "target": {
            "current_density_mA_cm2": config.max_current_density_mA_cm2,
            "temperature_C": config.target_temperature_C,
            "total_electrolyte_volume_L": config.total_volume_L,
        },
        "selected_design": selected.to_dict(),
        "candidate_count": len(evaluations),
        "feasible_candidate_count": len(feasible),
        "candidate_evaluations": [e.to_dict() for e in sorted(evaluations, key=lambda e: e.score)],
        "design_boundaries": [
            "Hydraulics are single-phase rectangular-channel sizing only; fittings, gas hold-up, manifolds, and deposit growth require verification.",
            "Electrochemical outputs use the repository screening physics and configured kinetic constants.",
            "The selected configuration is a deployable RC-1 reference cell, not a production-architecture selection.",
        ],
    }
