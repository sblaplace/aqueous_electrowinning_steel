"""Canonical RC-1 reference-cell integration boundary.

This module is the deliberately small seam between the repository's model
families.  It does not replace any of them.  It gives them one typed state and
one explicit data flow::

    RC-1 inputs
        -> CellPhysics (speciation + transport + voltage)
        -> gas_holdup.solve_coupled (H2 -> void fraction -> current -> FE)
        -> thermal_balance (heat and cooling duty)
        -> charge / iron / energy screening ledgers
        -> OperatingTwin (advisory safety boundary)

A measured run enters through ``run_record.build_qa_report`` and follows the
same state shape.  Measured values remain separate from predictions; model
predictions never become gate evidence.  The pipeline computes residuals and
reports calibration readiness, but it never silently fits or overwrites model
parameters.

Status
------
The integrated result is **screening level L0** until a real RC-1 dataset is
used to calibrate and validate it.  ``OperatingTwin`` is connected in
advisory/replay mode only.  The twin can emit a shutdown *request*; an
independent hardwired safety channel must execute any physical shutdown.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .cell_physics import BathRecipe, CellGeometry, CellPhysics, OperatingPoint, ProcessConditions
from .electrochemistry import FARADAY, M_FE_G, Z_FE, specific_energy_kWh_per_t
from .gas_holdup import ChannelGeometry, CoupledGasResult, solve_coupled
from .operating_twin import ControlCommand, OperatingTwin, SensorSnapshot, ShutdownRequest, TwinState
from .process_gates import EvidenceRecord, evaluate_all
from .reference_cell_design import (
    DEFAULT_CONFIG_PATH,
    ReferenceCellConfig,
    build_reference_cell_operating_twin,
    load_reference_cell_config,
)
from .run_record import DataContractError, RunRecord, build_qa_report, load_run_record
from .thermal_balance import CellThermalParams, simulate_thermal_transient


PIPELINE_CONTRACT = "aqueous-electrowinning.reference-cell-state"
PIPELINE_SCHEMA_VERSION = "1.0"
MODEL_LEVEL = "L0"
H3BO3_MOLAR_MASS_G_MOL = 61.83

# A run-record can omit these fields because they are optional in the current
# contract.  The integration layer uses these conservative fallbacks rather
# than inventing a measured value.
DEFAULT_AMBIENT_TEMPERATURE_C = 25.0
DEFAULT_RELATIVE_HUMIDITY = 0.50
DEFAULT_THERMAL_DURATION_HR = 1.0
DEFAULT_THERMAL_DT_S = 10.0
DEFAULT_RC1_FLOW_L_MIN = 0.25


# ---------------------------------------------------------------------------
# JSON and numeric helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:  # noqa: C901 - strict JSON boundary handles several scalar families
    """Convert dataclasses/numpy values to strict-JSON-compatible values.

    ``allow_nan=False`` is used when reports are written.  Non-finite model
    values are represented as ``None`` so a failed or unbounded screening
    branch cannot silently produce invalid JSON.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "value") and not isinstance(value, Mapping):
        # Enum values, including TwinMode.
        try:
            return _jsonable(value.value)
        except AttributeError:
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except TypeError:
            pass
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _finite_float(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return value if math.isfinite(value) else float(fallback)


def _mean_column(frame: Any, column: str, fallback: float) -> float:
    if frame is None or column not in frame:
        return float(fallback)
    values = np.asarray(frame[column], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float(fallback)


# ---------------------------------------------------------------------------
# Canonical state types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceCellInputs:
    """All runtime inputs needed by the integrated reference-cell state."""

    configuration_id: str
    active_area_cm2: float
    current_density_mA_cm2: float
    flow_L_min: float
    temperature_C: float
    ambient_temperature_C: float = DEFAULT_AMBIENT_TEMPERATURE_C
    initial_temperature_C: float | None = None
    relative_humidity: float = DEFAULT_RELATIVE_HUMIDITY
    thermal_duration_hr: float = DEFAULT_THERMAL_DURATION_HR
    thermal_dt_s: float = DEFAULT_THERMAL_DT_S
    cooling_active: bool = False

    def __post_init__(self) -> None:
        if not self.configuration_id.strip():
            raise ValueError("configuration_id is required")
        positive = (
            self.active_area_cm2,
            self.current_density_mA_cm2,
            self.flow_L_min,
            self.thermal_duration_hr,
            self.thermal_dt_s,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("active area, current density, flow, and thermal settings must be positive")
        temperatures = (
            self.temperature_C,
            self.ambient_temperature_C,
            self.initial_temperature_C,
        )
        if any(value is not None and not math.isfinite(float(value)) for value in temperatures):
            raise ValueError("temperatures must be finite")
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError("relative_humidity must lie in [0, 1]")

    @property
    def current_A(self) -> float:
        """Total current corresponding to the area and current density."""
        return self.current_density_mA_cm2 * self.active_area_cm2 / 1_000.0

    @classmethod
    def from_config(
        cls,
        config: ReferenceCellConfig,
        *,
        current_density_mA_cm2: float | None = None,
        flow_L_min: float | None = None,
        temperature_C: float | None = None,
        ambient_temperature_C: float = DEFAULT_AMBIENT_TEMPERATURE_C,
        initial_temperature_C: float | None = None,
        relative_humidity: float = DEFAULT_RELATIVE_HUMIDITY,
        thermal_duration_hr: float = DEFAULT_THERMAL_DURATION_HR,
        thermal_dt_s: float = DEFAULT_THERMAL_DT_S,
        cooling_active: bool = False,
    ) -> "ReferenceCellInputs":
        """Construct runtime inputs from the controlled RC-1 YAML basis."""
        nominal_flow = DEFAULT_RC1_FLOW_L_MIN
        if not config.flow_range_L_min[0] <= nominal_flow <= config.flow_range_L_min[1]:
            nominal_flow = float(sum(config.flow_range_L_min) / 2.0)
        flow = float(flow_L_min) if flow_L_min is not None else nominal_flow
        return cls(
            configuration_id=config.configuration_id,
            active_area_cm2=float(config.active_area_cm2),
            current_density_mA_cm2=(
                float(current_density_mA_cm2)
                if current_density_mA_cm2 is not None
                else float(config.max_current_density_mA_cm2)
            ),
            flow_L_min=flow,
            temperature_C=(
                float(temperature_C)
                if temperature_C is not None
                else float(config.target_temperature_C)
            ),
            ambient_temperature_C=float(ambient_temperature_C),
            initial_temperature_C=initial_temperature_C,
            relative_humidity=float(relative_humidity),
            thermal_duration_hr=float(thermal_duration_hr),
            thermal_dt_s=float(thermal_dt_s),
            cooling_active=bool(cooling_active),
        )

    def to_dict(self) -> dict[str, Any]:
        result = _jsonable(asdict(self))
        result["current_A"] = self.current_A
        return result


@dataclass(frozen=True)
class PredictedCellState:
    """Coupled model result; explicitly marked as screening prediction."""

    status: str
    operating: Mapping[str, Any]
    gas: Mapping[str, Any]
    thermal: Mapping[str, Any]
    ledgers: Mapping[str, Any]
    provenance: Mapping[str, Any]
    uncertainty: Mapping[str, Any]


@dataclass(frozen=True)
class MeasuredRunState:
    """Run-record data mapped into the canonical state without imputation."""

    run_id: str | None
    qa_status: str
    qa_report: Mapping[str, Any]
    metrics: Mapping[str, Any]
    ledgers: Mapping[str, Any]
    snapshots: tuple[SensorSnapshot, ...] = ()
    residuals: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyAssessment:
    """Advisory/replay assessment from the operating twin."""

    mode: str
    snapshot_count: int
    final_state: Mapping[str, Any]
    command: Mapping[str, Any]
    shutdown_request: Mapping[str, Any] | None
    trip_reasons: tuple[str, ...] = ()
    advisory_only: bool = True


@dataclass(frozen=True)
class GateEvaluation:
    """Measured gate result, kept separate from model predictions."""

    status: str
    evidence_count: int
    candidate_verdicts: tuple[Mapping[str, Any], ...] = ()
    note: str = ""


@dataclass
class ReferenceCellState:
    """The single serializable state exchanged by the integrated pipeline."""

    state_id: str
    inputs: ReferenceCellInputs
    predicted: PredictedCellState | None = None
    observed: MeasuredRunState | None = None
    safety: SafetyAssessment | None = None
    gates: GateEvaluation = field(
        default_factory=lambda: GateEvaluation(
            status="not_evaluated", evidence_count=0, note="No gate evaluation requested."
        )
    )
    calibration: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "contract": PIPELINE_CONTRACT,
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "state_id": self.state_id,
            "inputs": self.inputs.to_dict(),
            "predicted": self.predicted,
            "observed": self.observed,
            "safety": self.safety,
            "gates": self.gates,
            "calibration": self.calibration,
            "notes": self.notes,
        })

    def write_json(self, path: str | Path) -> Path:
        """Persist a strict JSON state report and return its path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination


# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------

class ReferenceCellPipeline:
    """Compose the RC-1 models and the measured-run/safety boundaries.

    The model and solver callables are injectable for fast contract tests and
    for later calibrated implementations.  Production defaults point to the
    repository's existing models; no duplicate electrochemistry is hidden in
    this orchestration layer.
    """

    def __init__(
        self,
        config: ReferenceCellConfig | None = None,
        *,
        config_path: str | Path | None = None,
        physics_factory: Callable[[BathRecipe, CellGeometry, ProcessConditions], Any] = CellPhysics,
        gas_solver: Callable[..., CoupledGasResult] = solve_coupled,
        thermal_solver: Callable[..., Mapping[str, Any]] = simulate_thermal_transient,
        registry_path: str | Path | None = None,
        gas_segments: int = 6,
        gas_iterations: int = 8,
    ) -> None:
        if config is not None and config_path is not None:
            raise ValueError("pass config or config_path, not both")
        self.config_path = None if config is not None else Path(config_path or DEFAULT_CONFIG_PATH)
        self.config = config or load_reference_cell_config(self.config_path)
        self.physics_factory = physics_factory
        self.gas_solver = gas_solver
        self.thermal_solver = thermal_solver
        self.registry_path = Path(registry_path) if registry_path is not None else None
        if gas_segments < 1 or gas_iterations < 1:
            raise ValueError("gas_segments and gas_iterations must be positive")
        self.gas_segments = int(gas_segments)
        self.gas_iterations = int(gas_iterations)

    def default_inputs(self, **overrides: Any) -> ReferenceCellInputs:
        """Return RC-1 inputs at the declared decision-current condition."""
        return ReferenceCellInputs.from_config(self.config, **overrides)

    def _conditions_for(self, inputs: ReferenceCellInputs, *, delta_m: float | None = None) -> ProcessConditions:
        conditions = replace(self.config.conditions, temperature_C=float(inputs.temperature_C))
        if delta_m is not None:
            conditions = replace(conditions, boundary_layer_m=float(delta_m))
        return conditions

    def _physics_for(self, inputs: ReferenceCellInputs, *, delta_m: float | None = None) -> Any:
        return self.physics_factory(
            self.config.bath,
            self.config.geometry,
            self._conditions_for(inputs, delta_m=delta_m),
        )

    def _channel_for(self, inputs: ReferenceCellInputs) -> ChannelGeometry:
        """Build gas-channel geometry from the same RC-1 geometry inputs."""
        return ChannelGeometry(
            height_m=self.config.channel_length_m,
            width_m=self.config.channel_width_m,
            depth_m=self.config.nominal_channel_depth_m,
            interelectrode_gap_m=self.config.geometry.interelectrode_gap_m,
            liquid_flow_L_min=inputs.flow_L_min,
        )

    def _fe_model_hook(self, inputs: ReferenceCellInputs) -> Callable[..., float]:
        """Return the gas model's FE hook backed by ``CellPhysics``."""
        cache: dict[tuple[float, float], float] = {}

        def fe_model(
            j_mA_cm2: float,
            delta_m: float,
            temperature_C: float,
            fe_conc_M: float,
            pH_bulk: float,
        ) -> float:
            # The gas solver passes temperature/concentration/pH explicitly.
            # The hook uses the pipeline's bath and conditions as the canonical
            # chemistry; those arguments are retained in the key so a future
            # calibrated solver cannot silently alias different states.
            key = (
                round(float(j_mA_cm2), 8),
                round(float(delta_m), 12),
                round(float(temperature_C), 6),
                round(float(fe_conc_M), 8),
                round(float(pH_bulk), 6),
            )
            if key not in cache:
                local_inputs = replace(inputs, temperature_C=float(temperature_C))
                point = self._physics_for(local_inputs, delta_m=float(delta_m)).solve_at_j(float(j_mA_cm2))
                cache[key] = float(point.current_efficiency)
            return cache[key]

        return fe_model

    @staticmethod
    def _thermal_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
        """Keep scalar thermal results in the canonical state, not large traces."""
        keys = (
            "T_ss_C",
            "T_max_C",
            "T_target_C",
            "cooling_duty_target_W",
            "cooling_duty_50C_W",
            "thermal_mass_kJ_K",
            "heat_gen_power_W",
        )
        summary = {key: raw.get(key) for key in keys if key in raw}
        for trace_key in ("time_hr", "temperature_C"):
            if trace_key in raw:
                try:
                    summary[f"{trace_key}_points"] = len(raw[trace_key])
                except TypeError:
                    pass
        return summary

    def simulate(
        self,
        inputs: ReferenceCellInputs | None = None,
        *,
        state_id: str | None = None,
    ) -> ReferenceCellState:
        """Run the coupled RC-1 screening chain and return one typed state."""
        inputs = inputs or self.default_inputs()
        point: OperatingPoint = self._physics_for(inputs).solve_at_j(inputs.current_density_mA_cm2)
        gas = self.gas_solver(
            j_mean_mA_cm2=inputs.current_density_mA_cm2,
            geometry=self._channel_for(inputs),
            temperature_C=inputs.temperature_C,
            kappa_S_m=float(point.conductivity_S_m),
            delta_forced_m=self.config.conditions.boundary_layer_m,
            fe_conc_M=self.config.bath.c_FeSO4_M,
            pH_bulk=self.config.bath.pH,
            n_segments=self.gas_segments,
            max_iterations=self.gas_iterations,
            fe_model=self._fe_model_hook(inputs),
        )

        v_coupled = float(point.V_cell + gas.ohmic_penalty_V)
        fe_uncoupled = float(point.current_efficiency)
        fe_coupled = float(gas.area_average_FE)
        energy_headline = float(specific_energy_kWh_per_t(v_coupled, fe_uncoupled))
        energy_with_fe_shift = float(specific_energy_kWh_per_t(v_coupled, fe_coupled))
        duration_s = inputs.thermal_duration_hr * 3_600.0
        charge_C = inputs.current_A * duration_s
        fe_mass_g = charge_C * fe_coupled * M_FE_G / (Z_FE * FARADAY)
        fe_mass_uncoupled_g = charge_C * fe_uncoupled * M_FE_G / (Z_FE * FARADAY)
        stack_energy_Wh = inputs.current_A * v_coupled * duration_s / 3_600.0

        thermal_params = CellThermalParams(
            V_cell=v_coupled,
            current_A=inputs.current_A,
            volume_L=self.config.total_volume_L,
            T_init_C=(
                inputs.initial_temperature_C
                if inputs.initial_temperature_C is not None
                else inputs.temperature_C
            ),
            T_amb_C=inputs.ambient_temperature_C,
            relative_humidity=inputs.relative_humidity,
            cooling_active=inputs.cooling_active,
            T_target_C=self.config.target_temperature_C,
        )
        thermal_raw = self.thermal_solver(
            thermal_params,
            t_end_hr=inputs.thermal_duration_hr,
            dt_s=inputs.thermal_dt_s,
        )

        operating = {
            "current_A": inputs.current_A,
            "j_mA_cm2": inputs.current_density_mA_cm2,
            "FE_uncoupled": fe_uncoupled,
            "FE_coupled": fe_coupled,
            "FE_shift_percentage_points": float(gas.FE_shift),
            "V_cell_uncoupled": float(point.V_cell),
            "V_cell_coupled": v_coupled,
            "gas_ohmic_penalty_V": float(gas.ohmic_penalty_V),
            "specific_energy_uncoupled_kWh_t": float(point.specific_energy_kWh_t),
            "specific_energy_coupled_kWh_t": energy_headline,
            "specific_energy_with_coupled_fe_kWh_t": energy_with_fe_shift,
            "transport_limit_mA_cm2": float(point.transport_limit_mA_cm2),
            "transport_margin": float(point.transport_limit_mA_cm2 / inputs.current_density_mA_cm2),
            "surface_pH": float(point.surface_pH),
            "surface_fe_M": float(point.surface_fe_M),
            "feoh2_supersaturation": float(point.feoh2_supersaturation),
            "precipitation_active": bool(point.precipitation_active),
            "deposition_rate_um_hr": float(point.deposition_rate_um_hr),
            "conductivity_S_m": float(point.conductivity_S_m),
            "V_decomposition": _jsonable({
                **dict(point.V_decomposition),
                "gas_ohmic_penalty_V": float(gas.ohmic_penalty_V),
                "V_cell_coupled": v_coupled,
            }),
            "transport_converged": bool(point.transport_converged),
        }
        ledgers = {
            "charge": {
                "duration_s": duration_s,
                "applied_cathodic_charge_C": charge_C,
                "predicted_fe_mass_g": fe_mass_g,
                "predicted_fe_mass_uncoupled_g": fe_mass_uncoupled_g,
                "predicted_her_fraction": 1.0 - fe_coupled,
                "hydrogen_flow_L_h_wet": float(gas.hydrogen_flow_L_h),
                "status": "model_projection_only",
            },
            "energy": {
                "stack_energy_Wh": stack_energy_Wh,
                "specific_energy_kWh_t": energy_headline,
                "specific_energy_with_coupled_fe_kWh_t": energy_with_fe_shift,
                "auxiliary_energy_Wh": None,
                "status": "model_projection_only",
            },
        }
        provenance = {
            "level": MODEL_LEVEL,
            "status": "screening_prediction",
            "configuration_id": self.config.configuration_id,
            "config_source": (
                str(self.config_path) if self.config_path is not None else "provided_config_object"
            ),
            "models": [
                "cell_physics.CellPhysics",
                "gas_holdup.solve_coupled",
                "thermal_balance.simulate_thermal_transient",
            ],
            "gate_evidence": False,
            "note": "No model output is experimental gate evidence.",
        }
        uncertainty = {
            "status": "not_calibrated",
            "parameter_uncertainty": "not yet estimated from RC-1 measurements",
            "dominant_unknowns": [
                "Fe/HER kinetics and Tafel parameters on the actual cathode",
                "membrane resistance and crossover",
                "bubble departure/coverage and bubbly-electrolyte conductivity",
                "thermal loss coefficients and auxiliary loads",
            ],
        }
        predicted = PredictedCellState(
            status="screening_prediction",
            operating=operating,
            gas=gas.to_dict(),
            thermal=self._thermal_summary(thermal_raw),
            ledgers=ledgers,
            provenance=provenance,
            uncertainty=uncertainty,
        )
        snapshot = SensorSnapshot(
            timestamp_s=0.0,
            current_A=inputs.current_A,
            voltage_V=v_coupled,
            temperature_C=inputs.temperature_C,
            pH=self.config.bath.pH,
            fe2_M=self.config.bath.c_FeSO4_M,
            cathode_area_cm2=inputs.active_area_cm2,
            source_run_id=state_id or f"screening-{self.config.configuration_id}",
        )
        safety = self._assess_safety((snapshot,))
        return ReferenceCellState(
            state_id=state_id or f"screening-{self.config.configuration_id}-{inputs.current_density_mA_cm2:g}",
            inputs=inputs,
            predicted=predicted,
            safety=safety,
            gates=GateEvaluation(
                status="not_evidence",
                evidence_count=0,
                note="Simulation output is deliberately excluded from process-gate evidence.",
            ),
            calibration={
                "status": "not_applied",
                "next_step": "ingest replicated RC-1 run records and fit only declared parameters",
            },
            notes=(
                "All integrated outputs are L0 screening predictions.",
                "Safety assessment is advisory; independent shutdown remains required.",
            ),
        )

    # ------------------------------------------------------------------
    # Operating-twin boundary
    # ------------------------------------------------------------------

    def _assess_safety(self, snapshots: Sequence[SensorSnapshot]) -> SafetyAssessment | None:
        if not snapshots:
            return None
        twin: OperatingTwin = build_reference_cell_operating_twin(self.config)
        for snapshot in snapshots:
            twin.update(snapshot, now_s=snapshot.timestamp_s)
        command: ControlCommand = twin.command()
        request: ShutdownRequest | None = twin.shutdown_request(snapshots[-1])
        state: TwinState = twin.state
        return SafetyAssessment(
            mode=state.mode.value,
            snapshot_count=len(snapshots),
            final_state=state.to_dict(),
            command=command.to_dict(),
            shutdown_request=None if request is None else request.to_dict(),
            trip_reasons=tuple(state.trip_reasons),
            advisory_only=True,
        )

    @staticmethod
    def _run_sign(manifest: Mapping[str, Any]) -> str:
        conventions = manifest.get("measurement_conventions", {})
        value = manifest.get("current_sign_convention")
        if value is None and isinstance(conventions, Mapping):
            value = conventions.get("cathodic_sign") or conventions.get("current_sign")
        return str(value or "negative").strip().lower()

    def _snapshots_from_record(self, record: RunRecord) -> tuple[SensorSnapshot, ...]:
        """Map a validated run trace to positive-cathodic safety snapshots."""
        composition = record.bath_batch.get("composition", {})
        default_pH = _finite_float(composition.get("pH"), self.config.bath.pH)
        default_fe2 = _finite_float(composition.get("fe2_g_L"), self.config.bath.c_FeSO4_M * M_FE_G) / M_FE_G
        default_temperature = _finite_float(
            record.metadata.get("temperature_C"), self.config.target_temperature_C
        )
        setup = record.manifest.get("setup", {})
        cathode = setup.get("cathode", {}) if isinstance(setup, Mapping) else {}
        area = _finite_float(cathode.get("area_cm2"), self.config.active_area_cm2)
        sign = self._run_sign(record.manifest)
        positive_cathodic = sign in {"positive", "cathodic_positive", "cathodic-positive"}

        snapshots: list[SensorSnapshot] = []
        for row in record.timeseries.to_dict("records"):
            raw_current = _finite_float(row.get("current_actual_A"), 0.0)
            current = raw_current if positive_cathodic else -raw_current
            # Safety uses the magnitude of the cathodic load; negative or
            # reverse-pulse values are not treated as a positive load.
            current = max(0.0, current)
            timestamp = _finite_float(row.get("timestamp_s"), 0.0)
            temperature = _finite_float(row.get("temperature_C"), default_temperature)
            pH = _finite_float(row.get("pH"), default_pH)
            fe2 = _finite_float(row.get("fe2_M"), default_fe2)
            voltage = max(0.0, _finite_float(row.get("voltage_V"), 0.0))
            snapshots.append(SensorSnapshot(
                timestamp_s=max(0.0, timestamp),
                current_A=current,
                voltage_V=voltage,
                temperature_C=temperature,
                pH=pH,
                fe2_M=max(0.0, fe2),
                cathode_area_cm2=area,
                source_run_id=str(record.manifest.get("run_id", "")),
                wind_gust_m_s=(
                    _finite_float(row["wind_gust_m_s"], 0.0)
                    if row.get("wind_gust_m_s") is not None else None
                ),
                flood_depth_m=(
                    _finite_float(row["flood_depth_m"], 0.0)
                    if row.get("flood_depth_m") is not None else None
                ),
                ingress_detected=bool(row.get("ingress_detected", False)),
            ))
        return tuple(snapshots)

    def _config_and_inputs_from_record(
        self,
        record: RunRecord,
    ) -> tuple[ReferenceCellConfig, ReferenceCellInputs]:
        """Create a prediction condition from measured run metadata.

        Only fields explicitly represented by the run contract are mapped.
        Missing supporting salt, membrane, and flow values remain the RC-1
        design defaults and are called out in the calibration report.
        """
        composition = record.bath_batch.get("composition", {})
        fe2_M = _finite_float(composition.get("fe2_g_L"), self.config.bath.c_FeSO4_M * M_FE_G) / M_FE_G
        h3bo3_M = _finite_float(composition.get("h3bo3_g_L"), self.config.bath.c_H3BO3_M * H3BO3_MOLAR_MASS_G_MOL) / H3BO3_MOLAR_MASS_G_MOL
        pH = _finite_float(composition.get("pH"), self.config.bath.pH)
        bath = replace(
            self.config.bath,
            c_FeSO4_M=max(fe2_M, 1e-9),
            c_H3BO3_M=max(h3bo3_M, 0.0),
            pH=pH,
        )
        setup = record.manifest.get("setup", {})
        cathode = setup.get("cathode", {}) if isinstance(setup, Mapping) else {}
        area = _finite_float(cathode.get("area_cm2"), self.config.active_area_cm2)
        measured_j = record.derived.current_density_mA_cm2
        if measured_j is None:
            raw = np.asarray(record.timeseries["current_actual_A"], dtype=float)
            measured_j = float(np.mean(np.abs(raw)) / area * 1_000.0)
        measured_temperature = _mean_column(
            record.timeseries, "temperature_C", _finite_float(record.metadata.get("temperature_C"), self.config.target_temperature_C)
        )
        observed_config = replace(self.config, bath=bath, active_area_cm2=area)
        inputs = ReferenceCellInputs.from_config(
            observed_config,
            current_density_mA_cm2=max(float(measured_j), 1e-9),
            temperature_C=measured_temperature,
        )
        return observed_config, inputs

    @staticmethod
    def _residuals(predicted: PredictedCellState, record: RunRecord) -> dict[str, Any]:
        """Compare predictions to observations without changing either."""
        residuals: dict[str, Any] = {}
        measured_fe = record.derived.faradaic_efficiency
        predicted_fe = predicted.operating.get("FE_coupled")
        if measured_fe is not None and predicted_fe is not None:
            residuals["faradaic_efficiency_apparent"] = {
                "predicted": predicted_fe,
                "measured": measured_fe,
                "residual": float(predicted_fe - measured_fe),
                "unit": "fraction",
                "measurement_basis": "apparent dry-mass FE until deposit composition is independently verified",
            }
        measured_voltage = record.derived.mean_voltage_V
        predicted_voltage = predicted.operating.get("V_cell_coupled")
        if measured_voltage is not None and predicted_voltage is not None:
            residuals["cell_voltage"] = {
                "predicted": predicted_voltage,
                "measured": measured_voltage,
                "residual": float(predicted_voltage - measured_voltage),
                "unit": "V",
            }
        mass_g = record.derived.net_deposit_mass_g
        energy_Wh = record.derived.energy_Wh
        predicted_energy = predicted.operating.get("specific_energy_coupled_kWh_t")
        if mass_g is not None and energy_Wh is not None and mass_g > 0 and predicted_energy is not None:
            measured_energy = float(energy_Wh / (mass_g / 1_000.0))
            residuals["specific_energy_apparent_product"] = {
                "predicted": predicted_energy,
                "measured": measured_energy,
                "residual": float(predicted_energy - measured_energy),
                "unit": "kWh/t",
                "measurement_basis": "dry mass gain, not composition-corrected Fe product",
            }
        return residuals

    def _gate_evaluation(self, qa_report: Mapping[str, Any]) -> GateEvaluation:
        if not qa_report.get("ready_for_analysis", False):
            return GateEvaluation(
                status="pending_qa",
                evidence_count=0,
                note="The run is not analysis-ready; no gate evidence is evaluated.",
            )
        evidence_rows = qa_report.get("gate_evidence", {}).get("records", [])
        records: list[EvidenceRecord] = []
        for row in evidence_rows:
            try:
                records.append(EvidenceRecord(
                    run_id=str(row["run_id"]),
                    candidate_id=str(row["candidate_id"]),
                    gate_id=str(row["gate_id"]),
                    metric=str(row["metric"]),
                    value=float(row["value"]),
                    unit=str(row.get("unit", "")),
                    source=str(row.get("source", "experimental")),
                    notes=str(row.get("notes", "")),
                ))
            except (KeyError, TypeError, ValueError):
                # run_record already reports malformed declarations.  Do not
                # turn a malformed evidence row into a gate pass here.
                continue
        if not records:
            return GateEvaluation(
                status="pending_no_evidence",
                evidence_count=0,
                note="QA is ready, but no explicit manifest gate evidence was declared.",
            )
        try:
            verdicts = evaluate_all(records, self.registry_path)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            return GateEvaluation(
                status="error",
                evidence_count=len(records),
                note=f"Gate registry/evaluation error: {exc}",
            )
        return GateEvaluation(
            status="evaluated",
            evidence_count=len(records),
            candidate_verdicts=tuple(verdict.to_dict() for verdict in verdicts),
            note="Gate verdicts use experimental evidence only.",
        )

    def ingest_run(self, run_dir: str | Path) -> ReferenceCellState:
        """Ingest one run record, replay safety, and compare to the model.

        Incomplete or invalid records produce a useful pending state and are
        never patched with model values.  A complete record is mapped to the
        same integrated state, then the model is evaluated at its measured
        current density/temperature for residual reporting.
        """
        qa_report = build_qa_report(run_dir)
        run_id = qa_report.get("run_id")
        if not qa_report.get("ready_for_analysis", False):
            observed = MeasuredRunState(
                run_id=run_id,
                qa_status="pending",
                qa_report=qa_report,
                metrics=qa_report.get("metrics", {}),
                ledgers=qa_report.get("ledgers", {}),
            )
            return ReferenceCellState(
                state_id=str(run_id or Path(run_dir).name),
                inputs=self.default_inputs(),
                observed=observed,
                gates=self._gate_evaluation(qa_report),
                calibration={
                    "status": "blocked",
                    "reason": "run record is not analysis-ready",
                    "next_step": "resolve data-contract errors and missing required files",
                },
                notes=("No model imputation was applied to the incomplete run.",),
            )

        try:
            record = load_run_record(run_dir, strict=True)
        except DataContractError:
            # This should be rare because build_qa_report was already checked,
            # but preserving the no-imputation boundary is more important than
            # hiding a race or file change between the two reads.
            observed = MeasuredRunState(
                run_id=run_id,
                qa_status="race_or_contract_error",
                qa_report=qa_report,
                metrics=qa_report.get("metrics", {}),
                ledgers=qa_report.get("ledgers", {}),
            )
            return ReferenceCellState(
                state_id=str(run_id or Path(run_dir).name),
                inputs=self.default_inputs(),
                observed=observed,
                gates=GateEvaluation(
                    status="pending_qa", evidence_count=0, note="Run changed while loading; no gate evaluation."
                ),
                calibration={"status": "blocked", "reason": "strict run-record load failed"},
            )

        observed_config, inputs = self._config_and_inputs_from_record(record)
        observed_pipeline = ReferenceCellPipeline(
            observed_config,
            physics_factory=self.physics_factory,
            gas_solver=self.gas_solver,
            thermal_solver=self.thermal_solver,
            registry_path=self.registry_path,
            gas_segments=self.gas_segments,
            gas_iterations=self.gas_iterations,
        )
        simulation = observed_pipeline.simulate(inputs, state_id=str(run_id or "observed-prediction"))
        assert simulation.predicted is not None
        snapshots = self._snapshots_from_record(record)
        residuals = self._residuals(simulation.predicted, record)
        observed = MeasuredRunState(
            run_id=run_id,
            qa_status="analysis_ready",
            qa_report=qa_report,
            metrics=qa_report.get("metrics", {}),
            ledgers=qa_report.get("ledgers", {}),
            snapshots=snapshots,
            residuals=residuals,
        )
        calibration = {
            "status": "calibration_candidate",
            "parameters_updated": [],
            "residuals_available": sorted(residuals),
            "measured_inputs_used": [
                "current density",
                "temperature",
                "Fe(II) concentration when present",
                "bulk pH when present",
                "boric acid concentration when present",
            ],
            "inputs_left_at_RC1_default": [
                "supporting salt concentration",
                "membrane resistance/crossover",
                "measured flow when absent from the run record",
            ],
            "next_step": "fit declared parameters against replicated runs and validate on held-out conditions",
        }
        return ReferenceCellState(
            state_id=str(run_id or Path(run_dir).name),
            inputs=inputs,
            predicted=simulation.predicted,
            observed=observed,
            safety=self._assess_safety(snapshots),
            gates=self._gate_evaluation(qa_report),
            calibration=calibration,
            notes=(
                "Predictions and measurements are stored in separate branches of the state.",
                "Gate verdicts are measurement-only; residuals are not gate evidence.",
                "OperatingTwin replay is advisory and cannot actuate hardware.",
            ),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> ReferenceCellState:
    """Run a screening state or ingest one experimental run directory."""
    parser = argparse.ArgumentParser(description="Run the unified RC-1 reference-cell pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="RC-1 YAML design basis")
    parser.add_argument("--run-dir", default=None, help="Validated run-record directory to ingest/replay")
    parser.add_argument("--j", type=float, default=None, help="Current density for a screening state (mA/cm²)")
    parser.add_argument("--flow", type=float, default=None, help="Cathode-channel flow (L/min)")
    parser.add_argument("--temperature", type=float, default=None, help="Electrolyte temperature (°C)")
    parser.add_argument("--out", default=None, help="Write the canonical state JSON here")
    args = parser.parse_args(argv)

    pipeline = ReferenceCellPipeline(config_path=args.config)
    if args.run_dir:
        state = pipeline.ingest_run(args.run_dir)
    else:
        inputs = pipeline.default_inputs(
            current_density_mA_cm2=args.j,
            flow_L_min=args.flow,
            temperature_C=args.temperature,
        )
        state = pipeline.simulate(inputs)
    text = json.dumps(state.to_dict(), indent=2, allow_nan=False) + "\n"
    if args.out:
        state.write_json(args.out)
    else:
        print(text, end="")
    return state


__all__ = [
    "PIPELINE_CONTRACT",
    "PIPELINE_SCHEMA_VERSION",
    "MODEL_LEVEL",
    "ReferenceCellInputs",
    "PredictedCellState",
    "MeasuredRunState",
    "SafetyAssessment",
    "GateEvaluation",
    "ReferenceCellState",
    "ReferenceCellPipeline",
    "main",
]


if __name__ == "__main__":
    main()
