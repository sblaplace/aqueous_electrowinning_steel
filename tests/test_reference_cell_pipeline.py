"""Contract tests for the unified RC-1 reference-cell pipeline."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from models.reference_cell_pipeline import (
    ReferenceCellInputs,
    ReferenceCellPipeline,
)
from models.reference_cell_design import load_reference_cell_config


class _FakePhysics:
    """Cheap deterministic stand-in used to test orchestration, not physics."""

    def __init__(self, bath, geometry, conditions):
        self.bath = bath
        self.geometry = geometry
        self.conditions = conditions

    def solve_at_j(self, j):
        return SimpleNamespace(
            j_mA_cm2=float(j),
            current_efficiency=0.80,
            surface_pH=self.bath.pH + 0.2,
            surface_fe_M=self.bath.c_FeSO4_M * 0.9,
            transport_limit_mA_cm2=600.0,
            diffusion_limit_mA_cm2=300.0,
            migration_enhancement=2.0,
            feoh2_supersaturation=0.1,
            film_potential_drop_V=0.01,
            precipitation_active=False,
            V_cell=2.4,
            V_decomposition={"E_cathode": -0.7, "E_anode": 1.7, "IR_drop": 0.3},
            specific_energy_kWh_t=2_878.0,
            deposition_rate_um_hr=100.0,
            free_fe2_activity=self.bath.c_FeSO4_M,
            conductivity_S_m=12.0,
            speciation={"source": "fake"},
            transport_converged=True,
        )


def _fake_thermal(params, t_end_hr=1.0, dt_s=10.0):
    return {
        "time_hr": [0.0, t_end_hr],
        "temperature_C": [params.T_init_C, params.T_init_C + 1.0],
        "T_ss_C": params.T_init_C + 1.0,
        "T_max_C": params.T_init_C + 1.0,
        "cooling_duty_50C_W": 2.0,
        "thermal_mass_kJ_K": 10.0,
        "heat_gen_power_W": 5.0,
    }


def _pipeline():
    config = load_reference_cell_config()
    return ReferenceCellPipeline(
        config=config,
        physics_factory=_FakePhysics,
        thermal_solver=_fake_thermal,
        gas_segments=3,
        gas_iterations=3,
    )


def test_inputs_share_area_and_current_density():
    pipeline = _pipeline()
    inputs = pipeline.default_inputs(current_density_mA_cm2=100.0, flow_L_min=0.25)
    assert isinstance(inputs, ReferenceCellInputs)
    assert inputs.current_A == pytest.approx(1.0)
    assert inputs.to_dict()["current_A"] == pytest.approx(1.0)


def test_simulation_wires_physics_gas_thermal_safety_and_ledgers():
    pipeline = _pipeline()
    state = pipeline.simulate(
        pipeline.default_inputs(
            current_density_mA_cm2=100.0,
            flow_L_min=0.25,
            thermal_duration_hr=0.01,
        )
    )

    assert state.predicted is not None
    assert state.predicted.provenance["models"] == [
        "cell_physics.CellPhysics",
        "gas_holdup.solve_coupled",
        "thermal_balance.simulate_thermal_transient",
    ]
    assert state.predicted.operating["FE_coupled"] == pytest.approx(0.80)
    assert state.predicted.operating["V_cell_coupled"] >= 2.4
    assert state.predicted.gas["converged"] is True
    assert state.predicted.ledgers["charge"]["applied_cathodic_charge_C"] > 0.0
    assert state.predicted.thermal["T_max_C"] > 0.0
    assert state.safety is not None
    assert state.safety.advisory_only is True
    assert state.gates.status == "not_evidence"

    unsafe = pipeline.simulate(pipeline.default_inputs(current_density_mA_cm2=400.0))
    assert unsafe.safety is not None
    assert unsafe.safety.mode == "tripped"
    assert "current_density_limit" in unsafe.safety.trip_reasons

    # The report must be strict JSON and keep model gates out of evidence.
    encoded = json.dumps(state.to_dict(), allow_nan=False)
    assert "screening_prediction" in encoded
    assert '"source": "experimental"' not in encoded


def _manifest():
    return {
        "schema_version": "1.0",
        "record_status": "complete",
        "run_id": "rc1-20260805-001",
        "date": "2026-08-05",
        "operator": "tester",
        "experiment_type": "divided_cell",
        "bath_batch": "B-001",
        "equipment": {
            "power_supply": {"model": "PS", "asset_id": "PS-01"},
            "cell": {"type": "RC-1", "volume_mL": 2000},
        },
        "setup": {
            "anode": {"material": "DSA"},
            "cathode": {"material": "316L", "area_cm2": 10.0},
        },
        "measurement_conventions": {"cathodic_sign": "negative"},
        "video": {"recording_status": "not_applicable"},
    }


def _bath():
    return {
        "schema_version": "1.0",
        "batch_id": "B-001",
        "date_mixed": "2026-08-05",
        "operator": "tester",
        "composition": {"fe2_g_L": 55.845, "h3bo3_g_L": 30.0, "pH": 2.0, "volume_mL": 1000.0},
        "source_chemicals": {"water_source": "DI"},
        "storage": {"container": "glass"},
    }


def _metadata():
    return {
        "sample_id": "rc1-20260805-001",
        "operator": "tester",
        "instrument": "PS-01/DMM-01",
        "calibration_date": "2026-08-01",
        "electrolyte_id": "B-001",
        "working_electrode": "316L, 10 cm2",
        "counter_electrode": "DSA",
        "reference_electrode": "none",
        "temperature_C": 60.0,
        "agitation": "recirculation",
        "preparation": "cleaned and dried",
    }


def _write_run(tmp_path):
    root = tmp_path / "rc1-20260805-001"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    (root / "bath_batch.json").write_text(json.dumps(_bath()), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    pd.DataFrame({
        "timestamp_s": [0.0, 1.0, 2.0],
        "current_actual_A": [-1.0, -1.0, -1.0],
        "voltage_V": [2.5, 2.5, 2.5],
        "temperature_C": [60.0, 60.0, 60.0],
        "pH": [2.0, 2.0, 2.0],
    }).to_csv(root / "timeseries.csv", index=False)
    return root


def test_incomplete_run_is_not_imputed_or_sent_to_gates(tmp_path):
    root = tmp_path / "planned"
    root.mkdir()
    manifest = _manifest()
    manifest["record_status"] = "planned"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    state = _pipeline().ingest_run(root)
    assert state.predicted is None
    assert state.observed is not None
    assert state.gates.status == "pending_qa"
    assert state.calibration["status"] == "blocked"
    assert "No model imputation" in state.notes[0]


def test_complete_run_replays_safety_and_reports_gate_pending(tmp_path):
    state = _pipeline().ingest_run(_write_run(tmp_path))
    assert state.predicted is not None
    assert state.observed is not None
    assert state.observed.qa_status == "analysis_ready"
    assert len(state.observed.snapshots) == 3
    assert state.observed.snapshots[0].current_A == pytest.approx(1.0)
    assert state.safety is not None
    assert state.safety.mode == "advisory"
    assert state.gates.status == "pending_no_evidence"
    assert state.calibration["status"] == "calibration_candidate"

    # No declared gate evidence means the gate engine was not invoked and no
    # candidate can be passed by this integration layer.
    assert state.gates.evidence_count == 0
    assert state.gates.candidate_verdicts == ()


def test_declared_experimental_gate_evidence_is_evaluated_separately(tmp_path):
    root = _write_run(tmp_path)
    manifest = _manifest()
    manifest["gate_evidence"] = [{
        "candidate_id": "divided_sulfate_dissolved_feed",
        "gate_id": "gate_2_gravimetric_fe",
        "metric": "faradaic_efficiency",
        "value_from": "apparent_faradaic_efficiency",
        "unit": "fraction",
    }]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # 2 C at 1 A would deposit 0.000578 g Fe; this mass is an apparent FE of
    # roughly 86%, enough to test the gate adapter without a model pass.
    pd.DataFrame({
        "mass_before_g": [1.0],
        "mass_after_g": [1.0005],
    }).to_csv(root / "mass_log.csv", index=False)

    state = _pipeline().ingest_run(root)
    assert state.gates.status == "evaluated"
    assert state.gates.evidence_count == 1
    candidate = next(
        verdict for verdict in state.gates.candidate_verdicts
        if verdict["candidate_id"] == "divided_sulfate_dissolved_feed"
    )
    gate = next(gate for gate in candidate["gates"] if gate["gate_id"] == "gate_2_gravimetric_fe")
    assert gate["status"] == "passed"
    assert state.observed is not None
    assert state.observed.residuals["faradaic_efficiency_apparent"]["measurement_basis"].startswith("apparent")
