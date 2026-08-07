"""Tests for the D6 L5 live-control wiring contract (closed_loop_twin).

Covers the fail-closed qualification gate (calibrated AND observed AND a
bounded actuation envelope), arming through the token guard, the closed loop
(observed state -> safety eval -> calibrated bounded command), and the
fail-closed paths (missing calibration, missing observability, calibration
with no viable CE floor).
"""

import pytest

from models.kinetics import DepositionKinetics
from models.operating_twin import OperatingTwin, SensorSnapshot, TwinConfig, TwinMode
from models.closed_loop_twin import (
    CellQualification,
    ClosedLoopLiveControlTwin,
    CalibrationEvidence,
    ObservabilityEvidence,
    build_reference_cell_qualification,
    evaluate_co_location_contract,
    DEFAULT_MIN_FE_FLOOR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_kinetics() -> DepositionKinetics:
    """A high-CE calibrated kinetics (Fe dominates HER), as for a real
    electrowinning reference cell: FE ~0.87 at 300 mA/cm^2."""
    return DepositionKinetics(
        pH=2.0, temperature_C=60.0,
        fe_i0=1.5e-2, her_i0=1.0e-5,
        fe_tafel_V=0.118, her_tafel_V=0.132,
        fe_conc_M=1.0, boundary_layer_m=4.2e-5,
    )


def _poor_kinetics() -> DepositionKinetics:
    """A kinetics whose CE never reaches the 0.8 floor (HER dominates due to
    high her_i0) -> no viable bounded actuation envelope."""
    return DepositionKinetics(
        pH=2.0, temperature_C=60.0,
        fe_i0=1.5e-2, her_i0=3.0e-4,
        fe_tafel_V=0.118, her_tafel_V=0.132,
        fe_conc_M=1.0, boundary_layer_m=4.2e-5,
    )


def _config(cell_id: str = "RC-1") -> TwinConfig:
    return TwinConfig(
        cell_id=cell_id,
        max_current_A=3.0,
        max_current_density_mA_cm2=300.0,
        max_voltage_V=8.0,
        min_temperature_C=50.0,
        max_temperature_C=70.0,
        min_fe2_M=0.2,
        max_fe2_M=2.0,
        min_pH=0.5,
        max_pH=5.0,
        target_current_A=3.0,
        target_temperature_C=60.0,
        current_ramp_A_per_s=0.5,
    )


def _snap(t=0.0, current=0.0) -> SensorSnapshot:
    return SensorSnapshot(
        timestamp_s=t, current_A=current, voltage_V=5.8,
        temperature_C=60.0, pH=2.0, fe2_M=0.9, cathode_area_cm2=10.0,
        source_run_id="D6-TEST",
    )


# ---------------------------------------------------------------------------
# Qualification gate (fail-closed)
# ---------------------------------------------------------------------------

def test_qualified_requires_calibration_plus_observability_plus_envelope():
    q = build_reference_cell_qualification(kinetics=_good_kinetics())
    assert q.qualified
    assert q.reasons() == []
    assert q.calibration_ok
    assert q.observability_ok
    assert q.observability.rank == 7
    assert q.bounded_viable_current_density_mA_cm2 == pytest.approx(300.0)
    assert q.bounded_target_current_A(10.0) == pytest.approx(3.0)


def test_fail_closed_without_calibration():
    q = build_reference_cell_qualification(kinetics=None, run_co_location_contract=False)
    assert not q.qualified
    assert not q.calibration_ok
    assert "calibration_missing_or_pending" in q.reasons()


def test_fail_closed_without_observability():
    q = build_reference_cell_qualification(
        kinetics=_good_kinetics(), run_co_location_contract=False
    )
    assert not q.qualified
    assert q.calibration_ok
    assert not q.observability_ok
    assert "observability_contract_not_holding" in q.reasons()


def test_fail_closed_when_no_bounded_actuation_envelope():
    # Both pillars present, but poor kinetics can never hold the CE floor.
    q = build_reference_cell_qualification(kinetics=_poor_kinetics())
    assert q.calibration_ok
    assert q.observability_ok
    assert not q.qualified
    assert q.bounded_viable_current_density_mA_cm2 == 0.0
    assert q.bounded_target_current_A(10.0) == 0.0
    assert any("no_viable_bounded_actuation" in r for r in q.reasons())


def test_reasons_is_single_source_of_truth_for_qualified():
    """qualified == not reasons() for every gate path (no contradiction)."""
    cases = [
        build_reference_cell_qualification(None, run_co_location_contract=False),
        build_reference_cell_qualification(_good_kinetics(), run_co_location_contract=False),
        build_reference_cell_qualification(_poor_kinetics()),
        build_reference_cell_qualification(_good_kinetics()),
    ]
    for q in cases:
        assert q.qualified == (len(q.reasons()) == 0)


def test_default_screening_kinetics_never_qualifies_calibration():
    # build_reference_cell_qualification(None) falls back to screening default
    # which is source='none' -> calibration does NOT qualify.
    q = build_reference_cell_qualification(None, run_co_location_contract=True)
    assert not q.calibration.qualified
    assert q.calibration.source == "none"
    assert not q.qualified


# ---------------------------------------------------------------------------
# Calibration + observability evidence pillars
# ---------------------------------------------------------------------------

def test_calibration_evidence_qualifies_only_with_a_fit():
    good = CalibrationEvidence(cell_id="RC-1", kinetics=_good_kinetics(), source="real_data")
    assert good.qualified
    assert good.efficiency_at(300.0) > DEFAULT_MIN_FE_FLOOR
    none = CalibrationEvidence(cell_id="RC-1", kinetics=_good_kinetics(), source="none")
    assert not none.qualified


def test_observability_evidence_requires_full_g0_contract():
    contract = evaluate_co_location_contract()
    assert contract.all_full_rank
    assert contract.all_cov_stable
    obs = ObservabilityEvidence(cell_id="RC-1", contract=contract, full_tags=tuple(contract.full_tags))
    assert obs.qualified
    assert obs.rank == 7
    empty = ObservabilityEvidence(cell_id="RC-1", contract=None, full_tags=())
    assert not empty.qualified


# ---------------------------------------------------------------------------
# Arming + closed loop
# ---------------------------------------------------------------------------

def test_arming_refuses_unqualified_cell():
    q = build_reference_cell_qualification(None, run_co_location_contract=False)
    twin = ClosedLoopLiveControlTwin(q, OperatingTwin(_config()), cathode_area_cm2=10.0)
    with pytest.raises(PermissionError):
        twin.arm_actuation("RC-1")


def test_arming_requires_exact_token_even_when_qualified():
    q = build_reference_cell_qualification(_good_kinetics())
    twin = ClosedLoopLiveControlTwin(q, OperatingTwin(_config()), cathode_area_cm2=10.0)
    with pytest.raises(PermissionError):
        twin.arm_actuation("WRONG-TOKEN")
    twin.arm_actuation("RC-1")
    assert twin.mode is TwinMode.ACTUATION


def test_closed_loop_ramps_bounded_command_and_reports_fe():
    q = build_reference_cell_qualification(_good_kinetics())
    twin = ClosedLoopLiveControlTwin(q, OperatingTwin(_config()), cathode_area_cm2=10.0)
    twin.arm_actuation("RC-1")
    cmd_current = 0.0
    last_ce = 0.0
    for i, t in enumerate(range(0, 6)):
        step = twin.step(_snap(t=float(t), current=cmd_current))
        assert step.qualified
        assert step.calibrated_fe > DEFAULT_MIN_FE_FLOOR
        # command never exceeds the calibrated-FE bounded target
        assert step.command_current_A <= q.bounded_target_current_A(10.0) + 1e-9
        cmd_current = step.command_current_A
        last_ce = step.calibrated_fe
    # The loop ramped current up (did not stay at 0 on the closed loop).
    assert cmd_current > 0.0
    assert last_ce > DEFAULT_MIN_FE_FLOOR


def test_closed_loop_stays_advisory_zero_without_pillars():
    q = build_reference_cell_qualification(None, run_co_location_contract=False)
    twin = ClosedLoopLiveControlTwin(q, OperatingTwin(_config()), cathode_area_cm2=10.0)
    step = twin.step(_snap(t=1.0))
    assert not step.qualified
    assert step.command_current_A == 0.0
    assert step.mode is TwinMode.ADVISORY


def test_over_current_trips_even_when_qualified():
    q = build_reference_cell_qualification(_good_kinetics())
    twin = ClosedLoopLiveControlTwin(q, OperatingTwin(_config()), cathode_area_cm2=10.0)
    twin.arm_actuation("RC-1")
    step = twin.step(_snap(t=1.0, current=50.0))
    assert step.mode is TwinMode.TRIPPED
    assert step.command_current_A == 0.0
    assert any("current_limit" in r for r in step.reasons)


def test_live_control_target_records_evidence_gates_pending():
    q = build_reference_cell_qualification(_good_kinetics())
    twin = ClosedLoopLiveControlTwin(q, OperatingTwin(_config()), cathode_area_cm2=10.0)
    target = twin.live_control_target()
    assert target["deliverable"] == "D6"
    assert target["level"] == "L5"
    assert target["qualification"]["qualified"]
    # Real Q1-Q5 + D1/D2/D3 evidence is still pending on the board -> honest.
    assert all(v == "pending" for v in target["evidence_gates"].values())
    assert target["observed_suite"] == [
        "TT-101", "TT-201", "pHAT-101", "CT-201",
        "VT-201", "THK-101", "CVT-201", "FE2P-101",
    ]
