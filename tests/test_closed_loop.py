"""Tests for Phase IV durability and closed-loop balances."""

import numpy as np
import pytest

from models.anode import AnodeKinetics, DSA_IRO2_TA2O5, NICO_SPINEL
from models.closed_loop import (
    AnodeDurabilityParams,
    ClosedLoopParams,
    PhaseIVClosedLoop,
    PhaseIVOperatingPoint,
)


def alkaline_model(**loop_overrides):
    anode = AnodeKinetics(NICO_SPINEL, electrolyte_type="alkaline", pH=14)
    loop_args = dict(
        volume_L=1000,
        feed_flow_L_hr=20,
        purge_flow_L_hr=20,
        fe_feed_M=1.25,
        ligand_feed_M=1.5,
        fe_initial_M=1.0,
        ligand_initial_M=1.3,
    )
    loop_args.update(loop_overrides)
    return PhaseIVClosedLoop(anode, ClosedLoopParams(**loop_args))


def test_operating_point_faraday_balance():
    op = PhaseIVOperatingPoint(current_density_mA_cm2=100, anode_area_m2=1, current_efficiency=1)
    assert op.current_A == pytest.approx(1000)
    assert op.fe_removal_mol_hr == pytest.approx(1000 * 3600 / (2 * 96485.3321))


def test_parameter_validation():
    with pytest.raises(ValueError):
        ClosedLoopParams(volume_L=0)
    with pytest.raises(ValueError):
        AnodeDurabilityParams(coating_loading_g_m2=-1)
    with pytest.raises(ValueError):
        PhaseIVOperatingPoint(current_efficiency=1.1)
    with pytest.raises(ValueError, match="constant-volume"):
        ClosedLoopParams(feed_flow_L_hr=20, purge_flow_L_hr=10)


def test_simulation_shapes_and_initial_state():
    result = alkaline_model().simulate(10, 1)
    assert len(result.time_hr) == 11
    assert result.fe_M[0] == 1.0
    assert result.ligand_M[0] == 1.3
    assert all(len(v) == 11 for v in result.as_columns().values())


def test_fe_balance_matches_one_euler_step_without_precipitation():
    model = alkaline_model(fe_solubility_M=2.0)
    result = model.simulate(1, 1)
    expected_rate = (
        20 * 1.25 - 20 * 1.0 - model.operating.fe_removal_mol_hr
    ) / 1000
    assert result.fe_M[1] == pytest.approx(1.0 + expected_rate)
    assert result.precipitated_fe_mol[1] == 0


def test_supersaturation_precipitates_iron():
    model = alkaline_model(fe_initial_M=1.8, fe_solubility_M=1.5, precipitation_rate_per_hr=0.5)
    result = model.simulate(1, 1)
    assert result.precipitated_fe_mol[-1] == pytest.approx(150.0)
    assert result.fe_M[-1] < result.fe_M[0]


def test_anode_wears_and_voltage_increases():
    model = alkaline_model()
    result = model.simulate(2000, 10)
    assert result.coating_remaining_fraction[-1] < 1
    assert result.cell_voltage_V[-1] > result.cell_voltage_V[0]
    assert result.anode_overpotential_V[-1] > result.anode_overpotential_V[0]


def test_zero_wear_preserves_coating_and_voltage():
    model = alkaline_model()
    model.durability = AnodeDurabilityParams(base_wear_mg_per_kAh=0)
    result = model.simulate(100, 5)
    assert np.all(result.coating_remaining_fraction == 1)
    assert np.ptp(result.cell_voltage_V) == pytest.approx(0)


def test_chloride_and_cer_accelerate_wear():
    anode = AnodeKinetics(
        DSA_IRO2_TA2O5, electrolyte_type="acidic_chloride", pH=1, a_Cl_molar=10
    )
    model = PhaseIVClosedLoop(anode)
    base = model.wear_rate_mg_per_kAh(0, 0)
    aggressive = model.wear_rate_mg_per_kAh(10, 0.5)
    assert aggressive > base


def test_closed_loop_chloride_updates_anode_activity():
    anode = AnodeKinetics(
        DSA_IRO2_TA2O5, electrolyte_type="acidic_chloride", pH=1, a_Cl_molar=10
    )
    model = PhaseIVClosedLoop(anode)
    assert model.degraded_anode(1.0, chloride_M=3.0).a_Cl_molar == 3.0


def test_quality_flags():
    model = alkaline_model(
        fe_initial_M=0.1,
        ligand_initial_M=0.01,
        impurity_initial_M=0.02,
    )
    result = model.simulate(1, 1)
    assert {"low_fe", "low_ligand_ratio", "high_impurity"} <= set(result.flags[0])


def test_time_step_stability_guard():
    with pytest.raises(ValueError, match="too large"):
        alkaline_model().simulate(100, 20)


def test_process_metrics_are_consistent():
    model = alkaline_model()
    result = model.simulate(100, 1)
    metrics = model.process_metrics(result)
    assert metrics["iron_produced_t"] > 0
    assert metrics["electricity_kWh"] > 0
    assert metrics["purge_volume_m3"] == pytest.approx(2.0)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert metrics["average_specific_energy_kWh_t"] == pytest.approx(
        integrate(model.operating.current_A * result.cell_voltage_V / 1000, result.time_hr)
        / metrics["iron_produced_t"]
    )


def test_summary_reports_eol_when_crossed():
    durability = AnodeDurabilityParams(
        coating_loading_g_m2=0.01,
        base_wear_mg_per_kAh=10,
        end_of_life_fraction=0.2,
    )
    model = alkaline_model()
    model.durability = durability
    result = model.simulate(2, 0.1)
    assert result.summary()["end_of_life_hr"] is not None
    assert "anode_end_of_life" in result.flags[-1]
