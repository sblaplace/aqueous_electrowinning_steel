import numpy as np
import pandas as pd
import pytest

from models.hull_cell import (
    FARADAY_CONSTANT_C_MOL,
    MOLAR_MASS_FE_G_MOL,
    HullCellGeometry,
    analyze_gravimetric_efficiency,
    cathodic_charge_C,
    current_density_window,
    gravimetric_faradaic_efficiency,
    hull_current_distribution,
    load_galvanostatic_trace,
    load_gravimetry,
    summarize_hull_distribution,
)


def test_variable_gap_distribution_conserves_current_and_decays_toward_far_edge():
    geometry = HullCellGeometry(
        panel_length_cm=10.0, panel_width_cm=5.0,
        near_edge_gap_cm=1.5, far_edge_gap_cm=9.0,
    )
    distribution = hull_current_distribution(geometry, total_current_A=1.0, n_segments=100)

    assert distribution["segment_current_A"].sum() == pytest.approx(1.0)
    assert distribution["segment_area_cm2"].sum() == pytest.approx(50.0)
    assert distribution["current_density_mA_cm2"].iloc[0] > distribution["current_density_mA_cm2"].iloc[-1]
    assert geometry.panel_angle_deg == pytest.approx(48.59, rel=1e-3)
    assert distribution["cumulative_current_fraction_at_far_edge"].iloc[-1] == pytest.approx(1.0)


def test_parallel_plate_limit_is_uniform():
    geometry = HullCellGeometry(
        panel_length_cm=10.0, panel_width_cm=5.0,
        near_edge_gap_cm=3.0, far_edge_gap_cm=3.0,
    )
    distribution = hull_current_distribution(geometry, total_current_A=1.0, n_segments=20)
    assert np.ptp(distribution["current_density_A_cm2"]) == pytest.approx(0.0, abs=1e-15)
    assert distribution["current_density_mA_cm2"].iloc[0] == pytest.approx(20.0)


def test_geometry_rejects_impossible_panel_angle():
    with pytest.raises(ValueError, match="must be less"):
        HullCellGeometry(
            panel_length_cm=5.0, panel_width_cm=5.0,
            near_edge_gap_cm=1.0, far_edge_gap_cm=7.0,
        )


def test_distribution_summary_and_density_window():
    geometry = HullCellGeometry()
    distribution = hull_current_distribution(geometry, total_current_A=1.0, n_segments=100)
    summary = summarize_hull_distribution(distribution)
    window = current_density_window(distribution, 10.0, 100.0)

    assert summary["total_current_A"] == pytest.approx(1.0)
    assert summary["panel_average_current_density_mA_cm2"] == pytest.approx(20.0)
    assert summary["near_edge_gap_cm"] == pytest.approx(1.5)
    assert summary["far_edge_gap_cm"] == pytest.approx(9.0)
    assert 0 < window["area_fraction"] <= 1
    assert window["position_start_cm_from_near_edge"] == pytest.approx(0.0)
    # The far end is below 10 mA/cm² in this geometry, so the selected
    # 10–100 mA/cm² strip range ends before the physical far edge.
    assert 0.0 < window["position_end_cm_from_near_edge"] < 10.0


def test_cathodic_charge_excludes_anodic_portion_and_supports_sign_convention():
    time = np.array([0.0, 10.0, 20.0])
    current_negative = np.array([-1.0, -1.0, 1.0])
    # Clipping first makes the final trapezoid contain the 1 A -> 0 A transition.
    assert cathodic_charge_C(time, current_negative) == pytest.approx(15.0)
    assert cathodic_charge_C(time, -current_negative, cathodic_sign="positive") == pytest.approx(15.0)


def test_gravimetric_fe_matches_faraday_mass_balance_and_propagates_uncertainty():
    time = np.array([0.0, 3600.0])
    current = np.array([-2.0, -2.0])
    charge = 7200.0
    theoretical_g = charge * MOLAR_MASS_FE_G_MOL / (2 * FARADAY_CONSTANT_C_MOL)
    result = gravimetric_faradaic_efficiency(
        time, current,
        mass_before_g=10.0,
        mass_after_g=10.0 + 0.9 * theoretical_g + 0.001,
        blank_mass_change_g=0.001,
        mass_uncertainty_g=0.0001,
        blank_mass_uncertainty_g=0.0001,
        charge_relative_uncertainty=0.002,
    )

    assert result.cathodic_charge_C == pytest.approx(charge)
    assert result.theoretical_fe_mass_g == pytest.approx(theoretical_g)
    assert result.apparent_faradaic_efficiency == pytest.approx(0.9)
    assert result.apparent_faradaic_efficiency_percent == pytest.approx(90.0)
    assert result.net_deposit_mass_uncertainty_g == pytest.approx(np.sqrt(3) * 0.0001)
    assert result.apparent_faradaic_efficiency_uncertainty_percent is not None


def test_gravimetric_fe_does_not_hide_over_100_percent_result():
    result = gravimetric_faradaic_efficiency(
        [0.0, 10.0], [-1.0, -1.0], mass_before_g=1.0, mass_after_g=1.01,
    )
    assert result.apparent_faradaic_efficiency_percent > 100.0


def test_trace_and_gravimetry_loaders_and_join(tmp_path):
    trace_path = tmp_path / "trace.csv"
    gravimetry_path = tmp_path / "mass.csv"
    pd.DataFrame({
        "timestamp_s": [0.0, 60.0], "current_A": [-1.0, -1.0],
        "working_electrode_area_cm2": [2.0, 2.0],
    }).to_csv(trace_path, index=False)
    pd.DataFrame({
        "mass_before_g": [10.0], "mass_after_g": [10.01],
        "blank_mass_change_g": [0.0], "mass_uncertainty_g": [0.0001],
    }).to_csv(gravimetry_path, index=False)

    trace = load_galvanostatic_trace(trace_path)
    gravimetry = load_gravimetry(gravimetry_path)
    result = analyze_gravimetric_efficiency(trace, gravimetry)

    assert "current_density_mA_cm2" in trace.columns
    assert trace["current_density_mA_cm2"].iloc[0] == pytest.approx(-500.0)
    assert result.cathodic_charge_C == pytest.approx(60.0)


def test_loader_and_charge_validation(tmp_path):
    path = tmp_path / "bad_trace.csv"
    pd.DataFrame({"timestamp_s": [0.0], "cell_voltage_V": [2.6]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="current_A"):
        load_galvanostatic_trace(path)
    with pytest.raises(ValueError, match="No cathodic charge"):
        cathodic_charge_C([0.0, 10.0], [1.0, 1.0])
