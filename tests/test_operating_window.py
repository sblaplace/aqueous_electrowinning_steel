"""Tests for pre-lab operating window classifier and parameter mapper."""

import numpy as np
from models.operating_window import (
    evaluate_operating_point,
    map_2d_operating_window,
    STATUS_PASS,
    FAIL_THERMAL_MEMBRANE
)


def test_evaluate_normal_pass_point():
    """Moderate current density (100 mA/cm2), pH 2.5, 50 °C should pass all constraints."""
    res = evaluate_operating_point(j_mA_cm2=100.0, pH_bulk=2.5, T_C=50.0, gap_cm=0.3)
    assert res["is_pass"] is True
    assert res["status_code"] == STATUS_PASS
    assert res["FE"] > 0.70
    assert res["V_cell"] < 3.50
    assert res["specific_energy_kWh_t"] < 4000.0


def test_evaluate_high_temperature_membrane_fail():
    """Temperature above membrane limit (85 °C > 75 °C) should fail thermal check."""
    res = evaluate_operating_point(j_mA_cm2=100.0, pH_bulk=2.5, T_C=85.0)
    assert res["is_pass"] is False
    assert res["status_code"] == FAIL_THERMAL_MEMBRANE


def test_map_2d_operating_window():
    """Map 2D grid over j and pH."""
    j_vals = np.array([50.0, 100.0, 150.0])
    pH_vals = np.array([2.0, 2.5, 3.0])

    res = map_2d_operating_window(
        param_x_name="j_mA_cm2", x_vals=j_vals,
        param_y_name="pH_bulk", y_vals=pH_vals,
        fixed_params={"T_C": 50.0, "c_Fe2_M": 1.0, "delta_um": 30.0, "j_mA_cm2": 100.0, "pH_bulk": 2.5}
    )

    assert res["pass_mask"].shape == (3, 3)
    assert 0.0 <= res["pass_fraction"] <= 1.0
