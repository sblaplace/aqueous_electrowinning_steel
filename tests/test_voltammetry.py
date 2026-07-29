import pandas as pd
import pytest
from models.experimental_data import load_measurements
from models.voltammetry import baseline_correct, extrema, scan_rate_V_s


def test_scan_rate_and_extrema():
    raw = pd.DataFrame({
        "timestamp_s": [0, 1, 2], "potential_V_vs_ref": [-1, 0, 1],
        "current_A": [-2, 1, 0], "working_electrode_area_cm2": [1, 1, 1]
    })
    data = load_measurements_from_frame(raw)
    assert scan_rate_V_s(data) == pytest.approx(1)
    result = extrema(baseline_correct(data, baseline_current_A=0))
    assert result["cathodic_peak_potential_V"] == -1
    assert result["anodic_peak_potential_V"] == 0


def test_scan_rate_requires_two_timestamps():
    data = load_measurements("experiments/data/voltammetry_template.csv")
    with pytest.raises(ValueError):
        scan_rate_V_s(data)


def load_measurements_from_frame(frame):
    frame["current_density_A_m2"] = frame["current_A"] / (frame["working_electrode_area_cm2"] * 1e-4)
    frame["current_density_mA_cm2"] = frame["current_A"] / frame["working_electrode_area_cm2"] * 100
    return frame
