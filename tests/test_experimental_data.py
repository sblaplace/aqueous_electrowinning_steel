import pandas as pd
import pytest

from models.experimental_data import load_measurements, summarize_run


def test_template_loads_and_derives_current_density():
    data = load_measurements("experiments/data/voltammetry_template.csv")
    assert data.loc[0, "current_density_mA_cm2"] == pytest.approx(0.0)
    assert summarize_run(data)["n_points"] == 1


def test_loader_rejects_missing_required_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp_s": [0], "current_A": [0],
                  "working_electrode_area_cm2": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="potential_V_vs_ref"):
        load_measurements(path)


def test_loader_rejects_non_monotonic_time(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp_s": [1, 0], "potential_V_vs_ref": [0, 0],
                  "current_A": [0, 0], "working_electrode_area_cm2": [1, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-decreasing"):
        load_measurements(path)
