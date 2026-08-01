import pandas as pd
from models.run_voltammetry import generate_synthetic_cv, main


def test_generate_synthetic_cv_properties(tmp_path):
    csv_file = tmp_path / "test_cv.csv"
    df = generate_synthetic_cv(csv_file, scan_rate_V_s=0.1, num_cycles=1, noise_level_A=1e-6)

    # Check file exists and DataFrame properties
    assert csv_file.exists()
    assert isinstance(df, pd.DataFrame)

    # Check required columns are present
    required_cols = {
        "timestamp_s", "potential_V_vs_ref", "current_A", "working_electrode_area_cm2",
        "cycle", "segment", "temperature_C", "pH", "fe2_concentration_M",
        "electrolyte_id", "reference_electrode", "notes"
    }
    assert required_cols.issubset(df.columns)

    # Check values
    assert (df["working_electrode_area_cm2"] == 1.0).all()
    assert (df["pH"] == 3.0).all()
    assert (df["fe2_concentration_M"] == 1.0).all()
    assert (df["reference_electrode"] == "Ag/AgCl").all()
    assert (df["cycle"] == 1).all()

    # Check timestamps are rounded/non-decreasing
    assert (df["timestamp_s"].diff().dropna() >= 0).all()

    # Potential range should be within expected bounds
    assert df["potential_V_vs_ref"].min() < -0.2
    assert df["potential_V_vs_ref"].max() > -1.2


def test_main_runs_without_error(monkeypatch, tmp_path):
    # Mock data and fig directories to use a temp directory during test
    temp_fig_dir = tmp_path / "figures"
    temp_data_dir = tmp_path / "data"
    temp_fig_dir.mkdir()
    temp_data_dir.mkdir()

    monkeypatch.setattr("models.run_voltammetry.FIG_DIR", temp_fig_dir)
    monkeypatch.setattr("models.run_voltammetry.DATA_DIR", temp_data_dir)

    # Run main() and check that output files are written
    main()

    assert (temp_data_dir / "synthetic_voltammetry.csv").exists()
    assert (temp_data_dir / "voltammetry_report.json").exists()
    assert (temp_fig_dir / "voltammetry_analysis.png").exists()
    assert (temp_fig_dir / "tafel_analysis.png").exists()
