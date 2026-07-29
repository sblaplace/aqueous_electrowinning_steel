import json

from models.run_hull_cell import (
    generate_synthetic_galvanostatic_trace,
    main,
)


def test_generate_synthetic_galvanostatic_trace(tmp_path):
    path = tmp_path / "synthetic_trace.csv"
    trace = generate_synthetic_galvanostatic_trace(path, duration_s=120.0, n_points=5)

    assert path.exists()
    assert len(trace) == 5
    assert {"timestamp_s", "current_A", "cell_voltage_V", "current_sign_convention"}.issubset(trace.columns)
    assert (trace["current_A"] < 0).all()
    assert trace["timestamp_s"].iloc[-1] == 120.0


def test_phase_ii_driver_writes_outputs(monkeypatch, tmp_path):
    figure_dir = tmp_path / "figures"
    data_dir = tmp_path / "data"
    figure_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.setattr("models.run_hull_cell.FIG_DIR", figure_dir)
    monkeypatch.setattr("models.run_hull_cell.DATA_DIR", data_dir)

    main()

    assert (figure_dir / "hull_cell_current_distribution.png").exists()
    assert (figure_dir / "gravimetric_faradaic_efficiency.png").exists()
    assert (data_dir / "synthetic_hull_cell_galvanostatic.csv").exists()
    assert (data_dir / "synthetic_hull_cell_gravimetry.csv").exists()
    report_path = data_dir / "hull_cell_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["hull_distribution_summary"]["total_current_A"] == 1.0
    assert report["gravimetric_faradaic_efficiency"]["apparent_faradaic_efficiency_percent"] > 90.0
