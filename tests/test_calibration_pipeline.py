"""Tests for the calibration pipeline.

Covers:
    - CSV ingestion handles missing columns gracefully
    - Fitting converges on synthetic data
    - Calibrated parameters are within physical bounds
    - load_calibrated_params() returns dict with expected keys
    - Model predictions improve after calibration (R² > 0.9 on synthetic)
    - Pipeline produces calibrated_parameters.json
    - Hall-Petch fit correct on linear data
    - EIS fit converges on synthetic spectrum
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models.calibration_pipeline import (
    CALIBRATED_PARAMS_FILENAME,
    FitResult,
    _check_physical_bounds,
    _r_squared,
    fit_hall_petch_domain,
    fit_eis_domain,
    load_calibrated_params,
    load_csv,
    load_csv_safe,
    run_calibration_pipeline,
    write_calibrated_parameters,
    CalibrationReport,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp(tmp_path):
    """Return a temporary directory for test outputs."""
    return tmp_path


# ─── 1. Ingestion handles missing columns gracefully ──────────────────────────

def test_ingestion_missing_columns_raises(tmp):
    """load_csv raises ValueError when required columns are missing."""
    bad_csv = tmp / "bad.csv"
    bad_csv.write_text("potential_V_vs_ref,garbage\n1.0,2.0\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_csv(bad_csv, "tafel")


def test_ingestion_safe_returns_none(tmp):
    """load_csv_safe returns None instead of raising."""
    missing = tmp / "missing.csv"
    assert load_csv_safe(missing, "tafel") is None

    bad_csv = tmp / "bad.csv"
    bad_csv.write_text("wrong_col\n1.0\n")
    assert load_csv_safe(bad_csv, "tafel") is None


def test_ingestion_empty_csv_raises(tmp):
    """Empty CSV raises ValueError."""
    empty = tmp / "empty.csv"
    empty.write_text("potential_V_vs_ref,current_density_A_m2\n")
    with pytest.raises(ValueError, match="empty"):
        load_csv(empty, "tafel")


# ─── 2. Fitting converges on synthetic data ────────────────────────────────────

def test_hall_petch_fit_converges():
    """Hall-Petch fit on synthetic linear data converges with R² > 0.99."""
    rng = np.random.default_rng(42)
    sigma0_true, k_true = 100.0, 0.50
    grain_um = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    d_m = grain_um * 1e-6
    y_true = sigma0_true + k_true / np.sqrt(d_m)
    y_noisy = y_true + rng.normal(0, 5, len(grain_um))

    data = pd.DataFrame({"grain_size_um": grain_um, "yield_MPa": y_noisy})
    result = fit_hall_petch_domain(data)

    assert result.converged
    assert result.r_squared > 0.99
    assert 80 < result.parameters["sigma0_MPa"] < 120
    assert 0.3 < result.parameters["k_HP_MPa_sqrt_m"] < 0.7


def test_eis_fit_converges_synthetic():
    """EIS Randles fit on a synthetic spectrum converges."""
    from models.eis import synthetic_randles_spectrum

    df = synthetic_randles_spectrum(
        rs_ohm=5.0, rct_ohm=100.0, cdl_F=1e-4,
        sigma_ohm_s_neg_half=50.0,
        freq_min_hz=0.1, freq_max_hz=1e5,
        points_per_decade=10, noise_rel=0.0,
    )
    result = fit_eis_domain(df)

    assert result.converged
    assert result.r_squared > 0.99
    assert abs(result.parameters["R_s_ohm"] - 5.0) < 1.0
    assert abs(result.parameters["R_ct_ohm"] - 100.0) < 20.0


# ─── 3. Calibrated parameters are within physical bounds ──────────────────────

def test_physical_bounds_check():
    """_check_physical_bounds returns True for valid and False for invalid params."""
    valid = {"sigma0_MPa": 100.0, "k_HP_MPa_sqrt_m": 0.5}
    assert _check_physical_bounds("hall_petch", valid)

    invalid = {"sigma0_MPa": -50.0}
    assert not _check_physical_bounds("hall_petch", invalid)

    invalid2 = {"k_softening": 0.5}  # way above 0.01
    assert not _check_physical_bounds("tempering", invalid2)


def test_hall_petch_params_within_bounds():
    """Fitted Hall-Petch parameters fall within physical bounds."""
    sigma0_true, k_true = 100.0, 0.50
    grain_um = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    d_m = grain_um * 1e-6
    y = sigma0_true + k_true / np.sqrt(d_m)
    data = pd.DataFrame({"grain_size_um": grain_um, "yield_MPa": y})
    result = fit_hall_petch_domain(data)

    assert _check_physical_bounds("hall_petch", result.parameters)


# ─── 4. load_calibrated_params() returns dict with expected keys ───────────────

def test_load_calibrated_params_from_file(tmp):
    """load_calibrated_params reads a JSON file and returns domain dicts."""
    payload = {
        "_metadata": {"version": "1.0"},
        "tafel": {
            "parameters": {"fe_i0_A_m2": 0.05, "her_i0_A_m2": 0.001},
        },
        "hall_petch": {
            "parameters": {"sigma0_MPa": 100.0, "k_HP_MPa_sqrt_m": 0.50},
        },
    }
    path = tmp / CALIBRATED_PARAMS_FILENAME
    path.write_text(json.dumps(payload))

    result = load_calibrated_params(path)
    assert "tafel" in result
    assert "hall_petch" in result
    assert result["tafel"]["fe_i0_A_m2"] == 0.05
    assert result["hall_petch"]["sigma0_MPa"] == 100.0


def test_load_calibrated_params_missing_file(tmp):
    """load_calibrated_params returns empty dict when no file found."""
    result = load_calibrated_params(tmp / "nonexistent.json")
    assert result == {}


# ─── 5. Model predictions improve after calibration ───────────────────────────

def test_r_squared_function():
    """_r_squared computes correct R² for perfect and imperfect fits."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _r_squared(y, y) == pytest.approx(1.0)

    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    r2 = _r_squared(y, y_pred)
    assert r2 > 0.95
    assert r2 < 1.0


def test_hall_petch_r2_above_09_synthetic():
    """Calibrated Hall-Petch model has R² > 0.9 on synthetic data."""
    sigma0_true, k_true = 100.0, 0.50
    grain_um = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    d_m = grain_um * 1e-6
    y = sigma0_true + k_true / np.sqrt(d_m)

    data = pd.DataFrame({"grain_size_um": grain_um, "yield_MPa": y})
    result = fit_hall_petch_domain(data)
    assert result.r_squared > 0.9


# ─── 6. Pipeline produces calibrated_parameters.json ──────────────────────────

def test_pipeline_produces_json(tmp):
    """run_calibration_pipeline writes calibrated_parameters.json."""
    # Create minimal synthetic CSVs
    sigma0, k = 100.0, 0.50
    grain_um = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    d_m = grain_um * 1e-6
    y = sigma0 + k / np.sqrt(d_m)

    data_dir = tmp / "data"
    data_dir.mkdir()
    (data_dir / "hall_petch.csv").write_text(
        "grain_size_um,yield_MPa\n" + "\n".join(f"{g},{yy}" for g, yy in zip(grain_um, y))
    )

    out_dir = tmp / "output"
    report = run_calibration_pipeline(
        data_dir=data_dir,
        output_dir=out_dir,
        domains=["hall_petch"],
        generate_figures=False,
    )

    assert report.output_path is not None
    assert report.output_path.exists()
    raw = json.loads(report.output_path.read_text())
    assert "hall_petch" in raw
    assert "parameters" in raw["hall_petch"]
    assert raw["_metadata"]["n_domains"] == 1


def test_write_and_roundtrip_calibrated_params(tmp):
    """Writing and re-reading calibrated_parameters.json preserves values."""
    result = FitResult(
        domain="test_domain",
        parameters={"param_a": 1.23, "param_b": 4.56},
        confidence_intervals={"param_a": (-0.1, 0.1), "param_b": (-0.2, 0.2)},
        r_squared=0.99,
        n_points=50,
        converged=True,
    )
    report = CalibrationReport(domain_results={"test_domain": result})
    path = write_calibrated_parameters(report, tmp)

    loaded = load_calibrated_params(path)
    assert "test_domain" in loaded
    assert loaded["test_domain"]["param_a"] == pytest.approx(1.23)
    assert loaded["test_domain"]["param_b"] == pytest.approx(4.56)


# ─── 7. Pipeline with empty/missing data dir ──────────────────────────────────

def test_pipeline_empty_data_dir(tmp):
    """Pipeline with no CSVs returns empty report gracefully."""
    empty_dir = tmp / "empty"
    empty_dir.mkdir()
    report = run_calibration_pipeline(
        data_dir=empty_dir, output_dir=tmp / "out", generate_figures=False
    )
    assert report.n_domains_fitted == 0
    assert report.output_path is None
