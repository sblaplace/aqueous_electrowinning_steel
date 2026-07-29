import json

import numpy as np
import pandas as pd
import pytest

from models.calibration import (
    calibrate_lsv_run,
    fit_eis_exchange_current,
    fit_total_cathodic_polarization,
)
from models.eis import synthetic_randles_spectrum
from models.kinetics import DepositionKinetics


def synthetic_polarization():
    potential_she = np.linspace(-0.66, -0.47, 80)
    kinetic = DepositionKinetics(
        pH=3.0, temperature_C=60.0, fe_conc_M=1.0, fe_i0=1e-2, her_i0=1e-4,
        fe_tafel_V=0.12, her_tafel_V=0.14, boundary_layer_m=5e-5,
    )
    total = kinetic.partial_currents(potential_she)[2]
    return pd.DataFrame({
        "potential_V_vs_ref": potential_she - 0.21,
        "current_density_A_m2": -total,
    })


def test_total_polarization_fit_recovers_synthetic_total_current():
    fit = fit_total_cathodic_polarization(
        synthetic_polarization(), pH=3.0, temperature_C=60.0, fe_conc_M=1.0,
        reference_to_she_V=0.21, initial_fe_i0_A_m2=1e-2, initial_her_i0_A_m2=1e-4,
    )
    assert fit.converged
    assert fit.n_points == 80
    assert fit.rmse_log10_current < 1e-5
    assert "total cathodic current" in fit.assumptions[0]


def test_total_polarization_fit_requires_enough_cathodic_data():
    with pytest.raises(ValueError, match="at least 10"):
        fit_total_cathodic_polarization(
            synthetic_polarization().iloc[:5], pH=3, temperature_C=60, fe_conc_M=1,
            reference_to_she_V=0.21,
        )


def test_eis_exchange_current_is_area_normalized(tmp_path):
    path = tmp_path / "eis.csv"
    synthetic_randles_spectrum(8, 12, 50e-6, 3, 0.01, 1e5, area_cm2=2.0).to_csv(path, index=False)
    report = fit_eis_exchange_current(path)
    assert report["fit"]["converged"]
    assert report["exchange_current_density_A_m2_from_rct"] == pytest.approx(
        report["exchange_current_A_from_rct"] / 2e-4
    )


def test_calibrate_lsv_run_requires_qa_ready_manifest_record(tmp_path):
    raw = tmp_path / "raw.csv"
    processed = tmp_path / "processed.csv"
    metadata = tmp_path / "metadata.json"
    char = tmp_path / "characterization.csv"
    raw.write_text("vendor", encoding="utf-8")
    # calibration loader needs area to derive the current density
    data = synthetic_polarization()
    data["timestamp_s"] = np.arange(len(data))
    data["current_A"] = data["current_density_A_m2"] * 1e-4
    data["working_electrode_area_cm2"] = 1.0
    data[["timestamp_s", "potential_V_vs_ref", "current_A", "working_electrode_area_cm2"]].to_csv(processed, index=False)
    metadata.write_text(json.dumps({
        "sample_id": "P1-1", "operator": "A", "instrument": "potentiostat", "calibration_date": "2026-07-29",
        "electrolyte_id": "bath", "working_electrode": "Fe", "counter_electrode": "Pt",
        "reference_electrode": "Ag/AgCl", "temperature_C": 60, "agitation": "RDE", "preparation": "documented",
    }), encoding="utf-8")
    char.write_text("not applicable characterization", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([["P1-1", "I", "LSV", "complete", raw.name, processed.name, metadata.name, char.name]],
                 columns=["run_id", "phase", "technique", "status", "raw_file", "processed_file", "metadata_file", "characterization_file"]).to_csv(manifest, index=False)
    report = calibrate_lsv_run(manifest, "P1-1", pH=3, temperature_C=60, fe_conc_M=1, reference_to_she_V=0.21)
    assert report["run_id"] == "P1-1"
    assert report["polarization_fit"]["converged"]
