import pandas as pd
import pytest

from models.characterization import load_characterization, summarize_characterization


COLUMNS = [
    "run_id", "coupon_id", "characterization_id", "technique", "analyte", "value", "unit",
    "uncertainty", "basis", "instrument", "calibration_date", "analysis_file",
]


def test_load_and_summarize_eds_composition(tmp_path):
    path = tmp_path / "characterization.csv"
    pd.DataFrame([
        ["P2-1", "C1", "EDS-1", "SEM_EDS", "Fe", 98.0, "wt%", 0.5, "area", "EDS", "2026-07-29", "raw/a.csv"],
        ["P2-1", "C1", "EDS-1", "SEM_EDS", "O", 2.0, "wt%", 0.2, "area", "EDS", "2026-07-29", "raw/a.csv"],
        ["P2-1", "C1", "COMB-1", "COMBUSTION", "C", 0.2, "wt%", 0.02, "bulk", "LECO", "2026-07-29", "raw/c.csv"],
    ], columns=COLUMNS).to_csv(path, index=False)
    summary = summarize_characterization(load_characterization(path))
    assert summary["quality_flags"] == []
    assert summary["composition_totals_wt_percent"]["P2-1/C1/SEM_EDS"] == pytest.approx(100)


def test_eds_total_is_flagged_not_renormalized(tmp_path):
    path = tmp_path / "characterization.csv"
    pd.DataFrame([
        ["P2-1", "C1", "EDS-1", "SEM_EDS", "Fe", 80.0, "wt%", 0.5, "area", "EDS", "2026-07-29", "raw/a.csv"],
    ], columns=COLUMNS).to_csv(path, index=False)
    summary = summarize_characterization(load_characterization(path))
    assert summary["quality_flags"][0].startswith("eds_total_outside")


def test_composition_requires_weight_percent_units(tmp_path):
    path = tmp_path / "characterization.csv"
    pd.DataFrame([
        ["P2-1", "C1", "EDS-1", "SEM_EDS", "Fe", 99.0, "ppm", 0.5, "area", "EDS", "2026-07-29", "raw/a.csv"],
    ], columns=COLUMNS).to_csv(path, index=False)
    with pytest.raises(ValueError, match="wt%"):
        load_characterization(path)
