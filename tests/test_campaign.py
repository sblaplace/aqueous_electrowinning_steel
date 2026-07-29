import json

import pandas as pd
import pytest

from models.campaign import load_manifest, validate_manifest


COLUMNS = [
    "run_id", "phase", "technique", "status", "raw_file", "processed_file", "metadata_file",
    "characterization_file",
]


def write_manifest(tmp_path, row):
    path = tmp_path / "manifest.csv"
    pd.DataFrame([row], columns=COLUMNS).to_csv(path, index=False)
    return path


def test_complete_run_with_linked_files_is_ready(tmp_path):
    metadata = {
        "sample_id": "P1-001", "operator": "A", "instrument": "potentiostat",
        "calibration_date": "2026-07-29", "electrolyte_id": "bath-1",
        "working_electrode": "Fe", "counter_electrode": "Pt", "reference_electrode": "Ag/AgCl",
        "temperature_C": 25, "agitation": "none", "preparation": "documented",
    }
    (tmp_path / "raw.csv").write_text("vendor export", encoding="utf-8")
    (tmp_path / "processed.csv").write_text("timestamp_s\n0\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "eds.csv").write_text("element,wt_percent\nFe,99\n", encoding="utf-8")
    path = write_manifest(
        tmp_path,
        [
            "P1-001", "I", "LSV", "complete", "raw.csv", "processed.csv", "metadata.json",
            "eds.csv",
        ],
    )
    report = validate_manifest(path)
    assert report["n_ready_for_analysis"] == 1
    assert report["runs"][0]["flags"] == []


def test_complete_run_missing_links_is_flagged(tmp_path):
    path = write_manifest(tmp_path, ["P1-001", "I", "LSV", "complete", "", "", "", ""])
    flags = validate_manifest(path)["runs"][0]["flags"]
    assert "missing_raw_file" in flags
    assert "missing_characterization_file" in flags


def test_planned_run_can_have_future_paths(tmp_path):
    path = write_manifest(
        tmp_path,
        [
            "P1-001", "I", "LSV", "planned", "raw.csv", "processed.csv", "metadata.json", "",
        ],
    )
    report = validate_manifest(path)
    assert report["n_ready_for_analysis"] == 0
    assert "file_not_found:raw_file" in report["runs"][0]["flags"]


def test_manifest_rejects_duplicate_run_id(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame([
        ["P1-001", "I", "LSV", "planned", "", "", "", ""],
        ["P1-001", "I", "EIS", "planned", "", "", "", ""],
    ], columns=COLUMNS).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_manifest(path)
