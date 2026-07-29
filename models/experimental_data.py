"""Utilities for loading and validating electrochemical experiment data.

The CSV format is intentionally long-form: one row per measurement, with
experiment metadata repeated in a sidecar JSON file or supplied as columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd

REQUIRED_COLUMNS = {
    "timestamp_s", "potential_V_vs_ref", "current_A", "working_electrode_area_cm2",
}
OPTIONAL_COLUMNS = {
    "cycle", "segment", "temperature_C", "pH", "fe2_concentration_M",
    "electrolyte_id", "reference_electrode", "notes",
}


def load_measurements(path: str | Path) -> pd.DataFrame:
    """Load a measurement CSV and normalize/validate its numeric columns."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    numeric = ["timestamp_s", "potential_V_vs_ref", "current_A",
               "working_electrode_area_cm2", "cycle", "temperature_C", "pH",
               "fe2_concentration_M"]
    for column in set(numeric) & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["working_electrode_area_cm2"] <= 0).any():
        raise ValueError("working_electrode_area_cm2 must be positive")
    if (frame["timestamp_s"].diff().dropna() < 0).any():
        raise ValueError("timestamp_s must be non-decreasing")
    frame["current_density_A_m2"] = frame["current_A"] / (frame["working_electrode_area_cm2"] * 1e-4)
    frame["current_density_mA_cm2"] = frame["current_A"] / frame["working_electrode_area_cm2"] * 100.0
    return frame


def summarize_run(data: pd.DataFrame) -> dict:
    """Return basic, unit-labelled metrics for a loaded run."""
    if data.empty:
        raise ValueError("Cannot summarize an empty run")
    return {
        "n_points": int(len(data)),
        "duration_s": float(data["timestamp_s"].iloc[-1] - data["timestamp_s"].iloc[0]),
        "potential_min_V": float(data["potential_V_vs_ref"].min()),
        "potential_max_V": float(data["potential_V_vs_ref"].max()),
        "current_density_mean_mA_cm2": float(data["current_density_mA_cm2"].mean()),
        "current_density_max_mA_cm2": float(data["current_density_mA_cm2"].max()),
    }
