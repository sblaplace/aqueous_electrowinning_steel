"""Validated long-form deposit-characterization records for experimental runs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "run_id", "coupon_id", "characterization_id", "technique", "analyte", "value", "unit",
    "basis", "instrument", "calibration_date", "analysis_file",
}
VALID_TECHNIQUES = {"SEM_EDS", "COMBUSTION", "XRD"}
WEIGHT_PERCENT_TECHNIQUES = {"SEM_EDS", "COMBUSTION"}


def load_characterization(path: str | Path) -> pd.DataFrame:
    """Load a canonical SEM/EDS, combustion, or XRD characterization table.

    Every row is a measured analyte or phase.  ``analysis_file`` links the
    exported spectrum/diffraction file without copying raw instrument data.
    """
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required characterization columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Characterization file must contain at least one result")
    for column in ("run_id", "coupon_id", "characterization_id", "technique", "analyte", "unit", "basis", "instrument", "calibration_date", "analysis_file"):
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"characterization column {column} must not be blank")
    invalid = set(frame["technique"]) - VALID_TECHNIQUES
    if invalid:
        raise ValueError(f"Unsupported characterization technique(s): {', '.join(sorted(invalid))}")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if "uncertainty" in frame:
        frame["uncertainty"] = pd.to_numeric(frame["uncertainty"], errors="raise")
        if (frame["uncertainty"] < 0).any():
            raise ValueError("uncertainty must be nonnegative")
    if (frame["value"] < 0).any():
        raise ValueError("characterization values must be nonnegative")
    weight_rows = frame["technique"].isin(WEIGHT_PERCENT_TECHNIQUES)
    bad_units = weight_rows & ~frame["unit"].str.lower().isin({"wt%", "mass%"})
    if bad_units.any():
        raise ValueError("SEM_EDS and COMBUSTION composition values must use wt% or mass%")
    return frame


def summarize_characterization(data: pd.DataFrame) -> dict:
    """Summarize composition coverage and retain QA flags instead of normalizing data."""
    if data.empty:
        raise ValueError("Cannot summarize an empty characterization table")
    composition = data[data["technique"].isin(WEIGHT_PERCENT_TECHNIQUES)].copy()
    totals = composition.groupby(["run_id", "coupon_id", "technique"])["value"].sum()
    flags: list[str] = []
    # EDS values are often normalized by software; combustion carbon is an
    # independent result. Never combine technique totals or force them to 100%.
    eds_totals = totals[totals.index.get_level_values("technique") == "SEM_EDS"]
    for index, total in eds_totals.items():
        if not 95.0 <= total <= 105.0:
            flags.append(f"eds_total_outside_95_105:{'/'.join(map(str, index))}={total:.3g}")
    return {
        "n_records": int(len(data)),
        "run_ids": sorted(data["run_id"].unique().tolist()),
        "techniques": sorted(data["technique"].unique().tolist()),
        "composition_totals_wt_percent": {"/".join(map(str, key)): float(value) for key, value in totals.items()},
        "quality_flags": flags,
        "note": "SEM/EDS, combustion, and XRD are retained as separate measurements; no cross-technique normalization is applied.",
    }
