"""Phase I analysis helpers for CV/LSV data."""
from __future__ import annotations
import numpy as np
import pandas as pd


def segment_sweeps(data: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a run into cycle/segment groups, preserving acquisition order."""
    keys = [c for c in ("cycle", "segment") if c in data]
    if not keys:
        return [data.copy()]
    return [group.copy() for _, group in data.groupby(keys, sort=False, dropna=False)]


def baseline_correct(data: pd.DataFrame, baseline_current_A: float | None = None) -> pd.DataFrame:
    """Add baseline-corrected current columns without modifying the input."""
    out = data.copy()
    baseline = float(data["current_A"].iloc[0] if baseline_current_A is None else baseline_current_A)
    out["current_baseline_A"] = baseline
    out["current_corrected_A"] = out["current_A"] - baseline
    out["current_corrected_mA_cm2"] = out["current_corrected_A"] / out["working_electrode_area_cm2"] * 1000.0
    return out


def scan_rate_V_s(data: pd.DataFrame) -> float:
    """Estimate scan rate from potential versus time using a robust median."""
    dt = data["timestamp_s"].diff().to_numpy()
    dE = data["potential_V_vs_ref"].diff().to_numpy()
    valid = (dt > 0) & np.isfinite(dt) & np.isfinite(dE)
    if not valid.any():
        raise ValueError("At least two points with increasing timestamps are required")
    return float(np.median(np.abs(dE[valid] / dt[valid])))


def extrema(data: pd.DataFrame) -> dict:
    """Return cathodic/anodic extrema from corrected or raw current."""
    column = "current_corrected_A" if "current_corrected_A" in data else "current_A"
    cath = data.loc[data[column].idxmin()]
    anod = data.loc[data[column].idxmax()]
    return {
        "cathodic_peak_potential_V": float(cath["potential_V_vs_ref"]),
        "cathodic_peak_current_A": float(cath[column]),
        "anodic_peak_potential_V": float(anod["potential_V_vs_ref"]),
        "anodic_peak_current_A": float(anod[column]),
    }


def plot_polarization(data: pd.DataFrame, ax=None):
    """Plot current density versus potential; returns the matplotlib Axes."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    current = "current_corrected_mA_cm2" if "current_corrected_mA_cm2" in data else "current_density_mA_cm2"
    for group in segment_sweeps(data):
        ax.plot(group["potential_V_vs_ref"], group[current], linewidth=1.5)
    ax.set(xlabel="Potential (V vs reference)", ylabel="Current density (mA cm$^{-2}$)")
    ax.axhline(0, color="0.5", linewidth=0.7)
    ax.grid(alpha=0.25)
    return ax
