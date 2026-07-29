"""Tafel-region selection and linear fitting for LSV data."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TafelFit:
    slope_V_decade: float
    intercept_log10_A: float
    exchange_current_A: float
    r_squared: float
    n_points: int
    potential_min_V: float
    potential_max_V: float


def fit_tafel(data: pd.DataFrame, *, potential_min_V: float, potential_max_V: float,
              equilibrium_potential_V: float = 0.0,
              current_column: str = "current_A") -> TafelFit:
    """Fit log10(cathodic current) against cathodic overpotential.

    Potentials in the selected interval must be on one monotonic cathodic
    branch. Current is treated as a magnitude, so either signed convention is
    accepted. The intercept is extrapolated to zero overpotential and is i0.
    """
    if potential_max_V <= potential_min_V:
        raise ValueError("potential_max_V must exceed potential_min_V")
    required = {"potential_V_vs_ref", current_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    selected = data[data["potential_V_vs_ref"].between(potential_min_V, potential_max_V)].copy()
    eta = equilibrium_potential_V - selected["potential_V_vs_ref"].to_numpy(float)
    current = np.abs(selected[current_column].to_numpy(float))
    valid = np.isfinite(eta) & np.isfinite(current) & (current > 0)
    if valid.sum() < 3:
        raise ValueError("Tafel fit requires at least three positive-current points")
    x, y = eta[valid], np.log10(current[valid])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return TafelFit(float(1.0 / slope), float(intercept), float(10 ** intercept),
                    float(r2), int(valid.sum()), float(selected["potential_V_vs_ref"].min()),
                    float(selected["potential_V_vs_ref"].max()))
