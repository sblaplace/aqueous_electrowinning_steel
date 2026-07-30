"""Phase II Hull-cell screening and gravimetric Faradaic-efficiency tools.

The Hull-cell calculation is a deliberately transparent *primary current*
model for an angled cathode opposite a planar anode.  It treats every narrow
cathode strip as an ohmic path through its local solution gap, so that
``j(s) ∝ 1 / gap(s)``.  The strip currents are normalized to the measured
applied current.  This makes the model useful for assigning coupon positions
and interpreting a Hull-panel screen, but it is not a replacement for a
calibrated cell map: edge effects, shields, anode shape, electrode kinetics,
mass transport, bubbles, and conductivity gradients are intentionally out of
scope.

Gravimetric efficiency is reported as *apparent Fe Faradaic efficiency*: net
coupon mass gain divided by the Fe mass predicted from the cathodic charge.
It is only an Fe efficiency when the dry deposit is verified as iron.  Retained
salts, oxides, codeposits, and incomplete drying can bias a mass-only result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import asin, degrees, log, sqrt
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


from .electrochemistry import FARADAY as FARADAY_CONSTANT_C_MOL, M_FE_G as MOLAR_MASS_FE_G_MOL, Z_FE as ELECTRONS_PER_FE

GALVANOSTATIC_REQUIRED_COLUMNS = {"timestamp_s", "current_A"}
GRAVIMETRY_REQUIRED_COLUMNS = {"mass_before_g", "mass_after_g"}

CathodicSign = Literal["negative", "positive"]


@dataclass(frozen=True)
class HullCellGeometry:
    """Physical dimensions for the variable-gap primary-current model.

    Parameters are measured along the cathode panel.  ``near_edge_gap_cm`` and
    ``far_edge_gap_cm`` are the perpendicular anode--cathode separations at
    the near and far ends, respectively.  Their difference creates the panel
    angle relative to the planar anode.
    """

    panel_length_cm: float = 10.0
    panel_width_cm: float = 5.0
    near_edge_gap_cm: float = 1.5
    far_edge_gap_cm: float = 9.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.far_edge_gap_cm < self.near_edge_gap_cm:
            raise ValueError(
                "far_edge_gap_cm must be at least near_edge_gap_cm; "
                "define the near edge as the edge closest to the anode"
            )
        # Δgap = L sin(theta).  The 90° limit is excluded because the
        # local parallel-strip approximation degenerates there.
        if self.far_edge_gap_cm - self.near_edge_gap_cm >= self.panel_length_cm:
            raise ValueError(
                "far_edge_gap_cm - near_edge_gap_cm must be less than panel_length_cm"
            )

    @property
    def panel_area_cm2(self) -> float:
        """Geometric cathode area exposed to the electrolyte."""
        return self.panel_length_cm * self.panel_width_cm

    @property
    def panel_angle_deg(self) -> float:
        """Cathode angle relative to the anode plane (0° is parallel)."""
        ratio = (self.far_edge_gap_cm - self.near_edge_gap_cm) / self.panel_length_cm
        return degrees(asin(np.clip(ratio, -1.0, 1.0)))

    def gap_cm(self, position_cm_from_near_edge: float | np.ndarray) -> np.ndarray:
        """Return local anode--cathode separation at panel position(s)."""
        position = np.asarray(position_cm_from_near_edge, dtype=float)
        if np.any(~np.isfinite(position)):
            raise ValueError("Panel position must be finite")
        if np.any(position < 0) or np.any(position > self.panel_length_cm):
            raise ValueError("Panel position must lie between 0 and panel_length_cm")
        slope = (self.far_edge_gap_cm - self.near_edge_gap_cm) / self.panel_length_cm
        return self.near_edge_gap_cm + slope * position


def hull_current_distribution(
    geometry: HullCellGeometry,
    total_current_A: float,
    n_segments: int = 100,
) -> pd.DataFrame:
    """Calculate a normalized primary-current map over an angled Hull panel.

    The calculation integrates the conductance of each panel strip exactly
    under the variable-gap approximation.  The returned ``segment_current_A``
    therefore sums to ``total_current_A`` (up to floating-point roundoff),
    rather than merely matching it after a sampled numerical integration.

    Parameters
    ----------
    geometry:
        Panel dimensions and near/far solution gaps.
    total_current_A:
        Positive magnitude of the externally applied cathodic current.
    n_segments:
        Number of equal-length panel strips.  Each row represents one strip
        centered at ``position_cm_from_near_edge``.
    """
    if not np.isfinite(total_current_A) or total_current_A <= 0:
        raise ValueError("total_current_A must be finite and positive")
    if isinstance(n_segments, bool) or int(n_segments) != n_segments or n_segments < 2:
        raise ValueError("n_segments must be an integer of at least 2")
    n_segments = int(n_segments)

    edges = np.linspace(0.0, geometry.panel_length_cm, n_segments + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    gap_edges = geometry.gap_cm(edges)
    gap_centers = geometry.gap_cm(centers)

    gap_change = geometry.far_edge_gap_cm - geometry.near_edge_gap_cm
    if np.isclose(gap_change, 0.0):
        # Parallel plates are the continuous limiting case of log(g_far/g_near).
        strip_current_fraction = widths / geometry.panel_length_cm
        cumulative_fraction = edges[1:] / geometry.panel_length_cm
    else:
        log_ratio = log(geometry.far_edge_gap_cm / geometry.near_edge_gap_cm)
        strip_current_fraction = np.log(gap_edges[1:] / gap_edges[:-1]) / log_ratio
        cumulative_fraction = np.log(
            gap_edges[1:] / geometry.near_edge_gap_cm
        ) / log_ratio

    segment_area_cm2 = geometry.panel_width_cm * widths
    segment_current_A = total_current_A * strip_current_fraction
    current_density_A_cm2 = segment_current_A / segment_area_cm2
    panel_average_A_cm2 = total_current_A / geometry.panel_area_cm2

    return pd.DataFrame({
        "position_cm_from_near_edge": centers,
        "position_fraction_from_near_edge": centers / geometry.panel_length_cm,
        "local_gap_cm": gap_centers,
        "segment_gap_near_edge_cm": gap_edges[:-1],
        "segment_gap_far_edge_cm": gap_edges[1:],
        "segment_width_cm": widths,
        "segment_area_cm2": segment_area_cm2,
        "segment_current_A": segment_current_A,
        "current_density_A_cm2": current_density_A_cm2,
        "current_density_mA_cm2": current_density_A_cm2 * 1000.0,
        "relative_to_panel_average": current_density_A_cm2 / panel_average_A_cm2,
        "cumulative_current_fraction_at_far_edge": cumulative_fraction,
    })


def summarize_hull_distribution(distribution: pd.DataFrame) -> dict:
    """Summarize a Hull-panel current map with explicit, unit-labelled keys."""
    required = {
        "segment_area_cm2", "segment_current_A", "current_density_mA_cm2",
        "local_gap_cm",
    }
    missing = required - set(distribution.columns)
    if missing:
        raise ValueError(f"Distribution is missing columns: {', '.join(sorted(missing))}")
    if distribution.empty:
        raise ValueError("Cannot summarize an empty Hull-cell distribution")
    total_current = float(distribution["segment_current_A"].sum())
    total_area = float(distribution["segment_area_cm2"].sum())
    return {
        "n_segments": int(len(distribution)),
        "panel_area_cm2": total_area,
        "total_current_A": total_current,
        "panel_average_current_density_mA_cm2": total_current / total_area * 1000.0,
        "near_edge_current_density_mA_cm2": float(
            distribution["current_density_mA_cm2"].iloc[0]
        ),
        "far_edge_current_density_mA_cm2": float(
            distribution["current_density_mA_cm2"].iloc[-1]
        ),
        "current_density_min_mA_cm2": float(distribution["current_density_mA_cm2"].min()),
        "current_density_max_mA_cm2": float(distribution["current_density_mA_cm2"].max()),
        "near_edge_gap_cm": float(
            distribution.get("segment_gap_near_edge_cm", distribution["local_gap_cm"]).iloc[0]
        ),
        "far_edge_gap_cm": float(
            distribution.get("segment_gap_far_edge_cm", distribution["local_gap_cm"]).iloc[-1]
        ),
    }


def current_density_window(
    distribution: pd.DataFrame,
    minimum_mA_cm2: float,
    maximum_mA_cm2: float,
) -> dict:
    """Report panel coverage that lies in an experimentally useful j window.

    Coverage is evaluated strip-by-strip, so the position endpoints are the
    bounds of the first and last accepted strips.  Increase ``n_segments`` in
    :func:`hull_current_distribution` when finer placement resolution is
    needed.
    """
    if not (np.isfinite(minimum_mA_cm2) and np.isfinite(maximum_mA_cm2)):
        raise ValueError("Current-density limits must be finite")
    if minimum_mA_cm2 < 0 or maximum_mA_cm2 <= minimum_mA_cm2:
        raise ValueError("Require 0 <= minimum_mA_cm2 < maximum_mA_cm2")
    required = {
        "current_density_mA_cm2", "segment_area_cm2", "segment_current_A",
        "position_cm_from_near_edge", "segment_width_cm",
    }
    missing = required - set(distribution.columns)
    if missing:
        raise ValueError(f"Distribution is missing columns: {', '.join(sorted(missing))}")

    selected = distribution.loc[
        distribution["current_density_mA_cm2"].between(minimum_mA_cm2, maximum_mA_cm2)
    ]
    total_area = float(distribution["segment_area_cm2"].sum())
    total_current = float(distribution["segment_current_A"].sum())
    if selected.empty:
        return {
            "minimum_mA_cm2": float(minimum_mA_cm2),
            "maximum_mA_cm2": float(maximum_mA_cm2),
            "n_segments": 0,
            "area_fraction": 0.0,
            "current_fraction": 0.0,
            "position_start_cm_from_near_edge": None,
            "position_end_cm_from_near_edge": None,
        }

    first, last = selected.iloc[0], selected.iloc[-1]
    return {
        "minimum_mA_cm2": float(minimum_mA_cm2),
        "maximum_mA_cm2": float(maximum_mA_cm2),
        "n_segments": int(len(selected)),
        "area_fraction": float(selected["segment_area_cm2"].sum() / total_area),
        "current_fraction": float(selected["segment_current_A"].sum() / total_current),
        "position_start_cm_from_near_edge": float(
            first["position_cm_from_near_edge"] - 0.5 * first["segment_width_cm"]
        ),
        "position_end_cm_from_near_edge": float(
            last["position_cm_from_near_edge"] + 0.5 * last["segment_width_cm"]
        ),
    }


def plot_hull_current_distribution(
    distribution: pd.DataFrame,
    geometry: HullCellGeometry,
    axes=None,
):
    """Plot an angled-panel schematic and its primary current-density map.

    ``axes`` may be a two-element array of Matplotlib axes.  The function
    returns the axes so callers can add experiment-specific annotations.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    required = {"position_cm_from_near_edge", "current_density_mA_cm2"}
    missing = required - set(distribution.columns)
    if missing:
        raise ValueError(f"Distribution is missing columns: {', '.join(sorted(missing))}")
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if len(axes) != 2:
        raise ValueError("axes must contain exactly two Matplotlib axes")
    ax_schematic, ax_density = axes

    theta = np.radians(geometry.panel_angle_deg)
    s_edges = np.linspace(0.0, geometry.panel_length_cm, len(distribution) + 1)
    x_edges = geometry.gap_cm(s_edges)
    y_edges = s_edges * np.cos(theta)
    points = np.column_stack([x_edges, y_edges])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    values = distribution["current_density_mA_cm2"].to_numpy(float)
    norm = plt.Normalize(values.min(), values.max())
    collection = LineCollection(segments, cmap="viridis", norm=norm, linewidth=8)
    collection.set_array(values)
    ax_schematic.add_collection(collection)
    anode_height = geometry.panel_length_cm * np.cos(theta)
    ax_schematic.plot([0, 0], [0, anode_height], color="#444444", linewidth=6,
                      solid_capstyle="butt", label="planar anode")
    ax_schematic.annotate(
        "near edge", xy=(x_edges[0], y_edges[0]), xytext=(x_edges[0] + 0.8, -0.9),
        arrowprops={"arrowstyle": "-", "color": "0.35"}, fontsize=9,
    )
    ax_schematic.annotate(
        "far edge", xy=(x_edges[-1], y_edges[-1]),
        xytext=(x_edges[-1] + 0.2, y_edges[-1] + 0.65),
        arrowprops={"arrowstyle": "-", "color": "0.35"}, fontsize=9,
    )
    ax_schematic.set(
        title=("Angled cathode — primary-current model\n"
               f"panel angle {geometry.panel_angle_deg:.1f}°"),
        xlabel="Perpendicular gap from anode (cm)", ylabel="Height along anode (cm)",
    )
    ax_schematic.set_aspect("equal", adjustable="box")
    ax_schematic.grid(alpha=0.2)
    colorbar = ax_schematic.figure.colorbar(collection, ax=ax_schematic, pad=0.02)
    colorbar.set_label("Cathode j (mA cm$^{-2}$)")

    ax_density.plot(
        distribution["position_cm_from_near_edge"], values, color="#1874b4", linewidth=2,
    )
    ax_density.fill_between(
        distribution["position_cm_from_near_edge"], values, alpha=0.18, color="#1874b4",
    )
    panel_average = float(
        distribution["segment_current_A"].sum() /
        distribution["segment_area_cm2"].sum() * 1000.0
    )
    ax_density.axhline(panel_average, color="0.35", linestyle="--", linewidth=1,
                       label=f"area average = {panel_average:.1f} mA cm$^{{-2}}$")
    ax_density.set(
        title="Local current-density distribution",
        xlabel="Distance from near panel edge (cm)",
        ylabel="Current density (mA cm$^{-2}$)",
        xlim=(0, geometry.panel_length_cm),
    )
    ax_density.grid(alpha=0.25)
    ax_density.legend(fontsize=8)
    return axes


def load_galvanostatic_trace(path: str | Path) -> pd.DataFrame:
    """Load a galvanostatic trace and derive signed current-density columns.

    The canonical schema is ``timestamp_s,current_A`` plus optional cell
    voltage and bath metadata.  Current follows the repository convention:
    cathodic current is negative.  The loader does not force a sign so that
    instrument exports can be reviewed before assigning the convention in the
    FE calculation.
    """
    frame = pd.read_csv(path)
    missing = GALVANOSTATIC_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    numeric = {
        "timestamp_s", "current_A", "cell_voltage_V", "working_electrode_area_cm2",
        "temperature_C", "pH", "fe2_concentration_M",
    }
    for column in numeric & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty:
        raise ValueError("Galvanostatic trace cannot be empty")
    if not np.isfinite(frame[["timestamp_s", "current_A"]].to_numpy(float)).all():
        raise ValueError("timestamp_s and current_A must be finite")
    if (frame["timestamp_s"].diff().dropna() < 0).any():
        raise ValueError("timestamp_s must be non-decreasing")
    if "working_electrode_area_cm2" in frame.columns:
        if (frame["working_electrode_area_cm2"] <= 0).any():
            raise ValueError("working_electrode_area_cm2 must be positive")
        frame["current_density_A_m2"] = (
            frame["current_A"] / (frame["working_electrode_area_cm2"] * 1e-4)
        )
        frame["current_density_mA_cm2"] = (
            frame["current_A"] / frame["working_electrode_area_cm2"] * 1000.0
        )
    return frame


def load_gravimetry(path: str | Path) -> pd.DataFrame:
    """Load coupon before/after masses for a gravimetric FE determination."""
    frame = pd.read_csv(path)
    missing = GRAVIMETRY_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    numeric = {
        "mass_before_g", "mass_after_g", "blank_mass_change_g",
        "mass_uncertainty_g", "blank_mass_uncertainty_g", "electrode_area_cm2",
    }
    for column in numeric & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty:
        raise ValueError("Gravimetry table cannot be empty")
    if not np.isfinite(frame[["mass_before_g", "mass_after_g"]].to_numpy(float)).all():
        raise ValueError("mass_before_g and mass_after_g must be finite")
    if (frame[["mass_before_g", "mass_after_g"]] < 0).any().any():
        raise ValueError("Recorded masses must be non-negative")
    for name in ("mass_uncertainty_g", "blank_mass_uncertainty_g"):
        if name in frame.columns and (frame[name] < 0).any():
            raise ValueError(f"{name} must be non-negative")
    return frame


def cathodic_charge_C(
    timestamp_s: np.ndarray | pd.Series,
    current_A: np.ndarray | pd.Series,
    cathodic_sign: CathodicSign = "negative",
) -> float:
    """Integrate only cathodic charge from a time/current trace.

    Current is clipped before trapezoidal integration, excluding reverse or
    anodic portions of pulse-reverse waveforms from the reduction charge.
    ``cathodic_sign='negative'`` matches the repository's electrochemical
    data convention; use ``'positive'`` only for exports that declare that
    convention explicitly.
    """
    time = np.asarray(timestamp_s, dtype=float)
    current = np.asarray(current_A, dtype=float)
    if time.ndim != 1 or current.ndim != 1 or len(time) != len(current):
        raise ValueError("timestamp_s and current_A must be one-dimensional and equal length")
    if len(time) < 2:
        raise ValueError("At least two time points are required to integrate charge")
    if not np.isfinite(time).all() or not np.isfinite(current).all():
        raise ValueError("timestamp_s and current_A must be finite")
    if (np.diff(time) < 0).any():
        raise ValueError("timestamp_s must be non-decreasing")
    if time[-1] <= time[0]:
        raise ValueError("Trace duration must be positive")
    if cathodic_sign == "negative":
        cathodic_current = np.clip(-current, 0.0, None)
    elif cathodic_sign == "positive":
        cathodic_current = np.clip(current, 0.0, None)
    else:
        raise ValueError("cathodic_sign must be 'negative' or 'positive'")
    charge = float(np.trapezoid(cathodic_current, time))
    if charge <= 0:
        raise ValueError(
            "No cathodic charge was found; confirm current sign and trace contents"
        )
    return charge


@dataclass(frozen=True)
class GravimetricFEResult:
    """Mass-balance result for one gravimetric iron-deposition run."""

    duration_s: float
    cathodic_charge_C: float
    equivalent_mean_cathodic_current_A: float
    mass_gain_g: float
    blank_mass_change_g: float
    net_deposit_mass_g: float
    theoretical_fe_mass_g: float
    apparent_faradaic_efficiency: float
    apparent_faradaic_efficiency_percent: float
    apparent_faradaic_efficiency_uncertainty_percent: float | None
    net_deposit_mass_uncertainty_g: float | None
    charge_relative_uncertainty: float
    n_electrons: int
    molar_mass_g_mol: float

    def summary(self) -> dict:
        """Return JSON-ready results with unit-labelled keys."""
        return asdict(self)


def gravimetric_faradaic_efficiency(
    timestamp_s: np.ndarray | pd.Series,
    current_A: np.ndarray | pd.Series,
    mass_before_g: float,
    mass_after_g: float,
    *,
    blank_mass_change_g: float = 0.0,
    mass_uncertainty_g: float | None = None,
    blank_mass_uncertainty_g: float = 0.0,
    charge_relative_uncertainty: float = 0.0,
    cathodic_sign: CathodicSign = "negative",
    n_electrons: int = ELECTRONS_PER_FE,
    molar_mass_g_mol: float = MOLAR_MASS_FE_G_MOL,
) -> GravimetricFEResult:
    """Calculate apparent Fe Faradaic efficiency from charge and dry mass gain.

    ``blank_mass_change_g`` is subtracted from coupon mass gain.  It can be
    used for a matched unpowered coupon or a validated handling blank.  Pass
    a per-weighing one-standard-deviation ``mass_uncertainty_g`` to propagate
    mass and optional blank uncertainty into the reported efficiency.

    The calculation intentionally does not cap efficiency at 100%; values
    above 100% are a useful QA flag for retained electrolyte, oxidation,
    incorrect charge/sign handling, or balance/drying errors.
    """
    for name, value in {
        "mass_before_g": mass_before_g,
        "mass_after_g": mass_after_g,
        "blank_mass_change_g": blank_mass_change_g,
        "blank_mass_uncertainty_g": blank_mass_uncertainty_g,
        "charge_relative_uncertainty": charge_relative_uncertainty,
        "molar_mass_g_mol": molar_mass_g_mol,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if mass_before_g < 0 or mass_after_g < 0:
        raise ValueError("Recorded masses must be non-negative")
    if blank_mass_uncertainty_g < 0 or charge_relative_uncertainty < 0:
        raise ValueError("Uncertainties must be non-negative")
    if mass_uncertainty_g is not None:
        if not np.isfinite(mass_uncertainty_g) or mass_uncertainty_g < 0:
            raise ValueError("mass_uncertainty_g must be finite and non-negative")
    if isinstance(n_electrons, bool) or int(n_electrons) != n_electrons or n_electrons <= 0:
        raise ValueError("n_electrons must be a positive integer")
    if molar_mass_g_mol <= 0:
        raise ValueError("molar_mass_g_mol must be positive")

    time = np.asarray(timestamp_s, dtype=float)
    charge_C = cathodic_charge_C(time, current_A, cathodic_sign=cathodic_sign)
    duration_s = float(time[-1] - time[0])
    theoretical_mass_g = charge_C * molar_mass_g_mol / (n_electrons * FARADAY_CONSTANT_C_MOL)
    mass_gain_g = float(mass_after_g - mass_before_g)
    net_deposit_mass_g = mass_gain_g - float(blank_mass_change_g)
    efficiency = net_deposit_mass_g / theoretical_mass_g

    net_mass_uncertainty = None
    efficiency_uncertainty_percent = None
    if mass_uncertainty_g is not None:
        net_mass_uncertainty = sqrt(
            2.0 * mass_uncertainty_g ** 2 + blank_mass_uncertainty_g ** 2
        )
        # sigma(net/theoretical) for independent mass and charge errors.
        sigma_efficiency = sqrt(
            (net_mass_uncertainty / theoretical_mass_g) ** 2 +
            (efficiency * charge_relative_uncertainty) ** 2
        )
        efficiency_uncertainty_percent = 100.0 * sigma_efficiency

    return GravimetricFEResult(
        duration_s=duration_s,
        cathodic_charge_C=charge_C,
        equivalent_mean_cathodic_current_A=charge_C / duration_s,
        mass_gain_g=mass_gain_g,
        blank_mass_change_g=float(blank_mass_change_g),
        net_deposit_mass_g=net_deposit_mass_g,
        theoretical_fe_mass_g=theoretical_mass_g,
        apparent_faradaic_efficiency=efficiency,
        apparent_faradaic_efficiency_percent=100.0 * efficiency,
        apparent_faradaic_efficiency_uncertainty_percent=efficiency_uncertainty_percent,
        net_deposit_mass_uncertainty_g=net_mass_uncertainty,
        charge_relative_uncertainty=float(charge_relative_uncertainty),
        n_electrons=int(n_electrons),
        molar_mass_g_mol=float(molar_mass_g_mol),
    )


def analyze_gravimetric_efficiency(
    trace: pd.DataFrame,
    gravimetry: pd.DataFrame,
    *,
    row: int = 0,
    cathodic_sign: CathodicSign = "negative",
    charge_relative_uncertainty: float = 0.0,
) -> GravimetricFEResult:
    """Join a loaded current trace to one gravimetry row and calculate FE."""
    trace_missing = GALVANOSTATIC_REQUIRED_COLUMNS - set(trace.columns)
    if trace_missing:
        raise ValueError(f"Trace is missing columns: {', '.join(sorted(trace_missing))}")
    mass_missing = GRAVIMETRY_REQUIRED_COLUMNS - set(gravimetry.columns)
    if mass_missing:
        raise ValueError(f"Gravimetry is missing columns: {', '.join(sorted(mass_missing))}")
    if row < 0 or row >= len(gravimetry):
        raise IndexError("gravimetry row is outside the table")
    measurement = gravimetry.iloc[row]
    return gravimetric_faradaic_efficiency(
        trace["timestamp_s"],
        trace["current_A"],
        float(measurement["mass_before_g"]),
        float(measurement["mass_after_g"]),
        blank_mass_change_g=float(measurement.get("blank_mass_change_g", 0.0)),
        mass_uncertainty_g=(
            float(measurement["mass_uncertainty_g"])
            if "mass_uncertainty_g" in gravimetry.columns else None
        ),
        blank_mass_uncertainty_g=float(measurement.get("blank_mass_uncertainty_g", 0.0)),
        charge_relative_uncertainty=charge_relative_uncertainty,
        cathodic_sign=cathodic_sign,
    )
