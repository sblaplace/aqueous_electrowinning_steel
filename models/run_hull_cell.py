#!/usr/bin/env python3
"""Phase II driver: Hull-panel primary current map and gravimetric FE.

This driver produces a clearly labelled synthetic example; it does not
represent wet-lab deposition or microscopy data.

Generates:
    experiments/data/synthetic_hull_cell_galvanostatic.csv
    experiments/data/synthetic_hull_cell_gravimetry.csv
    experiments/data/hull_cell_report.json
    docs/figures/hull_cell_current_distribution.png
    docs/figures/gravimetric_faradaic_efficiency.png

Usage:
    python -m models.run_hull_cell
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.hull_cell import (
    FARADAY_CONSTANT_C_MOL,
    MOLAR_MASS_FE_G_MOL,
    HullCellGeometry,
    analyze_gravimetric_efficiency,
    current_density_window,
    hull_current_distribution,
    load_galvanostatic_trace,
    load_gravimetry,
    plot_hull_current_distribution,
    summarize_hull_distribution,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def generate_synthetic_galvanostatic_trace(
    filepath: Path,
    duration_s: float = 3600.0,
    nominal_current_A: float = -1.50,
    n_points: int = 121,
) -> pd.DataFrame:
    """Generate a sign-convention-labelled synthetic Phase II current trace."""
    if duration_s <= 0 or nominal_current_A >= 0 or n_points < 2:
        raise ValueError("Use positive duration, negative cathodic current, and at least two points")
    time_s = np.linspace(0.0, duration_s, n_points)
    # Gentle deterministic control ripple; no random seed or hidden variability.
    current_A = nominal_current_A * (1.0 + 0.002 * np.sin(2.0 * np.pi * time_s / 600.0))
    cell_voltage_V = 2.62 + 0.018 * np.sin(2.0 * np.pi * time_s / 900.0)
    trace = pd.DataFrame({
        "timestamp_s": time_s,
        "current_A": current_A,
        "cell_voltage_V": cell_voltage_V,
        "working_electrode_area_cm2": 50.0,
        "temperature_C": 60.0,
        "pH": 3.0,
        "fe2_concentration_M": 1.0,
        "electrolyte_id": "FE-SO4-SYNTHETIC",
        "current_sign_convention": "cathodic_negative",
        "notes": "Synthetic 1 h galvanostatic Hull-panel example; not wet-lab data",
    })
    trace.to_csv(filepath, index=False)
    return trace


def generate_synthetic_gravimetry(
    filepath: Path,
    trace: pd.DataFrame,
    target_apparent_fe: float = 0.91,
    blank_mass_change_g: float = 0.0012,
) -> pd.DataFrame:
    """Generate synthetic pre/post masses consistent with a known apparent FE."""
    if not 0 < target_apparent_fe <= 1:
        raise ValueError("target_apparent_fe must lie in (0, 1]")
    time = trace["timestamp_s"].to_numpy(float)
    cathodic_current = np.clip(-trace["current_A"].to_numpy(float), 0.0, None)
    charge_C = float(np.trapezoid(cathodic_current, time))
    theoretical_mass_g = charge_C * MOLAR_MASS_FE_G_MOL / (2 * FARADAY_CONSTANT_C_MOL)
    mass_before_g = 25.0000
    mass_after_g = mass_before_g + target_apparent_fe * theoretical_mass_g + blank_mass_change_g
    gravimetry = pd.DataFrame({
        "run_id": ["SYNTHETIC-HULL-001"],
        "coupon_id": ["HULL-PANEL-01"],
        "mass_before_g": [mass_before_g],
        "mass_after_g": [mass_after_g],
        "blank_mass_change_g": [blank_mass_change_g],
        "mass_uncertainty_g": [0.0001],
        "blank_mass_uncertainty_g": [0.0001],
        "electrode_area_cm2": [50.0],
        "drying_protocol": ["Synthetic example only; actual coupons require validated rinse/dry protocol"],
        "notes": ["Synthetic masses constructed for 91% apparent gravimetric FE"],
    })
    gravimetry.to_csv(filepath, index=False)
    return gravimetry


def _plot_gravimetric_result(trace: pd.DataFrame, result, path: Path) -> None:
    """Write the synthetic trace and mass-balance diagnostic figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))

    ax = axes[0]
    time_min = trace["timestamp_s"] / 60.0
    ax.plot(time_min, trace["current_A"], color="#1874b4", linewidth=1.8,
            label="Cathode current")
    ax.set(xlabel="Time (min)", ylabel="Current (A)",
           title="Synthetic galvanostatic trace (cathodic negative)")
    ax.grid(alpha=0.25)
    voltage_axis = ax.twinx()
    voltage_axis.plot(time_min, trace["cell_voltage_V"], color="#d95f02", linewidth=1.3,
                      label="Cell voltage")
    voltage_axis.set_ylabel("Cell voltage (V)")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = voltage_axis.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="best")

    ax = axes[1]
    labels = ["Faradaic\ntheory", "Net dry\nmass gain"]
    masses = [result.theoretical_fe_mass_g, result.net_deposit_mass_g]
    bars = ax.bar(labels, masses, color=["#a6cee3", "#33a02c"], width=0.58)
    for bar, mass in zip(bars, masses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                f"{mass:.3f} g", ha="center", va="bottom", fontsize=10)
    uncertainty = result.apparent_faradaic_efficiency_uncertainty_percent
    uncertainty_text = ""
    if uncertainty is not None:
        uncertainty_text = f" ± {uncertainty:.2f}% (mass/charge input)"
    ax.text(0.5, 0.96,
            f"Apparent gravimetric FE = {result.apparent_faradaic_efficiency_percent:.1f}%"
            f"{uncertainty_text}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.6"})
    ax.set(ylabel="Iron mass (g)", title="Charge-to-mass balance")
    ax.set_ylim(0.0, max(masses) * 1.28)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Phase II gravimetric Faradaic efficiency — synthetic example", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Generate reproducible synthetic Phase II outputs and a JSON report."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PHASE II — HULL-CELL CURRENT MAP & GRAVIMETRIC FARADAIC EFFICIENCY")
    print("=" * 70)

    # Example 10 × 5 cm panel; a 1.5 -> 9 cm gap produces a 48.6° panel angle.
    geometry = HullCellGeometry(
        panel_length_cm=10.0,
        panel_width_cm=5.0,
        near_edge_gap_cm=1.5,
        far_edge_gap_cm=9.0,
    )
    applied_panel_current_A = 1.0
    distribution = hull_current_distribution(geometry, applied_panel_current_A, n_segments=100)
    distribution_summary = summarize_hull_distribution(distribution)
    operating_window = current_density_window(distribution, 10.0, 100.0)
    print("\nPrimary-current Hull-panel model (screening calculation):")
    print(f"  Panel: {geometry.panel_length_cm:.1f} × {geometry.panel_width_cm:.1f} cm "
          f"at {geometry.panel_angle_deg:.1f}° to the anode")
    print(f"  Applied current: {applied_panel_current_A:.2f} A")
    print("  Local j: "
          f"{distribution_summary['near_edge_current_density_mA_cm2']:.1f} → "
          f"{distribution_summary['far_edge_current_density_mA_cm2']:.1f} mA/cm² "
          "(near → far)")
    print(f"  10–100 mA/cm² coverage: {operating_window['area_fraction'] * 100:.1f}% of panel area")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    plot_hull_current_distribution(distribution, geometry, axes=axes)
    fig.suptitle("Phase II Hull-cell primary current-density screen", fontweight="bold")
    fig.tight_layout()
    distribution_path = FIG_DIR / "hull_cell_current_distribution.png"
    fig.savefig(distribution_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {distribution_path}")

    trace_path = DATA_DIR / "synthetic_hull_cell_galvanostatic.csv"
    gravimetry_path = DATA_DIR / "synthetic_hull_cell_gravimetry.csv"
    generate_synthetic_galvanostatic_trace(trace_path)
    raw_trace = load_galvanostatic_trace(trace_path)
    generate_synthetic_gravimetry(gravimetry_path, raw_trace)
    raw_gravimetry = load_gravimetry(gravimetry_path)
    fe_result = analyze_gravimetric_efficiency(raw_trace, raw_gravimetry)
    print("\nGravimetric Faradaic efficiency (synthetic dry-mass example):")
    print(f"  Cathodic charge: {fe_result.cathodic_charge_C:.1f} C")
    print(f"  Fe mass predicted by charge: {fe_result.theoretical_fe_mass_g:.4f} g")
    print(f"  Blank-corrected dry mass gain: {fe_result.net_deposit_mass_g:.4f} g")
    print(f"  Apparent gravimetric FE: {fe_result.apparent_faradaic_efficiency_percent:.2f}%")
    print(f"  Saved: {trace_path}\n  Saved: {gravimetry_path}")

    fe_path = FIG_DIR / "gravimetric_faradaic_efficiency.png"
    _plot_gravimetric_result(raw_trace, fe_result, fe_path)
    print(f"  Saved: {fe_path}")

    report = {
        "method_scope": {
            "hull_distribution": (
                "Primary ohmic variable-gap model normalized to applied current; excludes "
                "edge/shield, kinetic, mass-transfer, bubble, and conductivity-gradient effects."
            ),
            "gravimetric_fe": (
                "Apparent iron FE from blank-corrected dry mass gain / Fe mass predicted by "
                "cathodic charge; verify deposit composition before treating as Fe FE."
            ),
        },
        "hull_geometry": {
            **geometry.__dict__,
            "panel_area_cm2": geometry.panel_area_cm2,
            "panel_angle_deg": geometry.panel_angle_deg,
        },
        "hull_distribution_summary": distribution_summary,
        "current_density_window_10_to_100_mA_cm2": operating_window,
        "gravimetric_faradaic_efficiency": fe_result.summary(),
        "synthetic_inputs": {
            "galvanostatic_trace": trace_path.name,
            "gravimetry": gravimetry_path.name,
        },
    }
    report_path = DATA_DIR / "hull_cell_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"  Saved: {report_path}")
    print("\n✅ Phase II Hull-cell tooling driver complete!")


if __name__ == "__main__":
    main()
