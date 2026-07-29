#!/usr/bin/env python3
"""
Phase III executable analysis — co-deposition screening.

Usage:
    python experiments/notebooks/phase3_co_deposition.py
    # or import and call from a Jupyter notebook:
    from experiments.notebooks.phase3_co_deposition import run_phase3_analysis
    run_phase3_analysis()

This script connects synthetic (or experimental) co-deposition data to the
``models.co_deposition`` module and writes a summary CSV and plot.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure repository root is on PYTHONPATH when run standalone
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.co_deposition import (
    PhaseIIICoDeposition,
    build_phase3_model,
)


def load_experiment_csv(path: str | Path) -> pd.DataFrame:
    """Load a Phase III co-deposition CSV (synthetic or experimental)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Experiment file not found: {p}")
    df = pd.read_csv(p)
    required = {
        "current_density_mA_cm2",
        "fe_wt_percent_predicted",
        "ni_wt_percent_predicted",
        "current_efficiency_percent_predicted",
        "carbon_wt_percent_predicted",
        "mechanism_fe_ni",
        "notes",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Phase III CSV missing columns: {sorted(missing)}")
    return df


def run_phase3_analysis(
    csv_path: str | Path = "experiments/data/synthetic_co_deposition_hydroxide_suppression.csv",
    mechanism: str = "hydroxide_suppression",
    output_dir: str | Path = "experiments/data/",
) -> dict:
    """
    Execute the Phase III co-deposition analysis pipeline.

    Returns
    -------
    dict
        Summary dictionary with predictions and quality flags.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (synthetic or experimental)
    try:
        df = load_experiment_csv(csv_path)
        print(f"Loaded Phase III data: {csv_path} ({len(df)} rows)")
    except FileNotFoundError:
        # Fallback: generate synthetic dataset for demonstration
        print(f"File {csv_path} not found; generating synthetic dataset for mechanism '{mechanism}'.")
        from models.run_co_deposition import generate_synthetic_co_deposition_data
        df, model = generate_synthetic_co_deposition_data(csv_path, mechanism_fe_ni=mechanism)

    # Initialize model matching the dataset mechanism
    model = build_phase3_model(
        bath_fe_M=0.6,
        bath_ni_M=0.4,
        pH=3.5,
        temperature_C=60.0,
        carbon_particle_loading_g_L=2.5,
        mechanism_fe_ni=mechanism,
    )

    # Compute predictions for each current density in the dataset
    predictions = []
    for _, row in df.iterrows():
        j = float(row["current_density_mA_cm2"])
        res = model.run_at_current(j)
        predictions.append({
            "current_density_mA_cm2": j,
            "predicted_fe_wt_percent": res["alloy_kinetics"]["fe_wt_percent"],
            "predicted_ni_wt_percent": res["alloy_kinetics"]["ni_wt_percent"],
            "predicted_carbon_wt_percent": res["carbon_incorporation"]["predicted_carbon_wt_percent"],
            "predicted_ce_percent": res["integrated_metrics"]["adjusted_overall_current_efficiency_percent"],
            "is_anomalous": res["alloy_kinetics"]["is_anomalous"],
            "anomalous_description": res["integrated_metrics"]["anomalous_description"],
        })

    pred_df = pd.DataFrame(predictions)

    # Write predictions to CSV
    output_csv = output_dir / f"phase3_analysis_output_{mechanism}.csv"
    pred_df.to_csv(output_csv, index=False)
    print(f"Analysis output saved: {output_csv}")

    # Generate a simple summary plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))
        axes = axes.ravel()

        j_vals = pred_df["current_density_mA_cm2"].to_numpy()

        # Alloy composition
        ax = axes[0]
        ax.plot(j_vals, pred_df["predicted_fe_wt_percent"], color="#1874b4", marker="o", label="Fe (wt%)")
        ax.plot(j_vals, pred_df["predicted_ni_wt_percent"], color="#d95f02", marker="s", label="Ni (wt%)")
        ax.set(title="Predicted alloy composition", xlabel="j (mA/cm²)", ylabel="wt%")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

        # Carbon content
        ax = axes[1]
        ax.plot(j_vals, pred_df["predicted_carbon_wt_percent"], color="#e41a1c", marker="o")
        ax.fill_between(j_vals, pred_df["predicted_carbon_wt_percent"], alpha=0.15, color="#e41a1c")
        ax.set(title="Predicted carbon incorporation", xlabel="j (mA/cm²)", ylabel="C (wt%)")
        ax.grid(alpha=0.25)

        # Current efficiency
        ax = axes[2]
        ax.plot(j_vals, pred_df["predicted_ce_percent"], color="#1a9641", marker="s")
        ax.set(title="Predicted metal current efficiency", xlabel="j (mA/cm²)", ylabel="CE (%)")
        ax.grid(alpha=0.25)

        # Anomaly flag
        ax = axes[3]
        colors = ["#e31a1c" if a else "#377eb8" for a in pred_df["is_anomalous"]]
        ax.scatter(j_vals, pred_df["predicted_fe_wt_percent"], c=colors, s=50, zorder=3)
        ax.set(title="Anomalous regime (red = anomalous)", xlabel="j (mA/cm²)", ylabel="Fe wt%")
        ax.grid(alpha=0.25)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e31a1c", label="Anomalous"),
            Patch(facecolor="#377eb8", label="Normal"),
        ]
        ax.legend(handles=legend_elements, fontsize=9)

        fig.tight_layout()
        plot_path = output_dir / f"phase3_analysis_plot_{mechanism}.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"Plot saved: {plot_path}")
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")

    # Return summary statistics
    summary = {
        "mechanism": mechanism,
        "n_points": len(pred_df),
        "current_range_mA_cm2": [float(j_vals.min()), float(j_vals.max())],
        "mean_predicted_fe_wt_percent": float(pred_df["predicted_fe_wt_percent"].mean()),
        "mean_predicted_ni_wt_percent": float(pred_df["predicted_ni_wt_percent"].mean()),
        "mean_predicted_carbon_wt_percent": float(pred_df["predicted_carbon_wt_percent"].mean()),
        "mean_predicted_ce_percent": float(pred_df["predicted_ce_percent"].mean()),
        "fraction_anomalous": float(pred_df["is_anomalous"].mean()),
        "output_csv": str(output_csv),
    }
    return summary


if __name__ == "__main__":
    # Default execution: demonstrate with synthetic dataset
    result = run_phase3_analysis(
        csv_path="experiments/data/synthetic_co_deposition_hydroxide_suppression.csv",
        mechanism="hydroxide_suppression",
    )
    print("\nPhase III co-deposition summary:")
    for k, v in result.items():
        print(f"  {k}: {v}")
