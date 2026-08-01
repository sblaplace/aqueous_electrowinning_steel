"""CLI driver for pulsed electrodeposition Pareto optimization.

Runs the full parameter grid sweep, computes Pareto fronts, generates
four diagnostic figures and a JSON report with operating-point recommendations.

Usage::

    python -m models.run_pulse_optimization           # default output dir
    python -m models.run_pulse_optimization -o results # custom output dir

Outputs (to ``--output`` directory):
    1. pareto_grain_vs_efficiency.png
    2. pareto_strength_vs_energy.png
    3. pareto_carbon_vs_grain.png
    4. operating_window_heatmap.png
    5. pulse_optimization_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pulse_optimization import PulseOptimizationSweep


def _fig_pareto_grain_vs_efficiency(
    df: pd.DataFrame,
    pareto: pd.DataFrame,
    out: Path,
) -> None:
    """Figure 1: Grain size vs current efficiency Pareto front."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Background: all points, colored by waveform
    for wf, color, marker in [("pe", "#2196F3", "o"), ("pre", "#FF5722", "^")]:
        mask = df["waveform"] == wf
        ax.scatter(
            df.loc[mask, "grain_size_um"],
            df.loc[mask, "current_efficiency_pct"],
            c=color, marker=marker, alpha=0.15, s=12, label=f"All {wf.upper()}",
        )

    # Pareto front
    pf = pareto.sort_values("grain_size_um")
    ax.plot(
        pf["grain_size_um"], pf["current_efficiency_pct"],
        "k-o", markersize=5, linewidth=2, label="Pareto front", zorder=5,
    )

    ax.set_xlabel("Grain size (µm)")
    ax.set_ylabel("Current efficiency (%)")
    ax.set_title("Pareto Front: Min Grain Size vs Max Current Efficiency")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pareto_grain_vs_efficiency.png", dpi=150)
    plt.close(fig)


def _fig_pareto_strength_vs_energy(
    df: pd.DataFrame,
    pareto: pd.DataFrame,
    out: Path,
) -> None:
    """Figure 2: Yield strength vs energy cost Pareto front."""
    fig, ax = plt.subplots(figsize=(9, 6))

    valid = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["energy_cost_USD_per_kg"]
    )
    for wf, color, marker in [("pe", "#4CAF50", "o"), ("pre", "#9C27B0", "^")]:
        mask = valid["waveform"] == wf
        ax.scatter(
            valid.loc[mask, "energy_cost_USD_per_kg"],
            valid.loc[mask, "yield_strength_MPa"],
            c=color, marker=marker, alpha=0.15, s=12, label=f"All {wf.upper()}",
        )

    pf = pareto.sort_values("energy_cost_USD_per_kg")
    ax.plot(
        pf["energy_cost_USD_per_kg"], pf["yield_strength_MPa"],
        "k-o", markersize=5, linewidth=2, label="Pareto front", zorder=5,
    )

    ax.set_xlabel("Energy cost ($/kg Fe)")
    ax.set_ylabel("Yield strength (MPa)")
    ax.set_title("Pareto Front: Max Strength vs Min Energy Cost")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pareto_strength_vs_energy.png", dpi=150)
    plt.close(fig)


def _fig_pareto_carbon_vs_grain(
    df: pd.DataFrame,
    pareto: pd.DataFrame,
    out: Path,
) -> None:
    """Figure 3: Carbon incorporation vs grain size Pareto front."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for mech, color, marker in [
        ("hydroxide_suppression", "#E91E63", "o"),
        ("intermediate_adsorption", "#00BCD4", "s"),
        ("mixed_metal_intermediate", "#FFC107", "^"),
    ]:
        mask = df["mechanism"] == mech
        label = mech.replace("_", " ").title()
        ax.scatter(
            df.loc[mask, "grain_size_um"],
            df.loc[mask, "carbon_wt_pct"],
            c=color, marker=marker, alpha=0.15, s=12, label=label,
        )

    pf = pareto.sort_values("grain_size_um")
    ax.plot(
        pf["grain_size_um"], pf["carbon_wt_pct"],
        "k-o", markersize=5, linewidth=2, label="Pareto front", zorder=5,
    )

    ax.set_xlabel("Grain size (µm)")
    ax.set_ylabel("Carbon incorporation (wt%)")
    ax.set_title("Pareto Front: Max Carbon vs Min Grain Size")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pareto_carbon_vs_grain.png", dpi=150)
    plt.close(fig)


def _fig_operating_window_heatmap(
    df: pd.DataFrame,
    recs: list,
    out: Path,
) -> None:
    """Figure 4: Operating window — j_peak vs duty_cycle heatmap of yield strength."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, wf in zip(axes, ["pe", "pre"]):
        subset = df[df["waveform"] == wf]
        # Pivot: mean yield strength over (j_peak, duty_cycle), averaged
        # across frequencies and mechanisms
        pivot = subset.pivot_table(
            values="yield_strength_MPa",
            index="duty_cycle",
            columns="j_peak_mA_cm2",
            aggfunc="mean",
        )
        im = ax.imshow(
            pivot.values, aspect="auto", origin="lower", cmap="viridis",
            extent=[
                pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max(),
            ],
        )
        ax.set_xlabel("j_peak (mA/cm²)")
        ax.set_ylabel("Duty cycle")
        ax.set_title(f"Mean Yield Strength (MPa) — {wf.upper()}")
        fig.colorbar(im, ax=ax, label="MPa")

        # Mark recommended points
        for r in recs:
            if r["waveform"] == wf:
                ax.plot(
                    r["j_peak_mA_cm2"], r["duty_cycle"],
                    "r*", markersize=15, markeredgecolor="white", markeredgewidth=1.5,
                )

    fig.suptitle("Operating Window: Yield Strength Heatmap with Recommendations", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "operating_window_heatmap.png", dpi=150)
    plt.close(fig)


def build_report(
    df: pd.DataFrame,
    fronts: Dict[str, pd.DataFrame],
    recs: list,
) -> Dict[str, Any]:
    """Build the JSON report from sweep results."""
    grid_size = len(df)
    front_sizes = {name: len(fdf) for name, fdf in fronts.items()}

    # Summary statistics
    stats = {
        "grain_size_um": {
            "min": float(df["grain_size_um"].min()),
            "max": float(df["grain_size_um"].max()),
            "mean": float(df["grain_size_um"].mean()),
        },
        "yield_strength_MPa": {
            "min": float(df["yield_strength_MPa"].min()),
            "max": float(df["yield_strength_MPa"].max()),
            "mean": float(df["yield_strength_MPa"].mean()),
        },
        "current_efficiency_pct": {
            "min": float(df["current_efficiency_pct"].min()),
            "max": float(df["current_efficiency_pct"].max()),
            "mean": float(df["current_efficiency_pct"].mean()),
        },
        "energy_cost_USD_per_kg": {
            "min": float(df.replace([np.inf], np.nan)["energy_cost_USD_per_kg"].min()),
            "max": float(df.replace([np.inf], np.nan)["energy_cost_USD_per_kg"].max()),
            "mean": float(df.replace([np.inf], np.nan)["energy_cost_USD_per_kg"].mean()),
        },
        "carbon_wt_pct": {
            "min": float(df["carbon_wt_pct"].min()),
            "max": float(df["carbon_wt_pct"].max()),
            "mean": float(df["carbon_wt_pct"].mean()),
        },
    }

    # Clean recommendations for JSON
    def _clean_rec(r: dict) -> dict:
        return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in r.items()}

    return {
        "grid_size": grid_size,
        "parameter_counts": {
            "j_peak": len(df["j_peak_mA_cm2"].unique()),
            "duty_cycle": len(df["duty_cycle"].unique()),
            "frequency": len(df["frequency_Hz"].unique()),
            "waveform": len(df["waveform"].unique()),
            "mechanism": len(df["mechanism"].unique()),
        },
        "pareto_front_sizes": front_sizes,
        "summary_statistics": stats,
        "recommended_operating_points": [_clean_rec(r) for r in recs],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pulsed electrodeposition Pareto optimization sweep"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="results/pulse_optimization",
        help="Output directory for figures and JSON report",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pulsed Electrodeposition Pareto Optimization")
    print("=" * 60)

    sweep = PulseOptimizationSweep()
    print(f"Grid size: {sweep.grid_size()} combinations")

    t0 = time.time()
    print("Running parameter sweep...")
    df = sweep.run_full_sweep(progress=True)
    elapsed = time.time() - t0
    print(f"Sweep complete: {len(df)} valid points in {elapsed:.1f}s")

    print("Computing Pareto fronts...")
    fronts = sweep.compute_pareto_fronts(df)
    for name, fdf in fronts.items():
        print(f"  {name}: {len(fdf)} non-dominated points")

    print("Recommending operating points...")
    recs = sweep.recommend_operating_points(fronts)
    print(f"  {len(recs)} recommendations")

    print("Generating figures...")
    _fig_pareto_grain_vs_efficiency(df, fronts["grain_vs_efficiency"], out)
    _fig_pareto_strength_vs_energy(df, fronts["strength_vs_energy"], out)
    _fig_pareto_carbon_vs_grain(df, fronts["carbon_vs_grain"], out)
    _fig_operating_window_heatmap(df, recs, out)
    print(f"  4 figures saved to {out}/")

    report = build_report(df, fronts, recs)
    report_path = out / "pulse_optimization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report saved to {report_path}")

    # Also save full sweep CSV
    csv_path = out / "pulse_optimization_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Full sweep CSV saved to {csv_path}")

    print()
    print("Recommended operating points:")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. [{r['type']}] j_peak={r['j_peak_mA_cm2']:.0f} mA/cm², "
              f"duty={r['duty_cycle']:.1f}, f={r['frequency_Hz']:.0f} Hz, "
              f"{r['waveform'].upper()}, {r['mechanism']}")
        print(f"     Grain={r['grain_size_um']:.3f} µm, σ_y={r['yield_strength_MPa']:.0f} MPa, "
              f"FE={r['current_efficiency_pct']:.1f}%, C={r['carbon_wt_pct']:.2f} wt%, "
              f"Energy=${r['energy_cost_USD_per_kg']:.3f}/kg")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
