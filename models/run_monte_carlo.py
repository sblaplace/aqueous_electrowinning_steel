"""
Driver — Monte Carlo uncertainty propagation through the full model chain.

Usage
-----
python -m models.run_monte_carlo                  # defaults: N=1000, ASTM A36
python -m models.run_monte_carlo --n-samples 10000
python -m models.run_monte_carlo --spec-set CARBURIZED
python -m models.run_monte_carlo --n-jobs 1       # serial for debugging

Outputs
-------
* mc_output_distributions.png   – violin/box plots of key outputs
* mc_pass_rates.png             – bar chart of per-spec pass rates
* mc_sensitivity_tornado.png    – tornado diagram of top-5 params per output
* mc_correlation_heatmap.png    – output-output correlation matrix
* monte_carlo_report.json       – full machine-readable report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR = ROOT / "docs" / "figures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.uncertainty.monte_carlo import MonteCarloEngine, MonteCarloResult
from models.uncertainty.specification import (
    SPECS_A36, SPECS_1010, SPECS_1020, SPECS_CARBURIZED, SPECS_ELECTROWINNING,
)


SPEC_SETS = {
    "ASTM_A36": SPECS_A36,
    "AISI_1010": SPECS_1010,
    "AISI_1020": SPECS_1020,
    "CARBURIZED": SPECS_CARBURIZED,
    "ELECTROWINNING": SPECS_ELECTROWINNING,
}

# Key outputs to plot (skip design-point-only constants)
KEY_OUTPUTS = [
    "sigma_y_MPa", "uts_MPa", "vickers_hv", "elongation_pct",
    "grain_size_um", "porosity", "current_efficiency_percent",
    "specific_energy_kWh_per_kg", "case_depth_035_um", "surface_hv",
    "ni_wt_percent", "carbon_wt_percent",
]


def plot_output_distributions(result: MonteCarloResult, out_path: Path) -> None:
    """Violin plots of key output distributions."""
    data = []
    labels = []
    for key in KEY_OUTPUTS:
        arr = result.output_distributions.get(key)
        if arr is not None:
            valid = arr[~np.isnan(arr)]
            if len(valid) > 10:
                data.append(valid)
                labels.append(key.replace("_", "\n"))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 1.2), 6))
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_title(f"Output Distributions (N={result.n_samples})", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pass_rates(result: MonteCarloResult, out_path: Path) -> None:
    """Bar chart of per-spec pass rates."""
    if not result.pass_rates:
        return

    names = list(result.pass_rates.keys())
    rates = [result.pass_rates[n] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    colors = ["#4daf4a" if r >= 90 else "#ff7f00" if r >= 50 else "#e41a1c" for r in rates]
    bars = ax.bar(range(len(names)), rates, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 105)
    ax.axhline(90, color="green", ls="--", alpha=0.4, label="90% target")
    ax.set_title(
        f"Specification Pass Rates — Overall: {result.overall_confidence*100:.1f}%",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_sensitivity_tornado(result: MonteCarloResult, out_path: Path) -> None:
    """Tornado diagram: top params for each key output."""
    outputs_to_plot = [k for k in KEY_OUTPUTS if k in result.sensitivity and result.sensitivity[k]]
    if not outputs_to_plot:
        return

    n_outputs = len(outputs_to_plot)
    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, max(8, n_outputs * 1.5)))
    if n_outputs == 1:
        axes = [axes]

    for ax, out_key in zip(axes, outputs_to_plot):
        sens = result.sensitivity[out_key]
        params = list(sens.keys())
        values = [sens[p] for p in params]
        y_pos = range(len(params))
        ax.barh(y_pos, values, color="#1874b4", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params, fontsize=8)
        ax.set_xlabel("|Correlation|")
        ax.set_title(f"Sensitivity: {out_key}", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Parameter Sensitivity (Top-3 per output)", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(result: MonteCarloResult, out_path: Path) -> None:
    """Output-output correlation heatmap."""
    keys = [k for k in KEY_OUTPUTS if k in result.parameter_correlations]
    if len(keys) < 2:
        return

    n = len(keys)
    matrix = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            matrix[i, j] = result.parameter_correlations.get(ki, {}).get(kj, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [k.replace("_", "\n") for k in keys]
    ax.set_xticklabels(short_labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(short_labels, fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Output Correlation Matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(
    n_samples: int = 1000,
    spec_set: str = "ASTM_A36",
    n_jobs: int = -1,
    seed: int = 42,
):
    print("=" * 72)
    print(f"MONTE CARLO — N={n_samples}, specs={spec_set}")
    print("=" * 72)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    specs = SPEC_SETS.get(spec_set, SPECS_A36)

    engine = MonteCarloEngine(n_samples=n_samples, seed=seed, n_jobs=n_jobs)
    result = engine.run(specs=specs, spec_set_name=spec_set)

    print(f"\nCompleted in {result.elapsed_seconds:.1f}s")
    print(f"Outputs: {len(result.output_distributions)}")
    print(f"Overall confidence: {result.overall_confidence*100:.1f}%")
    print("Pass rates:")
    for name, rate in result.pass_rates.items():
        print(f"  {name}: {rate*100:.1f}%")

    # Plot
    print("\nGenerating figures...")
    plot_output_distributions(result, FIG_DIR / "mc_output_distributions.png")
    plot_pass_rates(result, FIG_DIR / "mc_pass_rates.png")
    plot_sensitivity_tornado(result, FIG_DIR / "mc_sensitivity_tornado.png")
    plot_correlation_heatmap(result, FIG_DIR / "mc_correlation_heatmap.png")
    print(f"  ✅ Figures saved to {FIG_DIR}")

    # JSON report
    report = result.summary_dict()
    report_path = DATA_DIR / "monte_carlo_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  ✅ Report saved to {report_path}")

    return result


def cli():
    parser = argparse.ArgumentParser(description="Monte Carlo uncertainty propagation")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of MC samples")
    parser.add_argument("--spec-set", type=str, default="ASTM_A36",
                        choices=list(SPEC_SETS.keys()), help="Specification set")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(n_samples=args.n_samples, spec_set=args.spec_set, n_jobs=args.n_jobs, seed=args.seed)


if __name__ == "__main__":
    cli()
