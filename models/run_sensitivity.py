"""
Driver — sensitivity analysis producing 4 figures.

Usage
-----
python -m models.run_sensitivity                  # defaults: N=1000, full
python -m models.run_sensitivity --n-samples 5000
python -m models.run_sensitivity --quick           # fast preview

Outputs
-------
* sensitivity_tornado_ys.png          – tornado chart for yield strength
* sensitivity_sobol_heatmap.png       – Sobol first-order heatmap (outputs × params)
* sensitivity_variance_decomposition.png – stacked bar: first-order vs interaction
* sensitivity_morris_screening.png    – Morris mu_star vs sigma
* sensitivity_report.json             – machine-readable report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.uncertainty.monte_carlo import MonteCarloEngine, DEFAULT_DESIGN_POINT
from models.uncertainty.parameter_registry import REGISTRY
from models.uncertainty.sensitivity import (
    sobol_analysis,
    tornado_chart,
    morris_screening,
    SobolResult,
    TornadoResult,
    MorrisResult,
)


# Key outputs for analysis (matching MC engine's outputs)
KEY_OUTPUTS = [
    "sigma_y_MPa", "uts_MPa", "vickers_hv", "elongation_pct",
    "grain_size_um", "porosity", "current_efficiency_percent",
    "specific_energy_kWh_per_kg", "case_depth_035_um", "surface_hv",
    "ni_wt_percent", "carbon_wt_percent",
]

# Short labels for plots
OUTPUT_LABELS = {
    "sigma_y_MPa": "YS (MPa)",
    "uts_MPa": "UTS (MPa)",
    "vickers_hv": "HV",
    "elongation_pct": "Elong (%)",
    "grain_size_um": "Grain (µm)",
    "porosity": "Porosity",
    "current_efficiency_percent": "CE (%)",
    "specific_energy_kWh_per_kg": "Energy (kWh/kg)",
    "case_depth_035_um": "Case depth (µm)",
    "surface_hv": "Surface HV",
    "ni_wt_percent": "Ni (wt%)",
    "carbon_wt_percent": "C (wt%)",
}


# ── Figure 1: Tornado chart for yield strength ───────────────────────────

def plot_tornado(tornado: TornadoResult, out_path: Path) -> None:
    """Horizontal bar chart of top parameters for one output."""
    ranked = tornado.ranked()
    if not ranked:
        return

    names = [r[0] for r in ranked]
    values = [r[1] for r in ranked]
    n = len(names)

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))
    y_pos = range(n)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, n))
    ax.barh(y_pos, values, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised sensitivity")
    ax.set_title(
        f"Tornado: {OUTPUT_LABELS.get(tornado.output_key, tornado.output_key)}",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    # Annotate with numeric values
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Figure 2: Sobol first-order heatmap ──────────────────────────────────

def plot_sobol_heatmap(sobol: SobolResult, out_path: Path, top_params: int = 20) -> None:
    """Heatmap of first-order Sobol indices (outputs × top parameters)."""
    # Select top parameters by average first-order index across outputs
    avg_si = np.mean(sobol.first_order, axis=0)
    top_idx = np.argsort(avg_si)[::-1][:top_params]

    param_labels = [sobol.parameter_names[i] for i in top_idx]
    out_labels = [OUTPUT_LABELS.get(k, k) for k in sobol.output_keys]

    matrix = sobol.first_order[:, top_idx]

    fig, ax = plt.subplots(figsize=(max(10, top_params * 0.6), max(6, len(out_labels) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.5, np.max(matrix)))
    ax.set_xticks(range(len(param_labels)))
    ax.set_xticklabels(param_labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(out_labels)))
    ax.set_yticklabels(out_labels, fontsize=7)
    fig.colorbar(im, ax=ax, label="First-order Sobol index S_i")
    ax.set_title("Sobol First-Order Sensitivity Indices", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Figure 3: Variance decomposition (stacked bar) ───────────────────────

def plot_variance_decomposition(sobol: SobolResult, out_path: Path, top_params: int = 8) -> None:
    """Stacked bar chart: first-order vs interaction contribution."""
    # Select outputs with non-zero variance
    valid_outputs = [i for i, v in enumerate(sobol.variance) if v > 1e-20]
    if not valid_outputs:
        return

    out_labels = [OUTPUT_LABELS.get(sobol.output_keys[i], sobol.output_keys[i])
                  for i in valid_outputs]

    # Top parameters by average first-order
    avg_si = np.mean(sobol.first_order[valid_outputs], axis=0)
    top_idx = np.argsort(avg_si)[::-1][:top_params]
    param_labels = [sobol.parameter_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(out_labels) * 1.5), 6))

    n_out = len(valid_outputs)
    n_par = len(top_idx)
    x = np.arange(n_out)
    width = 0.7

    # Stack: each parameter's contribution
    bottom = np.zeros(n_out)
    colors = plt.cm.tab20(np.linspace(0, 1, n_par))

    for k, (idx, label) in enumerate(zip(top_idx, param_labels)):
        vals = sobol.first_order[valid_outputs, idx]
        ax.bar(x, vals, width, bottom=bottom, label=label, color=colors[k])
        bottom += vals

    # Residual = 1 - sum(S_i) = interaction / higher-order
    residual = 1.0 - bottom
    residual = np.maximum(residual, 0)
    ax.bar(x, residual, width, bottom=bottom, label="Interactions",
           color="lightgray", edgecolor="gray", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(out_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Fraction of variance")
    ax.set_ylim(0, 1.15)
    ax.set_title("Variance Decomposition (First-Order + Interactions)", fontweight="bold")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Figure 4: Morris screening (mu_star vs sigma) ────────────────────────

def plot_morris(morris: MorrisResult, out_path: Path, top_n: int = 15) -> None:
    """Scatter plot of mu_star vs sigma for Morris screening."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot all points
    ax.scatter(morris.mu_star, morris.sigma, c="steelblue", alpha=0.6, s=30)

    # Annotate top-n
    order = np.argsort(morris.mu_star)[::-1][:top_n]
    for i in order:
        ax.annotate(
            morris.parameter_names[i],
            (morris.mu_star[i], morris.sigma[i]),
            fontsize=6, alpha=0.8,
            xytext=(3, 3), textcoords="offset points",
        )

    # Draw mu_star = sigma line (linear vs non-linear)
    lim = max(np.max(morris.mu_star), np.max(morris.sigma)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="μ* = σ (linear)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("μ* (mean |elementary effect|)")
    ax.set_ylabel("σ (std of elementary effects)")
    ax.set_title(
        f"Morris Screening ({morris.n_trajectories} trajectories, "
        f"{morris.n_params} params)",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main(
    n_samples_mc: int = 1000,
    n_samples_sobol: int = 3000,
    n_morris_trajectories: int = 30,
    seed: int = 42,
    quick: bool = False,
):
    print("=" * 72)
    print("SENSITIVITY ANALYSIS")
    print("=" * 72)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if quick:
        n_samples_mc = 200
        n_samples_sobol = 500
        n_morris_trajectories = 10

    report = {}
    t_total = time.perf_counter()

    # ── 1. Run MC for tornado ─────────────────────────────────────────
    print(f"\n[1/4] Monte Carlo tornado (N={n_samples_mc}) ...")
    engine = MonteCarloEngine(n_samples=n_samples_mc, seed=seed, n_jobs=1)
    mc_result = engine.run()
    print(f"  Completed in {mc_result.elapsed_seconds:.1f}s")

    # Tornado for yield strength
    ys_tornado = tornado_chart(mc_result, "sigma_y_MPa", top_n=10)
    plot_tornado(ys_tornado, FIG_DIR / "sensitivity_tornado_ys.png")
    report["tornado_ys"] = ys_tornado.to_dict()
    print(f"  ✅ Tornado chart saved")

    # ── 2. Sobol analysis ────────────────────────────────────────────
    print(f"\n[2/4] Sobol analysis (N={n_samples_sobol}) ...")
    # Use a representative subset from each model module for Sobol
    # (covers mechanical, carburization, kinetics, co-deposition)
    sobol_params = [
        # mechanical_properties (directly affect YS, UTS, HV, elongation)
        "sigma0_fe_MPa", "k_hp_MPa_sqrt_m", "k_ss_ni_MPa_per_wt",
        "k_carbon_MPa_per_wt", "tabor_factor", "uts_over_ys_base",
        "elongation_base_pct", "porosity_penalty_exp",
        # grain size
        "grain_d0_dc_ref_um", "grain_j_exponent", "grain_pe_factor_base",
        # carburization (affect case depth, surface HV)
        "D0_ferrite_m2_s", "Q_ferrite_kJ_mol", "HV_per_C_wt_Maynier",
        # tempering
        "C_HJ", "k_softening", "KM_alpha_K_inv",
        # kinetics (affect CE, energy)
        "fe_i0", "her_i0", "fe_tafel_V",
        # co-deposition (affect Ni, C content)
        "guglielmi_k_ref", "ni_i0",
    ]

    available_outputs = [k for k in KEY_OUTPUTS if k in mc_result.output_distributions]

    sobol_result = sobol_analysis(
        engine,
        n_samples=n_samples_sobol,
        param_subset=sobol_params,
        output_subset=available_outputs[:10],  # limit to 10 outputs
    )
    print(f"  {sobol_result.n_samples} evaluations in {sobol_result.elapsed_seconds:.1f}s")

    plot_sobol_heatmap(sobol_result, FIG_DIR / "sensitivity_sobol_heatmap.png")
    report["sobol"] = sobol_result.to_dict()
    print(f"  ✅ Sobol heatmap saved")

    # ── 3. Variance decomposition ─────────────────────────────────────
    print(f"\n[3/4] Variance decomposition ...")
    plot_variance_decomposition(sobol_result, FIG_DIR / "sensitivity_variance_decomposition.png")
    print(f"  ✅ Variance decomposition saved")

    # ── 4. Morris screening ───────────────────────────────────────────
    print(f"\n[4/4] Morris screening (R={n_morris_trajectories}) ...")
    morris_result = morris_screening(
        n_trajectories=n_morris_trajectories,
        n_levels=4,
        seed=seed,
        output_keys=["sigma_y_MPa", "vickers_hv", "elongation_pct"],
    )
    print(f"  {morris_result.n_params} params, {morris_result.n_trajectories} trajectories "
          f"in {morris_result.elapsed_seconds:.1f}s")

    plot_morris(morris_result, FIG_DIR / "sensitivity_morris_screening.png")
    report["morris"] = morris_result.to_dict()
    print(f"  ✅ Morris screening saved")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_total
    report["elapsed_total_seconds"] = round(elapsed_total, 2)

    report_path = DATA_DIR / "sensitivity_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  ✅ Report saved to {report_path}")
    print(f"\nTotal time: {elapsed_total:.1f}s")
    print("=" * 72)

    # Print top-5 tornado
    print("\nTop-5 parameters for yield strength:")
    for name, val in ys_tornado.ranked()[:5]:
        print(f"  {name}: {val:.4f}")

    # Print top-5 Morris
    print("\nTop-5 Morris screening (all outputs):")
    for name, val in morris_result.ranked(5):
        print(f"  {name}: μ*={val:.4f}")

    return report


def cli():
    parser = argparse.ArgumentParser(description="Sensitivity analysis")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="MC sample count for tornado")
    parser.add_argument("--n-sobol", type=int, default=3000,
                        help="Sample budget for Sobol analysis")
    parser.add_argument("--n-morris", type=int, default=30,
                        help="Morris trajectories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Fast preview mode")
    args = parser.parse_args()
    main(
        n_samples_mc=args.n_samples,
        n_samples_sobol=args.n_sobol,
        n_morris_trajectories=args.n_morris,
        seed=args.seed,
        quick=args.quick,
    )


if __name__ == "__main__":
    cli()
