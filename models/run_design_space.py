"""
Driver — Design space exploration for the aqueous electrowinning model.

Usage
-----
python -m models.run_design_space                          # defaults: 2D, N=100
python -m models.run_design_space --n-grid 6 --mc-samples 200
python -m models.run_design_space --skip-optimization      # fast, grid only

Outputs
-------
* design_space_confidence_map.png  — 2-D confidence surface (contour)
* design_space_pareto.png          — Pareto front (confidence vs cost vs energy)
* design_space_robust_optimum.png  — Bayesian optimization convergence
* design_space_margins.png         — Design margins at robust optimum
* design_space_report.json         — Full machine-readable report
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

from models.uncertainty.design_space import (
    explore_design_space,
    robust_optimum,
    pareto_front_robust,
    DesignSpaceResult,
    RobustOptimum,
    ParetoFront,
)
from models.uncertainty.specification import SPECS_A36, SPECS_ELECTROWINNING


# --------------------------------------------------------------------------- 
# Default sweep ranges (2 most impactful operating parameters)
# --------------------------------------------------------------------------- 

DEFAULT_RANGES_2D = {
    "j_avg": (50.0, 300.0),       # mA/cm²
    "T_bath": (30.0, 80.0),       # °C
}

# Full 4-D sweep (heavier)
FULL_RANGES_4D = {
    "j_avg": (50.0, 300.0),
    "T_bath": (30.0, 80.0),
    "pH": (2.0, 5.0),
    "carburizing_T": (800.0, 950.0),
}


# --------------------------------------------------------------------------- 
# Plotting functions
# --------------------------------------------------------------------------- 

def plot_confidence_map(result: DesignSpaceResult, out_path: Path) -> None:
    """Contour plot of the confidence surface over 2 operating dimensions."""
    if len(result.param_names) != 2:
        # For higher-D, plot slices
        _plot_confidence_slices(result, out_path)
        return

    x_vals = np.unique(result.grid_points[:, 0])
    y_vals = np.unique(result.grid_points[:, 1])
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = result.confidence_values.reshape(len(y_vals), len(x_vals))

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.linspace(0, 1, 21)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="RdYlGn")
    cs = ax.contour(X, Y, Z, levels=[0.90, 0.95], colors="black", linewidths=1.5)
    ax.clabel(cs, fmt="%.2f", fontsize=9)
    fig.colorbar(cf, ax=ax, label="P(all specs met)")

    # Mark optimum
    best = result.best_point
    ax.plot(best[result.param_names[0]], best[result.param_names[1]],
            "k*", markersize=15, label=f"Best: {result.max_confidence:.3f}")

    ax.set_xlabel(result.param_names[0])
    ax.set_ylabel(result.param_names[1])
    ax.set_title(
        f"Design Space Confidence Map\n"
        f"(N_grid={result.n_grid}, MC={result.mc_samples_per_point}/pt)",
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_confidence_slices(result: DesignSpaceResult, out_path: Path) -> None:
    """For >2-D, plot pairwise slices as a grid of subplots."""
    D = len(result.param_names)
    fig, axes = plt.subplots(D, D, figsize=(4 * D, 4 * D))
    if D == 1:
        axes = np.array([[axes]])

    for i in range(D):
        for j in range(D):
            ax = axes[i][j] if D > 1 else axes
            if i == j:
                # 1-D slice: scatter of param[i] vs confidence
                ax.scatter(
                    result.grid_points[:, i],
                    result.confidence_values,
                    c=result.confidence_values,
                    cmap="RdYlGn", vmin=0, vmax=1, s=20,
                )
                ax.set_xlabel(result.param_names[i])
                ax.set_ylabel("Confidence")
            elif i > j:
                # 2-D slice (i vs j)
                ax.scatter(
                    result.grid_points[:, j],
                    result.grid_points[:, i],
                    c=result.confidence_values,
                    cmap="RdYlGn", vmin=0, vmax=1, s=30,
                )
                ax.set_xlabel(result.param_names[j])
                ax.set_ylabel(result.param_names[i])
            else:
                ax.set_visible(False)

    fig.suptitle("Design Space Confidence (Pairwise Slices)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pareto(front: ParetoFront, out_path: Path) -> None:
    """Pareto front: confidence vs cost vs energy."""
    if not front.points:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    confs = [p.confidence for p in front.points]
    costs = [p.cost for p in front.points]
    energies = [p.energy for p in front.points]

    # Confidence vs Cost
    ax = axes[0]
    ax.scatter(costs, confs, c="green", s=80, zorder=5, label="Pareto front")
    ax.axhline(0.95, color="red", ls="--", alpha=0.5, label="95% target")
    ax.set_xlabel("Relative Cost")
    ax.set_ylabel("P(all specs met)")
    ax.set_title("Confidence vs Cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Confidence vs Energy
    ax = axes[1]
    ax.scatter(energies, confs, c="blue", s=80, zorder=5, label="Pareto front")
    ax.axhline(0.95, color="red", ls="--", alpha=0.5, label="95% target")
    ax.set_xlabel("Specific Energy (kWh/kg)")
    ax.set_ylabel("P(all specs met)")
    ax.set_title("Confidence vs Energy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Cost vs Energy (coloured by confidence)
    ax = axes[2]
    sc = ax.scatter(costs, energies, c=confs, cmap="RdYlGn", vmin=0, vmax=1,
                    s=80, zorder=5)
    fig.colorbar(sc, ax=ax, label="Confidence")
    ax.set_xlabel("Relative Cost")
    ax.set_ylabel("Specific Energy (kWh/kg)")
    ax.set_title("Cost vs Energy (Pareto)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Pareto Front — {front.n_evaluated} points, "
        f"{len(front.points)} on front ({front.dominated_count} dominated)",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_convergence(optimum: RobustOptimum, out_path: Path) -> None:
    """Bayesian optimization convergence: confidence vs iteration."""
    if len(optimum.all_confidences) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    iterations = np.arange(1, len(optimum.all_confidences) + 1)
    ax.scatter(iterations, optimum.all_confidences, alpha=0.5, s=15, c="steelblue")

    # Running best
    running_best = np.maximum.accumulate(optimum.all_confidences)
    ax.plot(iterations, running_best, "r-", lw=2, label="Running best")

    ax.axhline(optimum.target, color="green", ls="--", alpha=0.5,
               label=f"Target {optimum.target:.0%}")

    status = "ACHIEVED ✓" if optimum.achieved_target else "NOT ACHIEVED"
    ax.set_title(
        f"Robust Optimum Search — {status}\n"
        f"Best P = {optimum.optimum_confidence:.4f} "
        f"({optimum.n_calls} evaluations, {optimum.elapsed_seconds:.0f}s)",
        fontweight="bold",
    )
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("P(all specs met)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_margins(optimum: RobustOptimum, out_path: Path) -> None:
    """Design margins bar chart at the robust optimum."""
    if not optimum.design_margins:
        return

    params = list(optimum.design_margins.keys())
    lo_margins = [optimum.design_margins[p]["margin_lo_pct"] for p in params]
    hi_margins = [optimum.design_margins[p]["margin_hi_pct"] for p in params]

    fig, ax = plt.subplots(figsize=(max(8, len(params) * 1.5), 5))
    x = np.arange(len(params))
    width = 0.35

    ax.bar(x - width / 2, lo_margins, width, label="Margin (lo)", color="#2196F3", alpha=0.8)
    ax.bar(x + width / 2, hi_margins, width, label="Margin (hi)", color="#4CAF50", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Design Margin (% of range)")
    ax.set_title(
        f"Design Margins at Robust Optimum\n"
        f"(P >= 90% confidence window around optimum point)",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add optimum values as text
    for i, p in enumerate(params):
        opt_val = optimum.design_margins[p]["optimum"]
        ax.text(i, max(lo_margins[i], hi_margins[i]) + 1,
                f"opt={opt_val:.2f}", ha="center", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- 
# Main
# --------------------------------------------------------------------------- 

def main(
    n_grid: int = 8,
    mc_samples: int = 200,
    spec_set: str = "ASTM_A36",
    n_calls: int = 100,
    skip_optimization: bool = False,
    n_jobs: int = -1,
    seed: int = 42,
):
    print("=" * 72)
    print(f"DESIGN SPACE EXPLORATION — grid={n_grid}, MC={mc_samples}, specs={spec_set}")
    print("=" * 72)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    specs = SPECS_ELECTROWINNING if spec_set == "ELECTROWINNING" else SPECS_A36
    ranges = DEFAULT_RANGES_2D

    # ── Step 1: Grid exploration ──────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Step 1: Grid-based confidence surface")
    print(f"  Sweeping: {list(ranges.keys())}")
    print(f"  Grid: {n_grid}x{n_grid} = {n_grid**2} points, MC={mc_samples}/pt")

    ds_result = explore_design_space(
        ranges=ranges,
        specs=specs,
        n_grid=n_grid,
        mc_samples=mc_samples,
        seed=seed,
        n_jobs=n_jobs,
        spec_set_name=spec_set,
    )

    print(f"\n  Completed in {ds_result.elapsed_seconds:.1f}s")
    print(f"  Max confidence: {ds_result.max_confidence:.4f}")
    print(f"  Best point: {ds_result.best_point}")

    # Plot confidence map
    plot_confidence_map(ds_result, FIG_DIR / "design_space_confidence_map.png")
    print(f"  ✅ design_space_confidence_map.png")

    report: dict = {"grid_exploration": ds_result.summary_dict()}

    # ── Step 2: Robust optimum ────────────────────────────────────────
    if not skip_optimization:
        print(f"\n{'─' * 40}")
        print(f"Step 2: Bayesian optimization (target P >= 95%)")
        print(f"  Calls: {n_calls}, MC={mc_samples}/call")

        opt_result = robust_optimum(
            ranges=ranges,
            specs=specs,
            n_calls=n_calls,
            target=0.95,
            mc_samples=mc_samples,
            seed=seed,
            n_jobs=n_jobs,
            spec_set_name=spec_set,
        )

        status = "ACHIEVED ✓" if opt_result.achieved_target else "NOT ACHIEVED"
        print(f"\n  {status}")
        print(f"  Optimum confidence: {opt_result.optimum_confidence:.4f}")
        print(f"  Optimum point: {opt_result.optimum_point}")
        print(f"  Time: {opt_result.elapsed_seconds:.1f}s")

        if opt_result.design_margins:
            print(f"\n  Design margins (90% confidence window):")
            for param, margin in opt_result.design_margins.items():
                print(f"    {param}: {margin['lo_90']:.2f} — {margin['hi_90']:.2f} "
                      f"({margin['margin_lo_pct']:.0f}% / {margin['margin_hi_pct']:.0f}%)")

        plot_convergence(opt_result, FIG_DIR / "design_space_robust_optimum.png")
        plot_margins(opt_result, FIG_DIR / "design_space_margins.png")
        print(f"  ✅ design_space_robust_optimum.png")
        print(f"  ✅ design_space_margins.png")

        report["robust_optimum"] = opt_result.summary_dict()
    else:
        print("\n  (skipping optimization)")

    # ── Step 3: Pareto front ──────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Step 3: Pareto front (confidence vs cost vs energy)")
    print(f"  Grid: {max(4, n_grid // 2)}x{max(4, n_grid // 2)}, MC={mc_samples}/pt")

    pareto_result = pareto_front_robust(
        objectives=["confidence", "cost", "energy"],
        ranges=ranges,
        specs=specs,
        n_grid=max(4, n_grid // 2),
        mc_samples=mc_samples,
        seed=seed,
        n_jobs=n_jobs,
        spec_set_name=spec_set,
    )

    print(f"\n  {len(pareto_result.points)} points on Pareto front "
          f"({pareto_result.dominated_count} dominated)")
    print(f"  Best confidence: {pareto_result.best_confidence_point.confidence:.4f}")
    print(f"  Best cost: {pareto_result.best_cost_point.cost:.4f}")
    print(f"  Best energy: {pareto_result.best_energy_point.energy:.2f} kWh/kg")
    print(f"  Time: {pareto_result.elapsed_seconds:.1f}s")

    plot_pareto(pareto_result, FIG_DIR / "design_space_pareto.png")
    print(f"  ✅ design_space_pareto.png")

    report["pareto_front"] = pareto_result.summary_dict()

    # ── Save report ───────────────────────────────────────────────────
    report_path = DATA_DIR / "design_space_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  ✅ Report saved to {report_path}")

    print("\n" + "=" * 72)
    print("DESIGN SPACE EXPLORATION COMPLETE")
    print("=" * 72)

    return ds_result


def cli():
    parser = argparse.ArgumentParser(description="Design space exploration")
    parser.add_argument("--n-grid", type=int, default=8, help="Grid points per dimension")
    parser.add_argument("--mc-samples", type=int, default=200, help="MC samples per grid point")
    parser.add_argument("--spec-set", type=str, default="ASTM_A36",
                        choices=["ASTM_A36", "ELECTROWINNING"], help="Specification set")
    parser.add_argument("--n-calls", type=int, default=100,
                        help="Bayesian optimization evaluations")
    parser.add_argument("--skip-optimization", action="store_true",
                        help="Skip Bayesian optimization (grid only)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(
        n_grid=args.n_grid,
        mc_samples=args.mc_samples,
        spec_set=args.spec_set,
        n_calls=args.n_calls,
        skip_optimization=args.skip_optimization,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )


if __name__ == "__main__":
    cli()
