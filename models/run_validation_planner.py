"""
Driver — Validation experiment planner.

Produces 4 output plots:
  validation_experiment_sequence.png  — ordered experiment sequence with gain/$
  validation_uncertainty_reduction.png — cumulative variance reduction trajectory
  validation_cost_effectiveness.png   — marginal gain/$ bar chart
  validation_plan_gantt.png          — Gantt chart of experiment timeline

Usage:
  python -m models.run_validation_planner [--output-dir outputs/]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .uncertainty.parameter_registry import REGISTRY
from .uncertainty.validation_planner import (
    experiment_catalog,
    plan_validation_experiments,
    sequential_planner,
    uncertainty_reduction_trajectory,
    ValidationPlan,
    UncertaintyTrajectory,
)


def _plot_experiment_sequence(plan: ValidationPlan, ax):
    """Bar chart: experiment sequence with gain-per-dollar overlay."""
    names = [e.name.replace("_", "\n") for e in plan.experiments]
    costs = [e.cost_usd for e in plan.experiments]
    gains = plan.gain_per_dollar

    x = np.arange(len(names))
    width = 0.35

    bars_cost = ax.bar(x - width / 2, costs, width, label="Cost (USD)",
                       color="steelblue", alpha=0.7)
    ax.set_ylabel("Cost (USD)", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax.twinx()
    bars_gain = ax2.bar(x + width / 2, gains, width, label="Gain / $",
                        color="coral", alpha=0.7)
    ax2.set_ylabel("Information gain / dollar", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=0)
    ax.set_title("Validation Experiment Sequence (ranked by gain/$)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)


def _plot_uncertainty_reduction(traj: UncertaintyTrajectory, ax):
    """Line plot: cumulative uncertainty reduction trajectory."""
    ax.plot(range(len(traj.remaining_variance_frac)),
            [100.0 * f for f in traj.remaining_variance_frac],
            "o-", color="seagreen", linewidth=2, markersize=8,
            label="Remaining variance %")
    ax.fill_between(range(len(traj.remaining_variance_frac)),
                    [100.0 * f for f in traj.remaining_variance_frac],
                    alpha=0.15, color="seagreen")

    ax.set_xticks(range(len(traj.experiment_names)))
    ax.set_xticklabels(
        [n.replace("_", "\n") for n in traj.experiment_names],
        fontsize=7, rotation=30, ha="right",
    )
    ax.set_ylabel("Remaining output variance (%)")
    ax.set_title("Uncertainty Reduction Trajectory")
    ax.set_ylim(0, 105)
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="20% target")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_cost_effectiveness(plan: ValidationPlan, ax):
    """Horizontal bar chart: marginal gain/$ per experiment."""
    names = [e.name for e in plan.experiments]
    gains = plan.gain_per_dollar
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(names)))
    ax.barh(names, gains, color=colors)
    ax.set_xlabel("Information gain per dollar")
    ax.set_title("Cost-Effectiveness Ranking")
    ax.tick_params(axis="y", labelsize=8)


def _plot_gantt(plan: ValidationPlan, ax):
    """Horizontal bar (Gantt) chart: experiment timeline."""
    cumulative = 0.0
    for i, exp in enumerate(plan.experiments):
        ax.barh(i, exp.duration_hours, left=cumulative,
                color=plt.cm.Set2(i / max(len(plan.experiments), 1)),
                edgecolor="black", linewidth=0.5)
        ax.text(cumulative + exp.duration_hours / 2, i,
                f"${exp.cost_usd:.0f}",
                ha="center", va="center", fontsize=7, fontweight="bold")
        cumulative += exp.duration_hours

    ax.set_yticks(range(len(plan.experiments)))
    ax.set_yticklabels(
        [e.name.replace("_", " ") for e in plan.experiments], fontsize=8
    )
    ax.set_xlabel("Duration (hours)")
    ax.set_title("Validation Plan Timeline")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)


def run(output_dir: str = "outputs") -> dict:
    """Execute the validation planner and produce 4 plots + JSON summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Validation experiment planner")
    print("=" * 50)

    # 1. Initial plan
    print("\n1. Planning experiments (greedy by gain/$)...")
    plan = plan_validation_experiments(REGISTRY, budget=7)
    print(f"   Selected {len(plan.experiments)} experiments")
    print(f"   Total cost: ${plan.total_cost_usd:.0f}")
    print(f"   Total duration: {plan.total_duration_hours:.0f} h")

    # 2. Sequential planning simulation
    print("\n2. Sequential planning simulation...")
    completed: list = []
    sequential_plans: list = []
    for exp in plan.experiments:
        completed.append(exp.name)
        remaining = sequential_planner(REGISTRY, completed)
        sequential_plans.append(remaining)
        print(f"   After {exp.name}: {len(remaining.experiments)} remaining")

    # 3. Uncertainty trajectory
    print("\n3. Computing uncertainty reduction trajectory...")
    traj = uncertainty_reduction_trajectory(REGISTRY, plan)
    final_pct = traj.remaining_variance_frac[-1] * 100
    print(f"   Final remaining variance: {final_pct:.1f}%")

    # 4. Generate plots
    print("\n4. Generating figures...")

    fig, ax = plt.subplots(figsize=(12, 5))
    _plot_experiment_sequence(plan, ax)
    fig.tight_layout()
    fig.savefig(out / "validation_experiment_sequence.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_uncertainty_reduction(traj, ax)
    fig.tight_layout()
    fig.savefig(out / "validation_uncertainty_reduction.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_cost_effectiveness(plan, ax)
    fig.tight_layout()
    fig.savefig(out / "validation_cost_effectiveness.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    _plot_gantt(plan, ax)
    fig.tight_layout()
    fig.savefig(out / "validation_plan_gantt.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved to {out}/")

    # 5. JSON summary
    summary = {
        "n_experiments": len(plan.experiments),
        "total_cost_usd": plan.total_cost_usd,
        "total_duration_hours": plan.total_duration_hours,
        "final_remaining_variance_pct": final_pct,
        "experiments": [
            {
                "name": e.name,
                "cost_usd": e.cost_usd,
                "duration_hours": e.duration_hours,
                "constrained_params": e.constrained_params,
                "gain_per_dollar": g,
            }
            for e, g in zip(plan.experiments, plan.gain_per_dollar)
        ],
        "trajectory": {
            "experiment_names": traj.experiment_names,
            "remaining_variance_pct": [100.0 * f for f in traj.remaining_variance_frac],
            "cumulative_cost_usd": traj.cumulative_cost_usd,
        },
        "plots": [
            str(out / "validation_experiment_sequence.png"),
            str(out / "validation_uncertainty_reduction.png"),
            str(out / "validation_cost_effectiveness.png"),
            str(out / "validation_plan_gantt.png"),
        ],
    }

    (out / "validation_plan_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Summary saved to {out}/validation_plan_summary.json")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validation experiment planner — DOE to reduce uncertainty"
    )
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory for plots and JSON")
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(f"\nPlanner complete.")
    print(f"  {summary['n_experiments']} experiments, "
          f"${summary['total_cost_usd']:.0f} total")
    print(f"  Final remaining variance: "
          f"{summary['final_remaining_variance_pct']:.1f}%")


if __name__ == "__main__":
    main()
