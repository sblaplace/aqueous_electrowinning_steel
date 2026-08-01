"""Run membrane fouling analysis for the electrolyte recirculation loop.

Generates:
  - 4 figures (flux decline, cleaning cycles, impurity accumulation, cost breakdown)
  - JSON report

Usage:
    python -m models.run_membrane_fouling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.membrane_fouling import (
    CSTRFoulingCoupling,
    FoulingRateParams,
    HermiaModel,
    MembraneFoulingModel,
    MembraneParams,
    hermia_flux,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
DATA = ROOT / "experiments" / "data"


def build_base_model() -> MembraneFoulingModel:
    """Reproducible baseline screening case."""
    return MembraneFoulingModel(
        membrane=MembraneParams(
            area_m2=10.0,
            clean_water_flux_L_m2_hr=100.0,
            pore_diameter_um=0.1,
            membrane_cost_per_m2=200.0,
            replacement_labor_cost=500.0,
        ),
        fouling=FoulingRateParams(
            fe_oh3_base_rate=1.2e-3,
            caso4_base_rate=5.0e-4,
            organic_base_rate=2.0e-4,
            biofilm_base_rate=8.0e-4,
            fe_oh3_pH_threshold=3.0,
            caso4_hardness_mg_L=200.0,
            biofilm_idle_hr=24.0,
        ),
        coupling=CSTRFoulingCoupling(
            base_rejection=0.90,
            max_rejection=0.999,
            rejection_fouling_coupling=0.5,
            impurity_source_mol_hr=1.0e-3,
            loop_flow_L_hr=20.0,
            loop_volume_L=1000.0,
        ),
        hermia_variant=HermiaModel.CAKE_FILTRATION,
        operating_pH=2.0,
        hardness_mg_L=50.0,
        idle_time_hr=0.0,
        temperature_C=60.0,
    )


def plot_flux_decline(result, path: Path) -> None:
    """Figure 1: Flux decline with per-mechanism resistance overlay."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    t = result.flux_decline.time_hr

    ax1.plot(t, result.flux_decline.flux_L_m2_hr, "k-", lw=2, label="Total flux J(t)")
    ax1.set_xlabel("Time (hr)")
    ax1.set_ylabel("Flux (LMH)", color="k")
    ax1.tick_params(axis="y", labelcolor="k")

    ax2 = ax1.twinx()
    ax2.fill_between(t, 0, result.flux_decline.fe_oh3_resistance,
                      alpha=0.3, color="tab:red", label="Fe(OH)₃")
    ax2.fill_between(t, result.flux_decline.fe_oh3_resistance,
                      result.flux_decline.fe_oh3_resistance + result.flux_decline.caso4_resistance,
                      alpha=0.3, color="tab:blue", label="CaSO₄")
    cum = (result.flux_decline.fe_oh3_resistance + result.flux_decline.caso4_resistance
           + result.flux_decline.organic_resistance)
    ax2.fill_between(t,
                      result.flux_decline.fe_oh3_resistance + result.flux_decline.caso4_resistance,
                      cum,
                      alpha=0.3, color="tab:green", label="Organic")
    ax2.fill_between(t, cum,
                      cum + result.flux_decline.biofilm_resistance,
                      alpha=0.3, color="tab:orange", label="Biofilm")
    ax2.set_ylabel("Cumulative fouling resistance (a.u.)", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Membrane Flux Decline with Per-Mechanism Fouling\n"
                   "(Synthetic screening — not experimental data)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_cleaning_cycles(result, path: Path) -> None:
    """Figure 2: Flux with cleaning events marked."""
    fig, ax = plt.subplots(figsize=(10, 6))
    t = result.cleaning.time_hr
    ax.plot(t, result.cleaning.flux_L_m2_hr, "b-", lw=1.5, label="Flux with cleaning")

    # Mark cleaning events
    for time_hr, agent in result.cleaning.cleaning_events[:20]:  # limit to 20 markers
        color = {"acid_wash": "tab:red", "naoh_wash": "tab:green",
                 "backflush": "tab:orange"}.get(agent, "gray")
        ax.axvline(time_hr, color=color, alpha=0.4, lw=0.8)

    # Legend for cleaning agents
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="tab:red", alpha=0.4, label="Acid wash"),
        Line2D([0], [0], color="tab:green", alpha=0.4, label="NaOH wash"),
        Line2D([0], [0], color="tab:orange", alpha=0.4, label="Backflush"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Flux (LMH)")
    ax.set_title(f"Flux with Cleaning Cycles (interval = "
                 f"{result.cleaning.optimal_cleaning_interval_hr:.0f} hr)\n"
                 f"Total cleanings: {result.cleaning.n_cleanings}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_impurity_accumulation(result, path: Path) -> None:
    """Figure 3: Impurity concentration and rejection over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    t = result.flux_decline.time_hr

    ax1.plot(t, result.flux_decline.impurity_M * 1000, "r-", lw=2)
    ax1.set_xlabel("Time (hr)")
    ax1.set_ylabel("Impurity (mmol/L)")
    ax1.set_title("Impurity Accumulation in CSTR Loop")
    ax1.grid(alpha=0.3)

    ax2.plot(t, result.flux_decline.rejection * 100, "b-", lw=2)
    ax2.set_xlabel("Time (hr)")
    ax2.set_ylabel("Rejection (%)")
    ax2.set_title("Membrane Rejection vs Fouling")
    ax2.grid(alpha=0.3)

    fig.suptitle("Impurity-Rejection Coupling (C_imp = S / (Q × R_rej))",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_cost_breakdown(result, path: Path) -> None:
    """Figure 4: Economics — cost breakdown and Hermia model comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: cost breakdown
    costs = {
        "Cleaning": result.cleaning.total_cleaning_cost,
        "Membrane\nreplacement": result.cleaning.total_membrane_cost,
    }
    labels = list(costs.keys())
    values = list(costs.values())
    colors = ["#2196F3", "#E91E63"]
    bars = ax1.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1,
                 f"${val:.0f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Cost ($)")
    ax1.set_title("Membrane Fouling Cost Breakdown")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Hermia model comparison
    t = np.linspace(0, 4000, 200)
    J0 = 100.0
    K = 2e-4
    for model in HermiaModel:
        J = hermia_flux(t, J0, K, model)
        ax2.plot(t, J, lw=1.5, label=model.value.replace("_", " ").title())
    ax2.set_xlabel("Time (hr)")
    ax2.set_ylabel("Flux (LMH)")
    ax2.set_title("Hermia Fouling Model Variants")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Membrane Economics & Hermia Model Comparison",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MEMBRANE FOULING MODEL — CSTR LOOP DEGRADATION ANALYSIS")
    print("=" * 70)

    model = build_base_model()
    result = model.simulate(duration_hr=4000.0, dt_hr=1.0)

    # Print summary
    summary = result.summary()
    print(f"\n  Duration:            {summary['flux_decline']['duration_hr']:.0f} hr")
    print(f"  Initial flux:        {summary['flux_decline']['initial_flux_L_m2_hr']:.1f} LMH")
    print(f"  Final flux:          {summary['flux_decline']['final_flux_L_m2_hr']:.1f} LMH")
    print(f"  Flux decline:        {summary['flux_decline']['flux_decline_pct']:.1f}%")
    print(f"  Cleanings:           {summary['n_cleanings']}")
    print(f"  Cleaning cost:       ${summary['total_cleaning_cost']:.0f}")
    print(f"  Membrane cost:       ${summary['total_membrane_cost']:.0f}")
    print(f"  Optimal interval:    {summary['optimal_cleaning_interval_hr']:.0f} hr")

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_flux_decline(result, FIGURES / "membrane_flux_decline.png")
    plot_cleaning_cycles(result, FIGURES / "membrane_cleaning_cycles.png")
    plot_impurity_accumulation(result, FIGURES / "membrane_impurity_accumulation.png")
    plot_cost_breakdown(result, FIGURES / "membrane_cost_hermia.png")

    # Save JSON report
    print("\n" + "=" * 70)
    print("SAVING REPORT")
    print("=" * 70)
    report = {
        "provenance": "Synthetic membrane fouling screening; not experimental data.",
        "model_scope": (
            "Hermia-based flux decline with four parallel fouling mechanisms "
            "(Fe(OH)₃, CaSO₄, organic, biofilm), cleaning cycle optimization, "
            "and impurity-rejection coupling in a CSTR loop."
        ),
        "summary": summary,
        "hermia_variants": [m.value for m in HermiaModel],
        "fouling_mechanisms": ["fe_oh3", "caso4", "organic", "biofilm"],
        "membrane_params": {
            "area_m2": model.membrane.area_m2,
            "clean_water_flux_L_m2_hr": model.membrane.clean_water_flux_L_m2_hr,
            "membrane_cost_per_m2": model.membrane.membrane_cost_per_m2,
        },
    }
    report_path = DATA / "membrane_fouling_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n",
                           encoding="utf-8")
    print(f"  ✅ Saved: {report_path}")

    print("\n✅ Membrane fouling analysis complete!")


if __name__ == "__main__":
    main()
