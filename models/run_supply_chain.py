#!/usr/bin/env python3
"""
Driver for supply chain analysis.

Generates:
  - supply_material_balance.png  — Feedstock cost breakdown
  - supply_recycling.png         — Electrolyte recycling economics
  - supply_location_comparison.png — Multi-criteria site scoring
  - supply_sensitivity.png       — Electricity price sensitivity

Usage:
    python -m models.run_supply_chain
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.supply_chain import (
    DesignPoint, material_balance, electrolyte_recycling,
    compare_locations, electricity_sensitivity,
    CANDIDATE_LOCATIONS, LocationWeight, RAW_MATERIALS,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_material_balance(dp: DesignPoint, rate_kg_day: float) -> Path:
    """Bar chart: feedstock cost breakdown by chemical."""
    bal = material_balance(dp, rate_kg_day)
    names = [it.name for it in bal.items]
    costs = [it.annual_cost for it in bal.items]
    fracs = [it.fraction_of_total for it in bal.items]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of annual costs
    bars = ax1.barh(names, costs, color=colors[:len(names)], edgecolor="white")
    for bar, val in zip(bars, costs):
        ax1.text(bar.get_width() + max(costs) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"${val:,.0f}/yr", va="center", fontsize=10)
    ax1.set_xlabel("Annual Cost ($/yr)", fontsize=12)
    ax1.set_title(f"Feedstock Cost Breakdown\n({rate_kg_day:.0f} kg Fe/day)",
                  fontsize=13, fontweight="bold")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.invert_yaxis()

    # Pie chart of fractions
    ax2.pie(fracs, labels=names, autopct="%1.1f%%", colors=colors[:len(names)],
            startangle=140, textprops={"fontsize": 10})
    ax2.set_title("Cost Share by Chemical", fontsize=13, fontweight="bold")

    fig.suptitle(f"Material Balance — Specific feedstock cost: ${bal.specific_feedstock_cost_per_kg:.3f}/kg Fe",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / "supply_material_balance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_recycling(dp: DesignPoint, rate_kg_day: float) -> Path:
    """Bar + table: electrolyte recycling economics."""
    bal = material_balance(dp, rate_kg_day)
    rec = electrolyte_recycling(bal)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Cost breakdown
    labels = ["Makeup\nchemicals", "Purge\ntreatment"]
    costs = [rec.makeup_chemical_cost_per_day, rec.purge_treatment_cost_per_day]
    colors = ["#4CAF50", "#FF5722"]
    bars = ax1.bar(labels, costs, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, costs):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.05,
                 f"${val:.2f}/day", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Daily Cost ($/day)", fontsize=12)
    ax1.set_title("Electrolyte Recycling Costs", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Info table
    ax2.axis("off")
    info = [
        ("Fe²⁺ depletion rate", f"{rec.fe2_depletion_rate_kg_day:.1f} kg/day"),
        ("FeSO4 makeup rate", f"{rec.fe2_makeup_rate_kg_day:.1f} kg/day"),
        ("Purge rate", f"{rec.purge_rate_L_day:.1f} L/day"),
        ("Purge fraction", f"{rec.purge_fraction_per_day * 100:.2f}%/day"),
        ("Makeup cost/kg Fe", f"${rec.makeup_chemical_cost_per_kg_fe:.4f}"),
        ("Purge cost/kg Fe", f"${rec.purge_treatment_cost_per_day / rate_kg_day:.4f}" if rate_kg_day > 0 else "N/A"),
        ("Total recycling/kg Fe", f"${rec.total_recycling_cost_per_kg_fe:.4f}"),
    ]
    table = ax2.table(cellText=info, colLabels=["Parameter", "Value"],
                      loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax2.set_title("Recycling Summary", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    path = OUTPUT_DIR / "supply_recycling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_location_comparison(dp: DesignPoint, rate_kg_day: float) -> Path:
    """Radar + bar: multi-criteria location scoring."""
    ranking = compare_locations(CANDIDATE_LOCATIONS, dp, rate_kg_day)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart of total scores
    names = [loc.name for loc in ranking.locations]
    scores = [loc.total_score for loc in ranking.locations]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))

    bars = ax1.barh(names, scores, color=colors, edgecolor="white")
    for bar, val in zip(bars, scores):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=10)
    ax1.set_xlabel("Weighted Score (0-1)", fontsize=12)
    ax1.set_title("Location Ranking by Total Score", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, 1.0)
    ax1.invert_yaxis()

    # Stacked bar: contribution of each criterion
    criteria = list(ranking.locations[0].weighted_scores.keys())
    crit_labels = [c.replace("_", " ").title() for c in criteria]
    crit_colors = plt.cm.Set3(np.linspace(0, 1, len(criteria)))

    bottoms = np.zeros(len(names))
    for ci, (crit, label, color) in enumerate(zip(criteria, crit_labels, crit_colors)):
        vals = [loc.weighted_scores[crit] for loc in ranking.locations]
        ax2.barh(names, vals, left=bottoms, label=label, color=color, edgecolor="white")
        bottoms += np.array(vals)

    ax2.set_xlabel("Weighted Score Contribution", fontsize=12)
    ax2.set_title("Score Breakdown by Criterion", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right", ncol=2)
    ax2.invert_yaxis()

    plt.tight_layout()
    path = OUTPUT_DIR / "supply_location_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_sensitivity(dp: DesignPoint, rate_kg_day: float) -> Path:
    """Line chart: cost sensitivity to electricity price."""
    result = electricity_sensitivity(dp, rate_kg_day)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Stacked area
    prices = result.electricity_prices
    ax1.fill_between(prices, 0, result.feedstock_costs, alpha=0.6, label="Feedstock", color="#4CAF50")
    ax1.fill_between(prices, result.feedstock_costs,
                     [f + r for f, r in zip(result.feedstock_costs, result.recycling_costs)],
                     alpha=0.6, label="Recycling", color="#2196F3")
    ax1.fill_between(prices,
                     [f + r for f, r in zip(result.feedstock_costs, result.recycling_costs)],
                     result.total_costs,
                     alpha=0.6, label="Electricity", color="#FF9800")
    ax1.set_xlabel("Electricity Price ($/kWh)", fontsize=12)
    ax1.set_ylabel("Specific Cost ($/kg Fe)", fontsize=12)
    ax1.set_title("Cost vs Electricity Price\n(Stacked Components)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Electricity share
    shares = [e / t * 100 if t > 0 else 0
              for e, t in zip(result.electricity_costs_per_kg, result.total_costs)]
    ax2.plot(prices, shares, "o-", color="#FF9800", linewidth=2, markersize=4)
    ax2.set_xlabel("Electricity Price ($/kWh)", fontsize=12)
    ax2.set_ylabel("Electricity Share of Total Cost (%)", fontsize=12)
    ax2.set_title("Electricity Cost Fraction", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    path = OUTPUT_DIR / "supply_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    dp = DesignPoint()
    rate = dp.production_rate_kg_hr() * 24  # kg/day
    print(f"Design point production rate: {rate:.1f} kg Fe/day")
    print(f"Specific energy: {dp.energy_kWh_per_kg():.2f} kWh/kg Fe")

    print("\n--- Material Balance ---")
    plot_material_balance(dp, rate)

    print("\n--- Electrolyte Recycling ---")
    plot_recycling(dp, rate)

    print("\n--- Location Comparison ---")
    plot_location_comparison(dp, rate)

    print("\n--- Electricity Sensitivity ---")
    plot_sensitivity(dp, rate)

    print("\nDone. All plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
