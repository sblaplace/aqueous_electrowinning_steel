#!/usr/bin/env python3
"""
Driver for pilot-plant CAPEX/OPEX analysis.

Generates:
  - capex_by_scale.png        — CAPEX breakdown at 3 scales
  - capex_category.png        — CAPEX by equipment category
  - opex_breakdown_pilot.png  — OPEX pie chart at pilot scale
  - capex_tornado.png         — Tornado sensitivity chart
  - pilot_costing_report.json — Full JSON report

Usage:
    python -m models.run_pilot_costing
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pilot_costing import (
    estimate_capex,
    capex_at_all_scales,
    capex_by_category,
    capex_sensitivity_tornado,
    PilotOPEXModel,
    PilotCAPEXResult,
    PID_EQUIPMENT,
    SCALE_LAB,
    SCALE_PILOT,
    SCALE_PRODUCTION,
    equipment_table,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "data"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def plot_capex_by_scale(scales: dict) -> Path:
    """Bar chart: CAPEX at lab / pilot / production."""
    names = ["Lab\n1 kg/day", "Pilot\n10 kg/day", "Production\n100 kg/day"]
    keys = ["lab", "pilot", "production"]
    totals = [scales[k].total_capex for k in keys]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, totals, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                f"${val:,.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Total CAPEX ($)", fontsize=12)
    ax.set_title("Pilot CAPEX at Three Production Scales\n(Six-tenths rule scaling)", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = OUTPUT_DIR / "capex_by_scale.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_capex_category(scales: dict) -> Path:
    """Stacked bar: CAPEX by category at each scale."""
    keys = ["lab", "pilot", "production"]
    labels = ["Lab", "Pilot", "Production"]
    cats_order = ["tanks", "cell", "furnace", "gas", "instruments"]
    cat_labels = ["Tanks/Pumps", "Cell Stack", "Furnaces", "Gas Handling", "Instruments"]
    colors = ["#a6cee3", "#b2df8a", "#ff7f00", "#fb9a99", "#cab2d6"]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(keys))
    width = 0.5

    bottoms = np.zeros(len(keys))
    for ci, (cat, cat_label, color) in enumerate(zip(cats_order, cat_labels, colors)):
        vals = [capex_by_category(scales[k]).get(cat, 0) for k in keys]
        ax.bar(x, vals, width, bottom=bottoms, label=cat_label, color=color, edgecolor="white")
        bottoms += np.array(vals)

    # Add piping/eng/cont on top
    extras = [scales[k].piping_structural + scales[k].engineering + scales[k].contingency for k in keys]
    ax.bar(x, extras, width, bottom=bottoms, label="Piping/Eng/Contingency", color="#d9d9d9", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("CAPEX ($)", fontsize=12)
    ax.set_title("CAPEX Breakdown by Equipment Category", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = OUTPUT_DIR / "capex_category.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_opex_breakdown(opex_data: dict) -> Path:
    """Pie chart of OPEX at pilot scale."""
    categories = {
        "Electrowinning\nElectricity": opex_data["Electrowinning electricity ($/yr)"],
        "Gas costs": opex_data["Gas costs ($/yr)"],
        "Furnace\nElectricity": opex_data["Furnace electricity ($/yr)"],
        "Quench media": opex_data["Quench media ($/yr)"],
        "Ore": opex_data["Iron ore ($/yr)"],
        "Labor": opex_data["Labor ($/yr)"],
        "Maintenance": opex_data["General maintenance ($/yr)"] + opex_data["Instrument maintenance ($/yr)"],
        "Other": (opex_data["Electrolyte makeup ($/yr)"] + opex_data["Anode replacement ($/yr)"]
                  + opex_data["Membrane/filter replacement ($/yr)"]),
    }
    total = sum(categories.values())
    labels = list(categories.keys())
    values = list(categories.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.8,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(10)
    ax.set_title(f"Annual OPEX Breakdown (Pilot 10 kg/day)\nTotal: ${total:,.0f}/yr",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    path = OUTPUT_DIR / "opex_breakdown_pilot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_tornado(sensitivity: dict) -> Path:
    """Tornado chart: CAPEX sensitivity."""
    base_capex = sensitivity["base"]["CAPEX"]
    params = ["cell_current", "carburizing_temp", "production_scale"]
    param_labels = ["Cell Current\n(cost multiplier)", "Carburizing Temp\n(cost multiplier)", "Production Scale\n(kg/day)"]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(params))
    bar_height = 0.5

    for i, (param, label) in enumerate(zip(params, param_labels)):
        low_key = f"{param}_low"
        high_key = f"{param}_high"
        low_val = sensitivity[low_key]["CAPEX"]
        high_val = sensitivity[high_key]["CAPEX"]

        low_delta = low_val - base_capex
        high_delta = high_val - base_capex

        left = min(low_delta, high_delta, 0)
        right = max(low_delta, high_delta, 0)

        ax.barh(i, right - left, bar_height, left=base_capex + left,
                color="#E91E63" if abs(high_delta) > abs(low_delta) else "#4CAF50",
                alpha=0.7, edgecolor="white", linewidth=1)

        # Value labels
        low_label = f"${low_val:,.0f}"
        high_label = f"${high_val:,.0f}"
        if param == "production_scale":
            low_label += f"\n({sensitivity[low_key]['scale']:.0f} kg/d)"
            high_label += f"\n({sensitivity[high_key]['scale']:.0f} kg/d)"
        else:
            low_label += f"\n({sensitivity[low_key]['value']:.1f}x)"
            high_label += f"\n({sensitivity[high_key]['value']:.1f}x)"

        ax.text(base_capex + low_delta - 500, i, low_label,
                ha="right", va="center", fontsize=8, color="#4CAF50")
        ax.text(base_capex + high_delta + 500, i, high_label,
                ha="left", va="center", fontsize=8, color="#E91E63")

    ax.axvline(x=base_capex, color="black", linewidth=2)
    ax.text(base_capex, len(params) - 0.3, f"Base: ${base_capex:,.0f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_labels, fontsize=10)
    ax.set_xlabel("Total CAPEX ($)", fontsize=12)
    ax.set_title("CAPEX Sensitivity — Tornado Chart\n(Pilot scale, 10 kg/day)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = OUTPUT_DIR / "capex_tornado.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    print("=" * 70)
    print("PILOT CAPEX/OPEX MODEL — EQUIPMENT COSTING FROM P&ID")
    print("=" * 70)

    # CAPEX at all scales
    scales = capex_at_all_scales()
    print("\n┌─ CAPEX Summary ─────────────────────────────────────────────┐")
    for name, result in scales.items():
        print(f"│  {name:>12s} ({result.scale_kg_day:>6.0f} kg/day): ${result.total_capex:>12,.0f}  │")
    print("└──────────────────────────────────────────────────────────────┘")

    # Equipment table
    print("\n┌─ P&ID Equipment List (reference costs at 10 kg/day) ───────┐")
    for item in PID_EQUIPMENT:
        print(f"│  {item.tag:<8s} ${item.reference_cost_usd:>10,.0f}  {item.name[:38]:<38s}│")
    print("└──────────────────────────────────────────────────────────────┘")

    # OPEX at pilot
    opex_model = PilotOPEXModel()
    pilot_capex = scales["pilot"]
    opex = opex_model.estimate(SCALE_PILOT, pilot_capex)
    print(f"\n┌─ OPEX at Pilot Scale (10 kg/day) ──────────────────────────┐")
    for key in [
        "Electrowinning electricity ($/yr)", "Gas costs ($/yr)",
        "Furnace electricity ($/yr)", "Quench media ($/yr)",
        "Iron ore ($/yr)", "Labor ($/yr)",
        "General maintenance ($/yr)", "Instrument maintenance ($/yr)",
        "Total OPEX ($/yr)", "Specific OPEX ($/kg Fe)",
    ]:
        print(f"│  {key:<40s} ${opex[key]:>10,.0f}  │")
    print("└──────────────────────────────────────────────────────────────┘")

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    fig1 = plot_capex_by_scale(scales)
    fig2 = plot_capex_category(scales)
    fig3 = plot_opex_breakdown(opex)
    sensitivity = capex_sensitivity_tornado()
    fig4 = plot_tornado(sensitivity)

    # JSON report
    report = {
        "title": "Pilot CAPEX/OPEX Model — Equipment Costing from P&ID",
        "equipment_table": equipment_table(),
        "capex": {name: result.to_dict() for name, result in scales.items()},
        "opex_pilot": opex,
        "sensitivity": sensitivity,
    }
    report_path = REPORT_DIR / "pilot_costing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved: {report_path}")
    print("\n✅ Pilot costing analysis complete!")


if __name__ == "__main__":
    main()
