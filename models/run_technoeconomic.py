#!/usr/bin/env python3
"""
Run techno-economic analysis for aqueous electrowinning of iron/steel.

Generates:
  - Summary report (JSON)
  - Comparison charts (PNG)
  - Sensitivity analysis plots

Usage:
    python -m models.run_technoeconomic
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    CellVoltageModel,
    ElectrolyzerParams,
    CAPEXModel,
    OPEXModel,
    LevelizedCost,
    sensitivity_analysis,
    compare_routes,
    BENCHMARK_COSTS,
    specific_energy_kWh_per_t,
)

# ─── Output Directory ─────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "data"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_base_case():
    """Run the base-case techno-economic analysis."""
    print("=" * 70)
    print("AQUEOUS ELECTROWINNING — TECHNO-ECONOMIC ANALYSIS")
    print("=" * 70)

    # ─── Electrochemical Cell Model ──────────────────────────────────
    print("\n┌─ Cell Voltage Decomposition ─────────────────────────────────┐")
    cell = CellVoltageModel(
        E_cathode_eq=-0.440,   # Fe²⁺/Fe
        E_anode_eq=1.229,      # OER
        eta_cathode=0.30,      # cathode overpotential
        eta_anode=0.40,        # anode overpotential
        ir_drop=0.20,          # ohmic losses
    )
    for k, v in cell.summary().items():
        print(f"│  {k:<25s} {v:>8.3f}  │")
    print(f"│  {'─'*42}│")
    print(f"│  {'TOTAL CELL VOLTAGE':<25s} {cell.V_cell:>8.3f} V│")
    print("└──────────────────────────────────────────────────────────────┘")

    # ─── Electrolyzer Parameters ─────────────────────────────────────
    params = ElectrolyzerParams(
        current_density_mA_cm2=100.0,
        current_efficiency=0.90,
        cell_voltage=cell.V_cell,
        temperature_C=70.0,
        electrode_area_m2=1.0,
        n_cells=100,
        electrolyte_type="alkaline",
    )

    print(f"\n┌─ Electrolyzer Stack Parameters ──────────────────────────────┐")
    print(f"│  Current density:        {params.current_density_mA_cm2:>6.0f} mA/cm²       │")
    print(f"│  Current efficiency:     {params.current_efficiency*100:>6.0f} %             │")
    print(f"│  Cell voltage:           {params.cell_voltage:>6.2f} V              │")
    print(f"│  Electrode area:         {params.electrode_area_m2:>6.1f} m²              │")
    print(f"│  Cells per stack:        {params.n_cells:>6d}                  │")
    print(f"│  Stack power:            {params.stack_power_kW():>6.0f} kW             │")
    print(f"│  Production rate:        {params.production_rate_kg_hr():>6.1f} kg/hr           │")
    print(f"│  Annual production:      {params.production_rate_t_yr():>6.0f} t/yr/stack       │")
    print("└──────────────────────────────────────────────────────────────┘")

    # ─── Energy Consumption ──────────────────────────────────────────
    e_spec = specific_energy_kWh_per_t(cell.V_cell, params.current_efficiency)
    print(f"\n  Specific energy consumption: {e_spec:.0f} kWh/t Fe")
    print(f"  (Target from literature: <1,500 kWh/t Fe)")

    # ─── Plant Configuration ─────────────────────────────────────────
    n_stacks = 10
    capex_model = CAPEXModel()
    opex_model = OPEXModel(electricity_price_kWh=0.04)
    lc_model = LevelizedCost()

    capex = capex_model.estimate(params, n_stacks)
    opex = opex_model.estimate(params, capex["Total CAPEX ($)"], n_stacks)
    annual_prod = capex["Annual capacity (t/yr)"]

    print(f"\n┌─ Plant Configuration: {n_stacks} stacks ─────────────────────────────────┐")
    print(f"│  Annual capacity:      {annual_prod:>10,.0f} t/yr             │")
    print(f"│  Total CAPEX:          ${capex['Total CAPEX ($)']/1e6:>9.1f} M           │")
    print(f"│  Annual OPEX:          ${opex['Total OPEX ($/yr)']/1e6:>9.1f} M/yr        │")
    print("└──────────────────────────────────────────────────────────────┘")

    # ─── CAPEX Breakdown ─────────────────────────────────────────────
    print(f"\n┌─ CAPEX Breakdown ────────────────────────────────────────────┐")
    capex_items = [
        ("Electrodes", "Electrodes ($)"),
        ("Membranes/separators", "Membranes/separators ($)"),
        ("Cell hardware", "Cell hardware ($)"),
        ("Rectifiers", "Rectifiers ($)"),
        ("Electrolyte system", "Electrolyte system ($)"),
        ("Ore leaching", "Ore leaching ($)"),
        ("Infrastructure", "Infrastructure ($)"),
        ("Engineering", "Engineering ($)"),
        ("Contingency", "Contingency ($)"),
    ]
    for label, key in capex_items:
        val = capex[key]
        pct = val / capex["Total CAPEX ($)"] * 100
        bar = "█" * int(pct / 2)
        print(f"│  {label:<22s} ${val/1e6:>6.2f}M ({pct:>4.1f}%) {bar:<15s}│")
    print(f"│  {'─'*52}│")
    print(f"│  {'TOTAL':<22s} ${capex['Total CAPEX ($)']/1e6:>6.1f}M                 │")
    print("└──────────────────────────────────────────────────────────────┘")

    # ─── OPEX Breakdown ──────────────────────────────────────────────
    print(f"\n┌─ OPEX Breakdown (Annual) ────────────────────────────────────┐")
    opex_items = [
        ("Electricity", "Electricity ($/yr)"),
        ("Electrolyte makeup", "Electrolyte makeup ($/yr)"),
        ("Iron ore feedstock", "Iron ore feedstock ($/yr)"),
        ("Water", "Water ($/yr)"),
        ("Anode replacement", "Anode replacement ($/yr)"),
        ("Maintenance", "Maintenance ($/yr)"),
        ("Insurance", "Insurance ($/yr)"),
        ("Labor", "Labor ($/yr)"),
        ("Overhead", "Overhead ($/yr)"),
    ]
    for label, key in opex_items:
        val = opex[key]
        pct = val / opex["Total OPEX ($/yr)"] * 100
        bar = "█" * int(pct / 2)
        print(f"│  {label:<22s} ${val/1e6:>6.2f}M ({pct:>4.1f}%) {bar:<15s}│")
    print(f"│  {'─'*52}│")
    print(f"│  {'TOTAL':<22s} ${opex['Total OPEX ($/yr)']/1e6:>6.1f}M                 │")
    print("└──────────────────────────────────────────────────────────────┘")

    # ─── Levelized Cost ──────────────────────────────────────────────
    lcofe = lc_model.calculate(capex["Total CAPEX ($)"], opex["Total OPEX ($/yr)"], annual_prod)

    print(f"\n┌─ Levelized Cost of Iron (LCOFe) ─────────────────────────────┐")
    print(f"│  Capital recovery factor: {lcofe['Capital recovery factor']:>8.4f}            │")
    print(f"│  Annualized CAPEX:     ${lcofe['Annualized CAPEX ($/yr)']/1e6:>8.2f}M/yr          │")
    print(f"│  Annual OPEX:          ${lcofe['Annual OPEX ($/yr)']/1e6:>8.2f}M/yr          │")
    print(f"│  CAPEX share:            {lcofe['CAPEX share (%)']:>6.1f}%                │")
    print(f"│  OPEX share:             {lcofe['OPEX share (%)']:>6.1f}%                │")
    print(f"│  {'━'*52}│")
    print(f"│  LCOFe:                 ${lcofe['LCOFe ($/t Fe)']:>8.0f}/t Fe            │")
    print("└──────────────────────────────────────────────────────────────┘")

    return {
        "cell": cell,
        "params": params,
        "capex": capex,
        "opex": opex,
        "lcofe": lcofe,
        "n_stacks": n_stacks,
        "annual_prod": annual_prod,
    }


def plot_cost_comparison(base_lcofe: float):
    """Generate cost comparison bar chart."""
    comparison = compare_routes(base_lcofe, carbon_price_tCO2=50.0)

    routes = list(comparison.keys())
    base_costs = [comparison[r]["Base cost ($/t Fe)"] for r in routes]
    co2_costs = [comparison[r][f"Carbon cost @$50/tCO2 ($/t Fe)"] for r in routes]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(routes))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_costs, width, label="Base cost",
                   color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"], alpha=0.8)
    bars2 = ax.bar(x + width/2, co2_costs, width, label="Carbon cost (@$50/tCO₂)",
                   color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"], alpha=0.4,
                   edgecolor=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"])

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'${height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        total = base_costs[i] + co2_costs[i]
        ax.annotate(f'${total:.0f}',
                    xy=(b2.get_x() + b2.get_width() / 2, total),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel("Cost ($/t Fe)", fontsize=12)
    ax.set_title("Iron Production Cost Comparison\n(Base cost + carbon penalty @ $50/tCO₂)",
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace(" ", "\n") for r in routes], fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(base_costs) * 1.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "cost_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✅ Saved: {path}")
    return comparison


def plot_energy_breakdown(params, cell):
    """Plot cell voltage decomposition as a waterfall chart."""
    components = {
        "Thermodynamic\n(E°_OER − E°_Fe)": cell.E_thermodynamic,
        "Cathode\noverpotential (η_c)": cell.eta_cathode,
        "Anode\noverpotential (η_a)": cell.eta_anode,
        "Ohmic\n(iR drop)": cell.ir_drop,
    }

    labels = list(components.keys())
    values = list(components.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1565C0", "#E65100", "#E65100", "#E65100"]

    cumulative = 0
    bottoms = []
    for v in values:
        bottoms.append(cumulative)
        cumulative += v

    bars = ax.bar(labels, values, bottom=bottoms, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, val, bot in zip(bars, values, bottoms):
        ax.text(bar.get_x() + bar.get_width()/2, bot + val/2,
                f'{val:.2f} V', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    # Total line
    ax.axhline(y=cell.V_cell, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(len(labels) - 0.5, cell.V_cell + 0.05,
            f'Total: {cell.V_cell:.2f} V', color='red', fontsize=11, fontweight='bold')

    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.set_title("Cell Voltage Decomposition\n(Fe²⁺ electrowinning with OER anode)",
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, cell.V_cell * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "voltage_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_sensitivity(base_result):
    """Generate sensitivity tornado chart."""
    params = base_result["params"]
    sens = sensitivity_analysis(params, base_result["n_stacks"])
    base_lcofe = sens["base"]["LCOFe"]

    # Organize sensitivity results
    param_groups = {
        "Current efficiency": ("Current efficiency_low", "Current efficiency_high"),
        "Current density": ("Current density (mA/cm²)_low", "Current density (mA/cm²)_high"),
        "Cell voltage": ("Cell voltage (V)_low", "Cell voltage (V)_high"),
        "Electricity price": ("Electricity price_low", "Electricity price_high"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(param_groups))
    bar_height = 0.6

    for i, (param_name, (low_key, high_key)) in enumerate(param_groups.items()):
        low_val = sens[low_key]["LCOFe"]
        high_val = sens[high_key]["LCOFe"]

        low_delta = low_val - base_lcofe
        high_delta = high_val - base_lcofe

        # Draw bars from base
        left = min(low_delta, high_delta, 0)
        right = max(low_delta, high_delta, 0)

        ax.barh(i, right - left, bar_height, left=base_lcofe + left,
                color='#E91E63' if right > 0 else '#4CAF50', alpha=0.7,
                edgecolor='white', linewidth=1)

        # Labels
        ax.text(base_lcofe + low_delta - 5, i,
                f'${low_val:.0f}\n({sens[low_key]["value"]})',
                ha='right', va='center', fontsize=8, color='#4CAF50')
        ax.text(base_lcofe + high_delta + 5, i,
                f'${high_val:.0f}\n({sens[high_key]["value"]})',
                ha='left', va='center', fontsize=8, color='#E91E63')

    # Base line
    ax.axvline(x=base_lcofe, color='black', linewidth=2, linestyle='-')
    ax.text(base_lcofe, len(param_groups) - 0.3, f'Base: ${base_lcofe:.0f}/t',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_groups.keys(), fontsize=11)
    ax.set_xlabel("Levelized Cost of Iron ($/t Fe)", fontsize=12)
    ax.set_title("Sensitivity Analysis: One-at-a-Time Parameter Variation",
                 fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation
    ax.annotate("← Lower cost          Higher cost →",
                xy=(0.5, -0.15), xycoords='axes fraction',
                ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    path = OUTPUT_DIR / "sensitivity_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {path}")

    return sens


def plot_opex_breakdown(opex_data):
    """Pie chart of OPEX breakdown."""
    categories = {
        "Electricity": opex_data["Electricity ($/yr)"],
        "Ore feedstock": opex_data["Iron ore feedstock ($/yr)"],
        "Electrolyte": opex_data["Electrolyte makeup ($/yr)"],
        "Anode repl.": opex_data["Anode replacement ($/yr)"],
        "Labor": opex_data["Labor ($/yr)"],
        "Maintenance": opex_data["Maintenance ($/yr)"],
        "Other": (opex_data["Water ($/yr)"] + opex_data["Insurance ($/yr)"]
                  + opex_data["Overhead ($/yr)"]),
    }

    # Filter out very small slices
    total = sum(categories.values())
    filtered = {k: v for k, v in categories.items() if v / total > 0.02}
    if len(filtered) < len(categories):
        other_sum = sum(v for k, v in categories.items() if v / total <= 0.02)
        filtered["Other"] = filtered.get("Other", 0) + other_sum

    labels = list(filtered.keys())
    values = list(filtered.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=0.8,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    for t in texts:
        t.set_fontsize(10)

    ax.set_title(f"Annual OPEX Breakdown\nTotal: ${total/1e6:.1f}M/yr",
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    path = OUTPUT_DIR / "opex_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def save_report(base_result, comparison, sensitivity):
    """Save full results as JSON."""
    report = {
        "title": "Aqueous Electrowinning Techno-Economic Analysis",
        "date": "2026-07-29",
        "base_case": {
            "cell_voltage_V": base_result["cell"].V_cell,
            "current_density_mA_cm2": base_result["params"].current_density_mA_cm2,
            "current_efficiency": base_result["params"].current_efficiency,
            "n_stacks": base_result["n_stacks"],
            "annual_production_t": base_result["annual_prod"],
            "total_capex_M$": base_result["capex"]["Total CAPEX (M$)"],
            "annual_opex_M$": base_result["opex"]["Total OPEX (M$/yr)"],
            "LCOFe_$_per_t": base_result["lcofe"]["LCOFe ($/t Fe)"],
            "specific_energy_kWh_per_t": base_result["opex"]["Specific energy (kWh/t Fe)"],
            "electricity_cost_$_per_t": base_result["opex"]["Electricity cost ($/t Fe)"],
        },
        "capex_breakdown": {
            k: v for k, v in base_result["capex"].items()
            if "$" in k
        },
        "opex_breakdown": {
            k: v for k, v in base_result["opex"].items()
            if "$" in k or "energy" in k.lower() or "production" in k.lower()
        },
        "route_comparison": comparison,
        "sensitivity_analysis": {
            k: v for k, v in sensitivity.items()
        },
    }

    path = REPORT_DIR / "technoeconomic_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✅ Saved: {path}")


def main():
    # 1. Run base case
    base = run_base_case()

    # 2. Generate plots
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    plot_energy_breakdown(base["params"], base["cell"])
    comparison = plot_cost_comparison(base["lcofe"]["LCOFe ($/t Fe)"])
    plot_opex_breakdown(base["opex"])
    sensitivity = plot_sensitivity(base)

    # 3. Print route comparison
    print("\n" + "=" * 70)
    print("ROUTE COMPARISON (with $50/tCO₂ carbon price)")
    print("=" * 70)
    for route, data in comparison.items():
        print(f"\n  {route}:")
        for k, v in data.items():
            print(f"    {k}: {v}")

    # 4. Save JSON report
    print("\n" + "=" * 70)
    print("SAVING REPORT")
    print("=" * 70)
    save_report(base, comparison, sensitivity)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
