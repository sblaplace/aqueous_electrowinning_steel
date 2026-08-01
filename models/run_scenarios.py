#!/usr/bin/env python3
"""
Scenario comparison for aqueous electrowinning techno-economic analysis.

Evaluates multiple electrolyte/operating regimes and generates comparison charts.

Usage:
    python -m models.run_scenarios
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    ElectrolyzerParams,
    CAPEXModel,
    OPEXModel,
    LevelizedCost,
    BENCHMARK_COSTS,
    specific_energy_kWh_per_t,
)
from models.scenarios import ALL_SCENARIOS, Scenario

# ─── Output Directories ───────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "data"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def scenario_to_params(scenario: Scenario) -> ElectrolyzerParams:
    """Convert a Scenario to ElectrolyzerParams."""
    return ElectrolyzerParams(
        current_density_mA_cm2=scenario.current_density_mA_cm2,
        current_efficiency=scenario.current_efficiency,
        cell_voltage=scenario.V_cell,
        temperature_C=scenario.temperature_C,
        electrode_area_m2=1.0,
        n_cells=100,
        electrolyte_type=scenario.electrolyte_type,
    )


def evaluate_scenario(scenario: Scenario, n_stacks: int = 10):
    """Run full techno-economic evaluation for a scenario."""
    params = scenario_to_params(scenario)
    capex_model = CAPEXModel()
    opex_model = OPEXModel(
        electricity_price_kWh=scenario.electricity_price_kWh,
        electrolyte_makeup_per_t_Fe=scenario.electrolyte_makeup_per_t_Fe,
        ore_cost_per_t_Fe=scenario.ore_cost_per_t_Fe,
    )
    lc_model = LevelizedCost()

    capex = capex_model.estimate(params, n_stacks)
    capex["Total CAPEX ($)"] *= scenario.capex_modifier  # apply scenario modifier
    capex["Total CAPEX (M$)"] = round(capex["Total CAPEX ($)"] / 1e6, 2)

    opex = opex_model.estimate(params, capex["Total CAPEX ($)"], n_stacks)
    annual_prod = capex["Annual capacity (t/yr)"]
    lcofe = lc_model.calculate(capex["Total CAPEX ($)"], opex["Total OPEX ($/yr)"], annual_prod)

    specific_energy = specific_energy_kWh_per_t(
        scenario.V_cell, scenario.current_efficiency
    )

    return {
        "scenario": scenario,
        "params": params,
        "capex": capex,
        "opex": opex,
        "lcofe": lcofe,
        "annual_prod": annual_prod,
        "specific_energy_kWh_t": specific_energy,
        "electricity_cost_per_t": specific_energy * scenario.electricity_price_kWh,
    }


def print_scenario_table(results: list):
    """Print comparison table of all scenarios."""
    print("\n" + "=" * 100)
    print("SCENARIO COMPARISON — AQUEOUS ELECTROWINNING OF IRON")
    print("=" * 100)

    # Header
    print(f"\n{'Scenario':<22} {'j (mA/cm²)':>10} {'CE (%)':>7} {'V_cell':>7} "
          f"{'Energy':>9} {'CAPEX':>8} {'OPEX':>8} {'LCOFe':>8}")
    print(f"{'':22} {'':>10} {'':>7} {'(V)':>7} "
          f"{'(kWh/t)':>9} {'(M$)':>8} {'(M$/yr)':>8} {'($/t)':>8}")
    print("─" * 100)

    for r in results:
        s = r["scenario"]
        print(f"{s.name:<22} "
              f"{s.current_density_mA_cm2:>10.0f} "
              f"{s.current_efficiency*100:>6.1f}% "
              f"{s.V_cell:>7.2f} "
              f"{r['specific_energy_kWh_t']:>9.0f} "
              f"{r['capex']['Total CAPEX (M$)']:>8.1f} "
              f"{r['opex']['Total OPEX (M$/yr)']:>8.1f} "
              f"{r['lcofe']['LCOFe ($/t Fe)']:>8.0f}")

    print("─" * 100)

    # Benchmark reference
    print(f"{'BF-BOF (benchmark)':<22} {'—':>10} {'—':>7} {'—':>7} "
          f"{'—':>9} {'—':>8} {'—':>8} {'450':>8}")
    print(f"{'H₂-DRI + EAF':<22} {'—':>10} {'—':>7} {'—':>7} "
          f"{'—':>9} {'—':>8} {'—':>8} {'600':>8}")


def plot_scenario_comparison(results: list):
    """Multi-panel comparison chart for all scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r["scenario"].name for r in results]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    # Panel 1: LCOFe comparison
    ax = axes[0, 0]
    lcofes = [r["lcofe"]["LCOFe ($/t Fe)"] for r in results]
    bars = ax.bar(names, lcofes, color=colors, edgecolor='white', linewidth=1.5)
    ax.axhline(y=450, color='gray', linestyle='--', linewidth=1.5, label='BF-BOF ($450/t)')
    ax.axhline(y=600, color='gray', linestyle=':', linewidth=1.5, label='H₂-DRI ($600/t)')
    for bar, val in zip(bars, lcofes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'${val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("LCOFe ($/t Fe)", fontsize=11)
    ax.set_title("Levelized Cost of Iron", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:.0f}'))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=15)

    # Panel 2: Specific energy consumption
    ax = axes[0, 1]
    energies = [r["specific_energy_kWh_t"] for r in results]
    bars = ax.bar(names, energies, color=colors, edgecolor='white', linewidth=1.5)
    ax.axhline(y=1500, color='red', linestyle='--', linewidth=1.5, label='Target: <1,500 kWh/t')
    for bar, val in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Energy (kWh/t Fe)", fontsize=11)
    ax.set_title("Specific Energy Consumption", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=15)

    # Panel 3: Cell voltage decomposition (grouped bar)
    ax = axes[1, 0]
    x = np.arange(len(names))
    width = 0.2

    thermo = [abs(r["scenario"].E_anode_eq - r["scenario"].E_cathode_eq) for r in results]
    eta_c = [r["scenario"].eta_cathode for r in results]
    eta_a = [r["scenario"].eta_anode for r in results]
    ir = [r["scenario"].ir_drop for r in results]

    ax.bar(x - 1.5*width, thermo, width, label='Thermodynamic', color='#1565C0')
    ax.bar(x - 0.5*width, eta_c, width, label='η cathode', color='#E65100')
    ax.bar(x + 0.5*width, eta_a, width, label='η anode', color='#F57C00')
    ax.bar(x + 1.5*width, ir, width, label='iR drop', color='#FFB74D')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.set_ylabel("Voltage (V)", fontsize=11)
    ax.set_title("Cell Voltage Decomposition", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 4: CAPEX vs OPEX share
    ax = axes[1, 1]
    capex_shares = [r["lcofe"]["CAPEX share (%)"] for r in results]
    opex_shares = [r["lcofe"]["OPEX share (%)"] for r in results]

    ax.bar(names, capex_shares, label='CAPEX share', color='#1565C0', edgecolor='white')
    ax.bar(names, opex_shares, bottom=capex_shares, label='OPEX share',
           color='#4CAF50', edgecolor='white')

    for i, (cs, os_) in enumerate(zip(capex_shares, opex_shares)):
        ax.text(i, cs/2, f'{cs:.0f}%', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.text(i, cs + os_/2, f'{os_:.0f}%', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    ax.set_ylabel("Cost share (%)", fontsize=11)
    ax.set_title("CAPEX vs. OPEX Share of Levelized Cost", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=15)

    fig.suptitle("Aqueous Electrowinning: Scenario Comparison",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "scenario_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✅ Saved: {path}")


def plot_scenario_radar(results: list):
    """Radar chart comparing scenarios across multiple dimensions."""
    categories = [
        "Current\nEfficiency",
        "Current\nDensity",
        "Energy\nEfficiency",
        "Low\nCost",
        "Low\nCAPEX",
        "Scalability",
    ]
    N = len(categories)

    # Normalize metrics to 0–1 range (higher is better for all)
    max_j = max(r["scenario"].current_density_mA_cm2 for r in results)
    max_energy = max(r["specific_energy_kWh_t"] for r in results)
    max_lcofe = max(r["lcofe"]["LCOFe ($/t Fe)"] for r in results)
    max_capex = max(r["capex"]["Total CAPEX ($)"] for r in results)

    # Scalability scores (qualitative, 0-1)
    scalability = {
        "Conservative (Base)": 0.6,
        "Optimized Alkaline": 0.75,
        "AWARE Acidic": 0.85,
        "Future Target": 0.95,
    }

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, r in enumerate(results):
        s = r["scenario"]
        values = [
            s.current_efficiency,                                  # CE (already 0-1)
            s.current_density_mA_cm2 / max_j,                     # normalized j
            1.0 - (r["specific_energy_kWh_t"] / max_energy) * 0.5,  # energy efficiency (inverted, scaled)
            1.0 - r["lcofe"]["LCOFe ($/t Fe)"] / max_lcofe,       # cost (inverted)
            1.0 - r["capex"]["Total CAPEX ($)"] / max_capex,      # CAPEX (inverted)
            scalability.get(s.name, 0.5),                         # qualitative
        ]
        values += values[:1]  # close

        ax.plot(angles, values, 'o-', linewidth=2, label=s.name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color='gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Multi-Dimensional Scenario Comparison\n(higher = better)",
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    path = OUTPUT_DIR / "scenario_radar.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_energy_vs_cost(results: list):
    """Scatter plot: energy consumption vs. LCOFe with bubble size = current density."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, r in enumerate(results):
        s = r["scenario"]
        energy = r["specific_energy_kWh_t"]
        lcofe = r["lcofe"]["LCOFe ($/t Fe)"]
        j = s.current_density_mA_cm2

        # Bubble size proportional to current density
        size = j * 3

        ax.scatter(energy, lcofe, s=size, c=colors[i], alpha=0.7,
                   edgecolors='black', linewidth=1.5, zorder=5)

        # Label
        ax.annotate(
            f"{s.name}\n(j={j:.0f} mA/cm²)",
            (energy, lcofe),
            textcoords="offset points",
            xytext=(15, 10),
            fontsize=9,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
        )

    # Benchmark regions
    ax.axhline(y=450, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1500, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(1550, 420, 'Target: <1,500 kWh/t', color='red', fontsize=8, alpha=0.7)
    ax.text(2800, 460, 'BF-BOF: $450/t', color='gray', fontsize=8, alpha=0.7)

    # Shade the "competitive" zone
    ax.fill_between([0, 1500], 0, 450, alpha=0.05, color='green')
    ax.text(750, 200, 'Competitive\nzone', color='green', fontsize=10,
            alpha=0.5, ha='center', fontweight='bold')

    ax.set_xlabel("Specific Energy Consumption (kWh/t Fe)", fontsize=12)
    ax.set_ylabel("Levelized Cost of Iron ($/t Fe)", fontsize=12)
    ax.set_title("Energy vs. Cost: Scenario Trade-offs\n(bubble size ∝ current density)",
                 fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:.0f}'))
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "energy_vs_cost.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def save_scenario_report(results: list):
    """Save comprehensive scenario comparison as JSON."""
    report = {
        "title": "Aqueous Electrowinning Scenario Comparison",
        "date": "2026-07-29",
        "plant_config": {
            "n_stacks": 10,
            "electrode_area_m2": 1.0,
            "n_cells_per_stack": 100,
            "operating_hours_yr": 8000,
        },
        "scenarios": {},
    }

    for r in results:
        s = r["scenario"]
        report["scenarios"][s.name] = {
            "description": s.description,
            "electrolyte": s.electrolyte_composition,
            "operating_conditions": {
                "current_density_mA_cm2": s.current_density_mA_cm2,
                "current_efficiency": s.current_efficiency,
                "temperature_C": s.temperature_C,
                "cell_voltage_V": round(s.V_cell, 3),
            },
            "cell_voltage_breakdown": {
                "thermodynamic_V": round(abs(s.E_anode_eq - s.E_cathode_eq), 3),
                "eta_cathode_V": s.eta_cathode,
                "eta_anode_V": s.eta_anode,
                "ir_drop_V": s.ir_drop,
            },
            "results": {
                "specific_energy_kWh_t": round(r["specific_energy_kWh_t"], 0),
                "annual_production_t": round(r["annual_prod"], 0),
                "total_capex_M$": r["capex"]["Total CAPEX (M$)"],
                "annual_opex_M$": r["opex"]["Total OPEX (M$/yr)"],
                "LCOFe_$_t": r["lcofe"]["LCOFe ($/t Fe)"],
                "electricity_cost_$_t": round(r["electricity_cost_per_t"], 2),
                "capex_share_pct": r["lcofe"]["CAPEX share (%)"],
                "opex_share_pct": r["lcofe"]["OPEX share (%)"],
            },
            "references": s.references,
        }

    # Add benchmark comparison
    report["benchmarks"] = BENCHMARK_COSTS

    path = REPORT_DIR / "scenario_comparison_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✅ Saved: {path}")


def main():
    print("=" * 100)
    print("SCENARIO COMPARISON — AQUEOUS ELECTROWINNING OF IRON")
    print("=" * 100)

    # Evaluate all scenarios
    results = [evaluate_scenario(s) for s in ALL_SCENARIOS]

    # Print table
    print_scenario_table(results)

    # Print individual scenario details
    for r in results:
        s = r["scenario"]
        print(f"\n{'─'*60}")
        print(f"  📋 {s.name}")
        print(f"  {s.description}")
        print(f"  Electrolyte: {s.electrolyte_composition}")
        print(f"  Anode: {s.anode_type}")
        print(f"  References: {s.references}")
        print("  ─────────────────────────────────────")
        print(f"  V_cell = {abs(s.E_anode_eq - s.E_cathode_eq):.3f} (thermo) "
              f"+ {s.eta_cathode:.2f} (η_c) + {s.eta_anode:.2f} (η_a) "
              f"+ {s.ir_drop:.2f} (iR) = {s.V_cell:.3f} V")
        print(f"  Energy: {r['specific_energy_kWh_t']:.0f} kWh/t Fe "
              f"({r['electricity_cost_per_t']:.0f} $/t at ${s.electricity_price_kWh}/kWh)")
        print(f"  LCOFe: ${r['lcofe']['LCOFe ($/t Fe)']:.0f}/t Fe")

    # Generate plots
    print("\n" + "=" * 100)
    print("GENERATING FIGURES")
    print("=" * 100)

    plot_scenario_comparison(results)
    plot_scenario_radar(results)
    plot_energy_vs_cost(results)

    # Save JSON
    print("\n" + "=" * 100)
    print("SAVING REPORT")
    print("=" * 100)
    save_scenario_report(results)

    print("\n✅ Scenario analysis complete!")


if __name__ == "__main__":
    main()
