#!/usr/bin/env python3
"""
Run Life Cycle Assessment for aqueous electrowinning steel.

Generates:
  - lca_carbon_footprint.png   (GWP breakdown waterfall)
  - lca_comparison.png         (vs BOF/EAF/DRI bar chart)
  - lca_electricity_sensitivity.png
  - lca_water_footprint.png

Usage:
    python -m models.run_lca
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.lca import (
    ElectricityMix,
    LCAResult,
    compute_lca,
    compare_routes,
    sensitivity_to_electricity,
    breakeven_renewable_fraction,
    REFERENCE_ROUTES,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "data"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Typical specific energy from the electrochemistry model (kWh/kg Fe)
# At 100 mA/cm², 2.5 V cell, 90% CE → ~2.6 kWh/kg (refined below)
SPECIFIC_ENERGY_KWH_PER_KG = 2.6


def run_base_case():
    """Compute LCA for the base case (100% renewable electricity)."""
    print("=" * 70)
    print("AQUEOUS ELECTROWINNING — LIFE CYCLE ASSESSMENT")
    print("=" * 70)

    mix = ElectricityMix(renewable=1.0)
    result = compute_lca(SPECIFIC_ENERGY_KWH_PER_KG, electricity_mix=mix)

    print("\n  Electricity mix: 100% renewable")
    print(f"  Specific energy: {result.specific_energy_kWh_per_kg:.2f} kWh/kg")
    print(f"\n  GWP:             {result.gwp_kgCO2eq:.4f} kg CO₂-eq / kg steel")
    print(f"  Acidification:   {result.acidification_kgSO2eq:.6f} kg SO₂-eq / kg")
    print(f"  Eutrophication:  {result.eutrophication_kgPO4eq:.8f} kg PO₄-eq / kg")
    print(f"  Water:           {result.water_L:.2f} L / kg steel")
    print(f"  Land use:        {result.land_use_m2:.6f} m² / kg steel")
    print("\n  GWP breakdown:")
    print(f"    Electricity:     {result.electricity_gwp:.4f} kg CO₂-eq")
    print(f"    Chemicals/ore:   {result.chemicals_gwp:.4f} kg CO₂-eq")
    print(f"    Heat treatment:  {result.heat_treatment_gwp:.4f} kg CO₂-eq")
    print(f"    Waste treatment: {result.waste_gwp:.4f} kg CO₂-eq")

    return result


def plot_carbon_footprint(result: LCAResult):
    """Waterfall chart of GWP breakdown."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Electricity", "Chemicals/Ore", "Heat Treatment", "Waste", "Total"]
    values = [result.electricity_gwp, result.chemicals_gwp,
              result.heat_treatment_gwp, result.waste_gwp, result.gwp_kgCO2eq]

    colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bottoms = [0, values[0], values[0] + values[1],
               values[0] + values[1] + values[2], 0]

    ax.bar(categories[:4], values[:4], bottom=bottoms[:4], color=colours[:4],
           edgecolor="white", linewidth=0.5, width=0.6)
    ax.bar(categories[4:], values[4:], bottom=[0], color=colours[4:],
           edgecolor="white", linewidth=0.5, width=0.6, alpha=0.85)

    for i, v in enumerate(values):
        y = bottoms[i] + v / 2 if i < 4 else v / 2
        ax.text(i, y, f"{v:.4f}", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("kg CO₂-eq / kg steel")
    ax.set_title("GWP Breakdown — Aqueous Electrowinning (100% Renewable)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lca_carbon_footprint.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'lca_carbon_footprint.png'}")


def plot_comparison(result: LCAResult):
    """Bar chart: electrowinning vs reference routes."""
    table = compare_routes(result)

    fig, ax = plt.subplots(figsize=(10, 5))

    routes = list(REFERENCE_ROUTES.keys()) + ["Aq. Electrowinning"]
    lows = [REFERENCE_ROUTES[r]["gwp_low"] for r in REFERENCE_ROUTES] + [result.gwp_kgCO2eq]
    mids = [REFERENCE_ROUTES[r]["gwp_mid"] for r in REFERENCE_ROUTES] + [result.gwp_kgCO2eq]
    highs = [REFERENCE_ROUTES[r]["gwp_high"] for r in REFERENCE_ROUTES] + [result.gwp_kgCO2eq]

    colours = ["#78909C", "#78909C", "#78909C", "#78909C", "#4CAF50"]
    x = np.arange(len(routes))
    bars = ax.bar(x, mids, color=colours, edgecolor="white", width=0.6)

    # Error bars for ranges
    err_low = [m - l for m, l in zip(mids, lows)]
    err_high = [h - m for h, m in zip(highs, mids)]
    ax.errorbar(x, mids, yerr=[err_low, err_high], fmt="none",
                ecolor="#333", capsize=5, linewidth=1.5)

    for i, (lo, hi) in enumerate(zip(lows, highs)):
        ax.text(i, hi + 0.05, f"{lo:.2f}–{hi:.2f}", ha="center", fontsize=8)

    # Target line
    ax.axhline(y=0.5, color="#F44336", linestyle="--", linewidth=1, alpha=0.7, label="Target: <0.5 kg CO₂/kg")

    ax.set_xticks(x)
    ax.set_xticklabels(routes, rotation=15, ha="right")
    ax.set_ylabel("kg CO₂-eq / kg steel")
    ax.set_title("Carbon Footprint — Aqueous Electrowinning vs Conventional Routes")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lca_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'lca_comparison.png'}")


def plot_electricity_sensitivity():
    """Sensitivity of GWP to electricity mix."""
    mixes = {
        "100% Coal": ElectricityMix(coal=1.0, renewable=0.0),
        "75% Coal / 25% RE": ElectricityMix(coal=0.75, renewable=0.25),
        "50/50": ElectricityMix(coal=0.5, renewable=0.5),
        "Grid Average": ElectricityMix(grid_avg=1.0, renewable=0.0),
        "25% Coal / 75% RE": ElectricityMix(coal=0.25, renewable=0.75),
        "100% Renewable": ElectricityMix(renewable=1.0),
    }

    results = sensitivity_to_electricity(SPECIFIC_ENERGY_KWH_PER_KG, mixes)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names = list(results.keys())
    gwps = [results[n].gwp_kgCO2eq for n in names]
    waters = [results[n].water_L for n in names]

    colours = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(names)))

    ax = axes[0]
    bars = ax.barh(names, gwps, color=colours, edgecolor="white")
    ax.axvline(x=0.5, color="#F44336", linestyle="--", linewidth=1, alpha=0.7, label="Target")
    ax.set_xlabel("kg CO₂-eq / kg steel")
    ax.set_title("GWP Sensitivity to Electricity Mix")
    ax.legend()
    for bar, val in zip(bars, gwps):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    elec_gwps = [results[n].electricity_gwp for n in names]
    chem_gwps = [results[n].chemicals_gwp for n in names]
    ht_gwps = [results[n].heat_treatment_gwp for n in names]
    waste_gwps = [results[n].waste_gwp for n in names]

    y = np.arange(len(names))
    ax.barh(y, elec_gwps, color="#2196F3", label="Electricity", edgecolor="white")
    ax.barh(y, chem_gwps, left=elec_gwps, color="#4CAF50", label="Chemicals", edgecolor="white")
    left2 = [a + b for a, b in zip(elec_gwps, chem_gwps)]
    ax.barh(y, ht_gwps, left=left2, color="#FF9800", label="Heat Treatment", edgecolor="white")
    left3 = [a + b for a, b in zip(left2, ht_gwps)]
    ax.barh(y, waste_gwps, left=left3, color="#9C27B0", label="Waste", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("kg CO₂-eq / kg steel")
    ax.set_title("GWP Stacked Breakdown by Mix")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lca_electricity_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'lca_electricity_sensitivity.png'}")


def plot_water_footprint():
    """Water consumption comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    routes = list(REFERENCE_ROUTES.keys())
    waters_ref = [REFERENCE_ROUTES[r]["water_L_per_kg"] for r in routes]

    # Electrowinning at different mixes
    renew_result = compute_lca(SPECIFIC_ENERGY_KWH_PER_KG, electricity_mix=ElectricityMix(renewable=1.0))
    coal_result = compute_lca(SPECIFIC_ENERGY_KWH_PER_KG, electricity_mix=ElectricityMix(coal=1.0, renewable=0.0))

    all_routes = routes + ["Aq. EW (renewable)", "Aq. EW (coal)"]
    all_waters = waters_ref + [renew_result.water_L, coal_result.water_L]
    colours = ["#78909C"] * len(routes) + ["#4CAF50", "#FF5722"]

    x = np.arange(len(all_routes))
    ax.bar(x, all_waters, color=colours, edgecolor="white", width=0.6)
    for i, w in enumerate(all_waters):
        ax.text(i, w + 0.5, f"{w:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(all_routes, rotation=20, ha="right")
    ax.set_ylabel("L water / kg steel")
    ax.set_title("Water Footprint — Electrowinning vs Conventional Routes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lca_water_footprint.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'lca_water_footprint.png'}")


def run_breakeven_analysis():
    """Find breakeven renewable fraction for <0.5 kg CO₂/kg target."""
    target = 0.5
    frac = breakeven_renewable_fraction(SPECIFIC_ENERGY_KWH_PER_KG, target_co2_kg_per_kg=target)
    print("\n  Breakeven analysis:")
    print(f"    Target:    <{target} kg CO₂-eq / kg steel")
    print(f"    Required:  ≥{frac * 100:.1f}% renewable electricity in coal/renewable mix")
    return frac


def main():
    result = run_base_case()

    print("\n" + "─" * 70)
    print("  Generating charts...")
    print("─" * 70)
    plot_carbon_footprint(result)
    plot_comparison(result)
    plot_electricity_sensitivity()
    plot_water_footprint()

    frac = run_breakeven_analysis()

    # Comparison table
    table = compare_routes(result)
    print("\n" + "─" * 70)
    print("  Route comparison")
    print("─" * 70)
    table_dict = table.to_dict()
    for route, data in table_dict.items():
        print(f"\n  {route}:")
        for k, v in data.items():
            print(f"    {k}: {v}")

    # Save report
    report = {
        "base_case": result.to_dict(),
        "breakeven_renewable_fraction": frac,
        "comparison": table_dict,
    }
    report_path = REPORT_DIR / "lca_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
