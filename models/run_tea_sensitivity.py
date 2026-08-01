#!/usr/bin/env python3
"""
Techno-economic sensitivity — does the winning corner exist?

Monte Carlo sweep over the cost model to find the Pareto front of
(electricity price, current density, FE) that hits cost parity with DRI-H2.

Sweeps:
  j:         50–500 mA/cm²
  V_cell:    1.5–4.0 V
  FE:        0.50–0.95
  electricity: $0.01–0.10/kWh
  cell_cost:  $0–2000/m²  (electrode+membrane+hardware)

DRI-H2 benchmark: $450–1000/t Fe (mid $600/t)

Kill criteria:
  - If optimal requires j > 800 mA/cm² at 90% FE with $0.02/kWh -> no path
  - If optimal is at ~250 mA/cm² and ~78% FE -> plausible ground

Deliverables:
  docs/figures/tea_pareto_front.png
  docs/figures/tea_sensitivity_tornado.png
  docs/figures/tea_dri_h2_comparison.png
  experiments/data/tea_sensitivity_report.json
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.electrochemistry import (
    specific_energy_kWh_per_t,
    current_density_to_production,
)

# ─── Output paths ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "data"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── DRI-H2 Benchmark ─────────────────────────────────────────────────
DRI_H2_LOW = 450.0    # $/t Fe
DRI_H2_MID = 600.0
DRI_H2_HIGH = 1000.0


def compute_lcofe(
    j_mA_cm2: float,
    V_cell: float,
    FE: float,
    elec_price: float,     # $/kWh
    cell_cost_per_m2: float,  # $/m² total (electrode+membrane+hardware)
    n_cells: int = 100,
    electrode_area_m2: float = 1.0,
    n_stacks: int = 10,
    operating_hours: float = 8000.0,
    plant_lifetime_yr: int = 25,
    discount_rate: float = 0.08,
) -> float:
    """Compute levelized cost of iron ($/t Fe) for given operating point."""
    # Production per cell (kg/hr)
    prod_per_cell = current_density_to_production(j_mA_cm2, electrode_area_m2, FE)
    # Annual production (t/yr) for full plant
    annual_prod_t = prod_per_cell * n_cells * n_stacks * operating_hours / 1000.0
    if annual_prod_t <= 0:
        return float('inf')

    # Specific energy (kWh/t)
    spec_energy = specific_energy_kWh_per_t(V_cell, FE)

    # CAPEX
    area_total = electrode_area_m2 * n_cells * 2 * n_stacks  # both faces
    capex_stacks = cell_cost_per_m2 * area_total * 1.15  # assembly factor
    # BOP: rectifiers + electrolyte + leaching
    total_current_A = j_mA_cm2 * 10.0 * electrode_area_m2
    total_power_kW = total_current_A * V_cell * n_cells / 1000.0
    rectifier = 120.0 * total_power_kW * n_stacks
    elec_system = 5000.0 * 0.02 * area_total
    leaching = 50.0 * annual_prod_t
    bop = rectifier + elec_system + leaching
    direct = capex_stacks + bop
    capex_total = direct * (1 + 0.25 + 0.15) * (1 + 0.15)  # infra+eng+contingency

    # OPEX (annual)
    elec_cost = spec_energy * elec_price * annual_prod_t
    ore_cost = 40.0 * annual_prod_t
    water_cost = 2.0 * annual_prod_t
    electrolyte_cost = 15.0 * annual_prod_t
    anode_cost = 30.0 * area_total / 2  # per m² cathode area (half total)
    variable = elec_cost + ore_cost + water_cost + electrolyte_cost + anode_cost
    fixed = 0.03 * capex_total + 0.01 * capex_total + 2_000_000 + 0.10 * variable
    annual_opex = variable + fixed

    # LCOFe
    r = discount_rate
    n = plant_lifetime_yr
    crf = (r * (1 + r)**n) / ((1 + r)**n - 1)
    lcofe = (crf * capex_total + annual_opex) / annual_prod_t

    return lcofe


def run_sensitivity_sweep(n_samples: int = 100_000, seed: int = 42):
    """Run Monte Carlo sweep over parameter space."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    # Draw samples
    j = rng.uniform(50, 500, n_samples)         # mA/cm²
    V = rng.uniform(1.5, 4.0, n_samples)         # V
    FE = rng.uniform(0.50, 0.95, n_samples)
    elec = rng.uniform(0.01, 0.10, n_samples)    # $/kWh
    cell_cost = rng.uniform(0, 2000, n_samples)   # $/m²

    lcofe = np.empty(n_samples)
    for i in range(n_samples):
        lcofe[i] = compute_lcofe(j[i], V[i], FE[i], elec[i], cell_cost[i])

    elapsed = time.perf_counter() - t0
    print(f"  Sweep: {n_samples:,} samples in {elapsed:.1f}s")

    return {
        "j": j, "V": V, "FE": FE,
        "elec": elec, "cell_cost": cell_cost,
        "lcofe": lcofe,
    }


def find_pareto_front(data: dict, target: float = DRI_H2_MID) -> dict:
    """
    Find Pareto front: among samples that beat target, identify the
    efficient frontier of (j, FE) where LCOFe <= target with minimal j.
    """
    lcofe = data["lcofe"]
    j = data["j"]
    FE = data["FE"]
    V = data["V"]
    elec = data["elec"]
    cell_cost = data["cell_cost"]

    # Subset that beats DRI-H2
    mask = lcofe <= target
    n_beat = mask.sum()
    pct = n_beat / len(lcofe) * 100

    result = {
        "target_lcofe": target,
        "n_samples": len(lcofe),
        "n_below_target": int(n_beat),
        "pct_below_target": round(pct, 2),
    }

    if n_beat == 0:
        result["pareto_front"] = []
        result["message"] = "No samples beat DRI-H2 target. Winning corner is EMPTY."
        return result

    # Extract winning samples
    w_j = j[mask]
    w_FE = FE[mask]
    w_V = V[mask]
    w_elec = elec[mask]
    w_cost = cell_cost[mask]
    w_lcofe = lcofe[mask]

    # Discretize (j, FE) into bins and find best LCOFe per bin
    j_bins = np.linspace(50, 500, 20)
    fe_bins = np.linspace(0.50, 0.95, 20)

    pareto_points = []
    for ji in range(len(j_bins) - 1):
        for fi in range(len(fe_bins) - 1):
            in_bin = (
                (w_j >= j_bins[ji]) & (w_j < j_bins[ji + 1]) &
                (w_FE >= fe_bins[fi]) & (w_FE < fe_bins[fi + 1])
            )
            if in_bin.sum() == 0:
                continue
            best_idx = np.argmin(w_lcofe[in_bin])
            bin_mask = np.where(in_bin)[0][best_idx]
            pareto_points.append({
                "j_mA_cm2": round(float(w_j[bin_mask]), 1),
                "FE": round(float(w_FE[bin_mask]), 3),
                "V_cell": round(float(w_V[bin_mask]), 2),
                "elec_price": round(float(w_elec[bin_mask]), 3),
                "cell_cost_per_m2": round(float(w_cost[bin_mask]), 0),
                "LCOFe": round(float(w_lcofe[bin_mask]), 0),
            })

    # Sort by LCOFe
    pareto_points.sort(key=lambda p: p["LCOFe"])

    # Statistics of winners
    result["winning_stats"] = {
        "j_median": round(float(np.median(w_j)), 1),
        "j_p5": round(float(np.percentile(w_j, 5)), 1),
        "j_p95": round(float(np.percentile(w_j, 95)), 1),
        "FE_median": round(float(np.median(w_FE)), 3),
        "FE_p5": round(float(np.percentile(w_FE, 5)), 3),
        "FE_p95": round(float(np.percentile(w_FE, 95)), 3),
        "V_median": round(float(np.median(w_V)), 2),
        "elec_median": round(float(np.median(w_elec)), 3),
        "cell_cost_median": round(float(np.median(w_cost)), 0),
        "LCOFe_median": round(float(np.median(w_lcofe)), 0),
        "LCOFe_min": round(float(np.min(w_lcofe)), 0),
    }
    result["pareto_front"] = pareto_points[:30]  # top 30

    return result


def evaluate_kill_criteria(data: dict, pareto: dict) -> dict:
    """Evaluate kill criteria from the task spec."""
    criteria = {}

    # Criterion 1: Does optimal require j > 800 at 90% FE with $0.02/kWh?
    # We don't have j > 800 in our sweep (max 500), so check if the best
    # high-FE low-elec region is reachable
    lcofe = data["lcofe"]
    j = data["j"]
    FE = data["FE"]
    elec = data["elec"]

    # High-FE + low-elec subset
    opt_mask = (FE >= 0.85) & (elec <= 0.03)
    if opt_mask.sum() > 0:
        best_opt_lcofe = float(np.min(lcofe[opt_mask]))
        best_opt_j = float(j[opt_mask][np.argmin(lcofe[opt_mask])])
        criteria["high_FE_low_elec"] = {
            "condition": "FE >= 0.85 and elec <= $0.03/kWh",
            "best_LCOFe": round(best_opt_lcofe, 0),
            "best_j": round(best_opt_j, 1),
            "beats_DRI_H2": best_opt_lcofe < DRI_H2_MID,
        }
    else:
        criteria["high_FE_low_elec"] = {"condition": "no samples in region"}

    # Criterion 2: Is ~250 mA/cm² and ~78% FE plausible?
    mid_mask = (j >= 200) & (j <= 300) & (FE >= 0.73) & (FE <= 0.83)
    if mid_mask.sum() > 0:
        mid_lcofe = lcofe[mid_mask]
        criteria["mid_j_mid_FE"] = {
            "condition": "j 200-300 mA/cm², FE 0.73-0.83",
            "n_samples": int(mid_mask.sum()),
            "median_LCOFe": round(float(np.median(mid_lcofe)), 0),
            "min_LCOFe": round(float(np.min(mid_lcofe)), 0),
            "pct_below_DRI_H2": round(float((mid_lcofe < DRI_H2_MID).sum() / len(mid_lcofe) * 100), 1),
        }
    else:
        criteria["mid_j_mid_FE"] = {"condition": "no samples in region"}

    # Overall verdict
    pct_below = pareto["pct_below_target"]
    if pct_below == 0:
        verdict = "KILL: No winning corner exists. Zero samples beat DRI-H2 mid estimate."
    elif pct_below < 5:
        verdict = f"MARGINAL: Only {pct_below}% of parameter space beats DRI-H2. Requires optimistic assumptions."
    elif pct_below < 25:
        verdict = f"VIABLE: {pct_below}% of space beats DRI-H2. Winning corner exists but is narrow."
    else:
        verdict = f"STRONG: {pct_below}% of space beats DRI-H2. Wide winning corner."

    criteria["verdict"] = verdict
    criteria["pct_below_DRI_H2_mid"] = pct_below

    return criteria


def plot_pareto_front(data: dict, pareto: dict):
    """Scatter plot: j vs FE, colored by LCOFe, with DRI-H2 contour."""
    fig, ax = plt.subplots(figsize=(10, 7))

    lcofe = data["lcofe"]
    j = data["j"]
    FE = data["FE"]

    # Subsample for plotting (too many points)
    idx = np.random.default_rng(0).choice(len(lcofe), min(20_000, len(lcofe)), replace=False)

    sc = ax.scatter(
        j[idx], FE[idx],
        c=lcofe[idx],
        cmap="RdYlGn_r",
        s=2, alpha=0.4,
        vmin=200, vmax=2000,
    )
    cbar = plt.colorbar(sc, ax=ax, label="LCOFe ($/t Fe)")

    # Draw DRI-H2 contour (approximate: find samples near target)
    target = DRI_H2_MID
    near = np.abs(lcofe - target) < 50
    if near.sum() > 5:
        ax.scatter(j[near], FE[near], c='blue', s=3, alpha=0.3,
                   label=f'DRI-H2 ≈ ${target:.0f}/t')

    # Mark Pareto points
    if pareto["pareto_front"]:
        pj = [p["j_mA_cm2"] for p in pareto["pareto_front"][:20]]
        pfe = [p["FE"] for p in pareto["pareto_front"][:20]]
        ax.scatter(pj, pfe, c='red', s=40, marker='*', zorder=5,
                   label='Pareto winners')

    ax.set_xlabel("Current density (mA/cm²)", fontsize=12)
    ax.set_ylabel("Faradaic efficiency", fontsize=12)
    ax.set_title(
        "TEA Pareto Front: LCOFe vs (j, FE)\n"
        f"DRI-H2 target: ${DRI_H2_LOW}–${DRI_H2_HIGH}/t (mid ${DRI_H2_MID}/t)",
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=10, loc='lower left')
    ax.set_xlim(40, 510)
    ax.set_ylim(0.48, 0.97)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "tea_pareto_front.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return str(path)


def plot_tornado(pareto: dict, data: dict):
    """Tornado chart: sensitivity of LCOFe to each parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))

    lcofe = data["lcofe"]
    params = {
        "Cell voltage (V)": data["V"],
        "Current density (mA/cm²)": data["j"],
        "Faradaic efficiency": data["FE"],
        "Electricity price ($/kWh)": data["elec"],
        "Cell cost ($/m²)": data["cell_cost"],
    }

    # Compute rank correlation for each param
    corrs = {}
    for name, vals in params.items():
        valid = ~np.isnan(lcofe)
        if valid.sum() < 10:
            continue
        c = abs(float(np.corrcoef(lcofe[valid], vals[valid])[0, 1]))
        corrs[name] = c

    # Sort by correlation
    sorted_params = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_params]
    values = [p[1] for p in sorted_params]

    colors = ['#E91E63', '#FF5722', '#FF9800', '#FFC107', '#4CAF50']
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, values, color=colors[:len(names)], alpha=0.8,
                   edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("|Pearson correlation| with LCOFe", fontsize=12)
    ax.set_title("Sensitivity Tornado: Parameter Influence on LCOFe",
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "tea_sensitivity_tornado.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return str(path)


def plot_dri_h2_comparison(data: dict, pareto: dict):
    """Histogram of LCOFe distribution vs DRI-H2 benchmark range."""
    fig, ax = plt.subplots(figsize=(10, 6))

    lcofe = data["lcofe"]
    # Clip for display
    lcofe_clip = np.clip(lcofe, 0, 3000)

    ax.hist(lcofe_clip, bins=100, color='#2196F3', alpha=0.7,
            edgecolor='white', linewidth=0.5, density=True)

    # DRI-H2 bands
    ax.axvspan(DRI_H2_LOW, DRI_H2_HIGH, alpha=0.15, color='green',
               label=f'DRI-H2 range (${DRI_H2_LOW}–${DRI_H2_HIGH}/t)')
    ax.axvline(DRI_H2_MID, color='green', linewidth=2, linestyle='--',
               label=f'DRI-H2 mid (${DRI_H2_MID}/t)')
    ax.axvline(DRI_H2_LOW, color='green', linewidth=1, linestyle=':')

    # Median and percentiles
    med = float(np.median(lcofe))
    p5 = float(np.percentile(lcofe, 5))
    p95 = float(np.percentile(lcofe, 95))
    ax.axvline(med, color='red', linewidth=2, label=f'Median: ${med:.0f}/t')
    ax.axvline(p5, color='orange', linewidth=1, linestyle=':',
               label=f'P5: ${p5:.0f}/t')
    ax.axvline(p95, color='orange', linewidth=1, linestyle=':',
               label=f'P95: ${p95:.0f}/t')

    pct_below = (lcofe < DRI_H2_MID).sum() / len(lcofe) * 100
    ax.text(0.98, 0.95,
            f'{pct_below:.1f}% below DRI-H2 mid\n'
            f'n = {len(lcofe):,} samples',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel("LCOFe ($/t Fe)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Aqueous Electrowinning LCOFe Distribution vs DRI-H2 Benchmark",
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, 3000)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "tea_dri_h2_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return str(path)


def save_report(data: dict, pareto: dict, kill_criteria: dict):
    """Save full report as JSON."""
    # Output stats
    lcofe = data["lcofe"]
    stats = {
        "mean": round(float(np.mean(lcofe)), 0),
        "std": round(float(np.std(lcofe)), 0),
        "p5": round(float(np.percentile(lcofe, 5)), 0),
        "p25": round(float(np.percentile(lcofe, 25)), 0),
        "p50": round(float(np.percentile(lcofe, 50)), 0),
        "p75": round(float(np.percentile(lcofe, 75)), 0),
        "p95": round(float(np.percentile(lcofe, 95)), 0),
        "min": round(float(np.min(lcofe)), 0),
        "max": round(float(np.max(lcofe)), 0),
    }

    report = {
        "title": "Techno-Economic Sensitivity: Does the Winning Corner Exist?",
        "date": "2026-07-29",
        "sweep_parameters": {
            "j_mA_cm2": {"low": 50, "high": 500},
            "V_cell_V": {"low": 1.5, "high": 4.0},
            "FE": {"low": 0.50, "high": 0.95},
            "electricity_$/kWh": {"low": 0.01, "high": 0.10},
            "cell_cost_$/m2": {"low": 0, "high": 2000},
        },
        "n_samples": len(data["lcofe"]),
        "lcofe_distribution": stats,
        "dri_h2_benchmark": {
            "low": DRI_H2_LOW,
            "mid": DRI_H2_MID,
            "high": DRI_H2_HIGH,
        },
        "pareto_analysis": pareto,
        "kill_criteria": kill_criteria,
        "figures": [
            "docs/figures/tea_pareto_front.png",
            "docs/figures/tea_sensitivity_tornado.png",
            "docs/figures/tea_dri_h2_comparison.png",
        ],
    }

    path = REPORT_DIR / "tea_sensitivity_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✅ Saved: {path}")
    return str(path)


def main():
    print("=" * 70)
    print("TECHNO-ECONOMIC SENSITIVITY — DOES THE WINNING CORNER EXIST?")
    print("=" * 70)

    # 1. Run sweep
    print("\n▸ Running Monte Carlo sweep (100k samples)...")
    data = run_sensitivity_sweep(n_samples=100_000)

    # 2. Find Pareto front
    print("\n▸ Finding Pareto front vs DRI-H2...")
    pareto = find_pareto_front(data)
    ws = pareto.get("winning_stats", {})
    print(f"  Samples below DRI-H2 mid (${DRI_H2_MID}/t): {pareto['pct_below_target']}%")
    if ws:
        print(f"  Winners: j median={ws['j_median']} mA/cm², FE median={ws['FE_median']}")
        print(f"  Winner V_cell median={ws['V_median']}V, elec=${ws['elec_median']}/kWh")
        print(f"  Winner LCOFe range: ${ws['LCOFe_min']}–${ws['LCOFe_median']} /t")

    # 3. Kill criteria
    print("\n▸ Evaluating kill criteria...")
    kill = evaluate_kill_criteria(data, pareto)
    for k, v in kill.items():
        if k == "verdict":
            continue
        if isinstance(v, dict):
            print(f"  {k}: {v}")
    print(f"\n  ╔{'═'*60}╗")
    print(f"  ║  VERDICT: {kill['verdict']:<49s}║")
    print(f"  ╚{'═'*60}╝")

    # 4. Generate figures
    print("\n▸ Generating figures...")
    plot_pareto_front(data, pareto)
    plot_tornado(pareto, data)
    plot_dri_h2_comparison(data, pareto)

    # 5. Save report
    print("\n▸ Saving report...")
    save_report(data, pareto, kill)

    print("\n✅ TEA sensitivity analysis complete!")
    return pareto, kill


if __name__ == "__main__":
    main()
