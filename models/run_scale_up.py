#!/usr/bin/env python3
"""
Scale-up analysis driver: current distribution, mass transport, thermal.

Generates:
    docs/figures/scale_up_current_distribution.png
    docs/figures/scale_up_mass_transport.png
    docs/figures/scale_up_thermal.png
    docs/figures/scale_up_geometry_optimization.png
    experiments/data/scale_up_report.json

Usage:
    python -m models.run_scale_up
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.scale_up import (
    boundary_layer_thickness,
    mass_transport_scaling,
    optimize_geometry,
    primary_current_distribution,
    scale_up_analysis,
    thermal_management,
    uniformity_index,
    wagner_number,
)
from models.transport import D_FE

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


# ─── Figure 1: Current distribution across scales ────────────────────
def plot_current_distribution():
    """Primary current distribution at lab vs pilot scale."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (label, L, j_avg, color) in zip(axes, [
        ("Lab (10 cm²)", 0.1, 1000.0, "#1565c0"),
        ("Pilot (1000 cm²)", 0.316, 10000.0, "#c62828"),
    ]):
        x = np.linspace(-L / 2 * 0.95, L / 2 * 0.95, 501)
        j = primary_current_distribution(x, L, j_avg, kappa=10.0)
        uni = uniformity_index(j)
        Wa = wagner_number(kappa=10.0, j_ref=j_avg, L=L)

        ax.plot(x * 1000, j / j_avg, color=color, lw=2)
        ax.axhline(1.0, color="#616161", ls=":", lw=1, label="j_avg")
        ax.fill_between(x * 1000, 0.9, 1.1, alpha=0.1, color="#4caf50", label="±10% band")
        ax.set_xlabel("Position across cathode (mm)")
        ax.set_ylabel("j / j_avg")
        ax.set_title(f"{label}\nWa = {Wa:.3f}, uniformity = {uni:.1%}")
        ax.set_ylim(0.8, 3.2)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Primary Current Distribution: Edge Effects at Scale", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "scale_up_current_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")
    return Wa, uni


# ─── Figure 2: Mass transport scaling ────────────────────────────────
def plot_mass_transport():
    """Boundary layer growth and transport-limited current along cathode."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: boundary layer thickness vs electrode length
    Ls = np.logspace(-2, 0, 100)  # 1 cm to 1 m
    deltas = [boundary_layer_thickness(L, D=D_FE, v=0.1) * 1e3 for L in Ls]
    axes[0].semilogx(Ls * 100, deltas, color="#2e7d32", lw=2)
    axes[0].set_xlabel("Cathode length (cm)")
    axes[0].set_ylabel("Boundary layer δ (mm)")
    axes[0].set_title("Boundary layer growth (v = 0.1 m/s)")
    axes[0].grid(alpha=0.3, which="both")

    # Right: j_lim vs position for different flow velocities
    for v, color, label in [
        (0.05, "#c62828", "v = 0.05 m/s"),
        (0.1, "#ef6c00", "v = 0.1 m/s"),
        (0.2, "#1565c0", "v = 0.2 m/s"),
        (0.5, "#2e7d32", "v = 0.5 m/s"),
    ]:
        L = 0.3  # 30 cm cathode
        x = np.linspace(0.0, L, 201)
        j_local = np.full_like(x, 1000.0)
        mt = mass_transport_scaling(L, j_local, D=D_FE, v=v)
        axes[1].plot(mt.x * 100, mt.j_lim, color=color, lw=2, label=label)

    axes[1].axhline(1000.0, color="#616161", ls="--", lw=1.3, label="j = 1000 A/m²")
    axes[1].set_xlabel("Position along cathode (cm)")
    axes[1].set_ylabel("j_lim (A/m²)")
    axes[1].set_title("Transport-limited current (1 M Fe²⁺)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Mass Transport Scaling: Boundary Layer Growth", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "scale_up_mass_transport.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


# ─── Figure 3: Thermal management ────────────────────────────────────
def plot_thermal():
    """Temperature rise and heat balance vs current density."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: temperature vs current density for different areas
    j_range = np.linspace(100, 5000, 100)
    for area, color, label in [
        (0.001, "#1565c0", "10 cm² (lab)"),
        (0.01, "#ef6c00", "100 cm²"),
        (0.1, "#c62828", "1000 cm² (pilot)"),
    ]:
        Ts = []
        for j in j_range:
            t = thermal_management(float(j), area, gap_m=0.01, kappa=10.0)
            Ts.append(t.T_cell_C)
        axes[0].plot(j_range, Ts, color=color, lw=2, label=label)

    axes[0].axhline(80.0, color="#d32f2f", ls="--", lw=1.3, label="Boiling risk (80°C)")
    axes[0].set_xlabel("Average current density (A/m²)")
    axes[0].set_ylabel("Cell temperature (°C)")
    axes[0].set_title("Temperature vs current density")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Right: heat balance components for pilot scale
    j_range2 = np.linspace(100, 10000, 200)
    Q_gen, Q_conv, Q_rad = [], [], []
    for j in j_range2:
        t = thermal_management(float(j), 0.1, gap_m=0.01, kappa=10.0)
        Q_gen.append(t.Q_gen_W)
        Q_conv.append(t.Q_conv_W)
        Q_rad.append(t.Q_rad_W)

    axes[1].plot(j_range2, Q_gen, color="#c62828", lw=2, label="Q_gen (Joule)")
    axes[1].plot(j_range2, Q_conv, color="#1565c0", lw=2, label="Q_conv")
    axes[1].plot(j_range2, Q_rad, color="#2e7d32", lw=2, label="Q_rad")
    axes[1].set_xlabel("Average current density (A/m²)")
    axes[1].set_ylabel("Power (W)")
    axes[1].set_title("Heat balance (1000 cm² pilot cell)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Thermal Management: Scale Effects", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "scale_up_thermal.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


# ─── Figure 4: Geometry optimization ─────────────────────────────────
def plot_geometry_optimization():
    """Optimal geometry across a range of total currents."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    currents = np.linspace(100, 5000, 50)
    gaps, uniformities, energies = [], [], []
    for I in currents:
        result = optimize_geometry(float(I), area_m2=0.1, kappa=10.0, V_cell=2.0)
        gaps.append(result.gap_m * 1000)  # mm
        uniformities.append(result.uniformity)
        energies.append(result.energy_kWh_per_kg)

    # Left: optimal gap and uniformity vs total current
    ax1 = axes[0]
    ax1.plot(currents, gaps, "o-", color="#1565c0", lw=2, ms=3, label="Optimal gap (mm)")
    ax1.set_xlabel("Total current (A)")
    ax1.set_ylabel("Optimal gap (mm)", color="#1565c0")
    ax1.tick_params(axis="y", labelcolor="#1565c0")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(currents, uniformities, "s--", color="#c62828", lw=1.5, ms=3, label="Uniformity")
    ax2.axhline(0.9, color="#616161", ls=":", lw=1, label="90% threshold")
    ax2.set_ylabel("Uniformity index", color="#c62828")
    ax2.tick_params(axis="y", labelcolor="#c62828")
    ax2.set_ylim(0.5, 1.05)
    ax1.set_title("Optimal gap and uniformity")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # Right: specific energy vs total current
    axes[1].plot(currents, energies, "o-", color="#2e7d32", lw=2, ms=3)
    axes[1].set_xlabel("Total current (A)")
    axes[1].set_ylabel("Specific energy (kWh/kg)")
    axes[1].set_title("Energy efficiency vs scale")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Geometry Optimization for Pilot-Scale Cells", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "scale_up_geometry_optimization.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")
    return gaps, uniformities, energies, currents


# ─── Report ───────────────────────────────────────────────────────────
def build_report(geo_data) -> dict:
    """Build the JSON report from all analyses."""
    gaps, uniformities, energies, currents = geo_data

    # Compare lab vs pilot
    lab = scale_up_analysis(area_m2=0.001, total_current_A=10.0)
    pilot = scale_up_analysis(area_m2=0.1, total_current_A=1000.0)

    report = {
        "model": {
            "description": "Scale-up model: primary/secondary current distribution, "
                           "mass transport scaling, thermal management, geometry optimization",
            "components": [
                "Primary current distribution (Wagner number, conformal mapping)",
                "Secondary current distribution (1-D Poisson + Butler-Volmer)",
                "Mass transport scaling (boundary-layer growth, Levich-type)",
                "Thermal management (Joule heating, convective + radiative cooling)",
                "Geometry optimization (energy minimisation, uniformity constraint)",
            ],
        },
        "wagner_comparison": {
            "lab_10cm2": {
                "area_m2": 0.001,
                "wagner_number": round(lab.wagner, 4),
                "primary_uniformity": round(lab.primary_uniformity, 4),
                "T_cell_C": round(lab.thermal.T_cell_C, 1),
                "j_lim_min_A_m2": round(lab.mass_transport.j_lim_min, 0),
            },
            "pilot_1000cm2": {
                "area_m2": 0.1,
                "wagner_number": round(pilot.wagner, 4),
                "primary_uniformity": round(pilot.primary_uniformity, 4),
                "T_cell_C": round(pilot.thermal.T_cell_C, 1),
                "j_lim_min_A_m2": round(pilot.mass_transport.j_lim_min, 0),
            },
        },
        "thermal_sensitivity": [
            {
                "j_A_m2": int(j),
                "T_cell_C": round(
                    thermal_management(float(j), 0.1, 0.01, 10.0).T_cell_C, 1
                ),
                "boiling_risk": thermal_management(float(j), 0.1, 0.01, 10.0).boiling_risk,
            }
            for j in [500, 1000, 2000, 3000, 5000]
        ],
        "geometry_optimization": [
            {
                "total_current_A": int(c),
                "optimal_gap_mm": round(g, 1),
                "uniformity": round(u, 3),
                "energy_kWh_per_kg": round(e, 2),
            }
            for c, g, u, e in zip(currents[::10], gaps[::10], uniformities[::10], energies[::10])
        ],
        "uniformity_index": {
            "description": "Fraction of cathode area within ±10% of j_avg",
            "lab_10cm2": round(lab.primary_uniformity, 4),
            "pilot_1000cm2": round(pilot.primary_uniformity, 4),
        },
    }
    return report


def main():
    print("=" * 70)
    print("SCALE-UP MODEL — CURRENT DISTRIBUTION, MASS TRANSPORT, THERMAL")
    print("=" * 70)

    print("\nGenerating figures...\n")

    print("  [1/4] Current distribution...")
    plot_current_distribution()

    print("  [2/4] Mass transport scaling...")
    plot_mass_transport()

    print("  [3/4] Thermal management...")
    plot_thermal()

    print("  [4/4] Geometry optimization...")
    geo_data = plot_geometry_optimization()

    # Print summary tables
    print("\n" + "=" * 70)
    print("SCALE-UP COMPARISON")
    print("=" * 70)

    for label, area, current in [("Lab", 0.001, 10.0), ("Pilot", 0.1, 1000.0)]:
        result = scale_up_analysis(area_m2=area, total_current_A=current)
        print(f"\n  {label} ({area*1e4:.0f} cm², {current:.0f} A):")
        print(f"    Wagner number:      {result.wagner:.4f}")
        print(f"    Primary uniformity: {result.primary_uniformity:.1%}")
        print(f"    j_lim (min):        {result.mass_transport.j_lim_min:.0f} A/m²")
        print(f"    T_cell:             {result.thermal.T_cell_C:.1f} °C")
        print(f"    Boiling risk:       {'⚠️  YES' if result.thermal.boiling_risk else '✅ No'}")
        print(f"    Q_gen:              {result.thermal.Q_gen_W:.1f} W")

    print("\n  Geometry optimization (1000 cm², 1000 A):")
    opt = optimize_geometry(1000.0, area_m2=0.1, kappa=10.0)
    print(f"    Optimal gap:     {opt.gap_m*1000:.1f} mm")
    print(f"    Uniformity:      {opt.uniformity:.1%}")
    print(f"    Energy:          {opt.energy_kWh_per_kg:.2f} kWh/kg")
    print(f"    Feasible:        {'✅' if opt.feasible else '⚠️  No'}")

    # Build and save report
    report = build_report(geo_data)
    out = DATA_DIR / "scale_up_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  ✅ Saved: {out}")
    print("\n✅ Scale-up analysis complete!")


if __name__ == "__main__":
    main()
