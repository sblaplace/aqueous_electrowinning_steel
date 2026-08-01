"""
Driver for cold-rolling + recrystallization-anneal thermomechanical screening
of electrodeposited iron/steel foil.

Generates:
* docs/figures/thermomech_recrystallization.png   (JMAK + grain size vs time, multi-T)
* docs/figures/thermomech_temperature_sweep.png   (grain size / RX fraction / YS / energy vs T)
* docs/figures/thermomech_reduction_sweep.png     (RX grain size / YS vs cold reduction)
* docs/figures/thermomech_deposit_vs_annealed.png (deposit vs annealed property contrast)
* experiments/data/thermomechanical_report.json
* experiments/data/synthetic_thermomechanical.csv

Usage:
    python -m models.run_thermomechanical
    python -m models.run_thermomechanical --reduction 0.6 --anneal-temp 750 \\
        --anneal-time 90 --grain 1.5 --ni 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from models.thermomechanical import (
    ThermomechanicalModel,
    ThermomechanicalParams,
    RollingSchedule,
)

FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"


def _ns(
    reduction=None, passes=None, anneal_temp=None, anneal_time=None,
    grain=None, ni=None, carbon=None, ce=None,
):
    """Build a simple namespace from direct keyword overrides (run_all path)."""
    class _NS:
        pass
    args = _NS()
    args.reduction = reduction if reduction is not None else 0.50
    args.passes = passes if passes is not None else 2
    args.anneal_temp = anneal_temp if anneal_temp is not None else 700.0
    args.anneal_time = anneal_time if anneal_time is not None else 60.0
    args.grain = grain if grain is not None else 1.0
    args.ni = ni if ni is not None else 0.0
    args.carbon = carbon if carbon is not None else 0.0
    args.ce = ce if ce is not None else 95.0
    return args


def main(argv=None, reduction=None, passes=None, anneal_temp=None,
         anneal_time=None, grain=None, ni=None, carbon=None, ce=None):
    if argv is not None:
        parser = argparse.ArgumentParser(
            description="Thermomechanical (roll + recrystallize) screening driver")
        parser.add_argument("--reduction", type=float, default=0.50,
                            help="Total fractional cold reduction (0-0.95)")
        parser.add_argument("--passes", type=int, default=2,
                            help="Number of rolling passes")
        parser.add_argument("--anneal-temp", type=float, default=700.0,
                            help="Recrystallization anneal T (°C)")
        parser.add_argument("--anneal-time", type=float, default=60.0,
                            help="Anneal hold time (min)")
        parser.add_argument("--grain", type=float, default=1.0,
                            help="As-deposited grain size (µm)")
        parser.add_argument("--ni", type=float, default=0.0, help="Ni wt%")
        parser.add_argument("--carbon", type=float, default=0.0, help="C wt%")
        parser.add_argument("--ce", type=float, default=95.0,
                            help="Deposition current efficiency %")
        args = parser.parse_args(argv)
    else:
        args = _ns(reduction, passes, anneal_temp, anneal_time, grain,
                   ni, carbon, ce)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("THERMOMECHANICAL PROCESSING OF ELECTRODEPOSITED FOIL — SCREENING")
    print("=" * 72)
    print(f"Cold reduction={args.reduction:.0%} ({args.passes} pass), "
          f"anneal {args.anneal_temp:.0f}°C x {args.anneal_time:.0f} min, "
          f"deposit grain={args.grain:.2f} µm, Ni={args.ni}%, C={args.carbon}%")

    params = ThermomechanicalParams(
        deposit_grain_size_um=args.grain,
        rolling=RollingSchedule(total_reduction=args.reduction, n_passes=args.passes),
        anneal_temperature_C=args.anneal_temp,
        anneal_time_min=args.anneal_time,
        ni_wt_percent=args.ni,
        carbon_wt_percent=args.carbon,
        current_efficiency_percent=args.ce,
    )
    model = ThermomechanicalModel(params)
    result = model.predict()

    print("\nAs-deposited vs annealed:")
    print(f"  As-deposited : YS={result.deposit_yield_MPa:6.1f} MPa, "
          f"UTS={result.deposit_uts_MPa:6.1f} MPa, "
          f"Elong={result.deposit_elongation_pct:4.1f}%, "
          f"{result.deposit_grade}")
    print(f"  Annealed     : YS={result.annealed_yield_MPa:6.1f} MPa, "
          f"UTS={result.annealed_uts_MPa:6.1f} MPa, "
          f"Elong={result.annealed_elongation_pct:4.1f}%, "
          f"{result.annealed_grade}")
    print(f"  Grain: {result.deposit_grain_um:.2f} -> "
          f"{result.recrystallized_grain_um:.1f} (recryst.) -> "
          f"{result.final_grain_um:.1f} µm (after growth)")
    print(f"  Fraction recrystallized: {result.fraction_recrystallized:.3f} "
          f"(time to 99%: {result.t_full_rx_min:.1f} min)")
    print(f"  Anneal energy: {result.annealing_energy_kWh_per_kg:.3f} kWh/kg")
    print(f"  Flags: {result.flags if result.flags else 'none'}")

    # CSV time series
    df = pd.DataFrame({
        "time_s": result.time_s,
        "fraction_recrystallized": result.fraction_recrystallized_series,
        "grain_size_um": result.grain_size_series_um,
    })
    csv_path = DATA_DIR / "synthetic_thermomechanical.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved time series: {csv_path}")

    # ── Figure 1: recrystallization kinetics vs time at several T ─────────
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    temps = [550, 600, 650, 700, 750]
    for T in temps:
        mT = ThermomechanicalModel(
            ThermomechanicalParams(
                **{**params.__dict__, "anneal_temperature_C": T,
                   "anneal_time_min": 120.0})
        )
        rT = mT.predict()
        tm = rT.time_s / 60.0
        axes[0].plot(tm, rT.fraction_recrystallized_series, label=f"{T}°C")
        axes[1].plot(tm, rT.grain_size_series_um, label=f"{T}°C")
    axes[0].axhline(0.99, color="grey", ls=":", lw=1)
    axes[0].set(title="Recrystallization kinetics (JMAK, ε=0.69)",
                xlabel="Time (min)", ylabel="Fraction recrystallized",
                xlim=(0, 120), ylim=(0, 1.05))
    axes[1].set(title="Grain-size evolution during anneal",
                xlabel="Time (min)", ylabel="Grain size (µm)", xlim=(0, 120))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Thermomechanical recrystallization of electrodeposited foil",
                 fontweight="bold")
    fig.tight_layout()
    fig1 = FIG_DIR / "thermomech_recrystallization.png"
    fig.savefig(fig1, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig1}")

    # ── Figure 2: temperature sweep ───────────────────────────────────────
    ts = model.sweep_temperature(time_min=args.anneal_time)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot(ts["T_C"], ts["D_final_um"], "-o", color="#1874b4")
    axes[0].set(title="Final grain size vs T", xlabel="Anneal T (°C)",
                ylabel="Grain size (µm)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(ts["T_C"], ts["frac_rx"], "-o", color="#d95f02")
    axes[1].axhline(0.99, color="grey", ls=":", lw=1)
    axes[1].set(title="Fraction recrystallized vs T", xlabel="Anneal T (°C)",
                ylabel="Fraction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)
    axes[2].plot(ts["T_C"], ts["yield_MPa"], "-o", color="#4daf4a",
                 label="Yield")
    axes[2].set(title="Annealed yield strength vs T", xlabel="Anneal T (°C)",
                ylabel="Yield (MPa)")
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"Anneal temperature screen — {args.anneal_time:.0f} min hold",
                 fontweight="bold")
    fig.tight_layout()
    fig2 = FIG_DIR / "thermomech_temperature_sweep.png"
    fig.savefig(fig2, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig2}")

    # ── Figure 3: cold reduction sweep ────────────────────────────────────
    rs = model.sweep_reduction(time_min=args.anneal_time)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(rs["reduction"] * 100, rs["D_rx_um"], "-o", color="#1874b4",
                 label="Recrystallized grain")
    axes[0].plot(rs["reduction"] * 100, rs["D_final_um"], "--s",
                 color="#d95f02", label="Final (after growth)")
    axes[0].set(title="RX grain size vs cold reduction", xlabel="Reduction (%)",
                ylabel="Grain size (µm)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(rs["reduction"] * 100, rs["yield_MPa"], "-o", color="#4daf4a")
    axes[1].set(title="Annealed yield vs cold reduction",
                xlabel="Reduction (%)", ylabel="Yield (MPa)")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Cold-rolling reduction screen — anneal {args.anneal_temp:.0f}°C",
                 fontweight="bold")
    fig.tight_layout()
    fig3 = FIG_DIR / "thermomech_reduction_sweep.png"
    fig.savefig(fig3, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig3}")

    # ── Figure 4: deposit vs annealed contrast ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["Yield", "UTS"]
    dep = [result.deposit_yield_MPa, result.deposit_uts_MPa]
    ann = [result.annealed_yield_MPa, result.annealed_uts_MPa]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, dep, w, label="As-deposited", color="#e41a1c")
    b2 = ax.bar(x + w / 2, ann, w, label="Recrystallized-annealed",
                color="#4daf4a")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set(xticks=x, xticklabels=labels, ylabel="Strength (MPa)",
           title="Deposit vs thermomechanically-processed sheet")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.02,
            (f"Grain {result.deposit_grain_um:.1f}→{result.final_grain_um:.1f} µm\n"
             f"Elong {result.deposit_elongation_pct:.0f}→"
             f"{result.annealed_elongation_pct:.0f}%"),
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom")
    fig.tight_layout()
    fig4 = FIG_DIR / "thermomech_deposit_vs_annealed.png"
    fig.savefig(fig4, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig4}")

    # ── Report JSON ───────────────────────────────────────────────────────
    report = {
        "title": "Thermomechanical processing (cold roll + recrystallization) screening",
        "date": datetime.now().isoformat(),
        "params": {
            "deposit_grain_size_um": args.grain,
            "total_reduction_frac": args.reduction,
            "n_passes": args.passes,
            "anneal_temperature_C": args.anneal_temp,
            "anneal_time_min": args.anneal_time,
            "ni_wt_percent": args.ni,
            "carbon_wt_percent": args.carbon,
            "current_efficiency_percent": args.ce,
        },
        "summary": result.summary(),
        "figures": [str(fig1), str(fig2), str(fig3), str(fig4)],
        "csv": str(csv_path),
        "model_notes": [
            "JMAK recrystallization X=1-exp(-(k t)^n), k=k0·ε^m·exp(-Q/RT); Q≈210 kJ/mol, n=2 (screening means)",
            "Recrystallized grain size D_rx = D_rx_ref·(D0/D0_ref)^0.35·(ε_ref/ε)^0.55·(T/T_ref)^0.6",
            "Normal grain growth D²-D_rx² = K_gg·t·exp(-Q_gg/RT), Q_gg≈240 kJ/mol",
            "Annealed strength reuses mechanical_properties Hall-Petch + solid solution + dispersion machinery",
            "Energy is heating only (Cp=449 J/kg/K, 60% furnace eff.); no atmosphere or soak credit",
            "Does NOT model texture development, abnormal grain growth, recrystallized C redistribution, or hydrogen",
            "Screening constants must be calibrated with real rolling trials, EBSD, and hardness traverses",
        ],
    }
    report_path = DATA_DIR / "thermomechanical_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  ✅ Saved report: {report_path}")
    print("\n✅ Thermomechanical driver complete!")
    return report


if __name__ == "__main__":
    main()
