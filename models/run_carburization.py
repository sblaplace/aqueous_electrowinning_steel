"""
Driver for post-deposition gaseous carburization screening.

Generates:
* docs/figures/carburization_profiles.png
* docs/figures/carburization_case_depth.png
* docs/figures/carburization_hardness.png
* docs/figures/carburization_energy.png
* experiments/data/carburization_report.json
* experiments/data/synthetic_carburization.csv (time series) + profiles CSV

Usage:
    python -m models.run_carburization
    python -m models.run_carburization --temperature 920 --surface-c 1.2 --duration 6 --thickness 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.carburization import (
    CarburizationParams,
    CarburizationModel,
    carbon_diffusivity_m2_s,
    estimate_carburizing_time_for_case_depth,
)

FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"


def main(argv=None, temperature=None, surface_c=None, initial_c=None, thickness=None, duration=None, dt=None):
    """
    Main driver — accepts either CLI args (argv) or direct keyword overrides (for run_all).
    When called programmatically, pass temperature etc directly to avoid sys.argv clash.
    """
    if temperature is None and argv is None:
        # CLI mode
        parser = argparse.ArgumentParser(description="Carburization screening driver")
        parser.add_argument("--temperature", type=float, default=900.0, help="Carburizing T (°C)")
        parser.add_argument("--surface-c", type=float, default=1.10, help="Surface C wt% (potential)")
        parser.add_argument("--initial-c", type=float, default=0.02, help="Initial C wt%")
        parser.add_argument("--thickness", type=float, default=1000.0, help="Sheet thickness µm")
        parser.add_argument("--duration", type=float, default=4.0, help="Sim time hr")
        parser.add_argument("--dt", type=float, default=0.1, help="dt hr")
        parsed = parser.parse_args(argv)
        args = parsed
    else:
        # Programmatic mode — build simple namespace from kwargs
        class _NS: pass
        args = _NS()
        args.temperature = temperature if temperature is not None else 900.0
        args.surface_c = surface_c if surface_c is not None else 1.10
        args.initial_c = initial_c if initial_c is not None else 0.02
        args.thickness = thickness if thickness is not None else 1000.0
        args.duration = duration if duration is not None else 4.0
        args.dt = dt if dt is not None else 0.1
        # If argv supplied as list, parse it but allow empty
        if argv is not None:
            parser = argparse.ArgumentParser(description="Carburization screening driver")
            parser.add_argument("--temperature", type=float, default=args.temperature)
            parser.add_argument("--surface-c", type=float, default=args.surface_c)
            parser.add_argument("--initial-c", type=float, default=args.initial_c)
            parser.add_argument("--thickness", type=float, default=args.thickness)
            parser.add_argument("--duration", type=float, default=args.duration)
            parser.add_argument("--dt", type=float, default=args.dt)
            parsed = parser.parse_args(argv)
            args = parsed

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("POST-DEPOSITION GASEOUS CARBURIZATION — SCREENING")
    print("="*72)
    print(f"T={args.temperature}°C, Cs={args.surface_c} wt%, C0={args.initial_c} wt%, L={args.thickness} µm")

    params = CarburizationParams(
        temperature_C=args.temperature,
        surface_carbon_wt_percent=args.surface_c,
        initial_carbon_wt_percent=args.initial_c,
        sheet_thickness_um=args.thickness,
    )
    model = CarburizationModel(params)

    D, phase = carbon_diffusivity_m2_s(args.temperature)
    print(f"Diffusivity: D={D:.3e} m²/s, phase={phase} (D0/Q auto)")

    # Quick time estimate for 0.5 mm case
    try:
        t_est = estimate_carburizing_time_for_case_depth(
            target_case_depth_um=500.0,
            temperature_C=args.temperature,
            surface_c_wt=args.surface_c,
            threshold_c_wt=0.35,
            initial_c_wt=args.initial_c,
            D_m2_s=D,
        )
        print(f"Estimated time for 0.5 mm case @0.35%C: {t_est:.2f} hr (erfc inversion, semi-infinite)")
    except Exception as e:
        print(f"Time estimate failed: {e}")

    result = model.simulate(duration_hr=args.duration, dt_hr=args.dt, n_x=300, save_profiles_every_hr=1.0)

    print("\nTime series (last point):")
    print(result.summary())

    # Save time-series CSV
    df_ts = pd.DataFrame({
        "time_hr": result.time_hr,
        "case_depth_035_um": result.effective_case_depth_035_um,
        "case_depth_050_um": result.effective_case_depth_050_um,
        "carbon_uptake_g_m2": result.carbon_uptake_g_m2,
        "surface_hv": result.surface_hv,
        "core_c_wt": result.core_c_wt,
        "flags": [";".join(f) for f in result.flags],
    })
    ts_path = DATA_DIR / "synthetic_carburization.csv"
    df_ts.to_csv(ts_path, index=False)
    print(f"  ✅ Saved time series: {ts_path}")

    # Composite strength at final case depth
    comp = model.composite_strength_estimate(
        case_depth_um=result.effective_case_depth_035_um[-1],
        core_yield_MPa=300.0,
    )
    print("\nCase-core composite (final):")
    for k, v in comp.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

    # Energy estimate
    en = model.energy_estimate_kWh_per_kg(mass_kg=1.0)
    print("\nEnergy (screening):")
    for k, v in en.items():
        print(f"  {k}: {v}")

    # Figures
    # 1) Profiles
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for prof in result.profiles:
        ax.plot(prof.x_um, prof.c_wt_percent, label=f"{prof.time_hr:.1f} hr")
    ax.set(xlabel="Depth from surface (µm)", ylabel="C wt%", title=f"C profile — T={args.temperature}°C, Cs={args.surface_c}%",
           xlim=(0, min(args.thickness/2, 1200)), ylim=(0, args.surface_c*1.1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    # add threshold lines
    ax.axhline(0.35, color="#777777", ls=":", label="0.35% threshold")
    ax.axhline(0.50, color="#333333", ls="--", label="0.50% threshold")

    ax = axes[1]
    for prof in result.profiles:
        ax.plot(prof.x_um, prof.hv_predicted, label=f"{prof.time_hr:.1f} hr")
    ax.set(xlabel="Depth (µm)", ylabel="HV (kgf/mm²)", title="Hardness profile (as-quenched screening)",
           xlim=(0, min(args.thickness/2, 1200)), ylim=(100, 900))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(f"Carburization profiles — {args.thickness:.0f} µm sheet", fontweight="bold")
    fig.tight_layout()
    prof_fig = FIG_DIR / "carburization_profiles.png"
    fig.savefig(prof_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {prof_fig}")

    # 2) Case depth vs time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.time_hr, result.effective_case_depth_035_um, label="0.35% case", color="#1874b4", marker="o", ms=3)
    ax.plot(result.time_hr, result.effective_case_depth_050_um, label="0.50% case", color="#d95f02", marker="s", ms=3)
    ax.set(xlabel="Time (hr)", ylabel="Case depth (µm)", title=f"Case depth vs time — T={args.temperature}°C")
    ax.grid(alpha=0.25)
    ax.legend()
    case_fig = FIG_DIR / "carburization_case_depth.png"
    fig.tight_layout()
    fig.savefig(case_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {case_fig}")

    # 3) Surface HV and core C vs time
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(result.time_hr, result.surface_hv, color="#e41a1c", marker="o", ms=3)
    ax.set(title="Surface hardness vs time", xlabel="Time (hr)", ylabel="HV (kgf/mm²)")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(result.time_hr, result.core_c_wt, color="#4daf4a", marker="s", ms=3)
    ax.set(title="Core C vs time (midplane)", xlabel="Time (hr)", ylabel="Core C wt%")
    ax.grid(alpha=0.25)
    ax.axhline(0.30, color="#777777", ls=":", label="core high-C flag")
    ax.legend(fontsize=8)

    fig.suptitle("Carburization kinetics", fontweight="bold")
    fig.tight_layout()
    hv_fig = FIG_DIR / "carburization_hardness.png"
    fig.savefig(hv_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {hv_fig}")

    # 4) Energy / cost conceptual
    fig, ax = plt.subplots(figsize=(6, 4))
    temps = [800, 850, 900, 950]
    energies = []
    for T in temps:
        p_tmp = CarburizationParams(temperature_C=T, surface_carbon_wt_percent=args.surface_c,
                                    initial_carbon_wt_percent=args.initial_c,
                                    sheet_thickness_um=args.thickness)
        m_tmp = CarburizationModel(p_tmp)
        energies.append(m_tmp.energy_estimate_kWh_per_kg(1.0)["with_furnace_efficiency_kWh_kg"])
    ax.bar([str(t) for t in temps], energies, color="#fdbf6f")
    ax.set(title="Heating energy vs T (screening, 1 kg Fe, 60% furnace eff.)",
           ylabel="kWh/kg", xlabel="Carburizing T (°C)")
    ax.grid(axis="y", alpha=0.25)
    en_fig = FIG_DIR / "carburization_energy.png"
    fig.tight_layout()
    fig.savefig(en_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {en_fig}")

    # Report JSON
    report = {
        "title": "Post-deposition gaseous carburization screening",
        "date": __import__("datetime").datetime.now().isoformat(),
        "params": {
            "temperature_C": args.temperature,
            "surface_c_wt": args.surface_c,
            "initial_c_wt": args.initial_c,
            "thickness_um": args.thickness,
            "D_m2_s": D,
            "phase": phase,
            "duration_hr": args.duration,
        },
        "summary": result.summary(),
        "composite_strength_final": comp,
        "energy": en,
        "figures": [str(prof_fig), str(case_fig), str(hv_fig), str(en_fig)],
        "csv": str(ts_path),
        "model_notes": [
            "Fickian diffusion, finite slab with both sides carburized, Fourier superposition for Fo>0.2",
            "D = D0 exp(-Q/RT) with γ-Fe D0=2.3e-5 m2/s Q=148 kJ/mol, α-Fe D0=6.2e-7 Q=80 kJ/mol (screening means)",
            "Hardness: Maynier HV=127+949C (C<0.8 wt%) saturating at 900 HV, quench-rate knockdown for bainite",
            "Tempering optional via Hollomon-Jaffe parameter",
            "Does NOT model carbide precipitation, retained austenite, or decarburization on quench",
            "Energy is only heating, not soak or atmosphere — screening only",
        ],
        "time_estimate_500um_case": t_est if 't_est' in locals() else None,
    }
    report_path = DATA_DIR / "carburization_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  ✅ Saved report: {report_path}")
    print("\n✅ Carburization driver complete!")

    return report


if __name__ == "__main__":
    main()
