#!/usr/bin/env python3
"""
Phase V — Mechanical properties screening notebook (executable script).

Usage:
    python experiments/notebooks/phase5_mechanical.py
    python experiments/notebooks/phase5_mechanical.py --mechanism hydroxide_suppression --waveform pre

This bridges Phase III co-deposition composition predictions to structural
properties via Hall-Petch grain refinement, solid-solution, and dispersion
strengthening.

It does NOT replace real Vickers, tensile, or EBSD measurements — all
coefficients are screening-level and must be calibrated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.co_deposition import build_phase3_model
from models.mechanical_properties import (
    MechanicalPropertiesModel,
    build_mechanical_model_from_phase3_result,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mechanism", default="hydroxide_suppression",
                   choices=["hydroxide_suppression", "intermediate_adsorption", "mixed_metal_intermediate"])
    p.add_argument("--waveform", default="pe", choices=["dc", "pe", "pre"])
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--j-avg", type=float, default=100.0)
    p.add_argument("--j-peak", type=float, default=200.0)
    p.add_argument("--temperature-C", type=float, default=60.0)
    p.add_argument("--output", default="docs/figures/phase5_mechanical_example.png")
    return p


def main():
    args = build_parser().parse_args()

    print(f"Phase III mechanism: {args.mechanism}, waveform: {args.waveform}")

    p3_model = build_phase3_model(mechanism_fe_ni=args.mechanism)
    p3_res = p3_model.run_at_current(args.j_avg)

    print("Co-deposition at j_avg:")
    print(f"  Fe {p3_res['alloy_kinetics']['fe_wt_percent']} wt%, "
          f"Ni {p3_res['alloy_kinetics']['ni_wt_percent']} wt%, "
          f"C {p3_res['carbon_incorporation']['predicted_carbon_wt_percent']} wt%")

    mech_model = MechanicalPropertiesModel()
    result = build_mechanical_model_from_phase3_result(
        p3_res,
        j_avg_mA_cm2=args.j_avg,
        j_peak_mA_cm2=args.j_peak,
        duty_cycle=args.duty,
        waveform=args.waveform,
        temperature_C=args.temperature_C,
    )

    print("\nMechanical properties (screening):")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        j_vals = np.linspace(20, 400, 40)
        sweep = mech_model.sweep_current_density(
            j_vals, waveform=args.waveform, duty_cycle=args.duty,
            ni_wt_percent=result.ni_wt_percent,
            carbon_wt_percent=result.carbon_wt_percent,
            current_efficiency_percent=result.ce_percent,
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(sweep["j_mA_cm2"], sweep["yield_MPa"], label="YS", color="#1874b4")
        ax.plot(sweep["j_mA_cm2"], sweep["uts_MPa"], label="UTS", color="#d95f02")
        ax.set(title=f"YS/UTS vs j ({args.mechanism}, {args.waveform})",
               xlabel="j_avg (mA/cm²)", ylabel="Strength (MPa)")
        ax.grid(alpha=0.25)
        ax.legend()
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        print(f"\n  Wrote {out}")

    except Exception as exc:
        print(f"Plotting failed (non-fatal): {exc}")


if __name__ == "__main__":
    main()
