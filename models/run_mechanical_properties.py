"""
Driver for mechanical properties screening — maps Phase III co-deposition
predictions to Hall-Petch + solid-solution + dispersion strengthening and
structural grade estimates.

Generates:
* docs/figures/mechanical_properties_sweep.png
* docs/figures/alloy_vs_mechanical.png
* docs/figures/mechanical_process_flow.png (delegates to process_flow.py)
* experiments/data/mechanical_properties_report.json
* experiments/data/synthetic_mechanical_properties.csv
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mechanical_properties import MechanicalPropertiesModel, build_mechanical_model_from_phase3_result
from models.co_deposition import build_phase3_model
from models.process_flow import generate_process_flow_diagram, generate_detailed_flow_with_composition

DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR = ROOT / "docs" / "figures"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("Phase III → Mechanical Properties driver")
    print("="*72)

    model = MechanicalPropertiesModel()

    # Example Phase III operating points (3 mechanisms × 3 waveforms)
    mechanisms = ["hydroxide_suppression", "intermediate_adsorption", "mixed_metal_intermediate"]
    waveforms = [
        ("dc", 1.0, 100.0, 100.0),
        ("pe", 0.5, 100.0, 200.0),
        ("pre", 0.5, 100.0, 200.0),
    ]

    rows = []
    cases = {}

    for mech in mechanisms:
        p3 = build_phase3_model(
            bath_fe_M=0.6, bath_ni_M=0.4, pH=3.5, temperature_C=60.0,
            carbon_particle_loading_g_L=2.5,
            mechanism_fe_ni=mech,
        )
        p3_at_100 = p3.run_at_current(100.0)

        for wf, duty, j_avg, j_peak in waveforms:
            mech_res = build_mechanical_model_from_phase3_result(
                p3_at_100,
                j_avg_mA_cm2=j_avg,
                j_peak_mA_cm2=j_peak,
                duty_cycle=duty,
                waveform=wf,
                temperature_C=60.0,
            )
            s = mech_res.summary()
            key = f"{mech}_{wf}"
            cases[key] = s
            rows.append({
                "mechanism_fe_ni": mech,
                "waveform": wf,
                "duty_cycle": duty,
                "j_avg_mA_cm2": j_avg,
                "j_peak_mA_cm2": j_peak,
                "grain_size_um": s["grain_size_um"],
                "ni_wt_percent": s["composition"]["ni_wt_pct"],
                "carbon_wt_percent": s["composition"]["c_wt_pct"],
                "porosity": s["porosity"],
                "yield_MPa": s["yield_strength_MPa"],
                "uts_MPa": s["uts_MPa"],
                "elongation_pct": s["elongation_percent"],
                "vickers_hv": s["vickers_hv_kgf_mm2"],
                "grade_estimate": s["grade_estimate"],
                "flags": ";".join(s["flags"]),
            })
            print(f"  {key:45s} d={s['grain_size_um']:.2f} µm  YS={s['yield_strength_MPa']:.0f} MPa  "
                  f"UTS={s['uts_MPa']:.0f} MPa  HV={s['vickers_hv_kgf_mm2']:.0f}  {s['grade_estimate'][:50]}")

    # Save synthetic CSV
    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "synthetic_mechanical_properties.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Saved CSV: {csv_path} ({len(df)} rows)")

    # Generate sweeps (reuse logic from run_all)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j_vals = np.linspace(20, 400, 50)
    sweep_pe = model.sweep_current_density(j_vals, waveform="pe", duty_cycle=0.5,
                                           ni_wt_percent=2.0, carbon_wt_percent=0.8,
                                           current_efficiency_percent=93.0)
    sweep_dc = model.sweep_current_density(j_vals, waveform="dc", duty_cycle=1.0,
                                           ni_wt_percent=2.0, carbon_wt_percent=0.8,
                                           current_efficiency_percent=93.0)
    sweep_pre = model.sweep_current_density(j_vals, waveform="pre", duty_cycle=0.5,
                                            ni_wt_percent=2.0, carbon_wt_percent=0.8,
                                            current_efficiency_percent=93.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    ax = axes[0]
    ax.plot(sweep_dc["j_mA_cm2"], sweep_dc["yield_MPa"], label="DC", color="#555555")
    ax.plot(sweep_pe["j_mA_cm2"], sweep_pe["yield_MPa"], label="PE 50%", color="#1874b4")
    ax.plot(sweep_pre["j_mA_cm2"], sweep_pre["yield_MPa"], label="PRE 50%", color="#e41a1c")
    ax.set_title("Yield strength vs j (screening)")
    ax.set_xlabel("j_avg (mA/cm²)")
    ax.set_ylabel("YS (MPa)")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(sweep_dc["j_mA_cm2"], sweep_dc["uts_MPa"], label="DC", color="#555555")
    ax.plot(sweep_pe["j_mA_cm2"], sweep_pe["uts_MPa"], label="PE 50%", color="#1874b4")
    ax.plot(sweep_pre["j_mA_cm2"], sweep_pre["uts_MPa"], label="PRE 50%", color="#e41a1c")
    ax.set_title("UTS vs j")
    ax.set_xlabel("j_avg (mA/cm²)")
    ax.set_ylabel("UTS (MPa)")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2]
    ax.plot(sweep_dc["j_mA_cm2"], sweep_dc["hv"], label="DC", color="#555555")
    ax.plot(sweep_pe["j_mA_cm2"], sweep_pe["hv"], label="PE 50%", color="#1874b4")
    ax.plot(sweep_pre["j_mA_cm2"], sweep_pre["hv"], label="PRE 50%", color="#e41a1c")
    ax.set_title("Vickers hardness vs j")
    ax.set_xlabel("j_avg (mA/cm²)")
    ax.set_ylabel("HV (kgf/mm²)")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[3]
    ax.semilogy(sweep_dc["j_mA_cm2"], sweep_dc["grain_size_um"], label="DC", color="#555555")
    ax.semilogy(sweep_pe["j_mA_cm2"], sweep_pe["grain_size_um"], label="PE 50%", color="#1874b4")
    ax.semilogy(sweep_pre["j_mA_cm2"], sweep_pre["grain_size_um"], label="PRE 50%", color="#e41a1c")
    ax.set_title("Grain size vs j (log)")
    ax.set_xlabel("j_avg (mA/cm²)")
    ax.set_ylabel("d (µm)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("Mechanical properties sweep — Fe-2Ni-0.8C screening model", fontweight="bold")
    fig.tight_layout()
    sweep_fig = FIG_DIR / "mechanical_properties_sweep.png"
    fig.savefig(sweep_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {sweep_fig}")

    # Alloy vs mechanical: Ni sweep
    ni_range = np.linspace(0, 10, 30)
    ys_vs_ni = []
    for ni in ni_range:
        r = model.predict(j_avg_mA_cm2=100, j_peak_mA_cm2=200, duty_cycle=0.5,
                          waveform="pe", ni_wt_percent=float(ni), carbon_wt_percent=0.5,
                          current_efficiency_percent=93)
        ys_vs_ni.append(r.sigma_y_MPa)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ni_range, ys_vs_ni, color="#d95f02", marker="o", ms=3)
    ax.set_title("YS vs Ni wt% (j=100 mA/cm² PE, 0.5% C screening)")
    ax.set_xlabel("Ni wt%")
    ax.set_ylabel("YS (MPa)")
    ax.grid(alpha=0.25)
    alloy_fig = FIG_DIR / "alloy_vs_mechanical.png"
    fig.tight_layout()
    fig.savefig(alloy_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {alloy_fig}")

    # Process flow diagrams (also part of run_all but generate here)
    pf1 = generate_process_flow_diagram()
    pf2 = generate_detailed_flow_with_composition()
    print(f"  ✅ Saved process flow: {pf1}, {pf2}")

    # Report
    report = {
        "title": "Mechanical properties screening (Phase III → structural)",
        "date": __import__("datetime").datetime.now().isoformat(),
        "model": "Hall-Petch (k=0.5 MPa√m) + solid-solution Ni (38 MPa/wt%^0.75) + Guglielmi C dispersion (180 MPa/wt%^0.6) + porosity penalty",
        "cases": cases,
        "figures": [str(sweep_fig), str(alloy_fig), str(pf1), str(pf2)],
        "csv": str(csv_path),
        "calibration_note": "All coefficients are screening assumptions from literature compilations; must be calibrated with real HV, tensile, EBSD.",
        "limits": [
            "No texture / residual stress model",
            "No hydrogen embrittlement beyond porosity proxy",
            "Carbon described as dispersed particles, not dissolved interstitial (which would be stronger per wt%)",
            "Grain-size estimator is empirical, not phase-field",
        ],
    }
    report_path = DATA_DIR / "mechanical_properties_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  ✅ Saved report: {report_path}")
    print("\n✅ Mechanical properties driver complete!")
    return report


if __name__ == "__main__":
    main()
