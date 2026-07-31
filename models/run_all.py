"""
Master driver — runs the entire modeling suite end-to-end and writes a
consolidated report and dashboard.

Usage
-----
python -m models.run_all               # runs all modules
python -m models.run_all --quick       # skips heavy pulse/transport heavy grids
python -m models.run_all --out experiments/data/master_report.json

What it does
------------
1. Thermodynamic & kinetic baseline (electrochemistry)
2. Nernst-Planck transport with migration (transport)
3. Pulse / pulse-reverse transient dynamics (pulse)
4. Voltammetry + Tafel + EIS Phase I analysis (voltammetry, eis)
5. Hull-cell screened current + gravimetric FE Phase II (hull_cell)
6. Anomalous Fe-Ni + Guglielmi C co-deposition Phase III (co_deposition) +
   pulse-coupled co-deposition (surface pH recovery, enhanced transport)
7. Mechanical properties (grain → strength → hardness → grade)
8. Carburization post-processing (Fickian diffusion, case depth, HV profile)
9. Carbon potential / gas atmosphere (CO/CO2, CH4/H2, dew point → a_C, O2 probe)
10. Tempering + retained austenite (Ms, Koistinen-Marburger, Hollomon-Jaffe)
11. Foil + O2 probe calibration (inverse D, K offset, Hall-Petch fit) — via foil_calibration module, tested in synthetic examples
12. Anode durability + closed-loop CSTR Phase IV (closed_loop)
13. Techno-economics + scenario comparison (technoeconomic, scenarios)
14. Process-flow diagrams + pilot P&ID (process_flow, pid)
15. Dashboard summary figure + master JSON

Outputs
-------
* experiments/data/master_report.json   – consolidated report (all sub-reports)
* docs/figures/run_all_dashboard.png    – 2×3 dashboard summary
* docs/figures/process_flow_diagram.png
* docs/figures/process_flow_detailed.png
* docs/figures/mechanical_properties_sweep.png + alloy_vs_mechanical.png
and all standard figures from individual drivers (regenerated).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR = ROOT / "docs" / "figures"

# Ensure imports work when run as module or script
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the per-module drivers' core logic where possible to avoid subprocess overhead
from models.run_electrochemistry import main as run_electrochem_main
from models.run_transport import main as run_transport_main
from models.run_pulse import main as run_pulse_main
from models.run_voltammetry import main as run_volt_main
from models.run_eis import main as run_eis_main
from models.run_hull_cell import main as run_hull_main
from models.run_closed_loop import main as run_closed_loop_main
from models.run_technoeconomic import main as run_techno_main
from models.run_scenarios import main as run_scenarios_main
from models.run_mechanical_properties import main as run_mechanical_main
from models.run_carburization import main as run_carburization_main
from models.run_carbon_potential import main as run_carbon_potential_main
from models.run_tempering import main as run_tempering_main
from models.run_pid import main as run_pid_main
from models.run_speciation import main as run_speciation_main
from models.run_thermal_balance import main as run_thermal_balance_main
from models.run_operating_window import main as run_operating_window_main
from models.run_experimental_matrix import main as run_experimental_matrix_main
from models.run_cell_architecture import main as run_cell_architecture_main

from models.co_deposition import PhaseIIICoDeposition
from models.mechanical_properties import MechanicalPropertiesModel, build_mechanical_model_from_phase3_result
from models.process_flow import generate_process_flow_diagram, generate_detailed_flow_with_composition
from models.carburization import CarburizationModel, CarburizationParams


def _load_json(p: Path) -> dict:
    if p.exists():
        return json.loads(p.read_text())
    return {"_missing": str(p)}


def _run_co_deposition_full() -> dict:
    """Generate synthetic data and return summary for 3 mechanisms."""
    # Delegate to the existing driver which already handles all 3 mechanisms
    from models.run_co_deposition import main as co_main
    from models.co_deposition import build_phase3_model
    co_main()

    summary = {}
    for mech in ["hydroxide_suppression", "intermediate_adsorption", "mixed_metal_intermediate"]:
        csv_path = DATA_DIR / f"synthetic_co_deposition_{mech}.csv"
        model = build_phase3_model(mechanism_fe_ni=mech)
        r100 = model.run_at_current(100.0)
        # collect figures for this mechanism
        fig_glob = list(FIG_DIR.glob(f"co_deposition_*_{mech}.png"))
        summary[mech] = {
            "csv": str(csv_path),
            "figures": [str(p) for p in fig_glob],
            "at_100_mA_cm2": r100,
        }
    return summary


def _run_mechanical_properties(co_dep_summary: dict) -> dict:
    """Bridge Phase III -> mechanical properties, with sweeps and figures."""
    mech_model = MechanicalPropertiesModel()

    # Case matrix: DC, PE, PRE at 100 mA/cm2 avg with 3 co-dep mechanisms
    cases = {}
    waveforms = [
        ("dc", 1.0, 100.0, 100.0),
        ("pe", 0.5, 100.0, 200.0),
        ("pre", 0.5, 100.0, 200.0),
    ]

    for mech_name, mech_data in co_dep_summary.items():
        phase3_r100 = mech_data["at_100_mA_cm2"]
        for wf, duty, j_avg, j_peak in waveforms:
            res = build_mechanical_model_from_phase3_result(
                phase3_r100,
                j_avg_mA_cm2=j_avg,
                j_peak_mA_cm2=j_peak,
                duty_cycle=duty,
                waveform=wf,
                temperature_C=60.0,
            )
            key = f"{mech_name}_{wf}"
            cases[key] = res.summary()

    # Sweeps for plotting
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Mechanical sweep vs j for PE 50% duty, hydroxide_suppression case
    j_vals = np.linspace(20, 400, 50)
    sweep_pe = mech_model.sweep_current_density(j_vals, waveform="pe", duty_cycle=0.5,
                                                ni_wt_percent=2.0, carbon_wt_percent=0.8,
                                                current_efficiency_percent=93.0)
    sweep_dc = mech_model.sweep_current_density(j_vals, waveform="dc", duty_cycle=1.0,
                                                ni_wt_percent=2.0, carbon_wt_percent=0.8,
                                                current_efficiency_percent=93.0)
    sweep_pre = mech_model.sweep_current_density(j_vals, waveform="pre", duty_cycle=0.5,
                                                 ni_wt_percent=2.0, carbon_wt_percent=0.8,
                                                 current_efficiency_percent=93.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    # YS
    ax = axes[0]
    ax.plot(sweep_dc["j_mA_cm2"], sweep_dc["yield_MPa"], label="DC", color="#555555")
    ax.plot(sweep_pe["j_mA_cm2"], sweep_pe["yield_MPa"], label="PE 50%", color="#1874b4")
    ax.plot(sweep_pre["j_mA_cm2"], sweep_pre["yield_MPa"], label="PRE 50%", color="#e41a1c")
    ax.set_title("Yield strength vs j (screening)")
    ax.set_xlabel("j_avg (mA/cm²)")
    ax.set_ylabel("YS (MPa)")
    ax.grid(alpha=0.25)
    ax.legend()
    # UTS
    ax = axes[1]
    ax.plot(sweep_dc["j_mA_cm2"], sweep_dc["uts_MPa"], label="DC", color="#555555")
    ax.plot(sweep_pe["j_mA_cm2"], sweep_pe["uts_MPa"], label="PE 50%", color="#1874b4")
    ax.plot(sweep_pre["j_mA_cm2"], sweep_pre["uts_MPa"], label="PRE 50%", color="#e41a1c")
    ax.set_title("UTS vs j")
    ax.set_xlabel("j_avg (mA/cm²)")
    ax.set_ylabel("UTS (MPa)")
    ax.grid(alpha=0.25)
    ax.legend()
    # HV
    ax = axes[2]
    ax.plot(sweep_dc["j_mA_cm2"], sweep_dc["hv"], label="DC", color="#555555")
    ax.plot(sweep_pe["j_mA_cm2"], sweep_pe["hv"], label="PE 50%", color="#1874b4")
    ax.plot(sweep_pre["j_mA_cm2"], sweep_pre["hv"], label="PRE 50%", color="#e41a1c")
    ax.set_title("Vickers hardness vs j")
    ax.set_xlabel("j_avg (mA/cm²)")
    ax.set_ylabel("HV (kgf/mm²)")
    ax.grid(alpha=0.25)
    ax.legend()
    # Grain size
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

    # 2) Alloy vs mechanical mapping (Fe-Ni sweep at fixed j)
    ni_range = np.linspace(0, 10, 30)
    ys_vs_ni = []
    for ni in ni_range:
        r = mech_model.predict(j_avg_mA_cm2=100, j_peak_mA_cm2=200, duty_cycle=0.5,
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

    # Write report
    report = {
        "title": "Mechanical properties screening (run_all)",
        "model": "Hall-Petch + solid-solution Ni + Guglielmi C dispersion (screening, needs calibration)",
        "cases": cases,
        "sweep_figures": [str(sweep_fig), str(alloy_fig)],
        "note": "All predictions are screening-level; must be verified by Vickers, tensile, EBSD."
    }
    (DATA_DIR / "mechanical_properties_report.json").write_text(json.dumps(report, indent=2))

    return report


def _make_dashboard(quick: bool = False) -> Path:
    """Make a 2x3 dashboard from existing figures / reports."""

    # Load scenario report for text summary
    scenario_report = _load_json(DATA_DIR / "scenario_comparison_report.json")
    techno_report = _load_json(DATA_DIR / "technoeconomic_report.json")
    mech_report = _load_json(DATA_DIR / "mechanical_properties_report.json")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    # 1: Pourbaix (load existing png if present else blank with text)
    ax = axes[0]
    pp_path = FIG_DIR / "pourbaix_fe_h2o.png"
    if pp_path.exists():
        img = plt.imread(pp_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Pourbaix Fe-H2O")
    else:
        ax.text(0.5, 0.5, "Pourbaix\n(run electrochemistry)", ha="center", va="center")
        ax.axis("off")

    # 2: Transport pH comparison — generate simple bar from transport_report
    ax = axes[1]
    trans = _load_json(DATA_DIR / "transport_report.json")
    if "comparison" in trans:
        # Try to plot something meaningful else fallback
        try:
            labels = list(trans["comparison"].keys())[:3]
            vals = [trans["comparison"][k].get("current_efficiency_%", 0) for k in labels]
            ax.bar(labels, vals, color="#4daf4a")
            ax.set_title("Transport CE comparison (NP)")
            ax.tick_params(axis='x', rotation=20)
        except Exception:
            ax.text(0.5, 0.5, f"Transport\n{trans.get('note','')[:80]}", ha="center", va="center", fontsize=8)
            ax.axis("off")
    else:
        # Nernst-Planck figure
        np_path = FIG_DIR / "nernst_planck_profiles.png"
        if np_path.exists():
            img = plt.imread(np_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title("Nernst-Planck profiles")
        else:
            ax.axis("off")
            ax.set_title("Transport")

    # 3: Pulse reverse transient
    ax = axes[2]
    pr_path = FIG_DIR / "pulse_reverse_transient.png"
    if pr_path.exists():
        img = plt.imread(pr_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Pulse-reverse transient")
    else:
        ax.axis("off")

    # 4: Scenario comparison (cost)
    ax = axes[3]
    sc_path = FIG_DIR / "scenario_comparison.png"
    if sc_path.exists():
        img = plt.imread(sc_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Scenario comparison")
    else:
        ax.axis("off")

    # 5: Mechanical properties sweep
    ax = axes[4]
    ms_path = FIG_DIR / "mechanical_properties_sweep.png"
    if ms_path.exists():
        img = plt.imread(ms_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Mechanical sweep (new)")
    else:
        # fallback text from mech report
        txt = "\n".join([f"{k}: YS {v.get('yield_strength_MPa','?')} MPa, {v.get('grade_estimate','')[:35]}"
                         for k, v in list(mech_report.get("cases", {}).items())[:4]])
        ax.text(0.05, 0.95, txt or "Mechanical\n(not yet run)", va="top", ha="left", fontsize=7, transform=ax.transAxes)
        ax.axis("off")
        ax.set_title("Mechanical properties (screening)")

    # 6: Process flow
    ax = axes[5]
    pf_path = FIG_DIR / "process_flow_diagram.png"
    if pf_path.exists():
        img = plt.imread(pf_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Process flow")
    else:
        ax.axis("off")
        ax.set_title("Process flow TBD")

    fig.suptitle(f"Aqueous Electrowinning — Master Dashboard ({datetime.now().date()})", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "run_all_dashboard.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(quick: bool = False, master_out: Path = DATA_DIR / "master_report.json"):
    print("="*72)
    print("RUN_ALL — Aqueous Electrowinning Full Suite")
    print("="*72)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    master = {
        "title": "Aqueous Electrowinning — Master Report",
        "date": datetime.now().isoformat(),
        "quick": quick,
        "steps": {},
    }

    # 1 Electrochemistry
    print("\n[1/17] Electrochemistry (Pourbaix + kinetics)...")
    try:
        run_electrochem_main()
        master["steps"]["electrochemistry"] = _load_json(DATA_DIR / "electrochemistry_report.json")
        print("  ✅ electrochemistry")
    except Exception as e:
        print(f"  ❌ electrochemistry: {e}")
        master["steps"]["electrochemistry"] = {"error": str(e)}

    # 2 Transport
    print("\n[2/17] Transport (Nernst-Planck)...")
    try:
        run_transport_main()
        master["steps"]["transport"] = _load_json(DATA_DIR / "transport_report.json")
        print("  ✅ transport")
    except Exception as e:
        print(f"  ❌ transport: {e}")
        master["steps"]["transport"] = {"error": str(e)}

    # 3 Pulse
    print("\n[3/17] Pulse-reverse dynamics...")
    try:
        if not quick:
            run_pulse_main()
        master["steps"]["pulse"] = _load_json(DATA_DIR / "pulse_reverse_report.json")
        print("  ✅ pulse")
    except Exception as e:
        print(f"  ❌ pulse: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["pulse"] = {"error": str(e)}

    # 4 Voltammetry
    print("\n[4/17] Voltammetry / Tafel...")
    try:
        run_volt_main()
        master["steps"]["voltammetry"] = _load_json(DATA_DIR / "voltammetry_report.json")
        print("  ✅ voltammetry")
    except Exception as e:
        print(f"  ❌ voltammetry: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["voltammetry"] = {"error": str(e)}

    # 5 EIS
    print("\n[5/17] EIS...")
    try:
        run_eis_main()
        master["steps"]["eis"] = _load_json(DATA_DIR / "eis_report.json")
        print("  ✅ eis")
    except Exception as e:
        print(f"  ❌ eis: {e}")
        master["steps"]["eis"] = {"error": str(e)}

    # 6 Hull cell
    print("\n[6/17] Hull cell + gravimetric FE...")
    try:
        run_hull_main()
        master["steps"]["hull_cell"] = _load_json(DATA_DIR / "hull_cell_report.json")
        print("  ✅ hull_cell")
    except Exception as e:
        print(f"  ❌ hull_cell: {e}")
        master["steps"]["hull_cell"] = {"error": str(e)}

    # 7 Co-deposition (Phase III)
    print("\n[7/17] Co-deposition (Fe-Ni + C)...")
    try:
        co_summary = _run_co_deposition_full()
        master["steps"]["co_deposition"] = co_summary
        print("  ✅ co_deposition")
    except Exception as e:
        print(f"  ❌ co_deposition: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["co_deposition"] = {"error": str(e)}
        co_summary = {}

    # 8 Mechanical properties
    print("\n[8/17] Mechanical properties (Hall-Petch + ss + dispersion)...")
    try:
        mech_report = _run_mechanical_properties(co_summary) if co_summary else _run_mechanical_properties(
            {"hydroxide_suppression": {"at_100_mA_cm2": {"alloy_kinetics": {"ni_wt_percent": 2.0},
                                                        "carbon_incorporation": {"predicted_carbon_wt_percent": 0.5,
                                                                                "adjusted_ce_percent": 93}}}}
        )
        master["steps"]["mechanical_properties"] = mech_report
        print("  ✅ mechanical_properties")
    except Exception as e:
        print(f"  ❌ mechanical_properties: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["mechanical_properties"] = {"error": str(e)}

    # 8b Carburization post-processing
    print("\n[9/17] Carburization (Fickian case hardening)...")
    try:
        # Call with explicit kwargs to avoid argparse clash with --quick
        run_carburization_main(temperature=900.0, surface_c=1.10, initial_c=0.02, thickness=1000.0, duration=4.0, dt=0.2)
        master["steps"]["carburization"] = _load_json(DATA_DIR / "carburization_report.json")
        print("  ✅ carburization")
    except Exception as e:
        print(f"  ❌ carburization: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["carburization"] = {"error": str(e)}

    # 9 Carbon potential
    print("\n[10/17] Carbon potential (gas atmosphere)...")
    try:
        run_carbon_potential_main()
        master["steps"]["carbon_potential"] = _load_json(DATA_DIR / "carbon_potential_report.json")
        print("  ✅ carbon_potential")
    except Exception as e:
        print(f"  ❌ carbon_potential: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["carbon_potential"] = {"error": str(e)}

    # 10 Tempering + RA
    print("\n[11/17] Tempering + retained austenite...")
    try:
        run_tempering_main()
        master["steps"]["tempering"] = _load_json(DATA_DIR / "tempering_report.json")
        print("  ✅ tempering")
    except Exception as e:
        print(f"  ❌ tempering: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["tempering"] = {"error": str(e)}

    # 11 Foil calibration (synthetic example)
    print("\n[12/17] Foil + O2 probe calibration (synthetic)...")
    try:
        # Synthetic foil measurements: use foil_calibration module to fit D from its own synthetic data
        from models.foil_calibration import FoilMeasurement, fit_diffusivity_from_foil_data, fit_carbon_potential_offset
        # generate synthetic measurements mimicking 930C, pCO=0.2, pCO2=0.001
        synthetic_foils = [
            FoilMeasurement(time_hr=0.5, temperature_C=930, pCO_atm=0.2, pCO2_atm=0.001, foil_thickness_um=75, measured_avg_C_wt_percent=0.35, o2_probe_mV=1150),
            FoilMeasurement(time_hr=1.0, temperature_C=930, pCO_atm=0.2, pCO2_atm=0.001, foil_thickness_um=75, measured_avg_C_wt_percent=0.65, o2_probe_mV=1120),
            FoilMeasurement(time_hr=2.0, temperature_C=930, pCO_atm=0.2, pCO2_atm=0.001, foil_thickness_um=75, measured_avg_C_wt_percent=0.95, o2_probe_mV=1100),
            FoilMeasurement(time_hr=4.0, temperature_C=930, pCO_atm=0.2, pCO2_atm=0.001, foil_thickness_um=75, measured_avg_C_wt_percent=1.05, o2_probe_mV=1090),
        ]
        fit_D = fit_diffusivity_from_foil_data(synthetic_foils, initial_C_wt=0.02)
        fit_O2 = fit_carbon_potential_offset(synthetic_foils)
        master["steps"]["foil_calibration"] = {"D_fit": fit_D, "O2_offset": fit_O2}
        print(f"  ✅ foil calibration: D_fit={fit_D['D_fit_m2_s']:.2e} vs theory {fit_D['D_theory_m2_s']:.2e}, O2 offset={fit_O2.get('offset_factor_aC_probe_over_theory_mean',1):.3f}")
    except Exception as e:
        print(f"  ❌ foil calibration: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["foil_calibration"] = {"error": str(e)}

    # 12 Techno + Scenarios
    print("\n[13/17] Technoeconomics + scenarios...")
    try:
        run_techno_main()
        run_scenarios_main()
        master["steps"]["technoeconomic"] = _load_json(DATA_DIR / "technoeconomic_report.json")
        master["steps"]["scenarios"] = _load_json(DATA_DIR / "scenario_comparison_report.json")
        print("  ✅ technoeconomic & scenarios")
    except Exception as e:
        print(f"  ❌ technoeconomic/scenarios: {e}")
        master["steps"]["technoeconomic"] = {"error": str(e)}

    # 13 Process flow + PID
    print("\n[14/17] Process flow diagrams + pilot P&ID...")
    try:
        pf1 = generate_process_flow_diagram()
        pf2 = generate_detailed_flow_with_composition()
        # PID
        try:
            run_pid_main()
            pid_figs = [str(FIG_DIR / "pid_overview.png"), str(FIG_DIR / "pid_detailed.png")]
            pid_report = _load_json(DATA_DIR / "pid_report.json")
        except Exception as e_pid:
            print(f"    PID generation note: {e_pid}")
            pid_figs = []
            pid_report = {"error": str(e_pid)}
        master["steps"]["process_flow"] = {"figures": [str(pf1), str(pf2)]}
        master["steps"]["pid"] = {"figures": pid_figs, "report": pid_report}
        print(f"  ✅ process_flow: {pf1}, {pf2}")
        print(f"  ✅ pid: {pid_figs}")
    except Exception as e:
        print(f"  ❌ process_flow/pid: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["process_flow"] = {"error": str(e)}

    # 15 Pulse-coupled co-deposition analytics extra
    print("\n[15/17] Pulse-coupled co-deposition analytics...")
    try:
        from models.co_deposition import build_phase3_model
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Quick comparison DC vs PE vs PRE for hydroxide_suppression
        model = build_phase3_model(mechanism_fe_ni="hydroxide_suppression")
        j_avg = 50.0
        dc_res = model.run_at_current(j_avg)
        pe_res = model.run_at_current_pulsed(j_avg, j_avg*2, duty_cycle=0.5, waveform="pe")
        pre_res = model.run_at_current_pulsed(j_avg, j_avg*2, duty_cycle=0.5, waveform="pre")
        master["steps"]["pulse_coupled_co_deposition"] = {
            "dc_at_50": dc_res["alloy_kinetics"],
            "pe_avg50_peak100": pe_res["alloy_kinetics"],
            "pre_avg50_peak100": pre_res["alloy_kinetics"],
            "note": "PE/PRE surface pH lower (recovery) → less hydroxide suppression, higher Ni expected (if mechanism is hydroxide-driven)"
        }
        # Figure
        labels = ["DC 50", "PE 50/100", "PRE 50/100"]
        fe_vals = [dc_res["alloy_kinetics"]["fe_wt_percent"],
                   pe_res["alloy_kinetics"]["fe_wt_percent"],
                   pre_res["alloy_kinetics"]["fe_wt_percent"]]
        ni_vals = [dc_res["alloy_kinetics"]["ni_wt_percent"],
                   pe_res["alloy_kinetics"]["ni_wt_percent"],
                   pre_res["alloy_kinetics"]["ni_wt_percent"]]
        pH_vals = [model.kinetics_model.surface_pH(j_avg),
                   pe_res["alloy_kinetics"].get("pulsed_surface_pH", 0),
                   pre_res["alloy_kinetics"].get("pulsed_surface_pH", 0)]
        fig, axes = plt.subplots(1,2, figsize=(11,4.5))
        ax = axes[0]
        x = np.arange(3)
        ax.bar(x-0.15, fe_vals, width=0.3, label="Fe wt%", color="#1874b4")
        ax.bar(x+0.15, ni_vals, width=0.3, label="Ni wt%", color="#d95f02")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("wt%"); ax.set_title("Alloy composition DC vs PE vs PRE (screening)"); ax.legend(); ax.grid(alpha=0.25)
        ax = axes[1]
        ax.bar(labels, pH_vals, color="#4daf4a")
        ax.set_title("Surface pH DC vs pulse-coupled (recovery)"); ax.set_ylabel("pH surf"); ax.grid(alpha=0.25)
        fig.suptitle("Pulse-coupled co-deposition: pH recovery reduces hydroxide suppression", fontweight="bold")
        fig.tight_layout()
        fig_path = FIG_DIR / "pulse_coupled_co_deposition.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        master["steps"]["pulse_coupled_co_deposition"]["figure"] = str(fig_path)
        print(f"  ✅ pulse-coupled co-deposition: {fig_path}")
    except Exception as e:
        print(f"  ❌ pulse-coupled co-deposition: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["pulse_coupled_co_deposition"] = {"error": str(e)}

    # Pre-lab pre-experiment modeling suite (Speciation, Thermal Balance, Operating Window, DOE Matrix)
    print("\n[16/17] Pre-lab modeling suite (Speciation, Thermal, Operating Window, DOE Matrix)...")
    try:
        run_speciation_main()
        master["steps"]["speciation"] = _load_json(DATA_DIR / "speciation_report.json")
        run_thermal_balance_main()
        master["steps"]["thermal_balance"] = _load_json(DATA_DIR / "thermal_balance_report.json")
        run_operating_window_main()
        master["steps"]["operating_window"] = _load_json(DATA_DIR / "operating_window_report.json")
        run_experimental_matrix_main()
        master["steps"]["experimental_matrix"] = _load_json(DATA_DIR / "experimental_matrix_report.json")
        print("  ✅ pre-lab modeling suite")
    except Exception as e:
        print(f"  ❌ pre-lab modeling suite: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["pre_lab_suite"] = {"error": str(e)}

    # Cell architecture screen (areal productivity, $/m², kill criterion #3)
    print("\n[17/17] Cell architecture screen (harvesting, $/m², kill criterion #3)...")
    try:
        run_cell_architecture_main()
        master["steps"]["cell_architecture"] = _load_json(
            DATA_DIR / "cell_architecture_report.json"
        )
        print("  ✅ cell architecture")
    except Exception as e:
        print(f"  ❌ cell architecture: {e}")
        import traceback; traceback.print_exc()
        master["steps"]["cell_architecture"] = {"error": str(e)}

    # Dashboard
    print("\n[Dashboard] Generating master dashboard...")
    try:
        dash = _make_dashboard(quick=quick)
        master["dashboard_figure"] = str(dash)
        print(f"  ✅ dashboard: {dash}")
    except Exception as e:
        print(f"  ❌ dashboard: {e}")

    # Write master report
    master_out = Path(master_out)
    master_out.parent.mkdir(parents=True, exist_ok=True)
    master_out.write_text(json.dumps(master, indent=2))
    print("\n" + "="*72)
    print(f"✅ RUN_ALL complete — master report: {master_out}")
    print(f"   Dashboard: {FIG_DIR / 'run_all_dashboard.png'}")
    print("="*72)
    return master


def cli():
    """Entry point for aq-steel console script."""
    parser = argparse.ArgumentParser(description="Run all aqueous electrowinning models")
    parser.add_argument("--quick", action="store_true", help="Skip heavy grids (pulse comparisons)")
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "master_report.json"), help="Master report output path")
    args = parser.parse_args()
    main(quick=args.quick, master_out=Path(args.out))


if __name__ == "__main__":
    cli()
