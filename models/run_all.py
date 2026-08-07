"""
Master driver — runs the entire modeling suite end-to-end and writes a
consolidated report and dashboard.

Usage
-----
python -m models.run_all               # runs all modules (cached where possible)
python -m models.run_all --quick       # skips heavy pulse/transport heavy grids
python -m models.run_all --no-cache    # force full recompute (ignore step cache)
python -m models.run_all --force-step electrochemistry  # recompute one step
python -m models.run_all --out experiments/data/master_report.json

Incremental caching
-------------------
Each step is content-addressed by its source code + transitive deps + params.
If nothing changed, the step is skipped (⏩) and its cached outputs are reused.
Changing one model file only invalidates steps that transitively depend on it.
The cache manifest lives at ``experiments/data/.step_cache/manifest.json``.

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
15. Internal stress & coupon curvature (Stoney/bent-strip) + adhesion/peel (adhesion_peel, internal_stress)
16. RDE kinetics/transport separation — Levich + Koutecky-Levich (rde_levich)
17. Cathode-channel gas hold-up: void fraction, current redistribution, H2 safety (gas_holdup)
18. Unified RC-1 reference-cell state: physics + gas + thermal + ledgers + advisory safety (reference_cell_pipeline)
19. Dashboard summary figure + master JSON

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
from typing import Optional, Set
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
from models.run_technoeconomic import main as run_techno_main
from models.run_scenarios import main as run_scenarios_main
from models.run_carburization import main as run_carburization_main
from models.run_carbon_potential import main as run_carbon_potential_main
from models.run_tempering import main as run_tempering_main
from models.run_thermomechanical import main as run_thermomechanical_main
from models.run_pid import main as run_pid_main
from models.run_speciation import main as run_speciation_main
from models.run_thermal_balance import main as run_thermal_balance_main
from models.run_operating_window import main as run_operating_window_main
from models.run_experimental_matrix import main as run_experimental_matrix_main
from models.run_cell_architecture import main as run_cell_architecture_main
from models.run_adhesion_peel import main as run_adhesion_peel_main
from models.run_internal_stress import main as run_internal_stress_main
from models.run_gas_holdup import main as run_gas_holdup_main
from models.run_rde_levich import main as run_rde_levich_main
from models.run_physics_tranche3 import run_physics_tranche3

from models.mechanical_properties import MechanicalPropertiesModel, build_mechanical_model_from_phase3_result
from models.process_flow import generate_process_flow_diagram, generate_detailed_flow_with_composition

# Incremental step cache — content-addressed by source + deps + params
from models.step_cache import StepCache


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


def _step_outputs(report_name: str, fig_names: list[str] | None = None) -> list[Path]:
    """Build the list of output paths for a step (report JSON + figures)."""
    paths = [DATA_DIR / report_name]
    if fig_names:
        paths.extend(FIG_DIR / f for f in fig_names)
    return paths


def main(
    quick: bool = False,
    master_out: Path = DATA_DIR / "master_report.json",
    cache_enabled: bool = True,
    force_steps: Optional[Set[str]] = None,
):
    print("=" * 72)
    print("RUN_ALL — Aqueous Electrowinning Full Suite")
    print("=" * 72)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cache = StepCache(enabled=cache_enabled, force_steps=force_steps or set())
    if cache_enabled:
        print(f"  Step cache: enabled (manifest: experiments/data/.step_cache/manifest.json)")
    else:
        print(f"  Step cache: disabled (--no-cache)")

    master = {
        "title": "Aqueous Electrowinning — Master Report",
        "date": datetime.now().isoformat(),
        "quick": quick,
        "steps": {},
    }

    # 1 Electrochemistry
    print("\n[1/22] Electrochemistry (Pourbaix + kinetics)...")
    with cache.step("electrochemistry", "models.run_electrochemistry",
                    _step_outputs("electrochemistry_report.json",
                                  ["pourbaix_fe_h2o.png", "polarization_curves.png",
                                   "current_efficiency_map.png"])) as hit:
        if not hit:
            try:
                run_electrochem_main()
                print("  ✅ electrochemistry")
            except Exception as e:
                print(f"  ❌ electrochemistry: {e}")
    master["steps"]["electrochemistry"] = _load_json(DATA_DIR / "electrochemistry_report.json")

    # 2 Transport
    print("\n[2/22] Transport (Nernst-Planck)...")
    with cache.step("transport", "models.run_transport",
                    _step_outputs("transport_report.json",
                                  ["nernst_planck_profiles.png",
                                   "migration_enhancement.png",
                                   "transport_model_comparison.png"])) as hit:
        if not hit:
            try:
                run_transport_main()
                print("  ✅ transport")
            except Exception as e:
                print(f"  ❌ transport: {e}")
    master["steps"]["transport"] = _load_json(DATA_DIR / "transport_report.json")

    # 3 Pulse
    print("\n[3/22] Pulse-reverse dynamics...")
    with cache.step("pulse", "models.run_pulse",
                    _step_outputs("pulse_reverse_report.json",
                                  ["pulse_reverse_transient.png",
                                   "dc_vs_pulse_comparison.png"]),
                    params={"quick": quick}) as hit:
        if not hit:
            try:
                if not quick:
                    run_pulse_main()
                print("  ✅ pulse")
            except Exception as e:
                print(f"  ❌ pulse: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["pulse"] = _load_json(DATA_DIR / "pulse_reverse_report.json")

    # 4 Voltammetry
    print("\n[4/22] Voltammetry / Tafel...")
    with cache.step("voltammetry", "models.run_voltammetry",
                    _step_outputs("voltammetry_report.json",
                                  ["voltammetry_analysis.png",
                                   "tafel_analysis.png"])) as hit:
        if not hit:
            try:
                run_volt_main()
                print("  ✅ voltammetry")
            except Exception as e:
                print(f"  ❌ voltammetry: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["voltammetry"] = _load_json(DATA_DIR / "voltammetry_report.json")

    # 5 EIS
    print("\n[5/22] EIS...")
    with cache.step("eis", "models.run_eis",
                    _step_outputs("eis_report.json",
                                  ["eis_nyquist.png", "eis_bode.png"])) as hit:
        if not hit:
            try:
                run_eis_main()
                print("  ✅ eis")
            except Exception as e:
                print(f"  ❌ eis: {e}")
    master["steps"]["eis"] = _load_json(DATA_DIR / "eis_report.json")

    # 6 Hull cell
    print("\n[6/22] Hull cell + gravimetric FE...")
    with cache.step("hull_cell", "models.run_hull_cell",
                    _step_outputs("hull_cell_report.json",
                                  ["hull_cell_current_distribution.png",
                                   "gravimetric_faradaic_efficiency.png"])) as hit:
        if not hit:
            try:
                run_hull_main()
                print("  ✅ hull_cell")
            except Exception as e:
                print(f"  ❌ hull_cell: {e}")
    master["steps"]["hull_cell"] = _load_json(DATA_DIR / "hull_cell_report.json")

    # 7 Co-deposition (Phase III)
    print("\n[7/22] Co-deposition (Fe-Ni + C)...")
    with cache.step("co_deposition", "models.run_co_deposition",
                    _step_outputs("co_deposition_report.json")) as hit:
        if not hit:
            try:
                co_summary = _run_co_deposition_full()
                print("  ✅ co_deposition")
            except Exception as e:
                print(f"  ❌ co_deposition: {e}")
                import traceback; traceback.print_exc()
                co_summary = {}
        else:
            # Load cached co-deposition summary from existing report
            co_summary = _load_json(DATA_DIR / "co_deposition_report.json")
    master["steps"]["co_deposition"] = co_summary if isinstance(co_summary, dict) and "error" not in co_summary else _load_json(DATA_DIR / "co_deposition_report.json")

    # 8 Mechanical properties
    print("\n[8/22] Mechanical properties (Hall-Petch + ss + dispersion)...")
    with cache.step("mechanical_properties", "models.run_mechanical_properties",
                    _step_outputs("mechanical_properties_report.json",
                                  ["mechanical_properties_sweep.png",
                                   "alloy_vs_mechanical.png"])) as hit:
        if not hit:
            try:
                mech_report = _run_mechanical_properties(co_summary) if co_summary else _run_mechanical_properties(
                    {"hydroxide_suppression": {"at_100_mA_cm2": {"alloy_kinetics": {"ni_wt_percent": 2.0},
                                                                "carbon_incorporation": {"predicted_carbon_wt_percent": 0.5,
                                                                                        "adjusted_ce_percent": 93}}}}
                )
                print("  ✅ mechanical_properties")
            except Exception as e:
                print(f"  ❌ mechanical_properties: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["mechanical_properties"] = _load_json(DATA_DIR / "mechanical_properties_report.json")

    # 8b Carburization post-processing
    print("\n[9/22] Carburization (Fickian case hardening)...")
    with cache.step("carburization", "models.run_carburization",
                    _step_outputs("carburization_report.json",
                                  ["carburization_profiles.png",
                                   "carburization_case_depth.png",
                                   "carburization_hardness.png",
                                   "carburization_energy.png"]),
                    params={"temperature": 900.0, "surface_c": 1.10, "initial_c": 0.02,
                            "thickness": 1000.0, "duration": 4.0, "dt": 0.2}) as hit:
        if not hit:
            try:
                run_carburization_main(temperature=900.0, surface_c=1.10, initial_c=0.02,
                                       thickness=1000.0, duration=4.0, dt=0.2)
                print("  ✅ carburization")
            except Exception as e:
                print(f"  ❌ carburization: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["carburization"] = _load_json(DATA_DIR / "carburization_report.json")

    # 9 Carbon potential
    print("\n[10/22] Carbon potential (gas atmosphere)...")
    with cache.step("carbon_potential", "models.run_carbon_potential",
                    _step_outputs("carbon_potential_report.json",
                                  ["carbon_potential_map.png",
                                   "carbon_potential_dewpoint.png",
                                   "carbon_potential_Acm.png"])) as hit:
        if not hit:
            try:
                run_carbon_potential_main()
                print("  ✅ carbon_potential")
            except Exception as e:
                print(f"  ❌ carbon_potential: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["carbon_potential"] = _load_json(DATA_DIR / "carbon_potential_report.json")

    # 10 Tempering + RA
    print("\n[11/22] Tempering + retained austenite...")
    with cache.step("tempering", "models.run_tempering",
                    _step_outputs("tempering_report.json",
                                  ["tempering_curve.png",
                                   "tempering_energy.png",
                                   "retained_austenite.png",
                                   "case_tempered_hardness.png"])) as hit:
        if not hit:
            try:
                run_tempering_main()
                print("  ✅ tempering")
            except Exception as e:
                print(f"  ❌ tempering: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["tempering"] = _load_json(DATA_DIR / "tempering_report.json")

    # 10b Thermomechanical (cold roll + recrystallization anneal)
    print("\n[11b/22] Thermomechanical (roll + recrystallize foil to sheet)...")
    with cache.step("thermomechanical", "models.run_thermomechanical",
                    _step_outputs("thermomechanical_report.json",
                                  ["thermomech_recrystallization.png",
                                   "thermomech_temperature_sweep.png",
                                   "thermomech_reduction_sweep.png",
                                   "thermomech_deposit_vs_annealed.png"]),
                    params={"reduction": 0.5, "passes": 2, "anneal_temp": 700.0,
                            "anneal_time": 60.0, "grain": 1.0, "ni": 0.0,
                            "carbon": 0.0, "ce": 95.0}) as hit:
        if not hit:
            try:
                run_thermomechanical_main(reduction=0.5, passes=2, anneal_temp=700.0,
                                          anneal_time=60.0, grain=1.0, ni=0.0,
                                          carbon=0.0, ce=95.0)
                print("  ✅ thermomechanical")
            except Exception as e:
                print(f"  ❌ thermomechanical: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["thermomechanical"] = _load_json(DATA_DIR / "thermomechanical_report.json")

    # 11 Foil calibration (synthetic example)
    print("\n[12/22] Foil + O2 probe calibration (synthetic)...")
    with cache.step("foil_calibration", "models.foil_calibration",
                    []) as hit:
        if not hit:
            try:
                from models.foil_calibration import FoilMeasurement, fit_diffusivity_from_foil_data, fit_carbon_potential_offset
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
        else:
            master["steps"]["foil_calibration"] = _load_json(DATA_DIR / "foil_calibration_report.json")

    # 12 Techno + Scenarios
    print("\n[13/22] Technoeconomics + scenarios...")
    with cache.step("technoeconomic", "models.run_technoeconomic",
                    _step_outputs("technoeconomic_report.json",
                                  ["voltage_breakdown.png", "opex_breakdown.png",
                                   "cost_comparison.png", "energy_vs_cost.png"])) as hit:
        if not hit:
            try:
                run_techno_main()
                print("  ✅ technoeconomic")
            except Exception as e:
                print(f"  ❌ technoeconomic: {e}")
    master["steps"]["technoeconomic"] = _load_json(DATA_DIR / "technoeconomic_report.json")

    with cache.step("scenarios", "models.run_scenarios",
                    _step_outputs("scenario_comparison_report.json",
                                  ["scenario_comparison.png",
                                   "scenario_radar.png"])) as hit:
        if not hit:
            try:
                run_scenarios_main()
                print("  ✅ scenarios")
            except Exception as e:
                print(f"  ❌ scenarios: {e}")
    master["steps"]["scenarios"] = _load_json(DATA_DIR / "scenario_comparison_report.json")

    # 13 Process flow + PID
    print("\n[14/22] Process flow diagrams + pilot P&ID...")
    with cache.step("process_flow", "models.process_flow",
                    [FIG_DIR / "process_flow_diagram.png",
                     FIG_DIR / "process_flow_detailed.png"]) as hit:
        if not hit:
            try:
                pf1 = generate_process_flow_diagram()
                pf2 = generate_detailed_flow_with_composition()
                print(f"  ✅ process_flow: {pf1}, {pf2}")
            except Exception as e:
                print(f"  ❌ process_flow: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["process_flow"] = {"figures": [str(FIG_DIR / "process_flow_diagram.png"),
                                                    str(FIG_DIR / "process_flow_detailed.png")]}

    with cache.step("pid", "models.run_pid",
                    _step_outputs("pid_report.json",
                                  ["pid_overview.png", "pid_detailed.png"])) as hit:
        if not hit:
            try:
                run_pid_main()
                print("  ✅ pid")
            except Exception as e_pid:
                print(f"    PID generation note: {e_pid}")
    pid_report = _load_json(DATA_DIR / "pid_report.json")
    master["steps"]["pid"] = {"figures": [str(FIG_DIR / "pid_overview.png"),
                                          str(FIG_DIR / "pid_detailed.png")],
                              "report": pid_report}

    # 15 Pulse-coupled co-deposition analytics extra
    print("\n[15/22] Pulse-coupled co-deposition analytics...")
    with cache.step("pulse_coupled_co_deposition", "models.co_deposition",
                    _step_outputs("pulse_coupled_co_deposition_report.json",
                                  ["pulse_coupled_co_deposition.png"])) as hit:
        if not hit:
            try:
                from models.co_deposition import build_phase3_model
                # Quick comparison DC vs PE vs PRE for hydroxide_suppression
                model = build_phase3_model(mechanism_fe_ni="hydroxide_suppression")
                j_avg = 50.0
                dc_res = model.run_at_current(j_avg)
                pe_res = model.run_at_current_pulsed(j_avg, j_avg*2, duty_cycle=0.5, waveform="pe")
                pre_res = model.run_at_current_pulsed(j_avg, j_avg*2, duty_cycle=0.5, waveform="pre")
                pccd_result = {
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
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
                ax = axes[0]
                x = np.arange(3)
                ax.bar(x - 0.15, fe_vals, width=0.3, label="Fe wt%", color="#1874b4")
                ax.bar(x + 0.15, ni_vals, width=0.3, label="Ni wt%", color="#d95f02")
                ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("wt%")
                ax.set_title("Alloy composition DC vs PE vs PRE (screening)"); ax.legend(); ax.grid(alpha=0.25)
                ax = axes[1]
                ax.bar(labels, pH_vals, color="#4daf4a")
                ax.set_title("Surface pH DC vs pulse-coupled (recovery)"); ax.set_ylabel("pH surf"); ax.grid(alpha=0.25)
                fig.suptitle("Pulse-coupled co-deposition: pH recovery reduces hydroxide suppression", fontweight="bold")
                fig.tight_layout()
                fig_path = FIG_DIR / "pulse_coupled_co_deposition.png"
                fig.savefig(fig_path, dpi=180)
                plt.close(fig)
                pccd_result["figure"] = str(fig_path)
                # Write a report so the cache can track it
                (DATA_DIR / "pulse_coupled_co_deposition_report.json").write_text(
                    json.dumps(pccd_result, indent=2, default=str))
                print(f"  ✅ pulse-coupled co-deposition: {fig_path}")
            except Exception as e:
                print(f"  ❌ pulse-coupled co-deposition: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["pulse_coupled_co_deposition"] = _load_json(
        DATA_DIR / "pulse_coupled_co_deposition_report.json")

    # 16 Pre-lab modeling suite (Speciation, Thermal Balance, Operating Window, DOE Matrix)
    print("\n[16/22] Pre-lab modeling suite (Speciation, Thermal, Operating Window, DOE Matrix)...")
    with cache.step("speciation", "models.run_speciation",
                    _step_outputs("speciation_report.json",
                                  ["speciation_profiles.png"])) as hit:
        if not hit:
            try:
                run_speciation_main()
                print("  ✅ speciation")
            except Exception as e:
                print(f"  ❌ speciation: {e}")
    master["steps"]["speciation"] = _load_json(DATA_DIR / "speciation_report.json")

    with cache.step("thermal_balance", "models.run_thermal_balance",
                    _step_outputs("thermal_balance_report.json",
                                  ["thermal_balance_profiles.png"])) as hit:
        if not hit:
            try:
                run_thermal_balance_main()
                print("  ✅ thermal_balance")
            except Exception as e:
                print(f"  ❌ thermal_balance: {e}")
    master["steps"]["thermal_balance"] = _load_json(DATA_DIR / "thermal_balance_report.json")

    with cache.step("operating_window", "models.run_operating_window",
                    _step_outputs("operating_window_report.json",
                                  ["operating_window_map.png"])) as hit:
        if not hit:
            try:
                run_operating_window_main()
                print("  ✅ operating_window")
            except Exception as e:
                print(f"  ❌ operating_window: {e}")
    master["steps"]["operating_window"] = _load_json(DATA_DIR / "operating_window_report.json")

    with cache.step("experimental_matrix", "models.run_experimental_matrix",
                    _step_outputs("experimental_matrix_report.json",
                                  ["doe_matrix_summary.png"])) as hit:
        if not hit:
            try:
                run_experimental_matrix_main()
                print("  ✅ experimental_matrix")
            except Exception as e:
                print(f"  ❌ experimental_matrix: {e}")
    master["steps"]["experimental_matrix"] = _load_json(DATA_DIR / "experimental_matrix_report.json")

    # 17 Cell architecture screen
    print("\n[17/22] Cell architecture screen (harvesting, $/m², kill criterion #3)...")
    with cache.step("cell_architecture", "models.run_cell_architecture",
                    _step_outputs("cell_architecture_report.json",
                                  ["cell_architecture_comparison.png",
                                   "cell_architecture_sweeps.png",
                                   "cell_architecture_tradeoff.png"])) as hit:
        if not hit:
            try:
                run_cell_architecture_main()
                print("  ✅ cell architecture")
            except Exception as e:
                print(f"  ❌ cell architecture: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["cell_architecture"] = _load_json(DATA_DIR / "cell_architecture_report.json")

    # 18 Adhesion & peel screen
    print("\n[18/22] Adhesion & peel screen (does iron peel from a drum?)...")
    with cache.step("adhesion_peel", "models.run_adhesion_peel",
                    _step_outputs("adhesion_peel_report.json",
                                  ["adhesion_substrate_screen.png",
                                   "adhesion_foil_window.png",
                                   "adhesion_robustness.png"])) as hit:
        if not hit:
            try:
                run_adhesion_peel_main()
                print("  ✅ adhesion & peel")
            except Exception as e:
                print(f"  ❌ adhesion & peel: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["adhesion_peel"] = _load_json(DATA_DIR / "adhesion_peel_report.json")

    # 19 Internal stress & coupon curvature
    print("\n[19/22] Internal stress & coupon curvature (Stoney / bent-strip)...")
    with cache.step("internal_stress", "models.run_internal_stress",
                    _step_outputs("internal_stress_report.json",
                                  ["internal_stress_mechanism_decomposition.png",
                                   "internal_stress_coupon_measurability.png",
                                   "internal_stress_evolution_and_peel.png"])) as hit:
        if not hit:
            try:
                run_internal_stress_main()
                print("  ✅ internal stress")
            except Exception as e:
                print(f"  ❌ internal stress: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["internal_stress"] = _load_json(DATA_DIR / "internal_stress_report.json")

    # 20 RDE kinetics/transport separation
    print("\n[20/22] RDE kinetics/transport separation (Levich + Koutecky-Levich)...")
    with cache.step("rde_levich", "models.run_rde_levich",
                    _step_outputs("rde_levich_report.json",
                                  ["rde_levich_levich_kl.png",
                                   "rde_levich_polarization.png",
                                   "rde_levich_tafel.png"])) as hit:
        if not hit:
            try:
                run_rde_levich_main()
                print("  ✅ rde & levich")
            except Exception as e:
                print(f"  ❌ rde & levich: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["rde_levich"] = _load_json(DATA_DIR / "rde_levich_report.json")

    # 21 Cathode-channel gas hold-up
    print("\n[21/22] Cathode-channel gas hold-up (void fraction, redistribution, H2 safety)...")
    with cache.step("gas_holdup", "models.run_gas_holdup",
                    _step_outputs("gas_holdup_report.json",
                                  ["gas_holdup_axial_profiles.png",
                                   "gas_holdup_fe_coupling.png",
                                   "gas_holdup_scaling_safety.png"]),
                    params={"quick": quick}) as hit:
        if not hit:
            try:
                run_gas_holdup_main(quick=quick)
                print("  ✅ gas hold-up")
            except Exception as e:
                print(f"  ❌ gas hold-up: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["gas_holdup"] = _load_json(DATA_DIR / "gas_holdup_report.json")

    # 22 Unified RC-1 reference-cell pipeline
    print("\n[22/22] Unified RC-1 reference-cell pipeline (physics + gas + thermal + safety)...")
    with cache.step("reference_cell_pipeline", "models.reference_cell_pipeline",
                    _step_outputs("reference_cell_pipeline_report.json"),
                    params={"quick": quick, "gas_segments": 4, "gas_iterations": 6}) as hit:
        if not hit:
            try:
                from models.reference_cell_pipeline import ReferenceCellPipeline

                pipeline = ReferenceCellPipeline(gas_segments=4, gas_iterations=6)
                pipeline_inputs = pipeline.default_inputs(
                    current_density_mA_cm2=100.0 if quick else None,
                )
                integrated_state = pipeline.simulate(pipeline_inputs)
                # Write report so the cache can track it
                (DATA_DIR / "reference_cell_pipeline_report.json").write_text(
                    json.dumps(integrated_state.to_dict(), indent=2, default=str))
                print("  ✅ reference_cell_pipeline")
            except Exception as e:
                print(f"  ❌ reference_cell_pipeline: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["reference_cell_pipeline"] = _load_json(
        DATA_DIR / "reference_cell_pipeline_report.json")

    # 23 Advanced Physics & Chemistry Suite (Round 3)
    print("\n[23/23] Round 3 Advanced Physics & Chemistry Suite...")
    with cache.step("physics_tranche3", "models.run_physics_tranche3",
                    _step_outputs("physics_tranche3_report.json")) as hit:
        if not hit:
            try:
                run_physics_tranche3()
                print("  ✅ physics_tranche3")
            except Exception as e:
                print(f"  ❌ physics_tranche3: {e}")
                import traceback; traceback.print_exc()
    master["steps"]["physics_tranche3"] = _load_json(DATA_DIR / "physics_tranche3_report.json")

    # Dashboard — always regenerate (cheap, depends on everything above)
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

    # Cache summary
    print("\n" + "=" * 72)
    print(f"✅ RUN_ALL complete — master report: {master_out}")
    print(f"   Dashboard: {FIG_DIR / 'run_all_dashboard.png'}")
    print(f"   {cache.summary()}")
    print("=" * 72)
    return master


def cli():
    """Entry point for aq-steel console script."""
    parser = argparse.ArgumentParser(description="Run all aqueous electrowinning models")
    parser.add_argument("--quick", action="store_true", help="Skip heavy grids (pulse comparisons)")
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "master_report.json"), help="Master report output path")
    parser.add_argument("--no-cache", action="store_true", help="Disable incremental step cache (force full recompute)")
    parser.add_argument("--force-step", action="append", default=[], metavar="NAME",
                        help="Force recompute of a specific step (may be repeated)")
    args = parser.parse_args()
    main(
        quick=args.quick,
        master_out=Path(args.out),
        cache_enabled=not args.no_cache,
        force_steps=set(args.force_step),
    )


if __name__ == "__main__":
    cli()
