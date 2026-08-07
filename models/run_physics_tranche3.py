"""
Driver for the Round 3 advanced physical and chemical modeling suite.

Runs and generates a consolidated report for:
1. FeSO4 retrograde solubility and szomolnokite scaling (models/fe_sulfate_solubility.py)
2. Pulse RC double-layer filtering and cutoff frequency (models/pulse_rc_filter.py)
3. Bockris–Dražic–Despic (BDD) multi-step catalytic iron microkinetics (models/bdd_kinetics.py)
4. Multi-cell stack manifold ionic shunt currents (models/shunt_currents.py)
5. Hierarchical hydrogen trapping & McNabb–Foster bakeout optimization (models/hydrogen_trapping.py)
6. Primary ore leaching kinetics & reductive dissolution (models/ore_leaching.py)
7. Chemical osmosis and transmembrane water flux (models/chemical_osmosis.py)
8. 4-stage tempering metallurgy, LSW coarsening & Charpy DBTT (models/tempering_kinetics.py)
9. Solutal buoyancy, mixed convection, and flow reversal stability (models/solutal_convection.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from models.fe_sulfate_solubility import assess_heat_exchanger_scaling
from models.pulse_rc_filter import simulate_pulse_rc_response, max_practical_frequency_Hz
from models.bdd_kinetics import solve_bdd_kinetics
from models.shunt_currents import StackManifoldGeometry, solve_stack_shunt_currents
from models.hydrogen_trapping import compute_bakeout_schedule
from models.ore_leaching import OreSpec, simulate_ore_leaching
from models.chemical_osmosis import solve_transmembrane_water_flux
from models.tempering_kinetics import SteelMicrostructureSpec, simulate_tempering_kinetics
from models.solutal_convection import SolutalChannelParams, solve_solutal_mixed_convection

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "data"


def run_physics_tranche3() -> Dict[str, Any]:
    """Execute all 9 Round-3 physics workflows and return a structured summary report."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. FeSO4 retrograde scaling
    scaling_res = assess_heat_exchanger_scaling(
        bulk_temp_C=65.0,
        wall_temp_C=90.0,
        bulk_fe2_mol_L=1.50,
        background_sulfate_mol_L=0.50,
    )

    # 2. Pulse RC filter
    pulse_res = simulate_pulse_rc_response(
        frequency_Hz=500.0,
        duty_cycle=0.20,
        peak_current_mA_cm2=200.0,
    )
    max_freq = max_practical_frequency_Hz(duty_cycle=0.20, min_fidelity_ratio=0.85)

    # 3. BDD microkinetics
    bdd_res = solve_bdd_kinetics(overpotential_V=0.150, ph=2.5, fe2_mol_L=1.5)

    # 4. Stack shunt currents
    geom = StackManifoldGeometry(n_cells=50, cell_voltage_V=2.60, applied_current_A=300.0)
    shunt_res = solve_stack_shunt_currents(geom)

    # 5. Hydrogen trapping & bakeout
    bakeout_res = compute_bakeout_schedule(
        foil_thickness_um=100.0,
        total_initial_H_ppm_wt=5.0,
        bake_temperature_C=190.0,
    )

    # 6. Ore leaching
    ore = OreSpec(mineral_type="hematite", particle_p80_um=75.0)
    leach_direct = simulate_ore_leaching(ore, temperature_C=80.0, residence_time_hours=4.0, use_reductant=False)
    leach_reductive = simulate_ore_leaching(ore, temperature_C=80.0, residence_time_hours=4.0, use_reductant=True)

    # 7. Chemical osmosis
    osmosis_res = solve_transmembrane_water_flux(current_density_mA_cm2=200.0, water_activity_catholyte=0.925, water_activity_anolyte=0.965)

    # 8. Tempering kinetics & LSW coarsening
    steel_spec = SteelMicrostructureSpec(carbon_wt_percent=0.40)
    temper_res = simulate_tempering_kinetics(steel_spec, temperature_C=550.0, time_hours=2.0)

    # 9. Solutal mixed convection
    solutal_params = SolutalChannelParams(cathode_height_m=1.0)
    solutal_up = solve_solutal_mixed_convection(forced_velocity_m_s=0.15, flow_direction="upward", params=solutal_params)
    solutal_down = solve_solutal_mixed_convection(forced_velocity_m_s=0.04, flow_direction="downward", params=solutal_params)

    report = {
        "title": "Round 3 Advanced Physical & Chemical Modeling Report",
        "solubility_and_scaling": {
            "wall_supersaturation_ratio": round(scaling_res.supersaturation_ratio_wall, 3),
            "max_safe_wall_temp_C": round(scaling_res.max_safe_wall_temp_C, 1),
            "stable_solid_phase_wall": scaling_res.stable_phase_wall,
            "scaling_risk": scaling_res.critical_heat_flux_margin,
        },
        "pulse_rc_filtering": {
            "peak_attenuation_ratio": round(pulse_res.peak_attenuation_ratio, 3),
            "actual_peak_faradaic_mA_cm2": round(pulse_res.actual_peak_faradaic_mA_cm2, 1),
            "waveform_fidelity": pulse_res.waveform_fidelity,
            "max_practical_frequency_Hz": round(max_freq, 1),
        },
        "bdd_microkinetics": {
            "intermediate_coverage_theta": round(bdd_res.intermediate_coverage_theta, 3),
            "cathodic_current_density_mA_cm2": round(bdd_res.cathodic_current_density_A_m2 / 10.0, 2),
            "apparent_tafel_slope_mV_dec": round(bdd_res.apparent_tafel_slope_mV_dec, 1),
            "reaction_order_oh_minus": round(bdd_res.reaction_order_oh_minus, 3),
            "rate_determining_step": bdd_res.rate_determining_step,
        },
        "stack_shunt_currents": {
            "stack_voltage_V": shunt_res.stack_voltage_V,
            "total_shunt_current_A": round(shunt_res.total_shunt_current_A, 2),
            "faradaic_efficiency_loss_percent": round(shunt_res.stack_faradaic_efficiency_loss_percent, 4),
            "max_port_current_density_mA_cm2": round(shunt_res.max_port_current_density_mA_cm2, 1),
            "is_corrosion_threat": shunt_res.is_corrosion_threat,
            "recommended_port_length_m": round(shunt_res.recommended_min_port_length_m, 3),
        },
        "hydrogen_trapping_and_bakeout": {
            "foil_thickness_um": bakeout_res.foil_thickness_um,
            "bake_temperature_C": bakeout_res.bake_temperature_C,
            "required_bake_time_hours": round(bakeout_res.required_bake_time_hours, 2),
            "astm_f519_compliance": bakeout_res.astm_f519_compliance,
        },
        "ore_leaching": {
            "mineral": ore.mineral_type,
            "direct_acid_recovery_4h": round(leach_direct.fe_recovery_fraction * 100.0, 1),
            "reductive_recovery_4h": round(leach_reductive.fe_recovery_fraction * 100.0, 1),
            "reductive_fe2_ratio": round(leach_reductive.fe2_to_fe3_product_ratio, 2),
        },
        "chemical_osmosis": {
            "net_water_flux_L_m2_h": round(osmosis_res.net_water_flux_L_m2_h, 3),
            "zero_net_flux_current_density_mA_cm2": round(osmosis_res.zero_net_flux_current_density_mA_cm2, 1),
            "transport_regime": osmosis_res.transport_regime,
        },
        "tempering_lsw_kinetics": {
            "tempering_stage": temper_res.tempering_stage,
            "mean_carbide_radius_nm": round(temper_res.mean_carbide_radius_nm, 1),
            "yield_strength_MPa": round(temper_res.estimated_yield_strength_MPa, 0),
            "charpy_energy_J": round(temper_res.estimated_charpy_energy_J, 1),
            "dbtt_C": round(temper_res.dbtt_C, 1),
        },
        "solutal_mixed_convection": {
            "density_depletion_kg_m3": round(solutal_up.density_depletion_kg_m3, 1),
            "grashof_number_Gr_H": f"{solutal_up.grashof_number_Gr_H:.2e}",
            "upward_effective_boundary_layer_um": round(solutal_up.effective_boundary_layer_um, 1),
            "downward_effective_boundary_layer_um": round(solutal_down.effective_boundary_layer_um, 1),
            "critical_antireversal_velocity_m_s": round(solutal_down.critical_antireversal_velocity_m_s, 3),
            "is_downward_reversal_threat": solutal_down.is_flow_reversal_threat,
        },
    }

    out_file = DATA_DIR / "physics_tranche3_report.json"
    out_file.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    """CLI driver for physics tranche 3."""
    print("=" * 72)
    print("Running Round 3 Advanced Physics & Chemistry Workflows")
    print("=" * 72)
    rep = run_physics_tranche3()
    print(f"✅ Generated {DATA_DIR / 'physics_tranche3_report.json'}")
    for section, data in rep.items():
        if section != "title":
            print(f"\n[{section}]")
            for k, v in data.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
