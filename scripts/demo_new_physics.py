"""Demonstration of the four new chemistry/physics modules.

Run with: python -m scripts.demo_new_physics
(requires numpy + scipy for full functionality; this script is self-contained for the new modules)
"""

from models.mhd_convection import (
    compute_mhd_solution,
    effective_mass_transfer_with_mhd,
    measurement_protocol as mhd_protocol,
)
from models.sonoelectrochemistry import (
    compute_sonoelectro_result,
    effective_delta_with_ultrasound,
    measurement_protocol as us_protocol,
)
from models.bath_rheology import (
    BathRheologyParams,
    effective_viscosity_for_gas_holdup,
)
from models.eqcm_metrology import (
    simulate_eqcm_run,
)

print("=" * 70)
print("NEW PHYSICS MODULES DEMO — Aqueous Electrowinning Steel")
print("=" * 70)

# 1. MHD Convection
print("\n1. MHD CONVECTION (Lorentz-driven flow)")
sol = compute_mhd_solution(j_mean_mA_cm2=300.0, b_field_T=0.00025)
print(f"   B-field          : {sol.b_field_T*1000:.2f} mT (stray + Earth)")
print(f"   Lorentz velocity : {sol.lorentz_velocity_m_s*1000:.2f} mm/s")
print(f"   δ reduction      : {sol.effective_delta_reduction_factor:.3f}×")
print(f"   Notes            : {sol.notes}")

# 2. Sonoelectrochemistry
print("\n2. SONOELECTROCHEMISTRY (ultrasonic agitation)")
us = compute_sonoelectro_result()
print(f"   Streaming velocity : {us.acoustic_streaming_velocity_m_s*1000:.1f} mm/s")
print(f"   δ reduction        : {us.effective_delta_reduction_factor:.3f}×")
print(f"   Micro-jet velocity : {us.microjet_velocity_m_s:.1f} m/s")
print(f"   Degassing factor   : {us.degassing_factor:.2f}×")
print(f"   Cavitation active  : {us.cavitation_active}")

# 3. Bath Rheology
print("\n3. BATH RHEOLOGY (non-Newtonian viscosity)")
rheo = BathRheologyParams(
    yield_stress_Pa=0.08,
    consistency_index_Pa_s_n=0.0015,
    flow_index_n=0.82,
    phi_particles=0.04,
)
eta = effective_viscosity_for_gas_holdup(rheo, shear_rate_s=15.0)
print(f"   Yield stress     : {rheo.yield_stress_Pa:.3f} Pa")
print(f"   Effective η      : {eta*1000:.3f} mPa·s (at 15 s⁻¹)")
print(f"   Shear-thinning n : {rheo.flow_index_n}")

# 4. EQCM Metrology
print("\n4. EQCM METROLOGY (real-time mass + H inventory)")
eq = simulate_eqcm_run(
    charge_density_C_cm2=2160.0,   # ~30 min at 120 mA/cm²
    fe_efficiency=0.91,
    trapped_h_ppm=240.0,
    film_thickness_um=22.0,
)
print(f"   Mass gain        : {eq.mass_gain_ug_cm2:.1f} µg/cm²")
print(f"   Frequency shift  : {eq.frequency_shift_Hz:.0f} Hz")
print(f"   Trapped H (est)  : {eq.trapped_h_ppm:.0f} ppm")
print(f"   Viscoelastic corr: {eq.viscoelastic_correction:.2f}")

print("\n" + "=" * 70)
print("All modules are ready for integration into:")
print("  • diffusion_layer_1d.py (MHD + US + rheology δ_eff)")
print("  • gas_holdup.py (rheology + US degassing)")
print("  • reference_cell_pipeline.py / digital_twin.py (EQCM)")
print("  • pulse.py (US + MHD synergy)")
print("=" * 70)

print("\nMeasurement protocols available:")
print("  mhd_protocol()")
print("  us_protocol()")
print("\nRun the full suite with the new physics by importing the modules above.")