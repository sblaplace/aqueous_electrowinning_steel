"""Tests for hydrogen embrittlement screening model."""

import numpy as np
import pytest
from models.hydrogen_embrittlement import (
    HydrogenEmbrittlementModel,
    TrapSiteParams,
    h_diffusivity_m2_s,
    effective_diffusivity_m2_s,
    hydrogen_uptake_from_electrolysis,
    he_susceptibility_index,
    bakeout_time_hr,
    bakeout_schedule,
    trap_binding_factor,
    trap_density_m3,
    build_he_model_from_mechanical,
    synthetic_h_uptake_data,
    ipz_hydrogen_entry,
    ipz_parameters_from_permeation,
    IPZ_K_ABS_DEFAULT,
)


# ── Test 1: D_H increases with T ────────────────────────────────────────────


def test_h_diffusivity_increases_with_temperature():
    """D_H should increase monotonically with temperature for both phases."""
    temps = [25, 50, 100, 200, 500]
    D_alpha = [h_diffusivity_m2_s(T, "alpha")[0] for T in temps]
    D_gamma = [h_diffusivity_m2_s(T, "gamma")[0] for T in temps]

    # Alpha (bcc) should increase with T
    for i in range(1, len(D_alpha)):
        assert D_alpha[i] > D_alpha[i - 1], f"D_alpha not increasing at T={temps[i]}"

    # Gamma (fcc) should also increase with T
    for i in range(1, len(D_gamma)):
        assert D_gamma[i] > D_gamma[i - 1], f"D_gamma not increasing at T={temps[i]}"

    # Alpha is much faster than gamma at same T (below A3)
    D_a_25, _ = h_diffusivity_m2_s(25, "alpha")
    D_g_25, _ = h_diffusivity_m2_s(25, "gamma")
    assert D_a_25 > D_g_25, "α-Fe H diffusivity should be faster than γ-Fe at 25°C"

    # Physical reasonableness: D at 25°C for α-Fe should be ~1e-9 to 1e-7 m²/s
    assert 1e-12 < D_a_25 < 1e-5, f"D_alpha at 25°C unreasonable: {D_a_25}"


def test_h_diffusivity_auto_phase():
    """Auto phase selection should pick alpha below threshold, gamma above."""
    D_low, phase_low = h_diffusivity_m2_s(50, "auto")
    D_high, phase_high = h_diffusivity_m2_s(800, "auto")
    assert phase_low == "alpha"
    assert phase_high == "gamma"


# ── Test 2: Trap density reduces effective diffusivity ──────────────────────


def test_trap_density_reduces_effective_diffusivity():
    """Higher trap density should reduce D_eff below D_lattice."""
    # No traps (artificially)
    p_no_trap = TrapSiteParams(
        N_dislocation_m2=0, N_gb_per_um=0, N_carbide_per_wt_C=0
    )
    D_no_trap, D_lat_no, info_no = effective_diffusivity_m2_s(
        25.0, grain_size_um=3.0, carbon_wt_percent=0.0, params_trap=p_no_trap
    )
    assert abs(D_no_trap - D_lat_no) < 1e-15, "D_eff should equal D_lattice with no traps"

    # With standard traps
    D_eff, D_lat, info = effective_diffusivity_m2_s(
        25.0, grain_size_um=1.0, carbon_wt_percent=1.0
    )
    assert D_eff < D_lat, "D_eff must be less than D_lattice with traps present"
    assert D_eff > 0, "D_eff must be positive"

    # More traps (smaller grain + more C) → even lower D
    D_eff_fine, _, _ = effective_diffusivity_m2_s(
        25.0, grain_size_um=0.3, carbon_wt_percent=2.0
    )
    assert D_eff_fine < D_eff, "Finer grains + more C should further reduce D_eff"


def test_trap_binding_factor_increases_with_binding_energy():
    """Higher binding energy → larger K_t → stronger trap."""
    K_low = trap_binding_factor(11.0, 25.0)   # carbide
    K_med = trap_binding_factor(20.0, 25.0)    # GB
    K_high = trap_binding_factor(26.0, 25.0)   # dislocation
    assert K_low < K_med < K_high


def test_trap_binding_factor_decreases_with_temperature():
    """At higher T, traps release H more easily → lower K_t."""
    K_cold = trap_binding_factor(26.0, 25.0)
    K_hot = trap_binding_factor(26.0, 200.0)
    assert K_hot < K_cold, "Trap binding factor should decrease with temperature"


# ── Test 3: HE index increases with strength and H content ──────────────────


def test_he_index_increases_with_strength():
    """Higher yield strength → higher HE susceptibility."""
    he_low = he_susceptibility_index(300.0, 0.1)["I_HE"]
    he_high = he_susceptibility_index(1000.0, 0.1)["I_HE"]
    assert he_high > he_low


def test_he_index_increases_with_h_content():
    """Higher diffusible H → higher HE susceptibility."""
    he_low_H = he_susceptibility_index(600.0, 0.01)["I_HE"]
    he_high_H = he_susceptibility_index(600.0, 1.0)["I_HE"]
    assert he_high_H > he_low_H


def test_he_index_risk_levels():
    """Risk classification should work."""
    # Low: low strength, low H
    result = he_susceptibility_index(200.0, 0.001)
    assert result["risk_level"] == "low"

    # High / critical: high strength, high H
    result = he_susceptibility_index(1200.0, 10.0)
    assert result["risk_level"] in ("high", "critical")


# ── Test 4: Bake-out time decreases with temperature ────────────────────────


def test_bakeout_time_decreases_with_temperature():
    """Higher bake-out temperature → faster H removal → shorter time."""
    t_cold = bakeout_time_hr(1000, 0.5, 0.1, 120.0)["bakeout_time_hr"]
    t_hot = bakeout_time_hr(1000, 0.5, 0.1, 250.0)["bakeout_time_hr"]
    assert t_hot < t_cold, "Bake-out time should decrease with temperature"
    assert t_hot > 0, "Bake-out time must be positive"


def test_bakeout_time_decreases_with_thickness():
    """Thinner deposits → shorter bake-out (shorter diffusion path)."""
    t_thick = bakeout_time_hr(3000, 0.5, 0.1, 170.0)["bakeout_time_hr"]
    t_thin = bakeout_time_hr(500, 0.5, 0.1, 170.0)["bakeout_time_hr"]
    assert t_thin < t_thick


def test_bakeout_zero_when_already_below_target():
    """If initial H is already below target, bake-out time should be 0."""
    result = bakeout_time_hr(1000, 0.05, 0.1, 170.0)
    assert result["bakeout_time_hr"] == 0.0


def test_bakeout_schedule():
    """Bake-out schedule should return list of results for multiple temps."""
    schedule = bakeout_schedule(
        deposit_thickness_um=1000, initial_C_H_ppm=0.5, target_C_H_ppm=0.1
    )
    assert len(schedule) >= 3
    # Times should generally decrease with temperature
    times = [r["bakeout_time_hr"] for r in schedule]
    assert times[-1] < times[0], "Last temp should give shorter bake-out than first"


# ── Test 5: Integration with mechanical_properties output ───────────────────


def test_integration_with_mechanical_properties():
    """HE model should accept mechanical_properties summary dict."""
    # Simulate a mechanical result summary
    fake_mech = {
        "yield_strength_MPa": 500.0,
        "grain_size_um": 2.0,
        "composition": {"ni_wt_pct": 2.0, "c_wt_pct": 0.5},
    }

    model = HydrogenEmbrittlementModel()
    result = model.predict_with_integration(
        mechanical_result=fake_mech,
        current_density_mA_cm2=100.0,
        deposition_time_hr=2.0,
        bath_pH=3.5,
    )

    assert result.sigma_y_MPa == 500.0
    assert result.grain_size_um == 2.0
    assert result.ni_wt_percent == 2.0
    assert result.carbon_wt_percent == 0.5
    assert result.he_index["I_HE"] > 0
    assert result.bakeout["bakeout_time_hr"] >= 0


def test_build_he_model_from_mechanical():
    """Convenience adapter should work."""
    fake_mech = {
        "yield_strength_MPa": 400.0,
        "grain_size_um": 3.0,
        "composition": {"ni_wt_pct": 1.0, "c_wt_pct": 0.3},
    }
    result = build_he_model_from_mechanical(fake_mech, current_density_mA_cm2=80.0)
    assert result.sigma_y_MPa == 400.0
    assert result.he_index["I_HE"] > 0


# ── Additional tests ────────────────────────────────────────────────────────


def test_full_model_predict():
    """Full prediction pipeline should return reasonable values."""
    model = HydrogenEmbrittlementModel()
    result = model.predict(
        current_density_mA_cm2=100.0,
        deposition_time_hr=2.0,
        bath_pH=3.5,
        sigma_y_MPa=450.0,
        grain_size_um=2.0,
        carbon_wt_percent=0.5,
    )

    # H uptake should be positive
    assert result.uptake["C_H_diffusible_ppm"] > 0

    # Effective diffusivity should be positive and < lattice
    assert result.D_eff_m2_s > 0
    assert result.D_eff_m2_s <= result.D_lattice_m2_s

    # HE index should be positive
    assert result.he_index["I_HE"] > 0

    # Summary should be a dict
    s = result.summary()
    assert "I_HE" in s
    assert "risk_level" in s


def test_h_uptake_from_electrolysis():
    """H uptake: HER current rises with j and H entry is positive.

    At a fixed HER *fraction* the H-in-Fe concentration (ppm) falls as j
    rises, because Fe mass grows linearly with j while the IPZ entry flux
    grows only as √θ ∝ √j_HER — the physically correct 'higher current
    density → less diffusible H per kg iron' trend.  Assert the absolute
    H uptake (mol/m²) and HER current scale with j, and use the empirical
    model for the ppm-monotonic-in-j check the original test encoded.
    """
    up_low = hydrogen_uptake_from_electrolysis(50.0)
    up_high = hydrogen_uptake_from_electrolysis(200.0)
    assert up_high["her_current_A_m2"] > up_low["her_current_A_m2"]
    assert up_high["H_absorbed_mol_m2"] > up_low["H_absorbed_mol_m2"]
    assert up_low["absorption_fraction"] > 0
    # Empirical model retains the positive ppm-vs-j trend at fixed her_eff.
    emp_low = hydrogen_uptake_from_electrolysis(50.0, model="empirical")
    emp_high = hydrogen_uptake_from_electrolysis(200.0, model="empirical")
    assert emp_high["C_H_diffusible_ppm"] > emp_low["C_H_diffusible_ppm"]


def test_h_uptake_pH_effect():
    """Empirical model: lower pH → higher H uptake (more H⁺ available).

    The default IPZ model sets θ from the HER gas rate alone (at fixed
    j_HER), so pH shifts the required overpotential rather than the entry
    flux; the pH sensitivity the original test encoded belongs to the
    empirical correlation, which is exercised explicitly here.
    """
    up_low_pH = hydrogen_uptake_from_electrolysis(
        100.0, bath_pH=2.0, model="empirical"
    )
    up_high_pH = hydrogen_uptake_from_electrolysis(
        100.0, bath_pH=5.0, model="empirical"
    )
    assert up_low_pH["C_H_diffusible_ppm"] > up_high_pH["C_H_diffusible_ppm"]


# ── IPZ (Iyer–Pickering–Zamanzadeh) H-entry model ──────────────────────────


def test_ipz_theta_in_0_to_1():
    """Surface H coverage must remain a physical sub-monolayer fraction."""
    for j_her in [10.0, 100.0, 1000.0, 5000.0]:
        r = ipz_hydrogen_entry(j_her, bath_pH=3.0, temperature_C=60.0)
        assert 0.0 < r["theta_H"] <= 1.0
        assert r["eta_V"] > 0.0
        assert r["entry_flux_mol_m2_s"] > 0.0


def test_ipz_entry_efficiency_falls_with_her():
    """More HER gas → more recombination → a smaller fraction enters metal."""
    low = ipz_hydrogen_entry(100.0, bath_pH=3.0)
    high = ipz_hydrogen_entry(5000.0, bath_pH=3.0)
    assert high["entry_efficiency"] < low["entry_efficiency"]
    # But the absolute entry flux still rises with HER (more H around).
    assert high["entry_flux_mol_m2_s"] > low["entry_flux_mol_m2_s"]


def test_ipz_default_uptake_is_physically_scaled():
    """Default IPZ uptake should give a physically reasonable H level."""
    r = hydrogen_uptake_from_electrolysis(
        100.0, deposition_time_s=900, her_efficiency=0.15, bath_pH=3.0
    )
    assert r["model"] == "ipz"
    assert 0.0 < r["absorption_fraction"] < 0.15
    assert 0.1 < r["C_H_diffusible_ppm"] < 500.0
    assert "ipz" in r and 0.0 < r["ipz"]["theta_H"] <= 1.0


def test_ipz_zero_her_gives_zero_entry():
    """No HER means no hydrogen to absorb."""
    r = hydrogen_uptake_from_electrolysis(
        100.0, her_efficiency=0.0, bath_pH=3.0
    )
    assert r["C_H_diffusible_ppm"] == 0.0
    assert r["absorption_fraction"] == 0.0


def test_ipz_permeation_inversion_round_trips():
    """Recovering k_abs from a synthetic permeation flux should be exact."""
    j_her = 1000.0
    base = ipz_hydrogen_entry(j_her, bath_pH=3.0, k_abs=IPZ_K_ABS_DEFAULT)
    j_perm = base["entry_flux_mol_m2_s"] * 96485.3
    inv = ipz_parameters_from_permeation(j_perm, j_her, bath_pH=3.0)
    assert inv["k_abs"] == pytest.approx(IPZ_K_ABS_DEFAULT, rel=1e-6)
    assert 0.0 < inv["theta_H"] <= 1.0


def test_empirical_model_retained():
    """The empirical correlation must remain available and marked."""
    r = hydrogen_uptake_from_electrolysis(
        100.0, bath_pH=3.5, model="empirical"
    )
    assert r["model"] == "empirical"
    assert r["C_H_diffusible_ppm"] > 0



def test_synthetic_h_uptake_data():
    """Synthetic data generator should return arrays of consistent length."""
    syn = synthetic_h_uptake_data()
    assert len(syn["j_mA_cm2"]) == len(syn["H_vs_j_ppm"])
    assert len(syn["T_C"]) == len(syn["H_vs_T_ppm"])
    assert len(syn["pH"]) == len(syn["H_vs_pH_ppm"])
    assert np.all(syn["H_vs_j_ppm"] > 0)


def test_model_flags():
    """Model should flag critical conditions."""
    model = HydrogenEmbrittlementModel()
    # High strength + high H → critical HE risk
    result = model.predict(
        current_density_mA_cm2=200.0,
        deposition_time_hr=5.0,
        bath_pH=2.0,
        sigma_y_MPa=1000.0,
    )
    # At least one flag should trigger
    assert len(result.flags) > 0


def test_trap_density_components():
    """Trap density should have all components."""
    traps = trap_density_m3(grain_size_um=2.0, dislocation_density_m2=1e14, carbon_wt_percent=1.0)
    assert traps["dislocation_m3"] > 0
    assert traps["grain_boundary_m3"] > 0
    assert traps["carbide_m3"] > 0
    assert traps["total_m3"] > 0
    assert traps["total_m3"] == (
        traps["dislocation_m3"] + traps["grain_boundary_m3"] + traps["carbide_m3"]
    )


def test_integration_with_carburization():
    """Spatially-resolved HE risk with carburization profile."""
    fake_mech = {
        "yield_strength_MPa": 500.0,
        "grain_size_um": 2.0,
        "composition": {"ni_wt_pct": 2.0, "c_wt_pct": 0.8},
    }
    fake_carb = {
        "final_case_depth_035_um": 350.0,
        "final_surface_hv": 700.0,
    }

    model = HydrogenEmbrittlementModel()
    result = model.predict_with_integration(
        mechanical_result=fake_mech,
        carburization_result=fake_carb,
        current_density_mA_cm2=100.0,
    )

    assert result.spatial_he_risk is not None
    spatial = result.spatial_he_risk
    assert "surface_I_HE" in spatial
    assert "core_I_HE" in spatial
    # Surface (harder from carburization) should have higher HE risk than core
    assert spatial["surface_I_HE"] >= spatial["core_I_HE"]


def test_recombination_poison_wiring_raises_h():
    """Passing poison concentrations must raise absorbed-H and C_H_ppm (Round 5 B1)."""
    base = hydrogen_uptake_from_electrolysis(
        100.0, deposition_time_s=3600.0, her_efficiency=0.10, bath_pH=3.5,
        model="ipz",
    )
    poisoned = hydrogen_uptake_from_electrolysis(
        100.0, deposition_time_s=3600.0, her_efficiency=0.10, bath_pH=3.5,
        model="ipz", poison_concentrations_M={"sulfide": 1e-3, "arsenic": 5e-6},
        cathodic_overpotential_V=0.3,
    )
    assert poisoned["C_H_diffusible_ppm"] > base["C_H_diffusible_ppm"]
    assert poisoned["recombination_poison"]["promotion_factor"] > 1.0
    assert poisoned["recombination_poison"]["poisoned_absorption_fraction"] > 0.0


def test_recombination_poison_noop_without_concentrations():
    """Without poison concentrations, result has no poison key and H is unchanged."""
    res = hydrogen_uptake_from_electrolysis(
        100.0, deposition_time_s=3600.0, her_efficiency=0.10, bath_pH=3.5,
        model="ipz",
    )
    assert "recombination_poison" not in res
