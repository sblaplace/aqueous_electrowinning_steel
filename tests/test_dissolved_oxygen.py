"""
Unit tests for dissolved oxygen solubility, ORR, and homogeneous Fe(II) kinetics.
"""

from models.dissolved_oxygen import (
    DissolvedOxygenParams,
    pure_water_o2_saturation_M,
    dissolved_oxygen_solubility_M,
    cathodic_orr_limiting_current_A_m2,
    homogeneous_fe2_oxidation_rate_M_s,
    analyze_oxygen_ingress_control,
)


def test_pure_water_o2_saturation():
    """Verify dissolved O2 saturation in pure water decreases with temperature."""
    sat_25 = pure_water_o2_saturation_M(25.0)
    sat_60 = pure_water_o2_saturation_M(60.0)
    sat_80 = pure_water_o2_saturation_M(80.0)

    # 1. Physical bounds
    assert 1e-5 <= sat_25 <= 1e-3  # Typically ~0.25 mM
    # 2. Retrograde temperature trend: gas solubility decreases with T
    assert sat_25 > sat_60 > sat_80


def test_dissolved_oxygen_solubility_salinity():
    """Verify Sechenov salting-out effect reduces solubility in salt solutions."""
    # 25 °C saturation in pure water vs. 2.5 M salt
    sol_pure = dissolved_oxygen_solubility_M(25.0, ionic_strength_M=0.0)
    sol_salt = dissolved_oxygen_solubility_M(25.0, ionic_strength_M=2.5)

    assert sol_salt < sol_pure
    assert sol_salt > 0.0


def test_cathodic_orr_limiting_current():
    """Verify ORR cathodic limiting current density scales with DO concentration."""
    params = DissolvedOxygenParams(temperature_C=60.0, delta_um=100.0)

    j_lim_sat = cathodic_orr_limiting_current_A_m2(params, do_fraction_sat=1.0)
    j_lim_half = cathodic_orr_limiting_current_A_m2(params, do_fraction_sat=0.5)
    j_lim_zero = cathodic_orr_limiting_current_A_m2(params, do_fraction_sat=0.0)

    assert j_lim_zero == 0.0
    assert j_lim_sat > j_lim_half > 0.0
    # ORR at these conditions is typically on the order of ~1-10 A/m² (0.1 - 1.0 mA/cm²)
    assert 0.1 <= j_lim_sat <= 50.0


def test_homogeneous_fe2_oxidation_rate():
    """Verify homogeneous ferrous chemical oxidation is accelerated by pH and T."""
    # Test pH dependence: higher pH means higher [OH-], which catalyzes Fe(II) oxidation
    params_ph2 = DissolvedOxygenParams(temperature_C=25.0, pH=2.0)
    params_ph3 = DissolvedOxygenParams(temperature_C=25.0, pH=3.0)

    rate_ph2 = homogeneous_fe2_oxidation_rate_M_s(params_ph2, do_fraction_sat=1.0)
    rate_ph3 = homogeneous_fe2_oxidation_rate_M_s(params_ph3, do_fraction_sat=1.0)

    assert rate_ph3 > rate_ph2
    assert rate_ph2 >= 0.0

    # Test T dependence (Arrhenius)
    params_t25 = DissolvedOxygenParams(temperature_C=25.0, pH=3.0)
    params_t60 = DissolvedOxygenParams(temperature_C=60.0, pH=3.0)

    rate_t25 = homogeneous_fe2_oxidation_rate_M_s(params_t25, do_fraction_sat=1.0)
    rate_t60 = homogeneous_fe2_oxidation_rate_M_s(params_t60, do_fraction_sat=1.0)

    assert rate_t60 > rate_t25


def test_analyze_oxygen_ingress_control():
    """Verify nitrogen sweeps successfully reduce steady-state DO and penalties."""
    params = DissolvedOxygenParams(temperature_C=60.0, pH=2.5)

    # Air ingress with no nitrogen sweep vs. sweep
    analysis_no_n2 = analyze_oxygen_ingress_control(params, nitrogen_flow_L_min=0.0)
    analysis_with_n2 = analyze_oxygen_ingress_control(params, nitrogen_flow_L_min=2.0)

    assert analysis_with_n2.equilibrium_pO2_pct < analysis_no_n2.equilibrium_pO2_pct
    assert analysis_with_n2.steady_state_DO_mM < analysis_no_n2.steady_state_DO_mM
    assert analysis_with_n2.orr_penalty_A_m2 < analysis_no_n2.orr_penalty_A_m2
    assert analysis_with_n2.fe3_generation_ppm_hr < analysis_no_n2.fe3_generation_ppm_hr
