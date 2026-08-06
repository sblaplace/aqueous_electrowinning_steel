"""Assertions for the Fe³⁺ shuttle / O₂ bath-aging screening model (L0)."""
from math import isfinite
import pytest
from models.bath_startup import dissolved_o2_saturation_mol_L
from models.electrochemistry import FARADAY
from models.fe3_shuttle import (
    ShuttleParams,
    anolyte_crossover_fault,
    ce_penalty_at_j,
    fe3_solubility_cap_M,
    open_headspace,
    scenario_table,
    sealed_divided_cell,
    steady_state,
)
RC1 = ShuttleParams(pH=2.35, temperature_C=50.0)
def test_solubility_cap_matches_feoh3_window():
    """pH 2–3: Fe³⁺ caps between ~2 mM and ~2 µM (Ksp = 10^-38.7)."""
    assert fe3_solubility_cap_M(2.0) == pytest.approx(2.00e-3, rel=0.01)
    assert fe3_solubility_cap_M(3.0) == pytest.approx(2.00e-6, rel=0.01)
    # Steep hydrolysis: 3 decades per pH unit.
    assert fe3_solubility_cap_M(2.0) / fe3_solubility_cap_M(3.0) == pytest.approx(1e3)
def test_sealed_cell_shuttle_is_negligible_on_ce():
    """An intact divided cell leaks ~0 CE to the shuttle at j=300."""
    ss = steady_state(RC1, sealed_divided_cell())
    assert ss["i_shuttle_A_m2"] < 0.5  # A/m², ≪ j=3000 A/m²
    loss = ce_penalty_at_j(ss["i_shuttle_A_m2"], 300.0)
    assert loss < 1e-3  # <0.1% even with trace ingress
def test_open_headspace_sits_at_the_hydrolysis_cap():
    """Air-exposed RC-1 bath: Fe³⁺ accumulation is capped by Fe(OH)₃."""
    ss = steady_state(RC1, open_headspace())
    assert ss["feoh3_precipitation_active"]
    assert ss["fe3_ss_M"] == pytest.approx(ss["fe3_solubility_cap_M"])
    # Even cap-pinned, the shuttle current is a small CE leak at j=300.
    assert ce_penalty_at_j(ss["i_shuttle_A_m2"], 300.0) < 1e-3
def test_scenario_production_rate_ordering():
    """sealed ≪ open headspace ≪ membrane-fault (production side)."""
    sealed = steady_state(RC1, sealed_divided_cell())["fe3_production_M_s"]
    open_ = steady_state(RC1, open_headspace())["fe3_production_M_s"]
    fault = steady_state(RC1, anolyte_crossover_fault(300.0, 0.01))["fe3_production_M_s"]
    assert sealed < open_ < fault
    assert open_ / sealed > 10.0
    assert fault / open_ > 3.0
def test_shuttle_current_is_mass_transfer_limited_reduction():
    """i_shuttle = F·k_m·[Fe³⁺] (1 e⁻ per Fe³⁺)."""
    p = ShuttleParams()
    s = open_headspace()
    ss = steady_state(p, s)
    km = p.d_fe3_m2_s / p.boundary_layer_m
    assert ss["i_shuttle_A_m2"] == pytest.approx(
        FARADAY * km * ss["fe3_ss_M"] * 1000.0, rel=1e-12
    )
def test_shuttle_bounded_by_production_times_V_over_A():
    """Steady-state identity: i_sh ≤ F·(V/A)·r_prod (equality when no sludge)."""
    p = ShuttleParams(pH=1.5)  # cap moves up by ~8x vs pH 2 → no sludge in this cell
    s = open_headspace()
    ss = steady_state(p, s)
    v_over_a = (p.catholyte_volume_L / 1000.0) / p.cathode_area_m2
    # r_prod is mol/L/s; convert to mol/m³/s (×1000) for the mol/m³·m identity.
    bound = FARADAY * v_over_a * ss["fe3_production_M_s"] * 1000.0
    assert ss["i_shuttle_A_m2"] <= bound * (1.0 + 1e-9)
    if not ss["feoh3_precipitation_active"]:
        assert ss["i_shuttle_A_m2"] == pytest.approx(bound, rel=1e-9)
def test_sludge_only_when_cap_exceeded_and_conserves_mass():
    p = ShuttleParams(pH=2.35, temperature_C=50.0)
    ss = steady_state(p, open_headspace())
    assert ss["feoh3_precipitation_active"]
    # sludge sink (mol/L/s) + shuttle sink = production
    area_per_vol = p.cathode_area_m2 / (p.catholyte_volume_L / 1000.0)
    km = p.d_fe3_m2_s / p.boundary_layer_m
    shuttle_sink = km * area_per_vol * ss["fe3_ss_M"]
    sludge = ss["fe3_production_M_s"] - shuttle_sink
    assert ss["iron_sludge_loss_g_L_day"] == pytest.approx(
        sludge * 55.845 * 86400.0, rel=1e-9
    )
    assert ss["iron_sludge_loss_g_L_day"] > 0.0
def test_ce_penalty_scales_inverse_with_j():
    i_sh = steady_state(RC1, open_headspace())["i_shuttle_A_m2"]
    assert ce_penalty_at_j(i_sh, 600.0) == pytest.approx(
        ce_penalty_at_j(i_sh, 300.0) / 2.0, rel=1e-12
    )
def test_crossover_flux_scales_with_anode_current():
    f1 = anolyte_crossover_fault(300.0, 0.01)
    f2 = anolyte_crossover_fault(600.0, 0.01)
    assert f2.crossover_o2_flux_mol_m2_s == pytest.approx(
        2.0 * f1.crossover_o2_flux_mol_m2_s, rel=1e-12
    )
def test_temperature_trends_in_the_expected_direction():
    """Hotter bath: lower O₂ solubility but much faster oxidation kinetics —
    the kinetic term wins, so production rises with T (screening law)."""
    r25 = steady_state(ShuttleParams(pH=2.35, temperature_C=25.0),
                       open_headspace())["fe3_production_M_s"]
    r60 = steady_state(ShuttleParams(pH=2.35, temperature_C=60.0),
                       open_headspace())["fe3_production_M_s"]
    assert dissolved_o2_saturation_mol_L(60.0) < dissolved_o2_saturation_mol_L(25.0)
    assert r60 > 2.0 * r25
def test_scenario_table_is_finite_and_flagged():
    table = scenario_table(ShuttleParams(pH=2.35), j_mA_cm2=300.0)
    assert table["flag"] == "unvalidated (L0)"
    assert len(table["rows"]) == 3
    for row in table["rows"]:
        assert isfinite(row["i_shuttle_A_m2"]) and row["i_shuttle_A_m2"] >= 0.0
        assert isfinite(row["ce_loss_fraction_at_j"]) and row["ce_loss_fraction_at_j"] >= 0.0
