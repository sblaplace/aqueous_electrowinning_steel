"""Unit tests for the electrochemistry, Pourbaix, and kinetics modules."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.electrochemistry import (  # noqa: E402
    FARADAY,
    CellVoltageModel,
    current_density_to_production,
    production_rate_kg_per_hr,
    specific_energy_kWh_per_kg,
    specific_energy_kWh_per_t,
)
from models.kinetics import DepositionKinetics, limiting_current_density  # noqa: E402
from models.pourbaix import FePourbaix, her_line, oer_line  # noqa: E402


# ─── Faraday's law ─────────────────────────────────────────────────────
def test_production_rate_matches_faraday_hand_calc():
    # 1000 A at 100% CE: m = I*M*t/(zF)
    expected = 1000.0 * 55.845e-3 * 3600.0 / (2 * FARADAY)
    assert production_rate_kg_per_hr(1000.0, 1.0) == pytest.approx(expected, rel=1e-9)


def test_production_scales_linearly_with_efficiency():
    full = production_rate_kg_per_hr(500.0, 1.0)
    half = production_rate_kg_per_hr(500.0, 0.5)
    assert half == pytest.approx(full / 2.0, rel=1e-12)


def test_current_density_conversion_consistent():
    # 100 mA/cm^2 over 1 m^2 == 1000 A
    a = current_density_to_production(100.0, 1.0, 0.9)
    b = production_rate_kg_per_hr(1000.0, 0.9)
    assert a == pytest.approx(b, rel=1e-12)


# ─── Specific energy ───────────────────────────────────────────────────
def test_specific_energy_theoretical_minimum():
    # Thermodynamic minimum for Fe at 1.669 V, 100% CE is ~1.6 kWh/kg
    e = specific_energy_kWh_per_kg(1.669, 1.0)
    assert 1.55 < e < 1.65


def test_specific_energy_tonne_conversion():
    assert specific_energy_kWh_per_t(2.5, 0.9) == pytest.approx(
        specific_energy_kWh_per_kg(2.5, 0.9) * 1000.0
    )


def test_specific_energy_inverse_in_efficiency():
    assert specific_energy_kWh_per_kg(2.5, 0.5) == pytest.approx(
        2.0 * specific_energy_kWh_per_kg(2.5, 1.0)
    )


# ─── Cell voltage ──────────────────────────────────────────────────────
def test_cell_voltage_decomposition_sums():
    m = CellVoltageModel()
    assert m.V_cell == pytest.approx(
        m.E_thermodynamic + m.eta_cathode + m.eta_anode + m.ir_drop
    )
    assert m.E_thermodynamic == pytest.approx(1.669, abs=1e-3)


# ─── Pourbaix ──────────────────────────────────────────────────────────
def test_her_and_oer_lines_have_59mV_slope():
    slope_her = (her_line(1.0) - her_line(0.0)) * 1000.0
    slope_oer = (oer_line(1.0) - oer_line(0.0)) * 1000.0
    assert slope_her == pytest.approx(-59.16, abs=0.2)
    assert slope_oer == pytest.approx(-59.16, abs=0.2)


def test_water_window_is_1p23_V_at_all_pH():
    for pH in (0.0, 7.0, 14.0):
        assert oer_line(pH) - her_line(pH) == pytest.approx(1.229, abs=1e-6)


def test_her_line_zero_at_pH0():
    assert her_line(0.0) == pytest.approx(0.0, abs=1e-12)


def test_fe2_fe_potential_shifts_with_activity():
    dilute = FePourbaix(activity=1e-6).E_Fe2_Fe()
    concentrated = FePourbaix(activity=1.0).E_Fe2_Fe()
    assert concentrated == pytest.approx(-0.440, abs=1e-6)
    # dilution makes deposition harder (more negative)
    assert dilute < concentrated
    assert dilute == pytest.approx(-0.440 - 6 * 0.05916 / 2, abs=2e-3)


def test_hydrolysis_pH_increases_as_solution_dilutes():
    assert FePourbaix(activity=1e-6).pH_Fe2_FeOH2 > FePourbaix(activity=1.0).pH_Fe2_FeOH2


def test_fe3_hydrolyses_at_lower_pH_than_fe2():
    p = FePourbaix(activity=1e-2)
    assert p.pH_Fe3_FeOH3 < p.pH_Fe2_FeOH2


def test_deposition_always_below_her_line():
    """Core physics: Fe deposition is thermodynamically below HER at all pH."""
    p = FePourbaix(activity=1.0)
    for pH in np.linspace(0, 14, 29):
        assert p.deposition_potential(pH) < her_line(pH)
        assert p.her_margin(pH) > 0.0


def test_dominant_species_regions():
    p = FePourbaix(activity=1e-2)
    assert p.dominant_species(1.0, -1.2) == "Fe(s)"
    assert p.dominant_species(1.0, -0.2) == "Fe2+"
    assert p.dominant_species(1.0, 1.0) == "Fe3+"
    assert p.dominant_species(9.0, -0.5) == "Fe(OH)2(s)"
    assert p.dominant_species(9.0, 0.6) == "Fe(OH)3(s)"


def test_summary_covers_requested_pH_points():
    s = FePourbaix(activity=0.5).summary()
    assert set(s) == {0.0, 2.0, 7.0, 10.0, 14.0}
    assert all("HER margin (V)" in v for v in s.values())


# ─── Kinetics ──────────────────────────────────────────────────────────
def test_limiting_current_scales_with_concentration():
    a = limiting_current_density(1000.0)
    b = limiting_current_density(2000.0)
    assert b == pytest.approx(2 * a)


def test_limiting_current_inverse_with_boundary_layer():
    thin = limiting_current_density(1000.0, boundary_layer_m=1e-5)
    thick = limiting_current_density(1000.0, boundary_layer_m=1e-4)
    assert thin == pytest.approx(10 * thick)


def test_potential_solver_reproduces_target_current():
    k = DepositionKinetics()
    for j in (10.0, 50.0, 100.0):
        E = k.potential_at_current(j)
        assert k.partial_currents(E)[2] == pytest.approx(j * 10.0, rel=1e-6)


def test_current_efficiency_between_zero_and_one():
    k = DepositionKinetics()
    for j in (1.0, 10.0, 100.0, 400.0):
        ce = k.efficiency_at_current(j)
        assert 0.0 < ce < 1.0


def test_suppressing_her_exchange_current_raises_efficiency():
    active = DepositionKinetics(her_i0=1e-2)
    suppressed = DepositionKinetics(her_i0=1e-6)
    assert suppressed.efficiency_at_current(100.0) > active.efficiency_at_current(100.0)


def test_higher_pH_improves_efficiency():
    """Raising pH lowers the HER equilibrium potential, favouring Fe."""
    acidic = DepositionKinetics(pH=1.0)
    mild = DepositionKinetics(pH=5.0)
    assert mild.efficiency_at_current(100.0) > acidic.efficiency_at_current(100.0)


def test_efficiency_falls_when_mass_transport_limited():
    """Beyond i_lim the Fe branch saturates and HER takes the extra current."""
    k = DepositionKinetics(her_i0=1e-6, fe_conc_M=0.05, boundary_layer_m=1e-4)
    j_lim_mA_cm2 = k.i_lim / 10.0
    low = k.efficiency_at_current(0.2 * j_lim_mA_cm2)
    high = k.efficiency_at_current(3.0 * j_lim_mA_cm2)
    assert high < low


def test_agitation_recovers_efficiency():
    stagnant = DepositionKinetics(her_i0=1e-6, fe_conc_M=0.1, boundary_layer_m=2e-4)
    agitated = DepositionKinetics(her_i0=1e-6, fe_conc_M=0.1, boundary_layer_m=2e-5)
    assert agitated.efficiency_at_current(80.0) > stagnant.efficiency_at_current(80.0)


def test_deposition_rate_positive_and_monotone():
    k = DepositionKinetics()
    r1 = k.deposition_rate_um_hr(20.0)
    r2 = k.deposition_rate_um_hr(100.0)
    assert 0 < r1 < r2


def test_deposition_rate_against_analytic_value():
    k = DepositionKinetics()
    j = 50.0
    ce = k.efficiency_at_current(j)
    expected = (j * 10.0 * ce * 55.845e-3 / (2 * FARADAY)) / 7874.0 * 3600.0 * 1e6
    assert k.deposition_rate_um_hr(j) == pytest.approx(expected, rel=1e-9)


def test_hydrogen_flux_consistent_with_efficiency():
    k = DepositionKinetics()
    j = 100.0
    ce = k.efficiency_at_current(j)
    expected = (j * 10.0 * (1 - ce)) / (2 * FARADAY) * 3600.0
    assert k.hydrogen_flux_mol_m2_hr(j) == pytest.approx(expected, rel=1e-6)


def test_polarization_curve_shapes():
    k = DepositionKinetics()
    E, i_fe, i_h, i_tot, ce = k.polarization_curve()
    assert len(E) == len(i_fe) == len(i_h) == len(i_tot) == len(ce)
    assert np.all(np.diff(i_tot) <= 1e-6)  # more negative E -> larger current
    assert np.all(i_fe <= k.i_lim * (1 + 1e-9))


def test_summary_keys_present():
    s = DepositionKinetics().summary(100.0)
    for key in ("Current efficiency (%)", "Deposition rate (µm/hr)", "H₂ flux (mol/m²/hr)"):
        assert key in s


def test_summary_efficiency_matches_solver():
    k = DepositionKinetics()
    s = k.summary(120.0)
    assert s["Current efficiency (%)"] == pytest.approx(
        k.efficiency_at_current(120.0) * 100, abs=0.1
    )


# ─── Cross-module consistency ──────────────────────────────────────────
def test_kinetic_efficiency_feeds_energy_model_sanely():
    k = DepositionKinetics(her_i0=1e-6)
    ce = k.efficiency_at_current(100.0)
    energy = specific_energy_kWh_per_t(2.6, ce)
    assert 2000.0 < energy < 6000.0
    assert not math.isnan(energy)
